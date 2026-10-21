/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "brute_force_matcher_ref.h"

#include <TensorUtils.hpp>

#include <OpBruteForceMatcher.h>
#include <nvcv/TensorShape.h>

#include <bitset>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

// Enum for different matching strategies
enum class MatchingStrategy
{
    SIMPLE_MATCHING,                      // Simple matching without any check
    CROSS_CHECK_ONLY,                     // Only Cross Check
    DISTANCE_RATIO_TEST_ONLY,             // Only Distance Ratio Test
    CROSS_CHECK_WITH_DISTANCE_RATIO_TEST, // Cross check with distance ratio test
};

// Function to determine which matching strategy to use
MatchingStrategy determineMatchingStrategy(uint8_t enableCrossCheck, uint8_t enableDistanceRatioTest,
                                           int32_t queryCount, int32_t referenceCount, int32_t maxMatchesPerQuery)
{
    bool hasDistanceRatioTest = (enableDistanceRatioTest == 1 && queryCount > 1 && referenceCount > 1);
    bool hasCrossCheck        = (enableCrossCheck == 1);

    if (!hasCrossCheck && !hasDistanceRatioTest)
        return MatchingStrategy::SIMPLE_MATCHING;
    else if (hasCrossCheck && hasDistanceRatioTest)
        return MatchingStrategy::CROSS_CHECK_WITH_DISTANCE_RATIO_TEST;
    else if (hasCrossCheck && !hasDistanceRatioTest)
        return MatchingStrategy::CROSS_CHECK_ONLY;
    else // !hasCrossCheck && hasDistanceRatioTest
        return MatchingStrategy::DISTANCE_RATIO_TEST_ONLY;
}

void generateBruteForceMatcherDescriptorImpl(PVABriefDescriptor *descriptors, int32_t count)
{
    static std::mt19937 gen;
    std::uniform_int_distribution<> distrib(0, std::numeric_limits<uint8_t>::max());

    uint8_t *descriptorsU8Ptr_h = nullptr;
    CupvaMemGetHostPointer((void **)&descriptorsU8Ptr_h, (void *)descriptors);

    int32_t size = count * PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH;
    for (size_t i = 0; i < size; i++)
    {
        descriptorsU8Ptr_h[i] = static_cast<uint8_t>(distrib(gen));
    }
}

// "size" is the number of bytes to compute the Hamming distance.
int32_t computeHammingDistance(const uint8_t *lhs, const uint8_t *rhs, int32_t size)
{
    int32_t distance = 0;
    for (int32_t i = 0; i < size; i++)
    {
        distance += std::bitset<8>(lhs[i] ^ rhs[i]).count();
    }
    return distance;
}

bool sortMetric(std::pair<int32_t, int32_t> lhs, std::pair<int32_t, int32_t> rhs)
{
    if (lhs.second == rhs.second)
    {
        return lhs.first < rhs.first;
    }

    return lhs.second < rhs.second;
}

static inline bool isUniqueEnough(int32_t first_distance, int32_t second_distance, int32_t lowesTestThresholdNumerator,
                                  int32_t lowesTestThresholdDenominator)
{
    // first_distance * DENOMINATOR >= second_distance * NUMERATOR means NOT unique
    return (first_distance * lowesTestThresholdDenominator < second_distance * lowesTestThresholdNumerator);
}

// Simple matching without any check - Case 0
void performSimpleMatching(const PVABriefDescriptor *queryDescriptors_h, int32_t queryCount,
                           const PVABriefDescriptor *referenceDescriptors_h, int32_t referenceCount,
                           int32_t maxMatchesPerQuery, PVABFMatchesType *matches_h)
{
    for (int32_t i = 0; i < queryCount; i++)
    {
        std::vector<std::pair<int32_t, int32_t>> _matches;
        for (int32_t j = 0; j < referenceCount; j++)
        {
            int32_t distance = computeHammingDistance(queryDescriptors_h[i].data, referenceDescriptors_h[j].data,
                                                      PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH);
            _matches.push_back(std::make_pair(j, distance));
        }
        std::partial_sort(_matches.begin(), _matches.begin() + maxMatchesPerQuery, _matches.end(), sortMetric);

        for (int32_t j = 0; j < maxMatchesPerQuery; j++)
        {
            matches_h[i].refIndex[j] = static_cast<int16_t>(_matches[j].first);
            matches_h[i].distance[j] = static_cast<int16_t>(_matches[j].second);
        }
    }
}

// Cross check without distance ratio test - Case 1
void performCrossCheckWithoutDistanceRatioTest(const PVABriefDescriptor *queryDescriptors_h, int32_t queryCount,
                                               const PVABriefDescriptor *referenceDescriptors_h, int32_t referenceCount,
                                               PVABFMatchesType *matches_h)
{
    for (int32_t i = 0; i < queryCount; i++)
    {
        std::vector<std::pair<int32_t, int32_t>> matchesQ2R;
        for (int32_t j = 0; j < referenceCount; j++)
        {
            int32_t distance = computeHammingDistance(queryDescriptors_h[i].data, referenceDescriptors_h[j].data,
                                                      PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH);
            matchesQ2R.push_back(std::make_pair(j, distance));
        }
        std::partial_sort(matchesQ2R.begin(), matchesQ2R.begin() + 1, matchesQ2R.end(), sortMetric);
        std::pair<int32_t, int32_t> bestMatchQ2R = matchesQ2R.front();

        std::vector<std::pair<int32_t, int32_t>> matchesR2Q;
        for (int32_t j = 0; j < queryCount; j++)
        {
            int32_t distance = computeHammingDistance(referenceDescriptors_h[bestMatchQ2R.first].data,
                                                      queryDescriptors_h[j].data, PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH);
            matchesR2Q.push_back(std::make_pair(j, distance));
        }
        std::partial_sort(matchesR2Q.begin(), matchesR2Q.begin() + 1, matchesR2Q.end(), sortMetric);
        std::pair<int32_t, int32_t> bestMatchR2Q = matchesR2Q.front();

        if (bestMatchR2Q.first == i)
        {
            matches_h[i].refIndex[0] = static_cast<int16_t>(bestMatchQ2R.first);
            matches_h[i].distance[0] = static_cast<int16_t>(bestMatchQ2R.second);
        }
        else
        {
            matches_h[i].refIndex[0] = -1;
            matches_h[i].distance[0] = 0;
        }
    }
}

// Distance ratio test without cross check - Case 2
void performDistanceRatioTestWithoutCrossCheck(const PVABriefDescriptor *queryDescriptors_h, int32_t queryCount,
                                               const PVABriefDescriptor *referenceDescriptors_h, int32_t referenceCount,
                                               PVABFMatchesType *matches_h, int32_t lowesTestThresholdNumerator,
                                               int32_t lowesTestThresholdDenominator)
{
    for (int32_t i = 0; i < queryCount; i++)
    {
        std::vector<std::pair<int32_t, int32_t>> matchesQ2R;
        for (int32_t j = 0; j < referenceCount; j++)
        {
            int32_t distance = computeHammingDistance(queryDescriptors_h[i].data, referenceDescriptors_h[j].data,
                                                      PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH);
            matchesQ2R.push_back(std::make_pair(j, distance));
        }
        std::partial_sort(matchesQ2R.begin(), matchesQ2R.begin() + 2, matchesQ2R.end(), sortMetric);

        if (isUniqueEnough(matchesQ2R[0].second, matchesQ2R[1].second, lowesTestThresholdNumerator,
                           lowesTestThresholdDenominator))
        {
            matches_h[i].refIndex[0] = static_cast<int16_t>(matchesQ2R[0].first);
            matches_h[i].distance[0] = static_cast<int16_t>(matchesQ2R[0].second);
        }
    }
}

// Cross check with distance ratio test - Case 3
void performCrossCheckWithDistanceRatioTest(const PVABriefDescriptor *queryDescriptors_h, int32_t queryCount,
                                            const PVABriefDescriptor *referenceDescriptors_h, int32_t referenceCount,
                                            PVABFMatchesType *matches_h, int32_t lowesTestThresholdNumerator,
                                            int32_t lowesTestThresholdDenominator)
{
    for (int32_t i = 0; i < queryCount; i++)
    {
        std::vector<std::pair<int32_t, int32_t>> matchesQ2R;
        for (int32_t j = 0; j < referenceCount; j++)
        {
            int32_t distance = computeHammingDistance(queryDescriptors_h[i].data, referenceDescriptors_h[j].data,
                                                      PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH);
            matchesQ2R.push_back(std::make_pair(j, distance));
        }

        // Find top-2 matches for uniqueness test
        std::partial_sort(matchesQ2R.begin(), matchesQ2R.begin() + 2, matchesQ2R.end(), sortMetric);

        if (isUniqueEnough(matchesQ2R[0].second, matchesQ2R[1].second, lowesTestThresholdNumerator,
                           lowesTestThresholdDenominator))
        {
            std::pair<int32_t, int32_t> bestMatchQ2R = matchesQ2R.front();

            std::vector<std::pair<int32_t, int32_t>> matchesR2Q;
            for (int32_t j = 0; j < queryCount; j++)
            {
                int32_t distance =
                    computeHammingDistance(referenceDescriptors_h[bestMatchQ2R.first].data, queryDescriptors_h[j].data,
                                           PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH);
                matchesR2Q.push_back(std::make_pair(j, distance));
            }
            std::partial_sort(matchesR2Q.begin(), matchesR2Q.begin() + 2, matchesR2Q.end(), sortMetric);

            if (isUniqueEnough(matchesR2Q[0].second, matchesR2Q[1].second, lowesTestThresholdNumerator,
                               lowesTestThresholdDenominator))
            {
                std::pair<int32_t, int32_t> bestMatchR2Q = matchesR2Q.front();

                if (bestMatchR2Q.first == i)
                {
                    matches_h[i].refIndex[0] = static_cast<int16_t>(bestMatchQ2R.first);
                    matches_h[i].distance[0] = static_cast<int16_t>(bestMatchQ2R.second);
                }
            }
        }
    }
}

void bruteForceMatcherRefImpl(const PVABriefDescriptor *queryDescriptors, int32_t queryCount,
                              const PVABriefDescriptor *referenceDescriptors, int32_t referenceCount,
                              int32_t maxMatchesPerQuery, PVABFMatchesType *matches, uint8_t enableCrossCheck,
                              uint8_t enableDistanceRatioTest, int32_t lowesTestThresholdNumerator,
                              int32_t lowesTestThresholdDenominator)
{
    PVABriefDescriptor *queryDescriptors_h = nullptr;
    CupvaMemGetHostPointer((void **)&queryDescriptors_h, (void *)queryDescriptors);

    PVABriefDescriptor *referenceDescriptors_h = nullptr;
    CupvaMemGetHostPointer((void **)&referenceDescriptors_h, (void *)referenceDescriptors);

    PVABFMatchesType *matches_h = nullptr;
    CupvaMemGetHostPointer((void **)&matches_h, (void *)matches);

    const int32_t matchesCount = queryCount;
    for (int32_t i = 0; i < matchesCount; i++)
    {
        for (int32_t j = 0; j < PVA_MAX_BF_MATCHES_PER_DESCRIPTOR; j++)
        {
            matches_h[i].refIndex[j] = -1;
            matches_h[i].distance[j] = 0;
        }
    }

    // Determine matching strategy and execute once
    MatchingStrategy strategy = determineMatchingStrategy(enableCrossCheck, enableDistanceRatioTest, queryCount,
                                                          referenceCount, maxMatchesPerQuery);

    switch (strategy)
    {
    case MatchingStrategy::SIMPLE_MATCHING:
        performSimpleMatching(queryDescriptors_h, queryCount, referenceDescriptors_h, referenceCount,
                              maxMatchesPerQuery, matches_h);
        break;
    case MatchingStrategy::CROSS_CHECK_ONLY:
        performCrossCheckWithoutDistanceRatioTest(queryDescriptors_h, queryCount, referenceDescriptors_h,
                                                  referenceCount, matches_h);
        break;
    case MatchingStrategy::DISTANCE_RATIO_TEST_ONLY:
        performDistanceRatioTestWithoutCrossCheck(queryDescriptors_h, queryCount, referenceDescriptors_h,
                                                  referenceCount, matches_h, lowesTestThresholdNumerator,
                                                  lowesTestThresholdDenominator);
        break;
    case MatchingStrategy::CROSS_CHECK_WITH_DISTANCE_RATIO_TEST:
        performCrossCheckWithDistanceRatioTest(queryDescriptors_h, queryCount, referenceDescriptors_h, referenceCount,
                                               matches_h, lowesTestThresholdNumerator, lowesTestThresholdDenominator);
        break;
    default:
        printf("bruteForceMatcherRefImpl Error: Invalid matching strategy: %d\n", static_cast<int32_t>(strategy));
        break;
    }
}
} // namespace

void generateBruteForceMatcherDescriptor(PVABriefDescriptor *descriptors, int32_t count)
{
    generateBruteForceMatcherDescriptorImpl(descriptors, count);
}

int32_t checkBruteForceMatcherOutput(const PVABFMatchesType *refMatches, int32_t refMatchesCount,
                                     const PVABFMatchesType *pvaMatches)
{
    const PVABFMatchesType *refMatches_h = nullptr;
    CupvaMemGetHostPointer((void **)&refMatches_h, (void *)refMatches);

    const PVABFMatchesType *pvaMatches_h = nullptr;
    CupvaMemGetHostPointer((void **)&pvaMatches_h, (void *)pvaMatches);

    int32_t err = 0;
    for (int32_t i = 0; i < refMatchesCount; i++)
    {
        for (int32_t j = 0; j < PVA_MAX_BF_MATCHES_PER_DESCRIPTOR; j++)
        {
            if ((refMatches_h[i].refIndex[j] != pvaMatches_h[i].refIndex[j]) ||
                (refMatches_h[i].refIndex[j] != -1 && refMatches_h[i].distance[j] != pvaMatches_h[i].distance[j]))
            {
                if (!err)
                {
                    printf(
                        "Mismatch format[query index, 1st/2nd/3rd match]: ref = (refIndex, distance), pva = (refIndex, "
                        "distance)\n");
                }
                printf("Mismatch[%d, %d]: ref = (%d, %d), pva = (%d, %d)\n", i, j, refMatches_h[i].refIndex[j],
                       refMatches_h[i].distance[j], pvaMatches_h[i].refIndex[j], pvaMatches_h[i].distance[j]);
                err++;
            }
        }
        if (err > 10)
        {
            break;
        }
    }

    return err;
}

void bruteForceMatcherRef(const PVABriefDescriptor *queryDescriptors, int32_t queryCount,
                          const PVABriefDescriptor *referenceDescriptors, int32_t referenceCount,
                          int32_t maxMatchesPerQuery, PVABFMatchesType *matches, uint8_t enableCrossCheck,
                          uint8_t enableDistanceRatioTest, int32_t lowesTestThresholdNumerator,
                          int32_t lowesTestThresholdDenominator)
{
    bruteForceMatcherRefImpl(queryDescriptors, queryCount, referenceDescriptors, referenceCount, maxMatchesPerQuery,
                             matches, enableCrossCheck, enableDistanceRatioTest, lowesTestThresholdNumerator,
                             lowesTestThresholdDenominator);
}
