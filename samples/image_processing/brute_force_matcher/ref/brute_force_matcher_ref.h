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

/*
 * @file brute_force_matcher_ref.h
 *
 * @brief Defines functions for reference model verification
 */

#ifndef BRUTE_FORCE_MATCHER_REF_H
#define BRUTE_FORCE_MATCHER_REF_H

#include <OpBruteForceMatcher.h>
#include <nvcv/Tensor.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

void generateBruteForceMatcherDescriptor(PVABriefDescriptor *descriptors, int32_t count);

int32_t checkBruteForceMatcherOutput(const PVABFMatchesType *refMatches, int32_t refMatchesCount,
                                     const PVABFMatchesType *pvaMatches);

void bruteForceMatcherRef(const PVABriefDescriptor *queryDescriptors, int32_t queryCount,
                          const PVABriefDescriptor *referenceDescriptors, int32_t referenceCount,
                          int32_t maxMatchesPerQuery, PVABFMatchesType *matches, uint8_t enableCrossCheck,
                          uint8_t enableDistanceRatioTest, int32_t lowesTestThresholdNumerator,
                          int32_t lowesTestThresholdDenominator);

#ifdef __cplusplus
}
#endif

#endif /* BRUTE_FORCE_MATCHER_REF_H */
