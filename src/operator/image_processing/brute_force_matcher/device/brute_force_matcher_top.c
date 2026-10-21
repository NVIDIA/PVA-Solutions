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

#include "brute_force_matcher_knls.h"
#include "brute_force_matcher_param.h"

#include <cupva_device.h>

// The following variables are runtime parameters, which are passed from the host to the device.
VMEM(A, int32_t, queryCount);
VMEM(A, int32_t, referenceCount);
VMEM(A, int32_t, maxMatchesPerQuery);
VMEM(A, uint8_t, enableCrossCheck);
VMEM(A, uint8_t, enableDistanceRatioTest);
VMEM(A, int32_t, lowesTestThresholdNumerator);
VMEM(A, int32_t, lowesTestThresholdDenominator);

VMEM_SEQDF_HANDLER(A, querySQDFHdl);
VMEM_SEQDF_HANDLER(A, referenceSQDFHdl);
VMEM_SEQDF_HANDLER(A, matchesSQDFHdl);

// Operator expects queryBuf, referenceBuf, matches_source_buf (see OpBruteForceMatcher.cpp)
// queryBuf in C, referenceBuf in B to fit DMb (matches_source_buf, matches_reference_buf in A)
VMEM(C, uint8_t, queryBuf, PVA_MAX_BF_DESCRIPTOR_COUNT, PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH);
VMEM(B, uint8_t, referenceBuf, PVA_MAX_BF_DESCRIPTOR_COUNT, PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH);

// Working buffers for brute_force_matcher (R2Q and Q2R); matches_source_buf reused for output
VMEM(A, uint32_t, matches_source_buf, PVA_MAX_BF_DESCRIPTOR_COUNT *PVA_MAX_BF_MATCHES_PER_DESCRIPTOR);
VMEM(A, uint32_t, matches_reference_buf, PVA_MAX_BF_DESCRIPTOR_COUNT *PVA_MAX_BF_MATCHES_PER_DESCRIPTOR);

static uint32_t pad_descriptors_to_multiple_of_32(uint8_t descriptors[][PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH],
                                                  const uint32_t count)
{
    if (0 == count)
    {
        return count;
    }

    const uint32_t remainder = count % PVA_BF_QUERY_BATCH_SIZE;
    if (0 == remainder)
    {
        return count;
    }

    const uint32_t padded_count = count + (PVA_BF_QUERY_BATCH_SIZE - remainder);
    for (uint32_t pad_idx = count; pad_idx < padded_count; ++pad_idx)
    {
        for (uint32_t byte_idx = 0; byte_idx < PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH; ++byte_idx)
        {
            descriptors[pad_idx][byte_idx] = descriptors[count - 1][byte_idx];
        }
    }
    return padded_count;
}

// Convert internal packed format to PVABFMatchesType (int16_t refIndex, int16_t distance)
static void convert_matches_source_to_matches_buf(const uint32_t *matches_source, PVABFMatchesType *matches,
                                                  const uint32_t descriptor_count, const uint32_t max_matches_per_query)
{
    const int16_t *matches_source_s16 = (const int16_t *)matches_source;
    for (uint32_t query_idx = 0; query_idx < descriptor_count; ++query_idx)
    {
        for (uint32_t match_slot = 0; match_slot < PVA_MAX_BF_MATCHES_PER_DESCRIPTOR; ++match_slot)
        {
            if (match_slot >= max_matches_per_query)
            {
                matches[query_idx].refIndex[match_slot] = -1;
                matches[query_idx].distance[match_slot] = 0;
                continue;
            }

            const uint32_t entry_index = query_idx + (match_slot * PVA_MAX_BF_DESCRIPTOR_COUNT);
            const int16_t distance_s16 = matches_source_s16[entry_index * 2];
            const int16_t index_s16    = matches_source_s16[entry_index * 2 + 1];
            if (distance_s16 > (int16_t)PVA_BF_MAX_HAMMING_DISTANCE || (-1 == distance_s16 && -1 == index_s16))
            {
                matches[query_idx].refIndex[match_slot] = -1;
                matches[query_idx].distance[match_slot] = 0;
                continue;
            }

            const uint16_t index_u16                = ((uint16_t)index_s16) & PVA_BF_INDEX_MASK;
            matches[query_idx].refIndex[match_slot] = (int16_t)index_u16;
            matches[query_idx].distance[match_slot] = distance_s16;
        }
    }
}

CUPVA_VPU_MAIN()
{
    const int algo = enableCrossCheck + 2 * enableDistanceRatioTest;

    cupvaSQDFOpen(querySQDFHdl);
    cupvaSQDFUpdateTileSize(querySQDFHdl, 0, PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH, queryCount);
    cupvaSQDFFlushAndTrig(querySQDFHdl);
    cupvaSQDFSync(querySQDFHdl);
    cupvaSQDFClose(querySQDFHdl);

    cupvaSQDFOpen(referenceSQDFHdl);
    cupvaSQDFUpdateTileSize(referenceSQDFHdl, 0, PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH, referenceCount);
    cupvaSQDFFlushAndTrig(referenceSQDFHdl);
    cupvaSQDFSync(referenceSQDFHdl);
    cupvaSQDFClose(referenceSQDFHdl);

    const uint32_t padded_query_count = pad_descriptors_to_multiple_of_32(queryBuf, (uint32_t)queryCount);
    uint32_t padded_reference_count   = (uint32_t)referenceCount;

    if (0 != algo)
    {
        padded_reference_count = pad_descriptors_to_multiple_of_32(referenceBuf, (uint32_t)referenceCount);
    }

    for (uint32_t buf_idx = 0; buf_idx < PVA_MAX_BF_DESCRIPTOR_COUNT * PVA_MAX_BF_MATCHES_PER_DESCRIPTOR; ++buf_idx)
    {
        matches_source_buf[buf_idx]    = PVA_BF_INVALID_MATCH;
        matches_reference_buf[buf_idx] = PVA_BF_INVALID_MATCH;
    }

    brute_force_matcher((uint16_t *)queryBuf, (uint16_t *)referenceBuf, matches_source_buf, matches_reference_buf,
                        padded_query_count, padded_reference_count, algo, lowesTestThresholdNumerator,
                        lowesTestThresholdDenominator, (uint32_t)referenceCount, (uint32_t)queryCount);
    convert_matches_source_to_matches_buf(matches_reference_buf, (PVABFMatchesType *)matches_source_buf,
                                          (uint32_t)queryCount, (uint32_t)maxMatchesPerQuery);

    cupvaSQDFOpen(matchesSQDFHdl);
    cupvaSQDFUpdateTileSize(matchesSQDFHdl, 0, queryCount * sizeof(PVABFMatchesType), 1);
    cupvaSQDFFlushAndTrig(matchesSQDFHdl);
    cupvaSQDFSync(matchesSQDFHdl);
    cupvaSQDFClose(matchesSQDFHdl);

    return 0;
}
