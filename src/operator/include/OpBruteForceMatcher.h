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

/**
 * @file OpBruteForceMatcher.h
 *
 * @brief Defines types and functions to handle the Brute Force Matcher operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_BRUTE_FORCE_MATCHER BruteForceMatcher
 * @{
 */

#ifndef PVA_SOLUTIONS_OPBRUTEFORCEMATCHER_H
#define PVA_SOLUTIONS_OPBRUTEFORCEMATCHER_H
#include <PvaOperator.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Status.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructs an instance of the Brute Force Matcher operator.
 *
 * The Brute Force Matcher finds optimal matches between two sets of BRIEF (Binary Robust Independent
 * Elementary Features) descriptors. Each descriptor is 256 bits (32 bytes) in length.
 *
 * The matching process computes the Hamming distance between each query descriptor and all reference
 * descriptors. For each query descriptor, it can return up to PVA_MAX_BF_MATCHES_PER_DESCRIPTOR best matches,
 * sorted by ascending Hamming distance.
 *
 * Operating Modes:
 * 1. Standard Mode (Cross Check Disabled):
 *    - Returns up to PVA_MAX_BF_MATCHES_PER_DESCRIPTOR best matches per query descriptor
 *
 * 2. Cross Check Mode (Cross Check Enabled):
 *    - Returns exactly one match per query descriptor
 *    - A match is valid only if the query descriptor is also the best match for the reference descriptor
 *    - Invalid matches are indicated by a reference index of -1
 *    - Requires maxMatchesPerQuery to be set to 1
 *
 *
 * \b Limitations:
 *      The number of either query or reference descriptors should not be larger than PVA_MAX_BF_DESCRIPTOR_COUNT.
 *
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT       Handle is null.
 * @retval NVCV_ERROR_OUT_OF_MEMORY          Failed to allocate memory for the operator.
 * @retval NVCV_SUCCESS                      Operation executed successfully.
 */
NVCVStatus pvaBruteForceMatcherCreate(NVCVOperatorHandle *handle);

#ifdef __cplusplus
}

/**
 * Submits the BruteForceMatcher operator to a cuPVA stream.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid cuPVA stream.
 *                    + Must not be NULL.
 * @param [in] query Pointer to the query descriptors. \ref PVABriefDescriptor.
 *                   + Should point to a buffer that is accessible by PVA engine.
 *                   + The size of the buffer should be at least queryCount * sizeof(PVABriefDescriptor).
 * @param [in] queryCount Number of query descriptors.
 *                        + Must be larger than 0 and less than or equal to PVA_MAX_BF_DESCRIPTOR_COUNT.
 * @param [in] reference Pointer to the reference descriptors. \ref PVABriefDescriptor.
 *                       + Should point to a buffer that is accessible by PVA engine.
 *                       + The size of the buffer should be at least referenceCount * sizeof(PVABriefDescriptor).
 * @param [in] referenceCount Number of reference descriptors.
 *                            + Must be larger than 0 and less than or equal to PVA_MAX_BF_DESCRIPTOR_COUNT.
 * @param [in] maxMatchesPerQuery Maximum number of matches per query.
 *                                + Must be larger than 0 and less than or equal to PVA_MAX_BF_MATCHES_PER_DESCRIPTOR.
 *                                + When enableCrossCheck is 1 or enableDistanceRatioTest is 1, maxMatchesPerQuery must be 1
 * @param [out] matches Pointer to the matches. \ref PVABFMatchesType.
 *                      + Should point to a buffer that is accessible by PVA engine.
 *                      + The size of the buffer should be at least queryCount * sizeof(PVABFMatchesType).
 * @param [in] enableCrossCheck Whether to enable cross check.
 *                              + Must be 0 or 1.
 * @param [in] enableDistanceRatioTest Whether to enable distance ratio test (Lowe's ratio test).
 *                                + Must be 0 or 1.
 *                                + Only used when queryCount and referenceCount are larger than 1.
 * @param [in] lowesTestThresholdNumerator Numerator of the Lowe's ratio test threshold.
 *                                + Must be larger than 0.
 *                                + Only used when enableDistanceRatioTest is 1.
 * @param [in] lowesTestThresholdDenominator Denominator of the Lowe's ratio test threshold.
 *                                + Must be larger than 0.
 *                                + Only used when enableDistanceRatioTest is 1.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT       The possible cases might be the following:
 *                                            1) The handle or stream or query or reference is either NULL or points to an invalid address.
 *                                            2) The queryCount or referenceCount is not larger than 0 or larger than PVA_MAX_BF_DESCRIPTOR_COUNT.
 *                                            3) The maxMatchesPerQuery is not larger than 0 or less than or equal to PVA_MAX_BF_MATCHES_PER_DESCRIPTOR.
 *                                            4) The enableCrossCheck is not 0 or 1.
 *                                            5) The enableCrossCheck is 1 and maxMatchesPerQuery is not 1.
 *                                            6) The enableDistanceRatioTest is not 0 or 1.
 *                                            7) The enableDistanceRatioTest is 1 and maxMatchesPerQuery is not 1.
 *                                            8) The enableDistanceRatioTest is 1 and queryCount and referenceCount are not larger than 1.
 *                                            9) The enableDistanceRatioTest is 1 and lowesTestThresholdNumerator and lowesTestThresholdDenominator are not larger than 0.
 *                                            10) The enableCrossCheck is 0 and enableDistanceRatioTest is 0 and maxMatchesPerQuery is greater than referenceCount.
 * @retval NVCV_SUCCESS                      Operation executed successfully.
 */
NVCVStatus pvaBruteForceMatcherSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, PVABriefDescriptor *query,
                                      int32_t queryCount, PVABriefDescriptor *reference, int32_t referenceCount,
                                      int32_t maxMatchesPerQuery, PVABFMatchesType *matches, uint8_t enableCrossCheck,
                                      uint8_t enableDistanceRatioTest, int32_t lowesTestThresholdNumerator,
                                      int32_t lowesTestThresholdDenominator);

/**
 * Submits the BruteForceMatcher operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] query Pointer to the query descriptors.
 * @param [in] queryCount Number of query descriptors.
 * @param [in] reference Pointer to the reference descriptors.
 * @param [in] referenceCount Number of reference descriptors.
 * @param [in] maxMatchesPerQuery Maximum number of matches per query.
 * @param [out] matches Pointer to the matches.
 * @param [in] enableCrossCheck Whether to enable cross check.
 * @param [in] enableDistanceRatioTest Whether to enable distance ratio test.
 * @param [in] lowesTestThresholdNumerator Numerator of the Lowe's ratio test threshold.
 * @param [in] lowesTestThresholdDenominator Denominator of the Lowe's ratio test threshold.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT       The possible cases might be invalid parameters.
 * @retval NVCV_SUCCESS                      Operation executed successfully.
 */
NVCVStatus pvaBruteForceMatcherSubmit(NVCVOperatorHandle handle, cudaStream_t stream, PVABriefDescriptor *query,
                                      int32_t queryCount, PVABriefDescriptor *reference, int32_t referenceCount,
                                      int32_t maxMatchesPerQuery, PVABFMatchesType *matches, uint8_t enableCrossCheck,
                                      uint8_t enableDistanceRatioTest, int32_t lowesTestThresholdNumerator,
                                      int32_t lowesTestThresholdDenominator);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPBRUTEFORCEMATCHER_H */
