/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file coherent_integration_ref.h
 *
 * @brief Defines functions for reference model verification
 */

#ifndef COHERENT_INTEGRATION_REF_H
#define COHERENT_INTEGRATION_REF_H

#include <OpCoherentIntegration.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Target-arithmetic reference model for the Coherent Integration operator.
 *
 * Derives one adaptive exponent per range-Doppler cell, splits each input
 * component into two S16 limbs, completes the Rx dots in signed-48, and rounds
 * each completed complex component once. It compares beam power in fixed point
 * and converts the winning power to FP32 for the square root, scale restoration,
 * and FP32 fold.
 * The input geometry and steering tensor must satisfy the complete contract
 * documented by @ref pvaCoherentIntegrationCreate.
 *
 * @param [in] rangeDopplerMap Range-Doppler input, 2S32, HCW or NHCW with N = 1.
 * @param [in] steeringVectors Complex Q15 steering vectors, 2S16, shape
 *                             [numAzimuthBins, numElevationBins, numRx].
 * @param [out] coherentIntegrationRx Rx-coherently-integrated detection map,
 *                                    U32, shape [numRangeBins, numDopplerBins].
 * @param [out] coherentIntegrationFinal Final detection map after non-coherent
 *                                       Doppler-fold accumulation, U32, shape
 *                                       [numRangeBins, numDopplerBins / repeatFold].
 * @param [in] repeatFold Doppler fold factor. Must be at least 4, divisible by
 *                        4, and evenly divide numDopplerBins.
 * @param [in] inputLayout Range-Doppler ordering of @p rangeDopplerMap.
 * @return 0 on success, non-zero otherwise.
 */
int32_t coherentIntegrationTargetRef(NVCVTensorHandle rangeDopplerMap, NVCVTensorHandle steeringVectors,
                                     NVCVTensorHandle coherentIntegrationRx, NVCVTensorHandle coherentIntegrationFinal,
                                     int32_t repeatFold, PVAFFTLayout inputLayout);

/**
 * @brief Compares two U32 maps element-wise against an absolute tolerance.
 *
 * @param [in] refTensorHandle Reference tensor.
 * @param [in] pvaTensorHandle Tensor under test.
 * @param [in] tolerance Largest accepted absolute difference.
 *
 * @return 0 when every element is within tolerance, non-zero otherwise.
 */
int32_t checkCoherentIntegrationOutput(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle,
                                       uint32_t tolerance);

#ifdef __cplusplus
}
#endif

#endif /* COHERENT_INTEGRATION_REF_H */
