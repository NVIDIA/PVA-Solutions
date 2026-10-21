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

/**
 * @file OpCoherentIntegration.h
 *
 * @brief Defines the public interface for radar coherent integration.
 * @defgroup PVA_OPERATOR_ALGORITHM_RADAR_COHERENT_INTEGRATION RadarCoherentIntegration
 * @{
 */

#ifndef PVA_SOLUTIONS_OPCOHERENTINTEGRATION_H
#define PVA_SOLUTIONS_OPCOHERENTINTEGRATION_H

#include <PvaOperator.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Computes steering vectors for Coherent Integration.
 *
 * This utility computes steering vectors from Rx positions and angle bins and
 * writes them to a caller-owned tensor. The resulting tensor can be reused
 * across submissions while the Rx positions and angle bins remain unchanged.
 * A caller may instead provide steering vectors directly from the sensor.
 *
 * Steering vectors are computed as:
 *
 * @code
 * phase[k,az,el] =
 *     rxPositions[k].x * cos(el) * sin(az)
 *     + rxPositions[k].y * sin(el)
 *
 * steering[az,el,k] = exp(-j * 2 * pi * phase[k,az,el])
 * @endcode
 *
 * Angle-bin values are converted from degrees to radians for the
 * trigonometric operations.
 *
 * All four tensors must use host-mappable PVA memory.
 *
 * @param [in] rxPositions Wavelength-normalized Rx positions.
 *                         - Type: @c NVCV_DATA_TYPE_2F32.
 *                         - Layout/rank: `W`, rank 1.
 *                         - Shape: `[numRx]`, where `numRx` is 4, 8, 12, or
 *                           16.
 *                         - Component 0 is the horizontal position and
 *                           component 1 is the vertical position.
 *                         - All components must be finite.
 *
 * @param [in] azimuthBins Azimuth angle bins in degrees.
 *                         - Type: @c NVCV_DATA_TYPE_F32.
 *                         - Layout/rank: `W`, rank 1.
 *                         - Shape: `[numAzimuthBins]`, where
 *                           `numAzimuthBins > 0`.
 *                         - All values must be finite.
 *
 * @param [in] elevationBins Elevation angle bins in degrees.
 *                           - Type: @c NVCV_DATA_TYPE_F32.
 *                           - Layout/rank: `W`, rank 1.
 *                           - Shape: `[numElevationBins]`, where
 *                             `numElevationBins > 0`.
 *                           - All values must be finite.
 *
 * @param [out] steeringVectors Steering-vector output.
 *                              - Type: @c NVCV_DATA_TYPE_2S16 (complex Q15).
 *                              - Layout/rank: `CHW`, rank 3.
 *                              - Shape: `[numAzimuthBins, numElevationBins,
 *                                numRx]`.
 *                              - `C`, `H`, and `W` correspond to azimuth,
 *                                elevation, and Rx, respectively.
 *                              - `C * H * W` must not exceed 12,288 elements.
 *                              - Must be fully packed in
 *                                `[Azimuth][Elevation][Rx]` order.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT A tensor is NULL or does not satisfy
 *                                     the documented requirements.
 * @retval NVCV_ERROR_INTERNAL         Internal computation failure.
 * @retval NVCV_SUCCESS                Steering vectors computed successfully.
 */
NVCVStatus pvaCoherentIntegrationComputeSteeringVectors(NVCVTensorHandle rxPositions, NVCVTensorHandle azimuthBins,
                                                        NVCVTensorHandle elevationBins,
                                                        NVCVTensorHandle steeringVectors);

/**
 * Constructs a Coherent Integration operator.
 *
 * For complex range-Doppler input `X[r,k,d]` and steering-vector input
 * `S[az,el,k]`, the operator computes:
 *
 * @code
 * beam[r,d,az,el] =
 *     sum_k X[r,k,d] * S[az,el,k]
 *
 * C[r,d] = max_(az,el) abs(beam[r,d,az,el])
 * @endcode
 *
 * Each supplied Q15 steering value is used directly as the complex
 * multiplication factor; no implicit conjugation is applied. Steering vectors
 * may be provided by the sensor or computed using
 * @ref pvaCoherentIntegrationComputeSteeringVectors. `C` is true magnitude
 * (amplitude), not squared magnitude or power. The internal angle-bin traversal
 * order is unspecified, and no selected-angle index is produced.
 *
 * The folded map is:
 *
 * @code
 * F[r,q] = sum_(m=0..repeatFold-1)
 *              C[r, q + m * (numDopplerBins / repeatFold)]
 * @endcode
 *
 * The U32 outputs store `C / 2^rxShift` and `F / 2^finalShift`, truncated
 * toward zero and saturated to `INT32_MAX`, where:
 *
 * @code
 * rxShift    = ceil(log2(numRx)) + 1
 * finalShift = rxShift + ceil(log2(repeatFold))
 * @endcode
 *
 * The sum over `k` coherently integrates the complex inputs across the Rx
 * dimension. Because `C` contains magnitudes, the subsequent accumulation
 * across Tx-related Doppler folds to form `F` is non-coherent.
 *
 * All components in the range-Doppler input must use one common upstream
 * fixed-point exponent.
 *
 * \b Limitations (all constraints apply):
 *
 * - Dimensions: `4 <= numRangeBins <= 4096`,
 *   `128 <= numDopplerBins <= 4096`, and `numRx` is 4, 8, 12, or 16.
 * - `repeatFold` is a multiple of 4 in `[4, 128]` and evenly divides
 *   `numDopplerBins`.
 * - `numDopplerBins / repeatFold` is a multiple of 32.
 * - Input layout is `[Range][Doppler][Rx]` or `[Range][Rx][Doppler]`.
 * - Every Range, Doppler, and Rx input byte stride and every output byte
 *   stride is positive and fits in a 32-bit unsigned value.
 * - `[Range][Doppler][Rx]`: Rx stride is 8 bytes, Doppler stride is in
 *   `[numRx * 8, 65535]` bytes, and Range rows do not overlap.
 * - `[Range][Rx][Doppler]`: Doppler stride is 8 bytes, Rx stride is in
 *   `[numDopplerBins * 8, 65535]` bytes, and Range rows do not overlap.
 * - Both U32 outputs have a 4-byte W stride and non-overlapping Range rows.
 * - Steering vectors are packed in `[Azimuth][Elevation][Rx]` order with
 *   byte strides `[Elevation * Rx * 4, Rx * 4, 4]`.
 *
 * @note If the input or either output Range stride exceeds `2^23 - 1` bytes, the
 * operator remains correct but processes one Range row per tile, which may
 * reduce throughput.
 *
 * @param [out] handle Pointer to the created operator handle. Must not be
 *                     NULL. Destroy a successfully created handle with
 *                     @ref nvcvOperatorDestroy after all submitted work has
 *                     completed.
 *
 * @param [in] rangeDopplerMapTensorRequirements Input tensor requirements.
 *                     Must not be NULL.
 *                     - Type: @c NVCV_DATA_TYPE_2S32.
 *                     - Layout/rank: `HCW`, or `NHCW` with `N = 1`.
 *                     - For
 *                       @c PVA_FFT_LAYOUT_RANGE_RX_DOPPLER:
 *                       `H = numRangeBins`, `C = numRx`,
 *                       `W = numDopplerBins`.
 *                     - For
 *                       @c PVA_FFT_LAYOUT_RANGE_DOPPLER_RX:
 *                       `H = numRangeBins`, `C = numDopplerBins`,
 *                       `W = numRx`.
 *
 * @param [in] coherentIntegrationRxTensorRequirements Rx-coherently-integrated
 *                     detection-map output requirements. Must not be NULL.
 *                     - Type: @c NVCV_DATA_TYPE_U32.
 *                     - Layout/rank: `HW`, rank 2.
 *                     - Shape: `[numRangeBins, numDopplerBins]`.
 *
 * @param [in] coherentIntegrationFinalTensorRequirements Final detection-map
 *                     output requirements after non-coherent accumulation
 *                     across Tx-related Doppler folds. Must not be NULL.
 *                     - Type: @c NVCV_DATA_TYPE_U32.
 *                     - Layout/rank: `HW`, rank 2.
 *                     - Shape: `[numRangeBins,
 *                       numDopplerBins / repeatFold]`.
 *
 * @param [in] params Operator parameters. Must not be NULL. The parameter
 *                    structure and tensor requirements are copied before this
 *                    function returns, so the caller may release or modify
 *                    them after a successful call.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT     A pointer is NULL or a parameter,
 *                                         shape relationship, or enum value is
 *                                         invalid.
 * @retval NVCV_ERROR_INVALID_IMAGE_FORMAT A tensor type, rank, or layout is
 *                                         unsupported.
 * @retval NVCV_ERROR_OUT_OF_MEMORY        Insufficient memory to create the
 *                                         operator.
 * @retval NVCV_SUCCESS                    Operator created successfully.
 */
NVCVStatus pvaCoherentIntegrationCreate(NVCVOperatorHandle *handle,
                                        NVCVTensorRequirements const *const rangeDopplerMapTensorRequirements,
                                        NVCVTensorRequirements const *const coherentIntegrationRxTensorRequirements,
                                        NVCVTensorRequirements const *const coherentIntegrationFinalTensorRequirements,
                                        PVARadarCoherentIntegrationParams const *const params);

#ifdef __cplusplus
}

/**
 * Submits the Coherent Integration operator to a cuPVA stream.
 *
 * The range-Doppler and output tensors must match the complete requirements
 * supplied at creation, including type, shape, layout, strides, alignment, and
 * memory accessibility. The steering-vector tensor must satisfy the documented
 * runtime requirements and be PVA-accessible. No two submitted tensors may
 * overlap; in-place operation is unsupported.
 *
 * @param [in] handle Valid operator handle. Must not be NULL.
 * @param [in] stream Valid cuPVA stream.
 * @param [in] rangeDopplerMap Input range-Doppler tensor. Must not be NULL.
 * @param [in] steeringVectors Precomputed steering vectors. Must not be NULL.
 *                             The tensor may be populated by the sensor or by
 *                             @ref pvaCoherentIntegrationComputeSteeringVectors.
 *                             - Type: @c NVCV_DATA_TYPE_2S16 (complex Q15).
 *                             - Layout/rank: `CHW`, rank 3.
 *                             - Shape: `[numAzimuthBins, numElevationBins,
 *                               numRx]`, where all dimensions are positive and
 *                               `numRx` matches the input range-Doppler tensor.
 *                             - `C`, `H`, and `W` correspond to azimuth,
 *                               elevation, and Rx, respectively.
 *                             - `C * H * W` must not exceed 12,288 elements.
 * @param [out] coherentIntegrationRx Rx-coherently-integrated detection map
 *                                    before Doppler folding. Must not be NULL.
 * @param [out] coherentIntegrationFinal Final detection map after non-coherent
 *                                       accumulation across Tx-related Doppler
 *                                       folds. Must not be NULL.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT An argument is invalid or a tensor does
 *                                     not match the configured requirements.
 * @retval NVCV_ERROR_INTERNAL         Internal execution failure.
 * @retval NVCV_SUCCESS                Operation submitted successfully.
 */
NVCVStatus pvaCoherentIntegrationSubmit(NVCVOperatorHandle handle, cupvaStream_t stream,
                                        NVCVTensorHandle const rangeDopplerMap, NVCVTensorHandle const steeringVectors,
                                        NVCVTensorHandle coherentIntegrationRx,
                                        NVCVTensorHandle coherentIntegrationFinal);

/**
 * Submits the Coherent Integration operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * The range-Doppler and output tensors must match the complete requirements
 * supplied at creation, including type, shape, layout, strides, alignment, and
 * memory accessibility. The steering-vector tensor must satisfy the documented
 * runtime requirements and be PVA-accessible. No two submitted tensors may
 * overlap; in-place operation is unsupported.
 *
 * @param [in] handle Valid operator handle. Must not be NULL.
 * @param [in] stream Valid CUDA stream.
 * @param [in] rangeDopplerMap Input range-Doppler tensor. Must not be NULL.
 * @param [in] steeringVectors Precomputed steering vectors. Must not be NULL.
 *                             The tensor may be populated by the sensor or by
 *                             @ref pvaCoherentIntegrationComputeSteeringVectors.
 *                             - Type: @c NVCV_DATA_TYPE_2S16 (complex Q15).
 *                             - Layout/rank: `CHW`, rank 3.
 *                             - Shape: `[numAzimuthBins, numElevationBins,
 *                               numRx]`, where all dimensions are positive and
 *                               `numRx` matches the input range-Doppler tensor.
 *                             - `C`, `H`, and `W` correspond to azimuth,
 *                               elevation, and Rx, respectively.
 *                             - `C * H * W` must not exceed 12,288 elements.
 * @param [out] coherentIntegrationRx Rx-coherently-integrated detection map
 *                                    before Doppler folding. Must not be NULL.
 * @param [out] coherentIntegrationFinal Final detection map after non-coherent
 *                                       accumulation across Tx-related Doppler
 *                                       folds. Must not be NULL.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT An argument is invalid or a tensor does
 *                                     not match the configured requirements.
 * @retval NVCV_ERROR_INTERNAL         Internal execution failure.
 * @retval NVCV_ERROR_NOT_IMPLEMENTED  CUDA stream support is unavailable.
 * @retval NVCV_SUCCESS                Operation submitted successfully.
 */
NVCVStatus pvaCoherentIntegrationSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                        NVCVTensorHandle const rangeDopplerMap, NVCVTensorHandle const steeringVectors,
                                        NVCVTensorHandle coherentIntegrationRx,
                                        NVCVTensorHandle coherentIntegrationFinal);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPCOHERENTINTEGRATION_H */
