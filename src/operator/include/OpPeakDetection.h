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
 * @file OpPeakDetection.h
 *
 * @brief Defines types and functions to handle the Peak Detection operation.
 * @defgroup PVA_RADAR_OPERATOR_ALGORITHM_PEAK_DETECTION RadarPeakDetection
 * @{
 */

#ifndef PVA_SOLUTIONS_OPPEAKDETECTION_H
#define PVA_SOLUTIONS_OPPEAKDETECTION_H

#include <PvaOperator.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructs an instance of the Peak Detection operator.
 *
 * @param [out] handle Pointer to where the operator instance handle will be written.
 *                     Must not be NULL.
 *
 * @param [in] inTensorRequirements Pointer to a structure containing tensor rank, shape, layout, and data type information.
 *                                 + The inTensorRequirements corresponds to the dimensions of the FT2D tensor; other input tensors' dimensions are derived from it.
 *
 * Currently, the tensors have the following dimensions:
 * Nb – number of range bins (samples), Nr – number of Doppler bins (ramps), NofRx – number of RX antennas,
 * FT2D: std::complex<int32_t> [Nb][NofRx][Nr]
 * Data type: 2S32 (complex, fixed point Q15)
 * Data layout: HCW, H – number of range bins, C – number of RX antennas, W – number of Doppler bins
 *
 * nciRx: uint32_t [Nb][Nr]
 * Data type: U32 (fixed point Q15)
 * Data layout: HW, H – number of range bins, W – number of Doppler bins
 *
 * nciFinal: uint32_t [Nb][Nr/repeatFold]
 * Data type: U32 (fixed point Q15)
 * Data layout: HW, H – number of range bins, W – number of Doppler bins divided by repeatFold
 *
 * noiseEst: uint32_t [Nb]
 * Data type: U32 (fixed point Q15)
 * Data layout: W, W – number of range bins
 *
 * @param [in] outTensorRequirements Pointer to a structure containing tensor rank, shape, layout, and data type information.
 *                                   + The outTensorRequirements corresponds to the dimensions of the snap tensor;
 *                                   + peakIndex's shape[1] and snap's shape[0] are derived from it.
 *                                   + peakCount is fixed to be a 1x1 tensor.
 * peakCount: uint32_t [1]
 * Data type: U32
 * Data layout: W
 *
 * peakIndex: uint32_t [PVA_RADAR_PEAKDET_NUM_PEAK_INDICES][PEAKDET_MAX_PEAK_COUNT]
 * Data type: U32
 * Data layout: HW, H – PVA_RADAR_PEAKDET_NUM_PEAK_INDICES, W – maximum number of peaks. The first row corresponds to the range bin index, the second
 * row corresponds to the folded Doppler bin index, and the third row corresponds to the unfolded Doppler bin index.
 *
 * snap: 2S32 [PEAKDET_MAX_PEAK_COUNT][number_of_TX_antennas*number_of_RX_antennas]
 * Data type: 2S32 (same as the data type of FT2D)
 * Data layout: HW, H – maximum number of peaks, W – number_of_TX_antennas*number_of_RX_antennas
 *
 * @param [in] NofTx      The number of TX antennas.
 *                        + Currently only supports the value 4.
 *
 * @param [in] repeatFold The number of Doppler folds.
 *                        + Currently only supports the value 8.
 *                        + The value of repeatFold must be the same as the value of repeatFold in the inTensorRequirements.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT     Handle is null or some parameter is outside valid range.
 * @retval NVCV_ERROR_OUT_OF_MEMORY        Not enough memory to create the operator.
 * @retval NVCV_SUCCESS                    Operation executed successfully.
 */
NVCVStatus pvaPeakDetectionCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements const *const inTensorRequirements,
                                  NVCVTensorRequirements const *const outTensorRequirements, int32_t NofTx,
                                  int32_t repeatFold);

#ifdef __cplusplus
}

/**
 * Submits the PeakDetection operator to a cuPVA stream.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] inHandles Pointer to array of input NVCVTensorHandle.
 *                       + inHandles[0] is the FT2D tensor.
 *                       + inHandles[1] is the nciRx tensor.
 *                       + inHandles[2] is the nciFinal tensor.
 *                       + inHandles[3] is the noiseEst tensor.
 *
 * @param [out] outHandles Pointer to array of output NVCVTensorHandle.
 *                        + outHandles[0] is the peakCount tensor.
 *                        + outHandles[1] is the peakIndex tensor.
 *                        + Note: The order of each entry might not be the same among implementations.
 *                        + It is the caller's responsibility to sort if a specific order is required.
 *                        + outHandles[2] is the snap tensor.
 *
 * @param [in] inTensorCount Number of input tensors.
 *
 * @param [in] outTensorCount Number of output tensors.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null or some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaPeakDetectionSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle *inHandles,
                                  NVCVTensorHandle *outHandles, int inTensorCount, int outTensorCount);

/**
 * Submits the PeakDetection operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] inHandles Pointer to array of input NVCVTensorHandle.
 *                       + inHandles[0] is the FT2D tensor.
 *                       + inHandles[1] is the nciRx tensor.
 *                       + inHandles[2] is the nciFinal tensor.
 *                       + inHandles[3] is the noiseEst tensor.
 *
 * @param [out] outHandles Pointer to array of output NVCVTensorHandle.
 *                        + outHandles[0] is the peakCount tensor.
 *                        + outHandles[1] is the peakIndex tensor.
 *                        + Note: The order of each entry might not be the same among implementations.
 *                        + It is the caller's responsibility to sort if a specific order is required.
 *                        + outHandles[2] is the snap tensor.
 *
 * @param [in] inTensorCount Number of input tensors.
 *
 * @param [in] outTensorCount Number of output tensors.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null or some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaPeakDetectionSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle *inHandles,
                                  NVCVTensorHandle *outHandles, int inTensorCount, int outTensorCount);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPPEAKDETECTION_H */