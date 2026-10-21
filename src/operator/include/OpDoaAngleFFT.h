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
 * @file OpDoaAngleFFT.h
 *
 * @brief Defines types and functions for the DOA Angle FFT operator (8x8 MIMO; see parameter limitations below).
 *
 * This operator performs Direction of Arrival (DoA) angle estimation and outputs
 * target angles and target index maps as tensors.
 *
 * ** API inputs and outputs:**
 * - Inputs: detectionCount, snapshots, detectionList, ddmDopplerOffsets
 * - Outputs: targetCount, targetAngles, targetIndexMap
 *
 * @defgroup PVA_RADAR_OPERATOR_ALGORITHM_RADAR_DOA_ANGLE_FFT RadarDoaAngleFFT
 * @{
 */

#ifndef PVA_SOLUTIONS_OP_DOA_ANGLE_FFT_H
#define PVA_SOLUTIONS_OP_DOA_ANGLE_FFT_H

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
 * Constructs an instance of the DOA Angle FFT operator.
 *
 * This operator performs 2D Angle FFT for Direction of Arrival (DOA) estimation.
 * It processes virtual antenna snapshots and outputs target angles and target index maps.
 *
 * \b Limitations:
 *
 *      The maximum number of detections and targets is 8192.
 *      All tensors must be densely packed, with no padding between dimensions.
 *      The local peak detection is not supported yet. Parameter is ignored.
 *      The product of the number of virtual azimuth elements and the number of virtual elevation elements must equal
 *      the number of Tx antennas multiplied by the number of Rx antennas:
 *          numVirtualAzimuthElements * numVirtualElevationElements = numTxAntennas * numRxAntennas
 *
 *      Operator currently supports the following parameter values:
 *       - numTxAntennas = 8
 *       - numRxAntennas = 8
 *       - numVirtualAzimuthElements = 16
 *       - numVirtualElevationElements = 4
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] snapshotsTensorRequirements Snapshots tensor requirements
 *                     + Data Layout: [CHW] — C: maximum number of targets; H: number of Tx antennas; W: number of Rx antennas
 *                     + Rank: 3
 *                     + Data Type: NVCV_DATA_TYPE_2S32 (Complex SQ11.20)
 *
 * @param [in] detectionListTensorRequirements Range-Doppler Detection List tensor requirements
 *                     + Data Layout: [HW] — H: number of detections (numDetections); W: 2 (rangeIdx, dopplerIdx)
 *                     + Rank: 2
 *                     + Data Type: NVCV_DATA_TYPE_S32
 *
 * @param [in] ddmDopplerOffsetsTensorRequirements DDM Doppler Offsets tensor requirements
 *                     + Data Layout: [W] — W: number of Doppler folds (numDopplerFolds)
 *                     + Rank: 1
 *                     + Data Type: NVCV_DATA_TYPE_F32
 *
 * @param [in] targetAnglesTensorRequirements Target angles tensor requirements
 *                     + Data Layout: [HW] — H: 2 (azimuth, elevation); W: maximum number of targets
 *                     + Rank: 2
 *                     + Data Type: NVCV_DATA_TYPE_F32
 *
 * @param [in] targetIndexMapTensorRequirements Target index map tensor requirements
 *                     + Data Layout: [HW] — H: number of index map rows; W: maximum number of targets
 *                     + Rank: 2
 *                     + Data Type: NVCV_DATA_TYPE_S32
 *
 * @param [in] params Pointer to the DoaAngleFFT parameters. \ref PVADoaAngleFFTParams.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT     Handle is null or some parameter is outside valid range.
 * @retval NVCV_ERROR_OUT_OF_MEMORY        Not enough memory to create the operator.
 * @retval NVCV_ERROR_INVALID_IMAGE_FORMAT Image format is invalid.
 * @retval NVCV_SUCCESS                    Operation executed successfully.
 */
NVCVStatus pvaDoaAngleFFTCreate(NVCVOperatorHandle *handle,
                                NVCVTensorRequirements const *const snapshotsTensorRequirements,
                                NVCVTensorRequirements const *const detectionListTensorRequirements,
                                NVCVTensorRequirements const *const ddmDopplerOffsetsTensorRequirements,
                                NVCVTensorRequirements const *const targetAnglesTensorRequirements,
                                NVCVTensorRequirements const *const targetIndexMapTensorRequirements,
                                PVADoaAngleFFTParams const *const params);

#ifdef __cplusplus
}

/**
 * Submits the DoA Angle FFT operator to a cuPVA stream.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 *
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] detectionCount Input tensor: Detection Count
 *                            Data Layout: [W] where W=1
 *                            Data Type: NVCV_DATA_TYPE_S32
 *
 * @param [in] snapshots Input tensor: Snapshots
 *                       Data Layout: [CHW] where C=Maximum number of targets, H=Number of Tx antennas, W=Number of Rx antennas
 *                       Data Type: NVCV_DATA_TYPE_2S32 (Complex SQ11.20)
 *
 * @param [in] detectionList Input tensor: Range-Doppler Detection List
 *                           Data Layout: [HW] where H=numDetections, W=2
 *                           Data Type: NVCV_DATA_TYPE_S32
 *
 * @param [in] ddmDopplerOffsets Input tensor: DDM Doppler Offsets
 *                               Data Layout: [W] where W=numDopplerFolds
 *                               Data Type: NVCV_DATA_TYPE_F32
 *
 * @param [out] targetCount Output tensor: Target Count
 *                          Data Layout: [W] where W=1
 *                          Data Type: NVCV_DATA_TYPE_S32
 *
 * @param [out] targetAngles Output tensor: Target Angles
 *                           Data Layout: [HW] where H=2, W=Maximum number of targets
 *                           Data Type: NVCV_DATA_TYPE_F32
 *
 * @param [out] targetIndexMap Output tensor: Target Index Map
 *                             Data Layout: [HW]
 *                             Data Type: NVCV_DATA_TYPE_S32
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaDoaAngleFFTSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle const detectionCount,
                                NVCVTensorHandle const snapshots, NVCVTensorHandle const detectionList,
                                NVCVTensorHandle const ddmDopplerOffsets, NVCVTensorHandle targetCount,
                                NVCVTensorHandle targetAngles, NVCVTensorHandle targetIndexMap);

/**
 * Submits the DoA Angle FFT operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] detectionCount Input tensor: Detection Count
 * @param [in] snapshots Input tensor: Snapshots
 * @param [in] detectionList Input tensor: Detection List (Range-Doppler detection indices)
 * @param [in] ddmDopplerOffsets Input tensor: DDM Doppler Offsets
 * @param [out] targetCount Output tensor: Target Count
 * @param [out] targetAngles Output tensor: Target Angles
 * @param [out] targetIndexMap Output tensor: Target Index Map
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaDoaAngleFFTSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle const detectionCount,
                                NVCVTensorHandle const snapshots, NVCVTensorHandle const detectionList,
                                NVCVTensorHandle const ddmDopplerOffsets, NVCVTensorHandle targetCount,
                                NVCVTensorHandle targetAngles, NVCVTensorHandle targetIndexMap);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OP_DOA_ANGLE_FFT_H */
