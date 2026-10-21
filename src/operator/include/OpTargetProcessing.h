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
 * @file OpTargetProcessing.h
 *
 * @brief Defines types and functions for the Target Processing radar operator.
 *
 * The operator performs Range-Velocity (RV) decoupling and Cartesian coordinate computation for
 * detected targets: it converts range/Doppler bin indices and DOA angles into physical range,
 * velocity, and (X, Y, Z) coordinates.
 *
 * ** API inputs and outputs:**
 * - Inputs: targetCount, detectionCount, detectionList, targetIndexMap, targetAngles, ddmDopplerOffsets, nci (optional)
 * - Outputs: targetList (velocity, range, azimuth, elevation, X, Y, Z, optional power)
 *
 * @defgroup PVA_RADAR_OPERATOR_ALGORITHM_RADAR_TARGET_PROCESSING RadarTargetProcessing
 * @{
 */

#ifndef PVA_SOLUTIONS_OPTARGETPROCESSING_H
#define PVA_SOLUTIONS_OPTARGETPROCESSING_H

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
 * Constructs an instance of the Target Processing operator.
 *
 * The operator performs Range-Velocity (RV) decoupling and Cartesian coordinate computation for
 * detected radar targets. It consumes detection list indices, a per-target detection index map, DOA
 * angles (and optional NCI for quadratic interpolation) and writes a complete target list:
 * velocity, range, angles, Cartesian coordinates, and optional peak power.
 *
 * \b Limitations:
 *
 *      The maximum number of detections and targets is 8192.
 *      The maximum number of DDM Doppler folds is 32.
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] detectionListTensorRequirements Detection list tensor requirements
 *                     + Data Layout: [HW] — H: number of detections (numDetections); W: 2 (rangeIdx, dopplerIdx)
 *                     + Rank: 2
 *                     + Data Type: NVCV_DATA_TYPE_S32
 *
 * @param [in] targetIndexMapTensorRequirements Target index map tensor requirements
 *                     + Data Layout: [W] — W: number of targets (numTargets)
 *                     + Rank: 1
 *                     + Data Type: NVCV_DATA_TYPE_S32
 *                     + Description: Maps each target to a detection index; each element is the detection index for that target.
 *
 * @param [in] targetAnglesTensorRequirements Target angles tensor requirements
 *                     + Data Layout: [HW] — H: 3 (azimuth, elevation, power); W: number of targets (numTargets)
 *                     + Rank: 2
 *                     + Data Type: NVCV_DATA_TYPE_F32
 *                     + Description: DOA estimation — row 0: azimuth (degrees); row 1: elevation (degrees); row 2: peak power (dB)
 *
 * @param [in] ddmDopplerOffsetsTensorRequirements DDM Doppler Offsets tensor requirements
 *                     + Data Layout: [W] — W: number of Doppler folds (numDopplerFolds)
 *                     + Rank: 1
 *                     + Data Type: NVCV_DATA_TYPE_F32
 *
 * @param [in] targetListTensorRequirements Target list tensor requirements
 *                     + Data Layout: [HW] — H: 7 (without peak power) or 8 (with peak power); W: number of targets (numTargets)
 *                     + Rank: 2
 *                     + Data Type: NVCV_DATA_TYPE_F32
 *                     + Description: Target rows — 0: velocity (m/s); 1: range (m); 2: azimuth (degrees); 3: elevation (degrees); 4: X (m); 5: Y (m); 6: Z (m); 7: optional peak power (dB)
 *
 * @param [in] nciTensorRequirements NCI final output tensor requirements (optional, can be NULL)
 *                     + Data Layout: [HW] — H: number of range bins (numRangeBins); W: number of folded Doppler bins (numFoldedDopplerBins)
 *                     + Rank: 2
 *                     + Data Type: NVCV_DATA_TYPE_U32
 *                     + Description: NCI power for quadratic interpolation (sub-bin range and Doppler). If NULL, interpolation is off. Required when params->enableQuadraticInterpolation is true.
 *
 * @param [in] params Pointer to the Target Processing parameters. \ref PVATargetProcessingParams.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT     Handle is null or some parameter is outside valid range.
 * @retval NVCV_ERROR_OUT_OF_MEMORY        Not enough memory to create the operator.
 * @retval NVCV_SUCCESS                    Operation executed successfully.
 */
NVCVStatus pvaTargetProcessingCreate(NVCVOperatorHandle *handle,
                                     NVCVTensorRequirements const *const detectionListTensorRequirements,
                                     NVCVTensorRequirements const *const targetIndexMapTensorRequirements,
                                     NVCVTensorRequirements const *const targetAnglesTensorRequirements,
                                     NVCVTensorRequirements const *const ddmDopplerOffsetsTensorRequirements,
                                     NVCVTensorRequirements const *const targetListTensorRequirements,
                                     NVCVTensorRequirements const *const nciTensorRequirements,
                                     PVATargetProcessingParams const *const params);

#ifdef __cplusplus
}

/**
 * Submits the Target Processing operator to a cuPVA stream.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 *
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] targetCount Input tensor: Target Count (also used as detection count for processing)
 *                         Data Layout: [W] where W=1
 *                         Data Type: NVCV_DATA_TYPE_S32
 *
 * @param [in] detectionCount Input tensor: Detection Count
 *                            Data Layout: [W] where W=1
 *                            Data Type: NVCV_DATA_TYPE_S32
 *
 * @param [in] detectionList Input tensor: Range-Doppler Detection List
 *                           Data Layout: [HW] where H=numDetections, W=2
 *                           Data Type: NVCV_DATA_TYPE_S32
 *
 * @param [in] targetIndexMap Input tensor: Target Index Map
 *                            Data Layout: [W] where W=numTargets
 *                            Data Type: NVCV_DATA_TYPE_S32
 *
 * @param [in] targetAngles Input tensor: Target Angles from DOA
 *                          Data Layout: [HW] where H=3, W=numTargets
 *                          Data Type: NVCV_DATA_TYPE_F32
 *
 * @param [in] ddmDopplerOffsets Input tensor: DDM Doppler Offsets
 *                               Data Layout: [W] where W=numDopplerFolds
 *                               Data Type: NVCV_DATA_TYPE_F32
 *
 * @param [in] nci Input tensor: NCI (optional, can be NULL for no quadratic interpolation)
 *                 Data Layout: [HW] where H=numRangeBins, W=numFoldedDopplerBins
 *                 Data Type: NVCV_DATA_TYPE_U32
 *
 * @param [out] targetList Output tensor: Target List (complete target information)
 *                         Data Layout: [HW] where H=7 or 8, W=numTargets
 *                         Data Type: NVCV_DATA_TYPE_F32
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaTargetProcessingSubmit(NVCVOperatorHandle handle, cupvaStream_t stream,
                                     NVCVTensorHandle const targetCount, NVCVTensorHandle const detectionCount,
                                     NVCVTensorHandle const detectionList, NVCVTensorHandle const targetIndexMap,
                                     NVCVTensorHandle const targetAngles, NVCVTensorHandle const ddmDopplerOffsets,
                                     NVCVTensorHandle const nci, NVCVTensorHandle targetList);

/**
 * Submits the Target Processing operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] targetCount Input tensor: Target Count
 * @param [in] detectionCount Input tensor: Detection Count
 * @param [in] detectionList Input tensor: Detection List
 * @param [in] targetIndexMap Input tensor: Target Index Map
 * @param [in] targetAngles Input tensor: Target Angles
 * @param [in] ddmDopplerOffsets Input tensor: DDM Doppler Offsets
 * @param [in] nci Input tensor: NCI (optional, can be NULL)
 * @param [out] targetList Output tensor: Target List
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaTargetProcessingSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle const targetCount,
                                     NVCVTensorHandle const detectionCount, NVCVTensorHandle const detectionList,
                                     NVCVTensorHandle const targetIndexMap, NVCVTensorHandle const targetAngles,
                                     NVCVTensorHandle const ddmDopplerOffsets, NVCVTensorHandle const nci,
                                     NVCVTensorHandle targetList);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPTARGETPROCESSING_H */
