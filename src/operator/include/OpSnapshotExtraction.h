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
 * @file OpSnapshotExtraction.h
 *
 * @brief Defines types and functions to handle the Snapshot Extraction operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_RADAR_SNAPSHOT_EXTRACTION RadarSnapshotExtraction
 * @{
 */

#ifndef PVA_SOLUTIONS_OPSNAPSHOTEXTRACTION_H
#define PVA_SOLUTIONS_OPSNAPSHOTEXTRACTION_H

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
 * Constructs an instance of the Snapshot Extraction operator.
 *
 * This operator performs DDM (Doppler Division Multiplexing) disambiguation and
 * gathers snapshots from the Range-Doppler map for detected targets.
 *
 * \b Limitations:
 *
 *      The maximum number of detections is 8192.
 *      The only valid numTx X numRx configurations are 4x4 and 8x8.
 *      The maximum number of range bins is 512.
 *      The maximum number of doppler bins is 512 and the minimum number of doppler bins is 4.
 *      The valid number of doppler fold should be greater than or equal to the number of Tx antennas and less than or equal to 16.
 *      The valid number of doppler fold should be able to divide the number of doppler bins evenly.
 *      The maximum number of doppler folds is 16 and the minimum number of doppler folds is 4.
 *      rangeDopplerFormat must be PVA_FFT_LAYOUT_RANGE_RX_DOPPLER or PVA_FFT_LAYOUT_RANGE_DOPPLER_RX.
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] foldedDetectionListTensorRequirements Folded Detection List tensor requirements
 *                     + Data Layout: [HW] — H: number of detections (numDetections); W: 2 (rangeIdx, foldedDopplerIdx)
 *                     + Rank: 2
 *                     + Data Type: NVCV_DATA_TYPE_S32
 *
 * @param [in] ddmDopplerOffsetsTensorRequirements DDM Doppler Offsets tensor requirements
 *                     + Data Layout: [W] — W: number of Doppler folds (numDopplerFolds)
 *                     + Rank: 1
 *                     + Data Type: NVCV_DATA_TYPE_F32
 *
 * @param [in] nciRxTensorRequirements nciRx tensor requirements (magnitude accumulated over Rx channels)
 *                     + Data Layout: [HW] — H: number of range bins (numRangeBins); W: number of Doppler bins (numDopplerBins)
 *                     + Rank: 2
 *                     + Data Type: NVCV_DATA_TYPE_U32
 *
 * @param [in] rangeDopplerMapTensorRequirements Range-Doppler Map tensor requirements
 *                     + Data Layout: [HCW] or [NHCW] (with N = 1).
 *                         If rangeDopplerFormat is PVA_FFT_LAYOUT_RANGE_RX_DOPPLER:
 *                             H: number of range bins (numRangeBins)
 *                             C: number of Rx channels (numRx)
 *                             W: number of Doppler bins (numDopplerBins)
 *                         If rangeDopplerFormat is PVA_FFT_LAYOUT_RANGE_DOPPLER_RX:
 *                             H: number of range bins (numRangeBins)
 *                             C: number of Doppler bins (numDopplerBins)
 *                             W: number of Rx channels (numRx)
 *                     + Rank: 3 for [HCW]; 4 for [NHCW] (only when N = 1)
 *                     + Data Type: NVCV_DATA_TYPE_2S32 (Complex SQ11.20)
 *
 * @param [in] detectionListTensorRequirements Range-Doppler Detection List (unfolded) tensor requirements
 *                     + Data Layout: [HW] — H: number of detections (numDetections); W: 2 (rangeIdx, dopplerIdx)
 *                     + Rank: 2
 *                     + Data Type: NVCV_DATA_TYPE_S32
 *
 * @param [in] snapshotsTensorRequirements Snapshots tensor requirements
 *                     + Data Layout: [CHW] — C: number of detections (numDetections); H: number of Tx antennas; W: number of Rx antennas
 *                     + Rank: 3
 *                     + Data Type: NVCV_DATA_TYPE_2S32 (Complex SQ11.20)
 *
 * @param [in] calibrationWeightsTensorRequirements Calibration Weights tensor requirements (optional)
 *                     + Data Layout: [HW] — H: number of Tx antennas (numTx); W: number of Rx antennas (numRx)
 *                     + Rank: 2
 *                     + Data Type: NVCV_DATA_TYPE_2S32 (Complex Q28 fixed-point)
 *                     + Description: Can be NULL to use DDMA weights instead of calibration weights.
 *
 * @param [in] params Pointer to the SnapshotExtraction parameters. \ref PVASnapshotExtractionParams.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT     Handle is null or some parameter is outside valid range.
 * @retval NVCV_ERROR_OUT_OF_MEMORY        Not enough memory to create the operator.
 * @retval NVCV_ERROR_INVALID_IMAGE_FORMAT Image format is invalid.
 * @retval NVCV_SUCCESS                    Operation executed successfully.
 */
NVCVStatus pvaSnapshotExtractionCreate(NVCVOperatorHandle *handle,
                                       NVCVTensorRequirements const *const foldedDetectionListTensorRequirements,
                                       NVCVTensorRequirements const *const ddmDopplerOffsetsTensorRequirements,
                                       NVCVTensorRequirements const *const nciRxTensorRequirements,
                                       NVCVTensorRequirements const *const rangeDopplerMapTensorRequirements,
                                       NVCVTensorRequirements const *const detectionListTensorRequirements,
                                       NVCVTensorRequirements const *const snapshotsTensorRequirements,
                                       NVCVTensorRequirements const *const calibrationWeightsTensorRequirements,
                                       PVASnapshotExtractionParams const *const params);

#ifdef __cplusplus
}

/**
 * Submits the Snapshot Extraction operator to a cuPVA stream.
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
 * @param [in] foldedDetectionList Input tensor: Folded Detection List
 *                                 Data Layout: [HW] where H=numDetections, W=2 [rangeIdx, foldedDopplerIdx]
 *                                 Data Type: NVCV_DATA_TYPE_S32
 *
 * @param [in] ddmDopplerOffsets Input tensor: DDM Doppler Offsets
 *                               Data Layout: [W] where W=numDopplerFolds
 *                               Data Type: NVCV_DATA_TYPE_F32
 *
 * @param [in] nciRx Input tensor: nciRx (Magnitude accumulated over Rx channels)
 *                   Data Layout: [HW] where H=numRangeBins, W=numDopplerBins
 *                   Data Type: NVCV_DATA_TYPE_U32
 *
 * @param [in] rangeDopplerMap Input tensor: Range-Doppler Map
 *                             Data Layout: [HCW], or [NHCW] only when N = 1
 *                             If rangeDopplerFormat is PVA_FFT_LAYOUT_RANGE_RX_DOPPLER:
 *                                H: Number of range bins.
 *                                C: Number of RX channels.
 *                                W: Number of Doppler bins.
 *                             If rangeDopplerFormat is PVA_FFT_LAYOUT_RANGE_DOPPLER_RX:
 *                                H: Number of range bins.
 *                                C: Number of Doppler bins.
 *                                W: Number of RX channels.
 *                             Data Type: NVCV_DATA_TYPE_2S32 (Complex SQ11.20)
 *
 * @param [in] calibrationWeights Input tensor: Calibration Weights (optional)
 *                                Data Layout: [HW] where H=numTx, W=numRx
 *                                Data Type: NVCV_DATA_TYPE_2S32 (Complex Q28 fixed-point)
 *                                Note: Can be NULL if useCalibrationWeights=false
 *
 * @param [out] detectionList Output tensor: Range-Doppler Detection List (unfolded)
 *                            Data Layout: [HW] where H=numDetections, W=2 [rangeIdx, dopplerIdx]
 *                            Data Type: NVCV_DATA_TYPE_S32
 *
 * @param [out] snapshots Output tensor: Snapshots
 *                        Data Layout: [CHW] where C=numDetections, H=numTxAntennas, W=numRxAntennas
 *                        Data Type: NVCV_DATA_TYPE_2S32 (Complex SQ11.20)
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaSnapshotExtractionSubmit(NVCVOperatorHandle handle, cupvaStream_t stream,
                                       NVCVTensorHandle const detectionCount,
                                       NVCVTensorHandle const foldedDetectionList,
                                       NVCVTensorHandle const ddmDopplerOffsets, NVCVTensorHandle const nciRx,
                                       NVCVTensorHandle const rangeDopplerMap,
                                       NVCVTensorHandle const calibrationWeights, NVCVTensorHandle detectionList,
                                       NVCVTensorHandle snapshots);

/**
 * Submits the Snapshot Extraction operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 *
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] detectionCount Input tensor: Detection Count
 *                            Data Layout: [W] where W=1
 *                            Data Type: NVCV_DATA_TYPE_S32
 *
 * @param [in] foldedDetectionList Input tensor: Folded Detection List
 *                                 Data Layout: [HW] where H=numDetections, W=2 [rangeIdx, foldedDopplerIdx]
 *                                 Data Type: NVCV_DATA_TYPE_S32
 *
 * @param [in] ddmDopplerOffsets Input tensor: DDM Doppler Offsets
 *                               Data Layout: [W] where W=numDopplerFolds
 *                               Data Type: NVCV_DATA_TYPE_F32
 *
 * @param [in] nciRx Input tensor: nciRx (Magnitude accumulated over Rx channels)
 *                   Data Layout: [HW] where H=numRangeBins, W=numDopplerBins
 *                   Data Type: NVCV_DATA_TYPE_U32
 *
 * @param [in] rangeDopplerMap Input tensor: Range-Doppler Map
 *                             Data Layout: [HCW], or [NHCW] only when N = 1
 *                             If rangeDopplerFormat is PVA_FFT_LAYOUT_RANGE_RX_DOPPLER:
 *                                H: Number of range bins.
 *                                C: Number of RX channels.
 *                                W: Number of Doppler bins.
 *                             If rangeDopplerFormat is PVA_FFT_LAYOUT_RANGE_DOPPLER_RX:
 *                                H: Number of range bins.
 *                                C: Number of Doppler bins.
 *                                W: Number of RX channels.
 *                             Data Type: NVCV_DATA_TYPE_2S32 (Complex SQ11.20)
 *
 * @param [in] calibrationWeights Input tensor: Calibration Weights (optional)
 *                                Data Layout: [HW] where H=numTx, W=numRx
 *                                Data Type: NVCV_DATA_TYPE_2S32 (Complex Q28 fixed-point)
 *                                Note: Can be NULL if useCalibrationWeights=false
 *
 * @param [out] detectionList Output tensor: Range-Doppler Detection List (unfolded)
 *                            Data Layout: [HW] where H=numDetections, W=2 [rangeIdx, dopplerIdx]
 *                            Data Type: NVCV_DATA_TYPE_S32
 *
 * @param [out] snapshots Output tensor: Snapshots
 *                        Data Layout: [CHW] where C=numDetections, H=numTxAntennas, W=numRxAntennas
 *                        Data Type: NVCV_DATA_TYPE_2S32 (Complex SQ11.20)
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaSnapshotExtractionSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                       NVCVTensorHandle const detectionCount,
                                       NVCVTensorHandle const foldedDetectionList,
                                       NVCVTensorHandle const ddmDopplerOffsets, NVCVTensorHandle const nciRx,
                                       NVCVTensorHandle const rangeDopplerMap,
                                       NVCVTensorHandle const calibrationWeights, NVCVTensorHandle detectionList,
                                       NVCVTensorHandle snapshots);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPSNAPSHOTEXTRACTION_H */
