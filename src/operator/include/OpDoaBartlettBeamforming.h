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
 * @file OpDoaBartlettBeamforming.h
 *
 * @brief Defines types and functions to handle the Bartlett Beamforming DOA operation.
 *
 * This operator performs Direction of Arrival (DoA) angle estimation using Bartlett beamforming
 * and outputs target angles and target index map tensors.
 *
 * ** API inputs and outputs:**
 * - Inputs: detectionCount, snapshots, steeringVectors, azimuthBins, elevationBins, detectionList, ddmDopplerOffsets
 * - Outputs: targetCount, targetAngles (azimuth, elevation, power(optional)), targetIndexMap
 *
 * @defgroup PVA_OPERATOR_ALGORITHM_RADAR_DOA_BARTLETT_BEAMFORMING RadarBartlettBeamformerDoA (Direction of Arrival)
 * @{
 */

#ifndef PVA_SOLUTIONS_OPDOABARTLETTBEAMFORMING_H
#define PVA_SOLUTIONS_OPDOABARTLETTBEAMFORMING_H

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
 * Constructs an instance of the Bartlett Beamforming DOA operator.
 *
 * This operator performs Bartlett beamforming (conventional beamforming) for
 * Direction of Arrival (DOA) estimation using pre-computed steering vectors.
 * It outputs target angles (azimuth, elevation, and optional power) and a
 * target index map.
 *
 * \b Limitations:
 *
 *      - Maximum detections/targets: 8192
 *      - Virtual channels: numTxAntennas x numRxAntennas <= 64, must be divisible by 8
 *      - numAzimuthBins <= 256
 *      - numElevationBins <= 255
 *      - Full 2D mode: numAzimuthBins x numElevationBins <= 2048
 *
 * \b Separable mode:
 *
 * When separableParams.enableSeparable is true, uses a 2-step search
 * (azimuth at el~=0, then neighborhood x all elevations) for reduced
 * complexity: O(numAz + (2N+1) x numEl) vs O(numAz x numEl).
 * See \ref PVASeparableBartlettParams for configuration.
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] snapshotsTensorRequirements Snapshots tensor requirements.
 *                     + Data Layout: [CHW] — C: number of detections (numDetections); H: number of TX antennas (numTxAntennas); W: number of RX antennas (numRxAntennas).
 *                     + Rank: 3
 *                     + Data Type: NVCV_DATA_TYPE_2S32 (Complex SQ11.20 fixed-point).
 *
 * @param [in] detectionListTensorRequirements Range-Doppler Detection List tensor requirements.
 *                     + Data Layout: [HW] — H: number of detections (numDetections); W: 2 (rangeIdx, dopplerIdx).
 *                     + Rank: 2
 *                     + Data Type: NVCV_DATA_TYPE_S32.
 *
 * @param [in] ddmDopplerOffsetsTensorRequirements DDM Doppler Offsets tensor requirements.
 *                     + Data Layout: [W] — W: number of Doppler folds (numDopplerFolds), typically 1.
 *                     + Rank: 1
 *                     + Data Type: NVCV_DATA_TYPE_F32.
 *
 * @param [in] targetAnglesTensorRequirements Target angles tensor requirements.
 *                     + Data Layout: [HW] — H: 2 (without peak power) or 3 (with peak power); W: maximum number of targets (capacity).
 *                     + Rank: 2
 *                     + Data Type: NVCV_DATA_TYPE_F32.
 *                     + Target angles in SOA form — row 0: azimuth (degrees); row 1: elevation (degrees); row 2: optional peak power (dB). Valid target count is in targetCount tensor.
 *
 * @param [in] targetIndexMapTensorRequirements Target index map tensor requirements.
 *                     + Data Layout: [W] — W: maximum number of targets (capacity).
 *                     + Rank: 1
 *                     + Data Type: NVCV_DATA_TYPE_S32.
 *                     + Maps each target to its corresponding detection index.
 *
 * @param [in] params Pointer to the Bartlett Beamforming parameters. \ref PVADoaBartlettBeamformingParams.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT     Handle is null or some parameter is outside valid range.
 * @retval NVCV_ERROR_OUT_OF_MEMORY        Not enough memory to create the operator.
 * @retval NVCV_ERROR_INVALID_IMAGE_FORMAT Image format is invalid.
 * @retval NVCV_SUCCESS                    Operation executed successfully.
 */
NVCVStatus pvaDoaBartlettBeamformingCreate(NVCVOperatorHandle *handle,
                                           NVCVTensorRequirements const *const snapshotsTensorRequirements,
                                           NVCVTensorRequirements const *const detectionListTensorRequirements,
                                           NVCVTensorRequirements const *const ddmDopplerOffsetsTensorRequirements,
                                           NVCVTensorRequirements const *const targetAnglesTensorRequirements,
                                           NVCVTensorRequirements const *const targetIndexMapTensorRequirements,
                                           PVADoaBartlettBeamformingParams const *const params);

/**
 * Computes steering vectors for Bartlett beamforming DOA estimation.
 *
 * This utility function computes steering vectors based on virtual antenna array locations
 * and angle bins, and fills the provided tensor. The steering vectors can be reused
 * across multiple operator submissions as long as the antenna configuration and
 * angle bins remain unchanged.
 *
 * The complex value e^(-j·2π·phase) is quantized to Q15, then its complex
 * conjugate is stored to match the Bartlett operator convention, where
 * phase = vcPosition.x · cos(elevation) · sin(azimuth) + vcPosition.y · sin(elevation)
 * Virtual antenna array locations are in wavelengths.
 *
 * @param [in] virtualArrayLocations Virtual antenna array locations tensor (in wavelengths).
 *                                  Shape: [numTxAntennas, numRxAntennas].
 *                                  Data Type: NVCV_DATA_TYPE_2F32 (2-channel float32, representing [x, y] coordinates).
 *                                  For each TX-RX antenna pair, Channel 0: x position, Channel 1: y position.
 *                                  + Must not be NULL.
 *                                  + numTxAntennas × numRxAntennas ≤ 64 (maximum 64 virtual channels).
 *                                  + numTxAntennas × numRxAntennas must be divisible by 8.
 *
 * @param [in] azimuthBins Azimuth angle bins tensor (degrees).
 *                         Data Layout: [W] where W = numAzimuthBins.
 *                         Data Type: NVCV_DATA_TYPE_F32.
 *                         + Must not be NULL.
 *
 * @param [in] elevationBins Elevation angle bins tensor (degrees).
 *                           Data Layout: [W] where W = numElevationBins.
 *                           Data Type: NVCV_DATA_TYPE_F32.
 *                           + Must not be NULL.
 *
 * @param [out] steeringVectors Output tensor: Steering vectors.
 *                              Data Layout: [numAzimuthBins][numElevationBins][numTxAntennas][numRxAntennas].
 *                              Rank: 4. Data Type: NVCV_DATA_TYPE_2S16 (Complex Q15 fixed-point).
 *                              + Must not be NULL.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT     Some parameter is NULL or outside valid range, or tensor doesn't match requirements.
 * @retval NVCV_ERROR_INTERNAL             Internal error during computation.
 * @retval NVCV_SUCCESS                    Operation executed successfully.
 */
NVCVStatus pvaDoaBartlettBeamformingComputeSteeringVectors(NVCVTensorHandle virtualArrayLocations,
                                                           NVCVTensorHandle azimuthBins, NVCVTensorHandle elevationBins,
                                                           NVCVTensorHandle steeringVectors);

#ifdef __cplusplus
}

/**
 * Submits the Bartlett Beamforming DOA operator to a cuPVA stream.
 *
 * Performs beamforming on the provided snapshots using pre-computed steering vectors
 * to estimate azimuth and elevation angles for each detection.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 *
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] detectionCount Input tensor: Detection Count.
 *                            Data Layout: [W] where W=1.
 *                            Data Type: NVCV_DATA_TYPE_S32.
 *
 * @param [in] snapshots Input tensor: Snapshots.
 *                       Data Layout: [CHW] where C=numDetections, H=numTxAntennas, W=numRxAntennas.
 *                       Data Type: NVCV_DATA_TYPE_2S32 (Complex SQ11.20 fixed-point).
 *                       Antenna snapshots extracted from range-Doppler map for each detection.
 *
 * @param [in] steeringVectors Input tensor: Pre-computed steering vectors.
 *                             Data Layout: [numAzimuthBins][numElevationBins][numTxAntennas][numRxAntennas].
 *                             Rank: 4. Data Type: NVCV_DATA_TYPE_2S16 (Complex Q15 fixed-point).
 *                             + Must not be NULL.
 *                             + Must be computed using pvaDoaBartlettBeamformingComputeSteeringVectors.
 *                             + Must match the antenna configuration and angle bins.
 *
 * @param [in] azimuthBins Input tensor: Azimuth angle bins for peak index to angle mapping.
 *                         Data Layout: [W] where W = numAzimuthBins.
 *                         Data Type: NVCV_DATA_TYPE_F32.
 *                         + Must not be NULL.
 *                         + Must match the azimuth bins used to compute steering vectors.
 *
 * @param [in] elevationBins Input tensor: Elevation angle bins for peak index to angle mapping.
 *                           Data Layout: [W] where W = numElevationBins.
 *                           Data Type: NVCV_DATA_TYPE_F32.
 *                           + Must not be NULL.
 *                           + Must match the elevation bins used to compute steering vectors.
 *
 * @param [in] detectionList Input tensor: Range-Doppler Detection List.
 *                           Data Layout: [HW] where H=numDetections, W=2 (rangeIdx, dopplerIdx).
 *                           Data Type: NVCV_DATA_TYPE_S32.
 *
 * @param [in] ddmDopplerOffsets Input tensor: DDM Doppler Offsets.
 *                               Data Layout: [W] where W=numDopplerFolds (typically 1).
 *                               Data Type: NVCV_DATA_TYPE_F32.
 *
 * @param [out] targetCount Output tensor: Target Count.
 *                          Data Layout: [W] where W=1.
 *                          Data Type: NVCV_DATA_TYPE_S32.
 *
 * @param [out] targetAngles Output tensor: Target Angles.
 *                           Data Layout: [HW] where H=2 (without peak power) or 3 (with peak power),
 *                           W=Maximum number of targets (capacity).
 *                           Data Type: NVCV_DATA_TYPE_F32.
 *                           Actual valid target count is returned in targetCount tensor.
 *                           - Row 0: Azimuth angles (degrees)
 *                           - Row 1: Elevation angles (degrees)
 *                           - Row 2: Optional: Peak power (dB)
 *
 * @param [out] targetIndexMap Output tensor: Target Index Map.
 *                             Data Layout: [W] where W=Maximum number of targets (capacity).
 *                             Data Type: NVCV_DATA_TYPE_S32.
 *                             Maps each target to its corresponding detection index.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null or steering vectors don't match operator configuration.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaDoaBartlettBeamformingSubmit(NVCVOperatorHandle handle, cupvaStream_t stream,
                                           NVCVTensorHandle const detectionCount, NVCVTensorHandle const snapshots,
                                           NVCVTensorHandle const steeringVectors, NVCVTensorHandle const azimuthBins,
                                           NVCVTensorHandle const elevationBins, NVCVTensorHandle const detectionList,
                                           NVCVTensorHandle const ddmDopplerOffsets, NVCVTensorHandle targetCount,
                                           NVCVTensorHandle targetAngles, NVCVTensorHandle targetIndexMap);

/**
 * Submits the Bartlett Beamforming DOA operator to a CUDA stream.
 *
 * Performs beamforming on the provided snapshots using pre-computed steering vectors
 * to estimate azimuth and elevation angles for each detection.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] detectionCount Input tensor: Detection Count.
 *                            Data Layout: [W] where W=1.
 *                            Data Type: NVCV_DATA_TYPE_S32.
 * @param [in] snapshots Input tensor: Snapshots.
 *                       Data Layout: [CHW] where C=numDetections, H=numTxAntennas, W=numRxAntennas.
 *                       Data Type: NVCV_DATA_TYPE_2S32 (Complex SQ11.20 fixed-point).
 *                       Antenna snapshots extracted from range-Doppler map for each detection.
 * @param [in] steeringVectors Input tensor: Pre-computed steering vectors.
 *                             Data Layout: [numAzimuthBins][numElevationBins][numTxAntennas][numRxAntennas].
 *                             Rank: 4. Data Type: NVCV_DATA_TYPE_2S16 (Complex Q15 fixed-point).
 *                             + Must not be NULL.
 *                             + Must be computed using pvaDoaBartlettBeamformingComputeSteeringVectors.
 *                             + Must match the antenna configuration and angle bins.
 * @param [in] azimuthBins Input tensor: Azimuth angle bins for peak index to angle mapping.
 *                         Data Layout: [W] where W = numAzimuthBins.
 *                         Data Type: NVCV_DATA_TYPE_F32.
 *                         + Must not be NULL.
 *                         + Must match the azimuth bins used to compute steering vectors.
 * @param [in] elevationBins Input tensor: Elevation angle bins for peak index to angle mapping.
 *                           Data Layout: [W] where W = numElevationBins.
 *                           Data Type: NVCV_DATA_TYPE_F32.
 *                           + Must not be NULL.
 *                           + Must match the elevation bins used to compute steering vectors.
 * @param [in] detectionList Input tensor: Range-Doppler Detection List.
 *                           Data Layout: [HW] where H=numDetections, W=2 (rangeIdx, dopplerIdx).
 *                           Data Type: NVCV_DATA_TYPE_S32.
 * @param [in] ddmDopplerOffsets Input tensor: DDM Doppler Offsets.
 *                               Data Layout: [W] where W=numDopplerFolds (typically 1).
 *                               Data Type: NVCV_DATA_TYPE_F32.
 * @param [out] targetCount Output tensor: Target Count.
 *                          Data Layout: [W] where W=1.
 *                          Data Type: NVCV_DATA_TYPE_S32.
 * @param [out] targetAngles Output tensor: Target Angles.
 *                           Data Layout: [HW] where H=2 (without peak power) or 3 (with peak power),
 *                           W=Maximum number of targets (capacity).
 *                           Data Type: NVCV_DATA_TYPE_F32.
 *                           - Row 0: Azimuth angles (degrees)
 *                           - Row 1: Elevation angles (degrees)
 *                           - Row 2: Optional: Peak power (dB)
 * @param [out] targetIndexMap Output tensor: Target Index Map.
 *                             Data Layout: [W] where W=Maximum number of targets (capacity).
 *                             Data Type: NVCV_DATA_TYPE_S32.
 *                             Maps each target to its corresponding detection index.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null or steering vectors don't match operator configuration.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaDoaBartlettBeamformingSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                           NVCVTensorHandle const detectionCount, NVCVTensorHandle const snapshots,
                                           NVCVTensorHandle const steeringVectors, NVCVTensorHandle const azimuthBins,
                                           NVCVTensorHandle const elevationBins, NVCVTensorHandle const detectionList,
                                           NVCVTensorHandle const ddmDopplerOffsets, NVCVTensorHandle targetCount,
                                           NVCVTensorHandle targetAngles, NVCVTensorHandle targetIndexMap);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPDOABARTLETTBEAMFORMING_H */
