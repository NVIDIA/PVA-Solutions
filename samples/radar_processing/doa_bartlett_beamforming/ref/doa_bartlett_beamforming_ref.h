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

#pragma once

#include "bartlett_params.h"

#include <OpDoaBartlettBeamforming.h>
#include <PvaOperatorTypes.h>
#include <doa_angle_fft_ref.h>
#include <nvcv/Tensor.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

void generateDoaBartlettBeamformingInput(NVCVTensorHandle snapshotsTensorHandle);

/**
 * @brief Reference implementation for DOA Bartlett Beamforming operator (DOA-only)
 *
 * This function performs DoA angle estimation using Bartlett beamforming and outputs
 * intermediate results (targetAngles, targetIndexMap, targetCount). Target processing
 * should be performed separately using targetProcessingRef().
 *
 * @param detectionCountTensorHandle Input: Detection count [1] - S32
 * @param snapshotsTensorHandle Input: Snapshots tensor [numDetections, numTx, numRx] - Complex S32
 * @param steeringVectorsTensorHandle Input: Pre-computed steering vectors
 * @param azimuthBinsTensorHandle Input: Azimuth angle bins (degrees)
 * @param elevationBinsTensorHandle Input: Elevation angle bins (degrees)
 * @param targetCountTensorHandle Output: Target count [1] - S32
 * @param targetAnglesTensorHandle Output: Target angles [3, numTargets] - F32 (azimuth, elevation, power)
 * @param targetIndexMapTensorHandle Output: Target index map [numTargets] - S32 (maps targets to detections)
 * @param params Bartlett Beamforming parameters
 */
void doaBartlettBeamformingRef(NVCVTensorHandle detectionCountTensorHandle, NVCVTensorHandle snapshotsTensorHandle,
                               NVCVTensorHandle steeringVectorsTensorHandle, NVCVTensorHandle azimuthBinsTensorHandle,
                               NVCVTensorHandle elevationBinsTensorHandle, NVCVTensorHandle targetCountTensorHandle,
                               NVCVTensorHandle targetAnglesTensorHandle, NVCVTensorHandle targetIndexMapTensorHandle,
                               const PVADoaBartlettBeamformingParams *params);

/**
 * @brief Compute steering vectors from virtual antenna array locations tensor
 * @param steeringVectorsTensorHandle Output tensor for steering vectors with 2S16 (complex Q15)
 *                                    Data Layout: [numAzimuthBins][numElevationBins][numTx][numRx]
 *                                    Rank: 4
 * @param virtualArrayLocationsTensorHandle Input tensor for virtual antenna array locations
 *                                Shape: [numTxAntennas, numRxAntennas]
 *                                Data Type: NVCV_DATA_TYPE_2F32 (2 channels: [x, y] coordinates)
 *                                Memory: channel 0 = x, channel 1 = y
 *                                numTx and numRx are extracted from tensor shape
 * @param azimuthBinsTensorHandle Azimuth angle bins tensor in degrees
 *                                Data Layout: [W] where W = numAzimuthBins
 *                                Data Type: NVCV_DATA_TYPE_F32
 * @param elevationBinsTensorHandle Elevation angle bins tensor in degrees
 *                                  Data Layout: [W] where W = numElevationBins
 *                                  Data Type: NVCV_DATA_TYPE_F32
 */
void computeSteeringVectorsRef(NVCVTensorHandle steeringVectorsTensorHandle,
                               NVCVTensorHandle virtualArrayLocationsTensorHandle,
                               NVCVTensorHandle azimuthBinsTensorHandle, NVCVTensorHandle elevationBinsTensorHandle);

#ifdef __cplusplus
}
#endif

struct SteeringVectors
{
    std::vector<std::vector<std::pair<int16_t, int16_t>>> data;
    int32_t numVirtualChannels;
    int32_t numAngleBins;
};

struct BeamformerOutput
{
    std::vector<std::pair<int64_t, int64_t>> data;
    int32_t numAngleBins;
};

struct BartlettPowerSpectrum
{
    std::vector<float> data;
    std::vector<float> azimuthBins;
    std::vector<float> elevationBins;
    float maxPowerFloat;
    int32_t peakAngleIndex;
    int32_t numAzimuthBins;
    int32_t numElevationBins;
};

BeamformerOutput ComputeBeamforming(const int32_t *snapshotPtr, int32_t numTx, int32_t numRx,
                                    const int16_t *steeringVectorsPtr, int32_t numAzimuthBins,
                                    int32_t numElevationBins);

BartlettPowerSpectrum ComputeBartlettPowerSpectrum(const BeamformerOutput &beamformerOutput, const float *azimuthBins,
                                                   int32_t numAzimuthBins, const float *elevationBins,
                                                   int32_t numElevationBins);

std::pair<float, float> FindPeakWithInterpolation(const BartlettPowerSpectrum &powerSpec, bool enableInterpolation);

DoaResult DoaEstimationBartlettFixedPoint(const int32_t *snapshotsPtr, int32_t numDetections,
                                          int32_t numVirtualChannels, int32_t numTx, int32_t numRx,
                                          const int16_t *steeringVectorsPtr, const float *azimuthBins,
                                          int32_t numAzimuthBins, const float *elevationBins, int32_t numElevationBins,
                                          bool enableInterpolation);

/**
 * @brief Separable Bartlett DOA estimation using fixed-point beamforming
 *
 * This implements a two-step separable search for reduced complexity:
 * Step 1: Estimate azimuth with elevation fixed at 0 (or nearest bin)
 * Step 2: Estimate elevation using azimuth neighborhood search
 * Step 3: Apply interpolation (separate 1D or 2D on local 3x3)
 *
 * Complexity: O(numAz × numVC + numLocalAz × numEl × numVC) vs O(numAz × numEl × numVC)
 * Typical speedup: ~8x for 121 azimuth × 16 elevation bins
 *
 * @param snapshotsPtr Pointer to snapshot data [numDetections][numVC][2] complex int32
 * @param numDetections Number of detections to process
 * @param numVirtualChannels Number of virtual channels (numTx × numRx)
 * @param numTx Number of TX antennas
 * @param numRx Number of RX antennas
 * @param steeringVectorsPtr Pre-computed steering vectors [numAz][numEl][numTx][numRx] Q15
 * @param azimuthBins Azimuth angle bins (degrees)
 * @param numAzimuthBins Number of azimuth bins
 * @param elevationBins Elevation angle bins (degrees)
 * @param numElevationBins Number of elevation bins
 * @param azNeighborhood Azimuth neighborhood size for Step 2 (typically 3)
 * @param azFromEl0 If true, use azimuth from Step 1; if false, refine from Step 2
 * @param enableInterpolation Enable quadratic interpolation for sub-bin accuracy
 * @param azChannelMask Bitmask of VCs for Step 1 azimuth search (0 = no masking)
 * @return DoaResult with azimuth, elevation, and power for each detection
 */
DoaResult DoaEstimationBartlettSeparableFixedPoint(const int32_t *snapshotsPtr, int32_t numDetections,
                                                   int32_t numVirtualChannels, int32_t numTx, int32_t numRx,
                                                   const int16_t *steeringVectorsPtr, const float *azimuthBins,
                                                   int32_t numAzimuthBins, const float *elevationBins,
                                                   int32_t numElevationBins, int32_t azNeighborhood, bool azFromEl0,
                                                   bool enableInterpolation, uint64_t azChannelMask);

/**
 * @brief Load angle bins into a tensor
 *
 * @param angleBinsTensorHandle Tensor handle for angle bins
 * @param numBins Number of bins
 * @param start Start angle
 * @param step Step angle
 */
void loadAngleBins(NVCVTensorHandle angleBinsTensorHandle, int32_t numBins, float start, float step);

/**
 * @brief Copy steering vector data from a host pointer into the steering vectors tensor.
 * Tensor must be 4D with NVCV_DATA_TYPE_2S16.
 *
 * @param steeringVectorsTensorHandle Pre-constructed steering vectors tensor (PVA-allocated)
 * @param data Pointer to int16_t data (layout matching tensor: [numAz][numEl][numTx][numRx] * 2)
 * @param size Number of int16_t elements in data (must equal tensor numAz*numEl*numTx*numRx*2)
 * @throws std::runtime_error if size does not match the tensor element count
 */
void PopulateSteeringVectorsTensor(NVCVTensorHandle steeringVectorsTensorHandle, const int16_t *data, size_t size);

/**
 * @brief Copy virtual array locations data from a host pointer into the virtual array locations tensor.
 * Tensor must be 2D with NVCV_DATA_TYPE_2F32.
 *
 * @param virtualArrayLocationsTensorHandle Pre-constructed virtual array locations tensor (PVA-allocated)
 * @param numTx Number of TX antennas
 * @param numRx Number of RX antennas
 * @param azimuthAntennaSpacing Antenna spacing in wavelengths (TX)
 * @param elevationAntennaSpacing Antenna spacing in wavelengths (RX)
 */
void PopulateVirtualArrayLocationsTensor(NVCVTensorHandle virtualArrayLocationsTensorHandle, int32_t numTx,
                                         int32_t numRx, float azimuthAntennaSpacing, float elevationAntennaSpacing);