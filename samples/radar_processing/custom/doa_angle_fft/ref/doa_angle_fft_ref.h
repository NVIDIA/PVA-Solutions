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

#include <OpDoaAngleFFT.h>
#include <PvaOperatorTypes.h>
#include <nvcv/Tensor.h>
#include <snapshot_extraction_ref.h>

#ifdef __cplusplus
#    include <cstdint>
#    include <vector>

// Structure to hold DOA estimation results
struct DoaResult
{
    std::vector<float> azimuthDegrees;   // Azimuth angles in degrees
    std::vector<float> elevationDegrees; // Elevation angles in degrees
    std::vector<float> peakPowerDb;      // Peak power values in dB
};
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generate random input data for DOA Angle FFT snapshots
 *
 * @param snapshotsTensorHandle Handle to snapshots tensor [numDetections, numVirtualAntennas] - Complex S32
 * @param inDetectionCountTensorHandle Handle to detection count tensor [1] - S32
 * @param detectionCount Detection count
 */
void generateDoaAngleFFTInput(NVCVTensorHandle snapshotsTensorHandle, NVCVTensorHandle inDetectionCountTensorHandle,
                              int32_t detectionCount);
void generateDetectionListInput(NVCVTensorHandle detectionListTensorHandle);

/**
 * @brief Reference implementation for DOA Angle FFT operator (DOA-only)
 *
 * This function performs DoA angle estimation and outputs intermediate results
 * (targetAngles, targetIndexMap, targetCount). Target processing should be
 * performed separately using targetProcessingRef().
 *
 * @param detectionCountTensorHandle Input: Detection count [1] - S32
 * @param snapshotsTensorHandle Input: Snapshots tensor [numDetections, numVirtualAntennas] - Complex S32
 * @param targetCountTensorHandle Output: Target count [1] - S32
 * @param targetAnglesTensorHandle Output: Target angles [3, numTargets] - F32 (azimuth, elevation, power)
 * @param targetIndexMapTensorHandle Output: Target index map [numTargets] - S32 (maps targets to detections)
 * @param params DOA Angle FFT parameters
 */
void doaAngleFFTRef(NVCVTensorHandle detectionCountTensorHandle, NVCVTensorHandle snapshotsTensorHandle,
                    NVCVTensorHandle targetCountTensorHandle, NVCVTensorHandle targetAnglesTensorHandle,
                    NVCVTensorHandle targetIndexMapTensorHandle, const PVADoaAngleFFTParams *params);

#ifdef __cplusplus
}

// Data structures for intermediate results
struct Snapshot
{
    std::vector<std::vector<std::pair<int32_t, int32_t>>> data;
    int32_t numAzimuthVirtualElements;
    int32_t numElevationVirtualElements;
};

struct FFTData
{
    std::vector<std::vector<int32_t>> data;
    int32_t paddedAzimuthSize;
    int32_t paddedElevationSize;
};

struct PowerSpectrum
{
    std::vector<std::vector<float>> data;
    float maxPower;
    int32_t peakAzimuthIndex;
    int32_t peakElevationIndex;
};

FFTData Windowing2D(const Snapshot &snapshot, int32_t paddedAzimuthSize, int32_t paddedElevationSize);

void FFT2D(FFTData &fftData);

PowerSpectrum PowerSpectrumPeakDetection(const FFTData &fftData);

std::pair<int32_t, int32_t> QuadraticInterpolation(const PowerSpectrum &powerSpec);

std::pair<int32_t, int32_t> ConvertToAngles(const std::pair<int32_t, int32_t> &interpIndices,
                                            const std::pair<std::vector<int32_t>, std::vector<int32_t>> &angleBins,
                                            int32_t paddedAzimuthSize, int32_t paddedElevationSize);

// Helper function to initialize fixed-point angle bins
std::pair<std::vector<int32_t>, std::vector<int32_t>> InitializeAngleBins(int32_t paddedAzimuthSize,
                                                                          int32_t paddedElevationSize);

// Main DOA Estimation function
DoaResult DoaEstimationFixedPoint(int32_t *snapshotsPtr, int32_t numTargets, int32_t numTx, int32_t numRx,
                                  int32_t numVirtualAzimuthElements, int32_t numVirtualElevationElements,
                                  PVADoaAngleFFTPrecision angleEstimationPrecision);
#endif
