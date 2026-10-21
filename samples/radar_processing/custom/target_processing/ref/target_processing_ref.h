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

#include <PvaOperatorTypes.h>
#include <doa_angle_fft_ref.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
#    include <cstdint>
#    include <vector>

// Structure to hold target processing results
struct TargetProcessingResult
{
    std::vector<float> velocityValues;   // Velocity values in m/s
    std::vector<float> rangeValues;      // Range values in meters
    std::vector<float> azimuthDegrees;   // Azimuth angles in degrees
    std::vector<float> elevationDegrees; // Elevation angles in degrees
    std::vector<float> xCoordinates;     // X coordinates in meters
    std::vector<float> yCoordinates;     // Y coordinates in meters
    std::vector<float> zCoordinates;     // Z coordinates in meters
    std::vector<float> peakPowerDb;      // Peak power values in dB
};
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generate random input data for target processing
 *
 * @param detectionListTensorHandle Detection list tensor [numDetections, 2] - S32 (rangeIdx, dopplerIdx)
 * @param anglesTensorHandle Angles tensor [3, numTargets] - F32 (Azimuth, Elevation, Power)
 */
void generateTargetProcessingInput(NVCVTensorHandle detectionListTensorHandle, NVCVTensorHandle anglesTensorHandle);

/**
 * @brief Generate detection index map
 *
 * @param detectionIndexMapTensorHandle Detection index map tensor [numTargets] - S32
 * @param numDetections Number of detections
 * @param numTargets Number of targets (>= numDetections)
 */
void generateDetectionIndexMap(NVCVTensorHandle detectionIndexMapTensorHandle, int32_t numDetections,
                               int32_t numTargets);

/**
 * @brief Reference implementation for Target Processing
 *
 * @param targetCountTensorHandle Input: Target count [1] - S32
 * @param detectionListTensorHandle Input: Detection list [numDetections, 2] - S32
 * @param detectionIndexMapTensorHandle Input: Detection index map [numTargets] - S32
 * @param anglesTensorHandle Input: Angles [3, numTargets] - F32 (Azimuth, Elevation, Power)
 * @param ddmDopplerOffsetsTensorHandle Input: DDM Doppler offsets [numDopplerFolds] - F32
 * @param nciTensorHandle Input: NCI tensor for quadratic interpolation [nciHeight, nciWidth] - U32 (can be NULL)
 * @param targetListTensorHandle Output: Target list [8, numTargets] - F32
 * @param params Target processing parameters
 */
void targetProcessingRef(NVCVTensorHandle targetCountTensorHandle, NVCVTensorHandle detectionListTensorHandle,
                         NVCVTensorHandle detectionIndexMapTensorHandle, NVCVTensorHandle anglesTensorHandle,
                         NVCVTensorHandle ddmDopplerOffsetsTensorHandle, NVCVTensorHandle nciTensorHandle,
                         NVCVTensorHandle targetListTensorHandle, const PVATargetProcessingParams *params);

/**
 * @brief Check target processing output against reference
 *
 * @param targetCountTensorHandle Target count tensor
 * @param pvaTargetListTensorHandle PVA output target list tensor
 * @param refTargetListTensorHandle Reference target list tensor
 * @return 0 on success, non-zero on mismatch
 */
int32_t checkTargetProcessingOutput(NVCVTensorHandle targetCountTensorHandle,
                                    NVCVTensorHandle pvaTargetListTensorHandle,
                                    NVCVTensorHandle refTargetListTensorHandle);

#ifdef __cplusplus
}

// Target Processing (Fixed-Point) - Compute range, range-rate, and cartesian coordinates
// Note: This is essentially floating-point math, so it's identical to the floating-point version
// Supports both standard mode and RV decoupling mode based on PVATargetProcessingParams
// When enableQuadraticInterpolation is true, uses NCI data for sub-bin accuracy
// Optional waveParam: When provided (non-nullptr), use these pre-computed waveform parameters
// instead of computing from PVATargetProcessingParams (for MATLAB exact match testing)
TargetProcessingResult TargetProcessingFixedPoint(const std::vector<std::pair<int32_t, int32_t>> &tx1DetectionList,
                                                  const DoaResult &doaResult, float ddmDopplerOffset,
                                                  const PVATargetProcessingParams *params,
                                                  const uint32_t *nciData = nullptr, int32_t nciWidth = 0,
                                                  int32_t nciHeight = 0);
#endif
