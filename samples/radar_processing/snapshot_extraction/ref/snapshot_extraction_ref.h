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

#include <OpSnapshotExtraction.h>
#include <PvaOperatorTypes.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
#    include <complex>
#    include <cstdint>
#    include <vector>

// Spatial data extraction result structure
typedef struct SnapshotExtractionResultRec
{
    std::vector<std::vector<std::vector<std::complex<int32_t>>>> snapshots; // Virtual elements [detections x TX x RX]
    std::vector<std::pair<int32_t, int32_t>> tx1DetectionList;              // Detection list for TX1 [range, doppler]
} SnapshotExtractionResult;
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generate random input data for snapshot extraction
 *
 * @param detectionCountTensorHandle Detection count tensor [1] - S32
 * @param foldedDetectionListTensorHandle Folded detection list tensor [numDetections, 2] - S32
 * @param ddmDopplerOffsetsTensorHandle DDM Doppler offsets tensor [numDopplerFolds] - F32
 * @param rangeDopplerMapTensorHandle Range-Doppler map tensor [numRangeBins, numRx, numDopplerBins] - Complex S32
 * @param nciRxTensorHandle NCI Rx tensor [numRangeBins, numDopplerBins] - U32
 * @param calibrationWeightsTensorHandle Optional calibration weights tensor [numTx, numRx] - Complex S32 (NULL = skip)
 * @param detectionCount Number of detections (value to write to detection count tensor)
 * @param numDopplerFolds Number of doppler folds (for DDM disambiguation)
 * @param numRangeBins Number of range bins
 * @param numDopplerBins Number of doppler bins
 */
void generateSnapshotExtractionInput(NVCVTensorHandle detectionCountTensorHandle,
                                     NVCVTensorHandle foldedDetectionListTensorHandle,
                                     NVCVTensorHandle ddmDopplerOffsetsTensorHandle,
                                     NVCVTensorHandle rangeDopplerMapTensorHandle, NVCVTensorHandle nciRxTensorHandle,
                                     NVCVTensorHandle calibrationWeightsTensorHandle, int32_t detectionCount,
                                     int32_t numDopplerFolds, int32_t numRangeBins, int32_t numDopplerBins);

/**
 * @brief Generate calibration weights for antenna array (fixed-point Complex S32)
 *
 * Populates the calibration weights tensor with random phase/amplitude variations.
 * Call this when using calibration vector with snapshot extraction (e.g. in pipelines
 * that already have other inputs from prior stages).
 *
 * @param calibrationWeightsTensorHandle Calibration weights tensor [numTx, numRx] - Complex S32
 */
void generateCalibrationWeights(NVCVTensorHandle calibrationWeightsTensorHandle);

/**
 * @brief Generate DDM Doppler offsets for antenna array (fixed-point F32)
 *
 * Populates the DDM Doppler offsets tensor with random values.
 * Call this when using DDM with snapshot extraction (e.g. in pipelines
 * that already have other inputs from prior stages).
 *
 * @param ddmDopplerOffsetsTensorHandle DDM Doppler offsets tensor [numDopplerFolds] - F32
 * @param numDopplerFolds Number of doppler folds (for DDM disambiguation)
 */
void generateDdmDopplerOffsets(NVCVTensorHandle ddmDopplerOffsetsTensorHandle, int32_t numDopplerFolds);

#ifdef __cplusplus
/**
 * @brief Load DDM Doppler offsets
 *
 * @param ddmDopplerOffsetsTensorHandle DDM Doppler offsets tensor [numDopplerFolds] - F32
 * @param ddmDopplerOffsets DDM Doppler offsets
 */
void loadDdmDopplerOffsets(NVCVTensorHandle ddmDopplerOffsetsTensorHandle, const std::vector<float> &ddmDopplerOffsets);

/**
 * @brief Load unit calibration weights (real=2^28, imag=0) for DDM bypass
 *
 * @param calibrationWeightsTensorHandle Calibration weights tensor [numTx, numRx] - Complex S32
 * @param numTx Number of transmit antennas
 * @param numRx Number of receive antennas
 */
void loadCalibrationWeights(NVCVTensorHandle calibrationWeightsTensorHandle, int32_t numTx, int32_t numRx);
#endif

/**
 * @brief Reference implementation for Snapshot Extraction operator
 *
 * @param detectionCountTensorHandle Input: Detection count [1] - S32
 * @param foldedDetectionListTensorHandle Input: Folded detection list [numDetections, 2] - S32
 * @param ddmDopplerOffsetsTensorHandle Input: DDM Doppler offsets [numDopplerFolds] - F32
 * @param nciRxTensorHandle Input: NCI Rx [numRangeBins, numDopplerBins] - U32
 * @param rangeDopplerMapTensorHandle Input: Range-Doppler map [numRangeBins, numRx, numDopplerBins] - Complex S32
 * @param calibrationWeightsTensorHandle Input: Calibration weights [numTx, numRx] - Complex S32 (can be NULL)
 * @param detectionListTensorHandle Output: Unfolded detection list [numDetections, 2] - S32
 * @param snapshotsTensorHandle Output: Snapshots [numDetections, numVirtualAntennas] - Complex S32
 * @param params Snapshot extraction parameters
 */
void snapshotExtractionRef(NVCVTensorHandle detectionCountTensorHandle,
                           NVCVTensorHandle foldedDetectionListTensorHandle,
                           NVCVTensorHandle ddmDopplerOffsetsTensorHandle, NVCVTensorHandle nciRxTensorHandle,
                           NVCVTensorHandle rangeDopplerMapTensorHandle,
                           NVCVTensorHandle calibrationWeightsTensorHandle, NVCVTensorHandle detectionListTensorHandle,
                           NVCVTensorHandle snapshotsTensorHandle, const PVASnapshotExtractionParams *params);

/**
 * @brief Check snapshot extraction output against reference
 *
 * @param pvaDetectionCountTensorHandle PVA detection count tensor
 * @param refDetectionCountTensorHandle Reference detection count tensor
 * @param pvaDetectionListTensorHandle PVA operator output detection list tensor
 * @param refDetectionListTensorHandle Reference detection list tensor
 * @param pvaSnapshotsTensorHandle PVA operator output snapshots tensor
 * @param refSnapshotsTensorHandle Reference snapshots tensor
 * @return 0 on success, non-zero on mismatch
 */
int32_t checkSnapshotExtractionOutput(NVCVTensorHandle pvaDetectionCountTensorHandle,
                                      NVCVTensorHandle refDetectionCountTensorHandle,
                                      NVCVTensorHandle pvaDetectionListTensorHandle,
                                      NVCVTensorHandle refDetectionListTensorHandle,
                                      NVCVTensorHandle pvaSnapshotsTensorHandle,
                                      NVCVTensorHandle refSnapshotsTensorHandle);

#ifdef __cplusplus
}

constexpr int32_t DDMA_WEIGHTS_QBITS = 16;
constexpr int64_t DDMA_WEIGHTS_SCALE = (1LL << DDMA_WEIGHTS_QBITS);

constexpr int32_t CALIBRATION_WEIGHTS_QBITS = 28;
constexpr int64_t CALIBRATION_WEIGHTS_SCALE = (1LL << CALIBRATION_WEIGHTS_QBITS);

// C++ Helper function for snapshot extraction (similar to DoaEstimationFixedPoint)
SnapshotExtractionResult SnapshotExtractionFixedPoint(
    const std::vector<std::pair<int32_t, int32_t>> &foldedDetectionList, const float *ddmDopplerOffsets,
    int32_t numDopplerFolds, const uint32_t *nciRxPtr, int32_t numRangeBins, int32_t numDopplerBins,
    const int32_t *rangeDopplerMapPtr, int32_t numRx, int32_t numTx, PVAFFTLayout rangeDopplerFormat,
    const int32_t *calibrationWeightsPtr = nullptr);

#endif
