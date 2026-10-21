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
 * @file snapshot_extraction.cpp
 *
 * @brief Reference implementation for Snapshot Extraction operator
 *
 * This file provides a fixed-point reference implementation for the Snapshot Extraction
 * operation. This reference implementation is used for validation and corresponds to the
 * PVA operator defined in OpSnapshotExtraction.h with parameters PVASnapshotExtractionParams.
 *
 * The operator performs DDM (Doppler Division Multiplexing) disambiguation and gathers
 * antenna snapshots from the Range-Doppler map for detected targets.
 *
 * @see OpSnapshotExtraction.h for the PVA operator interface
 * @see PvaOperatorTypes.h for parameter structures
 */

#include "snapshot_extraction_ref.h"

#include <TensorUtils.hpp>

#include <cupva_host.h>
#include <nvcv/Tensor.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <iostream>
#include <limits>
#include <random>

// ============================================================================
// Public API Implementation using NVCVTensor
// ============================================================================

namespace {

template<typename T>
void generateDetectionListImpl(NVCVTensorHandle tensorHandle, int32_t numRangeBins, int32_t numChirpsPerFold)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);
    T *tensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&tensorPtr, (void *)tensorData.buffer.strided.basePtr);

    int64_t tensorShapeNCHW[4];
    nvcvTensorShapePermute(tensorData.layout, tensorData.shape, NVCV_TENSOR_NCHW, tensorShapeNCHW);

    const int32_t numDetections = tensorShapeNCHW[2];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int32_t> rangeDist(0, numRangeBins - 1);
    std::uniform_int_distribution<int32_t> dopplerDist(0, numChirpsPerFold - 1);

    for (int32_t i = 0; i < numDetections; i++)
    {
        tensorPtr[2 * i]     = rangeDist(gen);   // Range index
        tensorPtr[2 * i + 1] = dopplerDist(gen); // Doppler index
    }
}

template<typename T>
void generateComplexMapImpl(NVCVTensorHandle tensorHandle)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);
    T *tensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&tensorPtr, (void *)tensorData.buffer.strided.basePtr);

    int32_t bitsPerPixel;
    nvcvDataTypeGetBitsPerPixel(tensorData.dtype, &bitsPerPixel);
    int32_t elements = bitsPerPixel / 8 / sizeof(T);

    int64_t tensorShapeNCHW[4];
    nvcvTensorShapePermute(tensorData.layout, tensorData.shape, NVCV_TENSOR_NCHW, tensorShapeNCHW);

    const int32_t height   = tensorShapeNCHW[2];
    const int32_t channels = tensorShapeNCHW[1];
    const int32_t width    = tensorShapeNCHW[3];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int32_t> dist(-(1 << 19), (1 << 19) - 1);

    for (int32_t h = 0; h < height; h++)
    {
        for (int32_t c = 0; c < channels; c++)
        {
            for (int32_t w = 0; w < width; w++)
            {
                for (int32_t e = 0; e < elements; e++)
                {
                    tensorPtr[((h * channels + c) * width + w) * elements + e] = dist(gen);
                }
            }
        }
    }
}

template<typename T>
void generateNciImpl(NVCVTensorHandle tensorHandle)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);
    T *tensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&tensorPtr, (void *)tensorData.buffer.strided.basePtr);

    int64_t tensorShapeNCHW[4];
    nvcvTensorShapePermute(tensorData.layout, tensorData.shape, NVCV_TENSOR_NCHW, tensorShapeNCHW);

    const int32_t height = tensorShapeNCHW[2];
    const int32_t width  = tensorShapeNCHW[3];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dist(0, (1U << 20) - 1);

    for (int32_t h = 0; h < height; h++)
    {
        for (int32_t w = 0; w < width; w++)
        {
            tensorPtr[h * width + w] = dist(gen);
        }
    }
}

int32_t checkSnapshotExtractionOutputImpl(int32_t pvaDetectionCount, int32_t refDetectionCount, int32_t numTx,
                                          int32_t numRx, int32_t *pvaDetectionListPtr, int32_t *refDetectionListPtr,
                                          int32_t *pvaSnapshotsPtr, int32_t *refSnapshotsPtr)
{
    bool debug      = false;
    int32_t maxDiff = 1; // Maximum allowed difference for real and imaginary parts

    // Compare up to min(pva, ref) - extra PVA detections
    int32_t compareCount = (pvaDetectionCount < refDetectionCount) ? pvaDetectionCount : refDetectionCount;

    if (debug)
    {
        printf("\n========================================\n");
        printf("DEBUG: Snapshot Extraction Output Dump\n");
        printf("========================================\n");
        printf("Configuration:\n");
        printf("  PVA Detection Count: %d, Ref Detection Count: %d, Compare: %d\n", pvaDetectionCount,
               refDetectionCount, compareCount);
        printf("  Num Tx: %d\n", numTx);
        printf("  Num Rx: %d\n", numRx);
        printf("  Virtual Antennas: %d\n", numTx * numRx);
        printf("\n");
    }

    // Compare detection lists
    for (int32_t i = 0; i < compareCount; i++)
    {
        if (debug)
        {
            printf("Detection %d:\n", i);
            printf("  REF: range=%d, doppler=%d\n", refDetectionListPtr[2 * i], refDetectionListPtr[2 * i + 1]);
            printf("  PVA: range=%d, doppler=%d\n", pvaDetectionListPtr[2 * i], pvaDetectionListPtr[2 * i + 1]);
        }
        if (pvaDetectionListPtr[2 * i] != refDetectionListPtr[2 * i] ||
            pvaDetectionListPtr[2 * i + 1] != refDetectionListPtr[2 * i + 1])
        {
            printf("Detection list mismatch at index %d: ref = (%d, %d), pva = (%d, %d)\n", i,
                   refDetectionListPtr[2 * i], refDetectionListPtr[2 * i + 1], pvaDetectionListPtr[2 * i],
                   pvaDetectionListPtr[2 * i + 1]);
            return -1;
        }
    }

    if (debug)
    {
        printf("\n✓ All detection lists match\n\n");
    }

    // Compare snapshots
    for (int32_t i = 0; i < compareCount; i++)
    {
        if (debug)
        {
            printf("Snapshots for Detection %d:\n", i);
        }

        for (int32_t j = 0; j < numTx; j++)
        {
            if (debug)
            {
                printf("  Tx %d:\n", j);
            }

            for (int32_t k = 0; k < numRx; k++)
            {
                // Calculate linear index: [detection][tx][rx] in row-major order
                // Then multiply by 2 for complex (real, imag)
                int32_t snapshotIdx = 2 * (i * numTx * numRx + j * numRx + k);

                // Calculate absolute differences for both real and imaginary parts
                int32_t diffReal    = pvaSnapshotsPtr[snapshotIdx] - refSnapshotsPtr[snapshotIdx];
                int32_t diffImag    = pvaSnapshotsPtr[snapshotIdx + 1] - refSnapshotsPtr[snapshotIdx + 1];
                int32_t absDiffReal = (diffReal < 0) ? -diffReal : diffReal;
                int32_t absDiffImag = (diffImag < 0) ? -diffImag : diffImag;

                if (debug)
                {
                    printf("    Rx %d: REF=(%6d, %6d), PVA=(%6d, %6d)", k, refSnapshotsPtr[snapshotIdx],
                           refSnapshotsPtr[snapshotIdx + 1], pvaSnapshotsPtr[snapshotIdx],
                           pvaSnapshotsPtr[snapshotIdx + 1]);

                    if (absDiffReal <= maxDiff && absDiffImag <= maxDiff)
                    {
                        printf(" ✓\n");
                    }
                    else
                    {
                        printf(" ✗ MISMATCH (diff: real=%d, imag=%d)\n", diffReal, diffImag);
                    }
                }

                // Check if difference exceeds tolerance of maxDiff
                if (absDiffReal > maxDiff || absDiffImag > maxDiff)
                {
                    printf(
                        "Snapshots mismatch at detection %d, Tx %d, Rx %d: ref = (%d, %d), pva = (%d, %d), diff = (%d, "
                        "%d)\n",
                        i, j, k, refSnapshotsPtr[snapshotIdx], refSnapshotsPtr[snapshotIdx + 1],
                        pvaSnapshotsPtr[snapshotIdx], pvaSnapshotsPtr[snapshotIdx + 1], diffReal, diffImag);
                    return -1;
                }
            }
            if (debug)
            {
                printf("\n");
            }
        }
    }

    if (debug)
    {
        printf("✓ All snapshots match\n");
        printf("========================================\n\n");
    }

    // Report extra PVA detections
    if (pvaDetectionCount > refDetectionCount)
    {
        printf("Snapshot Extraction KPI: PVA has %d extra detections\n", pvaDetectionCount - refDetectionCount);
    }

    return 0;
}

/**
 * @brief Generate calibration weights for antenna array (fixed-point representation)
 *
 * This helper function generates example calibration weights in fixed-point format.
 * In a real application, these would be obtained from antenna calibration measurements.
 *
 * Generates random phase/amplitude variations simulating real calibration data.
 *
 * @param calibrationWeights Output buffer [numTx * numRx * 2] (complex int32)
 * @param numTx Number of TX antennas
 * @param numRx Number of RX antennas
 */
template<typename T>
void generateCalibrationWeightsImpl(NVCVTensorHandle tensorHandle)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);
    T *tensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&tensorPtr, (void *)tensorData.buffer.strided.basePtr);

    // Get tensor dimensions [numTx, numRx]
    int64_t tensorShapeHW[2];
    nvcvTensorShapePermute(tensorData.layout, tensorData.shape, NVCV_TENSOR_HW, tensorShapeHW);
    int32_t numTx = static_cast<int32_t>(tensorShapeHW[0]);
    int32_t numRx = static_cast<int32_t>(tensorShapeHW[1]);

    // Generate random calibration weights with small phase/amplitude variations
    // This simulates real antenna calibration data
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> amplitudeDist(0.9f, 1.1f); // ±10% amplitude variation
    std::uniform_real_distribution<float> phaseDist(-0.2f, 0.2f);    // ±0.2 radian phase variation

    for (int32_t txIdx = 0; txIdx < numTx; txIdx++)
    {
        for (int32_t rxIdx = 0; rxIdx < numRx; rxIdx++)
        {
            // Generate random amplitude and phase
            float amplitude = amplitudeDist(rng);
            float phase     = phaseDist(rng);

            // Convert to complex: amplitude * exp(j*phase) = amplitude * (cos(phase) + j*sin(phase))
            std::complex<float> weight(amplitude * std::cos(phase), amplitude * std::sin(phase));

            // Convert to fixed-point Q28 format
            int32_t idx        = (txIdx * numRx + rxIdx) * 2;
            tensorPtr[idx]     = static_cast<int32_t>(std::round(weight.real() * CALIBRATION_WEIGHTS_SCALE));
            tensorPtr[idx + 1] = static_cast<int32_t>(std::round(weight.imag() * CALIBRATION_WEIGHTS_SCALE));
        }
    }
}

template<typename T>
void generateDdmDopplerOffsetsImpl(NVCVTensorHandle ddmDopplerOffsetsTensorHandle, int32_t numDopplerFolds)
{
    // Set example DDM offsets
    std::vector<T> ddmDopplerOffsets(numDopplerFolds);
    for (int32_t i = 0; i < numDopplerFolds; i++)
    {
        // Example: TX antennas distributed across Doppler bins
        ddmDopplerOffsets[i] = ((float)i) / ((float)numDopplerFolds);
    }
    loadDdmDopplerOffsets(ddmDopplerOffsetsTensorHandle, ddmDopplerOffsets);
}

void setDetectionCountImpl(NVCVTensorHandle detectionCountTensorHandle, int32_t detectionCount)
{
    NVCVTensorData detectionCountTensorData;
    nvcvTensorExportData(detectionCountTensorHandle, &detectionCountTensorData);
    int32_t *detectionCountTensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&detectionCountTensorPtr, (void *)detectionCountTensorData.buffer.strided.basePtr);

    detectionCountTensorPtr[0] = detectionCount;
}

} // namespace

void generateSnapshotExtractionInput(NVCVTensorHandle detectionCountTensorHandle,
                                     NVCVTensorHandle foldedDetectionListTensorHandle,
                                     NVCVTensorHandle ddmDopplerOffsetsTensorHandle,
                                     NVCVTensorHandle rangeDopplerMapTensorHandle, NVCVTensorHandle nciRxTensorHandle,
                                     NVCVTensorHandle calibrationWeightsTensorHandle, int32_t detectionCount,
                                     int32_t numDopplerFolds, int32_t numRangeBins, int32_t numDopplerBins)
{
    NVCVTensorData detData, rdmData, nciData, ddmData, countData;
    nvcvTensorExportData(detectionCountTensorHandle, &countData);
    nvcvTensorExportData(foldedDetectionListTensorHandle, &detData);
    nvcvTensorExportData(rangeDopplerMapTensorHandle, &rdmData);
    nvcvTensorExportData(nciRxTensorHandle, &nciData);
    nvcvTensorExportData(ddmDopplerOffsetsTensorHandle, &ddmData);

    // Set detection count
    if (countData.dtype == NVCV_DATA_TYPE_S32)
        setDetectionCountImpl(detectionCountTensorHandle, detectionCount);

    if (detData.dtype == NVCV_DATA_TYPE_S32)
        generateDetectionListImpl<int32_t>(foldedDetectionListTensorHandle, numRangeBins,
                                           (numDopplerBins / numDopplerFolds));

    if (rdmData.dtype == NVCV_DATA_TYPE_2S32)
        generateComplexMapImpl<int32_t>(rangeDopplerMapTensorHandle);

    if (nciData.dtype == NVCV_DATA_TYPE_U32)
        generateNciImpl<uint32_t>(nciRxTensorHandle);

    // Generate DDM doppler offsets
    if (ddmData.dtype == NVCV_DATA_TYPE_F32)
        generateDdmDopplerOffsetsImpl<float>(ddmDopplerOffsetsTensorHandle, numDopplerFolds);

    // Generate calibration weights if tensor is provided
    if (calibrationWeightsTensorHandle != NULL)
    {
        NVCVTensorData calData;
        nvcvTensorExportData(calibrationWeightsTensorHandle, &calData);

        if (calData.dtype == NVCV_DATA_TYPE_2S32)
            generateCalibrationWeightsImpl<int32_t>(calibrationWeightsTensorHandle);
    }
}

void generateDdmDopplerOffsets(NVCVTensorHandle ddmDopplerOffsetsTensorHandle, int32_t numDopplerFolds)
{
    NVCVTensorData ddmData;
    nvcvTensorExportData(ddmDopplerOffsetsTensorHandle, &ddmData);
    if (ddmData.dtype == NVCV_DATA_TYPE_F32)
        generateDdmDopplerOffsetsImpl<float>(ddmDopplerOffsetsTensorHandle, numDopplerFolds);
}

void generateCalibrationWeights(NVCVTensorHandle calibrationWeightsTensorHandle)
{
    NVCVTensorData calData;
    nvcvTensorExportData(calibrationWeightsTensorHandle, &calData);
    if (calData.dtype == NVCV_DATA_TYPE_2S32)
        generateCalibrationWeightsImpl<int32_t>(calibrationWeightsTensorHandle);
}

void loadDdmDopplerOffsets(NVCVTensorHandle ddmDopplerOffsetsTensorHandle, const std::vector<float> &ddmDopplerOffsets)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(ddmDopplerOffsetsTensorHandle, &tensorData);
    float *tensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&tensorPtr, (void *)tensorData.buffer.strided.basePtr);

    // Load DDM Doppler offsets
    for (int32_t i = 0; i < ddmDopplerOffsets.size(); i++)
    {
        tensorPtr[i] = ddmDopplerOffsets[i];
    }
}

void loadCalibrationWeights(NVCVTensorHandle tensorHandle, int32_t numTx, int32_t numRx)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);
    int32_t *tensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&tensorPtr, (void *)tensorData.buffer.strided.basePtr);
    const int32_t numElements = numTx * numRx;
    for (int32_t i = 0; i < numElements; i++)
    {
        tensorPtr[i * 2]     = static_cast<int32_t>(CALIBRATION_WEIGHTS_SCALE);
        tensorPtr[i * 2 + 1] = 0;
    }
}

static inline int64_t my_round(int64_t x, int32_t shift)
{
    int64_t x_rounded = x;
    if (shift != 0)
    {
        int64_t half = 1 << (shift - 1);
        x_rounded    = (x + half) >> shift;
    }
    return x_rounded;
}

// C++ Helper function for snapshot extraction with fixed-point processing
SnapshotExtractionResult SnapshotExtractionFixedPoint(
    const std::vector<std::pair<int32_t, int32_t>> &foldedDetectionList, const float *ddmDopplerOffsetsPtr,
    int32_t numDopplerFolds, const uint32_t *nciRxPtr, int32_t numRangeBins, int32_t numDopplerBins,
    const int32_t *rangeDopplerMapPtr, int32_t numRx, int32_t numTx, PVAFFTLayout rangeDopplerFormat,
    const int32_t *calibrationWeightsPtr)
{
    SnapshotExtractionResult result;

    const int32_t numDetections       = static_cast<int32_t>(foldedDetectionList.size());
    const int32_t ddmRepeatFoldFactor = numDopplerFolds;

    // Copy DDM Doppler offsets to vector for easier access
    std::vector<float> ddmDopplerOffsets(numDopplerFolds);
    for (int32_t i = 0; i < numDopplerFolds; i++)
    {
        ddmDopplerOffsets[i] = ddmDopplerOffsetsPtr[i];
    }

    // Compute doppler offsets of TX antennas relative to TX1
    std::vector<int32_t> relativeDopplerOffsets(numTx);
    for (int32_t txIdx = 0; txIdx < numTx; txIdx++)
    {
        relativeDopplerOffsets[txIdx] =
            static_cast<int32_t>((ddmDopplerOffsets[0] - ddmDopplerOffsets[txIdx]) * numDopplerBins);
    }

    std::vector<int32_t> DDMspace(numTx, 0);
    for (int32_t txIdx = 0; txIdx < numTx; txIdx++)
    {
        DDMspace[txIdx] = ((relativeDopplerOffsets[txIdx] + numDopplerBins) % numDopplerBins);
    }

    // Determine weight type and Q-bits
    bool useCalibration  = (calibrationWeightsPtr != nullptr);
    int32_t weightsQbits = useCalibration ? CALIBRATION_WEIGHTS_QBITS : DDMA_WEIGHTS_QBITS;

    int32_t truncate_qbits = 16;                            // first truncate to 16 bits
    int32_t round_qbits    = weightsQbits - truncate_qbits; // then round to the remaining Q-bits

    // Compute DDMA weights if calibration is not provided
    // If calibration is provided, we'll access calibrationWeightsPtr directly
    std::vector<std::complex<int32_t>> ddmaWeights;
    if (!useCalibration)
    {
        // Compute DDMA weights in fixed-point
        constexpr float PI_CONST_F = 3.14159265358979323846f;
        ddmaWeights.resize(numTx);
        for (int32_t txIdx = 0; txIdx < numTx; txIdx++)
        {
            std::complex<float> imaginaryUnit(0.0f, 1.0f);
            float dopplerOffset             = ddmDopplerOffsets[txIdx];
            std::complex<float> weightFloat = std::conj(std::exp(-imaginaryUnit * 2.0f * PI_CONST_F * dopplerOffset));
            // Convert to fixed-point using DDMA_WEIGHTS_SCALE
            ddmaWeights[txIdx] =
                std::complex<int32_t>(static_cast<int32_t>(std::round(weightFloat.real() * DDMA_WEIGHTS_SCALE)),
                                      static_cast<int32_t>(std::round(weightFloat.imag() * DDMA_WEIGHTS_SCALE)));
        }
    }

    // Process detections in the order they arrive (NO SORTING)
    // This matches the PVA implementation behavior
    const int32_t numVirtualAntennas = numRx * numTx;
    const int32_t numChirpsPerFold   = numDopplerBins / ddmRepeatFoldFactor;

    // Initialize result
    result.tx1DetectionList.resize(numDetections);
    result.snapshots.resize(numDetections);
    for (int32_t i = 0; i < numDetections; i++)
    {
        result.snapshots[i].resize(numTx);
        for (int32_t tx = 0; tx < numTx; tx++)
        {
            result.snapshots[i][tx].resize(numRx);
        }
    }

    // Process each detection in the order it appears
    for (int32_t detectionIndex = 0; detectionIndex < numDetections; detectionIndex++)
    {
        int32_t detectionRangeIndex = foldedDetectionList[detectionIndex].first;
        int32_t foldedDopplerIndex  = foldedDetectionList[detectionIndex].second;

        int64_t maxSum = -1;
        int32_t tx1DopplerIndex;
        std::vector<int32_t> bestDetectionIndices(numTx);

        // Loop over RepeatFold number of candidates to find TX1 detections
        for (int32_t candidateIndex = 0; candidateIndex < ddmRepeatFoldFactor; candidateIndex++)
        {
            int32_t candidateDopplerDetectionIndex = foldedDopplerIndex + candidateIndex * numChirpsPerFold;
            // Compute doppler indices for all TX channels
            std::vector<int32_t> dopplerIndices(numTx);
            for (int32_t txIdx = 0; txIdx < numTx; txIdx++)
            {
                dopplerIndices[txIdx] = (candidateDopplerDetectionIndex + DDMspace[txIdx]) % numDopplerBins;
            }

            // Compute sum of NCI values for this candidate
            int64_t candidateSum = 0;
            for (int32_t txIdx = 0; txIdx < numTx; txIdx++)
            {
                int32_t nciIndex = detectionRangeIndex * numDopplerBins + dopplerIndices[txIdx];
                candidateSum += nciRxPtr[nciIndex];
            }

            // Check if this is the best candidate
            if (candidateSum > maxSum)
            {
                tx1DopplerIndex      = candidateDopplerDetectionIndex;
                bestDetectionIndices = dopplerIndices;
                maxSum               = candidateSum;
            }
        }

        // Write unfolded detection to result
        result.tx1DetectionList[detectionIndex].first  = detectionRangeIndex;
        result.tx1DetectionList[detectionIndex].second = tx1DopplerIndex;

        // Extract spatial data (virtual elements) and apply DDMA correction
        for (int32_t txIdx = 0; txIdx < numTx; txIdx++)
        {
            int32_t dopplerIdx = bestDetectionIndices[txIdx];
            for (int32_t rxIdx = 0; rxIdx < numRx; rxIdx++)
            {
                // Extract complex value from range-doppler map
                // Layout depends on rangeDopplerFormat:
                // - RANGE_RX_DOPPLER: [numRangeBins, numRx, numDopplerBins]
                // - RANGE_DOPPLER_RX: [numRangeBins, numDopplerBins, numRx]
                // Complex S32 (2 int32 per element)
                int32_t rdmIndex;
                if (rangeDopplerFormat == PVA_FFT_LAYOUT_RANGE_RX_DOPPLER)
                {
                    // RANGE_RX_DOPPLER format: [R, Rx, D]
                    rdmIndex = ((detectionRangeIndex * numRx + rxIdx) * numDopplerBins + dopplerIdx) * 2;
                }
                else
                {
                    // RANGE_DOPPLER_RX format: [R, D, Rx]
                    rdmIndex = ((detectionRangeIndex * numDopplerBins + dopplerIdx) * numRx + rxIdx) * 2;
                }
                std::complex<int32_t> spatialValue(rangeDopplerMapPtr[rdmIndex], rangeDopplerMapPtr[rdmIndex + 1]);

                // Get weight for this TX-RX pair (directly from calibration or DDMA weights)
                int32_t weightReal, weightImag;
                if (useCalibration)
                {
                    // Access calibration weight directly [numTx x numRx] - standard [real, imag] layout
                    int32_t calIndex = (txIdx * numRx + rxIdx) * 2;
                    weightReal       = calibrationWeightsPtr[calIndex];
                    weightImag       = calibrationWeightsPtr[calIndex + 1];
                }
                else
                {
                    // Access DDMA weight [numTx] - same weight for all RX
                    weightReal = ddmaWeights[txIdx].real();
                    weightImag = ddmaWeights[txIdx].imag();
                }

                // Apply weight using fixed-point complex multiplication
                // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
                int64_t real_in = static_cast<int64_t>(spatialValue.real());
                int64_t imag_in = static_cast<int64_t>(spatialValue.imag());

                int64_t ac = real_in * weightReal;
                int64_t bd = imag_in * weightImag;
                int64_t ad = real_in * weightImag;
                int64_t bc = imag_in * weightReal;

                ac = ac >> truncate_qbits;
                bd = bd >> truncate_qbits;
                ad = ad >> truncate_qbits;
                bc = bc >> truncate_qbits;

                int64_t real = ac - bd;
                int64_t imag = ad + bc;

                // Compensate for fixed-point scaling (use appropriate Q-bits based on weight type)
                real = my_round(real, round_qbits);
                imag = my_round(imag, round_qbits);

                // Store to result structure
                result.snapshots[detectionIndex][txIdx][rxIdx] =
                    std::complex<int32_t>(static_cast<int32_t>(real), static_cast<int32_t>(imag));
            }
        }
    }
    return result;
}

void snapshotExtractionRef(NVCVTensorHandle detectionCountTensorHandle,
                           NVCVTensorHandle foldedDetectionListTensorHandle,
                           NVCVTensorHandle ddmDopplerOffsetsTensorHandle, NVCVTensorHandle nciRxTensorHandle,
                           NVCVTensorHandle rangeDopplerMapTensorHandle,
                           NVCVTensorHandle calibrationWeightsTensorHandle, NVCVTensorHandle detectionListTensorHandle,
                           NVCVTensorHandle snapshotsTensorHandle, const PVASnapshotExtractionParams *params)
{
    // Extract detection count [1] - S32
    NVCVTensorData detectionCountData;
    nvcvTensorExportData(detectionCountTensorHandle, &detectionCountData);
    int32_t *detectionCountPtr = NULL;
    CupvaMemGetHostPointer((void **)&detectionCountPtr, (void *)detectionCountData.buffer.strided.basePtr);
    const int32_t numDetections = detectionCountPtr[0];

    // Extract folded detection list [numDetections, 2] - S32
    NVCVTensorData foldedDetData;
    nvcvTensorExportData(foldedDetectionListTensorHandle, &foldedDetData);
    int32_t *foldedDetPtr = NULL;
    CupvaMemGetHostPointer((void **)&foldedDetPtr, (void *)foldedDetData.buffer.strided.basePtr);

    std::vector<std::pair<int32_t, int32_t>> detectionList(numDetections);
    for (int32_t i = 0; i < numDetections; i++)
    {
        detectionList[i] = std::make_pair(foldedDetPtr[2 * i], foldedDetPtr[2 * i + 1]);
    }

    // Extract DDM Doppler offsets [numDopplerFolds] - F32
    NVCVTensorData ddmData;
    nvcvTensorExportData(ddmDopplerOffsetsTensorHandle, &ddmData);
    float *ddmPtr = NULL;
    CupvaMemGetHostPointer((void **)&ddmPtr, (void *)ddmData.buffer.strided.basePtr);

    // Extract NCI Rx [numRangeBins, numDopplerBins] - U32
    NVCVTensorData nciData;
    nvcvTensorExportData(nciRxTensorHandle, &nciData);
    uint32_t *nciPtr = NULL;
    CupvaMemGetHostPointer((void **)&nciPtr, (void *)nciData.buffer.strided.basePtr);

    int64_t nciShapeNCHW[4];
    nvcvTensorShapePermute(nciData.layout, nciData.shape, NVCV_TENSOR_NCHW, nciShapeNCHW);
    const int32_t numRangeBins   = nciShapeNCHW[2];
    const int32_t numDopplerBins = nciShapeNCHW[3];

    // Extract Range-Doppler map [numRangeBins, numRx, numDopplerBins] - Complex S32
    NVCVTensorData rdmData;
    nvcvTensorExportData(rangeDopplerMapTensorHandle, &rdmData);
    int32_t *rdmPtr = NULL;
    CupvaMemGetHostPointer((void **)&rdmPtr, (void *)rdmData.buffer.strided.basePtr);

    int64_t rdmShapeNCHW[4];
    nvcvTensorShapePermute(rdmData.layout, rdmData.shape, NVCV_TENSOR_NCHW, rdmShapeNCHW);

    // Extract numRx based on Range-Doppler format
    int32_t numRx;
    if (params->rangeDopplerFormat == PVA_FFT_LAYOUT_RANGE_RX_DOPPLER)
    {
        // RANGE_RX_DOPPLER format: [numRangeBins, numRx, numDopplerBins]
        numRx = rdmShapeNCHW[1]; // channels = numRx
    }
    else
    {
        // RANGE_DOPPLER_RX format: [numRangeBins, numDopplerBins, numRx]
        numRx = rdmShapeNCHW[3]; // width = numRx
    }

    // Extract calibration weights [numTx, numRx] - Complex S32 (optional - check if handle is not null)
    int32_t *calibrationWeightsPtr = nullptr;
    if (calibrationWeightsTensorHandle != nullptr)
    {
        NVCVTensorData calibrationData;
        nvcvTensorExportData(calibrationWeightsTensorHandle, &calibrationData);
        CupvaMemGetHostPointer((void **)&calibrationWeightsPtr, (void *)calibrationData.buffer.strided.basePtr);
    }

    // Prepare output: unfolded detection list
    NVCVTensorData outDetData;
    nvcvTensorExportData(detectionListTensorHandle, &outDetData);
    int32_t *outDetPtr = NULL;
    CupvaMemGetHostPointer((void **)&outDetPtr, (void *)outDetData.buffer.strided.basePtr);

    // Prepare output: snapshots [numDetections, numTxAntennas, numRxAntennas] - Complex S32
    NVCVTensorData snapshotsData;
    nvcvTensorExportData(snapshotsTensorHandle, &snapshotsData);

    int64_t snapshotsShapeCHW[3];
    nvcvTensorShapePermute(snapshotsData.layout, snapshotsData.shape, NVCV_TENSOR_CHW, snapshotsShapeCHW);
    const int32_t numTx = snapshotsShapeCHW[1];

    int32_t *snapshotsPtr = NULL;
    CupvaMemGetHostPointer((void **)&snapshotsPtr, (void *)snapshotsData.buffer.strided.basePtr);
    // Call the C++ helper function (calibration weights decision based on pointer being null or not)
    SnapshotExtractionResult cpuResult = SnapshotExtractionFixedPoint(
        detectionList, ddmPtr, params->numDopplerFolds, nciPtr, numRangeBins, numDopplerBins, rdmPtr, numRx, numTx,
        params->rangeDopplerFormat, calibrationWeightsPtr);

    // Write results to output tensors
    for (int32_t i = 0; i < numDetections; i++)
    {
        outDetPtr[2 * i]     = cpuResult.tx1DetectionList[i].first;
        outDetPtr[2 * i + 1] = cpuResult.tx1DetectionList[i].second;
    }

    const int32_t numVirtualAntennas = numRx * numTx;
    for (int32_t det = 0; det < numDetections; det++)
    {
        for (int32_t txIdx = 0; txIdx < numTx; txIdx++)
        {
            for (int32_t rxIdx = 0; rxIdx < numRx; rxIdx++)
            {
                int32_t snapshotIndex           = 2 * (det * numTx * numRx + txIdx * numRx + rxIdx);
                snapshotsPtr[snapshotIndex]     = cpuResult.snapshots[det][txIdx][rxIdx].real();
                snapshotsPtr[snapshotIndex + 1] = cpuResult.snapshots[det][txIdx][rxIdx].imag();
            }
        }
    }
}

int32_t checkSnapshotExtractionOutput(NVCVTensorHandle pvaDetectionCountTensorHandle,
                                      NVCVTensorHandle refDetectionCountTensorHandle,
                                      NVCVTensorHandle pvaDetectionListTensorHandle,
                                      NVCVTensorHandle refDetectionListTensorHandle,
                                      NVCVTensorHandle pvaSnapshotsTensorHandle,
                                      NVCVTensorHandle refSnapshotsTensorHandle)
{
    NVCVTensorData pvaDetectionCountData;
    nvcvTensorExportData(pvaDetectionCountTensorHandle, &pvaDetectionCountData);
    int32_t *pvaDetectionCountPtr = NULL;
    CupvaMemGetHostPointer((void **)&pvaDetectionCountPtr, (void *)pvaDetectionCountData.buffer.strided.basePtr);
    int32_t pvaDetectionCount = pvaDetectionCountPtr[0];

    NVCVTensorData refDetectionCountData;
    nvcvTensorExportData(refDetectionCountTensorHandle, &refDetectionCountData);
    int32_t *refDetectionCountPtr = NULL;
    CupvaMemGetHostPointer((void **)&refDetectionCountPtr, (void *)refDetectionCountData.buffer.strided.basePtr);
    int32_t refDetectionCount = refDetectionCountPtr[0];

    NVCVTensorData pvaDetectionListData;
    nvcvTensorExportData(pvaDetectionListTensorHandle, &pvaDetectionListData);
    int32_t *pvaDetectionListPtr = NULL;
    CupvaMemGetHostPointer((void **)&pvaDetectionListPtr, (void *)pvaDetectionListData.buffer.strided.basePtr);

    NVCVTensorData refDetectionListData;
    nvcvTensorExportData(refDetectionListTensorHandle, &refDetectionListData);
    int32_t *refDetectionListPtr = NULL;
    CupvaMemGetHostPointer((void **)&refDetectionListPtr, (void *)refDetectionListData.buffer.strided.basePtr);

    NVCVTensorData pvaSnapshotsData;
    nvcvTensorExportData(pvaSnapshotsTensorHandle, &pvaSnapshotsData);
    int32_t *pvaSnapshotsPtr = NULL;
    CupvaMemGetHostPointer((void **)&pvaSnapshotsPtr, (void *)pvaSnapshotsData.buffer.strided.basePtr);

    NVCVTensorData refSnapshotsData;
    nvcvTensorExportData(refSnapshotsTensorHandle, &refSnapshotsData);
    int32_t *refSnapshotsPtr = NULL;
    CupvaMemGetHostPointer((void **)&refSnapshotsPtr, (void *)refSnapshotsData.buffer.strided.basePtr);

    int64_t refSnapshotsShapeCHW[3];
    nvcvTensorShapePermute(refSnapshotsData.layout, refSnapshotsData.shape, NVCV_TENSOR_CHW, refSnapshotsShapeCHW);
    const int32_t numTx = refSnapshotsShapeCHW[1]; // H = numTx
    const int32_t numRx = refSnapshotsShapeCHW[2]; // W = numRx

    return checkSnapshotExtractionOutputImpl(pvaDetectionCount, refDetectionCount, numTx, numRx, pvaDetectionListPtr,
                                             refDetectionListPtr, pvaSnapshotsPtr, refSnapshotsPtr);
}
