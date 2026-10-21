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
 * @file target_processing.cpp
 *
 * @brief Reference implementation for Target Processing
 *
 * This file provides a reference implementation for the Target Processing component.
 * This reference implementation is used for validation
 *
 * The component performs Range-Velocity (RV) decoupling and Cartesian coordinate
 * computation, converting radar measurements (range, Doppler, angles) into physical
 * coordinates and velocity.
 *
 * @see PvaOperatorTypes.h for parameter structures
 */

#include "target_processing_ref.h"

#include "doa_angle_fft_ref.h"

#include <TensorUtils.hpp>

#include <cupva_host.h>
#include <nvcv/Tensor.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>

// Constants
#ifndef PI_CONST_F
#    define PI_CONST_F 3.14159265358979323846f
#endif

// ============================================================================
// RV Decoupling Helper Functions
// ============================================================================

/**
 * @brief 5-hypothesis velocity disambiguation
 *
 * @param tempVIn Base velocity value (f_fast * contVFast - f_slow * contVSlow)
 * @param tempVCoeff Array of 5 hypothesis coefficients
 * @param velocityAmbHigh High velocity bound (negative value in m/s)
 * @param velocityAmbLow Low velocity bound (positive value in m/s)
 * @return Disambiguated velocity in m/s
 */
static float calcVelocityRvDecouplingScalar(float tempVIn, const float *tempVCoeff, float velocityAmbHigh,
                                            float velocityAmbLow)
{
    for (int i = 0; i < 5; i++)
    {
        float tempV = tempVIn - tempVCoeff[i];
        if (tempV >= velocityAmbHigh && tempV <= velocityAmbLow)
        {
            return tempV;
        }
    }
    return 0.0f;
}

// ============================================================================
// Quadratic Interpolation Helper
// ============================================================================

/**
 * @brief Quadratic (parabolic) interpolation to find sub-bin peak offset
 *
 * Given 3 samples (y0, y1, y2) centered at y1 (the peak), compute the
 * fractional offset 'p' from the center to the true peak location.
 * Uses parabolic fit: p = (y0 - y2) / (2 * (y0 - 2*y1 + y2))
 *
 * @param y0 Value at index -1 (before peak)
 * @param y1 Value at index 0 (peak)
 * @param y2 Value at index +1 (after peak)
 * @return Fractional offset from center, in range [-1, 1]
 */
static float quadraticInterpolationOffset(float y0, float y1, float y2)
{
    float denominator = 2.0f * (y0 - 2.0f * y1 + y2);
    if (std::abs(denominator) < 1e-10f)
    {
        return 0.0f; // Avoid division by zero
    }
    float p = (y0 - y2) / denominator;
    // Clamp to valid range [-1, 1] - interpolation outside this range means center is not a valid peak
    return std::clamp(p, -1.0f, 1.0f);
}

// Target Processing (Fixed-Point) - Step 7: Compute range, range-rate, and cartesian coordinates
// Note: This function performs the same floating-point calculations as the reference implementation
// since target processing involves physical unit conversions that are inherently floating-point
// Supports both standard mode and RV decoupling mode based on params->enableRvDecoupling
// When enableQuadraticInterpolation is true, uses NCI data for sub-bin accuracy
TargetProcessingResult TargetProcessingFixedPoint(const std::vector<std::pair<int32_t, int32_t>> &tx1DetectionList,
                                                  const DoaResult &doaResult, float ddmDopplerOffset,
                                                  const PVATargetProcessingParams *params, const uint32_t *nciData,
                                                  int32_t nciWidth, int32_t nciHeight)
{
    TargetProcessingResult result;

    if (tx1DetectionList.empty())
    {
        return result;
    }

    // Extract parameters
    const float rangeResolution   = params->rangeResolution;
    const float sweepTime         = params->sweepTime;
    const float wavelength        = params->wavelength;
    const int32_t numChirps       = params->numChirps;
    const bool fftShiftedDoppler  = params->fftShiftedDoppler;
    const bool enableRvDecoupling = params->enableRvDecoupling;

    // Compute derived RV decoupling parameters from base FMCW parameters
    const float f_fast_delta    = params->fs / static_cast<float>(params->nofSamples);
    const float f_slow_delta    = 1.0f / (params->PRI * static_cast<float>(numChirps));
    const float contR           = params->contR;
    const float contVFast       = params->contVFast;
    const float contVSlow       = params->contVSlow;
    const float velocityAmbHigh = -1.0f * params->deltaV * static_cast<float>(numChirps) / 2.0f / 3.6f;
    const float velocityAmbLow  = params->deltaV * static_cast<float>(numChirps) / 2.0f / 3.6f;

    // Compute 5 hypothesis coefficients for velocity disambiguation
    float tempVCoeff[5];
    for (int i = 0; i < 5; i++)
    {
        tempVCoeff[i] = (1.0f / params->PRI) * (static_cast<float>(i) - 3.0f) * contVSlow;
    }

    int32_t numTargets = static_cast<int32_t>(tx1DetectionList.size());

    // Initialize result vectors
    result.velocityValues.resize(numTargets);
    result.rangeValues.resize(numTargets);
    result.azimuthDegrees.resize(numTargets);
    result.elevationDegrees.resize(numTargets);
    result.xCoordinates.resize(numTargets);
    result.yCoordinates.resize(numTargets);
    result.zCoordinates.resize(numTargets);
    result.peakPowerDb.resize(numTargets);

    // Compute derived parameters for standard mode
    float pulseRepetitionFrequency = 1.0f / sweepTime;

    // Check if quadratic interpolation is enabled and NCI data is available
    const bool enableQuadraticInterpolation =
        params->enableQuadraticInterpolation && nciData != nullptr && nciWidth > 0 && nciHeight > 0;

    // Process each target
    for (int32_t targetIndex = 0; targetIndex < numTargets; targetIndex++)
    {
        int32_t rangeBinIndex   = tx1DetectionList[targetIndex].first;
        int32_t dopplerBinIndex = tx1DetectionList[targetIndex].second;

        // Start with integer bin indices
        float interpolatedRangeIdx   = static_cast<float>(rangeBinIndex);
        float interpolatedDopplerIdx = static_cast<float>(dopplerBinIndex);

        // Apply quadratic interpolation if enabled
        if (enableQuadraticInterpolation)
        {
            // NCI data is unpadded [nciHeight][nciWidth]
            // Use circular/wrapped indexing to handle boundary cases (matches PVA's cyclic padding)

            // Helper lambda for circular indexing
            auto circularIndex = [](int32_t idx, int32_t size) -> int32_t {
                return ((idx % size) + size) % size;
            };

            // Detection list uses unfolded doppler indices, but NCI uses folded doppler
            // Fold the doppler index to match NCI dimensions
            int32_t foldedDopplerIdx = dopplerBinIndex % nciWidth;

            // Get 3x3 neighborhood centered at (rangeBinIndex, foldedDopplerIdx)
            // Using circular indexing for neighbors
            int32_t rCenter = rangeBinIndex;
            int32_t dCenter = foldedDopplerIdx;

            int32_t rMinus1 = circularIndex(rangeBinIndex - 1, nciHeight);
            int32_t rPlus1  = circularIndex(rangeBinIndex + 1, nciHeight);
            int32_t dMinus1 = circularIndex(foldedDopplerIdx - 1, nciWidth);
            int32_t dPlus1  = circularIndex(foldedDopplerIdx + 1, nciWidth);

            // Load 3x3 neighborhood
            // v11 v12 v13   (row rMinus1)
            // v21 v22 v23   (row rCenter)   <- v22 is the peak
            // v31 v32 v33   (row rPlus1)
            float v12 = static_cast<float>(nciData[rMinus1 * nciWidth + dCenter]);
            float v21 = static_cast<float>(nciData[rCenter * nciWidth + dMinus1]);
            float v22 = static_cast<float>(nciData[rCenter * nciWidth + dCenter]);
            float v23 = static_cast<float>(nciData[rCenter * nciWidth + dPlus1]);
            float v32 = static_cast<float>(nciData[rPlus1 * nciWidth + dCenter]);

            // Quadratic interpolation for range (vertical neighbors: v12, v22, v32)
            float pr = quadraticInterpolationOffset(v12, v22, v32);

            // Quadratic interpolation for doppler (horizontal neighbors: v21, v22, v23)
            float pd = quadraticInterpolationOffset(v21, v22, v23);

            // Apply sub-bin offsets
            interpolatedRangeIdx   = static_cast<float>(rangeBinIndex) + pr;
            interpolatedDopplerIdx = static_cast<float>(dopplerBinIndex) + pd;
        }

        float range, velocity;

        if (enableRvDecoupling)
        {
            // ================================================================
            // Original RV Decoupling Mode (Floating-Point)
            // Uses FMCW coupling formulas for accurate range/velocity
            // When quadratic interpolation is enabled, uses sub-bin indices
            // ================================================================

            // Compute f_fast and f_slow (using interpolated indices, +1 for 1-based indexing)
            float f_fast = (interpolatedRangeIdx + 1.0f) * f_fast_delta;
            float f_slow = (interpolatedDopplerIdx + 1.0f) * f_slow_delta;

            // Range with FMCW coupling correction
            range = (f_fast - f_slow) * contR;

            // Velocity with 5-hypothesis disambiguation
            float tempVBase = f_fast * contVFast - f_slow * contVSlow;
            float tempV     = calcVelocityRvDecouplingScalar(tempVBase, tempVCoeff, velocityAmbHigh, velocityAmbLow);

            // Convert from m/s to km/h
            velocity = tempV * 3.6f;
        }
        else
        {
            // ================================================================
            // Standard Mode (original implementation)
            // When quadratic interpolation is enabled, uses sub-bin indices
            // ================================================================

            // 1. Compute range values from (interpolated) range bin indices
            range = interpolatedRangeIdx * rangeResolution;

            // 2. Compute range-rate values from doppler bin indices (TX1 disambiguated doppler)
            int32_t adjustedDopplerIdx = dopplerBinIndex;

            if (!fftShiftedDoppler)
            {
                adjustedDopplerIdx = (dopplerBinIndex + numChirps / 2) % numChirps;
            }

            // Create doppler frequency map: (0:numChirps-1)/numChirps - 0.5
            float normalizedDopplerFreq =
                (static_cast<float>(adjustedDopplerIdx) / static_cast<float>(numChirps)) - 0.5f;

            // Apply Doppler correction for DDMA Doppler shift applied on TX1
            float dopplerValue = fmod(normalizedDopplerFreq + ddmDopplerOffset + 0.5f, 1.0f) - 0.5f;

            // Convert to actual frequency
            float dopplerFrequency = dopplerValue * pulseRepetitionFrequency;

            // Convert doppler frequency to speed (equivalent to MATLAB's dop2speed function)
            // Formula: speed = doppler_frequency * wavelength / 2 (factor of 2 for round-trip)
            velocity = (dopplerFrequency * wavelength) / 2.0f;
        }

        result.rangeValues[targetIndex]    = range;
        result.velocityValues[targetIndex] = velocity;

        // 3. Extract azimuth and elevation angles from DOA results
        result.azimuthDegrees[targetIndex]   = doaResult.azimuthDegrees[targetIndex];
        result.elevationDegrees[targetIndex] = doaResult.elevationDegrees[targetIndex];
        result.peakPowerDb[targetIndex]      = doaResult.peakPowerDb[targetIndex];

        // 4. Convert spherical coordinates (azimuth, elevation, range) to Cartesian coordinates (x, y, z)
        float azimuthRad   = result.azimuthDegrees[targetIndex] * PI_CONST_F / 180.0f;
        float elevationRad = result.elevationDegrees[targetIndex] * PI_CONST_F / 180.0f;

        // Height is always: Z = range * sin(elevation)
        result.zCoordinates[targetIndex] = range * std::sin(elevationRad);

        // Compute ground range based on projection mode
        float groundRange;
        if (params->groundProjection == 1) // GROUND_PLANE
        {
            groundRange = range * std::cos(elevationRad);
        }
        else // SLANT_RANGE
        {
            groundRange = range;
        }

        // Compute X and Y based on longitudinal axis convention
        if (params->longitudinalAxis == 1) // Y_FORWARD
        {
            result.xCoordinates[targetIndex] = groundRange * std::sin(azimuthRad); // Lateral (right positive)
            result.yCoordinates[targetIndex] = groundRange * std::cos(azimuthRad); // Longitudinal (forward)
        }
        else // X_FORWARD
        {
            result.xCoordinates[targetIndex] = groundRange * std::cos(azimuthRad); // Forward
            result.yCoordinates[targetIndex] = groundRange * std::sin(azimuthRad); // Lateral
        }
    }

    return result;
}

// ============================================================================
// Public API Implementation using NVCVTensor
// ============================================================================

namespace {

template<typename T>
void generateDetectionListImpl(NVCVTensorHandle tensorHandle)
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
    std::uniform_int_distribution<int32_t> rangeDist(0, 255);
    std::uniform_int_distribution<int32_t> dopplerDist(0, 127);

    for (int32_t i = 0; i < numDetections; i++)
    {
        tensorPtr[2 * i]     = rangeDist(gen);   // Range index
        tensorPtr[2 * i + 1] = dopplerDist(gen); // Doppler index
    }
}

template<typename T>
void generateAnglesImpl(NVCVTensorHandle tensorHandle)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);
    T *tensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&tensorPtr, (void *)tensorData.buffer.strided.basePtr);

    int64_t tensorShapeNCHW[4];
    nvcvTensorShapePermute(tensorData.layout, tensorData.shape, NVCV_TENSOR_NCHW, tensorShapeNCHW);

    const int32_t width = tensorShapeNCHW[3]; // numTargets

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> azDist(-45.0f, 45.0f);
    std::uniform_real_distribution<float> elDist(-30.0f, 30.0f);
    std::uniform_real_distribution<float> powerDist(10.0f, 40.0f);

    for (int32_t i = 0; i < width; i++)
    {
        tensorPtr[0 * width + i] = azDist(gen);    // Azimuth
        tensorPtr[1 * width + i] = elDist(gen);    // Elevation
        tensorPtr[2 * width + i] = powerDist(gen); // Power (dB)
    }
}

int32_t checkTargetListOutputImpl(int32_t numTargets, int32_t numProperties, float *pvaTargetListPtr, int32_t pvaStride,
                                  float *refTargetListPtr, int32_t refStride)
{
    float maxDiffFloat = 0.6f; // Tolerance for physical unit comparison
    bool debug         = false;

    if (debug == true)
    {
        // Dump both ref and PVA values for all properties, for each target, in 2 rows
        for (int32_t i = 0; i < numTargets; i++)
        {
            printf("Target %d:\n", i);
            printf("  REF: ");
            for (int32_t j = 0; j < numProperties; j++)
            {
                printf("%.6f ", refTargetListPtr[j * refStride + i]);
            }
            printf("\n  PVA: ");
            for (int32_t j = 0; j < numProperties; j++)
            {
                printf("%.6f ", pvaTargetListPtr[j * pvaStride + i]);
            }
            printf("\n");
        }
    }

    // Compute error statistics
    float maxError  = 0.0f;
    float sumError  = 0.0f;
    int maxErrorIdx = -1, maxErrorProp = -1;
    int numMismatches = 0;

    for (int32_t i = 0; i < numTargets; i++)
    {
        for (int32_t j = 0; j < numProperties; j++)
        {
            float error = std::abs(pvaTargetListPtr[j * pvaStride + i] - refTargetListPtr[j * refStride + i]);
            sumError += error;
            if (error > maxError)
            {
                maxError     = error;
                maxErrorIdx  = i;
                maxErrorProp = j;
            }
            if (error > maxDiffFloat)
            {
                numMismatches++;
            }
        }
    }

    if (maxError > maxDiffFloat)
    {
        printf("Target list mismatch at index %d, property %d: ref = %.6f, pva = %.6f\n", maxErrorIdx, maxErrorProp,
               refTargetListPtr[maxErrorProp * refStride + maxErrorIdx],
               pvaTargetListPtr[maxErrorProp * pvaStride + maxErrorIdx]);
        return -1;
    }
    return 0;
}

int32_t checkTargetListOutputImpl(int32_t numTargets, int32_t numProperties, float *pvaTargetListPtr,
                                  float *refTargetListPtr)
{
    // Use numTargets as stride for both (legacy function - assumes same stride)
    return checkTargetListOutputImpl(numTargets, numProperties, pvaTargetListPtr, numTargets, refTargetListPtr,
                                     numTargets);
}

} // namespace

void generateTargetProcessingInput(NVCVTensorHandle detectionListTensorHandle, NVCVTensorHandle anglesTensorHandle)
{
    NVCVTensorData detData, anglesData;
    nvcvTensorExportData(detectionListTensorHandle, &detData);
    nvcvTensorExportData(anglesTensorHandle, &anglesData);

    if (detData.dtype == NVCV_DATA_TYPE_S32)
        generateDetectionListImpl<int32_t>(detectionListTensorHandle);

    if (anglesData.dtype == NVCV_DATA_TYPE_F32)
        generateAnglesImpl<float>(anglesTensorHandle);
}

void generateDetectionIndexMap(NVCVTensorHandle detectionIndexMapTensorHandle, int32_t numDetections,
                               int32_t numTargets)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(detectionIndexMapTensorHandle, &tensorData);
    int32_t *tensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&tensorPtr, (void *)tensorData.buffer.strided.basePtr);

    // Simple 1:1 mapping (each target corresponds to one detection)
    for (int32_t i = 0; i < numTargets; i++)
    {
        tensorPtr[i] = (i < numDetections) ? i : numDetections - 1;
    }
}

void targetProcessingRef(NVCVTensorHandle targetCountTensorHandle, NVCVTensorHandle detectionListTensorHandle,
                         NVCVTensorHandle targetIndexMapTensorHandle, NVCVTensorHandle anglesTensorHandle,
                         NVCVTensorHandle ddmDopplerOffsetsTensorHandle, NVCVTensorHandle nciTensorHandle,
                         NVCVTensorHandle targetListTensorHandle, const PVATargetProcessingParams *params)
{
    // Extract target count
    NVCVTensorData targetCountData;
    nvcvTensorExportData(targetCountTensorHandle, &targetCountData);
    int32_t *targetCountPtr = NULL;
    CupvaMemGetHostPointer((void **)&targetCountPtr, (void *)targetCountData.buffer.strided.basePtr);
    const int32_t numTargets = targetCountPtr[0];

    // Extract target index map
    NVCVTensorData targetIndexMapData;
    nvcvTensorExportData(targetIndexMapTensorHandle, &targetIndexMapData);
    int32_t *targetIndexMapPtr = NULL;
    CupvaMemGetHostPointer((void **)&targetIndexMapPtr, (void *)targetIndexMapData.buffer.strided.basePtr);

    // Extract detection list
    NVCVTensorData detData;
    nvcvTensorExportData(detectionListTensorHandle, &detData);
    int32_t *detPtr = NULL;
    CupvaMemGetHostPointer((void **)&detPtr, (void *)detData.buffer.strided.basePtr);

    std::vector<std::pair<int32_t, int32_t>> rangeDopplerDetectionList(numTargets);
    for (int32_t i = 0; i < numTargets; i++)
    {
        int32_t detectionIndex       = targetIndexMapPtr[i];
        rangeDopplerDetectionList[i] = std::make_pair(detPtr[2 * detectionIndex], detPtr[2 * detectionIndex + 1]);
    }

    // Extract angles
    NVCVTensorData anglesData;
    nvcvTensorExportData(anglesTensorHandle, &anglesData);
    float *anglesPtr = NULL;
    CupvaMemGetHostPointer((void **)&anglesPtr, (void *)anglesData.buffer.strided.basePtr);

    int64_t anglesShapeCHW[3];
    nvcvTensorShapePermute(anglesData.layout, anglesData.shape, NVCV_TENSOR_CHW, anglesShapeCHW);
    const int32_t numAngles = anglesShapeCHW[2];

    DoaResult doaResult;
    doaResult.azimuthDegrees.resize(numTargets);
    doaResult.elevationDegrees.resize(numTargets);
    doaResult.peakPowerDb.resize(numTargets);
    for (int32_t i = 0; i < numTargets; i++)
    {
        doaResult.azimuthDegrees[i]   = anglesPtr[0 * numAngles + i];
        doaResult.elevationDegrees[i] = anglesPtr[1 * numAngles + i];
        doaResult.peakPowerDb[i]      = anglesPtr[2 * numAngles + i];
    }

    // Extract DDM Doppler offsets
    NVCVTensorData ddmData;
    nvcvTensorExportData(ddmDopplerOffsetsTensorHandle, &ddmData);
    float *ddmPtr = NULL;
    CupvaMemGetHostPointer((void **)&ddmPtr, (void *)ddmData.buffer.strided.basePtr);

    // Get the first Doppler offset (TX1)
    float ddmDopplerOffset = ddmPtr[0];

    // Extract NCI data for quadratic interpolation (if provided)
    const uint32_t *nciPtr = nullptr;
    int32_t nciWidth       = 0;
    int32_t nciHeight      = 0;

    if (nciTensorHandle != nullptr && params->enableQuadraticInterpolation)
    {
        NVCVTensorData nciData;
        nvcvTensorExportData(nciTensorHandle, &nciData);
        CupvaMemGetHostPointer((void **)&nciPtr, (void *)nciData.buffer.strided.basePtr);

        int64_t nciShapeHW[2];
        nvcvTensorShapePermute(nciData.layout, nciData.shape, NVCV_TENSOR_HW, nciShapeHW);
        nciHeight = static_cast<int32_t>(nciShapeHW[0]);
        nciWidth  = static_cast<int32_t>(nciShapeHW[1]);
    }

    // Run target processing (supports both standard and RV decoupling modes)
    TargetProcessingResult result = TargetProcessingFixedPoint(rangeDopplerDetectionList, doaResult, ddmDopplerOffset,
                                                               params, nciPtr, nciWidth, nciHeight);

    // Write output
    NVCVTensorData targetListData;
    nvcvTensorExportData(targetListTensorHandle, &targetListData);
    float *targetListPtr = NULL;
    CupvaMemGetHostPointer((void **)&targetListPtr, (void *)targetListData.buffer.strided.basePtr);

    // Use tensor stride (W dimension) rather than numTargets for correct memory layout
    int64_t targetListShapeNCHW[4];
    nvcvTensorShapePermute(targetListData.layout, targetListData.shape, NVCV_TENSOR_NCHW, targetListShapeNCHW);
    const int32_t stride           = targetListShapeNCHW[3]; // W dimension = tensor allocated width
    const int32_t targetListHeight = targetListShapeNCHW[2]; // H dimension = number of properties (7 or 8)

    for (int32_t i = 0; i < numTargets; i++)
    {
        targetListPtr[0 * stride + i] = result.velocityValues[i];   // Velocity
        targetListPtr[1 * stride + i] = result.rangeValues[i];      // Range
        targetListPtr[2 * stride + i] = result.azimuthDegrees[i];   // Azimuth
        targetListPtr[3 * stride + i] = result.elevationDegrees[i]; // Elevation
        targetListPtr[4 * stride + i] = result.xCoordinates[i];     // X
        targetListPtr[5 * stride + i] = result.yCoordinates[i];     // Y
        targetListPtr[6 * stride + i] = result.zCoordinates[i];     // Z
        if (targetListHeight == 8)
        {
            targetListPtr[7 * stride + i] = result.peakPowerDb[i]; // Power (optional)
        }
    }
}

int32_t checkTargetProcessingOutput(NVCVTensorHandle targetCountTensorHandle,
                                    NVCVTensorHandle pvaTargetListTensorHandle,
                                    NVCVTensorHandle refTargetListTensorHandle)
{
    NVCVTensorData targetCountData;
    nvcvTensorExportData(targetCountTensorHandle, &targetCountData);
    int32_t *targetCountPtr = NULL;
    CupvaMemGetHostPointer((void **)&targetCountPtr, (void *)targetCountData.buffer.strided.basePtr);
    const int32_t numTargetsToCompare = targetCountPtr[0];

    NVCVTensorData pvaTargetListData;
    nvcvTensorExportData(pvaTargetListTensorHandle, &pvaTargetListData);
    float *pvaTargetListPtr = NULL;
    CupvaMemGetHostPointer((void **)&pvaTargetListPtr, (void *)pvaTargetListData.buffer.strided.basePtr);

    NVCVTensorData refTargetListData;
    nvcvTensorExportData(refTargetListTensorHandle, &refTargetListData);
    float *refTargetListPtr = NULL;
    CupvaMemGetHostPointer((void **)&refTargetListPtr, (void *)refTargetListData.buffer.strided.basePtr);

    // Get tensor shape - use width (W) as stride for correct memory access
    int64_t pvaTargetListShapeNCHW[4];
    nvcvTensorShapePermute(pvaTargetListData.layout, pvaTargetListData.shape, NVCV_TENSOR_NCHW, pvaTargetListShapeNCHW);
    int64_t refTargetListShapeNCHW[4];
    nvcvTensorShapePermute(refTargetListData.layout, refTargetListData.shape, NVCV_TENSOR_NCHW, refTargetListShapeNCHW);

    const int32_t pvaStride     = pvaTargetListShapeNCHW[3]; // W dimension = tensor allocated width
    const int32_t refStride     = refTargetListShapeNCHW[3];
    const int32_t numProperties = refTargetListShapeNCHW[2]; // H dimension = 8 properties

    return checkTargetListOutputImpl(numTargetsToCompare, numProperties, pvaTargetListPtr, pvaStride, refTargetListPtr,
                                     refStride);
}
