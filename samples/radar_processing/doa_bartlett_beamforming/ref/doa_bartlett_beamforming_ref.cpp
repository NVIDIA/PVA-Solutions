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
 * @file doa_bartlett_beamforming_ref.cpp
 *
 * @brief Reference implementation for DOA Bartlett Beamforming
 *
 * Data Formats:
 * - Steering Vectors: Q15 (16-bit signed)
 * - Snapshots: SQ11.20 (32-bit signed)
 * - Beamformer Output: SQ18.23 (48-bit signed, after 12-bit shift, 64-ch accum)
 * - Power Spectrum: float
 * - Interpolation: float (quadratic peak interpolation on power)
 */

#include "doa_bartlett_beamforming_ref.h"

#include "../../custom/target_processing/ref/target_processing_ref.h"

#include <TensorUtils.hpp>

#include <OpDoaBartlettBeamforming.h>
#include <cupva_host.h>
#include <nvcv/Tensor.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

// Store steering vectors already conjugated in the tensor (imag negated).
// This matches the optimized device kernel which performs a direct complex multiply.
#define STEERING_VECTORS_STORED_CONJ 1

static inline int64_t my_round(int64_t x, int32_t shift)
{
    if (shift <= 0)
        return x;
    int64_t half = 1LL << (shift - 1);
    return (x + half) >> shift;
}

/**
 * @brief Core beamforming for a single angle bin
 *
 * Computes: beamOutput = Σ(steeringVector* × snapshot) across all virtual channels
 *
 * Q-Format Flow:
 *   Q15 (steering) × SQ11.20 (snapshot) = SQ12.35
 *   After shift by 12 → SQ12.23
 *   After accumulating 64 VCs: SQ18.23 (fits in 48 bits)
 *
 * @param snapshotPtr Pointer to snapshot data [TX, RX] with complex interleaved
 * @param numTx Number of TX antennas
 * @param numRx Number of RX antennas
 * @param steeringVectorsPtr Full steering vectors [numAz][numEl][numTx][numRx]
 * @param numElevationBins Total number of elevation bins (for indexing)
 * @param azIdx Azimuth index to compute
 * @param elIdx Elevation index to compute
 * @return Complex beamformer output (real, imag) as int64_t pair
 */
static std::pair<int64_t, int64_t> ComputeBeamformingForAngle(const int32_t *snapshotPtr, int32_t numTx, int32_t numRx,
                                                              const int16_t *steeringVectorsPtr,
                                                              int32_t numElevationBins, int32_t azIdx, int32_t elIdx)
{
    int64_t accumulatorReal = 0;
    int64_t accumulatorImag = 0;

    // Compute flat angle index for steering vector lookup
    int32_t angIdx = azIdx * numElevationBins + elIdx;

    for (int32_t txIdx = 0; txIdx < numTx; txIdx++)
    {
        for (int32_t rxIdx = 0; rxIdx < numRx; rxIdx++)
        {
            // Steering vector: [angIdx][txIdx][rxIdx] layout
            int32_t steerIdx  = (angIdx * numTx + txIdx) * numRx + rxIdx;
            int32_t steerReal = steeringVectorsPtr[2 * steerIdx];
            int32_t steerImag = steeringVectorsPtr[2 * steerIdx + 1];

            // Snapshot: [txIdx][rxIdx] layout (flat vcIdx = txIdx * numRx + rxIdx)
            int32_t vcIdx    = txIdx * numRx + rxIdx;
            int32_t snapReal = snapshotPtr[2 * vcIdx];
            int32_t snapImag = snapshotPtr[2 * vcIdx + 1];

#if STEERING_VECTORS_STORED_CONJ
            // steeringVectorsPtr already contains conj(steer), so do a direct complex multiply:
            // (steerReal + j*steerImag) * (snapReal + j*snapImag)
            int64_t realProduct = (int64_t)steerReal * (int64_t)snapReal - (int64_t)steerImag * (int64_t)snapImag;
            int64_t imagProduct = (int64_t)steerReal * (int64_t)snapImag + (int64_t)steerImag * (int64_t)snapReal;
#else
            // steeringVectorsPtr contains steer; apply Hermitian dot-product: conj(steer) * snapshot.
            int64_t realProduct = (int64_t)steerReal * (int64_t)snapReal + (int64_t)steerImag * (int64_t)snapImag;
            int64_t imagProduct = (int64_t)steerReal * (int64_t)snapImag - (int64_t)steerImag * (int64_t)snapReal;
#endif

            accumulatorReal += my_round(realProduct, BEAMFORMING_SHIFT);
            accumulatorImag += my_round(imagProduct, BEAMFORMING_SHIFT);
        }
    }

    return {accumulatorReal, accumulatorImag};
}

/**
 * @brief Beamforming computation for all angle bins (full 2D search)
 *
 * This is the standard Bartlett beamforming that searches over all Az × El combinations.
 * Complexity: O(numAz × numEl × numVC)
 *
 * @param snapshotPtr Pointer to snapshot data in memory layout [TX, RX] with complex interleaved
 * @param numTx Number of TX antennas
 * @param numRx Number of RX antennas
 * @param steeringVectorsPtr Pointer to steering vectors in memory layout [numAz][numEl][TX, RX]
 * @param numAzimuthBins Number of azimuth angle bins
 * @param numElevationBins Number of elevation angle bins
 */
// Forward declaration for ComputeBeamformingSubset
static BeamformerOutput ComputeBeamformingSubset(const int32_t *snapshotPtr, int32_t numTx, int32_t numRx,
                                                 const int16_t *steeringVectorsPtr, int32_t numElevationBins,
                                                 int32_t azStartIdx, int32_t azEndIdx, int32_t elStartIdx,
                                                 int32_t elEndIdx);

BeamformerOutput ComputeBeamforming(const int32_t *snapshotPtr, int32_t numTx, int32_t numRx,
                                    const int16_t *steeringVectorsPtr, int32_t numAzimuthBins, int32_t numElevationBins)
{
    // Delegate to ComputeBeamformingSubset with full range
    return ComputeBeamformingSubset(snapshotPtr, numTx, numRx, steeringVectorsPtr, numElevationBins, 0,
                                    numAzimuthBins - 1, 0, numElevationBins - 1);
}

/**
 * @brief Power spectrum calculation
 *
 * Computes |beamOutput|^2 = real^2 + imag^2 for each angle bin.
 * Interpolation is performed on power (not magnitude).
 */
BartlettPowerSpectrum ComputeBartlettPowerSpectrum(const BeamformerOutput &beamformerOutput, const float *azimuthBins,
                                                   int32_t numAzimuthBins, const float *elevationBins,
                                                   int32_t numElevationBins)
{
    BartlettPowerSpectrum powerSpec;
    powerSpec.numAzimuthBins   = numAzimuthBins;
    powerSpec.numElevationBins = numElevationBins;
    powerSpec.data.resize(beamformerOutput.numAngleBins);

    powerSpec.azimuthBins.assign(azimuthBins, azimuthBins + numAzimuthBins);
    powerSpec.elevationBins.assign(elevationBins, elevationBins + numElevationBins);

    powerSpec.maxPowerFloat  = 0.0f;
    powerSpec.peakAngleIndex = 0;

    for (int32_t angIdx = 0; angIdx < beamformerOutput.numAngleBins; angIdx++)
    {
        int64_t realPart = beamformerOutput.data[angIdx].first;
        int64_t imagPart = beamformerOutput.data[angIdx].second;

        // Convert to float
        float realFloat = static_cast<float>(realPart);
        float imagFloat = static_cast<float>(imagPart);

        // Store power (not magnitude) for interpolation
        float powerFloat = realFloat * realFloat + imagFloat * imagFloat;

        powerSpec.data[angIdx] = powerFloat;

        if (powerFloat > powerSpec.maxPowerFloat)
        {
            powerSpec.maxPowerFloat  = powerFloat;
            powerSpec.peakAngleIndex = angIdx;
        }
    }

    return powerSpec;
}

/**
 * @brief Quadratic peak interpolation for sub-bin accuracy
 *
 * Formula: x_peak = x2 + (y1 - y3) / (2 * (y1 - 2*y2 + y3))
 *
 * @param y1      Power at bin (peakIdx - 1)
 * @param y2      Power at bin peakIdx (center/peak)
 * @param y3      Power at bin (peakIdx + 1)
 * @param peakIdx Integer peak bin index
 * @return        Interpolated fractional index
 */
static float QuadraticPeakInterpFloat(float y1, float y2, float y3, int32_t peakIdx)
{
    float interpIdx   = static_cast<float>(peakIdx);
    float numerator   = y1 - y3;
    float denominator = 2.0f * (y1 - 2.0f * y2 + y3);

    // No clamp on delta, only check for division by zero
    if (std::abs(denominator) > std::numeric_limits<float>::epsilon())
    {
        float delta = numerator / denominator;
        interpIdx   = static_cast<float>(peakIdx) + delta;
    }

    return interpIdx;
}

/**
 * @brief 1D Quadratic interpolation on power spectrum
 *
 * @param spectrum Power spectrum array
 * @param peakIdx  Integer peak index
 * @param numBins  Number of bins in spectrum
 * @return Interpolated fractional index
 */
static float QuadraticInterpolation1DFixedPoint(const std::vector<float> &spectrum, int32_t peakIdx, int32_t numBins)
{
    if (peakIdx <= 0 || peakIdx >= numBins - 1)
    {
        return static_cast<float>(peakIdx);
    }
    return QuadraticPeakInterpFloat(spectrum[peakIdx - 1], spectrum[peakIdx], spectrum[peakIdx + 1], peakIdx);
}

/**
 * @brief Compute beamforming for a rectangular subset of angle bins
 *
 * Used by Separable Bartlett for:
 *   - Step 1: Azimuth-only search (elStart = elEnd = el0)
 *   - Step 2: Local elevation search (azStart..azEnd × all elevations)
 *
 * @param snapshotPtr Pointer to snapshot data
 * @param numTx Number of TX antennas
 * @param numRx Number of RX antennas
 * @param steeringVectorsPtr Full steering vectors [numAz][numEl][numTx][numRx]
 * @param numElevationBins Total number of elevation bins (for global indexing)
 * @param azStartIdx Starting azimuth index (inclusive)
 * @param azEndIdx Ending azimuth index (inclusive)
 * @param elStartIdx Starting elevation index (inclusive)
 * @param elEndIdx Ending elevation index (inclusive)
 * @return BeamformerOutput for the subset, stored in row-major [localAz][localEl] order
 */
static BeamformerOutput ComputeBeamformingSubset(const int32_t *snapshotPtr, int32_t numTx, int32_t numRx,
                                                 const int16_t *steeringVectorsPtr, int32_t numElevationBins,
                                                 int32_t azStartIdx, int32_t azEndIdx, int32_t elStartIdx,
                                                 int32_t elEndIdx)
{
    BeamformerOutput output;
    int32_t numLocalAz  = azEndIdx - azStartIdx + 1;
    int32_t numLocalEl  = elEndIdx - elStartIdx + 1;
    int32_t numLocalAng = numLocalAz * numLocalEl;
    output.numAngleBins = numLocalAng;
    output.data.resize(numLocalAng);

    int32_t localIdx = 0;
    for (int32_t azIdx = azStartIdx; azIdx <= azEndIdx; azIdx++)
    {
        for (int32_t elIdx = elStartIdx; elIdx <= elEndIdx; elIdx++)
        {
            // Reuse the core beamforming function
            output.data[localIdx++] = ComputeBeamformingForAngle(snapshotPtr, numTx, numRx, steeringVectorsPtr,
                                                                 numElevationBins, azIdx, elIdx);
        }
    }

    return output;
}

/**
 * @brief Compute power spectrum from beamformer output and find peak
 *
 * @param beamformerOutput Beamformer output
 * @param numLocalAz Number of local azimuth bins
 * @param numLocalEl Number of local elevation bins
 * @return Tuple of (power spectrum, peak local azimuth index, peak elevation index, max power)
 */
static std::tuple<std::vector<float>, int32_t, int32_t, float> ComputePowerSpectrumAndFindPeak(
    const BeamformerOutput &beamformerOutput, int32_t numLocalAz, int32_t numLocalEl)
{
    std::vector<float> powerSpectrum(beamformerOutput.numAngleBins);
    float maxPower      = 0.0f;
    int32_t peakLocalAz = 0;
    int32_t peakLocalEl = 0;

    for (int32_t i = 0; i < beamformerOutput.numAngleBins; i++)
    {
        float realFloat  = static_cast<float>(beamformerOutput.data[i].first);
        float imagFloat  = static_cast<float>(beamformerOutput.data[i].second);
        float power      = realFloat * realFloat + imagFloat * imagFloat;
        powerSpectrum[i] = power;

        if (power > maxPower)
        {
            maxPower    = power;
            peakLocalAz = i / numLocalEl;
            peakLocalEl = i % numLocalEl;
        }
    }

    return {powerSpectrum, peakLocalAz, peakLocalEl, maxPower};
}

/**
 * @brief Extract local 3x3 power spectrum from already-computed power spectrum
 *
 * Extracts a 3x3 neighborhood around the peak from the power spectrum computed in Step 2.
 * This avoids redundant beamforming computations.
 *
 * @param powerSpectrum Already-computed power spectrum from Step 2 [localAz * numEl + el]
 * @param localAzPeakIdx Local azimuth index of peak within the neighborhood
 * @param peakElIdx Elevation index of peak
 * @param numLocalAz Number of azimuth bins in the local spectrum
 * @param numEl Number of elevation bins
 * @return 3x3 power spectrum indexed as [elOffset][azOffset] where offset ∈ {0,1,2}
 *         (center at [1][1])
 */
static std::vector<std::vector<float>> ExtractLocal3x3FromPowerSpectrum(const std::vector<float> &powerSpectrum,
                                                                        int32_t localAzPeakIdx, int32_t peakElIdx,
                                                                        int32_t numLocalAz, int32_t numEl)
{
    std::vector<std::vector<float>> localSpectrum(3, std::vector<float>(3));

    // Compute clamped indices for 3x3 grid (handle boundaries)
    int32_t azIndices[3] = {std::max(0, localAzPeakIdx - 1), localAzPeakIdx,
                            std::min(numLocalAz - 1, localAzPeakIdx + 1)};
    int32_t elIndices[3] = {std::max(0, peakElIdx - 1), peakElIdx, std::min(numEl - 1, peakElIdx + 1)};

    for (int32_t ai = 0; ai < 3; ai++)
    {
        for (int32_t ei = 0; ei < 3; ei++)
        {
            // Extract from already-computed power spectrum
            int32_t idx           = azIndices[ai] * numEl + elIndices[ei];
            localSpectrum[ei][ai] = powerSpectrum[idx];
        }
    }

    return localSpectrum;
}

/**
 * @brief 2D quadratic interpolation on 3x3 local spectrum
 *
 * @param spectrum 3x3 power spectrum [el][az]
 * @param peakRow  Peak row index (should be 1 for center)
 * @param peakCol  Peak column index (should be 1 for center)
 * @return Pair of (interpolated azimuth offset, interpolated elevation offset)
 */
static std::pair<float, float> QuadraticInterpolation2DFixedPoint(const std::vector<std::vector<float>> &spectrum,
                                                                  int32_t peakRow, int32_t peakCol)
{
    float interpAzIdx = static_cast<float>(peakCol);
    float interpElIdx = static_cast<float>(peakRow);

    // Azimuth interpolation (along columns at center row)
    if (peakCol > 0 && peakCol < 2)
    {
        interpAzIdx = QuadraticPeakInterpFloat(spectrum[peakRow][peakCol - 1], spectrum[peakRow][peakCol],
                                               spectrum[peakRow][peakCol + 1], peakCol);
    }

    // Elevation interpolation (along rows at center column)
    if (peakRow > 0 && peakRow < 2)
    {
        interpElIdx = QuadraticPeakInterpFloat(spectrum[peakRow - 1][peakCol], spectrum[peakRow][peakCol],
                                               spectrum[peakRow + 1][peakCol], peakRow);
    }

    return std::make_pair(interpAzIdx, interpElIdx);
}

/**
 * @brief Convert float interpolated index to angle value
 *
 * @param interpIdx   Interpolated index as floating-point
 * @param bins        Array of angle bin values (degrees)
 * @param numBins     Number of bins in the array
 * @return            Interpolated angle value (degrees)
 */
static float InterpIdxToAngleFloat(float interpIdx, const std::vector<float> &bins, int32_t numBins)
{
    int32_t lowerIdx = static_cast<int32_t>(interpIdx);
    int32_t upperIdx = lowerIdx + 1;
    lowerIdx         = std::clamp(lowerIdx, 0, numBins - 1);
    upperIdx         = std::clamp(upperIdx, 0, numBins - 1);

    if (lowerIdx != upperIdx)
    {
        float fraction = interpIdx - static_cast<float>(lowerIdx);
        return bins[lowerIdx] + fraction * (bins[upperIdx] - bins[lowerIdx]);
    }
    return bins[lowerIdx];
}

/**
 * @brief Peak finding with optional quadratic interpolation
 */
std::pair<float, float> FindPeakWithInterpolation(const BartlettPowerSpectrum &powerSpec, bool enableInterpolation)
{
    int32_t peakAzIdx = powerSpec.peakAngleIndex / powerSpec.numElevationBins;
    int32_t peakElIdx = powerSpec.peakAngleIndex % powerSpec.numElevationBins;

    if (!enableInterpolation)
    {
        return std::make_pair(powerSpec.azimuthBins[peakAzIdx], powerSpec.elevationBins[peakElIdx]);
    }

    // Azimuth interpolation (using float)
    float interpAzIdx = static_cast<float>(peakAzIdx);
    if (peakAzIdx > 0 && peakAzIdx < powerSpec.numAzimuthBins - 1)
    {
        int32_t azLeftIdx   = (peakAzIdx - 1) * powerSpec.numElevationBins + peakElIdx;
        int32_t azCenterIdx = peakAzIdx * powerSpec.numElevationBins + peakElIdx;
        int32_t azRightIdx  = (peakAzIdx + 1) * powerSpec.numElevationBins + peakElIdx;

        interpAzIdx = QuadraticPeakInterpFloat(powerSpec.data[azLeftIdx], powerSpec.data[azCenterIdx],
                                               powerSpec.data[azRightIdx], peakAzIdx);
    }
    interpAzIdx = std::clamp(interpAzIdx, 0.0f, static_cast<float>(powerSpec.numAzimuthBins - 1));

    // Elevation interpolation (using float)
    float interpElIdx = static_cast<float>(peakElIdx);
    if (peakElIdx > 0 && peakElIdx < powerSpec.numElevationBins - 1)
    {
        int32_t elLowerIdx  = peakAzIdx * powerSpec.numElevationBins + (peakElIdx - 1);
        int32_t elCenterIdx = peakAzIdx * powerSpec.numElevationBins + peakElIdx;
        int32_t elUpperIdx  = peakAzIdx * powerSpec.numElevationBins + (peakElIdx + 1);

        interpElIdx = QuadraticPeakInterpFloat(powerSpec.data[elLowerIdx], powerSpec.data[elCenterIdx],
                                               powerSpec.data[elUpperIdx], peakElIdx);
    }
    interpElIdx = std::clamp(interpElIdx, 0.0f, static_cast<float>(powerSpec.numElevationBins - 1));

    // Convert to angles (using float interpolated indices)
    float peakAzimuth   = InterpIdxToAngleFloat(interpAzIdx, powerSpec.azimuthBins, powerSpec.numAzimuthBins);
    float peakElevation = InterpIdxToAngleFloat(interpElIdx, powerSpec.elevationBins, powerSpec.numElevationBins);

    return std::make_pair(peakAzimuth, peakElevation);
}

/**
 * @brief Main DOA estimation pipeline using fixed-point Bartlett beamforming
 */
DoaResult DoaEstimationBartlettFixedPoint(const int32_t *snapshotsPtr, int32_t numDetections,
                                          int32_t numVirtualChannels, int32_t numTx, int32_t numRx,
                                          const int16_t *steeringVectorsPtr, const float *azimuthBins,
                                          int32_t numAzimuthBins, const float *elevationBins, int32_t numElevationBins,
                                          bool enableInterpolation)
{
    DoaResult result;

    result.azimuthDegrees.resize(numDetections);
    result.elevationDegrees.resize(numDetections);
    result.peakPowerDb.resize(numDetections);

    for (int32_t targetIdx = 0; targetIdx < numDetections; targetIdx++)
    {
        const int32_t *targetSnapshotPtr = snapshotsPtr + (targetIdx * numVirtualChannels * 2);

        BeamformerOutput beamformerOutput =
            ComputeBeamforming(targetSnapshotPtr, numTx, numRx, steeringVectorsPtr, numAzimuthBins, numElevationBins);

        BartlettPowerSpectrum powerSpec = ComputeBartlettPowerSpectrum(beamformerOutput, azimuthBins, numAzimuthBins,
                                                                       elevationBins, numElevationBins);

        std::pair<float, float> angles = FindPeakWithInterpolation(powerSpec, enableInterpolation);

        // Convert power to dB: 10 * log10(power)
        // maxPowerFloat = real² + imag² where real/imag are SQ18.23
        // To normalize: divide by 2^46 (2^23 squared for power)
        // dB = 10 * log10(power / 2^46)
        constexpr float INV_BEAMFORMING_POWER_SCALE =
            1.0f / (static_cast<float>(1LL << BEAMFORMING_QBITS) * static_cast<float>(1LL << BEAMFORMING_QBITS));
        float maxPowerNormalized = powerSpec.maxPowerFloat * INV_BEAMFORMING_POWER_SCALE;
        float peakPowerDb        = 10.0f * std::log10(maxPowerNormalized + std::numeric_limits<float>::epsilon());

        result.azimuthDegrees[targetIdx]   = angles.first;
        result.elevationDegrees[targetIdx] = angles.second;
        result.peakPowerDb[targetIdx]      = peakPowerDb;
    }

    return result;
}

/**
 * @brief Separable Bartlett DOA estimation using fixed-point beamforming
 *
 * This implements a two-step separable search:
 * Step 1: Estimate azimuth with elevation fixed at 0
 * Step 2: Estimate elevation using azimuth neighborhood
 * Step 3: Apply interpolation (separate 1D or 2D on local 3x3)
 *
 * Complexity: O(numAz × numVC + numLocalAz × numEl × numVC) instead of O(numAz × numEl × numVC)
 */
DoaResult DoaEstimationBartlettSeparableFixedPoint(const int32_t *snapshotsPtr, int32_t numDetections,
                                                   int32_t numVirtualChannels, int32_t numTx, int32_t numRx,
                                                   const int16_t *steeringVectorsPtr, const float *azimuthBins,
                                                   int32_t numAzimuthBins, const float *elevationBins,
                                                   int32_t numElevationBins, int32_t azNeighborhood, bool azFromEl0,
                                                   bool enableInterpolation, uint64_t azChannelMask)
{
    DoaResult result;

    result.azimuthDegrees.resize(numDetections);
    result.elevationDegrees.resize(numDetections);
    result.peakPowerDb.resize(numDetections);

    // Create masked steering vectors for Step 1 if azChannelMask is set
    std::vector<int16_t> maskedSteeringVectors;
    const int16_t *step1SteeringPtr = steeringVectorsPtr;
    if (azChannelMask != 0)
    {
        int32_t totalElements = numAzimuthBins * numElevationBins * numVirtualChannels * 2;
        maskedSteeringVectors.assign(steeringVectorsPtr, steeringVectorsPtr + totalElements);

        // Zero out steering vector entries for masked channels
        for (int32_t angIdx = 0; angIdx < numAzimuthBins * numElevationBins; angIdx++)
        {
            for (int32_t txIdx = 0; txIdx < numTx; txIdx++)
            {
                for (int32_t rxIdx = 0; rxIdx < numRx; rxIdx++)
                {
                    int32_t vcIdx = txIdx * numRx + rxIdx;
                    if (!((azChannelMask >> vcIdx) & 1))
                    {
                        int32_t steerIdx                        = (angIdx * numTx + txIdx) * numRx + rxIdx;
                        maskedSteeringVectors[2 * steerIdx]     = 0;
                        maskedSteeringVectors[2 * steerIdx + 1] = 0;
                    }
                }
            }
        }
        step1SteeringPtr = maskedSteeringVectors.data();
    }

    // Find elevation index closest to 0 degrees for Step 1
    int32_t elZeroIdx = 0;
    float minElDist   = std::abs(elevationBins[0]);
    for (int32_t i = 1; i < numElevationBins; i++)
    {
        float dist = std::abs(elevationBins[i]);
        if (dist < minElDist)
        {
            minElDist = dist;
            elZeroIdx = i;
        }
    }

    // Copy angle bins to vectors for interpolation helper
    std::vector<float> azBinsVec(azimuthBins, azimuthBins + numAzimuthBins);
    std::vector<float> elBinsVec(elevationBins, elevationBins + numElevationBins);

    for (int32_t targetIdx = 0; targetIdx < numDetections; targetIdx++)
    {
        const int32_t *targetSnapshotPtr = snapshotsPtr + (targetIdx * numVirtualChannels * 2);

        // =====================================================================
        // Step 1: Azimuth estimation (elevation = 0 or closest bin)
        // Searches all azimuth bins with elevation fixed at el≈0
        // Complexity: O(numAz × numVC) - linear in azimuth bins only
        // =====================================================================
        BeamformerOutput beamformerAz =
            ComputeBeamformingSubset(targetSnapshotPtr, numTx, numRx, step1SteeringPtr, numElevationBins, 0,
                                     numAzimuthBins - 1, elZeroIdx, elZeroIdx);

        // Compute power spectrum and find peak
        std::vector<float> powerSpectrumAz(beamformerAz.numAngleBins);
        float maxPowerAz       = 0.0f;
        int32_t peakAzimuthIdx = 0;

        for (int32_t i = 0; i < beamformerAz.numAngleBins; i++)
        {
            float realFloat    = static_cast<float>(beamformerAz.data[i].first);
            float imagFloat    = static_cast<float>(beamformerAz.data[i].second);
            float power        = realFloat * realFloat + imagFloat * imagFloat;
            powerSpectrumAz[i] = power;

            if (power > maxPowerAz)
            {
                maxPowerAz     = power;
                peakAzimuthIdx = i;
            }
        }

        // =====================================================================
        // Step 2: Elevation estimation using azimuth neighborhood
        // =====================================================================
        int32_t azStartIdx     = std::max(0, peakAzimuthIdx - azNeighborhood);
        int32_t azEndIdx       = std::min(numAzimuthBins - 1, peakAzimuthIdx + azNeighborhood);
        int32_t numAzNeighbors = azEndIdx - azStartIdx + 1;

        // Complexity: O(numLocalAz × numEl × numVC) - searches small neighborhood
        BeamformerOutput beamformerLocal =
            ComputeBeamformingSubset(targetSnapshotPtr, numTx, numRx, steeringVectorsPtr, numElevationBins, azStartIdx,
                                     azEndIdx, 0, numElevationBins - 1);

        // Find peak in local 2D spectrum
        auto [powerSpectrumLocal, localAzPeakIdx, peakElIdx, maxPower] =
            ComputePowerSpectrumAndFindPeak(beamformerLocal, numAzNeighbors, numElevationBins);

        // Update azimuth index based on azFromEl0 option
        int32_t finalPeakAzIdx = peakAzimuthIdx; // Keep from Step 1 by default
        if (!azFromEl0)
        {
            // Update azimuth to the best one from the neighborhood search
            finalPeakAzIdx = azStartIdx + localAzPeakIdx;
        }

        // =====================================================================
        // Step 3: Apply interpolation
        // =====================================================================
        float peakAzimuth   = 0.0f;
        float peakElevation = 0.0f;

        if (enableInterpolation)
        {
            if (azFromEl0)
            {
                // Separate 1D interpolations when azFromEl0=true
                // This decouples azimuth and elevation interpolation for better accuracy

                // 1D azimuth interpolation on el=0 power spectrum (from Step 1)
                float interpAzIdx = QuadraticInterpolation1DFixedPoint(powerSpectrumAz, peakAzimuthIdx, numAzimuthBins);

                // 1D elevation interpolation on power spectrum where elevation peak was found
                std::vector<float> elevationSlice(numElevationBins);
                for (int32_t ei = 0; ei < numElevationBins; ei++)
                {
                    elevationSlice[ei] = powerSpectrumLocal[localAzPeakIdx * numElevationBins + ei];
                }
                float interpElIdx = QuadraticInterpolation1DFixedPoint(elevationSlice, peakElIdx, numElevationBins);

                // Convert interpolated indices to angles
                peakAzimuth   = InterpIdxToAngleFloat(interpAzIdx, azBinsVec, numAzimuthBins);
                peakElevation = InterpIdxToAngleFloat(interpElIdx, elBinsVec, numElevationBins);
            }
            else
            {
                // 2D quadratic interpolation using local 3x3 spectrum
                // Extract from already-computed powerSpectrumLocal (avoids redundant beamforming)
                std::vector<std::vector<float>> localSpectrum = ExtractLocal3x3FromPowerSpectrum(
                    powerSpectrumLocal, localAzPeakIdx, peakElIdx, numAzNeighbors, numElevationBins);

                // Apply 2D quadratic interpolation on the 3x3 local spectrum
                // Peak is at center (1, 1) of the 3x3 grid
                auto [interpAzOffset, interpElOffset] = QuadraticInterpolation2DFixedPoint(localSpectrum, 1, 1);

                // Convert offsets to actual indices (offset from center) and clamp
                float interpAzIdx = std::clamp(static_cast<float>(finalPeakAzIdx) + (interpAzOffset - 1.0f), 0.0f,
                                               static_cast<float>(numAzimuthBins - 1));
                float interpElIdx = std::clamp(static_cast<float>(peakElIdx) + (interpElOffset - 1.0f), 0.0f,
                                               static_cast<float>(numElevationBins - 1));

                // Convert interpolated indices to angles
                peakAzimuth   = InterpIdxToAngleFloat(interpAzIdx, azBinsVec, numAzimuthBins);
                peakElevation = InterpIdxToAngleFloat(interpElIdx, elBinsVec, numElevationBins);
            }
        }
        else
        {
            // Use peak bins directly without interpolation
            peakAzimuth   = azimuthBins[finalPeakAzIdx];
            peakElevation = elevationBins[peakElIdx];
        }

        // Convert power to dB
        constexpr float INV_BEAMFORMING_POWER_SCALE =
            1.0f / (static_cast<float>(1LL << BEAMFORMING_QBITS) * static_cast<float>(1LL << BEAMFORMING_QBITS));
        float maxPowerNormalized = maxPower * INV_BEAMFORMING_POWER_SCALE;
        float peakPowerDb        = 10.0f * std::log10(maxPowerNormalized + std::numeric_limits<float>::epsilon());

        result.azimuthDegrees[targetIdx]   = peakAzimuth;
        result.elevationDegrees[targetIdx] = peakElevation;
        result.peakPowerDb[targetIdx]      = peakPowerDb;
    }

    return result;
}

void computeSteeringVectorsRef(NVCVTensorHandle steeringVectorsTensorHandle,
                               NVCVTensorHandle virtualArrayLocationsTensorHandle,
                               NVCVTensorHandle azimuthBinsTensorHandle, NVCVTensorHandle elevationBinsTensorHandle)
{
    // Delegate to the operator's public API for steering vector computation
    pvaDoaBartlettBeamformingComputeSteeringVectors(virtualArrayLocationsTensorHandle, azimuthBinsTensorHandle,
                                                    elevationBinsTensorHandle, steeringVectorsTensorHandle);
}

void doaBartlettBeamformingRef(NVCVTensorHandle detectionCountTensorHandle, NVCVTensorHandle snapshotsTensorHandle,
                               NVCVTensorHandle steeringVectorsTensorHandle, NVCVTensorHandle azimuthBinsTensorHandle,
                               NVCVTensorHandle elevationBinsTensorHandle, NVCVTensorHandle targetCountTensorHandle,
                               NVCVTensorHandle targetAnglesTensorHandle, NVCVTensorHandle targetIndexMapTensorHandle,
                               const PVADoaBartlettBeamformingParams *params)
{
    // Step 1: Get detection count
    NVCVTensorData detectionCountData;
    nvcvTensorExportData(detectionCountTensorHandle, &detectionCountData);
    int32_t *detectionCountPtr = NULL;
    CupvaMemGetHostPointer((void **)&detectionCountPtr, (void *)detectionCountData.buffer.strided.basePtr);
    const int32_t detectionCount = detectionCountPtr[0];

    // Get target angles capacity to clamp target count
    NVCVTensorData targetAnglesData;
    nvcvTensorExportData(targetAnglesTensorHandle, &targetAnglesData);
    float *targetAnglesPtr = NULL;
    CupvaMemGetHostPointer((void **)&targetAnglesPtr, (void *)targetAnglesData.buffer.strided.basePtr);

    int64_t anglesShapeHW[2];
    nvcvTensorShapePermute(targetAnglesData.layout, targetAnglesData.shape, NVCV_TENSOR_HW, anglesShapeHW);
    const int32_t anglesStride = anglesShapeHW[1]; // W = numTargets (stride)

    // Clamp target count to not exceed buffer capacity
    const int32_t numDetections = std::min(detectionCount, static_cast<int32_t>(anglesStride));

    // Write target count (clamped to buffer capacity)
    NVCVTensorData targetCountData;
    nvcvTensorExportData(targetCountTensorHandle, &targetCountData);
    int32_t *targetCountPtr = NULL;
    CupvaMemGetHostPointer((void **)&targetCountPtr, (void *)targetCountData.buffer.strided.basePtr);
    targetCountPtr[0] = numDetections;

    if (numDetections == 0)
    {
        return;
    }

    // Step 2: Read snapshots
    NVCVTensorData snapshotsData;
    nvcvTensorExportData(snapshotsTensorHandle, &snapshotsData);
    int32_t *snapshotsPtr = NULL;
    CupvaMemGetHostPointer((void **)&snapshotsPtr, (void *)snapshotsData.buffer.strided.basePtr);

    int64_t snapshotsShapeCHW[3];
    nvcvTensorShapePermute(snapshotsData.layout, snapshotsData.shape, NVCV_TENSOR_CHW, snapshotsShapeCHW);
    const int32_t numTx              = snapshotsShapeCHW[1];
    const int32_t numRx              = snapshotsShapeCHW[2];
    const int32_t numVirtualChannels = numTx * numRx;

    // Step 3: Read steering vectors
    NVCVTensorData steeringVectorsData;
    nvcvTensorExportData(steeringVectorsTensorHandle, &steeringVectorsData);
    int16_t *steeringVectorsPtr = NULL;
    CupvaMemGetHostPointer((void **)&steeringVectorsPtr, (void *)steeringVectorsData.buffer.strided.basePtr);

    // Step 4: Read angle bins
    NVCVTensorData azimuthBinsData;
    nvcvTensorExportData(azimuthBinsTensorHandle, &azimuthBinsData);
    float *azimuthBinsPtr = NULL;
    CupvaMemGetHostPointer((void **)&azimuthBinsPtr, (void *)azimuthBinsData.buffer.strided.basePtr);
    const int32_t numAzimuthBins = static_cast<int32_t>(azimuthBinsData.shape[0]);

    NVCVTensorData elevationBinsData;
    nvcvTensorExportData(elevationBinsTensorHandle, &elevationBinsData);
    float *elevationBinsPtr = NULL;
    CupvaMemGetHostPointer((void **)&elevationBinsPtr, (void *)elevationBinsData.buffer.strided.basePtr);
    const int32_t numElevationBins = static_cast<int32_t>(elevationBinsData.shape[0]);

    // Step 5: Run Bartlett DOA estimation (separable or full 2D based on params)
    DoaResult doaResult;
    if (params->separableParams.enableSeparable)
    {
        doaResult = DoaEstimationBartlettSeparableFixedPoint(
            snapshotsPtr, numDetections, numVirtualChannels, numTx, numRx, steeringVectorsPtr, azimuthBinsPtr,
            numAzimuthBins, elevationBinsPtr, numElevationBins, params->separableParams.azNeighborhood,
            params->separableParams.azSource == PVA_BARTLETT_AZ_FROM_EL0, params->enableInterpolation,
            params->separableParams.azChannelMask);
    }
    else
    {
        doaResult = DoaEstimationBartlettFixedPoint(snapshotsPtr, numDetections, numVirtualChannels, numTx, numRx,
                                                    steeringVectorsPtr, azimuthBinsPtr, numAzimuthBins,
                                                    elevationBinsPtr, numElevationBins, params->enableInterpolation);
    }

    // Step 6: Write target angles [3, numTargets] - azimuth, elevation, power
    // Write target index map [numTargets] - 1:1 mapping (each detection produces one target)
    NVCVTensorData targetIndexMapData;
    nvcvTensorExportData(targetIndexMapTensorHandle, &targetIndexMapData);
    int32_t *targetIndexMapPtr = NULL;
    CupvaMemGetHostPointer((void **)&targetIndexMapPtr, (void *)targetIndexMapData.buffer.strided.basePtr);

    for (int32_t i = 0; i < numDetections; i++)
    {
        targetAnglesPtr[0 * anglesStride + i] = doaResult.azimuthDegrees[i];
        targetAnglesPtr[1 * anglesStride + i] = doaResult.elevationDegrees[i];
        targetAnglesPtr[2 * anglesStride + i] = doaResult.peakPowerDb[i];
        targetIndexMapPtr[i]                  = i;
    }
}

int32_t checkDoaBartlettBeamformingOutput(NVCVTensorHandle pvaTargetCountTensorHandle,
                                          NVCVTensorHandle refTargetCountTensorHandle,
                                          NVCVTensorHandle pvaTargetListTensorHandle,
                                          NVCVTensorHandle refTargetListTensorHandle)
{
    // First verify target counts match
    NVCVTensorData pvaTargetCountData, refTargetCountData;
    nvcvTensorExportData(pvaTargetCountTensorHandle, &pvaTargetCountData);
    nvcvTensorExportData(refTargetCountTensorHandle, &refTargetCountData);
    int32_t *pvaTargetCountPtr = NULL;
    int32_t *refTargetCountPtr = NULL;
    CupvaMemGetHostPointer((void **)&pvaTargetCountPtr, (void *)pvaTargetCountData.buffer.strided.basePtr);
    CupvaMemGetHostPointer((void **)&refTargetCountPtr, (void *)refTargetCountData.buffer.strided.basePtr);

    int32_t pvaCount = pvaTargetCountPtr[0];
    int32_t refCount = refTargetCountPtr[0];

    if (pvaCount < refCount)
    {
        printf("Target count FAIL: PVA has fewer targets (%d) than ref (%d)\n", pvaCount, refCount);
        return -1;
    }

    if (pvaCount > refCount)
    {
        printf("Target count: PVA has %d extra targets\n", pvaCount - refCount);
    }

    // Delegate to common target processing check - compare first refCount targets
    return checkTargetProcessingOutput(refTargetCountTensorHandle, pvaTargetListTensorHandle,
                                       refTargetListTensorHandle);
}

void loadAngleBins(NVCVTensorHandle angleBinsTensorHandle, int32_t numBins, float start, float step)
{
    NVCVTensorData angleBinsData;
    nvcvTensorExportData(angleBinsTensorHandle, &angleBinsData);
    float *angleBinsPtr = NULL;
    CupvaMemGetHostPointer((void **)&angleBinsPtr, (void *)angleBinsData.buffer.strided.basePtr);
    for (int32_t i = 0; i < numBins; i++)
    {
        angleBinsPtr[i] = start + i * step;
    }
}

void PopulateSteeringVectorsTensor(NVCVTensorHandle steeringVectorsTensorHandle, const int16_t *data, size_t size)
{
    NVCVTensorData steeringVectorsData;
    nvcvTensorExportData(steeringVectorsTensorHandle, &steeringVectorsData);

    const int32_t rank    = 4;
    int64_t totalElements = 1;
    for (int32_t i = 0; i < rank; i++)
    {
        totalElements *= steeringVectorsData.shape[i];
    }
    const size_t numInt16 = static_cast<size_t>(totalElements) * 2u;

    if (size != numInt16)
    {
        throw std::runtime_error("PopulateSteeringVectorsTensor: data size (" + std::to_string(size) +
                                 ") does not match tensor element count (" + std::to_string(numInt16) + ")");
    }

    int16_t *steeringVectorsPtr = NULL;
    CupvaMemGetHostPointer((void **)&steeringVectorsPtr, (void *)steeringVectorsData.buffer.strided.basePtr);
    std::memcpy(steeringVectorsPtr, data, numInt16 * sizeof(int16_t));
}

void PopulateVirtualArrayLocationsTensor(NVCVTensorHandle virtualArrayLocationsTensorHandle, int32_t numTx,
                                         int32_t numRx, float azimuthAntennaSpacing, float elevationAntennaSpacing)
{
    NVCVTensorData valData;
    nvcvTensorExportData(virtualArrayLocationsTensorHandle, &valData);
    if (valData.buffer.strided.basePtr != nullptr)
    {
        void *valHostPtr = nullptr;
        CupvaMemGetHostPointer(&valHostPtr, (void *)valData.buffer.strided.basePtr);
        if (valHostPtr != nullptr)
        {
            float *valPtr = reinterpret_cast<float *>(valHostPtr);
            for (int32_t txIdx = 0; txIdx < numTx; txIdx++)
            {
                for (int32_t rxIdx = 0; rxIdx < numRx; rxIdx++)
                {
                    int32_t idx     = (txIdx * numRx + rxIdx) * 2;
                    valPtr[idx]     = txIdx * azimuthAntennaSpacing;   // x position in wavelengths (TX spacing)
                    valPtr[idx + 1] = rxIdx * elevationAntennaSpacing; // y position in wavelengths (RX spacing)
                }
            }
        }
    }
}
