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
 * @file doa_angle_fft.cpp
 *
 * @brief Reference implementation for DOA Angle FFT operator
 *
 * This file provides a fixed-point reference implementation for the DOA (Direction of Arrival)
 * Angle FFT operation. This reference implementation is used for validation and corresponds
 * to the PVA operator defined in OpDoaAngleFFT.h with parameters PVADoaAngleFFTParams.
 *
 * The operator performs 2D Angle FFT processing on antenna snapshots to estimate azimuth
 * and elevation angles of detected radar targets.
 *
 * @see OpDoaAngleFFT.h for the PVA operator interface
 * @see PvaOperatorTypes.h for parameter structures
 */

#include "doa_angle_fft_ref.h"

#include <TensorUtils.hpp>

#include <cupva_host.h>
#include <nvcv/Tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <random>

#define TWIDDLE_QBITS (30)
#define TWIDDLE_SCALE (1 << TWIDDLE_QBITS)
#define WINDOW_QBITS (30)
#define WINDOW_SCALE (1 << WINDOW_QBITS)
#define POWER_QBITS (20)
#define ANGLE_QBITS (16)
#define ANGLE_SCALE (1 << ANGLE_QBITS)
#define TRUNCATE_QBITS (16)
#define ADDITIONAL_SHIFT_QBITS (2)

#ifndef PI_CONST_F
#    define PI_CONST_F 3.14159265358979323846f
#endif

/**
 * @brief Convert angle estimation precision enum to scaling factor.
 *
 * Maps the precision enum value to the corresponding FFT scaling factor:
 *   - LOW (0)     -> 2
 *   - DEFAULT (1) -> 4
 *   - HIGH (2)    -> 8
 *
 * @param precision The angle estimation precision level
 * @return The scaling factor (2, 4, or 8), defaults to 4 for invalid input
 */
static inline int32_t precisionToScalingFactor(PVADoaAngleFFTPrecision precision)
{
    switch (precision)
    {
    case PVA_ANGLE_ESTIMATION_PRECISION_LOW:
        return 2;
    case PVA_ANGLE_ESTIMATION_PRECISION_DEFAULT:
        return 4;
    case PVA_ANGLE_ESTIMATION_PRECISION_HIGH:
        return 8;
    default:
        return 4;
    }
}

// Helper function for rounding with scaling
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

// Helper function to determine the number of bits needed to represent n
int bit_length(int n)
{
    if (n <= 1)
    {
        return 0;
    }
    return (int)ceil(log2(n));
}

// Radix-4 digit reversal function
int r4_digit_reverse(int i, int log2n)
{
    int log4n = log2n / 2;
    int res   = 0;
    for (int k = 0; k < log4n; k++)
    {
        res <<= 2;
        res |= (i & 3);
        i >>= 2;
    }
    return res;
}

// Single stage of Radix-4 DIF FFT
void fft_fixedp_r4_dif_stage(int n, int k, int32_t *x_interleaved, int32_t *twiddle_interleaved, int32_t *y)
{
    int log2n   = bit_length(n);
    int ngroups = 1 << (2 * k);
    int groupsz = n / ngroups;
    int dist    = groupsz / 4;

    // Use Q30 format for twiddle factors
    int32_t qbits = ADDITIONAL_SHIFT_QBITS + (TWIDDLE_QBITS - TRUNCATE_QBITS);

    for (int i = 0; i < ngroups; i++)
    {
        for (int j = 0; j < dist; j++)
        {
            int didx = i * groupsz + j;

            // Twiddle indices
            int t1idx = j * (1 << (2 * k));
            int t2idx = 2 * t1idx;
            int t3idx = 3 * t1idx;

            // Read input data (Q20 format)
            int64_t x0r = x_interleaved[2 * didx];
            int64_t x0i = x_interleaved[2 * didx + 1];
            int64_t x1r = x_interleaved[2 * (didx + dist)];
            int64_t x1i = x_interleaved[2 * (didx + dist) + 1];
            int64_t x2r = x_interleaved[2 * (didx + 2 * dist)];
            int64_t x2i = x_interleaved[2 * (didx + 2 * dist) + 1];
            int64_t x3r = x_interleaved[2 * (didx + 3 * dist)];
            int64_t x3i = x_interleaved[2 * (didx + 3 * dist) + 1];

            // Fetch twiddle factors (Q30 format)
            int64_t t1r = twiddle_interleaved[2 * t1idx];
            int64_t t1i = twiddle_interleaved[2 * t1idx + 1];
            int64_t t2r = twiddle_interleaved[2 * t2idx];
            int64_t t2i = twiddle_interleaved[2 * t2idx + 1];
            int64_t t3r = twiddle_interleaved[2 * t3idx];
            int64_t t3i = twiddle_interleaved[2 * t3idx + 1];

            // Butterfly intermediate values
            int64_t z0r = x0r + x2r;
            int64_t z0i = x0i + x2i;
            int64_t z1r = x0r - x2r;
            int64_t z1i = x0i - x2i;
            int64_t z2r = x1r + x3r;
            int64_t z2i = x1i + x3i;
            int64_t z3r = x1r - x3r;
            int64_t z3i = x1i - x3i;

            // Butterfly outputs before twiddle multiplication
            int64_t w0r = z0r + z2r;
            int64_t w0i = z0i + z2i;
            int64_t w2r = z0r - z2r;
            int64_t w2i = z0i - z2i;
            int64_t w1r = z1r + z3i; // w1 = z1 - j*z3
            int64_t w1i = z1i - z3r;
            int64_t w3r = z1r - z3i; // w3 = z1 + j*z3
            int64_t w3i = z1i + z3r;

            // Apply twiddle factors and scale
            // For w0 (no twiddle factor), keep Q20 format
            y[2 * didx]     = my_round(w0r, ADDITIONAL_SHIFT_QBITS);
            y[2 * didx + 1] = my_round(w0i, ADDITIONAL_SHIFT_QBITS);

            // For w1, w2, w3: complex multiplication with Q30 twiddle factors
            y[2 * (didx + dist)] = my_round(((w1r * t1r) >> TRUNCATE_QBITS) - ((w1i * t1i) >> TRUNCATE_QBITS), qbits);
            y[2 * (didx + dist) + 1] =
                my_round(((w1r * t1i) >> TRUNCATE_QBITS) + ((w1i * t1r) >> TRUNCATE_QBITS), qbits);

            y[2 * (didx + 2 * dist)] =
                my_round(((w2r * t2r) >> TRUNCATE_QBITS) - ((w2i * t2i) >> TRUNCATE_QBITS), qbits);
            y[2 * (didx + 2 * dist) + 1] =
                my_round(((w2r * t2i) >> TRUNCATE_QBITS) + ((w2i * t2r) >> TRUNCATE_QBITS), qbits);

            y[2 * (didx + 3 * dist)] =
                my_round(((w3r * t3r) >> TRUNCATE_QBITS) - ((w3i * t3i) >> TRUNCATE_QBITS), qbits);
            y[2 * (didx + 3 * dist) + 1] =
                my_round(((w3r * t3i) >> TRUNCATE_QBITS) + ((w3i * t3r) >> TRUNCATE_QBITS), qbits);
        }
    }
}

// Radix-2 DIF FFT stage
void fft_fixedp_r2_dif_stage(int n, int k, int32_t *x, int32_t *twiddle, int32_t *y)
{
    int ngroups = 1 << k;
    int groupsz = n / ngroups;
    int dist    = groupsz / 2;
    int qbits   = 1 + (TWIDDLE_QBITS - TRUNCATE_QBITS);

    for (int i = 0; i < ngroups; i++)
    {
        for (int j = 0; j < dist; j++)
        {
            int didx = i * groupsz + j;
            int tidx = j * (1 << k);

            // Read input values
            int64_t ar = x[2 * didx];
            int64_t ai = x[2 * didx + 1];
            int64_t br = x[2 * (didx + dist)];
            int64_t bi = x[2 * (didx + dist) + 1];

            int64_t tr = twiddle[2 * tidx];
            int64_t ti = twiddle[2 * tidx + 1];

            // Compute butterfly
            int64_t cr = ar + br;
            int64_t ci = ai + bi;
            int64_t dr = ar - br;
            int64_t di = ai - bi;

            // Write output values
            y[2 * didx]              = my_round(cr, 1);
            y[2 * didx + 1]          = my_round(ci, 1);
            y[2 * (didx + dist)]     = my_round(((dr * tr) >> TRUNCATE_QBITS) - ((di * ti) >> TRUNCATE_QBITS), qbits);
            y[2 * (didx + dist) + 1] = my_round(((dr * ti) >> TRUNCATE_QBITS) + ((di * tr) >> TRUNCATE_QBITS), qbits);
        }
    }
}

// Full Radix-4 DIF FFT (for power-of-4 sizes)
void fft_fixedp_r4_dif(int n, int32_t *x_interleaved)
{
    int log2n = bit_length(n);
    int log4n = log2n / 2;

    if (log2n % 2 != 0)
    {
        printf("Error: FFT size 'n' must be a power of 4.\n");
        return;
    }
    int32_t twiddle_scale = TWIDDLE_SCALE;

    // Pre-compute twiddle factors
    std::vector<int32_t> twiddle_interleaved(2 * n);
    for (int i = 0; i < n; i++)
    {
        double angle                   = -2.0 * M_PI * i / n; // Negative sign for forward FFT
        twiddle_interleaved[2 * i]     = (int32_t)round(cos(angle) * twiddle_scale);
        twiddle_interleaved[2 * i + 1] = (int32_t)round(sin(angle) * twiddle_scale); // Positive sin for forward FFT
    }

    // Allocate working buffers
    std::vector<int32_t> stage_data(2 * n);
    std::vector<int32_t> temp_data(2 * n);

    // Copy input to working buffer (input is Q20, but we'll process as Q20)
    for (int i = 0; i < 2 * n; i++)
    {
        stage_data[i] = x_interleaved[i];
    }

    // Process FFT stages
    for (int k = 0; k < log4n; k++)
    {
        fft_fixedp_r4_dif_stage(n, k, stage_data.data(), twiddle_interleaved.data(), temp_data.data());
        // Swap buffers
        stage_data.swap(temp_data);
    }

    // Final permutation (Radix-4 digit reversal)
    for (int i = 0; i < n; i++)
    {
        int j                    = r4_digit_reverse(i, log2n);
        x_interleaved[2 * i]     = stage_data[2 * j];
        x_interleaved[2 * i + 1] = stage_data[2 * j + 1];
    }
}

// Mixed radix-4/radix-2 DIF FFT for odd-sized inputs
void fft_fixedp_r4_dif_odd(int n, int32_t *x_interleaved)
{
    int log2n = bit_length(n);
    int log4n = log2n / 2;

    // Allocate buffers
    std::vector<int32_t> twiddle(2 * n);
    std::vector<int32_t> buf1(2 * n);
    std::vector<int32_t> buf2(2 * n);
    int32_t *stage_in  = x_interleaved;
    int32_t *stage_out = buf1.data();

    // Pre-compute twiddle factors
    for (int i = 0; i < n; i++)
    {
        double angle       = -2.0 * M_PI * i / n;
        twiddle[2 * i]     = (int32_t)round(cos(angle) * TWIDDLE_SCALE);
        twiddle[2 * i + 1] = (int32_t)round(sin(angle) * TWIDDLE_SCALE);
    }

    // Process radix-4 stages
    for (int k = 0; k < log4n; k++)
    {
        fft_fixedp_r4_dif_stage(n, k, stage_in, twiddle.data(), stage_out);
        stage_in  = stage_out;
        stage_out = (k % 2) ? buf2.data() : buf1.data();
    }

    // Process final radix-2 stage
    for (int k = log2n - 1; k < log2n; k++)
    {
        fft_fixedp_r2_dif_stage(n, k, stage_in, twiddle.data(), stage_out);
        stage_in  = stage_out;
        stage_out = (k % 2) ? buf2.data() : buf1.data();
    }

    // Final permutation
    for (int i = 0; i < n; i++)
    {
        int k                    = i % (n / 2);
        int j                    = r4_digit_reverse(k, log2n - 1) * 2 + i / (n / 2);
        x_interleaved[2 * i]     = stage_in[2 * j];
        x_interleaved[2 * i + 1] = stage_in[2 * j + 1];
    }
}

// Helper function to check if log2n is even
inline bool is_log2n_even(int n)
{
    int log2n = bit_length(n);
    return ((log2n & 1) == 0);
}

// Generate fixed-point Hanning window in Q30 format
std::vector<int32_t> GenerateHanningWindowFixedPoint(int32_t length)
{
    std::vector<int32_t> window(length);

    if (length == 0)
        return window;

    if (length == 1)
    {
        window[0] = WINDOW_SCALE; // Full amplitude for single element
        return window;
    }

    for (int32_t i = 0; i < length; i++)
    {
        float hanningValue =
            0.5f * (1.0f - std::cos(2.0f * M_PI * static_cast<float>(i + 1) / static_cast<float>(length + 1)));
        window[i] = static_cast<int32_t>(hanningValue * WINDOW_SCALE);
    }

    return window;
}

// Apply fixed-point quadratic interpolation for sub-pixel accuracy
std::pair<int32_t, int32_t> QuadraticInterpolation(const PowerSpectrum &powerSpec)
{
    const std::vector<std::vector<float>> &spectrum = powerSpec.data;
    int32_t numElevBins                             = static_cast<int32_t>(spectrum.size());
    int32_t numAzBins                               = static_cast<int32_t>(spectrum[0].size());

    // Convert unshifted peak indices to shifted coordinates early
    int32_t elevIdx        = powerSpec.peakElevationIndex;
    int32_t azIdx          = powerSpec.peakAzimuthIndex;
    int32_t shiftedElevIdx = (elevIdx + numElevBins / 2) % numElevBins;
    int32_t shiftedAzIdx   = (azIdx + numAzBins / 2) % numAzBins;

    // Use Q16 format for interpolated indices (16 fractional bits for high precision)
    const int32_t INTERP_QBITS = ANGLE_QBITS; // Use same precision as angles
    const int32_t INTERP_SCALE = ANGLE_SCALE;

    // Initialize with shifted indices in Q16 format
    int32_t interpAzimuthIndex   = shiftedAzIdx * INTERP_SCALE;
    int32_t interpElevationIndex = shiftedElevIdx * INTERP_SCALE;

    // Quadratic interpolation for azimuth (horizontal neighbors) using circular indexing
    // Always safe to access neighbors with circular indexing
    {
        // Use circular indexing for neighbors (handles wraparound correctly)
        int32_t az_left   = (azIdx - 1 + numAzBins) % numAzBins; // Left neighbor
        int32_t az_center = azIdx;                               // Peak
        int32_t az_right  = (azIdx + 1) % numAzBins;             // Right neighbor

        float y1 = spectrum[elevIdx][az_left];   // Left neighbor
        float y2 = spectrum[elevIdx][az_center]; // Peak
        float y3 = spectrum[elevIdx][az_right];  // Right neighbor

        // Quadratic interpolation formula: x_peak = x2 + (y1 - y3) / (2 * (y1 - 2*y2 + y3))
        float numerator   = y1 - y3;
        float denominator = 2 * (y1 - 2 * y2 + y3);

        // Avoid division by zero
        if (denominator != 0)
        {
            float delta_az_float = numerator / denominator;
            int64_t delta_az     = (int64_t)(delta_az_float * INTERP_SCALE);

            // Clamp delta to reasonable range (±4 pixels in Q16)
            delta_az = std::clamp<int64_t>(delta_az, -4 * INTERP_SCALE, 4 * INTERP_SCALE);

            interpAzimuthIndex = shiftedAzIdx * INTERP_SCALE + static_cast<int32_t>(delta_az);
        }
    }

    // Quadratic interpolation for elevation (vertical neighbors) using circular indexing
    // Always safe to access neighbors with circular indexing
    {
        // Use circular indexing for neighbors (handles wraparound correctly)
        int32_t elev_lower  = (elevIdx - 1 + numElevBins) % numElevBins; // Lower neighbor
        int32_t elev_center = elevIdx;                                   // Peak
        int32_t elev_upper  = (elevIdx + 1) % numElevBins;               // Upper neighbor

        float y1 = spectrum[elev_lower][azIdx];  // Lower neighbor
        float y2 = spectrum[elev_center][azIdx]; // Peak
        float y3 = spectrum[elev_upper][azIdx];  // Upper neighbor

        // Quadratic interpolation formula: x_peak = x2 + (y1 - y3) / (2 * (y1 - 2*y2 + y3))
        float numerator   = y1 - y3;
        float denominator = 2 * (y1 - 2 * y2 + y3);

        // Avoid division by zero
        if (denominator != 0)
        {
            float delta_elev_float = numerator / denominator;
            int64_t delta_elev     = (int64_t)(delta_elev_float * INTERP_SCALE);

            // Clamp delta to reasonable range (±4 pixels in Q16)
            delta_elev = std::clamp<int64_t>(delta_elev, -4 * INTERP_SCALE, 4 * INTERP_SCALE);

            interpElevationIndex = shiftedElevIdx * INTERP_SCALE + static_cast<int32_t>(delta_elev);
        }
    }

    // Clamp interpolated indices to valid range in Q16 format
    interpAzimuthIndex   = std::clamp<int32_t>(interpAzimuthIndex, 0, (numAzBins - 1) * INTERP_SCALE);
    interpElevationIndex = std::clamp<int32_t>(interpElevationIndex, 0, (numElevBins - 1) * INTERP_SCALE);
    // Return indices already in shifted coordinates (deltas added directly)
    return std::make_pair(interpAzimuthIndex, interpElevationIndex);
}

// Initialize Q16 fixed-point angle bins for azimuth and elevation
std::pair<std::vector<int32_t>, std::vector<int32_t>> InitializeAngleBins(int32_t paddedAzimuthSize,
                                                                          int32_t paddedElevationSize)
{
    std::vector<int32_t> azimuthBins(paddedAzimuthSize);
    std::vector<int32_t> elevationBins(paddedElevationSize);

    // Generate azimuth angle bins: map FFT bin indices to angles in Q16 format (adjusted for removed conjugate)
    for (int32_t i = 0; i < paddedAzimuthSize; i++)
    {
        float normalizedIndex = 1.0f - 2.0f * (static_cast<float>(i) / static_cast<float>(paddedAzimuthSize));
        float angleDeg        = std::asin(std::max(-1.0f, std::min(1.0f, normalizedIndex))) * 180.0f / M_PI;
        azimuthBins[i]        = static_cast<int32_t>(angleDeg * ANGLE_SCALE);
    }

    // Generate elevation angle bins: map FFT bin indices to angles in Q16 format (adjusted for removed conjugate)
    for (int32_t i = 0; i < paddedElevationSize; i++)
    {
        float normalizedIndex = 1.0f - 2.0f * (static_cast<float>(i) / static_cast<float>(paddedElevationSize));
        float angleDeg        = std::asin(std::max(-1.0f, std::min(1.0f, normalizedIndex))) * 180.0f / M_PI;
        elevationBins[i]      = static_cast<int32_t>(angleDeg * ANGLE_SCALE);
    }

    return std::make_pair(azimuthBins, elevationBins);
}

// Apply joint 2D Hanning windowing and zero-padding, return padded FFTData structure
FFTData Windowing2D(const Snapshot &snapshot, int32_t paddedAzimuthSize, int32_t paddedElevationSize)
{
    std::vector<int32_t> azimuthWindow   = GenerateHanningWindowFixedPoint(snapshot.numAzimuthVirtualElements);
    std::vector<int32_t> elevationWindow = GenerateHanningWindowFixedPoint(snapshot.numElevationVirtualElements);

    FFTData result;
    result.paddedAzimuthSize   = paddedAzimuthSize;
    result.paddedElevationSize = paddedElevationSize;

    // Initialize padded array with zeros - interleaved format [real, imag, real, imag, ...]
    result.data.resize(paddedElevationSize, std::vector<int32_t>(2 * paddedAzimuthSize, 0));

    // Apply windowing and copy to padded array
    for (int32_t azIdx = 0; azIdx < snapshot.numAzimuthVirtualElements; azIdx++)
    {
        for (int32_t elIdx = 0; elIdx < snapshot.numElevationVirtualElements; elIdx++)
        {
            // Joint 2D window coefficient: Q30 * Q30 = Q60, shift to Q30
            int32_t jointWindow =
                my_round(((int64_t)azimuthWindow[azIdx] * (int64_t)elevationWindow[elIdx]), WINDOW_QBITS);

            // Apply windowing: Q30 * Q20 = Q50, shift to Q20
            int64_t windowedReal =
                my_round(((int64_t)snapshot.data[elIdx][azIdx].first * jointWindow) >> TRUNCATE_QBITS,
                         WINDOW_QBITS - TRUNCATE_QBITS);
            int64_t windowedImag =
                my_round(((int64_t)snapshot.data[elIdx][azIdx].second * jointWindow) >> TRUNCATE_QBITS,
                         WINDOW_QBITS - TRUNCATE_QBITS);

            // Copy windowed data to padded array (no conjugate)
            result.data[elIdx][2 * azIdx]     = static_cast<int32_t>(windowedReal); // real part
            result.data[elIdx][2 * azIdx + 1] = static_cast<int32_t>(windowedImag); // imaginary part
        }
    }

    return result;
}

// Perform 2D FFT using fixed-point radix-4/radix-2 algorithm
void FFT2D(FFTData &fftData)
{
    // FFT along azimuth dimension (rows)
    for (int32_t elIdx = 0; elIdx < fftData.paddedElevationSize; elIdx++)
    {
        // Extract row data in interleaved format for fixed-point FFT
        std::vector<int32_t> rowDataInterleaved(2 * fftData.paddedAzimuthSize, 0);
        for (int32_t azIdx = 0; azIdx < fftData.paddedAzimuthSize; azIdx++)
        {
            rowDataInterleaved[2 * azIdx]     = fftData.data[elIdx][2 * azIdx];     // real
            rowDataInterleaved[2 * azIdx + 1] = fftData.data[elIdx][2 * azIdx + 1]; // imaginary
        }

        // Perform fixed-point FFT on row
        if (is_log2n_even(fftData.paddedAzimuthSize))
        {
            fft_fixedp_r4_dif(fftData.paddedAzimuthSize, rowDataInterleaved.data());
        }
        else
        {
            fft_fixedp_r4_dif_odd(fftData.paddedAzimuthSize, rowDataInterleaved.data());
        }

        // Copy results back with range clamping to prevent overflow
        for (int32_t azIdx = 0; azIdx < fftData.paddedAzimuthSize; azIdx++)
        {
            fftData.data[elIdx][2 * azIdx] = std::clamp<int32_t>(rowDataInterleaved[2 * azIdx], INT32_MIN, INT32_MAX);
            fftData.data[elIdx][2 * azIdx + 1] =
                std::clamp<int32_t>(rowDataInterleaved[2 * azIdx + 1], INT32_MIN, INT32_MAX);
        }
    }

    // FFT along elevation dimension (columns)
    for (int32_t azIdx = 0; azIdx < fftData.paddedAzimuthSize; azIdx++)
    {
        // Extract column data in interleaved format for fixed-point FFT
        std::vector<int32_t> colDataInterleaved(2 * fftData.paddedElevationSize, 0);
        for (int32_t elIdx = 0; elIdx < fftData.paddedElevationSize; elIdx++)
        {
            colDataInterleaved[2 * elIdx]     = fftData.data[elIdx][2 * azIdx];     // real
            colDataInterleaved[2 * elIdx + 1] = fftData.data[elIdx][2 * azIdx + 1]; // imaginary
        }

        // Perform fixed-point FFT on column
        if (is_log2n_even(fftData.paddedElevationSize))
        {
            fft_fixedp_r4_dif(fftData.paddedElevationSize, colDataInterleaved.data());
        }
        else
        {
            fft_fixedp_r4_dif_odd(fftData.paddedElevationSize, colDataInterleaved.data());
        }

        // Copy results back with range clamping to prevent overflow
        for (int32_t elIdx = 0; elIdx < fftData.paddedElevationSize; elIdx++)
        {
            fftData.data[elIdx][2 * azIdx] = std::clamp<int32_t>(colDataInterleaved[2 * elIdx], INT32_MIN, INT32_MAX);
            fftData.data[elIdx][2 * azIdx + 1] =
                std::clamp<int32_t>(colDataInterleaved[2 * elIdx + 1], INT32_MIN, INT32_MAX);
        }
    }
}

// Calculate Q8 power spectrum and detect peak location
PowerSpectrum PowerSpectrumPeakDetection(const FFTData &fftData)
{
    PowerSpectrum result;

    // Initialize power spectrum array (transposed to match MATLAB format: [elevation x azimuth])
    result.data.resize(fftData.paddedElevationSize, std::vector<float>(fftData.paddedAzimuthSize));

    for (int32_t azIdx = 0; azIdx < fftData.paddedAzimuthSize; azIdx++)
    {
        for (int32_t elIdx = 0; elIdx < fftData.paddedElevationSize; elIdx++)
        {
            // Extract complex FFT result
            float realPart = static_cast<float>(fftData.data[elIdx][2 * azIdx]);
            float imagPart = static_cast<float>(fftData.data[elIdx][2 * azIdx + 1]);

            // Calculate power: magnitude^2 = real^2 + imag^2
            float power = realPart * realPart + imagPart * imagPart;

            result.data[elIdx][azIdx] = power;
        }
    }

    // Find peak in fixed-point power spectrum (direct search in unshifted coordinates)
    result.maxPower           = 0;
    result.peakAzimuthIndex   = 0;
    result.peakElevationIndex = 0;

    for (int32_t elIdx = 0; elIdx < fftData.paddedElevationSize; elIdx++)
    {
        for (int32_t azIdx = 0; azIdx < fftData.paddedAzimuthSize; azIdx++)
        {
            if (result.data[elIdx][azIdx] > result.maxPower)
            {
                result.maxPower           = result.data[elIdx][azIdx];
                result.peakAzimuthIndex   = azIdx;
                result.peakElevationIndex = elIdx;
            }
        }
    }
    return result;
}

// Convert interpolated indices to physical angles using fixed-point arithmetic
std::pair<int32_t, int32_t> ConvertToAngles(const std::pair<int32_t, int32_t> &interpIndices,
                                            const std::pair<std::vector<int32_t>, std::vector<int32_t>> &angleBins,
                                            int32_t paddedAzimuthSize, int32_t paddedElevationSize)
{
    // Extract azimuth and elevation bins from the pair
    const std::vector<int32_t> &azimuthBins   = angleBins.first;
    const std::vector<int32_t> &elevationBins = angleBins.second;

    // interpIndices are in Q16 format, azimuthBins/elevationBins are in Q16 format
    int32_t interpAzimuthIndex   = interpIndices.first;  // Q16 format
    int32_t interpElevationIndex = interpIndices.second; // Q16 format

    int32_t peakAzimuth   = 0; // Q16 format
    int32_t peakElevation = 0; // Q16 format

    // Linear interpolation for azimuth in fixed-point
    int32_t azLowerIdx = interpAzimuthIndex >> ANGLE_QBITS; // Extract integer part
    int32_t azUpperIdx = azLowerIdx + 1;
    azLowerIdx         = std::max(0, std::min(paddedAzimuthSize - 1, azLowerIdx));
    azUpperIdx         = std::max(0, std::min(paddedAzimuthSize - 1, azUpperIdx));

    if (azLowerIdx != azUpperIdx)
    {
        // Extract fractional part (Q16 format)
        int32_t fraction_q16 = interpAzimuthIndex - (azLowerIdx << ANGLE_QBITS);

        // Linear interpolation: lower + fraction * (upper - lower)
        // Q16 * Q16 = Q32, shift right by 16 to get Q16
        int64_t diff         = (int64_t)(azimuthBins[azUpperIdx] - azimuthBins[azLowerIdx]);
        int64_t interpolated = azimuthBins[azLowerIdx] + (my_round(diff * fraction_q16, ANGLE_QBITS));
        peakAzimuth          = static_cast<int32_t>(interpolated);
    }
    else
    {
        peakAzimuth = azimuthBins[azLowerIdx];
    }

    // Linear interpolation for elevation in fixed-point
    int32_t elLowerIdx = interpElevationIndex >> ANGLE_QBITS; // Extract integer part
    int32_t elUpperIdx = elLowerIdx + 1;
    elLowerIdx         = std::max(0, std::min(paddedElevationSize - 1, elLowerIdx));
    elUpperIdx         = std::max(0, std::min(paddedElevationSize - 1, elUpperIdx));

    if (elLowerIdx != elUpperIdx)
    {
        // Extract fractional part (Q16 format)
        int32_t fraction_q16 = interpElevationIndex - (elLowerIdx << ANGLE_QBITS);

        // Linear interpolation: lower + fraction * (upper - lower)
        // Q16 * Q16 = Q32, shift right by 16 to get Q16
        int64_t diff         = (int64_t)(elevationBins[elUpperIdx] - elevationBins[elLowerIdx]);
        int64_t interpolated = elevationBins[elLowerIdx] + (my_round(diff * fraction_q16, ANGLE_QBITS));
        peakElevation        = static_cast<int32_t>(interpolated);
    }
    else
    {
        peakElevation = elevationBins[elLowerIdx];
    }

    return std::make_pair(peakAzimuth, peakElevation);
}

// Fixed-point DOA estimation using 2D FFT with modular processing stages
DoaResult DoaEstimationFixedPoint(int32_t *snapshotsPtr, int32_t numTargets, int32_t numTx, int32_t numRx,
                                  int32_t numVirtualAzimuthElements, int32_t numVirtualElevationElements,
                                  PVADoaAngleFFTPrecision angleEstimationPrecision)
{
    DoaResult result;

    // Calculate dimensions for 2D FFT processing (enum value is the scaling factor directly)
    int32_t paddedAzimuthSize   = numVirtualAzimuthElements * precisionToScalingFactor(angleEstimationPrecision);
    int32_t paddedElevationSize = numVirtualElevationElements * precisionToScalingFactor(angleEstimationPrecision);

    // Pre-compute Q16 fixed-point angle bins (shared across all targets for efficiency)
    auto angleBins = InitializeAngleBins(paddedAzimuthSize, paddedElevationSize);

    // Process each target through the fixed-point pipeline - resize result vectors
    result.azimuthDegrees.resize(numTargets);
    result.elevationDegrees.resize(numTargets);
    result.peakPowerDb.resize(numTargets);

    for (int32_t targetIndex = 0; targetIndex < numTargets; targetIndex++)
    {
        Snapshot snapshot;
        snapshot.numAzimuthVirtualElements   = numVirtualAzimuthElements;
        snapshot.numElevationVirtualElements = numVirtualElevationElements;
        snapshot.data.resize(numVirtualElevationElements,
                             std::vector<std::pair<int32_t, int32_t>>(numVirtualAzimuthElements));
        for (int32_t elIdx = 0; elIdx < numVirtualElevationElements; elIdx++)
        {
            for (int32_t azIdx = 0; azIdx < numVirtualAzimuthElements; azIdx++)
            {
                int32_t virtualAntennaIdx = elIdx * numVirtualAzimuthElements + azIdx;
                int32_t txIdx             = virtualAntennaIdx % numTx;
                int32_t rxIdx             = virtualAntennaIdx / numTx;

                snapshot.data[elIdx][azIdx] =
                    std::make_pair(snapshotsPtr[2 * (targetIndex * numTx * numRx + txIdx * numRx + rxIdx)],
                                   snapshotsPtr[2 * (targetIndex * numTx * numRx + txIdx * numRx + rxIdx) + 1]);
            }
        }

        FFTData fftData = Windowing2D(snapshot, paddedAzimuthSize, paddedElevationSize);

        FFT2D(fftData);

        PowerSpectrum powerSpec = PowerSpectrumPeakDetection(fftData);

        std::pair<int32_t, int32_t> interpIndices = QuadraticInterpolation(powerSpec);

        std::pair<int32_t, int32_t> anglesFixed =
            ConvertToAngles(interpIndices, angleBins, paddedAzimuthSize, paddedElevationSize);

        // Convert fixed-point results to float for output
        // maxPowerFloat = real² + imag² where real/imag are Q12.20
        // To normalize: divide by 2^40 (2^20 squared for power)
        // dB = 10 * log10(power / 2^40)
        const float INV_POWER_SCALE =
            1.0f / (static_cast<float>(1LL << POWER_QBITS) * static_cast<float>(1LL << POWER_QBITS));
        const float INV_ANGLE_SCALE = 1.0f / static_cast<float>(ANGLE_SCALE);

        float maxPower = static_cast<float>(powerSpec.maxPower) * INV_POWER_SCALE;

        //compute peak power in dB: 10.0f * log10(maxPower + epsilon)
        //Due to lack of log10(x) function on PVA, we use log10(x) = log2(x) / log2(10.0f)
        float peakPowerDb =
            10.0f * std::log2(maxPower + std::numeric_limits<float>::epsilon()) * (1 / std::log2(10.0f));
        float peakAzimuthDeg   = static_cast<float>(anglesFixed.first) * INV_ANGLE_SCALE;
        float peakElevationDeg = static_cast<float>(anglesFixed.second) * INV_ANGLE_SCALE;

        // Store results
        result.azimuthDegrees[targetIndex]   = peakAzimuthDeg;
        result.elevationDegrees[targetIndex] = peakElevationDeg;
        result.peakPowerDb[targetIndex]      = peakPowerDb;
    }

    return result;
}

// ============================================================================
// Public API Implementation using NVCVTensor
// ============================================================================

namespace {

int32_t checkTargetListOutputImpl(int32_t pvaNumTargets, int32_t refNumTargets, int32_t numProperties, int32_t stride,
                                  float *pvaTargetListPtr, float *refTargetListPtr)
{
    float maxDiffFloat = 0.3f; // Tolerance for physical unit comparison
    bool debug         = false;

    if (pvaNumTargets != refNumTargets)
    {
        printf("Number of targets mismatch: pva = %d, ref = %d\n", pvaNumTargets, refNumTargets);
        return -1;
    }

    if (debug == true)
    {
        printf("Properties: Velocity, Range, Azimuth, Elevation, X, Y, Z, Power(optional)\n");
        printf("-------------------------------------------------------------------------\n");
        // Dump both ref and PVA values for all properties, for each target, in 2 rows
        for (int32_t i = 0; i < refNumTargets; i++)
        {
            printf("Target %d:\n", i);
            printf("  REF: ");
            for (int32_t j = 0; j < numProperties; j++)
            {
                printf("%.6f ", refTargetListPtr[j * stride + i]);
            }
            printf("\n  PVA: ");
            for (int32_t j = 0; j < numProperties; j++)
            {
                printf("%.6f ", pvaTargetListPtr[j * stride + i]);
            }
            printf("\n");
        }
    }

    // Compute error statistics
    float maxError  = 0.0f;
    float sumError  = 0.0f;
    int maxErrorIdx = -1, maxErrorProp = -1;
    int numMismatches = 0;

    for (int32_t i = 0; i < refNumTargets; i++)
    {
        for (int32_t j = 0; j < numProperties; j++)
        {
            float error = std::abs(pvaTargetListPtr[j * stride + i] - refTargetListPtr[j * stride + i]);
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

    float meanError = sumError / (refNumTargets * numProperties);
    (void)meanError; // Suppress unused variable warning

    if (maxError > maxDiffFloat)
    {
        printf("Target list mismatch at index %d, property %d: ref = %.6f, pva = %.6f\n", maxErrorIdx, maxErrorProp,
               refTargetListPtr[maxErrorProp * stride + maxErrorIdx],
               pvaTargetListPtr[maxErrorProp * stride + maxErrorIdx]);
        return -1;
    }
    return 0;
}

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
void generateComplexInputImpl(NVCVTensorHandle tensorHandle)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);
    T *tensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&tensorPtr, (void *)tensorData.buffer.strided.basePtr);

    int64_t tensorShapeNCHW[4];
    nvcvTensorShapePermute(tensorData.layout, tensorData.shape, NVCV_TENSOR_NCHW, tensorShapeNCHW);

    const int32_t channels = tensorShapeNCHW[1];
    const int32_t height   = tensorShapeNCHW[2];
    const int32_t width    = tensorShapeNCHW[3];

    // Generate random complex data in Q20 format
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<T> dist(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

    for (int32_t c = 0; c < channels; c++)
    {
        for (int32_t h = 0; h < height; h++)
        {
            for (int32_t w = 0; w < width; w++)
            {
                tensorPtr[2 * ((c * height + h) * width + w)]     = dist(gen) / 2; // Real part
                tensorPtr[2 * ((c * height + h) * width + w) + 1] = dist(gen) / 2; // Imaginary part
            }
        }
    }
}

void setDetectionCountImpl(NVCVTensorHandle inDetectionCountTensorHandle, int32_t detectionCount)
{
    NVCVTensorData inDetectionCountTensorData;
    nvcvTensorExportData(inDetectionCountTensorHandle, &inDetectionCountTensorData);
    int32_t *inDetectionCountTensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&inDetectionCountTensorPtr,
                           (void *)inDetectionCountTensorData.buffer.strided.basePtr);

    inDetectionCountTensorPtr[0] = detectionCount;
}

} // namespace

void generateDetectionListInput(NVCVTensorHandle detectionListTensorHandle)
{
    NVCVTensorData detData;
    nvcvTensorExportData(detectionListTensorHandle, &detData);

    if (detData.dtype == NVCV_DATA_TYPE_S32)
        generateDetectionListImpl<int32_t>(detectionListTensorHandle);
}

void generateDoaAngleFFTInput(NVCVTensorHandle snapshotsTensorHandle, NVCVTensorHandle inDetectionCountTensorHandle,
                              int32_t detectionCount)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(snapshotsTensorHandle, &tensorData);

    NVCVTensorData inDetectionCountTensorData;
    nvcvTensorExportData(inDetectionCountTensorHandle, &inDetectionCountTensorData);

    if (tensorData.dtype == NVCV_DATA_TYPE_2S32 && inDetectionCountTensorData.dtype == NVCV_DATA_TYPE_S32)
    {
        generateComplexInputImpl<int32_t>(snapshotsTensorHandle);
        setDetectionCountImpl(inDetectionCountTensorHandle, detectionCount);
    }
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input data type");
    }
}

void doaAngleFFTRef(NVCVTensorHandle detectionCountTensorHandle, NVCVTensorHandle snapshotsTensorHandle,
                    NVCVTensorHandle targetCountTensorHandle, NVCVTensorHandle targetAnglesTensorHandle,
                    NVCVTensorHandle targetIndexMapTensorHandle, const PVADoaAngleFFTParams *params)
{
    // Extract numDetections
    NVCVTensorData numDetectionsData;
    nvcvTensorExportData(detectionCountTensorHandle, &numDetectionsData);
    int32_t *detectionCountPtr = NULL;
    CupvaMemGetHostPointer((void **)&detectionCountPtr, (void *)numDetectionsData.buffer.strided.basePtr);

    const int32_t numDetections = detectionCountPtr[0];

    // Extract snapshots data
    NVCVTensorData snapshotsData;
    nvcvTensorExportData(snapshotsTensorHandle, &snapshotsData);
    int32_t *snapshotsPtr = NULL;
    CupvaMemGetHostPointer((void **)&snapshotsPtr, (void *)snapshotsData.buffer.strided.basePtr);

    int64_t snapshotsShapeCHW[3];
    nvcvTensorShapePermute(snapshotsData.layout, snapshotsData.shape, NVCV_TENSOR_CHW, snapshotsShapeCHW);
    const int32_t numTx = snapshotsShapeCHW[1];
    const int32_t numRx = snapshotsShapeCHW[2];

    // Run DOA estimation
    DoaResult doaResult =
        DoaEstimationFixedPoint(snapshotsPtr, numDetections, numTx, numRx, params->numVirtualAzimuthElements,
                                params->numVirtualElevationElements, params->angleEstimationPrecision);

    // Write target count output
    NVCVTensorData targetCountData;
    nvcvTensorExportData(targetCountTensorHandle, &targetCountData);
    int32_t *targetCountPtr = NULL;
    CupvaMemGetHostPointer((void **)&targetCountPtr, (void *)targetCountData.buffer.strided.basePtr);
    targetCountPtr[0] = numDetections;

    // Write target angles [3, numTargets] - azimuth, elevation, power
    NVCVTensorData targetAnglesData;
    nvcvTensorExportData(targetAnglesTensorHandle, &targetAnglesData);
    float *targetAnglesPtr = NULL;
    CupvaMemGetHostPointer((void **)&targetAnglesPtr, (void *)targetAnglesData.buffer.strided.basePtr);

    int64_t anglesShapeHW[2];
    nvcvTensorShapePermute(targetAnglesData.layout, targetAnglesData.shape, NVCV_TENSOR_HW, anglesShapeHW);
    const int32_t anglesStride = anglesShapeHW[1]; // W = numTargets (stride)

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
