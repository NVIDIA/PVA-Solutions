/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/* Scalar reference for Part 2 (Phases 4-12), per ROI. */

#include "find_local_max_ref.h"

#include <PvaOperatorTypes.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

float bitsToFloat(int32_t bits)
{
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

int32_t floatToBits(float value)
{
    int32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

struct Entry
{
    uint16_t power;
    uint16_t az_int;
    uint16_t el_int;
};

bool entryPrecedes(const Entry &a, const Entry &b)
{
    if (a.power != b.power)
        return a.power > b.power;
    if (a.el_int != b.el_int)
        return a.el_int < b.el_int;
    return a.az_int < b.az_int;
}

inline float quadratic_interp_1d(uint16_t i, float y_minus, float y0, float y_plus)
{
    const float scale = 1.0F / (4.0F * (y0 - 0.5F * y_minus - 0.5F * y_plus));
    if (std::isinf(scale))
        return static_cast<float>(i);
    const float offset = (y_plus - y_minus) * scale;
    return y0 > y_minus && y0 > y_plus && std::fabs(offset) <= 0.5F ? static_cast<float>(i) + offset
                                                                    : static_cast<float>(i);
}

/* Match VPU round-half-to-even semantics for positive and negative values. */
inline int32_t round_half_to_even_f32(float x)
{
    const float fl   = std::floor(x);
    const float frac = x - fl;
    int32_t iflr     = (int32_t)fl;
    if (frac < 0.5f)
        return iflr;
    if (frac > 0.5f)
        return iflr + 1;
    return (iflr & 1) ? iflr + 1 : iflr;
}

/* Round an integer right shift to nearest, with ties to even. */
inline int32_t round_half_to_even_shift_right_s64(int64_t x, int shift)
{
    if (shift <= 0)
        return (int32_t)x;
    const int64_t bias    = (int64_t)1 << (shift - 1);
    const int64_t mask    = ((int64_t)1 << shift) - 1;
    const int64_t shifted = (x + bias) >> shift;
    /* Undo the tie bias when it produced an odd result. */
    if ((x & mask) == bias)
    {
        if (shifted & 1)
            return (int32_t)(shifted - 1);
    }
    return (int32_t)shifted;
}

inline int32_t saturate_s64_to_s32(int64_t v, int32_t lo, int32_t hi)
{
    if (v < (int64_t)lo)
        return lo;
    if (v > (int64_t)hi)
        return hi;
    return static_cast<int32_t>(v);
}

/* Quantize one packed L-inverse with a shared 2^28 scale. */
void scalar_obtain_fixed_pt_L_inv_per_roi(uint16_t sv_dim, const float *L_inv_real, const float *L_inv_imag,
                                          int32_t *L_inv_fixed, float *scale_out)
{
    const uint32_t tri_len = (uint32_t)sv_dim * ((uint32_t)sv_dim + 1) / 2;
    float max_abs          = 0.f;
    for (uint32_t i = 0; i < tri_len; ++i)
    {
        float ar = std::fabs(L_inv_real[i]);
        float ai = std::fabs(L_inv_imag[i]);
        if (ar > max_abs)
            max_abs = ar;
        if (ai > max_abs)
            max_abs = ai;
    }
    const float scale    = (max_abs > 0.f) ? (0x1p28f / max_abs) : 0x1p28f;
    *scale_out           = scale;
    const int32_t sat_hi = (int32_t)(1 << 28);
    const int32_t sat_lo = -sat_hi;
    for (uint32_t i = 0; i < tri_len; ++i)
    {
        int32_t re_q = round_half_to_even_f32(L_inv_real[i] * scale);
        int32_t im_q = round_half_to_even_f32(L_inv_imag[i] * scale);
        if (re_q > sat_hi)
            re_q = sat_hi;
        if (re_q < sat_lo)
            re_q = sat_lo;
        if (im_q > sat_hi)
            im_q = sat_hi;
        if (im_q < sat_lo)
            im_q = sat_lo;
        L_inv_fixed[2 * i]     = re_q;
        L_inv_fixed[2 * i + 1] = im_q;
    }
}

/* Match the VPU's three-block accumulation, including the rounded and
 * saturated intermediate between the lower-right and lower-left blocks. Each
 * product is truncated by 16 bits before accumulation. */
void scalar_compute_L_inv_times_steering_matrix_per_roi(uint16_t sv_dim, uint16_t sv_num, const int32_t *L_inv_fixed,
                                                        const int32_t *sv_q22, int32_t *L_inv_sv)
{
    const int32_t sat_hi = (int32_t)(1 << 30);
    const int32_t sat_lo = -sat_hi;
    const uint32_t H     = (uint32_t)sv_dim / 2;

    /* Negate before shifting to preserve the VPU's signed truncation. */
    auto cmadd_term = [](int64_t &acc_re, int64_t &acc_im, int32_t Lr, int32_t Li, int32_t sr, int32_t si) {
        const int64_t prod_LrSr     = (int64_t)Lr * sr;
        const int64_t prod_negLi_Si = -((int64_t)Li * si); /* C[0][1] = -Li, full neg before shift */
        const int64_t prod_LiSr     = (int64_t)Li * sr;
        const int64_t prod_LrSi     = (int64_t)Lr * si;
        acc_re += (prod_LrSr >> 16) + (prod_negLi_Si >> 16);
        acc_im += (prod_LiSr >> 16) + (prod_LrSi >> 16);
    };

    for (uint32_t m = 0; m < sv_num; ++m)
    {
        for (uint32_t i = 0; i < sv_dim; ++i)
        {
            int64_t acc_re = 0, acc_im = 0;

            if (i < H)
            {
                /* Upper-left block. */
                for (uint32_t k = 0; k <= i; ++k)
                {
                    const uint32_t tri = k * sv_dim - k * (k - 1) / 2 + (i - k);
                    cmadd_term(acc_re, acc_im, L_inv_fixed[2 * tri], L_inv_fixed[2 * tri + 1],
                               sv_q22[2 * (m * sv_dim + k)], sv_q22[2 * (m * sv_dim + k) + 1]);
                }
            }
            else
            {
                /* Lower-right block. */
                for (uint32_t k = H; k <= i; ++k)
                {
                    const uint32_t tri = k * sv_dim - k * (k - 1) / 2 + (i - k);
                    cmadd_term(acc_re, acc_im, L_inv_fixed[2 * tri], L_inv_fixed[2 * tri + 1],
                               sv_q22[2 * (m * sv_dim + k)], sv_q22[2 * (m * sv_dim + k) + 1]);
                }
                /* Preserve the intermediate VPU store and reload. */
                int32_t s1_re = round_half_to_even_shift_right_s64(acc_re, 16);
                int32_t s1_im = round_half_to_even_shift_right_s64(acc_im, 16);
                s1_re         = saturate_s64_to_s32((int64_t)s1_re, sat_lo, sat_hi);
                s1_im         = saturate_s64_to_s32((int64_t)s1_im, sat_lo, sat_hi);

                acc_re = (int64_t)s1_re << 16;
                acc_im = (int64_t)s1_im << 16;
                for (uint32_t k = 0; k < H; ++k)
                {
                    const uint32_t tri = k * sv_dim - k * (k - 1) / 2 + (i - k);
                    cmadd_term(acc_re, acc_im, L_inv_fixed[2 * tri], L_inv_fixed[2 * tri + 1],
                               sv_q22[2 * (m * sv_dim + k)], sv_q22[2 * (m * sv_dim + k) + 1]);
                }
            }

            int32_t re_out                     = round_half_to_even_shift_right_s64(acc_re, 16);
            int32_t im_out                     = round_half_to_even_shift_right_s64(acc_im, 16);
            re_out                             = saturate_s64_to_s32((int64_t)re_out, sat_lo, sat_hi);
            im_out                             = saturate_s64_to_s32((int64_t)im_out, sat_lo, sat_hi);
            L_inv_sv[2 * (m * sv_dim + i)]     = re_out;
            L_inv_sv[2 * (m * sv_dim + i) + 1] = im_out;
        }
    }
}

/* Compute the reciprocal-spectrum denominator for one ROI. */
void scalar_compute_fp_rcp_spectrum_per_roi(uint16_t sv_dim, uint16_t sv_num, const int32_t *L_inv_sv,
                                            float *rcp_spectrum)
{
    for (uint32_t m = 0; m < sv_num; ++m)
    {
        float sum = 0.f;
        for (uint32_t i = 0; i < sv_dim; ++i)
        {
            float re = (float)L_inv_sv[2 * (m * sv_dim + i)];
            float im = (float)L_inv_sv[2 * (m * sv_dim + i) + 1];
            sum += re * re + im * im;
        }
        rcp_spectrum[m] = sum;
    }
}

/* Convert one reciprocal-spectrum denominator to fixed- and floating-point spectra. */
void scalar_obtain_spectrum_per_roi(uint16_t sv_num, const float *rcp_spectrum, uint16_t *fixed_pt_out,
                                    float *float_pt_out)
{
    assert(sv_num);

    float scale = bitsToFloat(0x7F800000);
    for (uint32_t s = 0; s < sv_num; ++s)
    {
        const float t = rcp_spectrum[s];
        assert(t >= 0.f);
        if (scale > t)
            scale = t;
    }

    scale *= 0x1p15f;
    scale = bitsToFloat(std::min(std::max(floatToBits(scale), floatToBits(1.0F / 0x1p113F)), 0x7F7FFFFF));

    for (uint32_t s = 0; s < sv_num; ++s)
    {
        const float spectrum = bitsToFloat(std::min(std::max(floatToBits(1.0F / rcp_spectrum[s]), 0), 0x7F7FFFFF));
        fixed_pt_out[s]      = static_cast<uint16_t>(std::min(1 << 15, round_half_to_even_f32(scale * spectrum)));
        float_pt_out[s]      = spectrum;
    }
}

/* Estimate one ROI's noise floor from the lower median. */
void scalar_estimate_noise_floor_per_roi(uint16_t sv_num, const uint16_t *fixed_pt_spectrum, uint16_t *noise_floor)
{
    assert(sv_num);

    std::vector<uint16_t> values(fixed_pt_spectrum, fixed_pt_spectrum + sv_num);
    std::sort(values.begin(), values.end());
    *noise_floor = static_cast<uint16_t>(values[sv_num / 2 - 1] & 0xFFC0);
}

/* Apply the 3x3 filter to one fixed 8x16 spectrum. */
void scalar_extract_3x3filtered_spectrum_in16az8el_out14az6el_per_roi(const uint16_t *fixed_pt_spectrum,
                                                                      uint16_t *local_max, uint16_t *local_sum_d8)
{
    typedef const uint16_t TI[8][16];
    typedef uint16_t TO[6][14];
    TI &P = *(TI *)fixed_pt_spectrum;
    TO &M = *(TO *)local_max;
    TO &S = *(TO *)local_sum_d8;

    for (uint32_t e = 0; e < 6; ++e)
        for (uint32_t a = 0; a < 14; ++a)
        {
            const uint16_t d[3][3] = {
                {P[e + 0][a + 0], P[e + 0][a + 1], P[e + 0][a + 2]},
                {P[e + 1][a + 0], P[e + 1][a + 1], P[e + 1][a + 2]},
                {P[e + 2][a + 0], P[e + 2][a + 1], P[e + 2][a + 2]},
            };

            M[e][a] = std::max({d[0][0], d[0][1], d[0][2], d[1][0], d[1][1], d[1][2], d[2][0], d[2][1], d[2][2]});

            S[e][a] = (uint16_t)(((d[0][0] + d[1][0] + d[2][0] + 1) / 2 + (d[0][1] + d[1][1] + d[2][1] + 1) / 2 +
                                  (d[0][2] + d[1][2] + d[2][2] + 1) / 2 + 2) /
                                 4);
        }
}

/* Mark local peak candidates for one ROI. */
void scalar_mark_peak_candidates_per_roi(uint16_t num_az, uint16_t num_el, const uint16_t *fixed_pt_spectrum,
                                         const uint16_t *local_max, const uint16_t *local_sum_d8, int32_t k_nb,
                                         uint16_t *candidate)
{
    const uint32_t out_az = (uint32_t)(num_az - 2);
    const uint32_t out_el = (uint32_t)(num_el - 2);

    for (uint32_t e = 0; e < out_el; ++e)
        for (uint32_t a = 0; a < out_az; ++a)
        {
            const uint16_t cen = fixed_pt_spectrum[(e + 1) * num_az + (a + 1)];
            const uint32_t y   = e * out_az + a;
            candidate[y]       = local_max[y] != cen || cen <= ((local_sum_d8[y] * k_nb + 64) >> 7) ? (uint16_t)0 : cen;
        }
}

/* Select the top three peaks for one ROI. */
void scalar_get_top3_peaks_per_roi(uint16_t num_az, uint16_t num_el, float noise_floor_ratio, const uint16_t *candidate,
                                   const float *float_pt_spectrum, float scale, uint16_t noise_floor,
                                   bool enable_sub_bin_interp, float *out_power, float *out_az, float *out_el)
{
    assert(num_az > 2);
    assert(num_el > 2);

    const uint16_t out_az_cnt = (uint16_t)(num_az - 2);
    const uint16_t out_el_cnt = (uint16_t)(num_el - 2);
    const uint32_t out_cnt    = (uint32_t)(out_az_cnt * out_el_cnt);
    assert(out_cnt >= 3);

    std::vector<Entry> entries(out_cnt);

    uint32_t n = 0;
    for (uint16_t e = 1; e < num_el - 1; ++e)
        for (uint16_t a = 1; a < num_az - 1; ++a)
        {
            entries[n].power  = candidate[(e - 1) * out_az_cnt + (a - 1)];
            entries[n].az_int = a;
            entries[n].el_int = e;
            ++n;
        }
    assert(n == out_cnt);

    std::sort(entries.begin(), entries.end(), entryPrecedes);

    const float p_abs_scale = scale * scale * (1.f / 0x1p20f);
    const float nf_th       = noise_floor_ratio * (float)noise_floor;

    for (uint32_t k = 0; k < 3; ++k)
    {
        const uint16_t az = entries[k].az_int;
        const uint16_t el = entries[k].el_int;

        const float *p    = float_pt_spectrum + (uint32_t)el * num_az + az;
        const float P_abs = p_abs_scale * p[0];

        out_power[k] = (entries[k].power <= nf_th || !(P_abs > 0.f)) ? PVA_DOA_CAPON_ROI_INVALID_PEAK_POWER : P_abs;
        /* Interpolation-off emits the raw integer argmax bin. */
        out_az[k] = enable_sub_bin_interp ? quadratic_interp_1d(az, p[-1], p[0], p[1]) : (float)az;
        out_el[k] = enable_sub_bin_interp ? quadratic_interp_1d(el, p[-(int)num_az], p[0], p[num_az]) : (float)el;
    }
}

} // namespace

void findLocalMaxRef(const float *inverseReal, const float *inverseImag, const int32_t *steeringVectors,
                     int32_t matrixSize, int32_t numAzimuthBins, int32_t numElevationBins, int32_t steeringQBits,
                     float noiseFloorRatio, bool enableSubBinInterpolation, float *peakPower, float *azimuthBins,
                     float *elevationBins)
{
    assert(numAzimuthBins == 16);
    assert(numElevationBins == 8);

    int32_t numSteeringVectors = numAzimuthBins * numElevationBins;
    uint32_t triangleSize      = (uint32_t)matrixSize * ((uint32_t)matrixSize + 1) / 2;

    // Phase 4: Quantize the packed inverse Cholesky factor and record its scale.
    std::vector<int32_t> fixedInverse(2 * triangleSize);
    float inverseScale = 0.0f;
    scalar_obtain_fixed_pt_L_inv_per_roi((uint16_t)matrixSize, inverseReal, inverseImag, fixedInverse.data(),
                                         &inverseScale);

    // Phase 5: Convert steering vectors to the VPU fixed-point layout and Q format.
    int32_t shiftBits    = 22 - steeringQBits;
    int64_t saturationHi = (int64_t)1 << 30;
    int64_t saturationLo = -saturationHi;
    std::vector<int32_t> fixedSteeringVectors(2 * (size_t)numSteeringVectors * (size_t)matrixSize);
    for (int32_t dimension = 0; dimension < matrixSize; ++dimension)
    {
        for (int32_t steering = 0; steering < numSteeringVectors; ++steering)
        {
            int32_t sourceIndex = dimension * numSteeringVectors * 2 + steering * 2;
            int64_t real        = (int64_t)steeringVectors[sourceIndex];
            int64_t imag        = (int64_t)steeringVectors[sourceIndex + 1];
            if (shiftBits >= 0)
            {
                real <<= shiftBits;
                imag <<= shiftBits;
            }
            else
            {
                real >>= -shiftBits;
                imag >>= -shiftBits;
            }
            real = std::min(std::max(real, saturationLo), saturationHi);
            imag = std::min(std::max(imag, saturationLo), saturationHi);

            int32_t destinationIndex                   = 2 * (steering * matrixSize + dimension);
            fixedSteeringVectors[destinationIndex]     = (int32_t)real;
            fixedSteeringVectors[destinationIndex + 1] = (int32_t)imag;
        }
    }

    // Phase 6: Multiply L_inv by every steering vector.
    std::vector<int32_t> inverseTimesSteering(2 * (size_t)numSteeringVectors * (size_t)matrixSize);
    scalar_compute_L_inv_times_steering_matrix_per_roi((uint16_t)matrixSize, (uint16_t)numSteeringVectors,
                                                       fixedInverse.data(), fixedSteeringVectors.data(),
                                                       inverseTimesSteering.data());

    // Phase 7: Form the reciprocal Capon spectrum from the complex products.
    std::vector<float> reciprocalSpectrum(numSteeringVectors);
    scalar_compute_fp_rcp_spectrum_per_roi((uint16_t)matrixSize, (uint16_t)numSteeringVectors,
                                           inverseTimesSteering.data(), reciprocalSpectrum.data());

    // Phase 8: Produce fixed-point and floating-point spectrum representations.
    std::vector<uint16_t> fixedSpectrum(numSteeringVectors);
    std::vector<float> floatSpectrum(numSteeringVectors);
    scalar_obtain_spectrum_per_roi((uint16_t)numSteeringVectors, reciprocalSpectrum.data(), fixedSpectrum.data(),
                                   floatSpectrum.data());

    // Phase 9: Estimate the noise floor from the fixed-point spectrum.
    uint16_t noiseFloor = 0;
    scalar_estimate_noise_floor_per_roi((uint16_t)numSteeringVectors, fixedSpectrum.data(), &noiseFloor);

    // Phase 10: Compute the 3x3 local maximum and neighborhood sum for each interior bin.
    int32_t filteredAzimuthBins   = numAzimuthBins - 2;
    int32_t filteredElevationBins = numElevationBins - 2;
    size_t filteredSize           = (size_t)filteredAzimuthBins * filteredElevationBins;
    std::vector<uint16_t> localMax(filteredSize);
    std::vector<uint16_t> localSumD8(filteredSize);
    scalar_extract_3x3filtered_spectrum_in16az8el_out14az6el_per_roi(fixedSpectrum.data(), localMax.data(),
                                                                     localSumD8.data());

    // Phase 11: Mark bins that satisfy the local-maximum and neighborhood tests.
    std::vector<uint16_t> candidates(filteredSize);
    scalar_mark_peak_candidates_per_roi((uint16_t)numAzimuthBins, (uint16_t)numElevationBins, fixedSpectrum.data(),
                                        localMax.data(), localSumD8.data(), 139, candidates.data());

    // Phase 12: Rank, threshold, and optionally interpolate the top three peaks.
    float nanValue = std::numeric_limits<float>::quiet_NaN();
    for (int32_t peak = 0; peak < 3; ++peak)
    {
        peakPower[peak]     = PVA_DOA_CAPON_ROI_INVALID_PEAK_POWER;
        azimuthBins[peak]   = nanValue;
        elevationBins[peak] = nanValue;
    }
    scalar_get_top3_peaks_per_roi((uint16_t)numAzimuthBins, (uint16_t)numElevationBins, noiseFloorRatio,
                                  candidates.data(), floatSpectrum.data(), inverseScale, noiseFloor,
                                  enableSubBinInterpolation, peakPower, azimuthBins, elevationBins);
}
