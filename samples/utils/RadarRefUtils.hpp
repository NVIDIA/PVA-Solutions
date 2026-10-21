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

#ifndef RADAR_REF_UTILS_HPP
#define RADAR_REF_UTILS_HPP

#include <RadarDefinitions.h>

#include <cmath>
#include <cstdint>

#define DATA_QBITS (20)

#define DATA_SCALE (1UL << DATA_QBITS)
#define WINDOW_SCALE (1UL << WINDOW_QBITS)
#define TWIDDLE_SCALE (1UL << TWIDDLE_QBITS)

namespace {
struct complex
{
    int64_t r;
    int64_t i;
};

// Helper function to determine the number of bits needed to represent n
inline int32_t bit_length(int32_t n)
{
    if (n <= 1)
    {
        return 0;
    }
    return (int32_t)ceil(log2(n));
}

bool is_log2n_even(int32_t n)
{
    int32_t log2n = bit_length(n);
    return ((log2n & 1) == 0);
}

bool is_power_of_2(int32_t n)
{
    return ((n > 0) && ((n & (n - 1)) == 0));
}

// Helper function for rounding with scaling
int64_t my_round(int64_t x, int32_t shift)
{
    int64_t half = 1UL << (shift - 1);
    return (x + half) >> shift;
}

inline complex cmul(complex v, complex tw)
{
    complex z;
    z.r = ((v.r * tw.r) >> TRUNCATE_QBITS) - ((v.i * tw.i) >> TRUNCATE_QBITS);
    z.i = ((v.r * tw.i) >> TRUNCATE_QBITS) + ((v.i * tw.r) >> TRUNCATE_QBITS);
    return z;
}

static inline complex cmul_scalar(complex v, int32_t scalar)
{
    complex z;
    z.r = (v.r * (int64_t)scalar) >> TRUNCATE_QBITS;
    z.i = (v.i * (int64_t)scalar) >> TRUNCATE_QBITS;
    return z;
}

inline complex cadd(complex &x, complex &y)
{
    complex z;
    z.r = x.r + y.r;
    z.i = x.i + y.i;
    return z;
}

inline complex csub(complex x, complex y)
{
    complex z;
    z.r = x.r - y.r;
    z.i = x.i - y.i;
    return z;
}

inline complex cshift(complex x, int32_t shift)
{
    complex z;
    if (shift > 0)
    {
        z.r = x.r >> shift;
        z.i = x.i >> shift;
    }
    else
    {
        z.r = x.r << (-shift);
        z.i = x.i << (-shift);
    }
    return z;
}

} // namespace

namespace RadarRefUtils {

/**
 * @brief Mixed-radix permutation for combined Radix-5, Radix-3, and Radix-2 stages
 * @param i The input index
 * @param n The FFT size
 * @param factorization The FFT factorization
 * @return The reversed index
 */
int32_t mixed_radix_reverse(int32_t i, int32_t n, const FFTFactorization &factorization);

/**
 * @brief Radix-2 DIF FFT stage
 * @param n The FFT size
 * @param k The stage index
 * @param groupsz The group size
 * @param x The input buffer
 * @param twiddle The twiddle buffer
 * @param y The output buffer
 */
void fft_fixedp_r2_dif_stage(int32_t n, int32_t k, int32_t groupsz, int32_t *x, int32_t *twiddle, int32_t *y);

/**
 * @brief Radix-4 DIF FFT stage
 * @param n The FFT size
 * @param k The stage index
 * @param groupsz The group size
 * @param x The input buffer
 * @param twiddle The twiddle buffer
 * @param y The output buffer
 */
void fft_fixedp_r4_dif_stage(int32_t n, int32_t k, int32_t groupsz, int32_t *x, int32_t *twiddle, int32_t *y);

/**
 * @brief Radix-3 DIF FFT stage
 * @param n The FFT size
 * @param k The stage index
 * @param groupsz The group size
 * @param x The input buffer
 * @param twiddle The twiddle buffer
 * @param y The output buffer
 */
void fft_fixedp_r3_dif_stage(int32_t n, int32_t k, int32_t groupsz, int32_t *x, int32_t *twiddle, int32_t *y);

/**
 * @brief Radix-5 DIF FFT stage
 * @param n The FFT size
 * @param k The stage index
 * @param groupsz The group size
 * @param x The input buffer
 * @param twiddle The twiddle buffer
 * @param y The output buffer
 */
void fft_fixedp_r5_dif_stage(int32_t n, int32_t k, int32_t groupsz, int32_t *x, int32_t *twiddle, int32_t *y);

} // namespace RadarRefUtils

#endif /* RADAR_REF_UTILS_HPP */