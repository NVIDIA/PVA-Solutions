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

#include "RadarRefUtils.hpp"

#include "RadarDefinitions.h"

#include <cmath>
#include <cstdint>

namespace RadarRefUtils {
// Mixed-radix permutation for combined Radix-5, Radix-3, and Radix-2 stages
int32_t mixed_radix_reverse(int32_t i, int32_t n, const FFTFactorization &factorization)
{
    // Mixed-radix digit reversal for DIF FFT
    // Generic algorithm that works for any combination of radix-4, radix-2, radix-3, and radix-5
    //
    // For N = R0 × R1 × R2 × ... × Rk:
    // Standard index: i = d0 + d1*R0 + d2*(R0*R1) + ...
    // Reversed index: j = dk + dk-1*Rk + ... + d0*(Rk*...*R1)
    //
    // Example for 400 = 4 × 4 × 5 × 5:
    //   Input:  i = d0 + d1*4  + d2*16  + d3*80    (radices: 4, 4, 5, 5)
    //   Output: j = d3 + d2*5  + d1*25  + d0*100   (reversed radices: 5, 5, 4, 4)
    //
    // Example for 192 = 4 × 4 × 4 × 3:
    //   Input:  i = d0 + d1*4  + d2*16  + d3*64    (radices: 4, 4, 4, 3)
    //   Output: j = d3 + d2*3  + d1*12  + d0*48    (reversed radices: 3, 4, 4, 4)

    // Generic algorithm for mixed radix-4/radix-2/radix-3/radix-5 factorizations
    int32_t digits[10];  // Store extracted digits
    int32_t radices[10]; // Store radix for each stage
    int32_t num_stages = 0;
    int32_t idx        = i;

    // Extract radix-5 digits
    for (int32_t stage = 0; stage < factorization.groups[RADIX5_GROUP_IDX].numStages; stage++)
    {
        radices[num_stages] = 5;
        digits[num_stages]  = idx % 5;
        idx /= 5;
        num_stages++;
    }

    // Extract radix-3 digits
    for (int32_t stage = 0; stage < factorization.groups[RADIX3_GROUP_IDX].numStages; stage++)
    {
        radices[num_stages] = 3;
        digits[num_stages]  = idx % 3;
        idx /= 3;
        num_stages++;
    }

    // Step 1: Extract digits in forward order (LSB to MSB)
    // Extract radix-4 digits
    for (int32_t stage = 0; stage < factorization.groups[RADIX4_GROUP_IDX].numStages; stage++)
    {
        radices[num_stages] = 4;
        digits[num_stages]  = idx % 4;
        idx /= 4;
        num_stages++;
    }

    // Extract radix-2 digit if odd number of radix-2 bits
    if (factorization.groups[RADIX4_GROUP_IDX].isOdd)
    {
        radices[num_stages] = 2;
        digits[num_stages]  = idx % 2;
        idx /= 2;
        num_stages++;
    }

    // Step 2: Reconstruct in reversed order (MSB to LSB)
    // Result = d[n-1] + d[n-2]*R[n-1] + d[n-3]*R[n-1]*R[n-2] + ...
    int32_t result = 0;
    int32_t stride = 1;
    for (int32_t stage = num_stages - 1; stage >= 0; stage--)
    {
        result += digits[stage] * stride;
        stride *= radices[stage];
    }

    return result;
}

// Radix-2 DIF FFT stage
void fft_fixedp_r2_dif_stage(int32_t n, int32_t k, int32_t groupsz, int32_t *x, int32_t *twiddle, int32_t *y)
{
    int32_t ngroups        = 1UL << k;
    int32_t groupsz_radix2 = groupsz * ngroups;
    int32_t ngroups_radix2 = n / groupsz_radix2;
    int32_t dist           = ngroups_radix2 / 2;
    int32_t qbits          = 1 + (TWIDDLE_QBITS - TRUNCATE_QBITS);

    for (int32_t g = 0; g < groupsz_radix2; g++)
    {
        for (int32_t j = 0; j < dist; j++)
        {
            int32_t didx = g * ngroups_radix2 + j;
            int32_t tidx = j * groupsz_radix2;

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

// Single stage of Radix-4 DIF FFT
void fft_fixedp_r4_dif_stage(int32_t n, int32_t k, int32_t groupsz, int32_t *x_interleaved,
                             int32_t *twiddle_interleaved, int32_t *y)
{
    int32_t ngroups        = 1UL << (2 * k);
    int32_t groupsz_radix4 = groupsz * ngroups;
    int32_t ngroups_radix4 = n / groupsz_radix4;
    int32_t dist           = ngroups_radix4 / 4;
    int32_t qbits          = 2 + (TWIDDLE_QBITS - TRUNCATE_QBITS);
    int32_t stride         = n / 4;

    for (int32_t i = 0; i < groupsz_radix4; i++)
    {
        for (int32_t j = 0; j < dist; j++)
        {
            int32_t didx = i * ngroups_radix4 + j;

            // Twiddle indices
            int32_t t1idx = j * (groupsz_radix4);
            int32_t t2idx = 2 * t1idx;
            int32_t t3idx = 3 * t1idx;

            // Read input values
            int64_t x0r = x_interleaved[2 * didx];
            int64_t x0i = x_interleaved[2 * didx + 1];
            int64_t x1r = x_interleaved[2 * (didx + dist)];
            int64_t x1i = x_interleaved[2 * (didx + dist) + 1];
            int64_t x2r = x_interleaved[2 * (didx + 2 * dist)];
            int64_t x2i = x_interleaved[2 * (didx + 2 * dist) + 1];
            int64_t x3r = x_interleaved[2 * (didx + 3 * dist)];
            int64_t x3i = x_interleaved[2 * (didx + 3 * dist) + 1];

            // Read twiddle factors
            int64_t t1r = twiddle_interleaved[2 * t1idx];
            int64_t t1i = twiddle_interleaved[2 * t1idx + 1];
            int64_t t2r = twiddle_interleaved[2 * t2idx];
            int64_t t2i = twiddle_interleaved[2 * t2idx + 1];
            int64_t t3r = twiddle_interleaved[2 * t3idx];
            int64_t t3i = twiddle_interleaved[2 * t3idx + 1];

            // Butterfly computation
            int64_t z0r = x0r + x2r;
            int64_t z0i = x0i + x2i;
            int64_t z1r = x0r - x2r;
            int64_t z1i = x0i - x2i;
            int64_t z2r = x1r + x3r;
            int64_t z2i = x1i + x3i;
            int64_t z3r = x1r - x3r;
            int64_t z3i = x1i - x3i;

            int64_t w0r = z0r + z2r;
            int64_t w0i = z0i + z2i;
            int64_t w2r = z0r - z2r;
            int64_t w2i = z0i - z2i;
            int64_t w1r = z1r + z3i;
            int64_t w1i = z1i - z3r;
            int64_t w3r = z1r - z3i;
            int64_t w3i = z1i + z3r;

            // Apply twiddle factors and write output
            y[2 * didx]     = my_round(w0r, 2);
            y[2 * didx + 1] = my_round(w0i, 2);

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

void fft_fixedp_r3_dif_stage(int32_t n, int32_t s, int32_t groupsz, int32_t *x_interleaved,
                             int32_t *twiddle_interleaved, int32_t *y)
{
    int32_t ngroups        = pow(3, s);
    int32_t groupsz_radix3 = groupsz * ngroups;
    int32_t ngroups_radix3 = n / groupsz_radix3;
    int32_t dist           = ngroups_radix3 / 3;
    int32_t qbits          = 2 + (TWIDDLE_QBITS - TRUNCATE_QBITS);
    int32_t stride         = n / 3;

    for (int32_t g = 0; g < groupsz_radix3; g++)
    {
        for (int32_t j = 0; j < dist; j++)
        {
            int32_t didx = g * ngroups_radix3 + j;

            int32_t x0r = x_interleaved[2 * didx];
            int32_t x0i = x_interleaved[2 * didx + 1];
            int32_t x1r = x_interleaved[2 * (didx + dist)];
            int32_t x1i = x_interleaved[2 * (didx + dist) + 1];
            int32_t x2r = x_interleaved[2 * (didx + 2 * dist)];
            int32_t x2i = x_interleaved[2 * (didx + 2 * dist) + 1];

            // Radix-3 butterfly twiddle factors: W_3^k = W_N^(k * N/3) for k=1,2
            // For N=192: W_3^1 = W_192^64, W_3^2 = W_192^128
            int32_t t1r = twiddle_interleaved[2 * (1 * stride)];
            int32_t t1i = twiddle_interleaved[2 * (1 * stride) + 1];
            int32_t t2r = twiddle_interleaved[2 * (2 * stride)];
            int32_t t2i = twiddle_interleaved[2 * (2 * stride) + 1];

            complex x0c         = {x0r, x0i};
            complex x1c         = {x1r, x1i};
            complex x2c         = {x2r, x2i};
            complex x0c_shifted = cshift(x0c, -(TWIDDLE_QBITS - TRUNCATE_QBITS));

            complex tw1c = {t1r, t1i};
            complex tw2c = {t2r, t2i};

            // Radix-3 DIF butterfly computation
            // z0 = x0 + x1 + x2
            // z1 = x0 + x1*W_3^1 + x2*W_3^2
            // z2 = x0 + x1*W_3^2 + x2*W_3^1
            complex z0c = cadd(x0c, x1c);
            z0c         = cadd(z0c, x2c);

            complex tmpc1 = cmul(x1c, tw1c);
            complex tmpc2 = cmul(x2c, tw2c);

            complex z1c = cadd(x0c_shifted, tmpc1);
            z1c         = cadd(z1c, tmpc2);

            tmpc1 = cmul(x1c, tw2c);
            tmpc2 = cmul(x2c, tw1c);

            complex z2c = cadd(x0c_shifted, tmpc1);
            z2c         = cadd(z2c, tmpc2);

            z1c = cshift(z1c, (TWIDDLE_QBITS - TRUNCATE_QBITS));
            z2c = cshift(z2c, (TWIDDLE_QBITS - TRUNCATE_QBITS));

            int32_t div_coeff = 21845;
            z0c               = cmul_scalar(z0c, div_coeff);
            z1c               = cmul_scalar(z1c, div_coeff);
            z2c               = cmul_scalar(z2c, div_coeff);

            // Intra-stage twiddle factors for radix-3 DIF
            // Twiddle for output k at position j within the butterfly: W_N^(j * k * intra_stride)
            // where intra_stride = groupsz_radix3 (remaining size after this radix-3 decomposition)
            int32_t intra_stride = groupsz_radix3;

            int32_t t0idx = 0; // (j * 0 * intra_stride) is always 0
            int32_t t1idx = (j * 1 * intra_stride);
            int32_t t2idx = (j * 2 * intra_stride);

            // Get intra-stage twiddle factors
            int32_t t0r1 = twiddle_interleaved[2 * t0idx];
            int32_t t0i1 = twiddle_interleaved[2 * t0idx + 1];
            int32_t t1r1 = twiddle_interleaved[2 * t1idx];
            int32_t t1i1 = twiddle_interleaved[2 * t1idx + 1];
            int32_t t2r1 = twiddle_interleaved[2 * t2idx];
            int32_t t2i1 = twiddle_interleaved[2 * t2idx + 1];

            complex tw0c1 = {t0r1, t0i1};
            complex tw1c1 = {t1r1, t1i1};
            complex tw2c1 = {t2r1, t2i1};

            z0c = cmul(z0c, tw0c1);
            z1c = cmul(z1c, tw1c1);
            z2c = cmul(z2c, tw2c1);

            // Write outputs and divide by 3
            // Fixed-point divide by 3: multiply by 21845 (≈ 65536/3) and shift right by 16
            y[2 * didx]                  = my_round(z0c.r, TWIDDLE_QBITS - TRUNCATE_QBITS);
            y[2 * didx + 1]              = my_round(z0c.i, TWIDDLE_QBITS - TRUNCATE_QBITS);
            y[2 * (didx + dist)]         = my_round(z1c.r, TWIDDLE_QBITS - TRUNCATE_QBITS);
            y[2 * (didx + dist) + 1]     = my_round(z1c.i, TWIDDLE_QBITS - TRUNCATE_QBITS);
            y[2 * (didx + 2 * dist)]     = my_round(z2c.r, TWIDDLE_QBITS - TRUNCATE_QBITS);
            y[2 * (didx + 2 * dist) + 1] = my_round(z2c.i, TWIDDLE_QBITS - TRUNCATE_QBITS);
        }
    }
}

void fft_fixedp_r5_dif_stage(int32_t n, int32_t s, int32_t groupsz, int32_t *x_interleaved,
                             int32_t *twiddle_interleaved, int32_t *y)
{
    int32_t ngroups        = pow(5, s);
    int32_t groupsz_radix5 = groupsz * ngroups;
    int32_t ngroups_radix5 = n / groupsz_radix5;
    int32_t dist           = ngroups_radix5 / 5;
    int32_t qbits          = 2 + (TWIDDLE_QBITS - TRUNCATE_QBITS);
    int32_t stride         = n / 5;

    for (int32_t g = 0; g < groupsz_radix5; g++)
    {
        for (int32_t j = 0; j < dist; j++)
        {
            int32_t didx = g * ngroups_radix5 + j;

            int32_t x0r = x_interleaved[2 * didx];
            int32_t x0i = x_interleaved[2 * didx + 1];
            int32_t x1r = x_interleaved[2 * (didx + dist)];
            int32_t x1i = x_interleaved[2 * (didx + dist) + 1];
            int32_t x2r = x_interleaved[2 * (didx + 2 * dist)];
            int32_t x2i = x_interleaved[2 * (didx + 2 * dist) + 1];
            int32_t x3r = x_interleaved[2 * (didx + 3 * dist)];
            int32_t x3i = x_interleaved[2 * (didx + 3 * dist) + 1];
            int32_t x4r = x_interleaved[2 * (didx + 4 * dist)];
            int32_t x4i = x_interleaved[2 * (didx + 4 * dist) + 1];

            // Radix-5 butterfly twiddle factors: W_5^k = W_N^(k * N/5) for k=1,2,3,4
            // For N=400: W_5^1 = W_400^80, W_5^2 = W_400^160, W_5^3 = W_400^240, W_5^4 = W_400^320
            int32_t t1r = twiddle_interleaved[2 * (1 * stride)];
            int32_t t1i = twiddle_interleaved[2 * (1 * stride) + 1];
            int32_t t2r = twiddle_interleaved[2 * (2 * stride)];
            int32_t t2i = twiddle_interleaved[2 * (2 * stride) + 1];
            int32_t t3r = twiddle_interleaved[2 * (3 * stride)];
            int32_t t3i = twiddle_interleaved[2 * (3 * stride) + 1];
            int32_t t4r = twiddle_interleaved[2 * (4 * stride)];
            int32_t t4i = twiddle_interleaved[2 * (4 * stride) + 1];

            complex x0c         = {x0r, x0i};
            complex x1c         = {x1r, x1i};
            complex x2c         = {x2r, x2i};
            complex x3c         = {x3r, x3i};
            complex x4c         = {x4r, x4i};
            complex x0c_shifted = cshift(x0c, -(TWIDDLE_QBITS - TRUNCATE_QBITS));

            complex tw1c = {t1r, t1i};
            complex tw2c = {t2r, t2i};
            complex tw3c = {t3r, t3i};
            complex tw4c = {t4r, t4i};

            // Butterfly computation
            complex z0c = cadd(x0c, x1c);
            z0c         = cadd(z0c, x2c);
            z0c         = cadd(z0c, x3c);
            z0c         = cadd(z0c, x4c);

            complex tmpc1 = cmul(x1c, tw1c);
            complex tmpc2 = cmul(x2c, tw2c);
            complex tmpc3 = cmul(x3c, tw3c);
            complex tmpc4 = cmul(x4c, tw4c);

            complex z1c = cadd(x0c_shifted, tmpc1);
            z1c         = cadd(z1c, tmpc2);
            z1c         = cadd(z1c, tmpc3);
            z1c         = cadd(z1c, tmpc4);

            tmpc1 = cmul(x1c, tw2c);
            tmpc2 = cmul(x2c, tw4c);
            tmpc3 = cmul(x3c, tw1c);
            tmpc4 = cmul(x4c, tw3c);

            complex z2c = cadd(x0c_shifted, tmpc1);
            z2c         = cadd(z2c, tmpc2);
            z2c         = cadd(z2c, tmpc3);
            z2c         = cadd(z2c, tmpc4);

            tmpc1 = cmul(x1c, tw3c);
            tmpc2 = cmul(x2c, tw1c);
            tmpc3 = cmul(x3c, tw4c);
            tmpc4 = cmul(x4c, tw2c);

            complex z3c = cadd(x0c_shifted, tmpc1);
            z3c         = cadd(z3c, tmpc2);
            z3c         = cadd(z3c, tmpc3);
            z3c         = cadd(z3c, tmpc4);

            tmpc1 = cmul(x1c, tw4c);
            tmpc2 = cmul(x2c, tw3c);
            tmpc3 = cmul(x3c, tw2c);
            tmpc4 = cmul(x4c, tw1c);

            complex z4c = cadd(x0c_shifted, tmpc1);
            z4c         = cadd(z4c, tmpc2);
            z4c         = cadd(z4c, tmpc3);
            z4c         = cadd(z4c, tmpc4);

            z1c = cshift(z1c, (TWIDDLE_QBITS - TRUNCATE_QBITS));
            z2c = cshift(z2c, (TWIDDLE_QBITS - TRUNCATE_QBITS));
            z3c = cshift(z3c, (TWIDDLE_QBITS - TRUNCATE_QBITS));
            z4c = cshift(z4c, (TWIDDLE_QBITS - TRUNCATE_QBITS));

            int32_t div_coeff = 13107;

            z0c = cmul_scalar(z0c, div_coeff);
            z1c = cmul_scalar(z1c, div_coeff);
            z2c = cmul_scalar(z2c, div_coeff);
            z3c = cmul_scalar(z3c, div_coeff);
            z4c = cmul_scalar(z4c, div_coeff);

            // Intra-stage twiddle factors for radix-5 DIF
            // Twiddle for output k at position j within the butterfly: W_N^(j * k * intra_stride)
            // where intra_stride = dist = groupsz_radix5 (remaining size after this radix-5 decomposition)
            // Note: Twiddle indices are independent of group g since each group processes the same pattern
            int32_t intra_stride = groupsz_radix5; // = groupsz_radix5

            int32_t t0idx = 0; // (j * 0 * intra_stride) is always 0
            int32_t t1idx = (j * 1 * intra_stride);
            int32_t t2idx = (j * 2 * intra_stride);
            int32_t t3idx = (j * 3 * intra_stride);
            int32_t t4idx = (j * 4 * intra_stride);

            // Scale twiddle factors by 4/5 to account for the division by 5
            int32_t t0r1 = twiddle_interleaved[2 * t0idx];
            int32_t t0i1 = twiddle_interleaved[2 * t0idx + 1];
            int32_t t1r1 = twiddle_interleaved[2 * t1idx];
            int32_t t1i1 = twiddle_interleaved[2 * t1idx + 1];
            int32_t t2r1 = twiddle_interleaved[2 * t2idx];
            int32_t t2i1 = twiddle_interleaved[2 * t2idx + 1];
            int32_t t3r1 = twiddle_interleaved[2 * t3idx];
            int32_t t3i1 = twiddle_interleaved[2 * t3idx + 1];
            int32_t t4r1 = twiddle_interleaved[2 * t4idx];
            int32_t t4i1 = twiddle_interleaved[2 * t4idx + 1];

            complex tw0c1 = {t0r1, t0i1};
            complex tw1c1 = {t1r1, t1i1};
            complex tw2c1 = {t2r1, t2i1};
            complex tw3c1 = {t3r1, t3i1};
            complex tw4c1 = {t4r1, t4i1};

            z0c = cmul(z0c, tw0c1);
            z1c = cmul(z1c, tw1c1);
            z2c = cmul(z2c, tw2c1);
            z3c = cmul(z3c, tw3c1);
            z4c = cmul(z4c, tw4c1);

            y[2 * didx]                  = my_round(z0c.r, TWIDDLE_QBITS - TRUNCATE_QBITS);
            y[2 * didx + 1]              = my_round(z0c.i, TWIDDLE_QBITS - TRUNCATE_QBITS);
            y[2 * (didx + dist)]         = my_round(z1c.r, TWIDDLE_QBITS - TRUNCATE_QBITS);
            y[2 * (didx + dist) + 1]     = my_round(z1c.i, TWIDDLE_QBITS - TRUNCATE_QBITS);
            y[2 * (didx + 2 * dist)]     = my_round(z2c.r, TWIDDLE_QBITS - TRUNCATE_QBITS);
            y[2 * (didx + 2 * dist) + 1] = my_round(z2c.i, TWIDDLE_QBITS - TRUNCATE_QBITS);
            y[2 * (didx + 3 * dist)]     = my_round(z3c.r, TWIDDLE_QBITS - TRUNCATE_QBITS);
            y[2 * (didx + 3 * dist) + 1] = my_round(z3c.i, TWIDDLE_QBITS - TRUNCATE_QBITS);
            y[2 * (didx + 4 * dist)]     = my_round(z4c.r, TWIDDLE_QBITS - TRUNCATE_QBITS);
            y[2 * (didx + 4 * dist) + 1] = my_round(z4c.i, TWIDDLE_QBITS - TRUNCATE_QBITS);
        }
    }
}
} // namespace RadarRefUtils