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

#include "fft_knl.h"

#include <cupva_device.h>

int32_t func(fft_batched_digit_reverse_transpose_init)(STYPE *src, STYPE *dst, AgenCFG *cfgs, int32_t &cfg_idx,
                                                       bool need_transpose, int32_t o_lane_ofst,
                                                       FFTFactorization *factorization, int32_t it, int32_t pitch,
                                                       int32_t groups, int32_t srcGroupStride, int32_t dstGroupStride,
                                                       int32_t out_round_override)
{
    const int32_t n   = factorization->n;
    int32_t out_round = (factorization->groups[RADIX4_GROUP_IDX].isOdd == 1) ? 1 : 2;
    const int32_t totalPowerOf2Stages =
        factorization->groups[RADIX4_GROUP_IDX].numStages + (int32_t)factorization->groups[RADIX4_GROUP_IDX].isOdd;
    out_round = (totalPowerOf2Stages == 0) ? 0 : out_round;
    /* A caller carrying its own fixed-point scaling downstream passes its shift here; -1 keeps the
     * growth-compensating default above. */
    out_round = (out_round_override >= 0) ? out_round_override : out_round;

    AgenWrapper wrapper;

    agen in      = init(src);
    wrapper.size = sizeof(STYPE);
    wrapper.n1   = n;
    wrapper.s1   = pitch;
    wrapper.n2   = groups;
    wrapper.s2   = srcGroupStride;
    INIT_AGEN2(in, wrapper);

    agen out  = init((VTYPE *)dst);
    out.round = out_round;
    if (need_transpose)
    {
        out.lane_ofst = o_lane_ofst;
    }

    wrapper.n1 = factorization->iters[0];
    wrapper.n2 = factorization->iters[1];
    wrapper.n3 = factorization->iters[2];
    wrapper.n4 = factorization->iters[3];
    wrapper.n5 = factorization->iters[4];
    wrapper.n6 = factorization->iters[5];
    wrapper.s1 = factorization->strides[0];
    wrapper.s2 = factorization->strides[1];
    wrapper.s3 = factorization->strides[2];
    wrapper.s4 = factorization->strides[3];
    wrapper.s5 = factorization->strides[4];
    wrapper.s6 = factorization->strides[5];
    // Put the group dimension in the first agen slot the FFT stages leave free (any of slots 1-5).
    if (groups > 1)
    {
        int32_t slot = 0;
        while ((slot < 6) && (factorization->iters[slot] != 0))
        {
            slot++;
        }
        switch (slot)
        {
        case 1:
            wrapper.n2 = groups;
            wrapper.s2 = dstGroupStride;
            break;
        case 2:
            wrapper.n3 = groups;
            wrapper.s3 = dstGroupStride;
            break;
        case 3:
            wrapper.n4 = groups;
            wrapper.s4 = dstGroupStride;
            break;
        case 4:
            wrapper.n5 = groups;
            wrapper.s5 = dstGroupStride;
            break;
        case 5:
            wrapper.n6 = groups;
            wrapper.s6 = dstGroupStride;
            break;
        default:
            break; // no free dimension: the host gate does not enable fusion here
        }
    }
    INIT_AGEN6(out, wrapper);

    cfgs[cfg_idx++] = extract_agen_cfg(in);
    cfgs[cfg_idx++] = extract_agen_cfg(out);

    it++;

    return it;
}

int32_t func(fft_batched_digit_reverse_power_init)(STYPE *src, STYPE *dst, AgenCFG *cfgs, int32_t &cfg_idx,
                                                   int32_t log2N, bool need_transpose, int32_t it, const int32_t niter1,
                                                   const int32_t round_qbits, int32_t in_width, int32_t in_pad,
                                                   int32_t out_width, int32_t out_pad, int32_t iter,
                                                   int32_t kernel_batch)
{
    const int32_t n           = 1 << log2N;
    const int32_t vecw        = chess_elementsof(DVTYPEX);
    const int32_t o_pad       = out_pad;
    const int32_t o_pitch     = need_transpose ? COMPLEX_ELEMENT_NUM : (COMPLEX_ELEMENT_NUM * out_width) + o_pad;
    const int32_t o_mag_pitch = need_transpose ? 1 : (out_width);
    const int32_t o_lane_ofst = (out_width * COMPLEX_ELEMENT_NUM + vecw - 1) / vecw;
    const int32_t in_pitch    = (COMPLEX_ELEMENT_NUM * in_width) + in_pad;
    const int32_t out_iter1_pitch =
        need_transpose ? ((vecw / COMPLEX_ELEMENT_NUM) * ((vecw * o_lane_ofst) + o_pad)) : vecw;
    const int32_t out_iter1_mag_pitch =
        need_transpose ? ((vecw / COMPLEX_ELEMENT_NUM) * ((vecw * o_lane_ofst) + o_pad)) : (vecw >> 1);

    const int32_t radix4Digits  = 4; // digits per radix-4 stage
    const int32_t knls_per_tile = iter;
    const int32_t row_iter      = kernel_batch / (vecw / COMPLEX_ELEMENT_NUM);

    AgenWrapper wrapper;

    agen in      = init(src);
    wrapper.size = sizeof(int32_t);
    wrapper.n1   = niter1;
    wrapper.n2   = row_iter;
    wrapper.n3   = n / niter1;
    wrapper.n4   = knls_per_tile;
    wrapper.s1   = in_pitch;
    wrapper.s2   = vecw;
    wrapper.s3   = niter1 * in_pitch;
    wrapper.s4   = row_iter * vecw;
    INIT_AGEN4(in, wrapper);

    agen out  = init(dst);
    out.round = round_qbits;
    if (need_transpose)
    {
        out.lane_ofst = o_lane_ofst;
    }

    agen power_spectrum       = init(dst);
    power_spectrum.minmax_opt = 3;

    // log2N must be in [3, 8]; the DoA host only admits angle FFT sizes 8 to 128.
    if (log2N == 8 || log2N == 7)
    {
        wrapper.size = sizeof(int32_t);
        wrapper.n1   = row_iter;
        wrapper.n2   = niter1;
        wrapper.n3   = radix4Digits;
        wrapper.n4   = radix4Digits;
        wrapper.n5   = radix4Digits;
        wrapper.n6   = knls_per_tile;
        wrapper.s1   = out_iter1_pitch;
        wrapper.s2   = 64 * o_pitch;
        wrapper.s3   = 16 * o_pitch;
        wrapper.s4   = 4 * o_pitch;
        wrapper.s5   = o_pitch;
        wrapper.s6   = row_iter * out_iter1_pitch;
        INIT_AGEN6(out, wrapper);

        wrapper.size = sizeof(float);
        wrapper.s1   = out_iter1_mag_pitch;
        wrapper.s2   = 64 * o_mag_pitch;
        wrapper.s3   = 16 * o_mag_pitch;
        wrapper.s4   = 4 * o_mag_pitch;
        wrapper.s5   = o_mag_pitch;
        wrapper.s6   = row_iter * out_iter1_mag_pitch;
        INIT_AGEN6(power_spectrum, wrapper);
    }
    else if (log2N == 6 || log2N == 5)
    {
        wrapper.size = sizeof(int32_t);
        wrapper.n1   = row_iter;
        wrapper.n2   = niter1;
        wrapper.n3   = radix4Digits;
        wrapper.n4   = radix4Digits;
        wrapper.n5   = knls_per_tile;
        wrapper.s1   = out_iter1_pitch;
        wrapper.s2   = 16 * o_pitch;
        wrapper.s3   = 4 * o_pitch;
        wrapper.s4   = o_pitch;
        wrapper.s5   = row_iter * out_iter1_pitch;
        INIT_AGEN5(out, wrapper);

        wrapper.size = sizeof(float);
        wrapper.s1   = out_iter1_mag_pitch;
        wrapper.s2   = 16 * o_mag_pitch;
        wrapper.s3   = 4 * o_mag_pitch;
        wrapper.s4   = o_mag_pitch;
        wrapper.s5   = row_iter * out_iter1_mag_pitch;
        INIT_AGEN5(power_spectrum, wrapper);
    }
    else if (log2N == 4 || log2N == 3)
    {
        wrapper.size = sizeof(int32_t);
        wrapper.n1   = row_iter;
        wrapper.n2   = niter1;
        wrapper.n3   = radix4Digits;
        wrapper.n4   = knls_per_tile;
        wrapper.s1   = out_iter1_pitch;
        wrapper.s2   = 4 * o_pitch;
        wrapper.s3   = o_pitch;
        wrapper.s4   = row_iter * out_iter1_pitch;
        INIT_AGEN4(out, wrapper);

        wrapper.size = sizeof(float);
        wrapper.s1   = out_iter1_mag_pitch;
        wrapper.s2   = 4 * o_mag_pitch;
        wrapper.s3   = o_mag_pitch;
        wrapper.s4   = row_iter * out_iter1_mag_pitch;
        INIT_AGEN4(power_spectrum, wrapper);
    }

    cfgs[cfg_idx++] = extract_agen_cfg(in);
    if (need_transpose)
    {
        cfgs[cfg_idx++] = extract_agen_cfg(out);
    }
    else
    {
        cfgs[cfg_idx++] = extract_agen_cfg(power_spectrum);
    }

    it++;
    return it;
}
