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

#ifndef FFT_WINDOWING_H
#define FFT_WINDOWING_H

#include <cupva_device.h>
#include <fft_knl.h>

/*
 * Windowing applied ahead of a batched FFT: every sample is multiplied by its coefficient with
 * dvmulw(..., VPU_TRUNC_16) and rounded by shift_qbits on the store, over a 1D window replayed across
 * `groups` batches.
 *
 * The input is always read with the transposed load. A transposed load with lane_ofst = 0 reads the
 * same contiguous vector as a plain load, so the range-innermost tiles (lane_ofst left at 0) and the
 * chirp-innermost or strided tiles (lane_ofst set by the caller) share one pipelined loop.
 *
 * `win_stride` selects consecutive coefficients (1) or every other one (2, real input, where each
 * coefficient is replicated across a real/imaginary lane pair by the permutation in the loop).
 */

static inline int32_t func(windowing_batched_init)(STYPE *src, STYPE *dst, STYPE *tmp, int32_t *win,
                                                   int32_t shift_qbits, AgenCFG *cfgs, int32_t &cfg_idx,
                                                   FFTFactorization *factorization, int32_t it, int32_t pitch,
                                                   int32_t groups, int32_t groupStride, int32_t transposed_input,
                                                   int32_t i_lane_ofst, int32_t srcGroupRowStride, int32_t win_stride,
                                                   int32_t route_dst_to_tmp)
{
    const int32_t n                   = factorization->n;
    const int32_t numR4Stages         = factorization->groups[RADIX4_GROUP_IDX].numStages;
    const int32_t numR3Stages         = factorization->groups[RADIX3_GROUP_IDX].numStages;
    const int32_t numR5Stages         = factorization->groups[RADIX5_GROUP_IDX].numStages;
    const int32_t isNumR4StagesOdd    = factorization->groups[RADIX4_GROUP_IDX].isOdd;
    const int32_t totalPowerOf2Stages = numR4Stages + isNumR4StagesOdd;
    const int32_t nonPowerOf2Stages   = numR3Stages + numR5Stages;
    // Single power-of-2 stage: the digit reverse reads tmp, so windowing writes into tmp.
    const bool toTmp  = (route_dst_to_tmp != 0) && (totalPowerOf2Stages <= 1) && (nonPowerOf2Stages == 0);
    STYPE *window_dst = toTmp ? tmp : dst;

    const int32_t niter1 = n;

    AgenWrapper wrapper;

    agen in      = init(src);
    wrapper.size = sizeof(STYPE);
    wrapper.n1   = niter1;
    wrapper.n2   = groups;
    if (transposed_input != 0)
    {
        // [batch][sample] tile: the transposed load gathers one row per lane; groups are srcGroupRowStride apart.
        in.lane_ofst = i_lane_ofst;
        wrapper.s1   = COMPLEX_ELEMENT_NUM;
        wrapper.s2   = srcGroupRowStride;
    }
    else
    {
        /* lane_ofst stays 0: the transposed load then reads one contiguous vector per sample. */
        wrapper.s1 = pitch;
        wrapper.s2 = groupStride;
    }
    INIT_AGEN2(in, wrapper);

    agen out  = init(window_dst);
    out.round = shift_qbits;
    // The intermediates are [sample][batch] whatever the input tile layout.
    wrapper.s1 = pitch;
    wrapper.s2 = groupStride;
    INIT_AGEN2(out, wrapper);

    agen w       = init(win);
    wrapper.size = sizeof(int32_t);
    wrapper.n1   = niter1;
    wrapper.s1   = win_stride;
    wrapper.n2   = groups;
    wrapper.s2   = 0; // every group reuses the same window
    INIT_AGEN2(w, wrapper);

    cfgs[cfg_idx++] = extract_agen_cfg(in);
    cfgs[cfg_idx++] = extract_agen_cfg(out);
    cfgs[cfg_idx++] = extract_agen_cfg(w);

    it++;

    return it;
}

static inline void func(windowing_batched)(AgenCFG *cfgs, int32_t &cfg_idx, int32_t n, int32_t groups,
                                           int32_t real_window_stride2)
{
    DVTYPEX v1, v2, vw;
    vshortx vpmt = real_window_stride2 ? vcollate_idx(replicateh(0)) >= chess_elementsof(vintx) : replicateh(0);

    agen_A in  = init_agen_A_from_cfg(cfgs[cfg_idx++]);
    agen_C out = init_agen_C_from_cfg(cfgs[cfg_idx++]);
    agen_B win = init_agen_B_from_cfg(cfgs[cfg_idx++]);

    const int32_t niter = max(((n * groups) + 7) & ~7, 24);

    for (int32_t i = 0; i < niter; i++) chess_loop_range(24, )
    chess_unroll_loop(8)
    chess_prepare_for_pipelining
    {
        v1 = dvint_load_transp2_di(in);
        vw = dvint_load_perm(win, vpmt);
        v2 = dvmulw(v1, vw, VPU_TRUNC_16);
        vstore_i(v2, out);
    }
}

#endif // FFT_WINDOWING_H
