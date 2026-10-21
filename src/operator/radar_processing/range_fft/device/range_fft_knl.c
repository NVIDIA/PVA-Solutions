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
#include "range_fft_knl.h"

#include "range_fft_param.h"

#include <RadarDefinitions.h>
#include <fft_knl.h>
#include <fft_radix2.h>
#include <fft_radix3.h>
#include <fft_radix4.h>
#include <fft_radix5.h>
#include <fft_windowing.h>

#define STYPE int32_t
#define DVTYPE dvint
#define DVTYPEX dvintx
#define VTYPE vint
#define VTYPEX vintx
#define DVLOAD(x) dvint_load(x)
#define DVLOAD_DI(x) dvint_load_di(x)
#define DVLOAD_TRANSP(x) dvint_load_transp(x)
#define VLOAD_TRANSP(x) vint_load_transp(x)

#define func(name) name##_int

// Post-process agen configs, in the order the init stores them and the exec reads them.
enum
{
    PP_CFG_IN1,
    PP_CFG_IN2,
    PP_CFG_CA,
    PP_CFG_CB,
    PP_CFG_OUT1,
    PP_CFG_OUT3,
    PP_CFG_DC_IN,
    PP_CFG_DC_OUT,
    PP_CFG_COUNT
};

int32_t func(fft_real_batched_post_process_transpose_init)(STYPE *src, STYPE *dst, int32_t *coef_a, int32_t *coef_b,
                                                           int32_t o_lane_ofst, int32_t shift_qbits, AgenCFG *cfgs,
                                                           int32_t &cfg_idx, FFTFactorization *factorization,
                                                           int32_t it, int32_t groups, int32_t midGroupStride,
                                                           int32_t dstGroupStride, int32_t chirp_innermost,
                                                           int32_t outRowStride)
{
    const int32_t n    = factorization->n;
    const int32_t vecw = chess_elementsof(DVTYPEX);

    AgenWrapper wrapper;

    agen in1     = init(src);
    wrapper.size = sizeof(STYPE);
    wrapper.n1   = n;
    wrapper.s1   = vecw;
    wrapper.n2   = groups;
    wrapper.s2   = midGroupStride;
    INIT_AGEN2(in1, wrapper);

    agen in2     = init(src + n * vecw);
    wrapper.size = sizeof(STYPE);
    wrapper.n1   = n;
    wrapper.s1   = -vecw;
    wrapper.n2   = groups;
    wrapper.s2   = midGroupStride;
    INIT_AGEN2(in2, wrapper);

    agen ca      = init(coef_a);
    wrapper.size = sizeof(int32_t);
    wrapper.n1   = n;
    wrapper.s1   = 2;
    wrapper.n2   = groups;
    wrapper.s2   = 0; // every group replays the same coefficient table
    INIT_AGEN2(ca, wrapper);

    agen cb      = init(coef_b);
    wrapper.size = sizeof(int16_t);
    wrapper.n1   = n;
    wrapper.s1   = 2;
    wrapper.n2   = groups;
    wrapper.s2   = 0;
    INIT_AGEN2(cb, wrapper);

    /* Range-innermost output: lane_ofst scatters each bin across the TILE_BATCH chirp rows, stepping one
     * bin (2 ints) along the row. Chirp-innermost output: the vector is one bin's row of chirps (the host
     * passes lane_ofst 0), and the agen steps one row per bin. */
    agen out1      = init(dst);
    out1.round     = 1 + shift_qbits;
    out1.lane_ofst = o_lane_ofst;
    wrapper.size   = sizeof(STYPE);
    wrapper.n1     = n;
    wrapper.s1     = (chirp_innermost != 0) ? outRowStride : 2;
    wrapper.n2     = groups;
    wrapper.s2     = dstGroupStride;
    INIT_AGEN2(out1, wrapper);

    /* The DC fill and the Nyquist tail run once per group, outside the main loop, so they cannot
     * share the strided agens above: those advance along n before they reach the group dimension.
     * They get their own configs whose only dimension is the group, one access each. */
    agen dcIn    = init(src);
    wrapper.size = sizeof(STYPE);
    wrapper.n1   = groups;
    wrapper.s1   = midGroupStride;
    INIT_AGEN1(dcIn, wrapper);

    agen dcOut   = init(src + n * vecw);
    wrapper.size = sizeof(STYPE);
    wrapper.n1   = groups;
    wrapper.s1   = midGroupStride;
    INIT_AGEN1(dcOut, wrapper);

    /* The Nyquist bin: one element past the last bin along the row, or one row past it. */
    agen out3      = init(dst + ((chirp_innermost != 0) ? (n * outRowStride) : (n * 2)));
    out3.round     = 1;
    out3.lane_ofst = o_lane_ofst;
    wrapper.size   = sizeof(STYPE);
    wrapper.n1     = groups;
    wrapper.s1     = dstGroupStride;
    INIT_AGEN1(out3, wrapper);

    cfgs[cfg_idx + PP_CFG_IN1]    = extract_agen_cfg(in1);
    cfgs[cfg_idx + PP_CFG_IN2]    = extract_agen_cfg(in2);
    cfgs[cfg_idx + PP_CFG_CA]     = extract_agen_cfg(ca);
    cfgs[cfg_idx + PP_CFG_CB]     = extract_agen_cfg(cb);
    cfgs[cfg_idx + PP_CFG_OUT1]   = extract_agen_cfg(out1);
    cfgs[cfg_idx + PP_CFG_OUT3]   = extract_agen_cfg(out3);
    cfgs[cfg_idx + PP_CFG_DC_IN]  = extract_agen_cfg(dcIn);
    cfgs[cfg_idx + PP_CFG_DC_OUT] = extract_agen_cfg(dcOut);
    cfg_idx += PP_CFG_COUNT;

    it++;

    return it;
}

void func(fft_real_batched_post_process_transpose)(AgenCFG *cfgs, int32_t &cfg_idx, int32_t n, int32_t groups)
{
    DVTYPEX v1, v2, vt1;

    // generate [0, ... ,0, 1, ... ,1]
    vshortx vpmta = vcollate_idx(replicateh(0)) >= chess_elementsof(vintx);

    /* Each stage initializes only its own agens, in its own scope: the DC fill, the main loop and the
     * Nyquist tail together need nine, more than the VPU keeps live without spilling. */
    const int32_t cfgBase = cfg_idx;
    cfg_idx += PP_CFG_COUNT;

    {
        /* The DC fill and the Nyquist tail step only along the group dimension; see the init. */
        agen_C in0  = init_agen_C_from_cfg(cfgs[cfgBase + PP_CFG_DC_IN]);
        agen_C out0 = init_agen_C_from_cfg(cfgs[cfgBase + PP_CFG_DC_OUT]);

        for (int32_t g = 0; g < groups; g++) chess_loop_range(1, )
        {
            v1 = DVLOAD_DI(in0);
            vstore_i(v1, out0);
        }
    }

    chess_separator_scheduler();

    {
        agen_C in1  = init_agen_C_from_cfg(cfgs[cfgBase + PP_CFG_IN1]);
        agen_C in2  = init_agen_C_from_cfg(cfgs[cfgBase + PP_CFG_IN2]);
        agen_A ca   = init_agen_A_from_cfg(cfgs[cfgBase + PP_CFG_CA]);
        agen_A cb   = init_agen_A_from_cfg(cfgs[cfgBase + PP_CFG_CB]);
        agen_A out1 = init_agen_A_from_cfg(cfgs[cfgBase + PP_CFG_OUT1]);

        const int32_t niter = max(((n * groups) + 3) & ~3, 12);

        for (int32_t i = 0; i < niter; i++) chess_loop_range(12, )
        chess_unroll_loop(4)
        chess_prepare_for_pipelining
        {
            v1 = DVLOAD_DI(in1);
            v2 = DVLOAD_DI(in2);

            dvintx vca  = dvint_load_perm(ca, vpmta);
            vshortx vcb = vshort_load_c2(cb);

            vt1 = dvcmulw_t16(v1, vca);

            vt1.lo = vdotp2_whw(v2.hi, v2.lo, vcb, vt1.lo, 1);
            vt1.hi = vdotpn2_whw(v2.lo, v2.hi, vcb, vt1.hi, 1);

            vstore_transp2_i(vt1, out1);
        }
    }

    chess_separator_scheduler();

    {
        agen_C in3  = init_agen_C_from_cfg(cfgs[cfgBase + PP_CFG_DC_IN]);
        agen_A out3 = init_agen_A_from_cfg(cfgs[cfgBase + PP_CFG_OUT3]);

        for (int32_t g = 0; g < groups; g++) chess_loop_range(1, )
        {
            v1    = DVLOAD_DI(in3);
            v2.lo = v1.lo - v1.hi;
            v2.hi = replicatew(0);
            vstore_transp2_i(v2, out3);
        }
    }
}

int32_t fft_batched_int_init(STYPE *src, STYPE *ping, STYPE *pong, STYPE *tmp, param_fft_batched_t *param,
                             int32_t groups, int32_t srcGroupStride, int32_t dstGroupStride, int32_t outRowStride)
{
    param->groups                     = groups;
    AgenCFG *cfgs                     = param->cfgs;
    int32_t cfg_idx                   = 0;
    FFTFactorization *factorization   = param->factorization;
    int32_t *win                      = param->windowBuffer;
    int32_t **twiddles                = param->twiddleTable;
    const int32_t window_shift_qbits  = param->window_shift_qbits;
    const int32_t twiddle_shift_qbits = param->twiddle_shift_qbits;
    const int32_t input_lane_offset   = param->input_lane_offset;
    const int32_t chirp_innermost     = param->out_chirp_innermost;
    const int32_t output_lane_offset  = param->output_lane_offset;
    const int32_t input_is_complex    = param->input_is_complex;
    int32_t numR4Stages               = factorization->groups[RADIX4_GROUP_IDX].numStages;
    const int32_t isNumR4StagesOdd    = factorization->groups[RADIX4_GROUP_IDX].isOdd;
    const int32_t totalPowerOf2Stages = numR4Stages + isNumR4StagesOdd;
    int32_t it                        = 0;
    STYPE *radix_tmp                  = input_is_complex ? tmp : NULL;
    /* The FFT element stride is one vector; the tile packs `groups` blocks of TILE_BATCH chirps.
     * srcGroupStride steps between those blocks in the DMA tile, dstGroupStride between them in the
     * output tile, and midGroupStride between each group's [n][vecw] slice of the intermediates. */
    const int32_t vecw = chess_elementsof(DVTYPEX);
    // The real path's DC fill copies bin 0 to slot n * vecw for the N - k read, so its slices need n + 1 vectors.
    const int32_t midGroupStride = (input_is_complex ? factorization->n : (factorization->n + 1)) * vecw;

    /* Read with the transposed load at the caller's lane offset, one block of TILE_BATCH chirp rows
     * per group; real input takes every other window coefficient. */
    it = func(windowing_batched_init)(src, pong, radix_tmp, win, window_shift_qbits, cfgs, cfg_idx, factorization, it,
                                      vecw, groups, midGroupStride, 1, input_lane_offset, srcGroupStride,
                                      input_is_complex ? 1 : 2, input_is_complex);

    if (factorization->groups[RADIX5_GROUP_IDX].numStages > 0)
    {
        it = func(fft_batched_radix5_init)(ping, pong, radix_tmp, twiddles[2], twiddle_shift_qbits, cfgs, cfg_idx,
                                           factorization, it, vecw, groups, midGroupStride);
    }

    if (factorization->groups[RADIX3_GROUP_IDX].numStages > 0)
    {
        it = func(fft_batched_radix3_init)(ping, pong, radix_tmp, twiddles[0], twiddle_shift_qbits, cfgs, cfg_idx,
                                           factorization, it, vecw, groups, midGroupStride);
    }

    if (totalPowerOf2Stages > 1)
    {
        it = func(fft_batched_radix4_init)(ping, pong, radix_tmp, twiddles[1], twiddle_shift_qbits, cfgs, cfg_idx,
                                           factorization, it, vecw, groups, midGroupStride, 2);
    }

    if (input_is_complex)
    {
        /* Complex path: the digit reverse writes straight to the output tile. */
        it = func(fft_batched_digit_reverse_transpose_init)(tmp, pong, cfgs, cfg_idx, true, output_lane_offset,
                                                            factorization, it, vecw, groups, midGroupStride,
                                                            dstGroupStride, -1);
        return 0;
    }

    /* Real path: the digit reverse stays inside the intermediates and the post process writes the
     * output tile, so only the latter uses dstGroupStride. */
    STYPE *srcTmp = (it & 1) ? pong : ping;
    it = func(fft_batched_digit_reverse_transpose_init)(srcTmp, tmp, cfgs, cfg_idx, false, 0, factorization, it, vecw,
                                                        groups, midGroupStride, midGroupStride, -1);

    int32_t *twiddlePPCoefA = twiddles[3];
    int32_t *twiddlePPCoefB = twiddlePPCoefA + factorization->twiddlePPSizePerCoef;
    it                      = func(fft_real_batched_post_process_transpose_init)(
        tmp, pong, twiddlePPCoefA, twiddlePPCoefB, output_lane_offset, twiddle_shift_qbits, cfgs, cfg_idx,
        factorization, it, groups, midGroupStride, dstGroupStride, chirp_innermost, outRowStride);

    return 0;
}

int32_t fft_batched_int_exec(void *args)
{
    param_fft_batched_t *param        = (param_fft_batched_t *)args;
    const int32_t groups              = param->groups;
    AgenCFG *cfgs                     = param->cfgs;
    int32_t cfg_idx                   = 0;
    FFTFactorization *factorization   = param->factorization;
    const int32_t n                   = factorization->n;
    const int32_t input_is_complex    = param->input_is_complex;
    const int32_t numR5Stages         = factorization->groups[RADIX5_GROUP_IDX].numStages;
    const int32_t numR3Stages         = factorization->groups[RADIX3_GROUP_IDX].numStages;
    const int32_t numR4Stages         = factorization->groups[RADIX4_GROUP_IDX].numStages;
    const int32_t isNumR4StagesOdd    = factorization->groups[RADIX4_GROUP_IDX].isOdd;
    const int32_t totalPowerOf2Stages = numR4Stages + isNumR4StagesOdd;

    func(windowing_batched)(cfgs, cfg_idx, n, groups, input_is_complex ? 0 : 1);

    for (int32_t i = 0; i < numR5Stages; i++)
    {
        func(fft_batched_radix5)(cfgs, cfg_idx, n, groups);
    }

    for (int32_t i = 0; i < numR3Stages; i++)
    {
        func(fft_batched_radix3)(cfgs, cfg_idx, n, groups);
    }

    for (int32_t i = 0; i < totalPowerOf2Stages - 1; i++)
    {
        func(fft_batched_radix4)(cfgs, cfg_idx, n, groups);
    }

    if (totalPowerOf2Stages == 0)
    {
        if (numR3Stages > 0)
        {
            func(fft_batched_radix3_digit_reverse_transpose)(cfgs, cfg_idx, factorization, groups);
        }
        else
        {
            func(fft_batched_radix5_digit_reverse_transpose)(cfgs, cfg_idx, factorization, groups);
        }
    }
    else
    {
        if (isNumR4StagesOdd == 1)
        {
            func(fft_batched_radix2_digit_reverse_transpose)(cfgs, cfg_idx, factorization, groups);
        }
        else
        {
            func(fft_batched_radix4_digit_reverse_transpose)(cfgs, cfg_idx, factorization, groups);
        }
    }

    if (!input_is_complex)
    {
        func(fft_real_batched_post_process_transpose)(cfgs, cfg_idx, n, groups);
    }

    return 0;
}
