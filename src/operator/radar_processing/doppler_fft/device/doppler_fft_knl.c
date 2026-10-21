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

#include "doppler_fft_knl.h"

#include "doppler_fft_param.h"
#include "fft_radix3.h"
#include "fft_radix4.h"
#include "fft_radix5.h"

#include <cupva_device.h>
#include <fft_radix2.h>
#include <fft_windowing.h>

int32_t fft_batched_int_init(STYPE *ping, STYPE *pong, STYPE *tmp, param_fft_batched_t *param, int32_t pitch,
                             int32_t groups, int32_t srcGroupStride, int32_t dstGroupStride, int32_t chirp_innermost,
                             int32_t i_lane_ofst, int32_t srcRowPitch)
{
    /* `pitch` is the chirp-to-chirp stride of the tile buffers, in int32: one vector for the classic
     * one-group tile, or the wider row stride when several range-bin groups share a tile. */
    AgenCFG *cfgs                     = param->cfgs;
    FFTFactorization *factorization   = param->factorization;
    int32_t cfg_idx                   = 0;
    const int32_t window_shift_qbits  = param->window_shift_qbits;
    const int32_t twiddle_shift_qbits = param->twiddle_shift_qbits;
    int32_t *win                      = param->windowBuffer;
    int32_t **twiddles                = param->twiddleTable;
    int32_t numR4Stages               = factorization->groups[RADIX4_GROUP_IDX].numStages;
    const int32_t isNumR4StagesOdd    = factorization->groups[RADIX4_GROUP_IDX].isOdd;
    const int32_t totalPowerOf2Stages = numR4Stages + isNumR4StagesOdd;
    int32_t it                        = 0;
    const bool needTranspose          = (param->outputLayout > 0);

    it = func(windowing_batched_init)(ping, pong, tmp, win, window_shift_qbits, cfgs, cfg_idx, factorization, it, pitch,
                                      groups, srcGroupStride, chirp_innermost, i_lane_ofst, TILE_BATCH * srcRowPitch, 1,
                                      1);

    if (factorization->groups[RADIX5_GROUP_IDX].numStages > 0)
    {
        it = func(fft_batched_radix5_init)(ping, pong, tmp, twiddles[2], twiddle_shift_qbits, cfgs, cfg_idx,
                                           factorization, it, pitch, groups, srcGroupStride);
    }

    if (factorization->groups[RADIX3_GROUP_IDX].numStages > 0)
    {
        it = func(fft_batched_radix3_init)(ping, pong, tmp, twiddles[0], twiddle_shift_qbits, cfgs, cfg_idx,
                                           factorization, it, pitch, groups, srcGroupStride);
    }

    if (totalPowerOf2Stages > 1)
    {
        it = func(fft_batched_radix4_init)(ping, pong, tmp, twiddles[1], twiddle_shift_qbits, cfgs, cfg_idx,
                                           factorization, it, pitch, groups, srcGroupStride, 2);
    }

    it = func(fft_batched_digit_reverse_transpose_init)(tmp, pong, cfgs, cfg_idx, needTranspose, param->o_lane_ofst,
                                                        factorization, it, pitch, groups, srcGroupStride,
                                                        dstGroupStride, -1);
    return 0;
}

// Initialize address generators for complex tile transpose [H][W/2][2] -> [W/2][H][2]
// Uses dvint_load_di to read 8 consecutive complex values from input row
// Uses vstore_transp2_i with lane_ofst to scatter them to 8 different output rows
void transpose_for_rx_packing_init(STYPE *src, STYPE *dst, param_transpose_t *param, int32_t width, int32_t height)
{
    const int32_t numComplex   = width / 2;
    const int32_t niter1       = (numComplex + 7) / 8;      // groups of 8 complex per row
    const int32_t vecw         = chess_elementsof(DVTYPEX); // 16 int32_t = 8 complex
    const int32_t outLinePitch = ((2 * height + (vecw - 1)) & ~(vecw - 1)) + 2;
    AgenWrapper wrapper;

    // Input agen: read 8 complex at a time from each row, then move to next row
    // Layout: [height][numComplex][2] stored row-major
    agen a_in    = init(src);
    wrapper.size = sizeof(STYPE);
    wrapper.n1   = niter1;         // groups of 8 complex per row
    wrapper.s1   = vecw;           // stride to next group (16 int32_t)
    wrapper.n2   = height;         // number of rows
    wrapper.s2   = numComplex * 2; // stride to next row
    INIT_AGEN2(a_in, wrapper);

    // Output agen: write with transpose via lane_ofst
    // Each vstore_transp2_i writes 8 complex to 8 different output rows
    agen a_out   = init(dst);
    wrapper.size = sizeof(STYPE);
    wrapper.n1   = niter1;           // groups of 8 output rows
    wrapper.s1   = outLinePitch * 8; // stride to next group of 8 rows
    wrapper.n2   = height;           // h iterations per group
    wrapper.s2   = 2;                // stride between h values (one complex)
    INIT_AGEN2(a_out, wrapper);
    // lane_ofst: each of 8 complex pairs goes to a different row, spaced outLinePitch apart
    // lane_ofst is in units of 64 bytes = 16 int32_t
    a_out.lane_ofst = outLinePitch / 16;

    param->cfgs[0] = extract_agen_cfg(a_in);
    param->cfgs[1] = extract_agen_cfg(a_out);
    param->niter   = niter1 * height;
}

// Execute complex tile transpose using vector load/store with agens
void transpose_for_rx_packing(param_transpose_t *param)
{
    agen_A a_in  = init_agen_A_from_cfg(param->cfgs[0]);
    agen_C a_out = init_agen_C_from_cfg(param->cfgs[1]);

    const int32_t niter = param->niter;

    // Adjust niter for unrolling and pipelining
    const int32_t niter_adj = (niter + 7) / 8 * 8;
    const int32_t niter_min = niter_adj > 16 ? niter_adj : 16;

    chess_separator_scheduler();
    for (int32_t i = 0; i < niter_min; i++) chess_unroll_loop(8)
    chess_loop_range(16, )
    chess_prepare_for_pipelining
    {
        DVTYPEX vec = DVLOAD_DI(a_in);
        vstore_transp2_i(vec, a_out);
    }
}

int32_t fft_batched_int_exec(void *args, int32_t groups)
{
    param_fft_batched_t *param = (param_fft_batched_t *)args;

    AgenCFG *cfgs                     = param->cfgs;
    int32_t cfg_idx                   = 0;
    FFTFactorization *factorization   = param->factorization;
    const int32_t n                   = factorization->n;
    const int32_t numR5Stages         = factorization->groups[RADIX5_GROUP_IDX].numStages;
    const int32_t numR3Stages         = factorization->groups[RADIX3_GROUP_IDX].numStages;
    const int32_t numR4Stages         = factorization->groups[RADIX4_GROUP_IDX].numStages;
    const int32_t isNumR4StagesOdd    = factorization->groups[RADIX4_GROUP_IDX].isOdd;
    const int32_t totalPowerOf2Stages = numR4Stages + isNumR4StagesOdd;

    // Execute the windowing stage
    func(windowing_batched)(cfgs, cfg_idx, n, groups, 0);

    // Execute the radix-5 stages
    for (int32_t i = 0; i < numR5Stages; i++)
    {
        func(fft_batched_radix5)(cfgs, cfg_idx, n, groups);
    }

    // Execute the radix-3 stages
    for (int32_t i = 0; i < numR3Stages; i++)
    {
        func(fft_batched_radix3)(cfgs, cfg_idx, n, groups);
    }

    // Execute the first several radix-4 stages
    for (int32_t i = 0; i < totalPowerOf2Stages - 1; i++)
    {
        func(fft_batched_radix4)(cfgs, cfg_idx, n, groups);
    }

    // Execute the digit-reverse stage and transpose the output
    // if transposeOutput is true
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

    return 0;
}
