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

#include "doa_fft_knl.h"

#include <fft_knl.h>
#include <fft_radix2.h>
#include <fft_radix4.h>

int32_t func(fft_batched_transpose_init)(STYPE *pong, STYPE *tmp, AgenCFG *cfgs, int32_t &cfg_idx, int32_t log2N,
                                         int32_t it)
{
    agen in, out;

    DVTYPEX v1;
    int32_t n     = 1 << log2N;
    int32_t vecw  = chess_elementsof(v1);
    int32_t pitch = FFT_TILE_WIDTH_PAD;

    STYPE *src = tmp;
    STYPE *dst = pong;

    AgenWrapper wrapper;

    in           = init((VTYPE *)src);
    in.lane_ofst = 1;
    wrapper.size = sizeof(STYPE);
    wrapper.n1   = 2;
    wrapper.n2   = n / vecw;
    wrapper.n3   = vecw / 2;
    wrapper.s1   = 1;
    wrapper.s2   = vecw * pitch;
    wrapper.s3   = 2;
    INIT_AGEN3(in, wrapper);

    out        = init((VTYPE *)dst);
    wrapper.n1 = n;
    wrapper.s1 = vecw;
    INIT_AGEN1(out, wrapper);

    cfgs[cfg_idx++] = extract_agen_cfg(in);
    cfgs[cfg_idx++] = extract_agen_cfg(out);

    it++;

    return it;
}

void func(fft_batched_transpose)(AgenCFG *cfgs, int32_t &cfg_idx, int32_t n)
{
    DVTYPEX v1, v2, v3, v4;

    agen_A in;
    agen_C out;

    in  = init_agen_A_from_cfg(cfgs[cfg_idx++]);
    out = init_agen_C_from_cfg(cfgs[cfg_idx++]);

    int32_t niter = n / 2;

    for (int32_t i = 0; i < niter; i++) chess_loop_range(16, )
    chess_unroll_loop(4)
    chess_prepare_for_pipelining
    {
        v1 = DVLOAD_TRANSP(in);
        v2 = DVLOAD_TRANSP(in);

        v3.lo = v1.lo;
        v3.hi = v2.lo;

        v4.lo = v1.hi;
        v4.hi = v2.hi;

        vstore_i(v3, out);
        vstore_i(v4, out);
    }
}

int32_t fft_batched_int_init(STYPE *ping, STYPE *pong, STYPE *tmp, int32_t *twiddle, int32_t &cfg_idx,
                             param_fft_batched_t *param)
{
    AgenCFG *cfgs       = param->cfgs;
    int32_t nbatch      = param->nbatch;
    int32_t log2N       = param->log2N;
    int32_t shift_qbits = param->shift_qbits;
    int32_t it          = 0;

    /* pitch is the batch row stride and the batch dimension is the engine's group dimension, one
     * vector apart. stage_round is 0: this operator keeps the full magnitude, its consumer scales. */
    const int32_t vecw   = chess_elementsof(DVTYPEX);
    const int32_t pitch  = nbatch * ELEMENT_NUM_OF_COMPLEX;
    const int32_t groups = pitch / vecw;
    it = func(fft_batched_radix4_init)(ping, pong, NULL, twiddle, shift_qbits, cfgs, cfg_idx, param->factorization, it,
                                       pitch, groups, vecw, 0);

    /* The digit reverse writes at the padded output pitch the host factorized with, without rounding. */
    STYPE *reverseSrc = (it & 1) ? pong : ping;
    it = func(fft_batched_digit_reverse_transpose_init)(reverseSrc, tmp, cfgs, cfg_idx, false, 0, param->factorization,
                                                        it, pitch, groups, 0, 0, 0);

    it = func(fft_batched_transpose_init)(pong, tmp, cfgs, cfg_idx, log2N, it);

    return it;
}

int32_t fft_batched_int_exec(int32_t &cfg_idx, void *args)
{
    param_fft_batched_t *param = (param_fft_batched_t *)args;

    AgenCFG *cfgs                   = param->cfgs;
    FFTFactorization *factorization = param->factorization;
    const int32_t n                 = 1 << param->log2N;
    const int32_t vecw              = chess_elementsof(DVTYPEX);
    const int32_t groups            = (param->nbatch * ELEMENT_NUM_OF_COMPLEX) / vecw;
    const int32_t isOdd             = factorization->groups[RADIX4_GROUP_IDX].isOdd;
    const int32_t powerOf2Stages    = factorization->groups[RADIX4_GROUP_IDX].numStages + isOdd;

    /* The radix-4 stages the initializer configured, then the digit reverse, which runs the last
     * power-of-2 stage. The transpose below de-interleaves the batches. */
    for (int32_t i = 0; i < powerOf2Stages - 1; i++)
    {
        func(fft_batched_radix4)(cfgs, cfg_idx, n, groups);
    }

    if (isOdd)
    {
        func(fft_batched_radix2_digit_reverse_transpose)(cfgs, cfg_idx, factorization, groups);
    }
    else
    {
        func(fft_batched_radix4_digit_reverse_transpose)(cfgs, cfg_idx, factorization, groups);
    }

    func(fft_batched_transpose)(cfgs, cfg_idx, n * groups);

    return 0;
}