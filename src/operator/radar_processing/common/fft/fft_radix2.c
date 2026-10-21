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

#include <cupva_device.h>
#include <fft_radix2.h>

void func(fft_batched_radix2_digit_reverse_transpose)(AgenCFG *cfgs, int32_t &cfg_idx, FFTFactorization *factorization,
                                                      int32_t groups)
{
    int32_t n = factorization->n;
    DVTYPEX v1, v2;
    DVTYPEX vt1, vt2;

    agen_A in  = init_agen_A_from_cfg(cfgs[cfg_idx++]);
    agen_C out = init_agen_C_from_cfg(cfgs[cfg_idx++]);

    const int32_t niter = max(((((int32_t)n * groups) / 2) + 3) & ~3, 12);

    /* Always the transposed store: when the output is not transposed the init leaves
     * lane_ofst at 0, and the transposed store then writes the same contiguous vector as a
     * plain one. */
    for (int32_t i = 0; i < niter; i++) chess_loop_range(12, )
    chess_unroll_loop(4)
    chess_prepare_for_pipelining
    {
        v1 = DVLOAD_DI(in);
        v2 = DVLOAD_DI(in);

        vaddsub(v1.lo, v2.lo, vt1.lo, vt2.lo);
        vaddsub(v1.hi, v2.hi, vt1.hi, vt2.hi);

        vstore_transp2_i(vt1, out);
        vstore_transp2_i(vt2, out);
    }
}

int32_t func(fft_batched_radix2_digit_reverse_power_init)(STYPE *src, STYPE *dst, AgenCFG *cfgs, int32_t &cfg_idx,
                                                          int32_t log2N, bool need_transpose, int32_t it,
                                                          int32_t round_qbits, int32_t in_width, int32_t in_pad,
                                                          int32_t out_width, int32_t out_pad, int32_t iter,
                                                          int32_t kernel_batch)
{
    return func(fft_batched_digit_reverse_power_init)(src, dst, cfgs, cfg_idx, log2N, need_transpose, it, 2,
                                                      round_qbits, in_width, in_pad, out_width, out_pad, iter,
                                                      kernel_batch);
}

float func(fft_batched_radix2_digit_reverse_power)(AgenCFG *cfgs, int32_t &cfg_idx, int32_t n, int32_t iter,
                                                   int32_t round_qbits)
{
    DVTYPEX v1, v2;
    DVTYPEX vt1, vt2;

    vfloatx vFFTFinal1;
    vfloatx vFFTFinal2;

    agen_A in             = init_agen_A_from_cfg(cfgs[cfg_idx++]);
    agen_C power_spectrum = init_agen_C_from_cfg(cfgs[cfg_idx++]);

    const int32_t niter = n / 2;

    for (int32_t i = 0; i < niter * iter; i++) chess_loop_range(16, )
    chess_unroll_loop(4)
    chess_prepare_for_pipelining
    {
        v1 = dvint_load_di(in);
        v2 = dvint_load_di(in);

        vaddsub(v1.lo, v2.lo, vt1.lo, vt2.lo);
        vaddsub(v1.hi, v2.hi, vt1.hi, vt2.hi);

        vFFTFinal1 = func(round_calc_power)(vt1, round_qbits);
        vFFTFinal2 = func(round_calc_power)(vt2, round_qbits);

        vstore(vFFTFinal1, power_spectrum);
        vstore(vFFTFinal2, power_spectrum);
    }

    int32_t power_max_val_int = power_spectrum.max_val;
    float power_max_val_float = *((float *)(&power_max_val_int));
    return power_max_val_float;
}
