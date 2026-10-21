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

#ifndef FFT_KNL_H
#define FFT_KNL_H

#include <RadarDefinitions.h>
#include <cupva_device.h>

#define STYPE int32_t
#define DVTYPE dvint
#define DVTYPEX dvintx
#define VTYPE vint
#define VTYPEX vintx
#define DVLOAD(x) dvint_load(x)
#define DVLOAD_DI(x) dvint_load_di(x)
#define DVLOAD_TRANSP(x) dvint_load_transp(x)
#define VLOAD_TRANSP(x) vint_load_transp(x)

#define TRUNCATE_QBITS (16)
#define TWIDDLE_QBITS (30)
#define WINDOW_QBITS (30)

#define func(name) name##_int

int32_t func(fft_batched_digit_reverse_transpose_init)(STYPE *src, STYPE *dst, AgenCFG *cfgs, int32_t &cfg_idx,
                                                       bool need_transpose, int32_t o_lane_ofst,
                                                       FFTFactorization *factorization, int32_t it, int32_t pitch,
                                                       int32_t groups, int32_t srcGroupStride, int32_t dstGroupStride,
                                                       int32_t out_round_override);

/*
 * Digit reverse for the two FFTs of a 2D angle FFT; log2N must be in [3, 8]. Two configs each:
 *   need_transpose == 0  the second FFT: the *_power forms write only |X|^2, with minmax tracking,
 *                        and return the maximum.
 *   need_transpose != 0  the first FFT: the plain func(fft_batched_radix{2,4}_digit_reverse_transpose)
 *                        writes the complex output, transposed for the second FFT.
 */

int32_t func(fft_batched_digit_reverse_power_init)(STYPE *src, STYPE *dst, AgenCFG *cfgs, int32_t &cfg_idx,
                                                   int32_t log2N, bool need_transpose, int32_t it, const int32_t niter1,
                                                   const int32_t round_qbits, int32_t in_width, int32_t in_pad,
                                                   int32_t out_width, int32_t out_pad, int32_t iter,
                                                   int32_t kernel_batch);

static inline vfloatx func(round_calc_power)(DVTYPEX &vc, int32_t round_qbits)
{
    vintx vc_real;
    vintx vc_imag;
    /* Round to reduce precision and prevent overflow */
    vc_real = vround(vc.lo, round_qbits);
    vc_imag = vround(vc.hi, round_qbits);

    DVTYPEX dvc_packed;
    dvc_packed.lo = vc_real;
    dvc_packed.hi = vc_imag;

    dvfloatx dvc_packed_float = dvintx_dvfp(dvc_packed);
    dvc_packed_float          = dvc_packed_float * dvc_packed_float;

    vfloatx vc_power = dvc_packed_float.lo + dvc_packed_float.hi;

    return vc_power;
}

#endif // FFT_KNL_H
