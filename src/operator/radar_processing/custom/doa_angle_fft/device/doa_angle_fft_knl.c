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

/**
 * @file doa_angle_fft_knl.c
 * @brief VPU kernel helper functions for DOA Angle FFT processing
 *
 * This file contains the building block functions used by the main VPU kernel
 * (doa_angle_fft_top.c) to perform 2D FFT-based Direction of Arrival estimation.
 *
 * - window_and_remap_antennas: 2D window and 8x8 (Tx x Rx) -> 16x4 virtual antenna re-mapping of one detection
 * - windowing_*_azimuth: per-detection windowing into a band of the packed azimuth grid
 * - fft_batched_*_azimuth / fft_batched_*_elevation: the two FFTs on the shared common/fft engine
 *
 * Peak search, interpolation and the snapshot/result transfers are in doa_angle_estimation_knl.c.
 *
 * Implementation Notes:
 * - Uses vectorized operations (dvintx, vintx) for SIMD processing
 * - AGEN (Address Generators) for efficient data access patterns
 * - Fixed-point arithmetic with configurable Q-format
 * - pragmas for loop unrolling and pipelining optimization
 */

#include "doa_angle_fft_knl.h"

#include "doa_angle_fft_param.h"

#include <fft_radix2.h>
#include <fft_radix4.h>

/*
 * Applies the 2D window and re-maps the 8 x 8 (Tx x Rx) snapshot into the 16 x 4 (azimuth x elevation)
 * virtual antenna grid.
 * The transposed load reads one complex sample from each of 8 rows; the output alternates between two
 * 8-row halves (the re-mapping) and is stored transposed for the azimuth FFT. `width` shapes the
 * source tile, `out_width` the destination row (wider when grids are packed) and `sub_width` the
 * complex written per row (one grid).
 */
static inline int32_t window_and_remap_antennas_init(STYPE *src, STYPE *dst, int32_t *win, int32_t shift_qbits,
                                                     AgenCFG *cfgs, int32_t &cfg_idx, int32_t log2N, int32_t it,
                                                     int32_t width, int32_t pad, int32_t out_width, int32_t sub_width)
{
    const int32_t n    = 1 << log2N;
    const int32_t vecw = chess_elementsof(DVTYPEX); // 16

    const int32_t in_pitch  = (COMPLEX_ELEMENT_NUM * width) * 2 + pad;
    const int32_t out_pitch = (COMPLEX_ELEMENT_NUM * out_width) + pad;
    const int32_t win_pitch = n;

    const int32_t i_lane_ofst = ((width * COMPLEX_ELEMENT_NUM * 2) + vecw - 1) / vecw;
    const int32_t o_lane_ofst = ((out_width * COMPLEX_ELEMENT_NUM) + vecw - 1) / vecw;

    AgenWrapper wrapper;

    agen in      = init(src);
    in.lane_ofst = i_lane_ofst;
    wrapper.size = sizeof(int32_t);
    wrapper.n1   = width * 2;
    wrapper.n2   = n / 16;
    wrapper.s1   = 2;
    wrapper.s2   = 8 * in_pitch;
    INIT_AGEN2(in, wrapper);

    agen out      = init(dst);
    out.round     = shift_qbits;
    out.lane_ofst = o_lane_ofst;
    wrapper.size  = sizeof(int32_t);
    wrapper.n1    = 2;
    wrapper.n2    = sub_width;
    wrapper.n3    = n / 16;
    wrapper.s1    = 8 * out_pitch;
    wrapper.s2    = 2;
    wrapper.s3    = 16 * out_pitch;
    INIT_AGEN3(out, wrapper);

    agen w       = init(win);
    wrapper.size = sizeof(int32_t);

    wrapper.n1 = 2;
    wrapper.n2 = 4;
    wrapper.n3 = 4;
    wrapper.n4 = 4;
    wrapper.s1 = vecw / 2;
    wrapper.s2 = win_pitch;
    wrapper.s3 = vecw;
    wrapper.s4 = 4 * win_pitch;
    INIT_AGEN4(w, wrapper);

    cfgs[cfg_idx++] = extract_agen_cfg(in);
    cfgs[cfg_idx++] = extract_agen_cfg(out);
    cfgs[cfg_idx++] = extract_agen_cfg(w);

    it++;

    return it;
}

static inline void window_and_remap_antennas(AgenCFG *cfgs, int32_t &cfg_idx, int32_t n, int32_t iter)
{
    DVTYPEX v1, v2, vw;

    agen_A in  = init_agen_A_from_cfg(cfgs[cfg_idx++]);
    agen_C out = init_agen_C_from_cfg(cfgs[cfg_idx++]);
    agen_B win = init_agen_B_from_cfg(cfgs[cfg_idx++]);

    // The smallest supported shape (32 x 8) gives n * iter = 32.
    const int32_t niter = n * iter;

    for (int32_t i = 0; i < niter; i++) chess_loop_range(32, )
    chess_unroll_loop(8)
    chess_prepare_for_pipelining
    {
        v1 = dvint_load_transp2_di(in);

        vw.lo = vint_load(win);
        vw.hi = vw.lo;

        v2 = dvmulw(v1, vw, VPU_TRUNC_16);

        vstore_transp2_i(v2, out);
    }
}

/* Windows each detection into its own column band of the packed buffer; the radix-4 stages and the
 * digit reverse then run once over all `groups` bands. */
int32_t windowing_init_azimuth(int32_t *src, int32_t *dst, int32_t *win, AgenCFG *cfgs, doa_angle_fft_params_t *param,
                               int32_t groups, int32_t slot)
{
    int32_t cfg_idx          = 0;
    int32_t it               = 0;
    const int32_t log2N      = param->log2N_azimuth_size;
    const int32_t el         = 1 << param->log2N_elevation_size;
    const int32_t line_width = TILE_BATCH;
    const int32_t in_pad     = 2;

    /* Column band of this detection, in int32: one complex per elevation bin. */
    int32_t *slot_dst = dst + (slot * el * COMPLEX_ELEMENT_NUM);

    return window_and_remap_antennas_init(src, slot_dst, win, param->window_shift_qbits, cfgs, cfg_idx, log2N, it,
                                          line_width, in_pad, groups * el, el);
}

int32_t fft_batched_init_azimuth(int32_t *ping, int32_t *pong, int32_t *tmp, int32_t **twiddles, AgenCFG *cfgs,
                                 doa_angle_fft_params_t *param, int32_t groups)
{
    int32_t cfg_idx                   = 0;
    const int32_t log2N               = param->log2N_azimuth_size;
    const int32_t twiddle_shift_qbits = param->twiddle_shift_qbits;
    const int32_t el                  = 1 << param->log2N_elevation_size;
    const int32_t out_line_width      = 1 << param->log2N_azimuth_size;
    const int32_t in_pad              = 2;
    const int32_t out_pad             = 2;
    /* Windowing already consumed iteration 0 and left its result in pong. */
    int32_t it = 1;
    /* Packed row: `groups` grids of `el` complex, side by side at the KERNEL_BATCH group stride. */
    const int32_t packed_width = groups * el;
    const int32_t packed_pitch = (COMPLEX_ELEMENT_NUM * packed_width) + in_pad;
    const int32_t iter         = packed_width / KERNEL_BATCH;

    it = fft_batched_radix4_init_int(ping, pong, tmp, twiddles[0], twiddle_shift_qbits, cfgs, cfg_idx,
                                     param->factorization_azimuth, it, packed_pitch, iter,
                                     KERNEL_BATCH * COMPLEX_ELEMENT_NUM, ADDITIONAL_SHIFT_QBITS);

    if (log2N & 1)
    {
        it = func(fft_batched_radix2_digit_reverse_power_init)(tmp, pong, cfgs, cfg_idx, log2N, 1, it,
                                                               ADDITIONAL_SHIFT_QBITS >> 1, packed_width, in_pad,
                                                               out_line_width, out_pad, iter, KERNEL_BATCH);
    }
    else
    {
        it = func(fft_batched_radix4_digit_reverse_power_init)(tmp, pong, cfgs, cfg_idx, log2N, 1, it,
                                                               ADDITIONAL_SHIFT_QBITS, packed_width, in_pad,
                                                               out_line_width, out_pad, iter, KERNEL_BATCH);
    }

    return 0;
}

int32_t fft_batched_init_elevation(int32_t *ping, int32_t *pong, int32_t *tmp, int32_t **twiddles, AgenCFG *cfgs,
                                   doa_angle_fft_params_t *param)
{
    int32_t cfg_idx                   = 0;
    const int32_t log2N               = param->log2N_elevation_size;
    const int32_t twiddle_shift_qbits = param->twiddle_shift_qbits;
    int32_t it                        = 1;
    const int32_t line_width          = (1 << param->log2N_azimuth_size);
    const int32_t pad                 = 2;
    const int32_t iter                = (1 << param->log2N_azimuth_size) / KERNEL_BATCH;

    it = fft_batched_radix4_init_int(ping, pong, tmp, twiddles[0], twiddle_shift_qbits, cfgs, cfg_idx,
                                     param->factorization_elevation, it, (COMPLEX_ELEMENT_NUM * line_width) + pad, iter,
                                     KERNEL_BATCH * COMPLEX_ELEMENT_NUM, ADDITIONAL_SHIFT_QBITS);
    if (log2N & 1)
    {
        it = func(fft_batched_radix2_digit_reverse_power_init)(tmp, pong, cfgs, cfg_idx, log2N, 0, it,
                                                               ADDITIONAL_SHIFT_QBITS >> 1, line_width, pad, line_width,
                                                               pad, iter, KERNEL_BATCH);
    }
    else
    {
        it = func(fft_batched_radix4_digit_reverse_power_init)(tmp, pong, cfgs, cfg_idx, log2N, 0, it,
                                                               ADDITIONAL_SHIFT_QBITS, line_width, pad, line_width, pad,
                                                               iter, KERNEL_BATCH);
    }

    return 0;
}

/* Windowing for one detection. Its trip count is per detection (n * el / KERNEL_BATCH); only the
 * destination stride differs between slots. */
void windowing_exec_azimuth(AgenCFG *cfgs, doa_angle_fft_params_t *param)
{
    int32_t cfg_idx    = 0;
    const int32_t n    = 1 << param->log2N_azimuth_size;
    const int32_t iter = (1 << param->log2N_elevation_size) / KERNEL_BATCH;

    window_and_remap_antennas(cfgs, cfg_idx, n, iter);
}

/* Radix-4 stages plus digit reverse over all `groups` packed grids in one run. */
int32_t fft_batched_exec_azimuth(AgenCFG *cfgs, doa_angle_fft_params_t *param, int32_t groups)
{
    int32_t cfg_idx     = 0;
    const int32_t log2N = param->log2N_azimuth_size;
    const int32_t n     = 1 << log2N;
    const int32_t iter  = (groups * (1 << param->log2N_elevation_size)) / KERNEL_BATCH;

    for (int32_t i = 0; i < (log2N - 1) / 2; i++)
    {
        fft_batched_radix4_int(cfgs, cfg_idx, n, iter);
    }

    if (log2N & 1)
    {
        func(fft_batched_radix2_digit_reverse_transpose)(cfgs, cfg_idx, param->factorization_azimuth, iter);
    }
    else
    {
        func(fft_batched_radix4_digit_reverse_transpose)(cfgs, cfg_idx, param->factorization_azimuth, iter);
    }

    return 0;
}

float fft_batched_exec_elevation(AgenCFG *cfgs, doa_angle_fft_params_t *param)
{
    int32_t cfg_idx     = 0;
    const int32_t log2N = param->log2N_elevation_size;
    const int32_t n     = 1 << log2N;
    const int32_t iter  = (1 << param->log2N_azimuth_size) / KERNEL_BATCH;
    float max_val       = 0.0f;

    // Execute the first several radix-4 stages
    for (int32_t i = 0; i < (log2N - 1) / 2; i++)
    {
        fft_batched_radix4_int(cfgs, cfg_idx, n, iter);
    }

    // Execute the digit-reverse stage and transpose the output
    // if transposeOutput is true
    if (log2N & 1)
    {
        // Execute radix-2 if complex number isn't power of 4
        max_val = func(fft_batched_radix2_digit_reverse_power)(cfgs, cfg_idx, n, iter, ADDITIONAL_SHIFT_QBITS >> 1);
    }
    else
    {
        // Execute radix-4 if complex number is power of 4
        max_val = func(fft_batched_radix4_digit_reverse_power)(cfgs, cfg_idx, n, iter, ADDITIONAL_SHIFT_QBITS);
    }

    return max_val;
}
