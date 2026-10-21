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

#ifndef FFT_RADIX4_H
#define FFT_RADIX4_H

#include <cupva_device.h>
#include <fft_knl.h>

/* `groups` runs G batches in one call, each agen stepping `groupStride` between them. */
/* `stage_round` is the per-stage rounding shift applied on the butterfly stores: 2 accounts for the
 * radix-4 growth and is what every operator wants unless it carries its own fixed-point scaling
 * downstream, in which case 0 leaves the magnitude untouched. */
int32_t func(fft_batched_radix4_init)(STYPE *ping, STYPE *pong, STYPE *tmp, int32_t *twiddle, int32_t shift_qbits,
                                      AgenCFG *cfgs, int32_t &cfg_idx, FFTFactorization *factorization, int32_t it,
                                      int32_t pitch, int32_t groups, int32_t groupStride, int32_t stage_round);

void func(fft_batched_radix4)(AgenCFG *cfgs, int32_t &cfg_idx, int32_t n, int32_t groups);

/* Digit reverse with the transposed store. The init decides the output form: a non-zero lane_ofst
 * transposes the output on the way out; lane_ofst 0 writes it straight. */
void func(fft_batched_radix4_digit_reverse_transpose)(AgenCFG *cfgs, int32_t &cfg_idx, FFTFactorization *factorization,
                                                      int32_t groups);

/* Digit reverse producing a power spectrum; see the note in fft_knl.h. */
int32_t func(fft_batched_radix4_digit_reverse_power_init)(STYPE *src, STYPE *dst, AgenCFG *cfgs, int32_t &cfg_idx,
                                                          int32_t log2N, bool need_transpose, int32_t it,
                                                          int32_t round_qbits, int32_t in_width, int32_t in_pad,
                                                          int32_t out_width, int32_t out_pad, int32_t iter,
                                                          int32_t kernel_batch);

float func(fft_batched_radix4_digit_reverse_power)(AgenCFG *cfgs, int32_t &cfg_idx, int32_t n, int32_t iter,
                                                   int32_t round_qbits);

#endif