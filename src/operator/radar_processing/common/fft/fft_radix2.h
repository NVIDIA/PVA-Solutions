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

#ifndef FFT_RADIX2_H
#define FFT_RADIX2_H

#include <cupva_device.h>
#include <fft_knl.h>

/*
 * Radix-2 digit reverse. Two families:
 *
 *   *_transpose        the plain form, carrying a group dimension so G batches issue as one call;
 *                      produces the complex output only. It also serves the first FFT of an angle
 *                      FFT, configured by func(fft_batched_digit_reverse_power_init).
 *   *_power            the second FFT of an angle FFT, where the final stage writes only |X|^2,
 *                      with minmax tracking, and returns the peak.
 *
 * The generic initializers both families call live in fft_knl.h, since radix-4 shares them.
 */

/* Transposed store (the output form follows lane_ofst, as in fft_radix4.h); `groups` runs G batches. */
void func(fft_batched_radix2_digit_reverse_transpose)(AgenCFG *cfgs, int32_t &cfg_idx, FFTFactorization *factorization,
                                                      int32_t groups);

int32_t func(fft_batched_radix2_digit_reverse_power_init)(STYPE *src, STYPE *dst, AgenCFG *cfgs, int32_t &cfg_idx,
                                                          int32_t log2N, bool need_transpose, int32_t it,
                                                          int32_t round_qbits, int32_t in_width, int32_t in_pad,
                                                          int32_t out_width, int32_t out_pad, int32_t iter,
                                                          int32_t kernel_batch);

float func(fft_batched_radix2_digit_reverse_power)(AgenCFG *cfgs, int32_t &cfg_idx, int32_t n, int32_t iter,
                                                   int32_t round_qbits);

#endif // FFT_RADIX2_H
