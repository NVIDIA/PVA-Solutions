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

#ifndef FFT_RADIX3_H
#define FFT_RADIX3_H

#include <cupva_device.h>
#include <fft_knl.h>

/* `groups` runs G batches in one call, each agen stepping `groupStride` between them. */
int32_t func(fft_batched_radix3_init)(STYPE *ping, STYPE *pong, STYPE *tmp, int32_t *twiddle, int32_t shift_qbits,
                                      AgenCFG *cfgs, int32_t &cfg_idx, FFTFactorization *factorization, int32_t it,
                                      int32_t pitch, int32_t groups, int32_t groupStride);

void func(fft_batched_radix3)(AgenCFG *cfgs, int32_t &cfg_idx, int32_t n, int32_t groups);

void func(fft_batched_radix3_digit_reverse_transpose)(AgenCFG *cfgs, int32_t &cfg_idx, FFTFactorization *factorization,
                                                      int32_t groups);

#endif