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

#ifndef DOA_FFT_KNL_H
#define DOA_FFT_KNL_H

#include "fft_params.h"

#include <RadarDefinitions.h>
#include <cupva_device.h>

typedef struct param_fft_batched_s
{
    AgenCFG *cfgs;
    int32_t log2N;
    int32_t nbatch;
    int32_t shift_qbits;
    /* Stage structure for the shared radix-4 engine, factorized on the host. */
    FFTFactorization *factorization;
} param_fft_batched_t;

int32_t fft_batched_int_init(int32_t *ping, int32_t *pong, int32_t *tmp, int32_t *twiddle, int32_t &cfg_idx,
                             param_fft_batched_t *param);

int32_t fft_batched_int_exec(int32_t &cfg_idx, void *args);

#endif // DOA_FFT_KNL_H
