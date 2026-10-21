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

#ifndef DOPPLER_FFT_KNL_H
#define DOPPLER_FFT_KNL_H

#include <RadarDefinitions.h>
#include <cupva_device.h>
#include <doppler_fft_param.h>

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

// Parameters for complex tile transpose kernel
typedef struct param_transpose_s
{
    AgenCFG cfgs[2]; // agen configs for input and output
    int32_t niter;   // iteration count
} param_transpose_t;

typedef struct param_fft_batched_s
{
    AgenCFG cfgs[FFT_BATCHED_MAX_CFGS];
    int32_t window_shift_qbits;
    int32_t twiddle_shift_qbits;
    int32_t o_lane_ofst;
    int32_t *windowBuffer;
    int32_t *twiddleTable[MAX_NUM_RADIX_GROUPS]; // [0] = twiddle3, [1] = twiddle4, [2] = twiddle5
    FFTFactorization *factorization;
    param_transpose_t transpose; // transpose kernel parameters
    int32_t outputLayout;
} param_fft_batched_t;

int32_t fft_batched_int_init(int32_t *ping, int32_t *pong, int32_t *tmp, param_fft_batched_t *param, int32_t pitch,
                             int32_t groups, int32_t srcGroupStride, int32_t dstGroupStride, int32_t chirp_innermost,
                             int32_t i_lane_ofst, int32_t srcRowPitch);

int32_t fft_batched_int_exec(void *args, int32_t groups);

// Complex tile transpose [H][W/2][2] -> [W/2][H][2]
void transpose_for_rx_packing_init(int32_t *src, int32_t *dst, param_transpose_t *param, int32_t width, int32_t height);

void transpose_for_rx_packing(param_transpose_t *param);

#endif