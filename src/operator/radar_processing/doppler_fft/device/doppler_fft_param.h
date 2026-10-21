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

#ifndef DOPPLER_FFT_PARAM_H
#define DOPPLER_FFT_PARAM_H

#include <RadarDefinitions.h>

#define TILE_BATCH (8)
/* Most range-bin groups packed into one tile on the direct path; the host picks G per geometry. */
#define TILE_GROUPS_MAX (8)
/* Most groups when the FFT is not fused, which needs one params_fft set per (buffer half, group). */
#define TILE_GROUPS_MAX_UNFUSED (4)
#define TILE_WIDTH (TILE_BATCH * COMPLEX_ELEMENT_NUM)
#define TILE_HEIGHT (FFT_SIZE_1024)
#define IN_TILE_WIDTH (TILE_WIDTH)
#define IN_TILE_HEIGHT (TILE_HEIGHT)
#define OUT_TILE_WIDTH (TILE_WIDTH)
#define OUT_TILE_HEIGHT (TILE_HEIGHT)
#define OUT_TILE_WIDTH_TRANS (FFT_SIZE_512 / 2 * COMPLEX_ELEMENT_NUM)
#define OUT_TILE_HEIGHT_TRANS (TILE_BATCH)

#define TWIDDLE_SIZE (TWIDDLE3_SIZE + TWIDDLE4_SIZE + TWIDDLE5_SIZE)
#define WINDOW_SIZE (FFT_SIZE_1024)

/* Agen configs one fft_batched_int_init() fills, one per address stream; the host checks this against
 * FFT_BATCHED_MAX_CFGS. Windowing: 3 (in, out, window). Each radix-5/radix-3 stage: 5 (in, 2 outs, twiddle,
 * intra-stage twiddle). Each power-of-2 stage but the last: 4 (in, 2 outs, twiddle). Digit reverse, which also
 * runs the last power-of-2 stage: 2 (in, out). */
#define FFT_BATCHED_CFGS_REQUIRED(numR5Stages, numR3Stages, totalPowerOf2Stages) \
    (3 + 5 * ((numR5Stages) + (numR3Stages)) + (((totalPowerOf2Stages) > 1) ? (4 * ((totalPowerOf2Stages)-1)) : 0) + 2)

#define FFT_BATCHED_MAX_CFGS (40)

#endif // DOPPLER_FFT_PARAM_H
