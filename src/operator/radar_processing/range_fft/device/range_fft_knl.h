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

#ifndef RANGE_FFT_KNL_H
#define RANGE_FFT_KNL_H

#include <RadarDefinitions.h>
#include <cupva_device.h>

typedef struct param_fft_batched_s
{
    AgenCFG cfgs[64];
    int32_t window_shift_qbits;
    int32_t twiddle_shift_qbits;
    int32_t input_lane_offset;
    int32_t output_lane_offset;
    int32_t input_is_complex; // 0 = real-input path, 1 = complex-input path
    /* 0 = range-innermost output ([..][Nb]): the final store scatters the TILE_BATCH chirps of a
     * vector across TILE_BATCH rows, one row per chirp.  1 = RANGE_RX_DOPPLER ([Nb][NofRx][Nr]):
     * chirps are innermost, so the same vector is stored contiguously as one row of a bin. */
    int32_t out_chirp_innermost;
    int32_t groups; // Chirp blocks per tile, stored by the init for the exec
    int32_t *windowBuffer;
    int32_t *
        twiddleTable[MAX_NUM_RADIX_GROUPS + 1]; // [0]=radix3, [1]=radix4, [2]=radix5, [3]=post-process (real path only)
    FFTFactorization *factorization;
} param_fft_batched_t;

/* `groups` packs G blocks of TILE_BATCH chirps into one tile and runs them as a single fused FFT;
 * groups == 1 is one block per tile. */
int32_t fft_batched_int_init(int32_t *src, int32_t *ping, int32_t *pong, int32_t *tmp, param_fft_batched_t *param,
                             int32_t groups, int32_t srcGroupStride, int32_t dstGroupStride, int32_t outRowStride);

int32_t fft_batched_int_exec(void *args);

#endif
