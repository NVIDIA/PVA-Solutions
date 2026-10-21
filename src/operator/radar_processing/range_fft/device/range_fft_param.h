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

#ifndef RANGE_FFT_PARAM_H
#define RANGE_FFT_PARAM_H

#include <RadarDefinitions.h>

#define TILE_BATCH (8)
// Most chirp blocks packed into one DMA tile; the host picks G per geometry (see OpRangeFFT.cpp).
#define TILE_GROUPS_MAX (8)
#define TILE_WIDTH (TILE_BATCH * COMPLEX_ELEMENT_NUM)
#define TILE_HEIGHT (FFT_SIZE_1024)
// A tile row holds FFT_SIZE_1024 int32: up to 1024 real samples or 512 complex samples per chirp.
#define IN_TILE_WIDTH (FFT_SIZE_1024)
#define IN_TILE_HEIGHT (TILE_BATCH)
#define OUT_TILE_WIDTH ((FFT_SIZE_1024 / 2 + 1) * COMPLEX_ELEMENT_NUM)
#define OUT_TILE_HEIGHT (TILE_BATCH)
// Complex-input cap: each chirp must fit in IN_TILE_WIDTH int32_t = FFT_SIZE_1024 ints = 512 complex samples.
#define MAX_COMPLEX_FFT_SIZE (FFT_SIZE_1024 / COMPLEX_ELEMENT_NUM)

#define TWIDDLE_POST_PROCESS_SIZE(SIZE) ALIGNED_SIZE(((SIZE) / 2) * COMPLEX_ELEMENT_NUM)
#define TWIDDLE_SIZE (TWIDDLE3_SIZE + TWIDDLE4_SIZE + TWIDDLE5_SIZE + TWIDDLE_POST_PROCESS_SIZE(FFT_SIZE_1024) * 2)
#define WINDOW_SIZE (FFT_SIZE_1024)

#define RANGE_BINS_REAL_FFT_CALCULATE(SIZE) ((SIZE) / COMPLEX_ELEMENT_NUM + 1)

#endif // RANGE_FFT_PARAM_H