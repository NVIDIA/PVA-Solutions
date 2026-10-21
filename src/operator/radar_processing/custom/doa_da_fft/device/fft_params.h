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

#ifndef FFT_PARAMS_H
#define FFT_PARAMS_H

#include "doa_params.h" // PEAK_SNAP_HEIGHT

/* One 64-entry radix-4 table; the engine reads it at a per-stage stride for the smaller stages. */
#define TWIDDLE_SIZE (3 * 64 * 2)

#define ELEMENT_NUM_OF_COMPLEX 2
#define BATCH_SIZE PEAK_SNAP_HEIGHT
#define FFT_LOG2_NUM_POINTS 8
#define FFT_NUM_POINTS (1 << FFT_LOG2_NUM_POINTS)
#define FFT_TILE_WIDTH (BATCH_SIZE * ELEMENT_NUM_OF_COMPLEX)
#define FFT_TILE_HEIGHT FFT_NUM_POINTS
#define FFT_TILE_SIZE (FFT_TILE_WIDTH * FFT_TILE_HEIGHT)
#define FFT_TILE_WIDTH_PAD (FFT_TILE_WIDTH + 1)
#define FFT_TILE_HEIGHT_PAD (FFT_TILE_HEIGHT + 1)
#define FFT_TILE_SIZE_PAD (FFT_TILE_WIDTH_PAD * FFT_TILE_HEIGHT_PAD)

#endif