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

#ifndef PEAK_DETECTION_PARAM_H
#define PEAK_DETECTION_PARAM_H

#include "PvaOperatorTypes.h"

#define PEAKDET_MAX_PEAK_COUNT (8192)

#define PEAKDET_SUPPORTED_TX_COUNT (4)

#define PEAKDET_SUPPORTED_REPEAT_FOLD (8)

#define TILE_Y (16)
#define REPEAT_FOLD (8)

#if defined(NCIRX_WIDTH_512)
#    define NCIRX_WIDTH (512)
// Tile width is set to be the number of ramps in nciFinal
#    define TILE_X (NCIRX_WIDTH / REPEAT_FOLD)
// The theoretical maximum number of peaks per tile is in chess pattern.
#    define MAX_PEAK_COUNT_PER_TILE (TILE_X * TILE_Y / 2)
#elif defined(NCIRX_WIDTH_1024)
#    define NCIRX_WIDTH (1024)
#    define TILE_X (NCIRX_WIDTH / REPEAT_FOLD)
#    define MAX_PEAK_COUNT_PER_TILE (TILE_X * TILE_Y / 2)
#endif

#define HALO_X (1)
#define HALO_Y (1)
#define OUTPUT_TILE_Y (800)
#define NUM_INT_PER_CMPLX (2)
#define RX_COUNT (4)
#define TX_COUNT (4)

typedef struct
{
    int32_t tbl1024[RX_COUNT * TX_COUNT];

    int32_t ddmSpaceTbl[RX_COUNT * TX_COUNT];

    uint32_t ddmSpaceBitMask;

    int niterY;
} PeakDetectionHost2DevParam;

// Per output tile data structure.
typedef struct PvaPeakTypeDeviceRec
{
    uint32_t rangeIdx[OUTPUT_TILE_Y];

    uint32_t dopplerFoldIdx[OUTPUT_TILE_Y];

    uint32_t dopplerIdx[OUTPUT_TILE_Y];

    int32_t snap[OUTPUT_TILE_Y][NUM_INT_PER_CMPLX * TX_COUNT * RX_COUNT];
} PvaPeakTypeDevice;
#endif
