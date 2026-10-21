/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CONTOUR_TRACING_KNL_H
#define CONTOUR_TRACING_KNL_H

#include "contour_tracing_param.h"

#include <cupva_device.h>
#include <stdint.h>

#define FILL_ZERO_EVEN (0)
#define FILL_ZERO_ODD (1)
#define FIND_START_IN (2)
#define FIND_START_BM (3)
#define REFORMAT_SRC (4)
#define REFORMAT_DST (5)
#define TRACE_SEG0 (6)
#define TRACE_SEG1 (7)
#define TRACE_SEG2 (8)
#define TRACE_SEG3 (9)
#define MEMCPY_SRC (10)
#define MEMCPY_DST (11)
#define CFG_NUM (12)

// Conv2D constants for 3x3 kernel
#define CONV_KW 3
#define CONV_KWA 4
#define CONV_KWR 1
#define CONV_WIDTH_PAD (MAX_RECT_WIDTH + 2 * CONV_KWR)
#define CONV_HEIGHT_PAD (MAX_RECT_HEIGHT / 2 + CONV_KWR)
#define CONV_BUF_SIZE (CONV_WIDTH_PAD * CONV_HEIGHT_PAD)

// Conv2D 3x3 outputs rows rounded up to multiple of 6
#define CONV_OUTPUT_HEIGHT (((MAX_RECT_HEIGHT / 2 - 1) / 3) + 1) * 6

static const int16_t X_OFFSETS[16] = {1, 1, 0, -1, -1, -1, 0, 1};
static const int16_t Y_OFFSETS[16] = {0, 1, 1, 1, 0, -1, -1, -1};
static const int16_t SHIFT_LUT[16] = {-4, -4, -8, -8, -12, -12, 0, 0};

// Hardcoded reformatted 3x3 bitmap kernel
static const uint8_t BITMAP_KNL_REFORMATTED[32] = {
    32, 64, 128, 0, 16, 0, 1, 0, 0, 0, 0, 0, 32, 64, 128, 0, 8, 4, 2, 0, 0, 0, 0, 0, 16, 0, 1, 0, 8, 4, 2, 0,
};

typedef struct
{
    int32_t x;
    int32_t y;
    int32_t width;
    int32_t height;
} BoundingRect;

#ifdef __cplusplus
extern "C" {
#endif

void fill_zero_init(AgenCFG *cfgs, uint8_t *evenBuf, uint8_t *oddBuf);

void fill_zero_exec(AgenCFG *cfgs, BoundingRect *roi);

void split_even_odd_exec(uint8_t *input, uint8_t *evenBuf, uint8_t *oddBuf, BoundingRect *roi);

void repack_bitmap_exec(uint8_t *buf, BoundingRect *roi);

void find_start_pixel_init(AgenCFG *cfgs);

// Find 4 start pixels from the horizontal and vertical midlines
void find_start_pixel_exec(AgenCFG *cfgs, uint8_t *input, uint8_t *bitmap, BoundingRect *roi, int16_t *startIndices);

void reformat_table_init(AgenCFG *cfgs);

// Reformat src into 4-way parallel subtable layout at dst.
// Each 64-byte input chunk produces 256 output bytes: T0[0..15] | T1[0..15] | T2[0..15] | T3[0..15] | ...
void reformat_table_exec(AgenCFG *cfgs, int32_t size);

void trace_contour_init(AgenCFG *cfgs);

// 4-way parallel contour traversal: Walks 4 segments simultaneously from 4 start points.
int32_t trace_contour_exec(AgenCFG *cfgs, uint8_t *bitmap4Way, uint16_t *dirTable4Way, BoundingRect *roi,
                           int16_t *startIndices, uint16_t *segLens);

void memcpy_u16_init(AgenCFG *cfgs);

void memcpy_u16_exec(AgenCFG *cfgs, uint16_t length);

#ifdef __cplusplus
}
#endif

#endif // CONTOUR_TRACING_KNL_H
