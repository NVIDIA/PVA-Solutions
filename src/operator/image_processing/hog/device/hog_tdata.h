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

#ifndef HOG_TDATA_H
#define HOG_TDATA_H

#define MAX_CHANNEL_COUNT (4)

#define ORIENTATIONS_NUM (18)
#define CELL_SIZE (4)
#define FEATURE_WIDTH (24)
#define FEATURE_HEIGHT (24)
#define FEATURE_SIZE (FEATURE_WIDTH * FEATURE_HEIGHT)
#define INPUT_WIDTH (FEATURE_WIDTH * CELL_SIZE)
#define INPUT_HEIGHT (FEATURE_HEIGHT * CELL_SIZE)
#define INPUT_SIZE (FEATURE_SIZE * CELL_SIZE * CELL_SIZE)
#define B_WIDTH (6)
#define B_HEIGHT (6)
#define B_SIZE (B_WIDTH * B_HEIGHT)
#define FULL_OUT_WIDTH (FEATURE_WIDTH * B_WIDTH)
#define FULL_OUT_HEIGHT (FEATURE_HEIGHT * B_HEIGHT)
#define FULL_OUT_SIZE (FULL_OUT_WIDTH * FULL_OUT_HEIGHT)
#define CELL_PADDING (1)

#define IN_TYPE uint8_t
#define OUT_TYPE uint8_t

// constants for YUV to RGB
#define INV_SCALE (0.00392157f)    // 1.0f/255.0f
#define CRMAX_FACTOR (0.00549804f) // CrMax/0.5f/Cscale=0.701f/0.5f/255.0f
#define CBMAX_FACTOR (0.00694902f) // CbMax/0.5f/Cscale=0.886f/0.5f/255.0f
#define INV_WG (1.70357751f)       // 1.0f/0.587f
#define WR_OVER_WG (0.50936968f)   // 0.299f/0.587f
#define WB_OVER_WG (0.19420874f)   // 0.114f/0.587f

#define RGB_WIDTH (FEATURE_WIDTH * CELL_SIZE + CELL_SIZE * 2 * CELL_PADDING)
#define RGB_HEIGHT (FEATURE_HEIGHT * CELL_SIZE + CELL_SIZE * 2 * CELL_PADDING)
#define ALIGN_64(x) ((((x) + 63) >> 6) << 6)
#define RGB_LP (ALIGN_64(RGB_WIDTH * sizeof(IN_TYPE)) + 2)
#define RGB_LP_LANE_OFST (RGB_LP >> 6)
#define RGB_BUF_SIZE ALIGN_64(RGB_HEIGHT *RGB_LP)
#define BIN_LP ((RGB_WIDTH + 31) / 32 * 32)

// constants for hist generating
#define HIST_PARALLISM (8)
#define HIST_WIDTH (FEATURE_WIDTH + 2 * CELL_PADDING)
#define HIST_HEIGHT (FEATURE_HEIGHT + 2 * CELL_PADDING)
#define HIST_SIZE (HIST_WIDTH * HIST_HEIGHT)
#define HIST_TYPE uint32_t
#define HIST_SIZE_P ((ALIGN_64(HIST_WIDTH * HIST_HEIGHT * sizeof(HIST_TYPE))) / sizeof(HIST_TYPE))
// divide by cell size
#define DIV_CELL_SIZE_BITS (2)
#define CORDS_QBITS (7)
#define WEIGHT_QBITS (5)
#define HIST_WIDTH_Q (HIST_WIDTH << CORDS_QBITS)
#define HALF_Q (1 << (CORDS_QBITS - 1))
#define ONE_Q (1 << CORDS_QBITS)
#define INTEG_MASK_Q (~((1 << CORDS_QBITS) - 1))
#define EPS (0.0001f)
#define COEF_SIZE_PER_PLANE (FEATURE_SIZE + 16)
#define DEQ_HIST_FACTOR (1.0f / (1 << WEIGHT_QBITS))
// use T4 in vmulw in order NOT overflow when converting to fp32
#define DEQ_NORM_FACTOR (1.0f / (1 << (2 * WEIGHT_QBITS - 4)))
#define OUT_MAX_F (255.0f)
#define ARCTAN_TBL_LOG2 6

// constants for bins voting
#define TAN_10DEG (0.17632698070846498)
#define TAN_30DEG (0.57735026918962573)
#define TAN_50DEG (1.19175359259421)
#define TAN_70DEG (2.7474774194546216)

// sat_opt
#define UNSIGNED_SATURATION (3)

#endif