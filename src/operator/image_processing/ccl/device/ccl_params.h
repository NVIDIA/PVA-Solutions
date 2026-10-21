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
#ifndef CCL_PARAMS_H
#define CCL_PARAMS_H

#include <stdint.h>

// CCL params
struct param_ccl
{
    // Number of tiles X dim
    int32_t nIterX;
    // Number of tiles Y dim
    int32_t nIterY;
    // 1 = 4-way connectivity, 0 = 8-way connectivity
    uint8_t isFourWay;
    // Threshold for connected components
    uint8_t threshold;
};

#define VMEM_ATOM_SIZE 64

#define CCL_TILEW 128
#define CCL_TILEH 32
#define CCL_HALOX 64 // We will over-process 64 width, to allow parallelogram access pattern
// Width limit is a result of tracking labels for the next tile row in a VMEM buffer
#define CCL_MAX_IMG_WIDTH 4096
// Max labels are limited by lookup table size and the data type size, with zero reserved for 'no label'
#define CCL_MAX_LABELS 65536

// Round LP up to multiple of VMEM atom size, then add 2 for transposed access
#define CCL_REFORMAT_TILE_W (CCL_TILEW + CCL_HALOX * 2)
#define CCL_REFORMAT_TILE_LP (((CCL_REFORMAT_TILE_W + VMEM_ATOM_SIZE - 1) / VMEM_ATOM_SIZE * VMEM_ATOM_SIZE) + 2)
#define CCL_REFORMAT_TILE_H (CCL_TILEH)
#define CCL_REFORMAT_TILE_LANE_OFFST (CCL_REFORMAT_TILE_LP / VMEM_ATOM_SIZE)

// Generally it seems that we very rarely need to go beyond 2 iterations
#define CCL_COLLAPSE_ITERATIONS 2
// Tilesize is a tradeoff. The number of DLUT tasks we need to run scales with the largest depth in the tile,
// so smaller tiles means outliers don't affect the majority of elements which have small depth.
// However, smaller tiles also means more overhead from the DLUT task and less opportunity to fill the
// DLUT pipeline. Try 2048 for now.
#define CCL_COLLAPSE_TILESIZE 2048

#endif
