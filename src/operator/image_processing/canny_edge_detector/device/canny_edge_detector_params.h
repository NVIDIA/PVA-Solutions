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
#ifndef CANNY_EDGE_DETECTOR_PARAMS_H
#define CANNY_EDGE_DETECTOR_PARAMS_H

enum EdgeType
{
    EDGE_NO     = 0,
    EDGE_WEAK   = 1,
    EDGE_STRONG = 2
};

enum DirectionType
{
    DIR_HORI  = 0,
    DIR_DIAG0 = 1, // 45 degree
    DIR_VERT  = 2,
    DIR_DIAG1 = 3, // -45 degree
    NUM_DIR   = 4
};

#define KH_ALIGNED(size) (((size) + 3) / 4 * 4)

#define TW 128
#define TH 64
#define LINEPITCH ((TW + 31) / 32 * 32 + 1)   // 32k + 1
#define LINEPITCH_T ((TH + 31) / 32 * 32 + 1) // 32k + 1
#define MAX_HALO 3
#define BLKW (TW + MAX_HALO * 2)
#define BLKH (TH + MAX_HALO * 2)
#define TW_NMS 96
#define TH_NMS 64
#define BLKW_NMS (TW_NMS + 2)
#define BLKH_NMS (TH_NMS + 2)
#define QBITS_7 3
#define CANNY_SHIFT 6
#define TG22_F 0.4142135623730950488016887242097f // sqrt(2) - 1
#define TG22 ((int32_t)(TG22_F * (1 << CANNY_SHIFT) + 0.5))
#define TG67_F 2.4142135623730950488016887242097f // sqrt(2) + 1
#define TG67 ((int32_t)(TG67_F * (1 << CANNY_SHIFT) + 0.5))

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