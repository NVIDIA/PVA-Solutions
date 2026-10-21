/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef FASTCORNER_PARAM_H
#define FASTCORNER_PARAM_H

#define CIRCLE_RADIUS (3)
#define HALO_X (CIRCLE_RADIUS)
#define HALO_Y (CIRCLE_RADIUS)
#define LOC_BUF_SIZE (10240 / 2) // 5120 - used by original flow (non-TopK)
#define LANE_ID_TBL_SZ (32)

// TopK-specific buffer size (per ping/pong buffer)
// Each buffer must hold one full tile: TILE_X * TILE_Y
#if defined(U8_IO) || defined(S8_IO)
#    define TOPK_LOC_BUF_SIZE (4096) // One tile = 128x32 = 4096 max features
#elif defined(U16_IO) || defined(S16_IO)
#    define TOPK_LOC_BUF_SIZE (2048) // One tile = 64x32 = 2048 max features
#endif

// Histogram configuration for TopK
#define HIST_BINS 256
#define HISTOGRAM_VECTOR_WIDTH 32

// Periodic reduction to prevent int16_t overflow
#if defined(U8_IO) || defined(S8_IO)
#    define TILES_PER_REDUCTION 8 // 128x32 = 4096 pixels/tile
#elif defined(U16_IO) || defined(S16_IO)
#    define TILES_PER_REDUCTION 16 // 64x32 = 2048 pixels/tile
#endif

// Maximum intermediate packed feature capacity for TopK
// Must match PVA_FAST_CORNER_MAX_TOPK_FEATURES in OpFastCorner.h
#define MAX_LOC_CAPACITY 250000

// Filter buffer capacity for TopK post-filter stage
#define FILTER_BUF_CAPACITY 2048

#if defined(U8_IO)
#    define TILE_X (128)
#    define TILE_Y (32)

#    define BASE_TYPE uint8_t

#    define VEC_TYPE vcharx
#    define DVEC_TYPE dvcharx
#    define DMEM_TYPE dvuchar
#    define LOAD_TYPE vuchar_load
#    define DLOAD_TYPE dvuchar_load

#    define REPLICATE replicateb
#elif defined(S8_IO)
#    define TILE_X (128)
#    define TILE_Y (32)

#    define BASE_TYPE int8_t

#    define VEC_TYPE vcharx
#    define DVEC_TYPE dvcharx
#    define DMEM_TYPE dvchar
#    define LOAD_TYPE vchar_load
#    define DLOAD_TYPE dvchar_load

#    define REPLICATE replicateb
#elif defined(U16_IO)
#    define TILE_X (64)
#    define TILE_Y (32)

#    define BASE_TYPE uint16_t

#    define VEC_TYPE vshortx
#    define DVEC_TYPE dvshortx
#    define DMEM_TYPE dvushort
#    define DLOAD_TYPE dvushort_load

#    define REPLICATE replicateh
#elif defined(S16_IO)
#    define TILE_X (64)
#    define TILE_Y (32)

#    define BASE_TYPE int16_t

#    define VEC_TYPE vshortx
#    define DVEC_TYPE dvshortx
#    define DMEM_TYPE dvshort
#    define DLOAD_TYPE dvshort_load

#    define REPLICATE replicateh
#else // For HOST's use.
#    include <utility>
template<typename T>
auto getTileSz()
{
    if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>)
        return std::make_pair(128, 32);
    else if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t>)
        return std::make_pair(64, 32);
    else
        return std::make_pair(-1, -1);
}
#endif

#define VECW pva_elementsof(VEC_TYPE)
#define DVECW pva_elementsof(DVEC_TYPE)
#define NUM_DVEC_PER_TILE (TILE_Y * TILE_X / DVECW)

typedef struct
{
    int width;

    int height;

    int niterX;

    int niterY;

    // Number of highest-scoring corners to return (0 = disabled, >0 = TopK enabled)
    int numTopKFeatures;
} FastcornerHost2DevParam;

#endif
