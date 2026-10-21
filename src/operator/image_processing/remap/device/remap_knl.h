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

#ifndef REMAP_PVA_KNL_H
#define REMAP_PVA_KNL_H

#include "remap_param.h"

#include <cupva_device.h>

#define TILE_INFO (0)
#define CLAMP_COORDS (1)
#define CALC_COORDS_UV (2)
#define TILE_INFO_UV (3)
#define CLAMP_COORDS_UV (4)
#define CFG_NUM (5)

typedef struct
{
    uint16_t tx;
    uint16_t ty;
    int32_t xBegin;
    int32_t yBegin;
} TileInfo;

typedef struct
{
    int32_t niter[CFG_NUM];
    AgenCFG in[CFG_NUM];
    AgenCFG out[CFG_NUM];
    AgenCFG mask[CFG_NUM];
} RemapParam;

typedef struct
{
    int32_t niter;
    AgenCFG inU;
    AgenCFG inV;
    AgenCFG out;
} MergeUVParam;

void get_tile_info_init(RemapParam *param, int32_t *coords);

void get_tile_info_exec(RemapParam *param, TileInfo *tileInfo, int32_t imageWidth, int32_t imageHeight);

void clamp_coords_init(RemapParam *param, int32_t *coords);

void clamp_coords_exec(RemapParam *param, int32_t imageWidth, int32_t imageHeight, BorderModeDev borderMode);

void calc_coords_UV_init(RemapParam *param, int32_t *coords, int32_t *coordsUV_mask);

void calc_coords_UV_exec(RemapParam *param, int32_t imageWidth, int32_t imageHeight);

void get_tile_info_UV_init(RemapParam *param);

void get_tile_info_UV_exec(RemapParam *param, TileInfo *tileInfoUV, int32_t imageWidth, int32_t imageHeight);

void clamp_coords_UV_init(RemapParam *param, int32_t *coordsUV_mask);

void clamp_coords_UV_exec(RemapParam *param, int32_t imageWidth, int32_t imageHeight, BorderModeDev borderMode);

void split_UV_exec(uint8_t *src, uint8_t *dstU, uint8_t *dstV, int32_t tx, int32_t ty);

void merge_UV_init(MergeUVParam *param, uint8_t *dstUV);

void merge_UV_exec(MergeUVParam *param, uint8_t *srcU, uint8_t *srcV);

void update_sampler(CupvaSampler *samplerCfg, uint8_t *src, int32_t *coords, uint8_t *dst, TileInfo *tileInfo);

#endif // REMAP_PVA_KNL_H
