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

#ifndef WARP_PERSPECTIVE_PVA_KNL_H
#define WARP_PERSPECTIVE_PVA_KNL_H

#include "warpPerspective_param.h"

#include <cupva_device.h>

typedef struct
{
    uint16_t tx;
    uint16_t ty;
    int32_t xBegin;
    int32_t yBegin;
    int32_t fullBlack;
} TileInfo;

typedef struct
{
    int32_t niter;
    AgenCFG a_in;
    AgenCFG a_out;
    AgenCFG a_mask;
    int32_t imageWidth;
    int32_t imageHeight;
} CalcCordsParamSP;

typedef struct
{
    int32_t niter;
    AgenCFG a_inU;
    AgenCFG a_inV;
    AgenCFG a_out;
} MergeUVParam;

typedef struct
{
    int32_t niter;
    AgenCFG a_in;
    AgenCFG a_out;
    AgenCFG a_mask;
} AddOfstCordsParamSP;

typedef struct
{
    int32_t niter;
    AgenCFG a_in;
    AgenCFG a_out;
} AddOfstCordsParam;

void init_cords_base(float *cordsBaseNum, float *cordsBaseDen, float *warpMat);

void calc_delta_x(float *dMatX, float *warpMat);

void calc_delta_y(float *dMatY, float *warpMat);

// DLUT only accepts unsigned coordinates, so we need to add offset to the coordinates to make it positive
void add_ofst_cords_init(AddOfstCordsParam *param, int32_t *cords);

void add_ofst_cords_exec(AddOfstCordsParam *param, int32_t *cords_rebased, const int32_t ofst_x, const int32_t ofst_y);

void update_for_next_tile(TileInfo *tileInfoCur, TileInfo *tileInfoNext, int32_t &nextTileIdX, int32_t &nextTileIdY,
                          const int32_t ntileX);

void calc_cords_for_next_tile(TileInfo *tileInfoNext, int32_t *cords, float *cordsBaseNum, float *cordsBaseDen,
                              float *dMatX, float *dMatY, const int32_t nextTileIdX, const int32_t nextTileIdY,
                              const int32_t imageWidth, const int32_t imageHeight, const int32_t qbits,
                              const TransformTypeDev transformType);

void split_UV_exec(uint8_t *src, uint8_t *dstU, uint8_t *dstV, const int32_t tx, const int32_t ty);

void calc_cords_SP_init(CalcCordsParamSP *param, int32_t *cordsSP_mask, const int32_t imageWidth,
                        const int32_t imageHeight);

void merge_UV_init(MergeUVParam *param, uint8_t *srcU, uint8_t *srcV, uint8_t *dst);

void merge_UV_exec(MergeUVParam *param, uint8_t *dst);

void add_ofst_cords_SP_init(AddOfstCordsParamSP *param, int32_t *cords, int32_t *cordsSP_mask);

void update_sampler(CupvaSampler *samplerCfg, uint8_t *src, int32_t *cords, uint8_t *dst, const int32_t tx,
                    const int32_t ty);

void generate_cords_SP(CalcCordsParamSP *calcCordsParamSP, AddOfstCordsParamSP *addOfstCordsParamSP,
                       AddOfstCordsParam *addOfstCordsParam, TileInfo *tileInfoNext, TileInfo *tileInfoSP,
                       int32_t *cordsSP, int32_t *cords, int32_t *cords_rebased, const int32_t inImageWidth,
                       const int32_t inImageHeight);

#endif // WARP_PERSPECTIVE_PVA_KNL_H
