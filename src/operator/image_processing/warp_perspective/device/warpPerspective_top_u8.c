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

#include "warpPerspective_knl.h"
#include "warpPerspective_param.h"

#include <cupva_device.h>
#include <stdint.h>

// Handlers for DMA operations
VMEM_GSDF_HANDLER(A, srcHdl, 1);
VMEM_SEQDF_HANDLER(A, dstHdl);
VMEM_POINTER(A, dstExtPtr);

VMEM(A, int32_t, outImageWidth);
VMEM(A, int32_t, outImageHeight);
VMEM(A, int32_t, inImageWidth);
VMEM(A, int32_t, inImageHeight);
VMEM(A, int32_t, outLinePitch);
VMEM(A, TransformTypeDev, transformType_v);
VMEM(A, InterpTypeDev, interpType_v);
VMEM(A, float, invWarpMat_v, 9);
VMEM(A, TileInfo, tileInfoNext);
VMEM(A, TileInfo, tileInfoCur);
VMEM(A, CupvaSampler, samplerCfg);
VMEM(A, AddOfstCordsParam, addOfstCordsParam);

// Double buffer for output with line pitch TILE_W
VMEM(B, uint8_t, dst_v, TILE_W *TILE_H * 2);
VMEM(A, int32_t, cords_v, TILE_W *TILE_H * 2);
// Q16.16 x/y coordinates in double buffer, note DLUT only accepts unsigned coordinates
VMEM(B, int32_t, cords_rebased, TILE_W *TILE_H * 4);
VMEM(C, uint8_t, src_v, MAX_TABLE_SIZE * 2);
VMEM(A, float, dMatX, 3);
VMEM(A, float, dMatY, 3);
// x/y numerator base, 64bytes interleaved
VMEM(C, float, cordsBaseNum, TILE_W *TILE_H * 2);
// z denominator base
VMEM(B, float, cordsBaseDen, TILE_W *TILE_H);

CUPVA_VPU_MAIN()
{
    add_ofst_cords_init(&addOfstCordsParam, &cords_v[0]);
    // line pitch equals tx, will update when having effective tile
    cupvaGSDFOpen(srcHdl, &src_v[0], 0, MAX_TABLE_SIZE);

    init_cords_base(&cordsBaseNum[0], &cordsBaseDen[0], &invWarpMat_v[0]);
    calc_delta_x(&dMatX[0], &invWarpMat_v[0]);
    calc_delta_y(&dMatY[0], &invWarpMat_v[0]);

    CupvaSamplerInput2D input = {
        .data           = NULL,
        .type           = SAMPLER_INPUT_TYPE_U8,
        .width          = TILE_W,
        .height         = TILE_H,
        .linePitch      = 0U,
        .inputAdv       = 0,
        .outOfRangeMode = SAMPLER_OUT_OF_RANGE_CONSTANT,
        .outOfRangeVal  = 0,
    };

    CupvaSamplerIndices2D indices = {
        .data               = NULL,
        .type               = SAMPLER_INDEX_TYPE_U32,
        .width              = TILE_W,
        .height             = TILE_H,
        .linePitch          = TILE_W,
        .fractionalBits     = QBITS,
        .fractionalHandling = interpType_v == NN ? SAMPLER_FRAC_HANDLING_ROUND : SAMPLER_FRAC_HANDLING_INTERPOLATE,
        .offsetX            = 0,
        .offsetY            = 0,
        .interleaving       = SAMPLER_INTERLEAVING_32B,
    };

    CupvaSamplerOutput output = {
        .data      = NULL,
        .pitch     = (uint16_t)TILE_W,
        .transMode = TRANS_MODE_NONE,
    };

    uint64_t dstDramBase = dstExtPtr.base + dstExtPtr.offset;
    cupvaSamplerSetup(&samplerCfg, &input, &indices, &output);

    const int32_t ntileX = (outImageWidth + TILE_W - 1) / TILE_W;
    const int32_t ntileY = (outImageHeight + TILE_H - 1) / TILE_H;

    // calculate idx for every tile
    uint32_t pingPong = 0;
    // only effective tile will update it
    uint32_t inPingPong    = 0;
    uint32_t outPingPong   = 0;
    tileInfoNext.fullBlack = 1;
    tileInfoCur.fullBlack  = 1;

    int32_t nextTileIdX = 0, nextTileIdY = 0;
    int32_t curTileIdX = 0, curTileIdY = 0;
    const int32_t cordsSize = TILE_W * TILE_H * 2;
    const int32_t dstSize   = TILE_W * TILE_H;
    calc_cords_for_next_tile(&tileInfoNext, &cords_v[0], &cordsBaseNum[0], &cordsBaseDen[0], &dMatX[0], &dMatY[0],
                             nextTileIdX, nextTileIdY, inImageWidth, inImageHeight, QBITS, transformType_v);

    // detect if first tile needs transfer
    if (!tileInfoNext.fullBlack)
    {
        // update tile info
        cupvaGSDFUpdateVmemAddr(srcHdl, &src_v[inPingPong * MAX_TABLE_SIZE], tileInfoNext.tx, MAX_TABLE_SIZE);
        cupvaGSDFUpdateTiles(srcHdl, &tileInfoNext.xBegin, &tileInfoNext.yBegin, tileInfoNext.tx, tileInfoNext.ty, 1);
        cupvaGSDFTrig(srcHdl);
        add_ofst_cords_exec(&addOfstCordsParam, &cords_rebased[pingPong * cordsSize], (tileInfoNext.xBegin) << QBITS,
                            (tileInfoNext.yBegin) << QBITS);
    }
    cupvaSQDFOpen(dstHdl);
    // calculate next tile id info
    update_for_next_tile(&tileInfoCur, &tileInfoNext, nextTileIdX, nextTileIdY, ntileX);

    int32_t validOutNum = 0;
    const int32_t ntile = ntileX * ntileY;
    for (int32_t i = 0; i < ntile; i++)
    {
        // generate current output through DLUT
        if (!tileInfoCur.fullBlack)
        {
            update_sampler(&samplerCfg, &src_v[inPingPong * MAX_TABLE_SIZE], &cords_rebased[pingPong * cordsSize],
                           &dst_v[outPingPong * dstSize], tileInfoCur.tx, tileInfoCur.ty);

            if (i > 0)
            {
                cupvaSQDFSync(dstHdl);
                validOutNum++;
            }

            // make sure current src is ready
            cupvaGSDFSync(srcHdl);
            // dlut to produce the output
            cupvaSamplerStart(&samplerCfg);
        }

        // calculate coordinates for next tile
        if (i != ntile - 1)
        {
            pingPong ^= 1;
            calc_cords_for_next_tile(&tileInfoNext, &cords_v[0], &cordsBaseNum[0], &cordsBaseDen[0], &dMatX[0],
                                     &dMatY[0], nextTileIdX, nextTileIdY, inImageWidth, inImageHeight, QBITS,
                                     transformType_v);
        }
        else
        {
            // end src transfer for last tile
            tileInfoNext.fullBlack = 1;
        }

        // detect if next tile needs transfer
        if (!tileInfoNext.fullBlack)
        {
            inPingPong ^= 1;
            cupvaGSDFUpdateVmemAddr(srcHdl, &src_v[inPingPong * MAX_TABLE_SIZE], tileInfoNext.tx, MAX_TABLE_SIZE);
            cupvaGSDFUpdateTiles(srcHdl, &tileInfoNext.xBegin, &tileInfoNext.yBegin, tileInfoNext.tx, tileInfoNext.ty,
                                 1);
            cupvaGSDFTrig(srcHdl);
            add_ofst_cords_exec(&addOfstCordsParam, &cords_rebased[pingPong * cordsSize],
                                (tileInfoNext.xBegin) << QBITS, (tileInfoNext.yBegin) << QBITS);
        }

        // wait for current output to be ready
        if (!tileInfoCur.fullBlack)
        {
            cupvaSamplerWait();
            uint64_t dst_d = dstDramBase + (curTileIdX * TILE_W + curTileIdY * TILE_H * outLinePitch) * sizeof(uint8_t);
            cupvaSQDFUpdateAddr(dstHdl, 0, cupvaGetVmemAddress(&dst_v[outPingPong * dstSize]), MEMTYPE_VMEM, dst_d,
                                MEMTYPE_DRAM);
            cupvaSQDFFlushAndTrig(dstHdl);
            outPingPong ^= 1;
        }

        // calculate next tile id info
        if (i != ntile - 1)
        {
            curTileIdX = nextTileIdX;
            curTileIdY = nextTileIdY;
            update_for_next_tile(&tileInfoCur, &tileInfoNext, nextTileIdX, nextTileIdY, ntileX);
        }
    }

    // once those last few tiles are all skipped, we don't have chance to sync the last dst buffer, check it here
    if ((validOutNum & 0x1) != outPingPong)
    {
        cupvaSQDFSync(dstHdl);
    }
    cupvaGSDFClose(srcHdl);
    cupvaSQDFClose(dstHdl);

    return 0;
}