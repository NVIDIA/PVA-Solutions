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
VMEM_GSDF_HANDLER(A, srcSPHdl, 1);
VMEM_SEQDF_HANDLER(A, dstHdl);
VMEM_SEQDF_HANDLER(A, dstSPHdl);
VMEM_POINTER(A, dstExtPtr);
VMEM_POINTER(A, dstSPExtPtr);

VMEM(A, int32_t, outImageWidth);
VMEM(A, int32_t, outImageHeight);
VMEM(A, int32_t, inImageWidth);
VMEM(A, int32_t, inImageHeight);
VMEM(A, int32_t, outLinePitch);
VMEM(A, int32_t, outLinePitchSP);
VMEM(A, TransformTypeDev, transformType_v);
VMEM(A, InterpTypeDev, interpType_v);
VMEM(A, float, invWarpMat_v, 9);
VMEM(A, TileInfo, tileInfoNext);
VMEM(A, TileInfo, tileInfoCur);
VMEM(A, TileInfo, tileInfoSP);
VMEM(A, CupvaSampler, samplerCfg);
VMEM(A, CupvaSampler, samplerCfgU);
VMEM(A, CupvaSampler, samplerCfgV);
VMEM(A, CalcCordsParamSP, calcCordsParamSP);
VMEM(A, MergeUVParam, mergeUVParam);
VMEM(A, AddOfstCordsParam, addOfstCordsParam);
VMEM(A, AddOfstCordsParamSP, addOfstCordsParamSP);

// Double buffer for output with line pitch TILE_W
VMEM(B, uint8_t, dst_v, TILE_W *TILE_H * 2);
VMEM(B, uint8_t, dstSP_v, TILE_W *(TILE_H >> 1) * 2);
VMEM(A, uint8_t, dstU, (TILE_W / 2) * (TILE_H / 2));
VMEM(C, uint8_t, dstV, (TILE_W / 2) * (TILE_H / 2));
VMEM(A, int32_t, cords_v, TILE_W *TILE_H * 2);
// Q16.16 x/y coordinates in double buffer, note DLUT only accepts unsigned coordinates
VMEM(B, int32_t, cords_rebased, TILE_W *TILE_H * 4);
VMEM(C, int32_t, cordsSP_v, (TILE_W >> 1) * (TILE_H >> 1) * 4);
VMEM(B, int32_t, cordsSP_mask_v, (TILE_W) * (TILE_H >> 1));

VMEM(C, uint8_t, src_v, MAX_TABLE_SIZE * 2);
VMEM(B, uint8_t, srcSP_v, MAX_TABLE_SP_SIZE * 2);
VMEM(A, uint8_t, srcU, MAX_TABLE_SP_SIZE / 2);
VMEM(A, uint8_t, srcV, MAX_TABLE_SP_SIZE / 2);
VMEM(A, float, dMatX, 3);
VMEM(A, float, dMatY, 3);
// x/y numerator base, 64bytes interleaved
VMEM(C, float, cordsBaseNum, TILE_W *TILE_H * 2);
// z denominator base
VMEM(B, float, cordsBaseDen, TILE_W *TILE_H);

CUPVA_VPU_MAIN()
{
    add_ofst_cords_init(&addOfstCordsParam, &cords_v[0]);
    // SP related agen init
    calc_cords_SP_init(&calcCordsParamSP, &cordsSP_mask_v[0], inImageWidth, inImageHeight);
    merge_UV_init(&mergeUVParam, &dstU[0], &dstV[0], &dstSP_v[0]);
    add_ofst_cords_SP_init(&addOfstCordsParamSP, &cordsSP_v[0], &cordsSP_mask_v[0]);

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

    uint64_t dstDramBase   = dstExtPtr.base + dstExtPtr.offset;
    uint64_t dstSPDramBase = dstSPExtPtr.base + dstSPExtPtr.offset;
    cupvaSamplerSetup(&samplerCfg, &input, &indices, &output);

    // plane U
    CupvaSamplerInput2D inputU = input;
    inputU.data                = &srcU[0];
    // according to VPI implementation, out of range value is 128
    inputU.outOfRangeVal            = 128;
    CupvaSamplerIndices2D indicesUV = indices;
    indicesUV.height                = TILE_H / 2;
    indicesUV.width                 = TILE_W / 2;
    indicesUV.linePitch             = TILE_W / 2;
    indicesUV.interleaving          = SAMPLER_INTERLEAVING_ELEMENTS;
    CupvaSamplerOutput outputU      = output;
    outputU.data                    = &dstU[0];
    outputU.pitch                   = (uint16_t)(TILE_W / 2);
    cupvaSamplerSetup(&samplerCfgU, &inputU, &indicesUV, &outputU);

    // plane V
    CupvaSamplerInput2D inputV = inputU;
    inputV.data                = &srcV[0];
    CupvaSamplerOutput outputV = outputU;
    outputV.data               = &dstV[0];
    cupvaSamplerSetup(&samplerCfgV, &inputV, &indicesUV, &outputV);

    cupvaSamplerLink(&samplerCfg, &samplerCfgU);
    cupvaSamplerLink(&samplerCfgU, &samplerCfgV);

    const int32_t ntileX = (outImageWidth + TILE_W - 1) / TILE_W;
    const int32_t ntileY = (outImageHeight + TILE_H - 1) / TILE_H;

    // three ping pong buffers for tracking input/index/output
    uint32_t pingPong    = 0;
    uint32_t inPingPong  = 0;
    uint32_t outPingPong = 0;

    // init the first tile info
    tileInfoNext.fullBlack = 1;
    tileInfoCur.fullBlack  = 1;

    int32_t nextTileIdX = 0, nextTileIdY = 0;
    int32_t curTileIdX = 0, curTileIdY = 0;
    const int32_t cordsSize   = TILE_W * TILE_H * 2;
    const int32_t cordsSPSize = (TILE_W >> 1) * (TILE_H >> 1) * 2;
    const int32_t dstSize     = TILE_W * TILE_H;
    const int32_t dstSPSize   = TILE_W * (TILE_H >> 1);

    calc_cords_for_next_tile(&tileInfoNext, &cords_v[0], &cordsBaseNum[0], &cordsBaseDen[0], &dMatX[0], &dMatY[0],
                             nextTileIdX, nextTileIdY, inImageWidth, inImageHeight, QBITS, transformType_v);

    // line pitch equals tx, will update when having effective tile
    cupvaGSDFOpen(srcHdl, &src_v[inPingPong * MAX_TABLE_SIZE], 0, MAX_TABLE_SIZE);
    cupvaGSDFOpen(srcSPHdl, &srcSP_v[inPingPong * MAX_TABLE_SP_SIZE], 0, MAX_TABLE_SP_SIZE);
    // detect if first tile needs transfer
    if (!tileInfoNext.fullBlack)
    {
        // update tile info
        cupvaGSDFUpdateVmemAddr(srcHdl, &src_v[inPingPong * MAX_TABLE_SIZE], tileInfoNext.tx, MAX_TABLE_SIZE);
        cupvaGSDFUpdateTiles(srcHdl, &tileInfoNext.xBegin, &tileInfoNext.yBegin, tileInfoNext.tx, tileInfoNext.ty, 1);
        cupvaGSDFTrig(srcHdl);

        // coordinates generating for sampler processing
        generate_cords_SP(&calcCordsParamSP, &addOfstCordsParamSP, &addOfstCordsParam, &tileInfoNext, &tileInfoSP,
                          &cordsSP_v[pingPong * cordsSPSize], &cords_v[0], &cords_rebased[pingPong * cordsSize],
                          inImageWidth, inImageHeight);

        // the source is UV interleaved plane, so we need to add 2x in x direction
        cupvaGSDFUpdateVmemAddr(srcSPHdl, &srcSP_v[inPingPong * MAX_TABLE_SP_SIZE], 2 * tileInfoSP.tx,
                                MAX_TABLE_SP_SIZE);
        int32_t xBeginSP = tileInfoSP.xBegin * 2;
        cupvaGSDFUpdateTiles(srcSPHdl, &xBeginSP, &tileInfoSP.yBegin, 2 * tileInfoSP.tx, tileInfoSP.ty, 1);
        cupvaGSDFTrig(srcSPHdl);
    }
    // calculate next tile id info
    update_for_next_tile(&tileInfoCur, &tileInfoNext, nextTileIdX, nextTileIdY, ntileX);

    // dst writing to DRMA through SQDF
    cupvaSQDFOpen(dstHdl);
    cupvaSQDFOpen(dstSPHdl);

    int32_t validOutNum = 0;
    const int32_t ntile = ntileX * ntileY;
    for (int32_t i = 0; i < ntile; i++)
    {
        // generate current output through DLUT
        if (!tileInfoCur.fullBlack)
        {
            // make sure current src is ready
            cupvaGSDFSync(srcHdl);
            cupvaGSDFSync(srcSPHdl);
            split_UV_exec(&srcSP_v[inPingPong * MAX_TABLE_SP_SIZE], &srcU[0], &srcV[0], 2 * tileInfoSP.tx,
                          tileInfoSP.ty);

            update_sampler(&samplerCfg, &src_v[inPingPong * MAX_TABLE_SIZE], &cords_rebased[pingPong * cordsSize],
                           &dst_v[outPingPong * dstSize], tileInfoCur.tx, tileInfoCur.ty);
            update_sampler(&samplerCfgU, &srcU[0], &cordsSP_v[pingPong * cordsSPSize], &dstU[0], tileInfoSP.tx,
                           tileInfoSP.ty);
            update_sampler(&samplerCfgV, &srcV[0], &cordsSP_v[pingPong * cordsSPSize], &dstV[0], tileInfoSP.tx,
                           tileInfoSP.ty);

            if (i > 0)
            {
                cupvaSQDFSync(dstHdl);
                cupvaSQDFSync(dstSPHdl);
                validOutNum++;
            }

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

            // coordinates generating for sampler processing
            generate_cords_SP(&calcCordsParamSP, &addOfstCordsParamSP, &addOfstCordsParam, &tileInfoNext, &tileInfoSP,
                              &cordsSP_v[pingPong * cordsSPSize], &cords_v[0], &cords_rebased[pingPong * cordsSize],
                              inImageWidth, inImageHeight);

            cupvaGSDFUpdateVmemAddr(srcSPHdl, &srcSP_v[inPingPong * MAX_TABLE_SP_SIZE], 2 * tileInfoSP.tx,
                                    MAX_TABLE_SP_SIZE);
            int32_t xBeginSP = tileInfoSP.xBegin * 2;
            cupvaGSDFUpdateTiles(srcSPHdl, &xBeginSP, &tileInfoSP.yBegin, 2 * tileInfoSP.tx, tileInfoSP.ty, 1);
            cupvaGSDFTrig(srcSPHdl);
        }

        // wait for current output to be ready
        if (!tileInfoCur.fullBlack)
        {
            cupvaSamplerWait();
            uint64_t dst_d = dstDramBase + (curTileIdX * TILE_W + curTileIdY * TILE_H * outLinePitch) * sizeof(uint8_t);
            cupvaSQDFUpdateAddr(dstHdl, 0, cupvaGetVmemAddress(&dst_v[outPingPong * dstSize]), MEMTYPE_VMEM, dst_d,
                                MEMTYPE_DRAM);
            cupvaSQDFFlushAndTrig(dstHdl);

            // process semiplanar dst
            // merge U and V into final dst
            merge_UV_exec(&mergeUVParam, &dstSP_v[outPingPong * dstSPSize]);
            uint64_t dstSP_d =
                dstSPDramBase + (curTileIdX * TILE_W + curTileIdY * (TILE_H / 2) * outLinePitchSP) * sizeof(uint8_t);
            cupvaSQDFUpdateAddr(dstSPHdl, 0, cupvaGetVmemAddress(&dstSP_v[outPingPong * dstSPSize]), MEMTYPE_VMEM,
                                dstSP_d, MEMTYPE_DRAM);
            cupvaSQDFFlushAndTrig(dstSPHdl);
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
        cupvaSQDFSync(dstSPHdl);
    }
    cupvaGSDFClose(srcHdl);
    cupvaGSDFClose(srcSPHdl);
    cupvaSQDFClose(dstHdl);
    cupvaSQDFClose(dstSPHdl);

    return 0;
}