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

#include "remap_knl.h"
#include "remap_param.h"

#include <cupva_device.h>
#include <stdint.h>

// Handlers for DMA operations
VMEM_RDF_UNIFIED(A, warpMapHdl);
#ifdef USE_GSDF
VMEM_GSDF_HANDLER(A, srcHdl, 1);
VMEM_GSDF_HANDLER(A, srcUVHdl, 1);
#else
VMEM_SEQDF_HANDLER(A, srcHdl);
VMEM_POINTER(A, srcExtPtr);
VMEM_SEQDF_HANDLER(A, srcUVHdl);
VMEM_POINTER(A, srcUVExtPtr);
#endif
VMEM_RDF_UNIFIED(A, dstHdl);
VMEM_RDF_UNIFIED(A, dstUVHdl);

VMEM(A, int32_t, inImageWidth);
VMEM(A, int32_t, inImageHeight);
VMEM(A, int32_t, outImageWidth);
VMEM(A, int32_t, outImageHeight);
VMEM(A, int32_t, inLinePitch);
VMEM(A, int32_t, inLinePitchUV);
VMEM(A, InterpTypeDev, interpType_v);
VMEM(A, BorderModeDev, borderMode_v);

VMEM(A, TileInfo, tileInfoNext);
VMEM(A, TileInfo, tileInfoCur);
VMEM(A, TileInfo, tileInfoUV);
VMEM(A, CupvaSampler, samplerCfg);
VMEM(A, CupvaSampler, samplerCfgU);
VMEM(A, CupvaSampler, samplerCfgV);
VMEM(A, RemapParam, remapParam);
VMEM(A, MergeUVParam, mergeUVParam);

VMEM(A, float, warpMap_v, RDF_DOUBLE(float, 2 * TILE_W, TILE_H)); // DMA -> VPU (pingpong)
VMEM(C, int32_t, coords_v, TILE_W *TILE_H * 2);                   // VPU
VMEM(A, int32_t, clamped_coords, TILE_W *TILE_H * 4);             // VPU -> DLUT (pingpong)
VMEM(A, int32_t, coordsUV_v, (TILE_W) * (TILE_H >> 1) * 2);       // [ch_x, ch_y], VPU -> DLUT (pingpong)
VMEM(B, int32_t, coordsUV_mask_v, (TILE_W) * (TILE_H >> 1) * 2);  // [left, right], VPU

VMEM(B, uint8_t, src_v, MAX_TABLE_SIZE * 2);   // DMA -> DLUT (pingpong)
VMEM(B, uint8_t, srcUV_v, MAX_TABLE_UV_SIZE);  // DMA -> VPU
VMEM(B, uint8_t, srcU, MAX_TABLE_UV_SIZE / 2); // VPU -> DLUT
VMEM(C, uint8_t, srcV, MAX_TABLE_UV_SIZE / 2); // VPU -> DLUT

VMEM(C, uint8_t, dst_v, RDF_DOUBLE(uint8_t, TILE_W, TILE_H));        // DLUT -> DMA (pingpong)
VMEM(C, uint8_t, dstU, (TILE_W / 2) * (TILE_H / 2));                 // DLUT -> VPU
VMEM(B, uint8_t, dstV, (TILE_W / 2) * (TILE_H / 2));                 // DLUT -> VPU
VMEM(C, uint8_t, dstUV_v, RDF_SINGLE(uint8_t, TILE_W, TILE_H >> 1)); // DLUT -> DMA

CUPVA_VPU_MAIN()
{
    get_tile_info_init(&remapParam, coords_v);
    clamp_coords_init(&remapParam, coords_v);
    // UV plane related agen init
    calc_coords_UV_init(&remapParam, coords_v, coordsUV_mask_v);
    get_tile_info_UV_init(&remapParam);
    clamp_coords_UV_init(&remapParam, coordsUV_mask_v);
    merge_UV_init(&mergeUVParam, dstUV_v);

    cupvaRasterDataFlowOpen(warpMapHdl, &warpMap_v[0]);
#ifdef USE_GSDF
    cupvaGSDFOpen(srcHdl, &src_v[0], 0, MAX_TABLE_SIZE);
    cupvaGSDFOpen(srcUVHdl, &srcUV_v[0], 0, MAX_TABLE_UV_SIZE);
#else
    cupvaSQDFOpen(srcHdl);
    cupvaSQDFOpen(srcUVHdl);
#endif
    cupvaRasterDataFlowOpen(dstHdl, &dst_v[0]);
    cupvaRasterDataFlowOpen(dstUVHdl, &dstUV_v[0]);

    CupvaSamplerInput2D input = {
        .data           = NULL,
        .type           = SAMPLER_INPUT_TYPE_U8,
        .width          = 0,
        .height         = 0,
        .linePitch      = 0,
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
        .interleaving       = SAMPLER_INTERLEAVING_32B, // indices are in 32 byte groups, x,x,x,….,y,y,y,….
    };

    CupvaSamplerOutput output = {
        .data      = NULL,
        .pitch     = TILE_W,
        .transMode = TRANS_MODE_NONE,
    };

    cupvaSamplerSetup(&samplerCfg, &input, &indices, &output);

    // Setup CupvaSampler for plane U
    CupvaSamplerInput2D inputU      = input;
    inputU.data                     = &srcU[0];
    inputU.outOfRangeVal            = 128; // VPI requirements
    CupvaSamplerIndices2D indicesUV = indices;
    indicesUV.height                = TILE_H / 2;
    indicesUV.width                 = TILE_W / 2;
    indicesUV.linePitch             = TILE_W / 2;
    indicesUV.interleaving          = SAMPLER_INTERLEAVING_ELEMENTS; // xyxyxy...
    CupvaSamplerOutput outputU      = output;
    outputU.data                    = &dstU[0];
    outputU.pitch                   = TILE_W / 2;
    cupvaSamplerSetup(&samplerCfgU, &inputU, &indicesUV, &outputU);

    // Setup CupvaSampler for plane V
    CupvaSamplerInput2D inputV = inputU;
    inputV.data                = &srcV[0];
    CupvaSamplerOutput outputV = outputU;
    outputV.data               = &dstV[0];
    cupvaSamplerSetup(&samplerCfgV, &inputV, &indicesUV, &outputV);

    cupvaSamplerLink(&samplerCfg, &samplerCfgU);
    cupvaSamplerLink(&samplerCfgU, &samplerCfgV);

#if !defined(USE_GSDF)
    uint64_t srcDramBase   = srcExtPtr.base + srcExtPtr.offset;
    uint64_t srcUVDramBase = srcUVExtPtr.base + srcUVExtPtr.offset;
#endif

    uint32_t pingPong    = 0;
    int32_t coordsSize   = TILE_W * TILE_H * 2;
    int32_t coordsUVSize = (TILE_W >> 1) * (TILE_H >> 1) * 2;

    // 1. RDF: get HWC format indices from DRAM
    float *warpMap = (float *)cupvaRasterDataFlowAcquire(warpMapHdl);
    // 2. VPU: reformat indices and get tile info
    cupvaModifyAgenCfgBase(&remapParam.in[TILE_INFO], warpMap);
    get_tile_info_exec(&remapParam, &tileInfoCur, inImageWidth, inImageHeight);
    cupvaRasterDataFlowRelease(warpMapHdl);

    // 3. GSDF/SQDF: update and trigger, get input for next tile
    {
#ifdef USE_GSDF
        cupvaGSDFUpdateVmemAddr(srcHdl, &src_v[pingPong * MAX_TABLE_SIZE], tileInfoCur.tx, MAX_TABLE_SIZE);
        cupvaGSDFUpdateTiles(srcHdl, &tileInfoCur.xBegin, &tileInfoCur.yBegin, tileInfoCur.tx, tileInfoCur.ty, 1);
        cupvaGSDFTrig(srcHdl);
#else
        uint64_t src_d = srcDramBase + (tileInfoCur.yBegin * inLinePitch + tileInfoCur.xBegin) * sizeof(uint8_t);
        cupvaSQDFUpdateAddr(srcHdl, 0, src_d, MEMTYPE_DRAM, inLinePitch,
                            cupvaGetVmemAddress(&src_v[pingPong * MAX_TABLE_SIZE]), MEMTYPE_VMEM, tileInfoCur.tx);
        cupvaSQDFUpdateTileSize(srcHdl, 0, tileInfoCur.tx, tileInfoCur.ty);
        cupvaSQDFFlushAndTrig(srcHdl);
#endif

        // [8 x', 8 y'] -> ((tl + tr + bl + br) * 0.25 - 0.5) * 0.5 -> [ch_x, ch_y]
        cupvaModifyAgenCfgBase(&remapParam.out[CALC_COORDS_UV], &coordsUV_v[pingPong * coordsUVSize]);
        calc_coords_UV_exec(&remapParam, inImageWidth, inImageHeight);

        // [ch_x, ch_y] -> {xBegin, yBegin, tx, ty}
        cupvaModifyAgenCfgBase(&remapParam.in[TILE_INFO_UV], &coordsUV_v[pingPong * coordsUVSize]);
        get_tile_info_UV_exec(&remapParam, &tileInfoUV, inImageWidth / 2, inImageHeight / 2);

        // the source is UV interleaved plane, so we need to add 2x in x direction
#ifdef USE_GSDF
        cupvaGSDFUpdateVmemAddr(srcUVHdl, &srcUV_v[0], 2 * tileInfoUV.tx, MAX_TABLE_UV_SIZE);
        int32_t xBeginUV = tileInfoUV.xBegin * 2;
        cupvaGSDFUpdateTiles(srcUVHdl, &xBeginUV, &tileInfoUV.yBegin, 2 * tileInfoUV.tx, tileInfoUV.ty, 1);
        cupvaGSDFTrig(srcUVHdl);
#else
        uint64_t srcUV_d = srcUVDramBase + (tileInfoUV.yBegin * inLinePitch + 2 * tileInfoUV.xBegin) * sizeof(uint8_t);
        cupvaSQDFUpdateAddr(srcUVHdl, 0, srcUV_d, MEMTYPE_DRAM, inLinePitch, cupvaGetVmemAddress(&srcUV_v[0]),
                            MEMTYPE_VMEM, 2 * tileInfoUV.tx);
        cupvaSQDFUpdateTileSize(srcUVHdl, 0, 2 * tileInfoUV.tx, tileInfoUV.ty);
        cupvaSQDFFlushAndTrig(srcUVHdl);
#endif

        cupvaModifyAgenCfgBase(&remapParam.out[CLAMP_COORDS], &clamped_coords[pingPong * coordsSize]);
        clamp_coords_exec(&remapParam, inImageWidth, inImageHeight, borderMode_v);

        cupvaModifyAgenCfgBase(&remapParam.in[CLAMP_COORDS_UV], &coordsUV_v[pingPong * coordsUVSize]);
        cupvaModifyAgenCfgBase(&remapParam.out[CLAMP_COORDS_UV], &coordsUV_v[pingPong * coordsUVSize]);
        clamp_coords_UV_exec(&remapParam, inImageWidth / 2, inImageHeight / 2, borderMode_v);
    }

    const int32_t ntileX = (outImageWidth + TILE_W - 1) / TILE_W;
    const int32_t ntileY = (outImageHeight + TILE_H - 1) / TILE_H;
    const int32_t ntile  = ntileX * ntileY;

    for (int32_t i = 0; i < ntile; i++)
    {
        if (i != ntile - 1)
        {
            // 1. RDF: get HWC format indices from DRAM
            float *warpMap = (float *)cupvaRasterDataFlowAcquire(warpMapHdl);
            // 2. VPU: reformat indices and get tile info
            cupvaModifyAgenCfgBase(&remapParam.in[TILE_INFO], warpMap);
            get_tile_info_exec(&remapParam, &tileInfoNext, inImageWidth, inImageHeight);
            cupvaRasterDataFlowRelease(warpMapHdl);
        }

        // 4. VPU: split srcUV into srcU and srcV
        {
            uint8_t *dst = (uint8_t *)cupvaRasterDataFlowAcquire(dstHdl);
            update_sampler(&samplerCfg, &src_v[pingPong * MAX_TABLE_SIZE], &clamped_coords[pingPong * coordsSize],
                           &dst[0], &tileInfoCur);
            update_sampler(&samplerCfgU, &srcU[0], &coordsUV_v[pingPong * coordsUVSize], &dstU[0], &tileInfoUV);
            update_sampler(&samplerCfgV, &srcV[0], &coordsUV_v[pingPong * coordsUVSize], &dstV[0], &tileInfoUV);
#ifdef USE_GSDF
            cupvaGSDFSync(srcHdl);
#else
            cupvaSQDFSync(srcHdl);
#endif

            // 3. GSDF/SQDF: update and trigger, get input for Y plane
            if (i != ntile - 1)
            {
                pingPong ^= 1;
#ifdef USE_GSDF
                cupvaGSDFUpdateVmemAddr(srcHdl, &src_v[pingPong * MAX_TABLE_SIZE], tileInfoNext.tx, MAX_TABLE_SIZE);
                cupvaGSDFUpdateTiles(srcHdl, &tileInfoNext.xBegin, &tileInfoNext.yBegin, tileInfoNext.tx,
                                     tileInfoNext.ty, 1);
                cupvaGSDFTrig(srcHdl);
#else
                uint64_t src_d =
                    srcDramBase + (tileInfoNext.yBegin * inLinePitch + tileInfoNext.xBegin) * sizeof(uint8_t);
                cupvaSQDFUpdateAddr(srcHdl, 0, src_d, MEMTYPE_DRAM, inLinePitch,
                                    cupvaGetVmemAddress(&src_v[pingPong * MAX_TABLE_SIZE]), MEMTYPE_VMEM,
                                    tileInfoNext.tx);
                cupvaSQDFUpdateTileSize(srcHdl, 0, tileInfoNext.tx, tileInfoNext.ty);
                cupvaSQDFFlushAndTrig(srcHdl);
#endif
            }

#ifdef USE_GSDF
            cupvaGSDFSync(srcUVHdl);
#else
            cupvaSQDFSync(srcUVHdl);
#endif
            split_UV_exec(srcUV_v, srcU, srcV, 2 * tileInfoUV.tx, tileInfoUV.ty);
        }

        // 5. DLUT: update sampler, start lookup
        cupvaSamplerStart(&samplerCfg);

        // 3. GSDF/SQDF: update and trigger, get input for next tile
        if (i != ntile - 1)
        {
            // [8 x', 8 y'] -> ((tl + tr + bl + br) * 0.25 - 0.5) * 0.5 -> [ch_x, ch_y]
            cupvaModifyAgenCfgBase(&remapParam.out[CALC_COORDS_UV], &coordsUV_v[pingPong * coordsUVSize]);
            calc_coords_UV_exec(&remapParam, inImageWidth, inImageHeight);

            // [ch_x, ch_y] -> {xBegin, yBegin, tx, ty}
            cupvaModifyAgenCfgBase(&remapParam.in[TILE_INFO_UV], &coordsUV_v[pingPong * coordsUVSize]);
            get_tile_info_UV_exec(&remapParam, &tileInfoUV, inImageWidth / 2, inImageHeight / 2);

            // the source is UV interleaved plane, so we need to add 2x in x direction
#ifdef USE_GSDF
            cupvaGSDFUpdateVmemAddr(srcUVHdl, &srcUV_v[0], 2 * tileInfoUV.tx, MAX_TABLE_UV_SIZE);
            int32_t xBeginUV = tileInfoUV.xBegin * 2;
            cupvaGSDFUpdateTiles(srcUVHdl, &xBeginUV, &tileInfoUV.yBegin, 2 * tileInfoUV.tx, tileInfoUV.ty, 1);
            cupvaGSDFTrig(srcUVHdl);
#else
            uint64_t srcUV_d =
                srcUVDramBase + (tileInfoUV.yBegin * inLinePitch + 2 * tileInfoUV.xBegin) * sizeof(uint8_t);
            cupvaSQDFUpdateAddr(srcUVHdl, 0, srcUV_d, MEMTYPE_DRAM, inLinePitch, cupvaGetVmemAddress(&srcUV_v[0]),
                                MEMTYPE_VMEM, 2 * tileInfoUV.tx);
            cupvaSQDFUpdateTileSize(srcUVHdl, 0, 2 * tileInfoUV.tx, tileInfoUV.ty);
            cupvaSQDFFlushAndTrig(srcUVHdl);
#endif

            cupvaModifyAgenCfgBase(&remapParam.out[CLAMP_COORDS], &clamped_coords[pingPong * coordsSize]);
            clamp_coords_exec(&remapParam, inImageWidth, inImageHeight, borderMode_v);

            cupvaModifyAgenCfgBase(&remapParam.in[CLAMP_COORDS_UV], &coordsUV_v[pingPong * coordsUVSize]);
            cupvaModifyAgenCfgBase(&remapParam.out[CLAMP_COORDS_UV], &coordsUV_v[pingPong * coordsUVSize]);
            clamp_coords_UV_exec(&remapParam, inImageWidth / 2, inImageHeight / 2, borderMode_v);
        }

        // 5. RDF: transfer output to DRAM
        {
            cupvaSamplerWait();
            cupvaRasterDataFlowRelease(dstHdl);

            cupvaRasterDataFlowAcquire(dstUVHdl);
            merge_UV_exec(&mergeUVParam, dstU, dstV);
            cupvaRasterDataFlowRelease(dstUVHdl);
        }

        if (i != ntile - 1)
        {
            tileInfoCur = tileInfoNext;
        }
    }

    cupvaRasterDataFlowClose(dstHdl);
    cupvaRasterDataFlowClose(dstUVHdl);
#ifdef USE_GSDF
    cupvaGSDFClose(srcHdl);
    cupvaGSDFClose(srcUVHdl);
#else
    cupvaSQDFClose(srcHdl);
    cupvaSQDFClose(srcUVHdl);
#endif
    cupvaRasterDataFlowClose(warpMapHdl);

    return 0;
}