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
#else
VMEM_SEQDF_HANDLER(A, srcHdl);
VMEM_POINTER(A, srcExtPtr);
#endif
VMEM_RDF_UNIFIED(A, dstHdl);

VMEM(A, int32_t, inImageWidth);
VMEM(A, int32_t, inImageHeight);
VMEM(A, int32_t, outImageWidth);
VMEM(A, int32_t, outImageHeight);
VMEM(A, int32_t, inLinePitch);
VMEM(A, InterpTypeDev, interpType_v);
VMEM(A, BorderModeDev, borderMode_v);

VMEM(A, TileInfo, tileInfoNext);
VMEM(A, TileInfo, tileInfoCur);
VMEM(A, CupvaSampler, samplerCfg);
VMEM(A, RemapParam, remapParam);

VMEM(A, float, warpMap_v, RDF_DOUBLE(float, 2 * TILE_W, TILE_H)); // DMA -> VPU (pingpong)
VMEM(C, int32_t, coords_v, TILE_W *TILE_H * 2);                   // VPU
VMEM(A, int32_t, clamped_coords, TILE_W *TILE_H * 4);             // VPU -> DLUT (pingpong)
VMEM(B, uint8_t, src_v, MAX_TABLE_SIZE * 2);                      // DMA -> DLUT (pingpong)
VMEM(C, uint8_t, dst_v, RDF_DOUBLE(uint8_t, TILE_W, TILE_H));     // DLUT -> DMA (pingpong)

CUPVA_VPU_MAIN()
{
    get_tile_info_init(&remapParam, coords_v);
    clamp_coords_init(&remapParam, coords_v);

    cupvaRasterDataFlowOpen(warpMapHdl, &warpMap_v[0]);
#ifdef USE_GSDF
    cupvaGSDFOpen(srcHdl, &src_v[0], 0, MAX_TABLE_SIZE);
#else
    cupvaSQDFOpen(srcHdl);
#endif
    cupvaRasterDataFlowOpen(dstHdl, &dst_v[0]);

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

#if !defined(USE_GSDF)
    uint64_t srcDramBase = srcExtPtr.base + srcExtPtr.offset;
#endif
    cupvaSamplerSetup(&samplerCfg, &input, &indices, &output);

    uint32_t pingPong  = 0;
    int32_t coordsSize = TILE_W * TILE_H * 2;

    // 1. RDF: get HWC format indices from DRAM
    float *warpMap = (float *)cupvaRasterDataFlowAcquire(warpMapHdl);
    // 2. VPU: reformat indices and get tile info
    cupvaModifyAgenCfgBase(&remapParam.in[TILE_INFO], warpMap);
    get_tile_info_exec(&remapParam, &tileInfoCur, inImageWidth, inImageHeight);
    cupvaRasterDataFlowRelease(warpMapHdl);

    // 3. GSDF/SQDF: update and trigger, get input for next tile
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
    cupvaModifyAgenCfgBase(&remapParam.out[CLAMP_COORDS], &clamped_coords[pingPong * coordsSize]);
    clamp_coords_exec(&remapParam, inImageWidth, inImageHeight, borderMode_v);

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

        // 4. DLUT: update sampler, start lookup
        uint8_t *dst = (uint8_t *)cupvaRasterDataFlowAcquire(dstHdl);
        update_sampler(&samplerCfg, &src_v[pingPong * MAX_TABLE_SIZE], &clamped_coords[pingPong * coordsSize], dst,
                       &tileInfoCur);
#ifdef USE_GSDF
        cupvaGSDFSync(srcHdl);
#else
        cupvaSQDFSync(srcHdl);
#endif
        cupvaSamplerStart(&samplerCfg);

        // 3. GSDF/SQDF: update and trigger, get input for next tile
        if (i != ntile - 1)
        {
            pingPong ^= 1;
#ifdef USE_GSDF
            cupvaGSDFUpdateVmemAddr(srcHdl, &src_v[pingPong * MAX_TABLE_SIZE], tileInfoNext.tx, MAX_TABLE_SIZE);
            cupvaGSDFUpdateTiles(srcHdl, &tileInfoNext.xBegin, &tileInfoNext.yBegin, tileInfoNext.tx, tileInfoNext.ty,
                                 1);
            cupvaGSDFTrig(srcHdl);
#else
            uint64_t src_d = srcDramBase + (tileInfoNext.yBegin * inLinePitch + tileInfoNext.xBegin) * sizeof(uint8_t);
            cupvaSQDFUpdateAddr(srcHdl, 0, src_d, MEMTYPE_DRAM, inLinePitch,
                                cupvaGetVmemAddress(&src_v[pingPong * MAX_TABLE_SIZE]), MEMTYPE_VMEM, tileInfoNext.tx);
            cupvaSQDFUpdateTileSize(srcHdl, 0, tileInfoNext.tx, tileInfoNext.ty);
            cupvaSQDFFlushAndTrig(srcHdl);
#endif
            cupvaModifyAgenCfgBase(&remapParam.out[CLAMP_COORDS], &clamped_coords[pingPong * coordsSize]);
            clamp_coords_exec(&remapParam, inImageWidth, inImageHeight, borderMode_v);
        }

        // 5. RDF: transfer output to DRAM
        cupvaSamplerWait();
        cupvaRasterDataFlowRelease(dstHdl);

        if (i != ntile - 1)
        {
            tileInfoCur = tileInfoNext;
        }
    }

    cupvaRasterDataFlowClose(dstHdl);
#ifdef USE_GSDF
    cupvaGSDFClose(srcHdl);
#else
    cupvaSQDFClose(srcHdl);
#endif
    cupvaRasterDataFlowClose(warpMapHdl);

    return 0;
}