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

#include "data_type_defs.h"
#include "image_resize_common.h"
#include "image_resize_param.h"

#include <cupva_device.h>

VMEM(A, int32_t, planeCount);

// sampler tile info
VMEM(A, uint32_t, srcOffsetsX, MAX_NTILE);
VMEM(A, uint32_t, srcOffsetsY, MAX_NTILE);
VMEM(A, uint16_t, srcTileSizesX, MAX_NTILE);
VMEM(A, uint16_t, srcTileSizesY, MAX_NTILE);
VMEM(A, int32_t, srcTileStartsX, MAX_NTILE);
VMEM(A, int32_t, srcTileStartsY, MAX_NTILE);
VMEM(A, int32_t, scaleX_q);
VMEM(A, int32_t, scaleY_q);
VMEM(A, int64_t, scaleX_q_high);
VMEM(A, int64_t, scaleY_q_high);

// Plane 0 data flows:
VMEM_GSDF_HANDLER(A, srcGSDFHdl0, 1);
VMEM_RDF_UNIFIED(A, dstRDFHdl0);

// Plane 1 data flows:
VMEM_GSDF_HANDLER(A, srcGSDFHdl1, 1);
VMEM_RDF_UNIFIED(A, dstRDFHdl1);

VMEM(A, IMAGE_RESIZE_CFG, cfg);
VMEM(A, uint8_t, channelCount, 2);
VMEM(A, uint16_t, srcImageWidth, 2);
VMEM(A, uint16_t, srcImageHeight, 2);
VMEM(A, uint16_t, dstImageWidth, 2);
VMEM(A, uint16_t, dstImageHeight, 2);
VMEM(A, uint16_t, srcTileWidth, 2);
VMEM(A, uint16_t, srcTileHeight, 2);
VMEM(A, uint16_t, srcTileLp, 2);
VMEM(A, uint16_t, dstTileWidth, 2);
VMEM(A, uint16_t, dstTileHeight, 2);
VMEM(A, int32_t, nTileX, 2);
VMEM(A, int32_t, nTileY, 2);

CUPVA_VPU_MAIN()
{
    // init auxiliary buffers
    void *preBufPtr, *postBufPtr;
    uint32_t *idxBufPtr, *laneIdsPtr;
    int32_t *transformMatPtr;

#if defined(NN_OP)
    laneIdsPtr = laneIds;
    idxBufPtr  = idxBuf;
#elif defined(BL_OP)
    transformMatPtr = transformMatrix;
#endif

#if COMPILE_PREPROCESS
    preBufPtr  = (void *)preBuf;
    postBufPtr = (void *)postBuf;
#endif

    for (int p = 0; p < planeCount; p++)
    {
        // init GSDF read dataflow
        GSDFHandler *srcGSDFHdl = (p == 0) ? srcGSDFHdl0 : srcGSDFHdl1;
        // align GSDF tile buffer size to 64
        int srcTileBufSize = (srcTileLp[p] * srcTileHeight[p] + 63) / 64 * 64;
        cupvaGSDFOpen(srcGSDFHdl, srcBufPing, srcTileLp[p], srcTileBufSize);

        // init RDF write dataflow
        UnifiedRDFHandler &dstRDFHdl = (p == 0) ? dstRDFHdl0 : dstRDFHdl1;
        cupvaRasterDataFlowOpen(dstRDFHdl, dstBuf);
        int dstTileLp = cupvaRasterDataFlowGetLinePitch(dstRDFHdl);

        int channelNum = channelCount[p];

        // for multi-channel image formats, CupvaSampler input is the transpose of each channel
        bool transpose_flag = (channelNum > 1);

        // init resize config
        INIT_RESIZE_CFG(cfg, (void *)srcBufPing, (void *)dstBuf, preBufPtr, postBufPtr, idxBufPtr, laneIdsPtr,
                        transformMatPtr, srcTileWidth[p], srcTileHeight[p], srcTileLp[p], dstTileWidth[p],
                        dstTileHeight[p], dstTileLp, channelNum, U8_FLAG);

        int32_t stepCol = transpose_flag ? scaleY_q : scaleX_q;
        int32_t stepRow = transpose_flag ? scaleX_q : scaleY_q;

        // init CupvaSampler
        CupvaSampler sampler = {};
        INIT_SAMPLER_CFG(cfg, sampler, stepCol, stepRow);

        // init tile starts, sizes, and index offsets
        init_sampler_tile_info(srcTileSizesY, srcTileStartsY, srcOffsetsY, nTileY[p], dstTileHeight[p],
                               dstImageHeight[p], scaleY_q_high);
        init_sampler_tile_info(srcTileSizesX, srcTileStartsX, srcOffsetsX, nTileX[p], dstTileWidth[p], dstImageWidth[p],
                               scaleX_q_high);

        // trig read DF for the first tile:
        int32_t tx  = srcTileStartsX[0] * channelNum;
        int32_t ty  = srcTileStartsY[0];
        uint16_t lx = srcTileSizesX[0] * channelNum;
        uint16_t ly = srcTileSizesY[0];

        cupvaGSDFUpdateTiles(srcGSDFHdl, &tx, &ty, &lx, &ly, 1);
        cupvaGSDFTrig(srcGSDFHdl);

        int pingPong = 0;
        for (int i = 0; i < nTileY[p]; i++)
        {
            for (int j = 0; j < nTileX[p]; j++)
            {

                cupvaGSDFSync(srcGSDFHdl);

                DATA_TYPE *src = pingPong ? srcBufPong : srcBufPing;
                DATA_TYPE *dst = (DATA_TYPE *)cupvaRasterDataFlowAcquire(dstRDFHdl);

                pingPong ^= 1;

                // trig read DF for the next tile
                if (i < nTileY[p] - 1 || j < nTileX[p] - 1)
                {

                    int j_next = (j < nTileX[p] - 1) ? j + 1 : 0;
                    int i_next = (j < nTileX[p] - 1) ? i : i + 1;

                    tx = srcTileStartsX[j_next] * channelNum;
                    ty = srcTileStartsY[i_next];
                    lx = srcTileSizesX[j_next] * channelNum;
                    ly = srcTileSizesY[i_next];

                    cupvaGSDFUpdateTiles(srcGSDFHdl, &tx, &ty, &lx, &ly, 1);
                    void *vmem_addr = pingPong ? srcBufPong : srcBufPing;
                    cupvaGSDFUpdateVmemAddr(srcGSDFHdl, vmem_addr, srcTileLp[p], srcTileBufSize);
                    cupvaGSDFTrig(srcGSDFHdl);
                }

                // fix the GSDF padding for multi-channels
                if (transpose_flag)
                {
                    // mult-channel inputs are all of U8 data type
                    if (srcTileStartsX[j] < 0)
                    {
                        // left border columns
                        fixMultiChannelPadding((uint8_t *)src, 0, -srcTileStartsX[j], channelNum, srcTileSizesY[i],
                                               srcTileLp[p], 1);
                    }
                    if (srcImageWidth[p] < srcTileStartsX[j] + srcTileSizesX[j])
                    {
                        // right border columns
                        fixMultiChannelPadding((uint8_t *)src, srcImageWidth[p] - srcTileStartsX[j],
                                               srcImageWidth[p] - 1 - srcTileStartsX[j], channelNum, srcTileSizesY[i],
                                               srcTileLp[p], 0);
                    }
                }

                // update sampler index offsets for each tile
                int32_t ofstCol = transpose_flag ? srcOffsetsY[i] : srcOffsetsX[j];
                int32_t ofstRow = transpose_flag ? srcOffsetsX[j] : srcOffsetsY[i];
                UPDATE_SAMPLER_OFFSETS(cfg, sampler, ofstCol, ofstRow);

                // update ping-pong buffer address for sampler if without pre/post processing
                if (!cfg._preprocess_flag)
                {
                    UPDATE_SAMPLER_ADDRS(cfg, sampler, src, dst);
                }

                for (int c = 0; c < channelNum; c++)
                {

                    // pre-processing
                    if (cfg._preprocess_flag)
                    {
                        cupvaModifyAgenCfgBase(&cfg._src_agen_cfg, src + c);
                        PRE_KNL_EXEC(cfg);
                    }

                    cupvaSamplerStart(&sampler);
                    cupvaSamplerWait();

                    // post-processing
                    if (cfg._preprocess_flag)
                    {
                        cupvaModifyAgenCfgBase(&cfg._dst_agen_cfg, dst + c);
                        POST_KNL_EXEC(cfg);
                    }
                }

                cupvaRasterDataFlowRelease(dstRDFHdl);
            }
        }

        // close data flows
        cupvaGSDFClose(srcGSDFHdl);
        cupvaRasterDataFlowClose(dstRDFHdl);
    }

    return 0;
}
