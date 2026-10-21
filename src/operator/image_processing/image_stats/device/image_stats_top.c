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

#include "data_type_defs.h"
#include "image_stats_common.h"
#include "image_stats_param.h"

#include <cupva_device.h>

VMEM(A, int32_t, planeCount);
VMEM(A, uint8_t, planeChannelNum, MAX_PLANE_COUNT);
VMEM(A, uint16_t, planeWidth, MAX_PLANE_COUNT);
VMEM(A, uint16_t, planeHeight, MAX_PLANE_COUNT);
VMEM(A, uint8_t, xSampleRatio, MAX_PLANE_COUNT);
VMEM(A, uint8_t, ySampleRatio, MAX_PLANE_COUNT);
VMEM(A, uint16_t, planeLinePitch, MAX_PLANE_COUNT);
VMEM(A, int32_t, planePixelCount, MAX_PLANE_COUNT);
VMEM(A, uint8_t, ntileX);
VMEM(A, uint8_t, ntileY);
VMEM(A, uint16_t, tileWidth, MAX_PLANE_COUNT);
VMEM(A, uint16_t, tileHeight, MAX_PLANE_COUNT);
VMEM(A, uint16_t, tileLp, MAX_PLANE_COUNT);
VMEM(A, IMAGE_STATS_CFG, cfg);

// input data flows:
VMEM_RDF_UNIFIED(A, srcRDFHdl0);
VMEM_RDF_UNIFIED(A, srcRDFHdl1);
VMEM(B, DATA_TYPE, srcBuf0, RDF_DOUBLE(DATA_TYPE, TILE_WIDTH *MAX_CHANNEL_NUM, TILE_HEIGHT));
VMEM(C, DATA_TYPE, srcBuf1, RDF_DOUBLE(DATA_TYPE, TILE_WIDTH *MAX_CHANNEL_NUM, TILE_HEIGHT));

// mask data flow:
#if defined(MASK_IO)
VMEM_RDF_UNIFIED(A, maskRDFHdl);
VMEM(A, uint8_t, maskBuf, RDF_DOUBLE(uint8_t, TILE_WIDTH, TILE_HEIGHT));
#endif

// output data flow:
VMEM_SEQDF_HANDLER(A, dstSQDFHdl);
VMEM(A, uint8_t, outBuf, IMAGE_STAT_OUTPUT_SIZE);

// accumulator buffers
VMEM(A, uint64_t, perChannelSumU64Buf, 4);
VMEM(A, uint64_t, covarianceMatU64Buf, 16);

// flags
VMEM(A, uint32_t, pixelCountFlag);
VMEM(A, uint32_t, sumFlag);
VMEM(A, uint32_t, meanFlag);
VMEM(A, uint32_t, varianceFlag);
VMEM(A, uint32_t, covarianceFlag);

CUPVA_VPU_MAIN()
{
    // init output buffer base address
    int32_t *pixelCountBuf   = (int32_t *)outBuf;
    float *perChannelSumBuf  = (float *)(outBuf + 4);
    float *perChannelMeanBuf = (float *)(outBuf + 20);
    float *covarianceMatBuf  = (float *)(outBuf + 36);

    // open SQDF write data flow
    cupvaSQDFOpen(dstSQDFHdl);

#if !defined(SUM_IO) && !defined(MASK_IO)
    // for cnt only op without mask, directly return the pixel count
    if (pixelCountFlag)
    {
        pixelCountBuf[0] = planeWidth[0] * planeHeight[0];
    }

    // TrigSQDF:
    cupvaSQDFTrig(dstSQDFHdl);
    cupvaSQDFSync(dstSQDFHdl);
    cupvaSQDFClose(dstSQDFHdl);

    return 0;
#endif

    // define array of src pointers
    DATA_TYPE *srcPtrs[MAX_PLANE_COUNT];
    srcPtrs[0] = srcBuf0;
    if (planeCount > 1)
    {
        srcPtrs[1] = srcBuf1;
    }

    // open RDF read data flow
    cupvaRasterDataFlowOpen(srcRDFHdl0, &srcBuf0[0]);
    tileLp[0] = cupvaRasterDataFlowGetLinePitch(srcRDFHdl0);
    if (planeCount > 1)
    {
        cupvaRasterDataFlowOpen(srcRDFHdl1, &srcBuf1[0]);
        tileLp[1] = cupvaRasterDataFlowGetLinePitch(srcRDFHdl1);
    }

    uint8_t *maskPtr = NULL;
    uint16_t maskLp  = 0;
#if defined(MASK_IO)
    cupvaRasterDataFlowOpen(maskRDFHdl, &maskBuf[0]);
    maskLp  = cupvaRasterDataFlowGetLinePitch(maskRDFHdl);
    maskPtr = maskBuf;
#endif

    // init kernel function config
    INIT_IMAGE_STATS_CFG(cfg, srcPtrs, maskPtr, tileWidth, tileHeight, tileLp, maskLp);

    // init accumulator buffers and flags
    image_stats_init_common_params(cfg, &planePixelCount[0], &perChannelSumU64Buf[0], &covarianceMatU64Buf[0],
                                   pixelCountFlag, sumFlag, meanFlag, varianceFlag, covarianceFlag);

    for (int i = 0; i < ntileY; i++)
    {
        for (int j = 0; j < ntileX; j++)
        {
            srcPtrs[0] = (DATA_TYPE *)cupvaRasterDataFlowAcquire(srcRDFHdl0);
            if (planeCount > 1)
            {
                srcPtrs[1] = (DATA_TYPE *)cupvaRasterDataFlowAcquire(srcRDFHdl1);
            }

#if defined(MASK_IO)
            maskPtr = (uint8_t *)cupvaRasterDataFlowAcquire(maskRDFHdl);
            IMAGE_STATS_EXEC(cfg, srcPtrs, maskPtr);
            cupvaRasterDataFlowRelease(maskRDFHdl);
#else
            IMAGE_STATS_EXEC(cfg, srcPtrs);
#endif

            cupvaRasterDataFlowRelease(srcRDFHdl0);
            if (planeCount > 1)
            {
                cupvaRasterDataFlowRelease(srcRDFHdl1);
            }
        }
    }

    cupvaRasterDataFlowClose(srcRDFHdl0);
    if (planeCount > 1)
    {
        cupvaRasterDataFlowClose(srcRDFHdl1);
    }
#if defined(MASK_IO)
    cupvaRasterDataFlowClose(maskRDFHdl);
#endif

    // post-processing
    image_stats_normalize_exec(cfg, planeCount, planeChannelNum, planeWidth, planeHeight, pixelCountBuf,
                               perChannelSumBuf, perChannelMeanBuf, covarianceMatBuf);

    // TrigSQDF:
    cupvaSQDFTrig(dstSQDFHdl);
    cupvaSQDFSync(dstSQDFHdl);
    cupvaSQDFClose(dstSQDFHdl);

    return 0;
}
