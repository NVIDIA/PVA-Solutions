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

#include "blur_filter_roi_knl_u8.h"

VMEM_RDF_UNIFIED(A, srcHdl0);
VMEM_RDF_UNIFIED(A, dstHdl0);
VMEM(A, RasterDataFlowHandler, srcHdl1);
VMEM(A, RasterDataFlowHandler, dstHdl1);

VMEM_SEQDF_HANDLER(A, rectHdl);

VMEM(A, int32_t, numPlanes);
VMEM(A, int32_t, numTiles, MAX_PLANE_COUNT);
VMEM(A, int32_t, numRects);
VMEM(A, BlurFilterROIRect, rectBuf, MAX_RECT_COUNT);
VMEM(A, int32_t, blockW);
VMEM(A, int32_t, blockH);

VMEM(A, int32_t, imageH);
VMEM(A, int32_t, tileW);
VMEM(A, int32_t, tileH);
VMEM(A, uint8_t, dataBuf, RECT_W *MAX_RECT_COUNT);

VMEM(B, uint8_t, inBuf, RDF_DOUBLE(uint8_t, MAX_BUF_SIZE, 1));
VMEM(C, uint8_t, outBuf, RDF_DOUBLE(uint8_t, MAX_BUF_SIZE, 1));

static inline int32_t heightOfRectInsideTile(int32_t rectY0, int32_t rectY1, int32_t tileY0, int32_t tileY1)
{
    return min(rectY1, tileY1) - max(rectY0, tileY0);
}

static inline void blurRectangleByBlockContinue(uint8_t *dataBuf, uint8_t *outBuf, int32_t linePitch, int32_t rectW,
                                                int rectH, int32_t blockW)
{
    for (int32_t bx = 0, blockIdx = 0; bx < rectW; bx += blockW, blockIdx++)
    {
        int32_t w    = min(blockW, rectW - bx);
        uint8_t data = dataBuf[blockIdx];
        uint8_t *dst = &outBuf[bx];
        blur_filter_roi_knl_exec(w, rectH, linePitch, data, dst);
    }
}

static inline void blurRectangleByBlock(uint8_t *inBuf, uint8_t *outBuf, int32_t linePitch, int32_t rectW,
                                        int32_t rectH, int32_t blockW, int32_t blockH, bool isContinueBottom,
                                        uint8_t *dataBuf)
{
    for (int32_t by = 0; by < rectH; by += blockH)
    {
        bool isLastBlockRow = ((by + blockH) >= rectH);

        int32_t h = min(blockH, rectH - by);
        for (int32_t bx = 0, blockIdx = 0; bx < rectW; bx += blockW, blockIdx++)
        {
            int32_t w    = min(blockW, rectW - bx);
            uint8_t data = inBuf[by * linePitch + bx];
            uint8_t *dst = &outBuf[by * linePitch + bx];
            blur_filter_roi_knl_exec(w, h, linePitch, data, dst);

            if (isLastBlockRow && isContinueBottom)
            {
                dataBuf[blockIdx] = data;
            }
        }
    }
}

CUPVA_VPU_MAIN()
{
    // transfer ROI rectangles

    cupvaSQDFOpen(rectHdl);
    cupvaSQDFTrig(rectHdl);
    cupvaSQDFSync(rectHdl);
    cupvaSQDFClose(rectHdl);

    // execute blur filter operation only on Y plane

    cupvaRasterDataFlowOpen(srcHdl0, &inBuf[0]);
    cupvaRasterDataFlowOpen(dstHdl0, &outBuf[0]);

    int32_t linePitch = cupvaRasterDataFlowGetLinePitch(srcHdl0);

    for (int32_t t = 0; t < numTiles[0]; t++)
    {
        uint8_t *srcTile = (uint8_t *)cupvaRasterDataFlowAcquire(srcHdl0);
        uint8_t *dstTile = (uint8_t *)cupvaRasterDataFlowAcquire(dstHdl0);

        vmemcpy_2d(dstTile, linePitch, srcTile, linePitch, tileW, tileH);

        int32_t tileY0 = t * tileH;
        int32_t tileY1 = min(tileY0 + tileH, imageH);
        for (int32_t i = 0; i < numRects; i++)
        {
            int32_t rectY0 = rectBuf[i].y;
            int32_t rectY1 = rectY0 + rectBuf[i].height;

            int32_t rectH = heightOfRectInsideTile(rectY0, rectY1, tileY0, tileY1);
            if (rectH > 0)
            {
                int32_t offset   = max(rectY0 - tileY0, 0) * linePitch + rectBuf[i].x;
                uint8_t *srcRect = srcTile + offset;
                uint8_t *dstRect = dstTile + offset;
                uint8_t *dataPtr = &dataBuf[i * MAX_RECT_COUNT];

                int32_t rectW = rectBuf[i].width;
                int32_t lastH = (tileY0 - rectY0) % blockH;
                if (lastH > 0)
                {
                    int32_t continueH = min(blockH - lastH, rectH);
                    blurRectangleByBlockContinue(dataPtr, dstRect, linePitch, rectW, continueH, blockW);

                    srcRect += continueH * linePitch;
                    dstRect += continueH * linePitch;
                    rectH -= continueH;
                }
                if (rectH > 0)
                {
                    bool isContinueBottom = ((rectH % blockH) && (rectY1 > tileY1));
                    blurRectangleByBlock(srcRect, dstRect, linePitch, rectW, rectH, blockW, blockH, isContinueBottom,
                                         dataPtr);
                }
            }
        }

        cupvaRasterDataFlowRelease(srcHdl0);
        cupvaRasterDataFlowRelease(dstHdl0);
    }

    cupvaRasterDataFlowClose(srcHdl0);
    cupvaRasterDataFlowClose(dstHdl0);

    // copy UV plane of input image buffer to output image buffer

    cupvaRasterDataFlowTrig(srcHdl1);
    for (int32_t t = 0; t < numTiles[1]; t++)
    {
        cupvaRasterDataFlowSync(srcHdl1);
        cupvaRasterDataFlowTrig(srcHdl1);

        // don't use cupvaRasterDataFlowAcquire and cupvaRasterDataFlowRelease
        // since DMA dataflow's fine control is needed, otherwise will need
        // distinguished input and output buffers and extra VPU kernel for memcpy
        cupvaRasterDataFlowTrig(dstHdl1);
        cupvaRasterDataFlowSync(dstHdl1);
    }

    return 0;
}