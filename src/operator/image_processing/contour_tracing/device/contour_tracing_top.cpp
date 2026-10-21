/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "pvaAplConv2dVpu.hpp"

#include "contour_tracing_knl.h"

#include <cupva_device.h>
#include <stdint.h>

// Handlers for DMA operations
VMEM_SEQDF_HANDLER(A, inHdl);
VMEM_SEQDF_HANDLER(A, outHdl);
VMEM_RDF_UNIFIED(A, rectHdl);
VMEM_RDF_UNIFIED(A, lenHdl);
VMEM_POINTER(A, inExtPtr);
VMEM_POINTER(A, outExtPtr);

VMEM(A, int32_t, numRects);
VMEM(A, int32_t, inLinePitch);
VMEM(A, int32_t, maxContourPixels);
VMEM(A, AgenCFG, cfgs, CFG_NUM);

// 1. Generate bitmap: inBuf -> (conv2dEvenBuf, conv2dOddBuf) -> bitmapBuf
// 2. Find start: inBuf + bitmapBuf -> startIndices
// 3. Trace contour: bitmapBuf + dirTable -> outBuf

VMEM(A, int16_t, startIndices, 16);
VMEM(A, uint16_t, segLens, 16);
VMEM(A, uint8_t, conv2dKnl, CONV_KWA *CONV_KWA * 2);
VMEM(A, BoundingRect, rectBuf, RDF_SINGLE(BoundingRect, MAX_RECT_NUM, 1));
VMEM(A, uint16_t, lenBuf, RDF_SINGLE(uint16_t, MAX_RECT_NUM, 1));
VMEM(A, uint8_t, conv2dOddBuf, CONV_BUF_SIZE);                   // 130 * 65 = 8.25KB
VMEM(A, uint8_t, bitmapBuf, MAX_RECT_WIDTH *CONV_OUTPUT_HEIGHT); // 128 * 132 = 16.5KB
VMEM(A, uint8_t, bitmap4WayBuf, MAX_RECT_AREA * 4);              // 128 * 128 * 4 = 64KB

VMEM(B, uint16_t, dirTable, NUM_NEIGHBOR_PERM);         // 256 * U16 = 0.5KB
VMEM(B, uint16_t, dirTable4Way, NUM_NEIGHBOR_PERM * 4); // 256 * U16 * 4 = 2KB
VMEM(B, uint8_t, inBuf, MAX_RECT_AREA * 2);             // 128 * 128 * pingpong = 32KB
VMEM(B, uint8_t, conv2dEvenBuf, CONV_BUF_SIZE);         // 130 * 65 = 8.25KB
VMEM(B, uint16_t, seg2Buf, MAX_RECT_AREA);              // 128 * 128 / 4 * 2 * U16 = 16KB
VMEM(B, uint16_t, seg3Buf, MAX_RECT_AREA);              // 128 * 128 / 4 * 2 * U16 = 16KB

VMEM(C, uint16_t, outBuf, MAX_RECT_AREA * 2 * 2); // 128 * 128 * 2 * pingpong * U16 = 128KB

CUPVA_VPU_MAIN()
{
    *((vchar *)conv2dKnl) = extract(zero_extend(*(vuchar *)BITMAP_KNL_REFORMATTED));

    // Transfer bounding rectangles
    cupvaRasterDataFlowOpen(rectHdl, &rectBuf[0]);
    cupvaRasterDataFlowAcquire(rectHdl);
    cupvaRasterDataFlowRelease(rectHdl);
    cupvaRasterDataFlowClose(rectHdl);

    cupvaSQDFOpen(inHdl);
    cupvaSQDFOpen(outHdl);

    int32_t pingPong     = 0;
    int32_t inPitchBytes = inLinePitch * sizeof(uint8_t);
    uint64_t inDramBase  = inExtPtr.base + inExtPtr.offset;
    uint64_t outDramBase = outExtPtr.base + outExtPtr.offset;

    // Transfer input for the first ROI
    BoundingRect roi       = rectBuf[0];
    uint16_t roiWidthBytes = roi.width * sizeof(uint8_t);

    uint8_t *input   = &inBuf[0];
    uint16_t *output = &outBuf[0];
    uint64_t inHost  = inDramBase + (roi.y * inLinePitch + roi.x) * sizeof(uint8_t);
    uint64_t outHost = outDramBase;

    cupvaSQDFUpdateAddr(inHdl, 0, inHost, MEMTYPE_DRAM, inPitchBytes, cupvaGetVmemAddress(input), MEMTYPE_VMEM,
                        roiWidthBytes);
    cupvaSQDFUpdateTileSize(inHdl, 0, roiWidthBytes, roi.height);
    cupvaSQDFFlushAndTrig(inHdl);

    fill_zero_init(cfgs, conv2dEvenBuf, conv2dOddBuf);
    find_start_pixel_init(cfgs);
    reformat_table_init(cfgs);
    trace_contour_init(cfgs);
    memcpy_u16_init(cfgs);

    // Reformat dirTable into 4-way parallel subtable layout
    cupvaModifyAgenCfgBase(&cfgs[REFORMAT_SRC], (uint8_t *)dirTable);
    cupvaModifyAgenCfgBase(&cfgs[REFORMAT_DST], (uint8_t *)dirTable4Way);
    reformat_table_exec(cfgs, NUM_NEIGHBOR_PERM * sizeof(uint16_t));

    for (int32_t i = 0; i < numRects; i++)
    {
        cupvaSQDFSync(inHdl);

        // Transfer input for the next ROI
        if (i != numRects - 1)
        {
            roi           = rectBuf[i + 1];
            roiWidthBytes = roi.width * sizeof(uint8_t);

            input  = &inBuf[(pingPong ^ 1) * MAX_RECT_AREA];
            inHost = inDramBase + (roi.y * inLinePitch + roi.x) * sizeof(uint8_t);

            cupvaSQDFUpdateAddr(inHdl, 0, inHost, MEMTYPE_DRAM, inPitchBytes, cupvaGetVmemAddress(input), MEMTYPE_VMEM,
                                roiWidthBytes);
            cupvaSQDFUpdateTileSize(inHdl, 0, roiWidthBytes, roi.height);
            cupvaSQDFFlushAndTrig(inHdl);
        }

        if (i > 1)
            cupvaSQDFSync(outHdl);

        // Transfer output for the previous ROI
        if (i > 0)
        {
            output  = &outBuf[(pingPong ^ 1) * MAX_RECT_AREA * 2];
            outHost = outDramBase + ((i - 1) * maxContourPixels * 2) * sizeof(uint16_t);
            cupvaSQDFUpdateAddr(outHdl, 0, cupvaGetVmemAddress(output), MEMTYPE_VMEM, outHost, MEMTYPE_DRAM);
            cupvaSQDFUpdateTileSize(outHdl, 0, lenBuf[i - 1] * 2 * sizeof(uint16_t), 1);
            cupvaSQDFFlushAndTrig(outHdl);
        }

        roi    = rectBuf[i];
        input  = &inBuf[pingPong * MAX_RECT_AREA];
        output = &outBuf[pingPong * MAX_RECT_AREA * 2];

        // 1. Generate 8-neighbor bitmap for every pixel in this ROI
        int32_t dstLp = roi.width <= 64 ? roi.width : 128;
        int32_t srcLp = roi.width + 2 * CONV_KWR;
        int32_t knlLp = CONV_KWA * 4;
        fill_zero_exec(cfgs, &roi);
        split_even_odd_exec(input, conv2dEvenBuf, conv2dOddBuf, &roi);

        int32_t cbLen = (roi.width * roi.height + 63) / 64 * 64;
        pvaApl::Conv2dVpu<CONV_KW, uint8_t, uint8_t> conv2d;
        conv2d.Init(conv2dEvenBuf, srcLp, conv2dOddBuf, srcLp, conv2dKnl, knlLp, bitmapBuf, dstLp, 0, roi.width,
                    roi.height, conv2dEvenBuf, conv2dOddBuf, cbLen);
        conv2d.Execute();

        if (dstLp != roi.width)
            repack_bitmap_exec(bitmapBuf, &roi);

        // 2. Find 4 start pixels: 2 on horizontal midline, 2 on vertical midline
        cupvaModifyAgenCfgBase(&cfgs[FIND_START_IN], &input[roi.width * roi.height / 2]);
        cupvaModifyAgenCfgBase(&cfgs[FIND_START_BM], &bitmapBuf[roi.width * roi.height / 2]);
        find_start_pixel_exec(cfgs, input, bitmapBuf, &roi, startIndices);

        // 3. Contour traversal
        cupvaModifyAgenCfgBase(&cfgs[REFORMAT_SRC], bitmapBuf);
        cupvaModifyAgenCfgBase(&cfgs[REFORMAT_DST], bitmap4WayBuf);
        reformat_table_exec(cfgs, roi.width * roi.height);
        if ((startIndices[0] | startIndices[1] | startIndices[2] | startIndices[3]) >= 0)
        {
            // The 4 lanes write into 4 different memory locations
            // vstore_i_wh with per-lane predicate writes halfwords at (base + 2 * segId)
            cupvaModifyAgenCfgBase(&cfgs[TRACE_SEG0], &output[0]);
            cupvaModifyAgenCfgBase(&cfgs[TRACE_SEG1], &output[MAX_RECT_AREA]);
            cupvaModifyAgenCfgBase(&cfgs[TRACE_SEG2], &seg2Buf[0]);
            cupvaModifyAgenCfgBase(&cfgs[TRACE_SEG3], &seg3Buf[0]);
            lenBuf[i] = trace_contour_exec(cfgs, bitmap4WayBuf, dirTable4Way, &roi, startIndices, segLens);

            // Append seg1, seg2, seg3 after seg0 via 3 memcpys
            uint16_t dstOffset = segLens[0] * 2;
            if (segLens[1] > 0)
            {
                cupvaModifyAgenCfgBase(&cfgs[MEMCPY_SRC], &output[MAX_RECT_AREA + 2]);
                cupvaModifyAgenCfgBase(&cfgs[MEMCPY_DST], &output[dstOffset]);
                memcpy_u16_exec(cfgs, segLens[1] * 2);
                dstOffset += segLens[1] * 2;
            }

            if (segLens[2] > 0)
            {
                cupvaModifyAgenCfgBase(&cfgs[MEMCPY_SRC], &seg2Buf[4]);
                cupvaModifyAgenCfgBase(&cfgs[MEMCPY_DST], &output[dstOffset]);
                memcpy_u16_exec(cfgs, segLens[2] * 2);
                dstOffset += segLens[2] * 2;
            }

            if (segLens[3] > 0)
            {
                cupvaModifyAgenCfgBase(&cfgs[MEMCPY_SRC], &seg3Buf[6]);
                cupvaModifyAgenCfgBase(&cfgs[MEMCPY_DST], &output[dstOffset]);
                memcpy_u16_exec(cfgs, segLens[3] * 2);
            }
        }

        pingPong ^= 1;
    }

    if (numRects > 1)
        cupvaSQDFSync(outHdl);

    // Transfer output for the last ROI
    output  = &outBuf[(pingPong ^ 1) * MAX_RECT_AREA * 2];
    outHost = outDramBase + ((numRects - 1) * maxContourPixels * 2) * sizeof(uint16_t);
    cupvaSQDFUpdateAddr(outHdl, 0, cupvaGetVmemAddress(output), MEMTYPE_VMEM, outHost, MEMTYPE_DRAM);
    cupvaSQDFUpdateTileSize(outHdl, 0, lenBuf[numRects - 1] * 2 * sizeof(uint16_t), 1);
    cupvaSQDFFlushAndTrig(outHdl);
    cupvaSQDFSync(outHdl);

    cupvaSQDFClose(inHdl);
    cupvaSQDFClose(outHdl);

    // Transfer contour lengths
    cupvaRasterDataFlowOpen(lenHdl, &lenBuf[0]);
    cupvaRasterDataFlowAcquire(lenHdl);
    cupvaRasterDataFlowRelease(lenHdl);
    cupvaRasterDataFlowClose(lenHdl);

    return 0;
}
