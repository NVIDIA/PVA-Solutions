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

/**
 * @file histogram_equalization_top.c
 *
 * @brief VMEM declarations and the task entry point. One submit runs the whole
 *        frame: count every tile, build the lookup table, then equalize every
 *        tile.
 */

#include "histogram_equalization_knl.h"
#include "histogram_equalization_param.h"

#include <cupva_device.h>

/*
 * VMEM bank plan. Every cycle of both hot loops touches three different
 * superbanks once each, so no read pair contends for a single bank's read port
 * (each superbank serves one read and one write per cycle in parallel).
 *
 *   A  input tiles, both passes           DMA writes / compute reads
 *   B  histogram, output tile, scan       histogram RMW; then compute writes
 *                                         the output tile while DMA reads it
 *   C  lookup table and column masks      compute reads only
 *
 * The histogram and the output tile share bank B because they are used in
 * disjoint phases: the histogram is finished and drained before the first
 * equalized tile is produced.
 */
VMEM(A, uint8_t, src_v, RDF_DOUBLE(uint8_t, HEQ_TW_MAX, HEQ_TH_MAX));

VMEM(B, uint8_t, dst_v, RDF_DOUBLE(uint8_t, HEQ_TW_MAX, HEQ_TH_MAX));
VMEM(B, int16_t, hist, HEQ_TABLE_ELEMS);
VMEM(B, int32_t, accum, HEQ_SCAN_ELEMS);
VMEM(B, int32_t, accumAlt, HEQ_SCAN_ELEMS);

VMEM(C, uint16_t, lut32, HEQ_TABLE_ELEMS);
VMEM(C, uint16_t, lut, HEQ_NBINS);
VMEM(C, int16_t, maskFull, HEQ_TW_MAX);
VMEM(C, int16_t, maskLast, HEQ_TW_MAX);

VMEM(A, int32_t, niterX);
VMEM(A, int32_t, niterY);
VMEM(A, int32_t, imageWidth);
VMEM(A, int32_t, imageHeight);
VMEM(A, int32_t, tileWidth);
VMEM(A, int32_t, tileHeight);

VMEM(A, HistogramEqualizationConfig, cfg);

/*
 * Three RasterDataFlow streams, each double-buffered: the image is read once to build the
 * lookup table and again to apply it, and written once. RDF_DOUBLE on the output as well as the
 * inputs is what lets tile N's write-out overlap tile N+1's compute.
 */
VMEM_RDF_UNIFIED(A, srcHdl);
VMEM_RDF_UNIFIED(A, src2Hdl);
VMEM_RDF_UNIFIED(B, dstHdl);

CUPVA_VPU_MAIN()
{
    const int32_t tw = tileWidth;
    const int32_t th = tileHeight;

    /*
     * Rows of the last tile row, and valid columns of the last tile column. The
     * short tile row is handled by shortening the issue count; the narrow tile
     * column by the zero-weighted mask, since the histogram must not count the
     * stale tile-buffer bytes past the image edge.
     */
    const int32_t thLast     = (imageHeight % th) ? (imageHeight % th) : th;
    const int32_t lastColumn = (imageWidth % tw) ? (imageWidth % tw) : tw;

    histogram_equalizationInitAgens(&src_v[0], &maskFull[0], &hist[0], &accum[0], &accumAlt[0], &lut[0], &lut32[0], tw,
                                    th, thLast, tw, &cfg);

    histogram_equalization_gen_mask(&maskFull[0], &maskLast[0], tw, lastColumn);

    /* --- Pass 1: count every active sample ------------------------------- */

    cupvaRasterDataFlowOpen(srcHdl, &src_v[0]);

    int32_t issuesSinceDrain = 0;

    for (int32_t y = 0; y < niterY; y++)
    {
        cfg.histNiter = (y == (niterY - 1)) ? cfg.histIssuesLast : cfg.histIssues;

        for (int32_t x = 0; x < niterX; x++)
        {
            void *inTile = cupvaRasterDataFlowAcquire(srcHdl);

            histogram_equalization_knl_hist(&cfg, (uint8_t *)inTile, (x == (niterX - 1)) ? &maskLast[0] : &maskFull[0]);

            cupvaRasterDataFlowRelease(srcHdl);

            /*
             * Each issue adds 1 to one bin per way, so no bin of any way can
             * exceed the issues since the last drain. Fold the ways into the
             * 32-bit counts before that reaches the signed 16-bit bin limit.
             */
            issuesSinceDrain += cfg.histNiter;
            if (issuesSinceDrain >= HEQ_DRAIN_ISSUES)
            {
                histogram_equalization_knl_drain(&cfg);
                issuesSinceDrain = 0;
            }
        }
    }

    cupvaRasterDataFlowClose(srcHdl);

    if (issuesSinceDrain > 0)
    {
        histogram_equalization_knl_drain(&cfg);
    }

    /* --- Build the lookup table ----------------------------------------- */

    histogram_equalization_knl_scan(&cfg);

    cfg.numSamples = imageWidth * imageHeight;
    cfg.delta      = histogram_equalization_knl_delta(&cfg);

    /*
     * scale = (255 << 15) / (numSamples - delta), matching the reference's integer division.
     * A constant image makes delta == numSamples; clamp the divisor to avoid /0.
     */
    const int32_t denom = cfg.numSamples - cfg.delta;
    cfg.scale           = HEQ_SCALE_NUM / ((denom > 0) ? denom : 1);

    histogram_equalization_knl_lut(&cfg);
    histogram_equalization_knl_replicate(&cfg);

    /* --- Pass 2: map every pixel through the table ----------------------- */

    cupvaRasterDataFlowOpen(src2Hdl, &src_v[0]);
    cupvaRasterDataFlowOpen(dstHdl, &dst_v[0]);

    for (int32_t y = 0; y < niterY; y++)
    {
        cfg.applyNiter = (y == (niterY - 1)) ? cfg.applyIssuesLast : cfg.applyIssues;

        for (int32_t x = 0; x < niterX; x++)
        {
            void *inTile  = cupvaRasterDataFlowAcquire(src2Hdl);
            void *outTile = cupvaRasterDataFlowAcquire(dstHdl);

            histogram_equalization_knl_apply(&cfg, (uint8_t *)inTile, (uint8_t *)outTile);

            cupvaRasterDataFlowRelease(src2Hdl);
            cupvaRasterDataFlowRelease(dstHdl);
        }
    }

    cupvaRasterDataFlowClose(src2Hdl);
    cupvaRasterDataFlowClose(dstHdl);

    return 0;
}
