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

/*
 * DepthToSpace top level -- one program walks every tile of one output row offset: BLK_SZ input RDFs +
 * 1 output RDF. Selecting "every BLK_SZ-th output row" is the OUTPUT DATAFLOW's job, so the
 * kernel just fills a dense tile. See doc/impl/operator/depthtospace.rst.
 *
 * Bank plan: input tiles alternate across superbanks A and B by bx, output tile and scalars
 * on C. A superbank serves one read and one write per cycle in parallel but serializes two
 * reads (PG 5.1-5.2). The operator does zero arithmetic and is DMA-bound.
 */

#include "depthtospace_rdf_knl.h"

VMEM_RDF_UNIFIED(C, srcHandler);
VMEM_RDF_UNIFIED(C, dstHandler);

VMEM(C, int32_t, niterX);
VMEM(C, int32_t, niterY);
VMEM(C, DepthToSpaceConfig, cfg);

/*
 * TRANS_MODE_1 pads the VMEM tile line pitch to 64k + 2 bytes, the pitch a byte-type
 * transposed load/store demands. Must match the .transpose(TRANS_MODE_1) the host sets.
 */
VMEM(A, uint8_t, srcBuf0, RDF_DOUBLE(uint8_t, TILE_W, TILE_H, 0, 0, TRANS_MODE_1));
VMEM(B, uint8_t, srcBuf1, RDF_DOUBLE(uint8_t, TILE_W, TILE_H, 0, 0, TRANS_MODE_1));
#if BLK_SZ >= 3
VMEM(A, uint8_t, srcBuf2, RDF_DOUBLE(uint8_t, TILE_W, TILE_H, 0, 0, TRANS_MODE_1));
#endif
#if BLK_SZ >= 4
VMEM(B, uint8_t, srcBuf3, RDF_DOUBLE(uint8_t, TILE_W, TILE_H, 0, 0, TRANS_MODE_1));
#endif

/* Widest buffer, and stores-only, so it gets superbank C to itself. */
VMEM(C, uint8_t, dstBuf, RDF_DOUBLE(uint8_t, OUT_TILE_W, OUT_TILE_H, 0, 0, TRANS_MODE_1));

CUPVA_VPU_MAIN()
{
    /* Opened before the tile setup so the DMA gets the earliest chance at the first input
     * tile. cupvaRasterDataFlowOpen() promises no timing, and dstHandler is a write flow with
     * nothing to fetch, so this only ever helps src. */
    cupvaRasterDataFlowOpen(srcHandler, &srcBuf0[0]);
    cupvaRasterDataFlowOpen(dstHandler, &dstBuf[0]);

    uint8_t *srcBufs[NUM_SRC_BUFFERS] = {
        &srcBuf0[0],
        &srcBuf1[0],
#if BLK_SZ >= 3
        &srcBuf2[0],
#endif
#if BLK_SZ >= 4
        &srcBuf3[0],
#endif
    };

    int32_t srcLinePitch = cupvaRasterDataFlowGetLinePitch(srcHandler);
    int32_t dstLinePitch = cupvaRasterDataFlowGetLinePitch(dstHandler);

    int32_t niter = EXEC_NITER;

    depthtospace_agen_init(srcLinePitch, dstLinePitch, &cfg);

    uint8_t *srcTiles[BLK_SZ];

    for (int32_t y = 0; y < niterY; y++)
    {
        for (int32_t x = 0; x < niterX; x++)
        {
            uint8_t *srcTile0 = (uint8_t *)cupvaRasterDataFlowAcquire(srcHandler);
            uint8_t *dstTile  = (uint8_t *)cupvaRasterDataFlowAcquire(dstHandler);

            /*
             * The BLK_SZ input RDFs share one handler and geometry, so their ping-pong parity is
             * identical: recover each sibling tile by applying the first tile's offset to its own base.
             */
            ptrdiff_t offset = srcTile0 - &srcBuf0[0];

            /* BLK_SZ is compile-time; enumerated so no scalar loop reaches the VPU. */
            srcTiles[0] = srcBufs[0] + offset;
            srcTiles[1] = srcBufs[1] + offset;
#if BLK_SZ >= 3
            srcTiles[2] = srcBufs[2] + offset;
#endif
#if BLK_SZ >= 4
            srcTiles[3] = srcBufs[3] + offset;
#endif

            depthtospace_agen_update(srcTiles, dstTile, &cfg);
            depthtospace_exec(niter, &cfg);

            cupvaRasterDataFlowRelease(srcHandler);
            cupvaRasterDataFlowRelease(dstHandler);
        }
    }

    cupvaRasterDataFlowClose(srcHandler);
    cupvaRasterDataFlowClose(dstHandler);

    return 0;
}
