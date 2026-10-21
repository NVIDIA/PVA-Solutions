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

#include "radar_transpose_knl.h"
#include "radar_transpose_param.h"

VMEM(A, int32_t, nRange);
VMEM(A, int32_t, nRx);
VMEM(A, int32_t, nDoppler);
VMEM(A, int32_t, nIter);
VMEM(A, int32_t, chunkRange);
VMEM(A, int32_t, outputLayout);
VMEM(A, int32_t, useVectorizedRdx);

VMEM(A, param_transpose_tile_rdx_t, rdxParam, 2);
VMEM(A, param_transpose_chunk_drx_t, drxParam, 2);

/* scatterBuf: single padded intermediate for DRxR vectorized scatter (bank A).
 * Compaction reads scatterBuf (bank A) and writes dstBuf[cur] (bank C) — no bank conflict. */
VMEM(A, int32_t, scatterBuf, MAX_TILE_SIZE);

VMEM(B, int32_t, srcBuf, MAX_TILE_SIZE * 2);
VMEM(C, int32_t, dstBuf, MAX_TILE_SIZE * 2);

VMEM_SEQDF_HANDLER(A, srcHdl);
VMEM_SEQDF_HANDLER(A, dstHdl);

CUPVA_VPU_MAIN()
{
    const int32_t totalIter    = nIter;
    const int32_t curNRx       = nRx;
    const int32_t curNDoppler  = nDoppler;
    const int32_t curChunk     = chunkRange;
    const int32_t layout       = outputLayout;
    const int32_t vectorizeRdx = useVectorizedRdx;

    cupvaSQDFOpen(srcHdl);
    cupvaSQDFOpen(dstHdl);

    /* Initialize agen configs once before the tile loop, for both buffer slots. */
    if (layout == RADAR_TRANSPOSE_OUTPUT_LAYOUT_RANGE_DOPPLER_RX)
    {
        transpose_tile_rdx_init(srcBuf, dstBuf, &rdxParam[0], curNRx, curNDoppler, vectorizeRdx);
        transpose_tile_rdx_init(srcBuf + MAX_TILE_SIZE, dstBuf + MAX_TILE_SIZE, &rdxParam[1], curNRx, curNDoppler,
                                vectorizeRdx);
    }
    else /* DOPPLER_RX_RANGE */
    {
        /* scatterBuf (bank A) is the shared padded scatter intermediate for both slots.
         * dstBuf[0/1] (bank C) are the compact output buffers read by the dst DMA. */
        transpose_chunk_drx_init(srcBuf, scatterBuf, dstBuf, &drxParam[0], curChunk, curNRx, curNDoppler);
        transpose_chunk_drx_init(srcBuf + MAX_TILE_SIZE, scatterBuf, dstBuf + MAX_TILE_SIZE, &drxParam[1], curChunk,
                                 curNRx, curNDoppler);
    }

    /* Pre-trigger first tile into srcBuf[0]. */
    cupvaSQDFTrig(srcHdl);

    for (int32_t i = 0; i < totalIter; i++)
    {
        const int32_t cur = i & 1;

        /* 1. Wait for tile i (i==0 was pre-triggered; others synced at step 4 below). */
        if (i == 0)
            cupvaSQDFSync(srcHdl);

        /* 2. Pre-trigger tile i+1 into srcBuf[1-cur] — overlaps compute below. */
        if (i < totalIter - 1)
            cupvaSQDFTrig(srcHdl);

        /* 3. Compute: srcBuf[cur] → dstBuf[cur].
         *    src DMA for tile i+1 (step 2) runs concurrently here. */
        if (layout == RADAR_TRANSPOSE_OUTPUT_LAYOUT_RANGE_DOPPLER_RX)
            transpose_tile_rdx(srcBuf + cur * MAX_TILE_SIZE, dstBuf + cur * MAX_TILE_SIZE, curNRx, curNDoppler,
                               &rdxParam[cur]);
        else
            transpose_chunk_drx(srcBuf + cur * MAX_TILE_SIZE, scatterBuf, dstBuf + cur * MAX_TILE_SIZE, curChunk,
                                curNRx, curNDoppler, &drxParam[cur]);

        /* 4. Ensure tile i+1 is ready in srcBuf[1-cur] before next iteration reads it. */
        if (i < totalIter - 1)
            cupvaSQDFSync(srcHdl);

        /* 5. Wait for the store of tile i-1; destination SQDF stores are serialized. */
        if (i > 0)
            cupvaSQDFSync(dstHdl);

        /* 6. Start storing dstBuf[cur] — overlaps with next iteration's compute. */
        cupvaSQDFTrig(dstHdl);
    }

    /* Flush the last pending dst store. */
    cupvaSQDFSync(dstHdl);

    cupvaSQDFClose(srcHdl);
    cupvaSQDFClose(dstHdl);

    return 0;
}
