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

#include "range_fft_knl.h"
#include "range_fft_param.h"

#include <RadarDefinitions.h>
#include <fft_knl.h>

VMEM(A, int32_t, niterX);
VMEM(A, int32_t, niterY);
// Selects which FFT path to run: 0 = real input, 1 = complex input.
VMEM(A, int32_t, inputIsComplex);
/* Chirp blocks packed into one DMA tile: the tile carries groupsPerTile blocks of TILE_BATCH chirps,
 * run as a single fused FFT. 1 = one block per tile. */
VMEM(A, int32_t, groupsPerTile);
/* 1 = RANGE_RX_DOPPLER output: chirps are the innermost tensor axis, so the final store is
 * contiguous and the VMEM tile is [bin][chirp] instead of [chirp][bin]. */
VMEM(A, int32_t, outChirpInnermost);
/* Rows in one output tile: TILE_BATCH * groupsPerTile for the range-innermost layouts; for the
 * chirp-innermost one, all N/2+1 bins (the arena height), a tile row being a bin rather than a chirp. */
VMEM(A, int32_t, outTileRows);

VMEM(A, FFTFactorization, factorization);
VMEM(A, int32_t, twiddleBuf, TWIDDLE_SIZE);
VMEM(A, int32_t, windowBuf, WINDOW_SIZE);

VMEM(B, int32_t, tmpBuf, TILE_WIDTH *TILE_HEIGHT);
/* Added 2nd tmpBuf for post processing stage to avoid overwriting the src buffer when doing
  in-place input buffer update for symmetry */
VMEM(B, int32_t, tmpBuf2, TILE_WIDTH *TILE_HEIGHT);
VMEM(A, int32_t, srcBuf, RDF_DOUBLE(int32_t, IN_TILE_WIDTH, IN_TILE_HEIGHT, 0, 0, TRANS_MODE_2));
VMEM(C, int32_t, dstBuf, RDF_DOUBLE(int32_t, OUT_TILE_WIDTH, OUT_TILE_HEIGHT, 0, 0, TRANS_MODE_2));

VMEM_SEQDF_HANDLER(A, coeffHdl);
VMEM_RDF_UNIFIED(A, srcHdl);
VMEM_RDF_UNIFIED(A, dstHdl);
VMEM_SEQDF_HANDLER(A, dstSqdfHdl);
VMEM(A, int32_t, useRxDrSqdf);
VMEM(A, int32_t, dstLinePitchVar);
VMEM(A, int32_t, outLaneOfstVar);

VMEM(A, param_fft_batched_t, params_fft, 2);

CUPVA_VPU_MAIN()
{
    cupvaSQDFOpen(coeffHdl);
    cupvaSQDFTrig(coeffHdl);

    const int32_t srcLinePitch = cupvaRasterDataFlowGetLinePitch(srcHdl);

    // RX_DOPPLER_RANGE SQDF path: dstHdl has no RDF configured, so use the host-computed values.
    // RDF path: query dstHdl as usual.
    const int32_t dstLinePitch = useRxDrSqdf ? dstLinePitchVar : cupvaRasterDataFlowGetLinePitch(dstHdl);
    // The chirp-innermost output RDF is not transposed, so the lane offset is 0.
    int32_t output_lo = 0;
    if (useRxDrSqdf != 0)
    {
        output_lo = outLaneOfstVar;
    }
    else if (outChirpInnermost == 0)
    {
        output_lo = cupvaTransposeLaneOfst(dstHdl, TRANS_MODE_2, sizeof(int32_t));
    }

    // Chirp blocks are TILE_BATCH line pitches apart; lane_ofst addresses chirps within a block.
    const int32_t input_lo       = cupvaTransposeLaneOfst(srcHdl, TRANS_MODE_2, sizeof(int32_t));
    const int32_t srcGroupStride = IN_TILE_HEIGHT * srcLinePitch;
    /* Chirp-innermost: groups sit side by side within a bin row, TILE_WIDTH ints apart.
     * Range-innermost: groups are the next block of TILE_BATCH rows. */
    const int32_t dstGroupStride = (outChirpInnermost != 0) ? TILE_WIDTH : (OUT_TILE_HEIGHT * dstLinePitch);
    // Ping-pong half sizes follow the DMA tile layout.
    const int32_t inTileBufSize  = IN_TILE_HEIGHT * groupsPerTile * srcLinePitch;
    const int32_t outTileBufSize = outTileRows * dstLinePitch;

    for (int32_t i = 0; i < 2; i++) chess_unroll_loop(*)
    {
        params_fft[i].factorization       = &factorization;
        params_fft[i].windowBuffer        = (int32_t *)windowBuf;
        params_fft[i].twiddleTable[0]     = (int32_t *)twiddleBuf;                                   // twiddle3Buf
        params_fft[i].twiddleTable[1]     = (int32_t *)twiddleBuf + factorization.twiddleOffsets[0]; // twiddle4Buf
        params_fft[i].twiddleTable[2]     = (int32_t *)twiddleBuf + factorization.twiddleOffsets[1]; // twiddle5Buf
        params_fft[i].window_shift_qbits  = WINDOW_QBITS - TRUNCATE_QBITS;
        params_fft[i].twiddle_shift_qbits = TWIDDLE_QBITS - TRUNCATE_QBITS;
        params_fft[i].input_lane_offset   = input_lo;
        params_fft[i].output_lane_offset  = output_lo;
        params_fft[i].input_is_complex    = inputIsComplex;
        params_fft[i].out_chirp_innermost = outChirpInnermost;

        if (inputIsComplex == 0)
        {
            params_fft[i].twiddleTable[3] =
                (int32_t *)twiddleBuf + factorization.twiddleOffsets[2]; // twiddle post processBuf
        }
    }

    if (inputIsComplex == 0)
    {
        fft_batched_int_init(srcBuf, tmpBuf2, dstBuf, tmpBuf, &params_fft[0], groupsPerTile, srcGroupStride,
                             dstGroupStride, dstLinePitch);
        fft_batched_int_init(srcBuf + inTileBufSize, tmpBuf2, dstBuf + outTileBufSize, tmpBuf, &params_fft[1],
                             groupsPerTile, srcGroupStride, dstGroupStride, dstLinePitch);
    }
    else
    {
        fft_batched_int_init(srcBuf, srcBuf, dstBuf, tmpBuf, &params_fft[0], groupsPerTile, srcGroupStride,
                             dstGroupStride, dstLinePitch);
        fft_batched_int_init(srcBuf + inTileBufSize, srcBuf + inTileBufSize, dstBuf + outTileBufSize, tmpBuf,
                             &params_fft[1], groupsPerTile, srcGroupStride, dstGroupStride, dstLinePitch);
    }

    cupvaSQDFSync(coeffHdl);
    cupvaSQDFClose(coeffHdl);

    cupvaRasterDataFlowOpen(srcHdl, &srcBuf[0]);

    if (useRxDrSqdf)
    {
        const int32_t totalTiles = niterX * niterY;

        cupvaSQDFOpen(dstSqdfHdl);

        for (int32_t i = 0; i < totalTiles; i++)
        {
            cupvaRasterDataFlowAcquire(srcHdl);
            // Sync previous output DMA (wait for tile i-1 transfer to complete)
            if (i > 0)
            {
                cupvaSQDFSync(dstSqdfHdl);
            }

            // FFT: reads srcBuf[i&1], writes dstBuf[i&1]
            fft_batched_int_exec(&params_fft[i & 1]);

            // Trigger output DMA for current tile (overlaps the next iteration's input acquire)
            cupvaSQDFTrig(dstSqdfHdl);

            // Release current input slot (the next tile is acquired at the top of the next iteration)
            cupvaRasterDataFlowRelease(srcHdl);
        }

        // Drain: wait for the last output DMA
        cupvaSQDFSync(dstSqdfHdl);

        cupvaSQDFClose(dstSqdfHdl);
    }
    else
    {
        cupvaRasterDataFlowOpen(dstHdl, &dstBuf[0]);

        for (int32_t i = 0; i < niterX * niterY; i++)
        {
            cupvaRasterDataFlowAcquire(srcHdl);
            cupvaRasterDataFlowAcquire(dstHdl);

            fft_batched_int_exec(&params_fft[i & 1]);

            cupvaRasterDataFlowRelease(srcHdl);
            cupvaRasterDataFlowRelease(dstHdl);
        }

        cupvaRasterDataFlowClose(dstHdl);
    }

    cupvaRasterDataFlowClose(srcHdl);

    return 0;
}
