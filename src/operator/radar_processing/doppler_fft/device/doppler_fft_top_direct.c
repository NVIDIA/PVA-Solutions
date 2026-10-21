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

/* VPU entry point for the direct output layouts (DOPPLER_RX_RANGE, RANGE_RX_DOPPLER, RX_RANGE_DOPPLER),
 * which write the FFT output straight to DRAM. RANGE_RX_DOPPLER input arrives through a RasterDataFlow. */

#include "doppler_fft_knl.h"
#include "doppler_fft_param.h"

VMEM(A, int32_t, niterX);
VMEM(A, int32_t, niterY);
VMEM(A, int32_t, inTileBufSize);
VMEM(A, int32_t, outTileBufSize);
VMEM(A, int32_t, outLaneOfst);
VMEM(A, int32_t, outputLayout);
/* Wide-tile parameters: groupsPerTile range-bin groups per tile (1 = one group). */
VMEM(A, int32_t, groupsPerTile);
VMEM(A, int32_t, srcGroupStride);
VMEM(A, int32_t, dstGroupStride);
VMEM(A, int32_t, tilePitch);
/* Non-zero: run a tile's G groups as one fused call instead of G calls. */
VMEM(A, int32_t, fuseGroups);
/* 1 = RANGE_RX_DOPPLER input: windowing gathers each group's bins with a transposed load. */
VMEM(A, int32_t, inChirpInnermost);
VMEM(A, int32_t, inLaneOfst);
VMEM(A, int32_t, inRowPitch);

VMEM(A, FFTFactorization, factorization);
VMEM(A, int32_t, twiddleBuf, TWIDDLE_SIZE);
VMEM(A, int32_t, windowBuf, WINDOW_SIZE);

VMEM(A, int32_t, tmpBuf, TILE_WIDTH *TILE_HEIGHT);
VMEM(B, int32_t, srcBuf, RDF_DOUBLE(int32_t, IN_TILE_WIDTH, IN_TILE_HEIGHT));
VMEM(C, int32_t, dstBuf, RDF_DOUBLE(int32_t, OUT_TILE_WIDTH, OUT_TILE_HEIGHT));

VMEM_SEQDF_HANDLER(A, coeffHdl);
VMEM_SEQDF_HANDLER(A, srcHdl);
/* RANGE_RX_DOPPLER input arrives through a RasterDataFlow. */
VMEM_RDF_UNIFIED(A, srcRdfHdl);
VMEM_SEQDF_HANDLER(A, dstHdl);

VMEM_POINTER(A, srcExtPtr);
VMEM_POINTER(A, dstExtPtr);

VMEM(A, param_fft_batched_t, params_fft, 2 * TILE_GROUPS_MAX_UNFUSED);

/* Per tile: wait for the input, prefetch the next tile, then run the FFT. */
static inline void tileInputAndFFT(int32_t i, int32_t totalTiles)
{
    if (inChirpInnermost != 0)
    {
        /* RDF manages its own ping-pong and prefetch; acquire blocks until this tile has landed. */
        cupvaRasterDataFlowAcquire(srcRdfHdl);
    }
    else
    {
        // Sync current tile (i==0 was pre-triggered before loop, others synced at end of prev iteration)
        if (i == 0)
        {
            cupvaSQDFSync(srcHdl);
        }

        // Prefetch next tile (overlaps DMA with FFT computation)
        if (i < totalTiles - 1)
        {
            cupvaSQDFTrig(srcHdl);
        }
    }

    // === FFT COMPUTATION ===
    /* One FFT per range-bin group packed into this tile. */
    const int32_t nCalls    = (fuseGroups != 0) ? 1 : groupsPerTile;
    const int32_t fftGroups = (fuseGroups != 0) ? groupsPerTile : 1;
    for (int32_t g = 0; g < nCalls; g++)
    {
        fft_batched_int_exec(&params_fft[(i & 1) * nCalls + g], fftGroups);
    }

    if (inChirpInnermost != 0)
    {
        cupvaRasterDataFlowRelease(srcRdfHdl);
    }
    else
    {
        // Complete prefetch before next iteration uses that buffer
        if (i < totalTiles - 1)
        {
            cupvaSQDFSync(srcHdl);
        }
    }
}

/* Direct output path: every output layout but RANGE_DOPPLER_RX, written straight to DRAM. */
static void processTilesDirect()
{
    const int32_t totalTiles = niterX * niterY;

    // Pre-fetch first input tile before entering loop (SQDF only; RDF prefetches on open)
    if (inChirpInnermost == 0)
    {
        cupvaSQDFTrig(srcHdl);
    }

    for (int32_t x = 0; x < niterX; x++)
    {
        for (int32_t y = 0; y < niterY; y++)
        {
            int32_t i = x * niterY + y;

            // === INPUT + FFT ===
            tileInputAndFFT(i, totalTiles);

            // === OUTPUT STAGE ===
            if (i > 0)
            {
                cupvaSQDFSync(dstHdl); // Wait for previous output
            }
            cupvaSQDFTrig(dstHdl); // Output current tile to DRAM
        }
    }

    // === EPILOGUE ===
    cupvaSQDFSync(dstHdl); // Complete last output
}

CUPVA_VPU_MAIN()
{
    cupvaSQDFOpen(coeffHdl);
    cupvaSQDFTrig(coeffHdl);

    for (int32_t i = 0; i < 2 * TILE_GROUPS_MAX_UNFUSED; i++) chess_unroll_loop(*)
    {
        params_fft[i].window_shift_qbits  = WINDOW_QBITS - TRUNCATE_QBITS;
        params_fft[i].twiddle_shift_qbits = TWIDDLE_QBITS - TRUNCATE_QBITS;
        params_fft[i].o_lane_ofst         = outLaneOfst;
        params_fft[i].outputLayout        = outputLayout;
        params_fft[i].windowBuffer        = (int32_t *)windowBuf;
        params_fft[i].twiddleTable[0]     = (int32_t *)twiddleBuf;
        params_fft[i].twiddleTable[1]     = (int32_t *)twiddleBuf + factorization.twiddleOffsets[0];
        params_fft[i].twiddleTable[2]     = (int32_t *)twiddleBuf + factorization.twiddleOffsets[1];
        params_fft[i].factorization       = &factorization;
    }

    /* One agen-config set per (buffer half, group); tilePitch is the chirp stride. */
    const int32_t nCalls    = (fuseGroups != 0) ? 1 : groupsPerTile;
    const int32_t fftGroups = (fuseGroups != 0) ? groupsPerTile : 1;
    for (int32_t h = 0; h < 2; h++)
    {
        for (int32_t g = 0; g < nCalls; g++)
        {
            fft_batched_int_init(srcBuf + h * inTileBufSize + g * srcGroupStride,
                                 dstBuf + h * outTileBufSize + g * dstGroupStride, tmpBuf + g * srcGroupStride,
                                 &params_fft[h * nCalls + g], tilePitch, fftGroups, srcGroupStride, dstGroupStride,
                                 inChirpInnermost, inLaneOfst, inRowPitch);
        }
    }

    cupvaSQDFSync(coeffHdl);
    cupvaSQDFClose(coeffHdl);

    if (inChirpInnermost != 0)
    {
        cupvaRasterDataFlowOpen(srcRdfHdl, &srcBuf[0]);
    }
    else
    {
        cupvaSQDFOpen(srcHdl);
    }
    cupvaSQDFOpen(dstHdl);

    processTilesDirect();

    if (inChirpInnermost != 0)
    {
        cupvaRasterDataFlowClose(srcRdfHdl);
    }
    else
    {
        cupvaSQDFClose(srcHdl);
    }
    cupvaSQDFClose(dstHdl);

    return 0;
}
