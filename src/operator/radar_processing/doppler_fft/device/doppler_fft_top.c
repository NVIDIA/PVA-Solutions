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

#include "doppler_fft_knl.h"
#include "doppler_fft_param.h"

VMEM(A, int32_t, niterX);
VMEM(A, int32_t, niterY);
VMEM(A, int32_t, inTileBufSize);
VMEM(A, int32_t, outTileBufSize);
VMEM(A, int32_t, outTileWidth);
VMEM(A, int32_t, outTileHeight);
VMEM(A, int32_t, outLaneOfst);
// PVAFFTLayout value; the host passes RX_RANGE_DOPPLER as RANGE_RX_DOPPLER.
VMEM(A, int32_t, outputLayout);
VMEM(A, int32_t, drainCount);

VMEM(A, FFTFactorization, factorization);
VMEM(A, int32_t, twiddleBuf, TWIDDLE_SIZE);
VMEM(A, int32_t, windowBuf, WINDOW_SIZE);

VMEM(A, int32_t, tmpBuf, TILE_WIDTH *TILE_HEIGHT);
VMEM(B, int32_t, srcBuf, RDF_DOUBLE(int32_t, IN_TILE_WIDTH, IN_TILE_HEIGHT));
VMEM(C, int32_t, dstBuf, RDF_DOUBLE(int32_t, OUT_TILE_WIDTH, OUT_TILE_HEIGHT));

VMEM_SEQDF_HANDLER(A, coeffHdl);
VMEM_SEQDF_HANDLER(A, srcHdl);
VMEM_SEQDF_HANDLER(A, dstHdl);
VMEM_SEQDF_HANDLER(A, toL2Hdl);
VMEM_SEQDF_HANDLER(A, fromL2Hdl);

VMEM_POINTER(A, srcExtPtr);
VMEM_POINTER(A, dstExtPtr);

VMEM(A, param_fft_batched_t, params_fft, 2);

/* Per tile: wait for the input, prefetch the next tile, then run the FFT. */
static inline void tileInputAndFFT(int32_t i, int32_t totalTiles)
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

    // === FFT COMPUTATION ===
    fft_batched_int_exec(&params_fft[i & 1], 1);

    // Complete prefetch before next iteration uses that buffer
    if (i < totalTiles - 1)
    {
        cupvaSQDFSync(srcHdl);
    }
}

/* RANGE_DOPPLER_RX output: FFT -> L2 -> transpose -> DRAM, with L2 holding niterX tiles. */
static void processTilesRxPacked()
{
    const int32_t totalTiles = niterX * niterY;

    // Pre-fetch first input tile before entering loop
    cupvaSQDFTrig(srcHdl);

    for (int32_t x = 0; x < niterX; x++)
    {
        for (int32_t y = 0; y < niterY; y++)
        {
            int32_t i = x * niterY + y;

            // === INPUT + FFT ===
            tileInputAndFFT(i, totalTiles);

            // Fill L2 for the first niterX tiles, then output one tile per iteration.

            if (i > niterX)
            {
                cupvaSQDFSync(dstHdl); // Wait for previous DRAM output
            }

            cupvaSQDFTrig(toL2Hdl); // Send FFT result to L2

            if (i >= niterX)
            {
                // L2 has niterX tiles buffered - start draining
                cupvaSQDFTrig(fromL2Hdl); // Fetch oldest tile from L2
                cupvaSQDFSync(toL2Hdl);
                cupvaSQDFSync(fromL2Hdl);
                transpose_for_rx_packing(&params_fft[i & 1].transpose); // Rearrange Rx dim
                cupvaSQDFTrig(dstHdl);                                  // Output transposed tile to DRAM
            }
            else
            {
                cupvaSQDFSync(toL2Hdl); // Just wait for L2 write (fill phase)
            }
        }
    }

    // === EPILOGUE ===
    if (niterY > 1)
    {
        cupvaSQDFSync(dstHdl);
    }

    // Drain the drainCount tiles still in L2.
    int32_t drainBufIdx = totalTiles & 1;
    for (int32_t i = 0; i < drainCount; i++)
    {
        cupvaSQDFTrig(fromL2Hdl);
        cupvaSQDFSync(fromL2Hdl);
        transpose_for_rx_packing(&params_fft[drainBufIdx & 1].transpose);
        cupvaSQDFTrig(dstHdl);
        cupvaSQDFSync(dstHdl);
        drainBufIdx++;
    }
}

CUPVA_VPU_MAIN()
{
    cupvaSQDFOpen(coeffHdl);
    cupvaSQDFTrig(coeffHdl);

    for (int32_t i = 0; i < 2; i++) chess_unroll_loop(*)
    {
        params_fft[i].window_shift_qbits  = WINDOW_QBITS - TRUNCATE_QBITS;
        params_fft[i].twiddle_shift_qbits = TWIDDLE_QBITS - TRUNCATE_QBITS;
        params_fft[i].o_lane_ofst         = outLaneOfst;
        params_fft[i].outputLayout        = outputLayout;
        // The twiddleTable[] uses host-computed offsets relative to twiddleBuf starting address.
        params_fft[i].windowBuffer    = (int32_t *)windowBuf;
        params_fft[i].twiddleTable[0] = (int32_t *)twiddleBuf;                                   // twiddle3Buf
        params_fft[i].twiddleTable[1] = (int32_t *)twiddleBuf + factorization.twiddleOffsets[0]; // twiddle4Buf
        params_fft[i].twiddleTable[2] = (int32_t *)twiddleBuf + factorization.twiddleOffsets[1]; // twiddle5Buf
        params_fft[i].factorization   = &factorization;
    }

    fft_batched_int_init(srcBuf, dstBuf, tmpBuf, &params_fft[0], chess_elementsof(DVTYPEX), 1, 0, 0, 0, 0, 0);
    fft_batched_int_init(srcBuf + inTileBufSize, dstBuf + outTileBufSize, tmpBuf, &params_fft[1],
                         chess_elementsof(DVTYPEX), 1, 0, 0, 0, 0, 0);

    // Initialize transpose agens for both double-buffer slots
    const int32_t transposeHeight = niterX;
    const int32_t transposeWidth  = outTileWidth * outTileHeight / niterX;
    transpose_for_rx_packing_init(tmpBuf, dstBuf, &params_fft[0].transpose, transposeWidth, transposeHeight);
    transpose_for_rx_packing_init(tmpBuf, dstBuf + outTileBufSize, &params_fft[1].transpose, transposeWidth,
                                  transposeHeight);

    cupvaSQDFSync(coeffHdl);
    cupvaSQDFClose(coeffHdl);

    cupvaSQDFOpen(srcHdl);
    cupvaSQDFOpen(dstHdl);
    cupvaSQDFOpen(toL2Hdl);
    cupvaSQDFOpen(fromL2Hdl);

    processTilesRxPacked();

    cupvaSQDFClose(toL2Hdl);
    cupvaSQDFClose(fromL2Hdl);
    cupvaSQDFClose(srcHdl);
    cupvaSQDFClose(dstHdl);

    return 0;
}