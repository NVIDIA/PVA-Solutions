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

#include "gaussian_pyramid_knl.h"
#include "gaussian_pyramid_param.h"

VMEM(B, int32_t, niterY);
VMEM(B, int32_t, niterX);

VMEM(B, int32_t, imgWidth);
VMEM(B, int32_t, imgHeight);
VMEM(B, int32_t, totalLevels);

VMEM(C, uint16_t, weightBuffer, 3 * 128);

VMEM(A, BASE_TYPE, level0EvenBuffer,
     RDF_CIRCULAR(BASE_TYPE, L0_LINE_PITCH_ALIGN_VECW, TH0_SPLIT_BUF, HALO_L0, HALO_L0_Y_SPLIT_BUF));
VMEM(B, BASE_TYPE, level0OddBuffer,
     RDF_CIRCULAR(BASE_TYPE, L0_LINE_PITCH_ALIGN_VECW, TH0_SPLIT_BUF, HALO_L0, HALO_L0_Y_SPLIT_BUF));
VMEM(A, BASE_TYPE, level1EvenBuffer,
     RDF_DOUBLE(BASE_TYPE, L1_LINE_PITCH_ALIGN_VECW, L1_TILE_ARENA_Y) + L1_TILE_ADJUSTMENT_MAX);
VMEM(C, BASE_TYPE, level1OddBuffer,
     RDF_DOUBLE(BASE_TYPE, L1_LINE_PITCH_ALIGN_VECW, L1_TILE_ARENA_Y) + L1_TILE_ADJUSTMENT_MAX);
VMEM(B, BASE_TYPE, level2Buffer, RDF_DOUBLE(BASE_TYPE, L2_TILE_ARENA_X, L2_TILE_ARENA_Y));

VMEM_RDF_UNIFIED(B, level0HandlerEven);
VMEM_RDF_UNIFIED(B, level0HandlerOdd);
VMEM_RDF_UNIFIED(B, level1Handler);
VMEM_RDF_UNIFIED(B, level2Handler);

VMEM(C, uint32_t, padValue);
VMEM(C, AgenCFG, level0EvenInputCfg);
VMEM(C, AgenCFG, level0OddInputCfg);
VMEM(C, AgenCFG, level2EvenOutputCfg);
VMEM(C, AgenCFG, level2OddOutputCfg);

VMEM(C, uint16_t, gaussianKernel, (KW + 1) * (KW + 1));

// Inline function to set zero padding
static inline void memset_vpu_zero_pad(BASE_TYPE *dst, const int32_t width, const int32_t height)
{
    DIN_VEC_TYPE dvzero;
    dvzero.lo          = REPLICATE(padValue);
    dvzero.hi          = REPLICATE(padValue);
    const int32_t iter = (width + VECW - 1) / VECW;
    for (int32_t i = 0; i < height; i++)
    {
        for (int32_t k = 0; k < iter; k++)
        {
            *(IN_MEM_TYPE *restrict)(void *)(dst + (i * width) + (k * VECW)) = extract(dvzero);
        }
    }
}

// Inline function to apply border padding to the even and odd buffers
static inline void apply_border_padding(BASE_TYPE *evenBuffer, BASE_TYPE *oddBuffer, int32_t startAddress,
                                        int32_t linePitch, uint32_t padVal)
{
    for (int32_t h = 0; h < (TH1_SPLIT_BUF + HALO_L1); h++)
    {
        for (int32_t w = 0; w < KERNEL_HALO; w++)
        {
            evenBuffer[startAddress + h * linePitch + w] = padVal;
            oddBuffer[startAddress + h * linePitch + w]  = padVal;
        }
    }
}

int32_t image_pyramid_compute_1level(const int32_t niterY, const int32_t niterX, const int32_t level1ImageWidth,
                                     const int32_t level1ImageHeight, UnifiedRDFHandler &level0HandlerEven,
                                     UnifiedRDFHandler &level0HandlerOdd, UnifiedRDFHandler &level1Handler)
{
    cupvaRasterDataFlowOpen(level0HandlerEven, &level0EvenBuffer[0]);
    cupvaRasterDataFlowOpen(level0HandlerOdd, &level0OddBuffer[0]);
    cupvaRasterDataFlowOpen(level1Handler, &level1EvenBuffer[0]);

    int32_t level0LinePitchEven = cupvaRasterDataFlowGetLinePitch(level0HandlerEven);
    int32_t level0LinePitchOdd  = cupvaRasterDataFlowGetLinePitch(level0HandlerOdd);

    int32_t level1LinePitch = cupvaRasterDataFlowGetLinePitch(level1Handler);
    int32_t level0CbLenEven = cupvaRasterDataFlowGetCbLen(level0HandlerEven);
    int32_t level0CbLenOdd  = cupvaRasterDataFlowGetCbLen(level0HandlerOdd);

    int32_t level1ImageHeightOdd  = level1ImageHeight >> 1;
    int32_t level1ImageHeightEven = level1ImageHeight - level1ImageHeightOdd;

    AgenWrapper w0, w1;
    // ------------------------------------------- //
    // config for 1st level compute
    // ------------------------------------------- //
    const size_t level1NiterX = L0_LINE_PITCH_ALIGN_VECW / VECW;
    agen in                   = init(0);
    w0.size                   = sizeof(BASE_TYPE);
    w0.n1                     = 6;
    w0.n2                     = 3;
    w0.n3                     = level1NiterX;
    w0.n4                     = NITER_Y_TILE0;
    w0.s1                     = level0LinePitchEven;
    w0.s2                     = 2;
    w0.s3                     = VECW;
    w0.s4                     = KERNEL_TH0_SPLIT_BUF * level0LinePitchEven;
    INIT_AGEN4(in, w0);
    in                         = update_agen_cb_start(in, (intptr_t)(&level0EvenBuffer[0]));
    in                         = update_agen_cb_size(in, level0CbLenEven * sizeof(BASE_TYPE));
    AgenCFG level0EvenInputCfg = extract_agen_cfg(in);

    w0.s1 = level0LinePitchOdd;
    w0.s4 = KERNEL_TH0_SPLIT_BUF * level0LinePitchOdd;
    INIT_AGEN4(in, w0);
    in                        = update_agen_cb_start(in, (intptr_t)(&level0OddBuffer[0]));
    in                        = update_agen_cb_size(in, level0CbLenOdd * sizeof(BASE_TYPE));
    AgenCFG level0OddInputCfg = extract_agen_cfg(in);

    agen out = init(0);
    w1.size  = sizeof(BASE_TYPE);
    w1.n1    = 2;
    w1.n2    = 3;
    w1.n3    = level1NiterX;
    w1.n4    = NITER_Y_TILE0;
    w1.s1    = level1LinePitch;
    w1.s2    = 0;
    w1.s3    = VECW >> 1;
    w1.s4    = KERNEL_TH1_SPLIT_BUF * level1LinePitch;
    INIT_AGEN4(out, w1);
    out.round = QBITS;

    AgenCFG level1EvenOutputCfg = extract_agen_cfg(out);
    AgenCFG level1OddOutputCfg  = extract_agen_cfg(out);

    for (int32_t i = 0; i < niterY; i++)
    {
        for (int32_t j = 0; j < niterX; j++)
        {
            BASE_TYPE *inTileEven    = (BASE_TYPE *)cupvaRasterDataFlowAcquire(level0HandlerEven);
            BASE_TYPE *inTileOdd     = (BASE_TYPE *)cupvaRasterDataFlowAcquire(level0HandlerOdd);
            BASE_TYPE *outL1TileEven = (BASE_TYPE *)cupvaRasterDataFlowAcquire(level1Handler);

            BASE_TYPE *outL1TileOdd = &level1OddBuffer[0] + (outL1TileEven - level1EvenBuffer);

            cupvaModifyAgenCfgBase(&level0EvenInputCfg, inTileEven);
            cupvaModifyAgenCfgBase(&level0OddInputCfg, inTileOdd);

            cupvaModifyAgenCfgBase(&level1EvenOutputCfg, outL1TileEven);
            cupvaModifyAgenCfgBase(&level1OddOutputCfg, outL1TileOdd);

            nonSep6x6FIR_downsamp2x2_regblk4(&level0EvenInputCfg, &level0OddInputCfg, &level1EvenOutputCfg,
                                             &level1OddOutputCfg, level1NiterX, &weightBuffer[0]);

            cupvaRasterDataFlowRelease(level0HandlerEven);
            cupvaRasterDataFlowRelease(level0HandlerOdd);
            cupvaRasterDataFlowRelease(level1Handler);
        }
    }
    cupvaRasterDataFlowClose(level0HandlerEven);
    cupvaRasterDataFlowClose(level0HandlerOdd);
    cupvaRasterDataFlowClose(level1Handler);

    return 0;
}

int32_t image_pyramid_compute_2levels(const int32_t niterY, const int32_t niterX, const int32_t level1ImageWidth,
                                      const int32_t level1ImageHeight, UnifiedRDFHandler &level0HandlerEven,
                                      UnifiedRDFHandler &level0HandlerOdd, UnifiedRDFHandler &level1Handler,
                                      UnifiedRDFHandler &level2Handler)
{
    cupvaRasterDataFlowOpen(level0HandlerEven, &level0EvenBuffer[0]);
    cupvaRasterDataFlowOpen(level0HandlerOdd, &level0OddBuffer[0]);
    cupvaRasterDataFlowOpen(level1Handler, &level1EvenBuffer[L1_TILE_ADJUSTMENT_MAX]);
    cupvaRasterDataFlowOpen(level2Handler, &level2Buffer[0]);

    int32_t level0LinePitchEven = cupvaRasterDataFlowGetLinePitch(level0HandlerEven);
    int32_t level0LinePitchOdd  = cupvaRasterDataFlowGetLinePitch(level0HandlerOdd);

    int32_t level1LinePitch = cupvaRasterDataFlowGetLinePitch(level1Handler);
    int32_t level2LinePitch = cupvaRasterDataFlowGetLinePitch(level2Handler);
    int32_t level0CbLenEven = cupvaRasterDataFlowGetCbLen(level0HandlerEven);
    int32_t level0CbLenOdd  = cupvaRasterDataFlowGetCbLen(level0HandlerOdd);

    int32_t level1ImageHeightOdd  = level1ImageHeight >> 1;
    int32_t level1ImageHeightEven = level1ImageHeight - level1ImageHeightOdd;

    int32_t const l1Adjustment = HALO_L1 + (level1LinePitch * HALO_L1_Y_SPLIT_BUF);

    AgenWrapper w0, w1;
    // ------------------------------------------- //
    // config for 1st level compute
    // ------------------------------------------- //
    const size_t level1NiterX = L0_LINE_PITCH_ALIGN_VECW / VECW;
    agen in                   = init(0);
    w0.size                   = sizeof(BASE_TYPE);
    w0.n1                     = 6;
    w0.n2                     = 3;
    w0.n3                     = level1NiterX;
    w0.n4                     = NITER_Y_TILE0;
    w0.s1                     = level0LinePitchEven;
    w0.s2                     = 2;
    w0.s3                     = VECW;
    w0.s4                     = KERNEL_TH0_SPLIT_BUF * level0LinePitchEven;
    INIT_AGEN4(in, w0);
    in                         = update_agen_cb_start(in, (intptr_t)(&level0EvenBuffer[0]));
    in                         = update_agen_cb_size(in, level0CbLenEven * sizeof(BASE_TYPE));
    AgenCFG level0EvenInputCfg = extract_agen_cfg(in);

    w0.s1 = level0LinePitchOdd;
    w0.s4 = KERNEL_TH0_SPLIT_BUF * level0LinePitchOdd;
    INIT_AGEN4(in, w0);
    in                        = update_agen_cb_start(in, (intptr_t)(&level0OddBuffer[0]));
    in                        = update_agen_cb_size(in, level0CbLenOdd * sizeof(BASE_TYPE));
    AgenCFG level0OddInputCfg = extract_agen_cfg(in);

    agen out = init(0);
    w1.size  = sizeof(BASE_TYPE);
    w1.n1    = 2;
    w1.n2    = 3;
    w1.n3    = level1NiterX;
    w1.n4    = NITER_Y_TILE0;
    w1.s1    = level1LinePitch;
    w1.s2    = 0;
    w1.s3    = VECW >> 1;
    w1.s4    = KERNEL_TH1_SPLIT_BUF * level1LinePitch;
    INIT_AGEN4(out, w1);
    out.round = QBITS;

    AgenCFG level1EvenOutputCfg = extract_agen_cfg(out);
    AgenCFG level1OddOutputCfg  = extract_agen_cfg(out);

    // ------------------------------------------- //
    // config for 2nd level compute
    // ------------------------------------------- //
    const size_t level2NiterX = L1_LINE_PITCH_ALIGN_VECW / VECW;
    in                        = init(0);
    w0.size                   = sizeof(BASE_TYPE);
    w0.n1                     = 4;
    w0.n2                     = 3;
    w0.n3                     = level2NiterX;
    w0.n4                     = NITER_Y_TILE1;
    w0.s1                     = level1LinePitch;
    w0.s2                     = 2;
    w0.s3                     = VECW;
    w0.s4                     = KERNEL_TH1_SPLIT_BUF * level1LinePitch;
    INIT_AGEN4(in, w0);

    AgenCFG level1EvenInputCfg = extract_agen_cfg(in);
    AgenCFG level1OddInputCfg  = extract_agen_cfg(in);

    out     = init(0);
    w1.size = sizeof(BASE_TYPE);
    w1.n1   = level2NiterX;
    w1.n2   = NITER_Y_TILE1;
    w1.s1   = VECW >> 1;
    w1.s2   = KERNEL_TH2 * level2LinePitch;
    INIT_AGEN2(out, w1);
    out.round = QBITS;

    AgenCFG level2EvenOutputCfg = extract_agen_cfg(out);
    AgenCFG level2OddOutputCfg  = extract_agen_cfg(out);

    for (int32_t i = 0; i < niterY; i++)
    {
        for (int32_t j = 0; j < niterX; j++)
        {
            BASE_TYPE *inTileEven = (BASE_TYPE *)cupvaRasterDataFlowAcquire(level0HandlerEven);
            BASE_TYPE *inTileOdd  = (BASE_TYPE *)cupvaRasterDataFlowAcquire(level0HandlerOdd);

            BASE_TYPE *outL1TileEven = (BASE_TYPE *)cupvaRasterDataFlowAcquire(level1Handler);
            BASE_TYPE *outL2TileEven = (BASE_TYPE *)cupvaRasterDataFlowAcquire(level2Handler);

            // Negative adjustment to L1 tile to offset the tile arena.
            outL1TileEven -= l1Adjustment;

            // Set offset pointers.
            BASE_TYPE *outL1TileOdd = &level1OddBuffer[0] + (outL1TileEven - level1EvenBuffer);
            BASE_TYPE *outL2TileOdd = outL2TileEven + level2LinePitch;

            BASE_TYPE *inTileL1Even = outL1TileEven;
            BASE_TYPE *inTileL1Odd  = outL1TileOdd;

            cupvaModifyAgenCfgBase(&level0EvenInputCfg, inTileEven);
            cupvaModifyAgenCfgBase(&level0OddInputCfg, inTileOdd);

            cupvaModifyAgenCfgBase(&level1EvenOutputCfg, outL1TileEven);
            cupvaModifyAgenCfgBase(&level1OddOutputCfg, outL1TileOdd);

            nonSep6x6FIR_downsamp2x2_regblk4(&level0EvenInputCfg, &level0OddInputCfg, &level1EvenOutputCfg,
                                             &level1OddOutputCfg, level1NiterX, &weightBuffer[0]);

            if (j == 0)
            {
                apply_border_padding(inTileL1Even, inTileL1Odd, 0, level1LinePitch, padValue);
            }
            if (j == niterX - 1)
            {
                int32_t startAddress = level1ImageWidth - j * TW1 + HALO_L1;
                apply_border_padding(inTileL1Even, inTileL1Odd, startAddress, level1LinePitch, padValue);
            }

            if (i == 0)
            {
                memset_vpu_zero_pad(inTileL1Even, level1LinePitch, HALO_L1_Y_SPLIT_BUF);
                memset_vpu_zero_pad(inTileL1Odd, level1LinePitch, HALO_L1_Y_SPLIT_BUF);
            }
            if ((i == niterY - 2) || (i == niterY - 1))
            {
                int32_t startAddressEven = level1ImageHeightEven - i * TH1_SPLIT_BUF + HALO_L1_Y_SPLIT_BUF;
                int32_t startAddresstOdd = level1ImageHeightOdd - i * TH1_SPLIT_BUF + HALO_L1_Y_SPLIT_BUF;
                int32_t offsetEven       = level1LinePitch * startAddressEven;
                int32_t offsetOdd        = level1LinePitch * startAddresstOdd;
                if (startAddressEven < TH1_SPLIT_BUF + HALO_L1)
                {
                    memset_vpu_zero_pad(inTileL1Even + offsetEven, level1LinePitch, HALO_L1_Y_SPLIT_BUF);
                }
                if (startAddresstOdd < TH1_SPLIT_BUF + HALO_L1)
                {
                    memset_vpu_zero_pad(inTileL1Odd + offsetOdd, level1LinePitch, HALO_L1_Y_SPLIT_BUF);
                }
            }

            cupvaModifyAgenCfgBase(&level1EvenInputCfg, inTileL1Even);
            cupvaModifyAgenCfgBase(&level1OddInputCfg, inTileL1Odd);

            cupvaModifyAgenCfgBase(&level2EvenOutputCfg, outL2TileEven);
            cupvaModifyAgenCfgBase(&level2OddOutputCfg, outL2TileOdd);

            nonSep6x6FIR_downsamp2x2_regblk2(&level1EvenInputCfg, &level1OddInputCfg, &level2EvenOutputCfg,
                                             &level2OddOutputCfg, level2NiterX, &weightBuffer[0]);

            cupvaRasterDataFlowRelease(level0HandlerEven);
            cupvaRasterDataFlowRelease(level0HandlerOdd);
            cupvaRasterDataFlowRelease(level1Handler);
            cupvaRasterDataFlowRelease(level2Handler);
        }
    }
    cupvaRasterDataFlowClose(level0HandlerEven);
    cupvaRasterDataFlowClose(level0HandlerOdd);
    cupvaRasterDataFlowClose(level1Handler);
    cupvaRasterDataFlowClose(level2Handler);

    return 0;
}

CUPVA_VPU_MAIN()
{
    // convert coef to weight once
    nonSep6x6FIR_downsamp2x2_weight_init(&gaussianKernel[0], &weightBuffer[0]);

    switch (totalLevels)
    {
    case 2:
        image_pyramid_compute_1level(niterY, niterX, (imgWidth + 1) / 2, (imgHeight + 1) / 2, level0HandlerEven,
                                     level0HandlerOdd, level1Handler);
        break;
    case 3:
        image_pyramid_compute_2levels(niterY, niterX, (imgWidth + 1) / 2, (imgHeight + 1) / 2, level0HandlerEven,
                                      level0HandlerOdd, level1Handler, level2Handler);
        break;
    default:
        return -1;
    }

    return 0;
}
