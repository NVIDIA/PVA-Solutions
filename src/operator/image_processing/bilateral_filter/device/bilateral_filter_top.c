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

#include "bilateral_filter_knl.h"

#include <cupva_device.h>

// Input buffers
VMEM(A, uint8_t, srcBuffer, RDF_CIRCULAR(uint8_t, TILE_WIDTH, TILE_HEIGHT, RADIUS, RADIUS));
VMEM(B, uint8_t, weightTable, COLOR_8BIT *SPACE2_PARAMS_SIZE);
VMEM(C, uint8_t, weightTableReplicate32, COLOR_8BIT *SPACE2_PARAMS_SIZE * 32);

// Intermediate buffers
VMEM(B, uint32_t, valueSumBuffer, TILE_HEIGHT *TILE_WIDTH);
VMEM(C, uint32_t, weightSumBuffer, TILE_HEIGHT *TILE_WIDTH);

// Output buffers
VMEM(A, uint8_t, dstBuffer, RDF_DOUBLE(uint8_t, TILE_WIDTH, TILE_HEIGHT));

// Parameters
VMEM(C, int32_t, niterX);
VMEM(C, int32_t, niterY);
VMEM(C, int32_t, imageHeight);
VMEM(C, BilateralFilterConfig, cfg);

VMEM_RDF_UNIFIED(C, srcHandler);
VMEM_RDF_UNIFIED(C, dstHandler);

CUPVA_VPU_MAIN()
{
#ifdef KSIZE_3X3
    int32_t lookupOffsets[1] = {1 * COLOR_8BIT};
#endif
#ifdef KSIZE_5X5
    int32_t lookupOffsets[3 * 3] = {1 * COLOR_8BIT, 2 * COLOR_8BIT, 1 * COLOR_8BIT, 2 * COLOR_8BIT, 3 * COLOR_8BIT,
                                    2 * COLOR_8BIT, 1 * COLOR_8BIT, 2 * COLOR_8BIT, 1 * COLOR_8BIT};
#endif
#ifdef KSIZE_7X7
    int32_t lookupOffsets[5 * 5] = {1 * COLOR_8BIT, 2 * COLOR_8BIT, 3 * COLOR_8BIT, 2 * COLOR_8BIT, 1 * COLOR_8BIT,
                                    2 * COLOR_8BIT, 4 * COLOR_8BIT, 5 * COLOR_8BIT, 4 * COLOR_8BIT, 2 * COLOR_8BIT,
                                    3 * COLOR_8BIT, 5 * COLOR_8BIT, 6 * COLOR_8BIT, 5 * COLOR_8BIT, 3 * COLOR_8BIT,
                                    2 * COLOR_8BIT, 4 * COLOR_8BIT, 5 * COLOR_8BIT, 4 * COLOR_8BIT, 2 * COLOR_8BIT,
                                    1 * COLOR_8BIT, 2 * COLOR_8BIT, 3 * COLOR_8BIT, 2 * COLOR_8BIT, 1 * COLOR_8BIT};
#endif

    int32_t srcCircularBufLen = cupvaRasterDataFlowGetCbLen(srcHandler);
    int32_t srcLinePitch      = cupvaRasterDataFlowGetLinePitch(srcHandler);

    BilateralFilterConfig cfg;
    bilateral_filter_init(lookupOffsets, valueSumBuffer, weightSumBuffer, srcLinePitch, srcCircularBufLen, srcBuffer, 0,
                          &cfg);

    replicate_table_char_32(weightTableReplicate32, weightTable, COLOR_8BIT * SPACE2_PARAMS_SIZE);

    cupvaRasterDataFlowOpen(srcHandler, &srcBuffer[0]);
    cupvaRasterDataFlowOpen(dstHandler, &dstBuffer[0]);

    // check whether image height is multiple of tile height
    int32_t lastTileMaskValid = imageHeight % TILE_HEIGHT;

    for (int32_t i = 0; i < niterY; i++)
    {
        if (lastTileMaskValid && i == (niterY - 1))
        {
            // set tile height for last tile row
            bilateral_filter_init(lookupOffsets, valueSumBuffer, weightSumBuffer, srcLinePitch, srcCircularBufLen,
                                  srcBuffer, lastTileMaskValid, &cfg);
        }

        for (int32_t j = 0; j < niterX; j++)
        {
            uint8_t *srcTile = (uint8_t *)cupvaRasterDataFlowAcquire(srcHandler);
            uint8_t *dstTile = (uint8_t *)cupvaRasterDataFlowAcquire(dstHandler);

            ptrdiff_t srcOffset = &srcTile[0] - &srcBuffer[0];

            ptrdiff_t src_corner_start = ((RADIUS + srcOffset) >= srcCircularBufLen)
                                             ? ((RADIUS + srcOffset) - srcCircularBufLen)
                                             : (RADIUS + srcOffset);
            cupvaModifyAgenCfgBase(&cfg.cfg_ld_img_corner, srcBuffer + src_corner_start);

            ptrdiff_t src_square_start = ((srcLinePitch + 1 + srcOffset) >= srcCircularBufLen)
                                             ? ((srcLinePitch + 1 + srcOffset) - srcCircularBufLen)
                                             : (srcLinePitch + 1 + srcOffset);
            cupvaModifyAgenCfgBase(&cfg.cfg_ld_img_square, srcBuffer + src_square_start);

            ptrdiff_t src_central_start = ((RADIUS * srcLinePitch + RADIUS + srcOffset) >= srcCircularBufLen)
                                              ? ((RADIUS * srcLinePitch + RADIUS + srcOffset) - srcCircularBufLen)
                                              : (RADIUS * srcLinePitch + RADIUS + srcOffset);
            cupvaModifyAgenCfgBase(&cfg.cfg_ld_img_central, srcBuffer + src_central_start);

            cupvaModifyAgenCfgBase(&cfg.cfg_st_img, dstTile);

            bilateral_filter_accumulate_stage(weightTableReplicate32, &cfg);
            bilateral_filter_division_stage(&cfg);

            cupvaRasterDataFlowRelease(srcHandler);
            cupvaRasterDataFlowRelease(dstHandler);
        }
    }
    cupvaRasterDataFlowClose(srcHandler);
    cupvaRasterDataFlowClose(dstHandler);

    return 0;
}