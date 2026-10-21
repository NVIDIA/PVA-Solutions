/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "image_blend_knl_u8.h"

VMEM_RDF_UNIFIED(A, src0Handler);
VMEM_RDF_UNIFIED(A, src1Handler);
VMEM_RDF_UNIFIED(A, dstHandler);

VMEM(A, int32_t, niterX);
VMEM(A, int32_t, niterY);
VMEM(A, int32_t, imagePlaneCount);
VMEM(A, uint32_t, alpha);
VMEM(A, ImageBlendConfig, cfg);

// buffer allocation for src0 image
VMEM(A, uint8_t, src0Buffer0, RDF_DOUBLE(uint8_t, TILE_WIDTH, TILE_HEIGHT));
VMEM(A, uint8_t, src0Buffer1, RDF_DOUBLE(uint8_t, TILE_WIDTH, TILE_HEIGHT));
VMEM(A, uint8_t, src0Buffer2, RDF_DOUBLE(uint8_t, TILE_WIDTH, TILE_HEIGHT));
VMEM(A, uint8_t, src0Buffer3, RDF_DOUBLE(uint8_t, TILE_WIDTH, TILE_HEIGHT));

// buffer allocation for src1 image
VMEM(B, uint8_t, src1Buffer0, RDF_DOUBLE(uint8_t, TILE_WIDTH, TILE_HEIGHT));
VMEM(B, uint8_t, src1Buffer1, RDF_DOUBLE(uint8_t, TILE_WIDTH, TILE_HEIGHT));
VMEM(B, uint8_t, src1Buffer2, RDF_DOUBLE(uint8_t, TILE_WIDTH, TILE_HEIGHT));
VMEM(B, uint8_t, src1Buffer3, RDF_DOUBLE(uint8_t, TILE_WIDTH, TILE_HEIGHT));

// buffer allocation for dst image
VMEM(C, uint8_t, dstBuffer0, RDF_DOUBLE(uint8_t, TILE_WIDTH, TILE_HEIGHT));
VMEM(C, uint8_t, dstBuffer1, RDF_DOUBLE(uint8_t, TILE_WIDTH, TILE_HEIGHT));
VMEM(C, uint8_t, dstBuffer2, RDF_DOUBLE(uint8_t, TILE_WIDTH, TILE_HEIGHT));
VMEM(C, uint8_t, dstBuffer3, RDF_DOUBLE(uint8_t, TILE_WIDTH, TILE_HEIGHT));

CUPVA_VPU_MAIN()
{
    int niter = TILE_WIDTH / chess_elementsof(dvchar) * TILE_HEIGHT;

    int32_t src0LinePitch = cupvaRasterDataFlowGetLinePitch(src0Handler);
    int32_t src1LinePitch = cupvaRasterDataFlowGetLinePitch(src1Handler);
    int32_t dstLinePitch  = cupvaRasterDataFlowGetLinePitch(dstHandler);

    cupvaRasterDataFlowOpen(src0Handler, &src0Buffer0[0]);
    cupvaRasterDataFlowOpen(src1Handler, &src1Buffer0[0]);
    cupvaRasterDataFlowOpen(dstHandler, &dstBuffer0[0]);

    image_blend_agen_init(src0LinePitch, src1LinePitch, dstLinePitch, &cfg);

    uint8_t *src0Tiles[MAX_CHANNEL_COUNT];
    uint8_t *src1Tiles[MAX_CHANNEL_COUNT];
    uint8_t *dstTiles[MAX_CHANNEL_COUNT];

    for (int32_t y = 0; y < niterY; y++)
    {
        for (int32_t x = 0; x < niterX; x++)
        {
            src0Tiles[0] = (uint8_t *)cupvaRasterDataFlowAcquire(src0Handler);
            src1Tiles[0] = (uint8_t *)cupvaRasterDataFlowAcquire(src1Handler);
            dstTiles[0]  = (uint8_t *)cupvaRasterDataFlowAcquire(dstHandler);

            // We alias 4 RDFs to a single handler. This is only possible when all RDFs have the same
            // geometry. To find correct pointers in the other three tiles, we find the offset of the
            // first tile from the base of its buffer
            ptrdiff_t offset = src0Tiles[0] - &src0Buffer0[0];

            // Since src0/src1/dst have the same tile size, they can share the offset.
            src0Tiles[1] = &src0Buffer1[offset];
            src0Tiles[2] = &src0Buffer2[offset];
            src0Tiles[3] = &src0Buffer3[offset];

            src1Tiles[1] = &src1Buffer1[offset];
            src1Tiles[2] = &src1Buffer2[offset];
            src1Tiles[3] = &src1Buffer3[offset];

            dstTiles[1] = &dstBuffer1[offset];
            dstTiles[2] = &dstBuffer2[offset];
            dstTiles[3] = &dstBuffer3[offset];

            for (int p = 0; p < imagePlaneCount; p++)
            {
                image_blend_agen_update(src0Tiles[p], src1Tiles[p], dstTiles[p], &cfg);
                image_blend_exec(alpha, niter, &cfg);
            }

            cupvaRasterDataFlowRelease(src0Handler);
            cupvaRasterDataFlowRelease(src1Handler);
            cupvaRasterDataFlowRelease(dstHandler);
        }
    }

    cupvaRasterDataFlowClose(src0Handler);
    cupvaRasterDataFlowClose(src1Handler);
    cupvaRasterDataFlowClose(dstHandler);

    return 0;
}