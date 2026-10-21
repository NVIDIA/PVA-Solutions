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

#include "template_matching_knl.h"

#include <cupva_device.h>

// Tile replicated into superbanks A and B by one DMA transfer: three reads feed
// each correlation issue and a superbank serves one per cycle.
// The +6 rows are slack for RDF ring granularity, not a derived bound.
VMEM(A, uint8_t, srcTileA, RDF_DOUBLE(uint8_t, TM_TILE_WIDTH, TM_TILE_HEIGHT + 6, TM_MAX_HALO, TM_MAX_HALO));
VMEM(B, uint8_t, srcTileB, RDF_DOUBLE(uint8_t, TM_TILE_WIDTH, TM_TILE_HEIGHT + 6, TM_MAX_HALO, TM_MAX_HALO));
VMEM(C, float, outTile, RDF_DOUBLE(float, TM_TILE_WIDTH, TM_TILE_HEIGHT));

// Vertical prefix sums of the squared-window row sums, h+2 rows live.
VMEM(B, int32_t, vRing, TM_V_ROWS *TM_SCRATCH_ROW);

// Two output rows of one pass. Kept off bank C so the output tile is the only
// streaming buffer there.
VMEM(B, int32_t, corrScratch, 2 * TM_SCRATCH_ROW);

VMEM(C, uint8_t, coefBlocks, TM_COEF_BYTES);

// DVLDPermHBU halfword indices, resident in bank A. The image pattern takes
// halfwords 0..15 into .lo and 4..19 into .hi, the window VFilt4x2x2 reads per
// group of 8 output lanes. The coefficient pattern takes the two sets, each
// replicated four times because src2 repeats per 8 byte lanes.
VMEM_NOEXPORT(A, uint8_t, permTable, TM_PERM_ENTRIES) = {
    0, 1, 2, 3, 4, 5, 6,  7,  8,  9,  10, 11, 12, 13, 14, 15, /* image .lo */
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, /* image .hi */
    0, 1, 2, 3, 0, 1, 2,  3,  0,  1,  2,  3,  0,  1,  2,  3,  /* coef  .lo */
    4, 5, 6, 7, 4, 5, 6,  7,  4,  5,  6,  7,  4,  5,  6,  7,  /* coef  .hi */
};

// The template as transferred. The transfer writes only w by h; the rest of the
// buffer has to read as zero, which the energy sweep, the row -1 lookup in the
// packing and the padded tap of a one-row template all rely on. Measured to hold
// on both the emulator and silicon: a submit that dirties this buffer at exit
// still sees it zeroed on the next submit. Bank A so the packing loads here and
// stores to the bank C coefficients without a superbank conflict.
VMEM(A, uint8_t, tmplStage, TM_STAGE_BYTES);

VMEM(A, TemplateMatchingParams, algoParams);

VMEM_SEQDF_HANDLER(A, tmplHandler);
VMEM_RDF_UNIFIED(C, imgHandler);
VMEM_RDF_UNIFIED(C, dstHandler);

// Wrap a tile-buffer byte offset into the circular ring. A single conditional
// subtraction suffices because the offset is at most one tile past the ring.
static inline int32_t tmWrap(int32_t offset, int32_t cbLen)
{
    return (offset >= cbLen) ? (offset - cbLen) : offset;
}

CUPVA_VPU_MAIN()
{
    const int32_t imgLinePitch = cupvaRasterDataFlowGetLinePitch(imgHandler);
    const int32_t outLinePitch = cupvaRasterDataFlowGetLinePitch(dstHandler);

    // RDF may lay the halo'd tile out circularly even under RDF_DOUBLE, so every
    // image offset below is wrapped against these bounds. Inert if it does not.
    const int32_t imgCbLen = cupvaRasterDataFlowGetCbLen(imgHandler);

    const int32_t th     = algoParams.templateHeight;
    const int32_t haloX  = algoParams.haloX;
    const int32_t haloY  = algoParams.haloY;
    const int32_t vRows  = algoParams.vRows;
    const int32_t niterX = algoParams.niterX;
    const int32_t niterY = algoParams.niterY;

    // Byte offsets of the peeled VSumSq pair inside an image row.
    const int32_t lastTapAOfs = 2 * (algoParams.sqPairs - 1);
    const int32_t lastTapBOfs = lastTapAOfs + 1;

    cupvaSQDFOpen(tmplHandler);
    cupvaSQDFTrig(tmplHandler);

    TemplateMatchingConfig cfg;
    template_matching_init_agens(&algoParams, imgLinePitch, outLinePitch, &srcTileA[0], &srcTileB[0], imgCbLen, &cfg);

    cupvaSQDFSync(tmplHandler);
    cfg.templateEnergy = template_matching_template_energy(&tmplStage[0]);
    template_matching_pack_coefficients(&tmplStage[0], &coefBlocks[0], &algoParams);
    cupvaSQDFClose(tmplHandler);

    cupvaRasterDataFlowOpen(imgHandler, &srcTileA[0]);
    cupvaRasterDataFlowOpen(dstHandler, &outTile[0]);

    for (int32_t ty = 0; ty < niterY; ty++)
    {
        for (int32_t tx = 0; tx < niterX; tx++)
        {
            uint8_t *tileA = (uint8_t *)cupvaRasterDataFlowAcquire(imgHandler);
            float *tileOut = (float *)cupvaRasterDataFlowAcquire(dstHandler);

            // Acquire wraps the head of the ring to the tail for the buffer it
            // owns; the replicate is reached by pointer math, so do it by hand.
            cupvaCircularBufferMemcpy(&srcTileB[0], imgCbLen);

            // Tile origin: the RDF writes the symmetric halo first, so skip
            // (haloX, haloY). Row offsets are wrapped at each use site; the
            // tap offsets inside a row are not, because those agens carry the
            // ring bounds and wrap in hardware.
            // Both buffers hold the same tile at the same offset.
            const int32_t originOfs = (int32_t)(tileA - &srcTileA[0]) + (haloY * imgLinePitch) + haloX;

            // V(-1) = 0 so that EI(0) = V(h-1) - V(-1) needs no special case.
            template_matching_zero_v_row(&vRing[0], &cfg);

            // Ring slot holding V(r) is (r + 1) mod vRows, tracked incrementally.
            int32_t prevSlot = 0;
            int32_t curSlot  = 1;

            // Prologue: rows 0..h-2 of the window. The vertical prefix forces
            // the rows to be produced in order.
            for (int32_t r = 0; r < (th - 1); r++)
            {
                uint8_t *rowA = &srcTileA[tmWrap(originOfs + (r * imgLinePitch), imgCbLen)];
                uint8_t *rowB = &srcTileB[tmWrap(originOfs + (r * imgLinePitch), imgCbLen)];
                template_matching_box_row(rowA, rowB + 1, rowA + lastTapAOfs, rowB + lastTapBOfs,
                                          &vRing[prevSlot * TM_SCRATCH_ROW], &vRing[curSlot * TM_SCRATCH_ROW], &cfg);
                prevSlot = curSlot;
                curSlot++;
                if (curSlot == vRows)
                {
                    curSlot = 0;
                }
            }

            // Consumption slots: V(y-1), V(y+h-1) for output row y and
            // V(y), V(y+h) for output row y+1.
            int32_t oldSlot0 = 0;
            int32_t oldSlot1 = 1;
            int32_t newSlot0 = curSlot;
            int32_t newSlot1 = ((curSlot + 1) == vRows) ? 0 : (curSlot + 1);

            for (int32_t y = 0; y < TM_TILE_HEIGHT; y += 2)
            {
                // Produce V(y+h-1) and V(y+h).
                for (int32_t k = 0; k < 2; k++)
                {
                    const int32_t r = y + (th - 1) + k;
                    uint8_t *rowA   = &srcTileA[tmWrap(originOfs + (r * imgLinePitch), imgCbLen)];
                    uint8_t *rowB   = &srcTileB[tmWrap(originOfs + (r * imgLinePitch), imgCbLen)];
                    template_matching_box_row(rowA, rowB + 1, rowA + lastTapAOfs, rowB + lastTapBOfs,
                                              &vRing[prevSlot * TM_SCRATCH_ROW], &vRing[curSlot * TM_SCRATCH_ROW],
                                              &cfg);
                    prevSlot = curSlot;
                    curSlot++;
                    if (curSlot == vRows)
                    {
                        curSlot = 0;
                    }
                }

                template_matching_correlate(&srcTileA[tmWrap(originOfs + (y * imgLinePitch), imgCbLen)],
                                            &srcTileB[tmWrap(originOfs + ((y + 1) * imgLinePitch), imgCbLen)],
                                            &coefBlocks[0], &permTable[0], &corrScratch[0], &cfg);

                template_matching_score(&corrScratch[0], &vRing[newSlot0 * TM_SCRATCH_ROW],
                                        &vRing[oldSlot0 * TM_SCRATCH_ROW], &vRing[newSlot1 * TM_SCRATCH_ROW],
                                        &vRing[oldSlot1 * TM_SCRATCH_ROW], tileOut + (y * outLinePitch), &cfg);

                oldSlot0 += 2;
                if (oldSlot0 >= vRows)
                {
                    oldSlot0 -= vRows;
                }
                oldSlot1 += 2;
                if (oldSlot1 >= vRows)
                {
                    oldSlot1 -= vRows;
                }
                newSlot0 += 2;
                if (newSlot0 >= vRows)
                {
                    newSlot0 -= vRows;
                }
                newSlot1 += 2;
                if (newSlot1 >= vRows)
                {
                    newSlot1 -= vRows;
                }
            }

            cupvaRasterDataFlowRelease(imgHandler);
            cupvaRasterDataFlowRelease(dstHandler);
        }
    }

    cupvaRasterDataFlowClose(imgHandler);
    cupvaRasterDataFlowClose(dstHandler);

    return 0;
}
