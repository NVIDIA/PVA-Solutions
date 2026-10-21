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
 * @file radar_cfar_top.c
 *
 * @brief VPU entry point for the RadarCFAR operator.
 *
 * Phase 0 marches along x over CFAR_LANES-row strips and writes the horizontal decision
 * mask. Phase 1 marches along y over CFAR_LANES-column strips, combines with that mask,
 * and drains the surviving cells to the detection list. Both phases run inside one
 * submit, so the per-submit cost is paid once per frame rather than once per tile.
 *
 * Four RDF handlers stream the raster tiles. Three SEQDF handlers cover the one-shot
 * 28 KB table load, the detection-list drain and the count word; the drain needs a
 * device-computed destination offset (cupvaSQDFUpdateAddr / cupvaSQDFUpdateTileSize),
 * which RDF cannot express.
 *
 * VMEM bank plan (superbank = 128 KB each; ~295 KB of 384 KB used):
 *   A  DMA landing buffer for both passes, handlers, scalars, reciprocal tables  (92 KB)
 *   B  private padded strip written by the pad kernels, read by the marching loops (97 KB)
 *   C  horizontal decision mask, detection staging, frame detection list        (106 KB)
 * The compute-hot buffer (B) is on its own superbank, away from the DMA landing buffer in
 * A. The mask in C is DMA-written in phase 2 as well as DMA-read in phase 1, but the
 * phases are sequential, so no two DMA streams are ever live on one superbank at once.
 */

#include "radar_cfar_knl.h"
#include "radar_cfar_param.h"

#include <cupva_device.h>
#include <cupva_device/sequence_dataflow.h>

/*
 * Buffers are sized for the widest supported image and shared between the two phases,
 * which are sequential: aliasing keeps the working set at ~295 KB of the 384 KB VMEM
 * instead of the ~720 KB independent worst-case buffers would need.
 */

/* Host parameters. */
VMEM(A, CfarHost2DevParam, cfarParam);

/*
 * Per-position reciprocal and threshold-factor tables, DMA'd from DRAM once at startup.
 * They are too large for the 8 KB parameter block, so the host stages them in DRAM and a
 * SequenceDataFlow copies them in before either pass runs.
 */
VMEM(A, int32_t, tableBuf, CFAR_NUM_TABLES *CFAR_TABLE_STRIDE);

/* Lane index vector, used to turn a lane position into an image x. */
VMEM(A, int32_t, laneIdxBuf, CFAR_LANES);

/*
 * Superbank A: the DMA landing buffer, shared by both passes. A row strip is
 * CFAR_LANES x width and a column strip is height x CFAR_LANES, so one
 * CFAR_LANES * CFAR_MAX_DIM buffer serves both.
 */
VMEM(A, uint32_t, inBuffer, CFAR_LANES *CFAR_MAX_DIM);

/*
 * Superbank B: the private padded strip, written by cfarPadRowStrip/cfarPadColStrip and
 * read by the marching loops. The 16k+1 element pitch gives transposed access an integral
 * lane offset, and the buffer is large enough for a full-height column strip plus halos.
 */
VMEM(B, uint32_t, padBuffer, CFAR_PAD_BUF_ELEMS);

/*
 * Superbank C: the horizontal mask, written transposed in phase 0 and read back
 * naturally in phase 1, plus the detection staging drained to DRAM.
 */
VMEM(C, int32_t, maskBuffer, CFAR_MASK_BUF_ELEMS);
VMEM(C, int32_t, stageY, CFAR_STAGE_CAPACITY + CFAR_STAGE_SLOP);
VMEM(C, int32_t, stageX, CFAR_STAGE_CAPACITY + CFAR_STAGE_SLOP);
VMEM(C, int32_t, detectionCountBuf);

/*
 * The frame's detections, in the order the strips produce them. Coordinates are bounded by
 * CFAR_MAX_DIM so they fit a halfword, which is what lets the whole list sit in VMEM.
 */
VMEM(C, int16_t, collectY, CFAR_MAX_DETECTIONS);
VMEM(C, int16_t, collectX, CFAR_MAX_DETECTIONS);

/* Dataflow handlers. */
VMEM_SEQDF_HANDLER(A, tableHdl);
VMEM_RDF_UNIFIED(A, hInHdl);
VMEM_RDF_UNIFIED(A, hMaskOutHdl);
VMEM_RDF_UNIFIED(A, vInHdl);
VMEM_RDF_UNIFIED(A, vMaskInHdl);
VMEM_SEQDF_HANDLER(A, listHdl);
VMEM_SEQDF_HANDLER(A, countHdl);

/* Base of the detection list in DRAM, for the device-computed drain offset. */
VMEM_POINTER(A, listExtMemPtr);

/*
 * Order the frame's detections the way the reference emits them.
 *
 * cfar_processing_opt.cpp walks 8-row bands, then 8-column blocks, then the rows and columns
 * inside a block, so the list is ordered by (y >> 3, x >> 3, y, x). Snapshot extraction indexes
 * it positionally in its SQDF mode, which is what makes the order part of the contract.
 *
 * A strip emits y-ascending and every 8-column block lies inside one strip, so the collection
 * order already supplies the last two keys. One stable counting sort on the block index
 * (y >> 3) * nbx + (x >> 3) supplies the first two.
 *
 * Both scratch buffers are borrowed: inBuffer's input tiles and padBuffer's padded strip are
 * finished by the time this runs.
 */
static inline void cfarSortBlocks(int32_t n, int32_t nKeys, int32_t nbx, int16_t const *srcY, int16_t const *srcX,
                                  int16_t *dstY, int16_t *dstX, int32_t *offset)
{
    for (int32_t k = 0; k < nKeys; k++)
    {
        offset[k] = 0;
    }
    for (int32_t i = 0; i < n; i++)
    {
        offset[((int32_t)srcY[i] >> 3) * nbx + ((int32_t)srcX[i] >> 3)]++;
    }
    int32_t acc = 0;
    for (int32_t k = 0; k < nKeys; k++)
    {
        int32_t const c = offset[k];
        offset[k]       = acc;
        acc += c;
    }
    for (int32_t i = 0; i < n; i++)
    {
        int32_t const d = offset[((int32_t)srcY[i] >> 3) * nbx + ((int32_t)srcX[i] >> 3)]++;
        dstY[d]         = srcY[i];
        dstX[d]         = srcX[i];
    }
}

/**
 * @brief Drain @p n staged detections to the detection list in DRAM.
 *
 * The list is {verticalIdx, horizontalIdx} pairs, so one transfer writes the y column
 * and a second the x column, both with a destination line pitch of two elements.
 */
static inline void cfarDrain(SequenceDataFlowHandler &hdl, uint64_t listBase, int32_t written, int32_t n)
{
    uint64_t const dst = listBase + (uint64_t)written * 2u * sizeof(int32_t);

    cupvaSQDFUpdateAddr(hdl, 0, cupvaGetVmemAddress(&stageY[0]), MEMTYPE_VMEM, dst, MEMTYPE_DRAM);
    cupvaSQDFUpdateTileSize(hdl, 0, (uint16_t)sizeof(int32_t), (uint16_t)n);

    cupvaSQDFUpdateAddr(hdl, 1, cupvaGetVmemAddress(&stageX[0]), MEMTYPE_VMEM, dst + sizeof(int32_t), MEMTYPE_DRAM);
    cupvaSQDFUpdateTileSize(hdl, 1, (uint16_t)sizeof(int32_t), (uint16_t)n);

    cupvaSQDFFlushAndTrig(hdl);
    cupvaSQDFSync(hdl);
}

CUPVA_VPU_MAIN()
{
    CfarHost2DevParam const *restrict p = &cfarParam;

    int32_t const width  = p->width;
    int32_t const height = p->height;

    cfarInitLaneIdx(laneIdxBuf);

    /* Load the per-position tables before either pass needs them. */
    cupvaSQDFOpen(tableHdl);
    cupvaSQDFTrig(tableHdl);
    cupvaSQDFSync(tableHdl);
    cupvaSQDFClose(tableHdl);

    int32_t const *restrict horRecipTotal = &tableBuf[CFAR_TBL_HOR_TOTAL * CFAR_TABLE_STRIDE];
    int32_t const *restrict horRecipTrail = &tableBuf[CFAR_TBL_HOR_TRAIL * CFAR_TABLE_STRIDE];
    int32_t const *restrict horRecipLead  = &tableBuf[CFAR_TBL_HOR_LEAD * CFAR_TABLE_STRIDE];
    int32_t const *restrict verRecipTotal = &tableBuf[CFAR_TBL_VER_TOTAL * CFAR_TABLE_STRIDE];
    int32_t const *restrict verRecipTrail = &tableBuf[CFAR_TBL_VER_TRAIL * CFAR_TABLE_STRIDE];
    int32_t const *restrict verRecipLead  = &tableBuf[CFAR_TBL_VER_LEAD * CFAR_TABLE_STRIDE];
    int32_t const *restrict horFactorQ14  = &tableBuf[CFAR_TBL_HOR_FACTOR * CFAR_TABLE_STRIDE];

    /* ---------------- Phase 0: horizontal pass ---------------- */
    cupvaRasterDataFlowOpen(hInHdl, inBuffer);
    cupvaRasterDataFlowOpen(hMaskOutHdl, maskBuffer);

    int32_t const hInLp        = cupvaRasterDataFlowGetLinePitch(hInHdl);
    int32_t const hMaskLp      = cupvaRasterDataFlowGetLinePitch(hMaskOutHdl);
    int32_t const maskLaneOfst = cupvaTransposeLaneOfst(hMaskOutHdl, TRANS_MODE_1, sizeof(int32_t));

    for (int32_t s = 0; s < p->numHorStrips; s++)
    {
        uint32_t *inTile = (uint32_t *)cupvaRasterDataFlowAcquire(hInHdl);
        int32_t *outTile = (int32_t *)cupvaRasterDataFlowAcquire(hMaskOutHdl);

        int32_t const rowBase = s * CFAR_LANES;
        int32_t const rows    = (height - rowBase) < CFAR_LANES ? (height - rowBase) : CFAR_LANES;

        cfarPadRowStrip(inTile, hInLp, padBuffer, width, p->horWindow, rows, p->horCyclic);

        cfarHorizontalStrip(padBuffer, outTile, hMaskLp, maskLaneOfst, horRecipTotal, horRecipTrail, horRecipLead,
                            &horFactorQ14[rowBase], p);

        cupvaRasterDataFlowRelease(hInHdl);
        cupvaRasterDataFlowRelease(hMaskOutHdl);
    }

    cupvaRasterDataFlowClose(hInHdl);
    cupvaRasterDataFlowClose(hMaskOutHdl);

    /* ---------------- Phase 1: vertical pass and emission ---------------- */
    cupvaRasterDataFlowOpen(vInHdl, inBuffer);
    cupvaRasterDataFlowOpen(vMaskInHdl, maskBuffer);

    int32_t const vInLp   = cupvaRasterDataFlowGetLinePitch(vInHdl);
    int32_t const vMaskLp = cupvaRasterDataFlowGetLinePitch(vMaskInHdl);

    cupvaSQDFOpen(listHdl);
    uint64_t const listBase = listExtMemPtr.base + listExtMemPtr.offset;

    int32_t collected = 0; /* detections held in VMEM for the whole frame */

    for (int32_t s = 0; s < p->numVerStrips; s++)
    {
        uint32_t *inTile  = (uint32_t *)cupvaRasterDataFlowAcquire(vInHdl);
        int32_t *maskTile = (int32_t *)cupvaRasterDataFlowAcquire(vMaskInHdl);

        cfarPadColStrip(inTile, vInLp, padBuffer, height, p->verWindow, p->verCyclic);

        int32_t const room = p->maxDetections - collected;
        if (room > 0)
        {
            int32_t const cap = room < CFAR_STAGE_CAPACITY ? room : CFAR_STAGE_CAPACITY;

            int32_t got = cfarVerticalStrip(padBuffer, maskTile, vMaskLp, laneIdxBuf, verRecipTotal, verRecipTrail,
                                            verRecipLead, s * CFAR_LANES, 0, height, &stageY[0], &stageX[0], cap, p);

            if (got == CFAR_STAGE_CAPACITY && room > CFAR_STAGE_CAPACITY)
            {
                int32_t const step = CFAR_STAGE_CAPACITY / CFAR_LANES;
                for (int32_t y0 = 0; y0 < height && collected < p->maxDetections; y0 += step)
                {
                    int32_t const rows = (height - y0) < step ? (height - y0) : step;
                    int32_t const left = p->maxDetections - collected;
                    int32_t const n =
                        cfarVerticalStrip(padBuffer, maskTile, vMaskLp, laneIdxBuf, verRecipTotal, verRecipTrail,
                                          verRecipLead, s * CFAR_LANES, y0, rows, &stageY[0], &stageX[0],
                                          left < CFAR_STAGE_CAPACITY ? left : CFAR_STAGE_CAPACITY, p);
                    for (int32_t i = 0; i < n; i++)
                    {
                        collectY[collected + i] = (int16_t)stageY[i];
                        collectX[collected + i] = (int16_t)stageX[i];
                    }
                    collected += n;
                }
                got = 0;
            }

            /* Append this strip's run; the reorder below needs the whole frame. */
            for (int32_t i = 0; i < got; i++)
            {
                collectY[collected + i] = (int16_t)stageY[i];
                collectX[collected + i] = (int16_t)stageX[i];
            }
            collected += got;
        }

        cupvaRasterDataFlowRelease(vInHdl);
        cupvaRasterDataFlowRelease(vMaskInHdl);
    }

    cupvaRasterDataFlowClose(vInHdl);
    cupvaRasterDataFlowClose(vMaskInHdl);

    int16_t *const sortedY = (int16_t *)&inBuffer[0];
    int16_t *const sortedX = (int16_t *)&inBuffer[CFAR_MAX_DETECTIONS / 2];

    if (collected > 0)
    {
        int32_t const nbx = (width + 7) / 8;
        cfarSortBlocks(collected, nbx * ((height + 7) / 8), nbx, &collectY[0], &collectX[0], sortedY, sortedX,
                       (int32_t *)&padBuffer[0]);
    }

    int32_t written = 0;
    while (written < collected)
    {
        int32_t const chunk = (collected - written) < CFAR_STAGE_CAPACITY ? (collected - written) : CFAR_STAGE_CAPACITY;
        for (int32_t i = 0; i < chunk; i++)
        {
            stageY[i] = sortedY[written + i];
            stageX[i] = sortedX[written + i];
        }
        cfarDrain(listHdl, listBase, written, chunk);
        written += chunk;
    }
    cupvaSQDFClose(listHdl);

    detectionCountBuf = collected;
    cupvaSQDFOpen(countHdl);
    cupvaSQDFTrig(countHdl);
    cupvaSQDFSync(countHdl);
    cupvaSQDFClose(countHdl);

    return 0;
}
