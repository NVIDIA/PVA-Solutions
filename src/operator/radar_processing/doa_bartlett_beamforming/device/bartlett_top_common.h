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

#ifndef BARTLETT_TOP_COMMON_H
#define BARTLETT_TOP_COMMON_H

#include "bartlett_knl.h"

#include <cupva_device.h>

typedef struct
{
    BartlettGlobalParams *GP;
    uint32_t *detectionCount;
    int32_t *maxTargetCount;
    int32_t *targetCount;
    int32_t *targetMap;

    int16_t *steeringVectorsBuffer;
    int32_t *snapshotBuffer;
    float *powerSpectrum;
    float *anglesOutputBuffer;

    int32_t *peakAzIdx;
    int32_t *peakElIdx;
    float *azY0, *azY1, *azY2;
    float *elY0, *elY1, *elY2;
    float *peakMagnitude;

    SequenceDataFlowHandler *detectionCntSqdfHdl;
    SequenceDataFlowHandler *snapshotsSqdfHdl;
    SequenceDataFlowHandler *anglesOutputSqdfHdl;
    SequenceDataFlowHandler *steeringVectorsSqdfHdl;
    SequenceDataFlowHandler *targetCountAndMapSqdfHdl;

    ExtMemPointer *snapshotsExtAddr;
    ExtMemPointer *anglesOutputExtAddr;
    ExtMemPointer *targetCountExtAddr;

    int16_t *step1SteeringBuffer; // NULL for full2D
    SequenceDataFlowHandler *step1SteeringSqdfHdl;

    ExtMemPointer *steeringDramExtAddr; // Non-NULL: DRAM (separable), NULL: L2 (full2D)
} BartlettVmemPtrs;

typedef void (*BartlettKernelFunc)(BartlettContext *ctx, int32_t detIdx, BartlettNeighborData *outNeighbors);
typedef void (*BartlettAgenInitFunc)(BartlettContext *ctx, BartlettGlobalParams const *GP);

static inline int32_t bartlettRun(const BartlettVmemPtrs *vmem, BartlettKernelFunc kernelFunc,
                                  BartlettAgenInitFunc agenInitFunc)
{
    BartlettContext ctx;

    // Load detection count
    cupvaSQDFOpen(*vmem->detectionCntSqdfHdl);
    cupvaSQDFTrig(*vmem->detectionCntSqdfHdl);
    cupvaSQDFSync(*vmem->detectionCntSqdfHdl);
    cupvaSQDFClose(*vmem->detectionCntSqdfHdl);

    // Initialize context
    const int32_t steerSize     = STEERING_BATCH_WIDTH * STEERING_BATCH_HEIGHT;
    const int32_t snapSize      = SNAPSHOT_BUFFER_WIDTH * SNAPSHOT_BUFFER_HEIGHT;
    const int32_t anglesOutSize = SNAPSHOT_BATCH_SIZE * ANGLES_NUM_ROWS_WITH_POWER;

    ctx.steeringBuffers[0]     = vmem->steeringVectorsBuffer;
    ctx.steeringBuffers[1]     = vmem->steeringVectorsBuffer + steerSize;
    ctx.steeringSqdfHdl        = vmem->steeringVectorsSqdfHdl;
    ctx.snapshotBuffers[0]     = vmem->snapshotBuffer;
    ctx.snapshotBuffers[1]     = vmem->snapshotBuffer + snapSize;
    ctx.powerSpectrum          = vmem->powerSpectrum;
    ctx.peakAzIdx              = vmem->peakAzIdx;
    ctx.peakElIdx              = vmem->peakElIdx;
    ctx.azY0                   = vmem->azY0;
    ctx.azY1                   = vmem->azY1;
    ctx.azY2                   = vmem->azY2;
    ctx.elY0                   = vmem->elY0;
    ctx.elY1                   = vmem->elY1;
    ctx.elY2                   = vmem->elY2;
    ctx.peakMagnitude          = vmem->peakMagnitude;
    ctx.anglesOutputBuffers[0] = vmem->anglesOutputBuffer;
    ctx.anglesOutputBuffers[1] = vmem->anglesOutputBuffer + anglesOutSize;
    ctx.anglesOutputSqdfHdl    = vmem->anglesOutputSqdfHdl;
    ctx.anglesOutputExtAddr    = vmem->anglesOutputExtAddr;
    ctx.outputPingPong         = 0;
    ctx.outputInFlight         = false;
    ctx.pingPong               = 0;
    ctx.nextPeakValid          = 0;
    ctx.azSpecBuf              = 0;

    // Steering vector source: DRAM (separable) or L2 (full2D)
    ExtMemPointer l2Ptr;
    if (!vmem->steeringDramExtAddr)
        cupvaGetL2ExtMemPointer(&l2Ptr, 0U);
    ctx.steeringBase = vmem->steeringDramExtAddr ? vmem->steeringDramExtAddr->base + vmem->steeringDramExtAddr->offset
                                                 : l2Ptr.base + l2Ptr.offset;
    ctx.steeringMemType = vmem->steeringDramExtAddr ? MEMTYPE_DRAM : MEMTYPE_L2SRAM;

    // Precompute beamforming constants
    vintx bfRoundW   = replicatew(BEAMFORMING_ROUND);
    ctx.bfRoundDV.lo = bfRoundW;
    ctx.bfRoundDV.hi = bfRoundW;
    ctx.zeroVec      = replicatew(0);

    ctx.step1SteeringBuffer  = vmem->step1SteeringBuffer;
    ctx.step1SteeringSqdfHdl = vmem->step1SteeringBuffer ? vmem->step1SteeringSqdfHdl : NULL;

    // Initialize AGENs and cache GP-derived constants
    BartlettGlobalParams const *GP = vmem->GP;
    agenInitFunc(&ctx, GP);

    // Open SQDFs
    cupvaSQDFOpen(*vmem->snapshotsSqdfHdl);
    cupvaSQDFOpen(*vmem->anglesOutputSqdfHdl);
    cupvaSQDFOpen(*vmem->steeringVectorsSqdfHdl);
    if (vmem->step1SteeringSqdfHdl)
        cupvaSQDFOpen(*vmem->step1SteeringSqdfHdl);

    // Main processing loop
    *vmem->targetCount    = 0;
    *vmem->detectionCount = *vmem->detectionCount > (uint32_t)*vmem->maxTargetCount ? (uint32_t)*vmem->maxTargetCount
                                                                                    : *vmem->detectionCount;

    if (*vmem->detectionCount != 0)
    {
        *vmem->targetCount = *vmem->detectionCount;
        for (int32_t i = 0; i < (int32_t)*vmem->detectionCount; i++) vmem->targetMap[i] = i;

        const int32_t snapStride = SNAPSHOT_STRIDE_BYTES(ctx.numVC);
        const int32_t numBatches = (*vmem->detectionCount + SNAPSHOT_BATCH_SIZE - 1) / SNAPSHOT_BATCH_SIZE;

        // Trigger first snapshot DMA (overlaps with steering pre-load below)
        updateAndTriggerInputSQDF(*vmem->snapshotsSqdfHdl, 0, snapStride, vmem->snapshotsExtAddr,
                                  ctx.snapshotBuffers[0]);

        // Pre-load el=0 steering vectors (separable mode only)
        if (vmem->step1SteeringSqdfHdl)
        {
            const int64_t vcStride = ctx.steeringStridePerAngle;
            const int64_t srcPitch = (int64_t)ctx.numEl * vcStride;
            const int64_t srcBase  = ctx.steeringBase + (int64_t)ctx.elZeroIdx * vcStride;

            cupvaSQDFUpdateTileSize(*vmem->step1SteeringSqdfHdl, 0, (int32_t)vcStride, ctx.numAz);
            // Safe under separable constraints: numEl<=255, vcStride<=256, so pitches fit uint16_t.
#ifdef SQDF_NO_PITCH_OVERLOAD
            // cuPVA < 2.7.0: 8-arg cupvaSQDFUpdateAddr not available, use 6-arg + manual LP_CNTL
            cupvaSQDFUpdateAddr(*vmem->step1SteeringSqdfHdl, 0, (uint64_t)srcBase, ctx.steeringMemType,
                                (uint64_t)cupvaGetVmemAddress(ctx.step1SteeringBuffer), MEMTYPE_VMEM);
            {
                uint32_t descId = vmem->step1SteeringSqdfHdl->metadata.transfers[0].desc_id;
                uint32_t lpAddr = CUPVA_GET_DESC_ATTR_ADDR(descId, LP_CNTL);
                uint32_t lpVal  = CUPVA_DMA_FIELD(LP_CNTL, SLP_ADV, (uint32_t)(uint16_t)srcPitch) |
                                 CUPVA_DMA_FIELD(LP_CNTL, DLP_ADV, (uint32_t)(uint16_t)vcStride);
                uint32_t idx                                               = vmem->step1SteeringSqdfHdl->entryCount;
                vmem->step1SteeringSqdfHdl->vpuCTable.entries[idx].address = lpAddr;
                vmem->step1SteeringSqdfHdl->vpuCTable.entries[idx].value   = lpVal;
                vmem->step1SteeringSqdfHdl->entryCount                     = idx + 1;
            }
#else
            // cuPVA >= 2.7.0: use 8-arg overload with line pitches
            cupvaSQDFUpdateAddr(*vmem->step1SteeringSqdfHdl, 0, (uint64_t)srcBase, ctx.steeringMemType,
                                (uint16_t)srcPitch, (uint64_t)cupvaGetVmemAddress(ctx.step1SteeringBuffer),
                                MEMTYPE_VMEM, (uint16_t)vcStride);
#endif
            cupvaSQDFFlushAndTrig(*vmem->step1SteeringSqdfHdl);
            cupvaSQDFSync(*vmem->step1SteeringSqdfHdl);

            // Apply azimuth channel mask to Step 1 steering vectors in-place.
            // For each masked VC, zero its (real, imag) pair across all numAz rows.
            if (ctx.hasAzChannelMask)
            {
                int32_t azM, vc;
                const int32_t numAz    = ctx.numAz;
                const int32_t numVC    = ctx.numVC;
                const int32_t stride32 = ctx.steeringStrideElements / 2; // stride in int32_t units
                uint32_t maskLo        = ctx.azChannelMaskLo;
                uint32_t maskHi        = ctx.azChannelMaskHi;

                for (vc = 0; vc < numVC; vc++)
                {
                    uint32_t bit = (vc < 32) ? ((maskLo >> vc) & 1U) : ((maskHi >> (vc - 32)) & 1U);
                    if (!bit)
                    {
                        int32_t *col = (int32_t *)(ctx.step1SteeringBuffer + vc * 2);
                        for (azM = 0; azM < numAz; azM++)
                        {
                            *col = 0;
                            col += stride32;
                        }
                    }
                }
            }
        }

        for (int32_t b = 0; b < numBatches; b++)
        {
            const int32_t start = b * SNAPSHOT_BATCH_SIZE;
            const int32_t end   = min(start + SNAPSHOT_BATCH_SIZE, (int32_t)*vmem->detectionCount);
            const int32_t size  = end - start;

            cupvaSQDFSync(*vmem->snapshotsSqdfHdl);

            if (b + 1 < numBatches)
                updateAndTriggerInputSQDF(*vmem->snapshotsSqdfHdl, (b + 1) * SNAPSHOT_BATCH_SIZE, snapStride,
                                          vmem->snapshotsExtAddr, ctx.snapshotBuffers[ctx.pingPong ^ 1]);

            BartlettNeighborData neighbors;

            ctx.batchSize = size;

            for (int32_t i = 0; i < size; i++)
            {
                kernelFunc(&ctx, i, &neighbors);
                bartlettWriteNeighbors(&ctx, i, &neighbors);
            }

            bartlettProcessBatchOutput(&ctx, size, start);

            ctx.pingPong ^= 1;
        }

        if (ctx.outputInFlight)
            cupvaSQDFSync(*vmem->anglesOutputSqdfHdl);
    }

    // Close SQDFs
    cupvaSQDFClose(*vmem->snapshotsSqdfHdl);
    cupvaSQDFClose(*vmem->anglesOutputSqdfHdl);
    cupvaSQDFClose(*vmem->steeringVectorsSqdfHdl);
    if (vmem->step1SteeringSqdfHdl)
        cupvaSQDFClose(*vmem->step1SteeringSqdfHdl);

    // Write target count + map output (addresses set by host via OffsetPointer / UpdateDataFlows)
    cupvaSQDFOpen(*vmem->targetCountAndMapSqdfHdl);
    cupvaSQDFTrig(*vmem->targetCountAndMapSqdfHdl);
    cupvaSQDFSync(*vmem->targetCountAndMapSqdfHdl);
    cupvaSQDFClose(*vmem->targetCountAndMapSqdfHdl);

    return 0;
}

#endif // BARTLETT_TOP_COMMON_H
