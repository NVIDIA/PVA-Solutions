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

#ifndef PVA_SOLUTIONS_PRIV_BARTLETT_KNL_H
#define PVA_SOLUTIONS_PRIV_BARTLETT_KNL_H

#include "bartlett_params.h"

#include <cupva_device.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct
{
    // Steering vectors (double-buffered)
    int16_t *steeringBuffers[2];
    int32_t batchSize; /* Detections in the batch being processed; the last one may be short. */

    /* Separable software pipeline: step 1 of detection i+1 runs inside call i, so the step-2
     * steering DMA has work to hide behind. Only these need to survive across calls. */
    int32_t nextPeakValid;
    int32_t nextPeak;
    int32_t azSpecBuf; /* Which half of the azimuth spectrum is live; toggles per detection. */
    SequenceDataFlowHandler *steeringSqdfHdl;

    uint64_t steeringBase;
    MemType steeringMemType;

    // Step 1 steering (separable mode only)
    int16_t *step1SteeringBuffer;
    SequenceDataFlowHandler *step1SteeringSqdfHdl;

    // Snapshot (double-buffered)
    int32_t *snapshotBuffers[2];

    // Power spectrum
    float *powerSpectrum;

    // Per-batch interpolation arrays
    int32_t *peakAzIdx;
    int32_t *peakElIdx;
    float *azY0, *azY1, *azY2;
    float *elY0, *elY1, *elY2;
    float *peakMagnitude;

    // Output buffer (double-buffered)
    float *anglesOutputBuffers[2];
    SequenceDataFlowHandler *anglesOutputSqdfHdl;
    ExtMemPointer *anglesOutputExtAddr;

    // State
    int32_t outputPingPong;
    bool outputInFlight;
    int32_t pingPong;

    // AGEN configurations
    AgenCFG cfgSnapTpl;
    AgenCFG cfgSteerTpl;
    AgenCFG cfgSteer2D[4];
    /* One shape for the fused single-pass path; the v=0 and v=1 cursors rebase it. */
    AgenCFG cfgSteerFused[2];
    AgenCFG cfgStep1Steer2D[4];
    AgenCFG cfgPowSearch;
    AgenCFG cfgLocalPowSearch;
    AgenCFG cfgOutSpectrum;
    AgenCFG cfgOutSpectrumVec; /* stride-8 twin: one vstore retires eight angles */
    AgenCFG cfgOutPeakScan;

    // Output processing AGENs
    AgenCFG cfgAzY0, cfgAzY1, cfgAzY2;
    AgenCFG cfgElY0, cfgElY1, cfgElY2;
    AgenCFG cfgPeakAzIdx, cfgPeakElIdx;
    AgenCFG cfgPowIn;
    AgenCFG cfgOutAz, cfgOutEl;
    AgenCFG cfgOutPower;

    // Cached GP scalars
    int32_t numAz;
    int32_t numEl;
    int32_t numAngleBins;
    int32_t azNeighborhood;
    int32_t enableInterpolation;
    int32_t enablePowerOutput;
    int32_t azFromEl0;
    int32_t elZeroIdx;
    float azScale;
    float elScale;
    float azMin;
    float elMin;

    // Precomputed constants
    int32_t numVC;
    int32_t numVecIterations;
    int32_t steeringStrideElements;
    int64_t steeringStridePerAngle;
    int64_t steeringBatchStride;

    // Beamforming constants
    dvintx bfRoundDV;
    vintx zeroVec;

    // Azimuth channel mask for Step 1 (separable mode only)
    // Raw bitmask stored as two uint32_t halves; bit i = 1 keeps VC i, bit i = 0 zeroes it
    uint32_t azChannelMaskLo; // Bits  0-31 (VCs  0-31)
    uint32_t azChannelMaskHi; // Bits 32-63 (VCs 32-63)
    int32_t hasAzChannelMask; // Non-zero if azChannelMask is active
} BartlettContext;

typedef struct
{
    int32_t peakAzIdx;
    int32_t peakElIdx;
    float azY0, azY1, azY2;
    float elY0, elY1, elY2;
    float maxPower;
} BartlettNeighborData;

void bartlettInitAgensCommon(BartlettContext *ctx, BartlettGlobalParams const *GP);
void bartlettInitAgensSeparable(BartlettContext *ctx, BartlettGlobalParams const *GP);
void bartlettInitAgensFull2D(BartlettContext *ctx, BartlettGlobalParams const *GP);

void bartlettKnl(BartlettContext *ctx, int32_t detIdx, BartlettNeighborData *outNeighbors);
void bartlettSeparableKnl(BartlettContext *ctx, int32_t detIdx, BartlettNeighborData *outNeighbors);

static inline void bartlettWriteNeighbors(BartlettContext *ctx, int32_t detIdx, const BartlettNeighborData *neighbors)
{
    ctx->peakAzIdx[detIdx]     = neighbors->peakAzIdx;
    ctx->peakElIdx[detIdx]     = neighbors->peakElIdx;
    ctx->azY0[detIdx]          = neighbors->azY0;
    ctx->azY1[detIdx]          = neighbors->azY1;
    ctx->azY2[detIdx]          = neighbors->azY2;
    ctx->elY0[detIdx]          = neighbors->elY0;
    ctx->elY1[detIdx]          = neighbors->elY1;
    ctx->elY2[detIdx]          = neighbors->elY2;
    ctx->peakMagnitude[detIdx] = neighbors->maxPower;
}

void bartlettProcessBatchOutput(BartlettContext *ctx, int32_t batchSize, int32_t batchStartIdx);

static inline void updateAndTriggerInputSQDF(SequenceDataFlowHandler &sqdfHdl, int32_t batchStartIdx,
                                             int32_t snapshotStride, ExtMemPointer *snapshotsExtPtr, int32_t *vmemDst)
{
    int64_t srcAddr = snapshotsExtPtr->base + snapshotsExtPtr->offset + batchStartIdx * snapshotStride;
    cupvaSQDFUpdateAddr(sqdfHdl, 0, srcAddr, MEMTYPE_DRAM, cupvaGetVmemAddress(vmemDst), MEMTYPE_VMEM);
    cupvaSQDFFlushAndTrig(sqdfHdl);
}

void bartlettBeamformBatch(BartlettContext *ctx, AgenCFG *cfgSteer2D, int16_t *steer, int32_t anglesInBatch,
                           float *outPtr);

#endif // PVA_SOLUTIONS_PRIV_BARTLETT_KNL_H
