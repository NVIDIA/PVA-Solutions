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

#include "bartlett_common.h"

static inline void readNeighborsFromSpectrum(int32_t numAz, int32_t numEl, int32_t azFromEl0,
                                             int32_t enableInterpolation, int32_t peakAzIdx, int32_t peakElIdx,
                                             float maxPower, float *azSpectrum, float *localSpectrum,
                                             int32_t numLocalAz, int32_t localAzPeakIdx, BartlettNeighborData *out)
{
    out->peakAzIdx = peakAzIdx;
    out->peakElIdx = peakElIdx;
    out->maxPower  = maxPower;

    const int32_t elL       = max(0, peakElIdx - 1);
    const int32_t elH       = min(numEl - 1, peakElIdx + 1);
    const int32_t elRowBase = localAzPeakIdx * numEl;

    float elY0 = localSpectrum[elRowBase + elL];
    float elY1 = localSpectrum[elRowBase + peakElIdx];
    float elY2 = localSpectrum[elRowBase + elH];

    const int32_t azL_el0   = max(0, peakAzIdx - 1);
    const int32_t azR_el0   = min(numAz - 1, peakAzIdx + 1);
    const int32_t azL_local = max(0, localAzPeakIdx - 1);
    const int32_t azR_local = min(numLocalAz - 1, localAzPeakIdx + 1);

    float azY0_el0   = azSpectrum[azL_el0];
    float azY1_el0   = azSpectrum[peakAzIdx];
    float azY2_el0   = azSpectrum[azR_el0];
    float azY0_local = localSpectrum[azL_local * numEl + peakElIdx];
    float azY1_local = localSpectrum[localAzPeakIdx * numEl + peakElIdx];
    float azY2_local = localSpectrum[azR_local * numEl + peakElIdx];

    float azY0 = mux(azFromEl0, azY0_el0, azY0_local);
    float azY1 = mux(azFromEl0, azY1_el0, azY1_local);
    float azY2 = mux(azFromEl0, azY2_el0, azY2_local);

    // When interpolation is disabled, set neighbors equal to peak power
    // so the quadratic formula produces delta = 0 (no sub-bin offset)
    out->azY0 = mux(enableInterpolation, azY0, azY1);
    out->azY1 = azY1;
    out->azY2 = mux(enableInterpolation, azY2, azY1);
    out->elY0 = mux(enableInterpolation, elY0, elY1);
    out->elY1 = elY1;
    out->elY2 = mux(enableInterpolation, elY2, elY1);
}

static float bartlettBeamformBatchWithPeak(BartlettContext *ctx, AgenCFG *cfgSteer2D, int16_t *steer,
                                           int32_t anglesInBatch, float *outPtr, int32_t *outPeakIdx)
{
    bartlettBeamformBatch(ctx, cfgSteer2D, steer, anglesInBatch, outPtr);

    cupvaModifyAgenCfgBase(&ctx->cfgOutPeakScan, outPtr);
    float maxPower = 0.0f;
    *outPeakIdx    = peakSearchVec(&ctx->cfgOutPeakScan, outPtr, anglesInBatch, &maxPower);
    return maxPower;
}

void bartlettSeparableKnl(BartlettContext *ctx, int32_t detIdx, BartlettNeighborData *outNeighbors)
{
    const int32_t numAz                  = ctx->numAz;
    const int32_t numEl                  = ctx->numEl;
    const int32_t azNeighborhood         = ctx->azNeighborhood;
    const int32_t steeringStrideElements = ctx->steeringStrideElements;
    const int64_t steeringBatchStride    = ctx->steeringBatchStride;

    int32_t *snapshot = ctx->snapshotBuffers[ctx->pingPong] + detIdx * steeringStrideElements;

    /* Start each batch from a known state. */
    if (detIdx == 0)
    {
        ctx->nextPeakValid = 0;
        ctx->azSpecBuf     = 0;
    }

    float *azSpectrum    = ctx->powerSpectrum + ctx->azSpecBuf * numAz;
    float *localSpectrum = ctx->powerSpectrum + 2 * numAz;

    // Step 1: Azimuth search at elevation ~0 (steering vectors pre-loaded by runner).
    // Detections after the first reuse the peak the previous call computed.
    int32_t peakAzimuthIdx;
    if (ctx->nextPeakValid)
    {
        peakAzimuthIdx = ctx->nextPeak;
    }
    else
    {
        cupvaModifyAgenCfgBase(&ctx->cfgSnapTpl, snapshot);
        bartlettBeamformBatchWithPeak(ctx, ctx->cfgStep1Steer2D, ctx->step1SteeringBuffer, numAz, azSpectrum,
                                      &peakAzimuthIdx);
    }

    // Step 2: Elevation search in azimuth neighborhood
    int32_t azStartIdx           = max(0, peakAzimuthIdx - azNeighborhood);
    const int64_t localStartAddr = ctx->steeringBase + azStartIdx * numEl * ctx->steeringStridePerAngle;

    int32_t azEndIdx              = min(numAz - 1, peakAzimuthIdx + azNeighborhood);
    int32_t numLocalAz            = azEndIdx - azStartIdx + 1;
    int32_t numLocalAngles        = numLocalAz * numEl;
    const int32_t numStep2Batches = (numLocalAngles + STEERING_ANGLE_BATCH_SIZE - 1) / STEERING_ANGLE_BATCH_SIZE;

    /* A detection consumes numLocalAngles of steering, not a full batch. They are contiguous
     * from localStartAddr, so one row covers them. */
    cupvaSQDFUpdateTileSize(*ctx->steeringSqdfHdl, 0,
                            min(numLocalAngles, STEERING_ANGLE_BATCH_SIZE) * ctx->steeringStridePerAngle, 1);

    cupvaSQDFUpdateAddr(*ctx->steeringSqdfHdl, 0, localStartAddr, ctx->steeringMemType,
                        cupvaGetVmemAddress(ctx->steeringBuffers[0]), MEMTYPE_VMEM);
    cupvaSQDFFlushAndTrig(*ctx->steeringSqdfHdl);

    /* The transfer is in flight. Step 1 of the next detection does not depend on it and has to
     * happen anyway, so run it here and the DMA latency lands under real work, not an idle sync. */
    const int32_t hasNext = (detIdx + 1) < ctx->batchSize;
    if (hasNext)
    {
        int32_t *snapNext = ctx->snapshotBuffers[ctx->pingPong] + (detIdx + 1) * steeringStrideElements;
        cupvaModifyAgenCfgBase(&ctx->cfgSnapTpl, snapNext);
        bartlettBeamformBatchWithPeak(ctx, ctx->cfgStep1Steer2D, ctx->step1SteeringBuffer, numAz,
                                      ctx->powerSpectrum + (ctx->azSpecBuf ^ 1) * numAz, &ctx->nextPeak);
    }
    ctx->nextPeakValid = hasNext;

    /* Step 2 reads this detection's snapshot. Restore unconditionally: the lookahead above left
     * the agen on i+1, and the path that skips step 1 never set it at all. */
    cupvaModifyAgenCfgBase(&ctx->cfgSnapTpl, snapshot);

    int32_t steerBuf = 0;

    float *outPtr = localSpectrum;
    for (int32_t batch = 0; batch < numStep2Batches; batch++)
    {
        const int32_t angleStart    = batch * STEERING_ANGLE_BATCH_SIZE;
        const int32_t angleEnd      = min(angleStart + STEERING_ANGLE_BATCH_SIZE, numLocalAngles);
        const int32_t anglesInBatch = angleEnd - angleStart;

        int16_t *steer = ctx->steeringBuffers[steerBuf];
        cupvaSQDFSync(*ctx->steeringSqdfHdl);

        if (batch + 1 < numStep2Batches)
        {
            const int32_t remaining = numLocalAngles - (batch + 1) * STEERING_ANGLE_BATCH_SIZE;
            cupvaSQDFUpdateTileSize(*ctx->steeringSqdfHdl, 0,
                                    min(remaining, STEERING_ANGLE_BATCH_SIZE) * ctx->steeringStridePerAngle, 1);
            cupvaSQDFUpdateAddr(*ctx->steeringSqdfHdl, 0, localStartAddr + (batch + 1) * steeringBatchStride,
                                ctx->steeringMemType, cupvaGetVmemAddress(ctx->steeringBuffers[steerBuf ^ 1]),
                                MEMTYPE_VMEM);
            cupvaSQDFFlushAndTrig(*ctx->steeringSqdfHdl);
        }

        bartlettBeamformBatch(ctx, ctx->cfgSteer2D, steer, anglesInBatch, outPtr);
        outPtr += anglesInBatch;

        steerBuf ^= 1;
    }

    // Find 2D peak in local spectrum
    float maxPower = 0.0f;
    cupvaModifyAgenCfgBase(&ctx->cfgLocalPowSearch, localSpectrum);
    int32_t peakLinear = peakSearchVec(&ctx->cfgLocalPowSearch, localSpectrum, numLocalAngles, &maxPower);

    int32_t localAzPeak = peakLinear / numEl;
    int32_t peakElIdx   = peakLinear - localAzPeak * numEl;

    int32_t finalPeakAzIdx = ctx->azFromEl0 ? peakAzimuthIdx : (azStartIdx + localAzPeak);

    readNeighborsFromSpectrum(numAz, numEl, ctx->azFromEl0, ctx->enableInterpolation, finalPeakAzIdx, peakElIdx,
                              maxPower, azSpectrum, localSpectrum, numLocalAz, localAzPeak, outNeighbors);

    /* The next detection's spectrum was written to the other half; hand it over. */
    if (ctx->nextPeakValid)
    {
        ctx->azSpecBuf ^= 1;
    }
}

void bartlettInitAgensSeparable(BartlettContext *ctx, BartlettGlobalParams const *GP)
{
    bartlettInitAgensCommon(ctx, GP);

    // Store azimuth channel bitmask for Step 1 masking
    // Bit i = 1 keeps VC i, bit i = 0 zeroes VC i in Step 1 azimuth search
    uint32_t maskLo       = GP->azChannelMaskLo;
    uint32_t maskHi       = GP->azChannelMaskHi;
    ctx->azChannelMaskLo  = maskLo;
    ctx->azChannelMaskHi  = maskHi;
    ctx->hasAzChannelMask = (maskLo | maskHi) ? 1 : 0;

    const int32_t numVecIterations       = ctx->numVecIterations;
    const int32_t steeringStrideElements = ctx->steeringStrideElements;
    const int32_t numAz                  = ctx->numAz;

    AgenWrapper wrap2DStep1;
    wrap2DStep1.size = sizeof(int16_t);
    wrap2DStep1.n1   = numVecIterations;
    wrap2DStep1.s1   = 16;
    wrap2DStep1.n2   = STEP1_AZ_BATCH_SIZE / 4;
    wrap2DStep1.s2   = steeringStrideElements * 4;

    for (int32_t i = 0; i < 4; i++)
    {
        agen aS = init((vshortx *)(ctx->step1SteeringBuffer + i * steeringStrideElements));
        INIT_AGEN2(aS, wrap2DStep1);
        ctx->cfgStep1Steer2D[i] = extract_agen_cfg(aS);
    }

    const int32_t vecw = chess_elementsof(vfloatx);
    /* The azimuth spectrum is double buffered (step 1 of detection i+1 runs while detection i
     * still needs its own), so the local spectrum starts after both halves. */
    float *localSpectrum       = ctx->powerSpectrum + 2 * numAz;
    const int32_t maxLocalVecs = (SEPARABLE_POWER_SPECTRUM_SIZE - 2 * numAz) / vecw;

    AgenWrapper wrapLocalPow;
    wrapLocalPow.size = sizeof(float);
    wrapLocalPow.n1   = maxLocalVecs;
    wrapLocalPow.s1   = vecw;

    agen aLocalPow = init((vfloatx *)localSpectrum);
    INIT_AGEN1(aLocalPow, wrapLocalPow);
    ctx->cfgLocalPowSearch = extract_agen_cfg(aLocalPow);
}
