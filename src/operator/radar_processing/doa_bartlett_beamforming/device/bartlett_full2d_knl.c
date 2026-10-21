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

static inline void readNeighborsFromSpectrum2D(int32_t numAz, int32_t numEl, int32_t enableInterpolation,
                                               float *powerSpectrum, int32_t peakIdx, float maxPower,
                                               BartlettNeighborData *out)
{
    int32_t peakAz = peakIdx / numEl;
    int32_t peakEl = peakIdx - peakAz * numEl;

    out->peakAzIdx = peakAz;
    out->peakElIdx = peakEl;
    out->maxPower  = maxPower;

    const int32_t peakRowOfs = peakAz * numEl;
    float peakPower          = powerSpectrum[peakRowOfs + peakEl];

    int32_t azL = max(0, peakAz - 1);
    int32_t azR = min(numAz - 1, peakAz + 1);
    int32_t elL = max(0, peakEl - 1);
    int32_t elH = min(numEl - 1, peakEl + 1);

    float azY0 = powerSpectrum[azL * numEl + peakEl];
    float azY2 = powerSpectrum[azR * numEl + peakEl];
    float elY0 = powerSpectrum[peakRowOfs + elL];
    float elY2 = powerSpectrum[peakRowOfs + elH];

    out->azY0 = mux(enableInterpolation, azY0, peakPower);
    out->azY2 = mux(enableInterpolation, azY2, peakPower);
    out->elY0 = mux(enableInterpolation, elY0, peakPower);
    out->elY2 = mux(enableInterpolation, elY2, peakPower);
    out->azY1 = peakPower;
    out->elY1 = peakPower;
}

void bartlettKnl(BartlettContext *ctx, int32_t detIdx, BartlettNeighborData *outNeighbors)
{
    const int32_t numAngleBins = ctx->numAngleBins;

    int32_t *snapshot    = ctx->snapshotBuffers[ctx->pingPong] + detIdx * ctx->steeringStrideElements;
    float *powerSpectrum = ctx->powerSpectrum;

    const int32_t numBatches = (numAngleBins + STEERING_ANGLE_BATCH_SIZE - 1) / STEERING_ANGLE_BATCH_SIZE;
    int32_t steerBuf         = 0;

    cupvaModifyAgenCfgBase(&ctx->cfgSnapTpl, snapshot);

    /* Each batch transfers only the angles it consumes; the last one is short. */
    cupvaSQDFUpdateTileSize(*ctx->steeringSqdfHdl, 0,
                            min(numAngleBins, STEERING_ANGLE_BATCH_SIZE) * ctx->steeringStridePerAngle, 1);
    cupvaSQDFUpdateAddr(*ctx->steeringSqdfHdl, 0, ctx->steeringBase, ctx->steeringMemType,
                        cupvaGetVmemAddress(ctx->steeringBuffers[0]), MEMTYPE_VMEM);
    cupvaSQDFFlushAndTrig(*ctx->steeringSqdfHdl);

    for (int32_t batch = 0; batch < numBatches; batch++)
    {
        const int32_t angleStart    = batch * STEERING_ANGLE_BATCH_SIZE;
        const int32_t angleEnd      = min(angleStart + STEERING_ANGLE_BATCH_SIZE, numAngleBins);
        const int32_t anglesInBatch = angleEnd - angleStart;

        int16_t *steer = ctx->steeringBuffers[steerBuf];
        cupvaSQDFSync(*ctx->steeringSqdfHdl);

        if (batch + 1 < numBatches)
        {
            int64_t l2Ofs = (batch + 1) * ctx->steeringBatchStride;
            const int32_t nextAngles =
                min(numAngleBins - (batch + 1) * STEERING_ANGLE_BATCH_SIZE, STEERING_ANGLE_BATCH_SIZE);
            cupvaSQDFUpdateTileSize(*ctx->steeringSqdfHdl, 0, nextAngles * ctx->steeringStridePerAngle, 1);
            cupvaSQDFUpdateAddr(*ctx->steeringSqdfHdl, 0, ctx->steeringBase + l2Ofs, ctx->steeringMemType,
                                cupvaGetVmemAddress(ctx->steeringBuffers[steerBuf ^ 1]), MEMTYPE_VMEM);
            cupvaSQDFFlushAndTrig(*ctx->steeringSqdfHdl);
        }

        bartlettBeamformBatch(ctx, ctx->cfgSteer2D, steer, anglesInBatch, &powerSpectrum[angleStart]);

        steerBuf ^= 1;
    }

    float maxPower  = 0.0f;
    int32_t peakIdx = peakSearchVec(&ctx->cfgPowSearch, ctx->powerSpectrum, numAngleBins, &maxPower);

    readNeighborsFromSpectrum2D(ctx->numAz, ctx->numEl, ctx->enableInterpolation, powerSpectrum, peakIdx, maxPower,
                                outNeighbors);
}

void bartlettInitAgensFull2D(BartlettContext *ctx, BartlettGlobalParams const *GP)
{
    bartlettInitAgensCommon(ctx, GP);

    const int32_t vecw        = chess_elementsof(vfloatx);
    const int32_t numVecIters = ctx->numAngleBins / vecw;

    AgenWrapper wrapper;

    agen aPow    = init((vfloatx *)ctx->powerSpectrum);
    wrapper.size = sizeof(float);
    wrapper.n1   = numVecIters;
    wrapper.s1   = vecw;
    INIT_AGEN1(aPow, wrapper);
    ctx->cfgPowSearch = extract_agen_cfg(aPow);
}
