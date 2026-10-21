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

// ============================================================================
// Beamforming inner-loop helper macros
// Macro is used to avoid function call overhead in the inner loop.
// ============================================================================
#define BF_MUL(snap, steerAgen) ((dvcmulwhw((snap), vshort_load(steerAgen)) + bfRoundDV) >> BEAMFORMING_SHIFT)

/* Multiply, shift and accumulate in one issue: VShiftAdd shifts right by -n and adds acc. */
#define BF_MAC(snap, steerAgen, acc) \
    dvshiftadd((dvcmulwhw((snap), vshort_load(steerAgen)) + bfRoundDV), -BEAMFORMING_SHIFT, (acc))

// Reduce 4 accumulators to float power and store via output AGEN
#define BF_REDUCE_4(a0, a1, a2, a3, aOut)             \
    float_store(reducePower((a0).lo, (a0).hi), aOut); \
    float_store(reducePower((a1).lo, (a1).hi), aOut); \
    float_store(reducePower((a2).lo, (a2).hi), aOut); \
    float_store(reducePower((a3).lo, (a3).hi), aOut)

static void bartlettComputeAngles(BartlettContext *ctx, int32_t numVecIters)
{
    const float azScale  = ctx->azScale;
    const float elScale  = ctx->elScale;
    const float azMin    = ctx->azMin;
    const float elMin    = ctx->elMin;
    const float azMaxIdx = (float)(ctx->numAz - 1);
    const float elMaxIdx = (float)(ctx->numEl - 1);

    const vfloatx vZero     = replicatef(0.0f);
    const vfloatx vAzMaxIdx = replicatef(azMaxIdx);
    const vfloatx vElMaxIdx = replicatef(elMaxIdx);

    agen aOutAz  = init_agen_from_cfg(ctx->cfgOutAz);
    agen aOutEl  = init_agen_from_cfg(ctx->cfgOutEl);
    agen aAzY0   = init_agen_from_cfg(ctx->cfgAzY0);
    agen aAzY1   = init_agen_from_cfg(ctx->cfgAzY1);
    agen aAzY2   = init_agen_from_cfg(ctx->cfgAzY2);
    agen aElY0   = init_agen_from_cfg(ctx->cfgElY0);
    agen aElY1   = init_agen_from_cfg(ctx->cfgElY1);
    agen aElY2   = init_agen_from_cfg(ctx->cfgElY2);
    agen aPeakAz = init_agen_from_cfg(ctx->cfgPeakAzIdx);
    agen aPeakEl = init_agen_from_cfg(ctx->cfgPeakElIdx);

    for (int32_t v = 0; v < numVecIters; v++) chess_prepare_for_pipelining
    {
        dvfloatx dvY0, dvY1, dvY2;
        dvY0.lo = vfloat_load(aAzY0);
        dvY0.hi = vfloat_load(aElY0);
        dvY1.lo = vfloat_load(aAzY1);
        dvY1.hi = vfloat_load(aElY1);
        dvY2.lo = vfloat_load(aAzY2);
        dvY2.hi = vfloat_load(aElY2);

        dvfloatx dvOfs = dvQuadraticPeakOffset(dvY0, dvY1, dvY2);

        vintx vPeakAzInt = vint_load(aPeakAz);
        vintx vPeakElInt = vint_load(aPeakEl);
        vfloatx vPeakAzF = vintx_vfp(vPeakAzInt);
        vfloatx vPeakElF = vintx_vfp(vPeakElInt);

        vfloatx vAzIdx = vPeakAzF + dvOfs.lo;
        vfloatx vElIdx = vPeakElF + dvOfs.hi;

        vAzIdx = vmux(vAzIdx < 0.0f, vZero, vAzIdx);
        vAzIdx = vmux(vAzIdx > azMaxIdx, vAzMaxIdx, vAzIdx);
        vElIdx = vmux(vElIdx < 0.0f, vZero, vElIdx);
        vElIdx = vmux(vElIdx > elMaxIdx, vElMaxIdx, vElIdx);

        vfloatx vAzAngle = vAzIdx * azScale + azMin;
        vfloatx vElAngle = vElIdx * elScale + elMin;

        vstore(vAzAngle, aOutAz);
        vstore(vElAngle, aOutEl);
    }
}

static inline void bartlettConvertPowerToDb(BartlettContext *ctx, int32_t numVecIters)
{
    const vfloatx vEpsilon     = replicatef(EPSILON_AVOID_LOG_ZERO);
    const vfloatx vHalfDbScale = replicatef(DB_SCALE_LOG2_TO_DB10 * 0.5f);
    const vfloatx vDbOffset    = replicatef(-(float)BEAMFORMING_QBITS * DB_SCALE_LOG2_TO_DB10);

    agen aPowIn    = init_agen_from_cfg(ctx->cfgPowIn);
    agen aOutPower = init_agen_from_cfg(ctx->cfgOutPower);

    for (int32_t v = 0; v < numVecIters; v++) chess_prepare_for_pipelining
    {
        vfloatx vPow = vfloat_load(aPowIn);
        vPow         = vPow + vEpsilon;
        vfloatx vDb  = vflog2(vPow) * vHalfDbScale + vDbOffset;

        vstore(vDb, aOutPower);
    }
}

static inline void bartlettTriggerOutputDma(BartlettContext *ctx, float *currentOutBuf, int32_t batchSize,
                                            int32_t batchStartIdx)
{
    if (ctx->outputInFlight)
        cupvaSQDFSync(*ctx->anglesOutputSqdfHdl);

    const int64_t baseAddr         = ctx->anglesOutputExtAddr->base + ctx->anglesOutputExtAddr->offset;
    const int32_t tileSizeBytes    = batchSize * sizeof(float);
    const int32_t batchOffsetBytes = batchStartIdx * sizeof(float);
    const int32_t numRows = ctx->enablePowerOutput ? ANGLES_NUM_ROWS_WITH_POWER : ANGLES_NUM_ROWS_WITHOUT_POWER;

    cupvaSQDFUpdateTileSize(*ctx->anglesOutputSqdfHdl, 0, tileSizeBytes, numRows);
    cupvaSQDFUpdateAddr(*ctx->anglesOutputSqdfHdl, 0, cupvaGetVmemAddress(currentOutBuf), MEMTYPE_VMEM,
                        baseAddr + batchOffsetBytes, MEMTYPE_DRAM);
    cupvaSQDFFlushAndTrig(*ctx->anglesOutputSqdfHdl);

    ctx->outputInFlight = true;
    ctx->outputPingPong ^= 1;
}

void bartlettProcessBatchOutput(BartlettContext *ctx, int32_t batchSize, int32_t batchStartIdx)
{
    float *currentOutBuf = ctx->anglesOutputBuffers[ctx->outputPingPong];
    float *outAz         = currentOutBuf + ANGLES_ROW_AZ * SNAPSHOT_BATCH_SIZE;
    float *outEl         = currentOutBuf + ANGLES_ROW_EL * SNAPSHOT_BATCH_SIZE;

    const int32_t vecw        = chess_elementsof(vfloatx);
    const int32_t numVecIters = (batchSize + vecw - 1) / vecw;

    cupvaModifyAgenCfgBase(&ctx->cfgOutAz, outAz);
    cupvaModifyAgenCfgBase(&ctx->cfgOutEl, outEl);

    bartlettComputeAngles(ctx, numVecIters);

    if (ctx->enablePowerOutput)
    {
        float *outPower = currentOutBuf + ANGLES_ROW_POWER * SNAPSHOT_BATCH_SIZE;
        cupvaModifyAgenCfgBase(&ctx->cfgOutPower, outPower);
        bartlettConvertPowerToDb(ctx, numVecIters);
    }

    bartlettTriggerOutputDma(ctx, currentOutBuf, batchSize, batchStartIdx);
}

typedef struct
{
    /* Lane selectors for bfPack4, which reads base + 0 .. base + 2 for base 0 and 4. */
    vintx one[7];
    vintx lowHalf;
} bfLaneMasks;

static inline void bfInitLaneMasks(bfLaneMasks *m)
{
    const vintx lane = vcollate_idx(replicatew(0));
    m->one[0]        = (lane == 0);
    m->one[1]        = (lane == 1);
    m->one[2]        = (lane == 2);
    m->one[4]        = (lane == 4);
    m->one[5]        = (lane == 5);
    m->one[6]        = (lane == 6);
    m->lowHalf       = (lane < 4);
}

static inline vintx bfPack4(vintx s0, vintx s1, vintx s2, vintx s3, const bfLaneMasks *m, int32_t base)
{
    return vmux(m->one[base + 0], s0, vmux(m->one[base + 1], s1, vmux(m->one[base + 2], s2, s3)));
}

static inline void bfStorePower8(vintx re, vintx im, agen &aPow)
{
    vfloatx fr = vintx_vfp(re);
    vfloatx fi = vintx_vfp(im);
    /* Operand order must match reducePower: vmaddf rounds once, so swapping the terms moves
     * the result by an ULP - enough to pick a different bin out of a tied spectrum. */
    vstore(vmaddf(fr, fr, vmulf(fi, fi)), aPow);
}

/* Pack four reduced angles into lanes base..base+3; two groups fill a vector. */
#define BF_PACK4(a0, a1, a2, a3, base, re, im)                                                        \
    do                                                                                                \
    {                                                                                                 \
        (re) = bfPack4(vsumr((a0).lo), vsumr((a1).lo), vsumr((a2).lo), vsumr((a3).lo), &lm_, (base)); \
        (im) = bfPack4(vsumr((a0).hi), vsumr((a1).hi), vsumr((a2).hi), vsumr((a3).hi), &lm_, (base)); \
    } while (0)

/*
 * One group of four angles: zero the accumulators, walk the snapshot once, pack into lanes
 * base..base+3. A macro, not a function, so the chess_storage pinning on the accumulators survives.
 */
#define BF_ACCUM4(base, re, im)                                                     \
    do                                                                              \
    {                                                                               \
        agen aSnap = snapTpl;                                                       \
        dvintx chess_storage(DV0) acc0;                                             \
        dvintx chess_storage(DV1) acc1;                                             \
        dvintx chess_storage(DV2) acc2;                                             \
        dvintx chess_storage(DV3) acc3;                                             \
        acc0.lo = zeroVec;                                                          \
        acc0.hi = zeroVec;                                                          \
        acc1.lo = zeroVec;                                                          \
        acc1.hi = zeroVec;                                                          \
        acc2.lo = zeroVec;                                                          \
        acc2.hi = zeroVec;                                                          \
        acc3.lo = zeroVec;                                                          \
        acc3.hi = zeroVec;                                                          \
                                                                                    \
        for (int32_t v = 0; v < numVecIterations; v++) chess_prepare_for_pipelining \
        chess_loop_range(1, 8)                                                      \
        {                                                                           \
            dvintx snap = dvint_load_di(aSnap);                                     \
            acc0        = BF_MAC(snap, aS0, acc0);                                  \
            acc1        = BF_MAC(snap, aS1, acc1);                                  \
            acc2        = BF_MAC(snap, aS2, acc2);                                  \
            acc3        = BF_MAC(snap, aS3, acc3);                                  \
        }                                                                           \
                                                                                    \
        BF_PACK4(acc0, acc1, acc2, acc3, (base), (re), (im));                       \
    } while (0)

static void bartlettBeamformBatchDirect(BartlettContext *ctx, AgenCFG *cfgSteer2D, int16_t *steer,
                                        int32_t anglesInBatch, float *outPtr)
{
    const int32_t numVecIterations       = ctx->numVecIterations;
    const int32_t steeringStrideElements = ctx->steeringStrideElements;
    const dvintx bfRoundDV               = ctx->bfRoundDV;
    const vintx zeroVec                  = ctx->zeroVec;

    /* One resident config per cursor. Copying an AgenCFG to a local per batch costs ~6%. */
    cupvaModifyAgenCfgBase(&cfgSteer2D[0], &steer[0]);
    cupvaModifyAgenCfgBase(&cfgSteer2D[1], &steer[steeringStrideElements]);
    cupvaModifyAgenCfgBase(&cfgSteer2D[2], &steer[steeringStrideElements * 2]);
    cupvaModifyAgenCfgBase(&cfgSteer2D[3], &steer[steeringStrideElements * 3]);

    agen aS0 = init_agen_from_cfg(cfgSteer2D[0]);
    agen aS1 = init_agen_from_cfg(cfgSteer2D[1]);
    agen aS2 = init_agen_from_cfg(cfgSteer2D[2]);
    agen aS3 = init_agen_from_cfg(cfgSteer2D[3]);

    agen snapTpl = init_agen_from_cfg(ctx->cfgSnapTpl);

    const int32_t anglesQuad = anglesInBatch & ~3;
    const int32_t anglesOct  = anglesInBatch & ~7;

    bfLaneMasks lm_;
    bfInitLaneMasks(&lm_);

    cupvaModifyAgenCfgBase(&ctx->cfgOutSpectrumVec, outPtr);
    agen aOutV = init_agen_from_cfg(ctx->cfgOutSpectrumVec);

    for (int32_t a = 0; a < anglesOct; a += 8) chess_prepare_for_pipelining
    {
        vintx reA, imA, reB, imB;
        BF_ACCUM4(0, reA, imA);
        BF_ACCUM4(4, reB, imB);
        bfStorePower8(vmux(lm_.lowHalf, reA, reB), vmux(lm_.lowHalf, imA, imB), aOutV);
    }

    cupvaModifyAgenCfgBase(&ctx->cfgOutSpectrum, &outPtr[anglesOct]);
    agen aOut = init_agen_from_cfg(ctx->cfgOutSpectrum);

    for (int32_t a = anglesOct; a < anglesQuad; a += 4) chess_prepare_for_pipelining
    {
        agen aSnap = snapTpl;
        dvintx chess_storage(DV0) acc0;
        dvintx chess_storage(DV1) acc1;
        dvintx chess_storage(DV2) acc2;
        dvintx chess_storage(DV3) acc3;
        acc0.lo = zeroVec;
        acc0.hi = zeroVec;
        acc1.lo = zeroVec;
        acc1.hi = zeroVec;
        acc2.lo = zeroVec;
        acc2.hi = zeroVec;
        acc3.lo = zeroVec;
        acc3.hi = zeroVec;

        for (int32_t v = 0; v < numVecIterations; v++) chess_prepare_for_pipelining
        chess_loop_range(1, 8)
        {
            dvintx snap = dvint_load_di(aSnap);
            acc0        = BF_MAC(snap, aS0, acc0);
            acc1        = BF_MAC(snap, aS1, acc1);
            acc2        = BF_MAC(snap, aS2, acc2);
            acc3        = BF_MAC(snap, aS3, acc3);
        }

        BF_REDUCE_4(acc0, acc1, acc2, acc3, aOut);
    }

    int32_t remainderOfs = anglesQuad * steeringStrideElements;
    for (int32_t a = anglesQuad; a < anglesInBatch; a++)
    {
        agen aSnap = snapTpl;
        cupvaModifyAgenCfgBase(&ctx->cfgSteerTpl, &steer[remainderOfs]);
        agen aSteer = init_agen_from_cfg(ctx->cfgSteerTpl);
        remainderOfs += steeringStrideElements;

        dvintx chess_storage(DV0) acc;
        acc.lo = zeroVec;
        acc.hi = zeroVec;
        for (int32_t v = 0; v < numVecIterations; v++) chess_prepare_for_pipelining
        chess_loop_range(1, 8)
        {
            dvintx snap = dvint_load_di(aSnap);
            acc         = BF_MAC(snap, aSteer, acc);
        }
        float_store(reducePower(acc.lo, acc.hi), aOut);
    }
}

void bartlettBeamformBatch(BartlettContext *ctx, AgenCFG *cfgSteer2D, int16_t *steer, int32_t anglesInBatch,
                           float *outPtr)
{
    const int32_t numVecIterations       = ctx->numVecIterations;
    const int32_t steeringStrideElements = ctx->steeringStrideElements;
    const dvintx bfRoundDV               = ctx->bfRoundDV;
    const vintx zeroVec                  = ctx->zeroVec;

    const int32_t anglesQuad    = anglesInBatch & ~3;
    const int32_t numQuadGroups = anglesQuad >> 2;

    agen aOut;

    // Use the direct (angle-outer, v-inner) path when numVecIterations > 2
    // or too few angle groups to pipeline.
    if (numQuadGroups < 4 || numVecIterations > 2)
    {
        bartlettBeamformBatchDirect(ctx, cfgSteer2D, steer, anglesInBatch, outPtr);
        return;
    }

    // ================================================================
    // Fused single-pass path (numVecIterations <= 2)
    //
    // All snapshots are preloaded into DV registers.  One 1D steering
    // AGEN per v-iteration walks the angle array in lockstep.  Each
    // quad-group computes all v-iterations and reduces to output in a
    // single inner-loop iteration.
    // Two separate loops avoid branches that would prevent pipelining.
    // ================================================================

    cupvaModifyAgenCfgBase(&ctx->cfgSteerFused[0], steer);
    agen aS0 = init_agen_from_cfg(ctx->cfgSteerFused[0]);

    /* Eight angles per vector store. */
    const int32_t anglesOct = anglesInBatch & ~7;
    bfLaneMasks lm_;
    bfInitLaneMasks(&lm_);
    cupvaModifyAgenCfgBase(&ctx->cfgOutSpectrumVec, outPtr);
    agen aOutV = init_agen_from_cfg(ctx->cfgOutSpectrumVec);
    cupvaModifyAgenCfgBase(&ctx->cfgOutSpectrum, &outPtr[anglesOct]);
    aOut = init_agen_from_cfg(ctx->cfgOutSpectrum);

    if (numVecIterations == 1)
    {
        agen aSnap                      = init_agen_from_cfg(ctx->cfgSnapTpl);
        dvintx chess_storage(DV4) snap0 = dvint_load_di(aSnap);

        for (int32_t a = 0; a < anglesOct; a += 8) chess_prepare_for_pipelining
        chess_loop_range(1, 32)
        {
            dvintx chess_storage(DV0) a0;
            dvintx chess_storage(DV1) a1;
            dvintx chess_storage(DV2) a2;
            dvintx chess_storage(DV3) a3;
            vintx reA, imA, reB, imB;

            a0 = BF_MUL(snap0, aS0);
            a1 = BF_MUL(snap0, aS0);
            a2 = BF_MUL(snap0, aS0);
            a3 = BF_MUL(snap0, aS0);
            BF_PACK4(a0, a1, a2, a3, 0, reA, imA);

            a0 = BF_MUL(snap0, aS0);
            a1 = BF_MUL(snap0, aS0);
            a2 = BF_MUL(snap0, aS0);
            a3 = BF_MUL(snap0, aS0);
            BF_PACK4(a0, a1, a2, a3, 4, reB, imB);

            bfStorePower8(vmux(lm_.lowHalf, reA, reB), vmux(lm_.lowHalf, imA, imB), aOutV);
        }

        /* At most one quad remains; no chess_loop_range - the trip count is 0 or 1. */
        for (int32_t a = anglesOct; a < anglesQuad; a += 4) chess_prepare_for_pipelining
        {
            dvintx chess_storage(DV0) a0;
            dvintx chess_storage(DV1) a1;
            dvintx chess_storage(DV2) a2;
            dvintx chess_storage(DV3) a3;

            a0 = BF_MUL(snap0, aS0);
            a1 = BF_MUL(snap0, aS0);
            a2 = BF_MUL(snap0, aS0);
            a3 = BF_MUL(snap0, aS0);

            BF_REDUCE_4(a0, a1, a2, a3, aOut);
        }
    }
    else
    { // numVecIterations == 2
        agen aSnap                      = init_agen_from_cfg(ctx->cfgSnapTpl);
        dvintx chess_storage(DV4) snap0 = dvint_load_di(aSnap);
        dvintx chess_storage(DV5) snap1 = dvint_load_di(aSnap);

        cupvaModifyAgenCfgBase(&ctx->cfgSteerFused[1], &steer[16]);
        agen aS1 = init_agen_from_cfg(ctx->cfgSteerFused[1]);

        for (int32_t a = 0; a < anglesOct; a += 8) chess_prepare_for_pipelining
        chess_loop_range(1, 32)
        {
            dvintx chess_storage(DV0) a0;
            dvintx chess_storage(DV1) a1;
            dvintx chess_storage(DV2) a2;
            dvintx chess_storage(DV3) a3;
            vintx reA, imA, reB, imB;

            a0 = BF_MUL(snap0, aS0) + BF_MUL(snap1, aS1);
            a1 = BF_MUL(snap0, aS0) + BF_MUL(snap1, aS1);
            a2 = BF_MUL(snap0, aS0) + BF_MUL(snap1, aS1);
            a3 = BF_MUL(snap0, aS0) + BF_MUL(snap1, aS1);
            BF_PACK4(a0, a1, a2, a3, 0, reA, imA);

            a0 = BF_MUL(snap0, aS0) + BF_MUL(snap1, aS1);
            a1 = BF_MUL(snap0, aS0) + BF_MUL(snap1, aS1);
            a2 = BF_MUL(snap0, aS0) + BF_MUL(snap1, aS1);
            a3 = BF_MUL(snap0, aS0) + BF_MUL(snap1, aS1);
            BF_PACK4(a0, a1, a2, a3, 4, reB, imB);

            bfStorePower8(vmux(lm_.lowHalf, reA, reB), vmux(lm_.lowHalf, imA, imB), aOutV);
        }

        /* At most one quad remains; no chess_loop_range - the trip count is 0 or 1. */
        for (int32_t a = anglesOct; a < anglesQuad; a += 4) chess_prepare_for_pipelining
        {
            dvintx chess_storage(DV0) a0;
            dvintx chess_storage(DV1) a1;
            dvintx chess_storage(DV2) a2;
            dvintx chess_storage(DV3) a3;

            a0 = BF_MUL(snap0, aS0) + BF_MUL(snap1, aS1);
            a1 = BF_MUL(snap0, aS0) + BF_MUL(snap1, aS1);
            a2 = BF_MUL(snap0, aS0) + BF_MUL(snap1, aS1);
            a3 = BF_MUL(snap0, aS0) + BF_MUL(snap1, aS1);

            BF_REDUCE_4(a0, a1, a2, a3, aOut);
        }
    }

    // Remainder: non-quad-aligned angles use original v-inner approach
    {
        agen snapTpl         = init_agen_from_cfg(ctx->cfgSnapTpl);
        int32_t remainderOfs = anglesQuad * steeringStrideElements;
        for (int32_t a = anglesQuad; a < anglesInBatch; a++)
        {
            agen aSnap = snapTpl;
            cupvaModifyAgenCfgBase(&ctx->cfgSteerTpl, &steer[remainderOfs]);
            agen aSteer = init_agen_from_cfg(ctx->cfgSteerTpl);
            remainderOfs += steeringStrideElements;

            dvintx chess_storage(DV0) acc;
            acc.lo = zeroVec;
            acc.hi = zeroVec;
            for (int32_t v = 0; v < numVecIterations; v++) chess_prepare_for_pipelining
            chess_loop_range(1, 8)
            {
                dvintx snap = dvint_load_di(aSnap);
                acc         = BF_MAC(snap, aSteer, acc);
            }
            float_store(reducePower(acc.lo, acc.hi), aOut);
        }
    }
}

void bartlettInitAgensCommon(BartlettContext *ctx, BartlettGlobalParams const *GP)
{
    const int32_t numVC            = GP->numTx * GP->numRx;
    const int32_t numVecIterations = numVC / COMPLEX_PER_VEC_LOAD;
    ctx->numVC                     = numVC;

    ctx->numAz               = GP->numAzimuthBins;
    ctx->numEl               = GP->numElevationBins;
    ctx->numAngleBins        = ctx->numAz * ctx->numEl;
    ctx->azNeighborhood      = GP->azNeighborhood;
    ctx->enableInterpolation = GP->enableInterpolation;
    ctx->enablePowerOutput   = GP->enablePowerOutput;
    ctx->azFromEl0           = GP->azFromEl0;
    ctx->elZeroIdx           = GP->elZeroIdx;
    ctx->azScale             = (GP->azimuthMax - GP->azimuthMin) / (float)max(1, ctx->numAz - 1);
    ctx->elScale             = (GP->elevationMax - GP->elevationMin) / (float)max(1, ctx->numEl - 1);
    ctx->azMin               = GP->azimuthMin;
    ctx->elMin               = GP->elevationMin;

    const int32_t steeringStrideElements = numVC * ELEMENT_NUM_OF_COMPLEX;
    ctx->numVecIterations                = numVecIterations;
    ctx->steeringStrideElements          = steeringStrideElements;
    ctx->steeringStridePerAngle          = (int64_t)steeringStrideElements * sizeof(int16_t);
    ctx->steeringBatchStride             = STEERING_ANGLE_BATCH_SIZE * ctx->steeringStridePerAngle;
    ctx->nextPeakValid                   = 0;
    ctx->azSpecBuf                       = 0;

    AgenWrapper wrapper;

    agen snapTpl = init((vintx *)ctx->snapshotBuffers[0]);
    wrapper.size = sizeof(int32_t);
    wrapper.n1   = numVecIterations;
    wrapper.s1   = 16;
    INIT_AGEN1(snapTpl, wrapper);
    ctx->cfgSnapTpl = extract_agen_cfg(snapTpl);

    agen steerTpl = init((vshortx *)ctx->steeringBuffers[0]);
    wrapper.size  = sizeof(int16_t);
    wrapper.n1    = numVecIterations;
    wrapper.s1    = 16;
    INIT_AGEN1(steerTpl, wrapper);
    ctx->cfgSteerTpl = extract_agen_cfg(steerTpl);

    AgenWrapper wrap2D;
    wrap2D.size = sizeof(int16_t);
    wrap2D.n1   = numVecIterations;
    wrap2D.s1   = 16;
    wrap2D.n2   = STEERING_ANGLE_BATCH_SIZE / 4;
    wrap2D.s2   = steeringStrideElements * 4;

    for (int32_t i = 0; i < 4; i++)
    {
        agen aS2D = init((vshortx *)ctx->steeringBuffers[0]);
        INIT_AGEN2(aS2D, wrap2D);
        ctx->cfgSteer2D[i] = extract_agen_cfg(aS2D);
    }

    // Fused single-pass steering AGENs (for numVecIterations<=2 path).
    // Two 1D AGENs walk consecutive angles at v=0 and v=1 offsets respectively.
    // n1 set to STEP1_AZ_BATCH_SIZE (256) to cover both full2D batches (max 128)
    // and separable Step 1 (max 256); the inner loop exits early for smaller batches.
    for (int32_t i = 0; i < 2; i++)
    {
        agen aSFused = init((vshortx *)(ctx->steeringBuffers[0] + i * 16));
        wrapper.size = sizeof(int16_t);
        wrapper.n1   = STEP1_AZ_BATCH_SIZE;
        wrapper.s1   = steeringStrideElements;
        INIT_AGEN1(aSFused, wrapper);
        ctx->cfgSteerFused[i] = extract_agen_cfg(aSFused);
    }

    agen aOutSpec = init((float *)ctx->powerSpectrum);
    wrapper.size  = sizeof(float);
    wrapper.n1    = STEP1_AZ_BATCH_SIZE;
    wrapper.s1    = 1;
    INIT_AGEN1(aOutSpec, wrapper);
    ctx->cfgOutSpectrum = extract_agen_cfg(aOutSpec);

    agen aOutSpecV = init((float *)ctx->powerSpectrum);
    wrapper.size   = sizeof(float);
    wrapper.n1     = STEP1_AZ_BATCH_SIZE / 8;
    wrapper.s1     = 8;
    INIT_AGEN1(aOutSpecV, wrapper);
    ctx->cfgOutSpectrumVec = extract_agen_cfg(aOutSpecV);

    agen aPeakScan = init((float *)ctx->powerSpectrum);
    wrapper.size   = sizeof(float);
    wrapper.n1     = STEP1_AZ_BATCH_SIZE / 8;
    wrapper.s1     = 8;
    INIT_AGEN1(aPeakScan, wrapper);
    ctx->cfgOutPeakScan = extract_agen_cfg(aPeakScan);

    const int32_t vecw = 8;

    AgenWrapper wrapOutF;
    wrapOutF.size = sizeof(float);
    wrapOutF.n1   = 1;
    wrapOutF.s1   = vecw;

    AgenWrapper wrapOutI;
    wrapOutI.size = sizeof(int32_t);
    wrapOutI.n1   = 1;
    wrapOutI.s1   = vecw;

    agen a;
    a = init((vfloatx *)ctx->azY0);
    INIT_AGEN1(a, wrapOutF);
    ctx->cfgAzY0 = extract_agen_cfg(a);
    a            = init((vfloatx *)ctx->azY1);
    INIT_AGEN1(a, wrapOutF);
    ctx->cfgAzY1 = extract_agen_cfg(a);
    a            = init((vfloatx *)ctx->azY2);
    INIT_AGEN1(a, wrapOutF);
    ctx->cfgAzY2 = extract_agen_cfg(a);
    a            = init((vfloatx *)ctx->elY0);
    INIT_AGEN1(a, wrapOutF);
    ctx->cfgElY0 = extract_agen_cfg(a);
    a            = init((vfloatx *)ctx->elY1);
    INIT_AGEN1(a, wrapOutF);
    ctx->cfgElY1 = extract_agen_cfg(a);
    a            = init((vfloatx *)ctx->elY2);
    INIT_AGEN1(a, wrapOutF);
    ctx->cfgElY2 = extract_agen_cfg(a);
    a            = init((vintx *)ctx->peakAzIdx);
    INIT_AGEN1(a, wrapOutI);
    ctx->cfgPeakAzIdx = extract_agen_cfg(a);
    a                 = init((vintx *)ctx->peakElIdx);
    INIT_AGEN1(a, wrapOutI);
    ctx->cfgPeakElIdx = extract_agen_cfg(a);
    a                 = init((vfloatx *)ctx->peakMagnitude);
    INIT_AGEN1(a, wrapOutF);
    ctx->cfgPowIn = extract_agen_cfg(a);

    float *outBuf0 = ctx->anglesOutputBuffers[0];
    a              = init((vfloatx *)(outBuf0 + ANGLES_ROW_AZ * SNAPSHOT_BATCH_SIZE));
    INIT_AGEN1(a, wrapOutF);
    ctx->cfgOutAz = extract_agen_cfg(a);
    a             = init((vfloatx *)(outBuf0 + ANGLES_ROW_EL * SNAPSHOT_BATCH_SIZE));
    INIT_AGEN1(a, wrapOutF);
    ctx->cfgOutEl = extract_agen_cfg(a);
    a             = init((vfloatx *)(outBuf0 + ANGLES_ROW_POWER * SNAPSHOT_BATCH_SIZE));
    INIT_AGEN1(a, wrapOutF);
    ctx->cfgOutPower = extract_agen_cfg(a);
}
