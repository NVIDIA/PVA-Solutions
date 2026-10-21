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

#include "doa_knl.h"

#include "doa_params.h"

void clearFFTInputInit(int32_t &cfg_idx, DOAParams *params)
{
    int32_t *upperPtr = (int32_t *)params->upperPtr;
    int32_t *lowerPtr = (int32_t *)params->lowerPtr;
    int32_t vecw      = chess_elementsof(DVTYPEX);
    int32_t niters    = (FFT_NUM_POINTS * ELEMENT_NUM_OF_COMPLEX * PEAK_SNAP_HEIGHT) / vecw;

    AgenCFG *cfgs = params->cfgs;

    AgenWrapper wrapper;
    agen upperSt, lowerSt;

    upperSt      = init(upperPtr);
    wrapper.size = sizeof(STYPE);
    wrapper.n1   = niters;
    wrapper.s1   = vecw;
    INIT_AGEN1(upperSt, wrapper);

    lowerSt      = init(lowerPtr);
    wrapper.size = sizeof(STYPE);
    wrapper.n1   = niters;
    wrapper.s1   = vecw;
    INIT_AGEN1(lowerSt, wrapper);

    cfgs[cfg_idx++] = extract_agen_cfg(upperSt);
    cfgs[cfg_idx++] = extract_agen_cfg(lowerSt);

    params->clearFFTInputIters = niters;
}

void processSnapDataInit(int32_t &cfg_idx, DOAParams *params)
{
    int32_t *snapSrcPtr             = params->peakSnapPtr;
    int32_t *snapDstPtr             = params->snapProcessedPtr;
    int32_t *snapProcessedTranspPtr = params->snapProcessedTranspPtr;

    int32_t vecw = pva_elementsof(DVTYPEX);
    int32_t lp   = PEAK_SNAP_PROCESSED_TRANSP_WIDTH;

    // 4-dimension agen for reversing 4 digits (256 = 2^8 = 4^4)
    int32_t niter1    = PEAK_SNAP_WIDTH / vecw;
    int32_t niter2    = PEAK_SNAP_HEIGHT;
    int32_t niter3    = PEAK_TILE_HEIGHT / PEAK_SNAP_HEIGHT;
    int32_t niters    = niter1 * niter2;
    AgenCFG *cfgs     = params->cfgs;
    int32_t qbits     = max(params->GP->calibVectorQbits, 16);
    int32_t snapQbits = qbits - 16;

    AgenWrapper wrapper;
    agen in, out1, out2, snapLd, snapSt;

    in           = init((DVTYPE *)snapSrcPtr);
    wrapper.size = sizeof(STYPE);
    wrapper.n1   = niter1;
    wrapper.s1   = vecw;
    wrapper.n2   = niter2;
    wrapper.s2   = vecw * niter1;
    wrapper.n3   = niter3;
    wrapper.s3   = vecw * niter1 * niter2;
    INIT_AGEN3(in, wrapper);

    out1           = init((DVTYPEX *)snapProcessedTranspPtr);
    out1.lane_ofst = lp / vecw;
    wrapper.size   = sizeof(STYPE);
    wrapper.n1     = niter1 / 2;
    wrapper.s1     = lp;
    wrapper.n2     = niter2;
    wrapper.s2     = 2;
    wrapper.n3     = PEAK_ITERS;
    wrapper.s3     = 0;
    INIT_AGEN3(out1, wrapper);
    out1.round = snapQbits;

    out2           = init((DVTYPEX *)(snapProcessedTranspPtr + 1));
    out2.lane_ofst = lp / vecw;
    wrapper.size   = sizeof(STYPE);
    wrapper.n1     = niter1 / 2;
    wrapper.s1     = lp;
    wrapper.n2     = niter2;
    wrapper.s2     = 2;
    wrapper.n3     = PEAK_ITERS;
    wrapper.s3     = 0;
    INIT_AGEN3(out2, wrapper);
    out2.round = snapQbits;

    snapLd       = init((DVTYPEX *)snapProcessedTranspPtr);
    wrapper.size = sizeof(STYPE);
    wrapper.n1   = niter2 * 2;
    wrapper.s1   = lp;
    wrapper.n2   = PEAK_ITERS;
    wrapper.s2   = 0;
    INIT_AGEN2(snapLd, wrapper);

    snapSt       = init((DVTYPEX *)snapDstPtr);
    wrapper.size = sizeof(STYPE);
    wrapper.n1   = niter2 * 2;
    wrapper.s1   = vecw;
    wrapper.n2   = PEAK_ITERS;
    wrapper.s2   = 0;
    INIT_AGEN2(snapSt, wrapper);

    params->snapIters = niters / PEAK_SNAP_UNROLLING;

    cfgs[cfg_idx++] = extract_agen_cfg(in);
    cfgs[cfg_idx++] = extract_agen_cfg(out1);
    cfgs[cfg_idx++] = extract_agen_cfg(out2);
    cfgs[cfg_idx++] = extract_agen_cfg(snapLd);
    cfgs[cfg_idx++] = extract_agen_cfg(snapSt);
}

void processFFTInit(int32_t &cfg_idx, DOAParams *params)
{
    fft_batched_int_init(params->upperPtr, params->fftUpperPtr, params->fftTmpPtr, params->twiddleTable, cfg_idx,
                         params->fftParams);
    fft_batched_int_init(params->lowerPtr, params->fftLowerPtr, params->fftTmpPtr, params->twiddleTable, cfg_idx,
                         params->fftParams);
}

void calcAzimuthInit(int32_t &cfg_idx, DOAParams *params)
{
    int32_t vecw   = pva_elementsof(DVTYPEX);
    int32_t niter1 = (FFT_NUM_POINTS * 2) / vecw;
    int32_t niter2 = PEAK_SNAP_HEIGHT;
    int32_t niter3 = PEAK_ITERS;
    int32_t pitch  = FFT_NUM_POINTS * 2;

    int32_t *upperPtr = (int32_t *)params->fftUpperPtr;
    int32_t *lowerPtr = (int32_t *)params->fftLowerPtr;
    AgenCFG *cfgs     = params->cfgs;

    AgenWrapper wrapper;
    agen upperIn, lowerIn;
    upperIn      = init((DVTYPE *)upperPtr);
    wrapper.size = sizeof(STYPE);
    wrapper.n1   = niter1;
    wrapper.n2   = niter2;
    wrapper.s1   = vecw;
    wrapper.s2   = pitch;
    wrapper.n3   = PEAK_ITERS;
    wrapper.s3   = 0;
    INIT_AGEN3(upperIn, wrapper);

    lowerIn      = init((DVTYPE *)lowerPtr);
    wrapper.size = sizeof(STYPE);
    wrapper.n1   = niter1;
    wrapper.n2   = niter2;
    wrapper.s1   = vecw;
    wrapper.s2   = pitch;
    wrapper.n3   = PEAK_ITERS;
    wrapper.s3   = 0;
    INIT_AGEN3(lowerIn, wrapper);

    params->NCIfftIter1 = niter1;
    params->NCIfftIter2 = niter2;

    cfgs[cfg_idx++] = extract_agen_cfg(upperIn);
    cfgs[cfg_idx++] = extract_agen_cfg(lowerIn);

    CupvaSampler *sampler = params->samplerPhase;

    CupvaSamplerInput1D const input = {.data           = params->atanTablePtr,
                                       .type           = SAMPLER_INPUT_TYPE_S32,
                                       .length         = (uint32_t)(ATAN_TABLE_SIZE),
                                       .outOfRangeMode = SAMPLER_OUT_OF_RANGE_CONSTANT};

    CupvaSamplerIndices1D const indices = {.data               = params->indexPtr,
                                           .type               = SAMPLER_INDEX_TYPE_U32,
                                           .width              = (uint32_t)(PEAK_SNAP_HEIGHT),
                                           .height             = 1,
                                           .fractionalBits     = 13,
                                           .fractionalHandling = SAMPLER_FRAC_HANDLING_INTERPOLATE};

    CupvaSamplerOutput const output = {.data      = params->anglePtr,
                                       .pitch     = PEAK_SNAP_HEIGHT,
                                       .transMode = TRANS_MODE_NONE};

    cupvaSamplerSetup(sampler, &input, &indices, &output);

    sampler->task_info = sampler->task_info | 1;
}

void processSubBinInit(int32_t &cfg_idx, DOAParams *params, bool isLastIter)
{
    int32_t remaingPeaks      = isLastIter ? params->GP->remainPeakIters * PEAK_SNAP_HEIGHT : PEAK_TILE_HEIGHT;
    int32_t remainingNCINodes = remaingPeaks > NUM_NCI_NODES ? NUM_NCI_NODES : remaingPeaks;
    int32_t vecw              = pva_elementsof(DVTYPEX);
    int32_t niter1            = (remainingNCINodes + vecw - 1) / vecw;
    int32_t niter2            = (remaingPeaks + NUM_NCI_NODES - 1) / NUM_NCI_NODES;
    int32_t pitch             = PEAK_TILE_HEIGHT;

    AgenCFG *cfgs = params->cfgs;

    AgenWrapper wrapper;
    agen in0, in1, nciLd;
    agen azimuthIn, elevationIn, elevationSignIn;
    agen velocityOut, rangeOut, elevationOut, targetXOut, targetYOut, targetZOut;

    int32_t *rindPtr     = (int32_t *)params->peakIndicesPtr;
    int32_t *dfoldindPtr = (int32_t *)params->peakIndicesPtr + PEAK_TILE_HEIGHT;
    int32_t *dindPtr     = (int32_t *)params->peakIndicesPtr + PEAK_TILE_HEIGHT * 2;

    params->rindPtr     = rindPtr;
    params->dfoldindPtr = dfoldindPtr;
    params->dindPtr     = dindPtr;

    float *targetVelocityPtr  = params->targetResultsPtr + TARGET_DET_VELOCITY_OFFSET * PEAK_TILE_HEIGHT;
    float *targetAzimuthPtr   = params->targetResultsPtr + TARGET_DET_AZIMUTH_OFFSET * PEAK_TILE_HEIGHT;
    float *targetRangePtr     = params->targetResultsPtr + TARGET_DET_RANGE_OFFSET * PEAK_TILE_HEIGHT;
    float *targetElevationPtr = params->targetResultsPtr + TARGET_DET_ELEVATION_OFFSET * PEAK_TILE_HEIGHT;
    float *targetXPtr         = params->targetResultsPtr + TARGET_DET_X_OFFSET * PEAK_TILE_HEIGHT;
    float *targetYPtr         = params->targetResultsPtr + TARGET_DET_Y_OFFSET * PEAK_TILE_HEIGHT;
    float *targetZPtr         = params->targetResultsPtr + TARGET_DET_Z_OFFSET * PEAK_TILE_HEIGHT;

    in0          = init((DVTYPE *)rindPtr);
    wrapper.size = sizeof(STYPE);
    wrapper.n1   = niter1;
    wrapper.s1   = vecw;
    wrapper.n2   = niter2;
    wrapper.s2   = vecw * niter1;
    wrapper.n3   = PEAK_ITERS;
    wrapper.s3   = 0;
    INIT_AGEN3(in0, wrapper);

    in1          = init((DVTYPE *)dindPtr);
    wrapper.size = sizeof(STYPE);
    wrapper.n1   = niter1;
    wrapper.s1   = vecw;
    wrapper.n2   = niter2;
    wrapper.s2   = vecw * niter1;
    wrapper.n3   = PEAK_ITERS;
    wrapper.s3   = 0;
    INIT_AGEN3(in1, wrapper);

    nciLd           = init((DVUTYPE *)params->nciFinalPtr);
    nciLd.lane_ofst = 1;
    wrapper.size    = sizeof(UTYPE);
    wrapper.n1      = 9;
    wrapper.s1      = 1;
    wrapper.n2      = niter1;
    wrapper.s2      = NCI_NODE_SIZE * vecw;
    wrapper.n3      = PEAK_ITERS;
    wrapper.s3      = 0;
    INIT_AGEN3(nciLd, wrapper);

    azimuthIn    = init((dvfloat *)targetAzimuthPtr);
    wrapper.size = sizeof(float);
    wrapper.n1   = PEAK_TILE_HEIGHT / vecw;
    wrapper.s1   = vecw;
    wrapper.n2   = PEAK_ITERS;
    wrapper.s2   = 0;
    INIT_AGEN2(azimuthIn, wrapper);

    elevationIn  = init((DVTYPE *)params->elevationAnglePtr);
    wrapper.size = sizeof(STYPE);
    wrapper.n1   = PEAK_TILE_HEIGHT / vecw;
    wrapper.s1   = vecw;
    wrapper.n2   = PEAK_ITERS;
    wrapper.s2   = 0;
    INIT_AGEN2(elevationIn, wrapper);

    elevationSignIn = init((DVTYPE *)params->elevationIndexSignPtr);
    wrapper.size    = sizeof(STYPE);
    wrapper.n1      = PEAK_TILE_HEIGHT / vecw;
    wrapper.s1      = vecw;
    wrapper.n2      = PEAK_ITERS;
    wrapper.s2      = 0;
    INIT_AGEN2(elevationSignIn, wrapper);

    velocityOut  = init((DVTYPE *)targetVelocityPtr);
    rangeOut     = init((DVTYPE *)targetRangePtr);
    elevationOut = init((DVTYPE *)targetElevationPtr);
    wrapper.size = sizeof(STYPE);
    wrapper.n1   = PEAK_TILE_HEIGHT / vecw;
    wrapper.s1   = vecw;
    wrapper.n2   = PEAK_ITERS;
    wrapper.s2   = 0;
    INIT_AGEN2(velocityOut, wrapper);

    rangeOut     = init((DVTYPE *)targetRangePtr);
    wrapper.size = sizeof(STYPE);
    wrapper.n1   = PEAK_TILE_HEIGHT / vecw;
    wrapper.s1   = vecw;
    wrapper.n2   = PEAK_ITERS;
    wrapper.s2   = 0;
    INIT_AGEN2(rangeOut, wrapper);

    elevationOut = init((dvfloat *)targetElevationPtr);
    wrapper.size = sizeof(float);
    wrapper.n1   = PEAK_TILE_HEIGHT / vecw;
    wrapper.s1   = vecw;
    wrapper.n2   = PEAK_ITERS;
    wrapper.s2   = 0;
    INIT_AGEN2(elevationOut, wrapper);

    targetXOut   = init((dvfloat *)targetXPtr);
    wrapper.size = sizeof(float);
    wrapper.n1   = PEAK_TILE_HEIGHT / vecw;
    wrapper.s1   = vecw;
    wrapper.n2   = PEAK_ITERS;
    wrapper.s2   = 0;
    INIT_AGEN3(targetXOut, wrapper);

    targetYOut   = init((dvfloat *)targetYPtr);
    wrapper.size = sizeof(float);
    wrapper.n1   = PEAK_TILE_HEIGHT / vecw;
    wrapper.s1   = vecw;
    wrapper.n2   = PEAK_ITERS;
    wrapper.s2   = 0;
    INIT_AGEN3(targetYOut, wrapper);

    targetZOut   = init((dvfloat *)targetZPtr);
    wrapper.size = sizeof(float);
    wrapper.n1   = PEAK_TILE_HEIGHT / vecw;
    wrapper.s1   = vecw;
    wrapper.n2   = PEAK_ITERS;
    wrapper.s2   = 0;
    INIT_AGEN3(targetZOut, wrapper);

    cfgs[cfg_idx++] = extract_agen_cfg(in0);
    cfgs[cfg_idx++] = extract_agen_cfg(in1);
    cfgs[cfg_idx++] = extract_agen_cfg(nciLd);
    cfgs[cfg_idx++] = extract_agen_cfg(azimuthIn);
    cfgs[cfg_idx++] = extract_agen_cfg(elevationIn);
    cfgs[cfg_idx++] = extract_agen_cfg(elevationSignIn);
    cfgs[cfg_idx++] = extract_agen_cfg(velocityOut);
    cfgs[cfg_idx++] = extract_agen_cfg(rangeOut);
    cfgs[cfg_idx++] = extract_agen_cfg(elevationOut);
    cfgs[cfg_idx++] = extract_agen_cfg(targetXOut);
    cfgs[cfg_idx++] = extract_agen_cfg(targetYOut);
    cfgs[cfg_idx++] = extract_agen_cfg(targetZOut);

    params->subBinIters2     = (remaingPeaks + NUM_NCI_NODES - 1) / NUM_NCI_NODES;
    params->subBinIters1     = (remainingNCINodes + vecw - 1) / vecw;
    params->subBinTotalIters = params->subBinIters1 * params->subBinIters2;

    CupvaSampler *sampler = params->samplerElevation;

    CupvaSamplerInput1D const input = {.data           = params->asinTablePtr,
                                       .type           = SAMPLER_INPUT_TYPE_S32,
                                       .length         = ASIN_TABLE_SIZE,
                                       .outOfRangeMode = SAMPLER_OUT_OF_RANGE_CONSTANT};

    CupvaSamplerIndices1D const indices = {.data               = params->elevationIndexPtr,
                                           .type               = SAMPLER_INDEX_TYPE_U32,
                                           .width              = (uint16_t)(remaingPeaks),
                                           .height             = 1,
                                           .fractionalBits     = 13,
                                           .fractionalHandling = SAMPLER_FRAC_HANDLING_INTERPOLATE};

    CupvaSamplerOutput const output = {.data      = params->elevationAnglePtr,
                                       .pitch     = PEAK_TILE_HEIGHT,
                                       .transMode = TRANS_MODE_NONE};

    cupvaSamplerSetup(sampler, &input, &indices, &output);

    sampler->task_info = sampler->task_info | 1;
}

static inline vintx calcMagnitude(DVTYPEX &vc)
{
    vintx vMagInt     = vsumsq_t16(vc.lo, vc.hi);
    vfloatx vMagFloat = vintx_vfp(vMagInt);
    return vfp_vintx_trunc(vfsqrt(vMagFloat * (1.0f / (1 << 24))) * (1 << 15));
}

void updateInputSQDF(SequenceDataFlowHandler &inSQDFIndices, SequenceDataFlowHandler &inSQDFData, int32_t iter,
                     int32_t pingPong, DOAParams *params)
{
    ExtMemPointer *indicesExtAddr = params->extMemPtrs[0];
    ExtMemPointer *snapExtAddr    = params->extMemPtrs[1];

    int64_t indicesBase   = indicesExtAddr->base + indicesExtAddr->offset;
    int64_t snapBase      = snapExtAddr->base + snapExtAddr->offset;
    int32_t indicesOffset = iter * PEAK_TILE_HEIGHT * sizeof(uint32_t);
    int32_t snapOffset    = iter * PEAK_SNAP_WIDTH * PEAK_TILE_HEIGHT * sizeof(int32_t);

    cupvaSQDFUpdateAddr(inSQDFIndices, 0, indicesBase + indicesOffset, MEMTYPE_DRAM,
                        cupvaGetVmemAddress((void *)params->indicesTilePtr[pingPong]), MEMTYPE_VMEM);
    cupvaSQDFUpdateAddr(inSQDFData, 0, snapBase + snapOffset, MEMTYPE_DRAM,
                        cupvaGetVmemAddress((void *)params->snapTilePtr[pingPong]), MEMTYPE_VMEM);

    cupvaSQDFFlushAndTrig(inSQDFIndices);
    cupvaSQDFFlushAndTrig(inSQDFData);
}

void updateOutputSQDF(SequenceDataFlowHandler &outSQDF, ExtMemPointer &targetExtAddr, int32_t iter, DOAParams *params)
{
    int64_t targetBase   = targetExtAddr.base + targetExtAddr.offset;
    int32_t targetOffset = iter * PEAK_TILE_HEIGHT * sizeof(float);

    cupvaSQDFUpdateAddr(outSQDF, 0, cupvaGetVmemAddress((void *)params->targetResultsPtr), MEMTYPE_VMEM,
                        targetBase + targetOffset, MEMTYPE_DRAM);
    cupvaSQDFFlushAndTrig(outSQDF);
}

void clearFFTInput(int32_t &cfg_idx, DOAParams *params)
{
    AgenCFG *cfgs  = &params->cfgs[cfg_idx];
    agen_A upperSt = init_agen_A_from_cfg(cfgs[0]);
    agen_B lowerSt = init_agen_B_from_cfg(cfgs[1]);

    DVTYPEX vZero;
    vZero.lo = replicatew(0);
    vZero.hi = replicatew(0);

    int32_t niters = params->clearFFTInputIters;
    for (int32_t i = 0; i < niters; i++)
    {
        vstore(vZero, upperSt);
        vstore(vZero, lowerSt);
    }

    cfg_idx += 2;
}

// Function to process the snap data
void processSnapData(int32_t &cfg_idx, DOAParams *params, int32_t i)
{
    agen_A in;
    agen_C out1;
    agen_C out2;

    AgenCFG *cfgs = &params->cfgs[cfg_idx];

    cupvaModifyAgenCfgBase(&cfgs[0], params->peakSnapPtr + PEAK_SNAP_WIDTH * PEAK_SNAP_HEIGHT * i);
    in   = init_agen_A_from_cfg(cfgs[0]);
    out1 = init_agen_C_from_cfg(cfgs[1]);
    out2 = init_agen_C_from_cfg(cfgs[2]);

    int32_t calibShift = max(16 - params->GP->calibVectorQbits, 0);

    dvint *calibVectorPtr = (dvint *)params->calibVectorPtr;

    dvintx vCalib1 = sign_extend(*calibVectorPtr);
    dvintx vCalib2 = sign_extend(*(calibVectorPtr + 1));

    vCalib1 = vCalib1 << calibShift;
    vCalib2 = vCalib2 << calibShift;

    for (int32_t i = 0; i < params->snapIters; i++)
    {
        DVTYPEX v1 = DVLOAD_DI(in);
        DVTYPEX v2 = DVLOAD_DI(in);

        DVTYPEX v3 = dvcmulw_t16(v1, vCalib1);
        DVTYPEX v4 = dvcmulw_t16(v2, vCalib2);

        DVTYPEX v5, v6;
        v5.lo = v3.lo;
        v5.hi = v4.lo;
        v6.lo = -v3.hi;
        v6.hi = -v4.hi;

        vstore_transp(v5, out1);
        vstore_transp(v6, out2);
    }

    agen_A snapLd;
    agen_B snapSt;

    snapLd = init_agen_A_from_cfg(cfgs[3]);
    snapSt = init_agen_B_from_cfg(cfgs[4]);

    for (int32_t i = 0; i < params->snapIters; i++)
    {
        DVTYPEX v1 = DVLOAD(snapLd);
        DVTYPEX v2 = DVLOAD(snapLd);

        vstore(v1, snapSt);
        vstore(v2, snapSt);
    }

    DVTYPE *upperInPtr        = (DVTYPE *)(params->GP->apUpper);
    DVTYPE *lowerInPtr        = (DVTYPE *)(params->GP->apLower);
    DVTYPE *upperOutPtr       = (DVTYPE *)(params->GP->apUpperInd);
    DVTYPE *lowerOutPtr       = (DVTYPE *)(params->GP->apLowerInd);
    int32_t *snapProcessedPtr = (int32_t *)params->snapProcessedPtr;

    int32_t *upperPtr = (int32_t *)params->upperPtr;
    int32_t *lowerPtr = (int32_t *)params->lowerPtr;
    int32_t vecw      = chess_elementsof(DVTYPEX);
    int32_t pred      = 0xffffffff;

    for (int32_t i = 0; i < params->snapIters; i++)
    {
        DVTYPEX vUpperInd0   = sign_extend(*upperInPtr);
        DVTYPEX vLowerInd0   = sign_extend(*lowerInPtr);
        DVTYPEX vUpperOutInd = sign_extend(*upperOutPtr);
        DVTYPEX vLowerOutInd = sign_extend(*lowerOutPtr);

        DVTYPEX vUpper0, vLower0;
        vUpper0 = vlookup_16w(snapProcessedPtr, vUpperInd0);
        vLower0 = vlookup_16w(snapProcessedPtr, vLowerInd0);

        vast_16w(upperPtr, vUpperOutInd, vUpper0, pred);
        vast_16w(lowerPtr, vLowerOutInd, vLower0, pred);

        upperInPtr++;
        lowerInPtr++;
        upperOutPtr++;
        lowerOutPtr++;
    }

    cfg_idx += 5;
}

void processFFT(int32_t &cfg_idx, DOAParams *params)
{
    fft_batched_int_exec(cfg_idx, params->fftParams);
    fft_batched_int_exec(cfg_idx, params->fftParams);
}

void calcAzimuth(int32_t &cfg_idx, int32_t fft_idx, DOAParams *params)
{
    agen_A upperIn;
    agen_B lowerIn;

    AgenCFG *cfgs        = &params->cfgs[cfg_idx];
    int32_t *fftUpperPtr = (int32_t *)params->fftUpperPtr;
    int32_t *fftLowerPtr = (int32_t *)params->fftLowerPtr;
    int32_t fftQbits     = params->GP->fftQbits;
    int32_t lowMaxShift  = max(16 - fftQbits, 0);

    upperIn = init_agen_A_from_cfg(cfgs[0]);
    lowerIn = init_agen_B_from_cfg(cfgs[1]);

    int32_t vecw = pva_elementsof(VTYPEX);

    int32_t *upperMaxPoints = (int32_t *)params->fftUpperMaxPtr;
    int32_t *lowerMaxPoints = (int32_t *)params->fftLowerMaxPtr;
    int32_t NCIfftIter1     = params->NCIfftIter1;
    int32_t NCIfftIter2     = params->NCIfftIter2;

    int32_t *upperRowPtr, *lowerRowPtr;
    for (int32_t i = 0; i < NCIfftIter2; i++)
    {
        int32_t maxIdx = 0;
        int32_t maxMag = 0;
        for (int32_t j = 0; j < NCIfftIter1; j++)
        {
            DVTYPEX v1 = DVLOAD_DI(upperIn);
            DVTYPEX v2 = DVLOAD_DI(lowerIn);

            VTYPEX vMagUpper = calcMagnitude(v1);
            VTYPEX vMagLower = calcMagnitude(v2);
            VTYPEX vMag      = vadd(vMagUpper, vMagLower);

            int32_t currMaxIdx;
            int32_t currMaxMag;

            vmaxrid_s(vMag, currMaxMag, currMaxIdx);
            currMaxIdx += j * vecw;
            maxIdx = currMaxMag > maxMag ? currMaxIdx : maxIdx;
            maxMag = max(currMaxMag, maxMag);
        }

        maxIdx                               = maxIdx * 2;
        upperRowPtr                          = fftUpperPtr + FFT_NUM_POINTS * ELEMENT_NUM_OF_COMPLEX * i;
        lowerRowPtr                          = fftLowerPtr + FFT_NUM_POINTS * ELEMENT_NUM_OF_COMPLEX * i;
        upperMaxPoints[i]                    = upperRowPtr[maxIdx];
        upperMaxPoints[PEAK_SNAP_HEIGHT + i] = upperRowPtr[maxIdx + 1];
        lowerMaxPoints[i]                    = lowerRowPtr[maxIdx];
        lowerMaxPoints[PEAK_SNAP_HEIGHT + i] = -lowerRowPtr[maxIdx + 1];
    }

    chess_separator_scheduler();

    DVTYPEX vUpperMax = sign_extend(*(DVTYPE *)upperMaxPoints);
    DVTYPEX vLowerMax = sign_extend(*(DVTYPE *)lowerMaxPoints);
    vLowerMax         = vLowerMax << lowMaxShift;

    DVTYPEX magMaxAbs, magMaxSign;
    DVTYPEX magMax = dvcmulw_t16(vUpperMax, vLowerMax);
    dvsignmag(magMax, magMaxSign, magMaxAbs);
    dvfloatx magMaxFloat;
    magMaxFloat.lo            = vintx_vfp(magMaxAbs.lo);
    magMaxFloat.hi            = vintx_vfp(magMaxAbs.hi);
    vintx sel                 = magMaxAbs.hi > magMaxAbs.lo;
    vfloatx rcpY              = vfrcp(magMaxFloat.hi);
    vfloatx rcpX              = vfrcp(magMaxFloat.lo);
    vfloatx num               = vmux(sel, magMaxFloat.lo, magMaxFloat.hi);
    vfloatx denom             = vmux(sel, rcpY, rcpX);
    vfloatx angleIndexFloat   = num * denom;
    vintx angleIndex          = vfp_vint_round(angleIndexFloat * (1 << 22));
    *(vint *)params->indexPtr = extract(angleIndex);

    cupvaSamplerStart(params->samplerPhase);
    vintx isQuad4 = magMaxSign.lo == 0 && magMaxSign.hi == 1;
    vintx isQuad2 = magMaxSign.lo == 1 && magMaxSign.hi == 0;
    vintx isQuad3 = magMaxSign.lo == 1 && magMaxSign.hi == 1;
    cupvaSamplerWait();
    vintx angle        = sign_extend(*(vint *)params->anglePtr);
    vintx angleYLarger = replicatew(Q15_90_DEGREES) - angle;
    angle              = vmux(sel, angleYLarger, angle);
    vintx angleQuad4   = -angle;
    vintx angleQuad2   = replicatew(Q15_180_DEGREES) - angle;
    vintx angleQuad3   = angle - Q15_180_DEGREES;

    angle = vmux(isQuad4, angleQuad4, angle);
    angle = vmux(isQuad2, angleQuad2, angle);
    angle = vmux(isQuad3, angleQuad3, angle);

    vfloatx angleFloat = vintx_vfp(angle);

    dvintx vPhase;
    vPhase.lo = vfp_vint_round(vfcos(angleFloat * (1.0f / (360.0f * (1 << 15)))) * (1 << 16));
    vPhase.hi = vfp_vint_round(vfsin(angleFloat * (1.0f / (360.0f * (1 << 15)))) * (1 << 16));

    float *targetAzimuthPtr =
        params->targetResultsPtr + TARGET_DET_AZIMUTH_OFFSET * PEAK_TILE_HEIGHT + fft_idx * BATCH_SIZE;
    float *aIndexTable             = params->aIndexTable;
    int32_t *elevationIndexPtr     = params->elevationIndexPtr + fft_idx * BATCH_SIZE;
    int32_t *elevationIndexSignPtr = params->elevationIndexSignPtr + fft_idx * BATCH_SIZE;
    vintx elevationInt             = vfp_vintx_round(angleFloat * params->GP->elevationFactor);
    vintx elevationIntSign, elevationIntAbs;
    vsignmag(elevationInt, elevationIntSign, elevationIntAbs);

    *(vint *)elevationIndexPtr     = extract(elevationIntAbs);
    *(vint *)elevationIndexSignPtr = extract(elevationIntSign);

    chess_separator_scheduler();
    vintx indices = replicatew(0);

    for (int32_t i = 0; i < NCIfftIter2; i++)
    {
        dvintx vPhaseCurr;
        vPhaseCurr.lo = vpermute(vPhase.lo, indices);
        vPhaseCurr.hi = vpermute(vPhase.hi, indices);

        int32_t maxIdx = 0;
        int32_t maxMag = 0;
        for (int32_t j = 0; j < NCIfftIter1; j++)
        {
            DVTYPEX v1 = DVLOAD_DI(upperIn);
            DVTYPEX v2 = DVLOAD_DI(lowerIn);

            DVTYPEX lowerPhaseAdj = dvcmulw_t16(v2, vPhaseCurr);
            DVTYPEX sum           = v1 + lowerPhaseAdj;
            VTYPEX vFFTFinal      = calcMagnitude(sum);
            int32_t currMaxIdx;
            int32_t currMaxMag;
            vmaxrid_s(vFFTFinal, currMaxMag, currMaxIdx);
            currMaxIdx += j * vecw;
            maxIdx = (currMaxMag > maxMag) ? currMaxIdx : maxIdx;
            maxMag = max(currMaxMag, maxMag);
        }
        targetAzimuthPtr[i] = aIndexTable[maxIdx];
        indices             = indices + 1;
    }

    cfg_idx += 2;
}

static inline void quadraticInterpolation(dvfloatx &y0, dvfloatx &y1, dvfloatx &y2, dvfloatx &p)
{
    dvfloatx vDen    = y0 - y1 * 2.0f + y2;
    dvfloatx vDenRcp = dvfrcp(vDen * 2.0f);
    p                = (y0 - y2) * vDenRcp;
}

static inline void calcVelocity(dvfloatx &vTempVIn, dvfloatx &vTempVOut, PvaRadarGPPriv *GP)
{
    vTempVOut.lo = replicatef(0.0f);
    vTempVOut.hi = replicatef(0.0f);

    for (int i = 0; i < 5; i++)
    {
        dvfloatx vTempV = vTempVIn - GP->tempVCoeff[i];
        dvintx found    = (vTempV >= GP->velocityAmbHigh) && (vTempV <= GP->velocityAmbLow) && (vTempVOut == 0.0f);
        vTempVOut       = dvmux(found, vTempV, vTempVOut);
    }
}

void processSubBin(int32_t &cfg_idx, DOAParams *params, bool isLastIter)
{
    cupvaSamplerStart(params->samplerElevation);

    AgenCFG *cfgs                               = &params->cfgs[cfg_idx];
    chess_protect_access agen_A rindIn          = init_agen_A_from_cfg(cfgs[0]);
    chess_protect_access agen_B dindIn          = init_agen_B_from_cfg(cfgs[1]);
    chess_protect_access agen_C nciLdIn         = init_agen_C_from_cfg(cfgs[2]);
    chess_protect_access agen_A azimuthIn       = init_agen_A_from_cfg(cfgs[3]);
    chess_protect_access agen_B elevationIn     = init_agen_B_from_cfg(cfgs[4]);
    chess_protect_access agen_C elevationSignIn = init_agen_C_from_cfg(cfgs[5]);
    chess_protect_access agen_A velocityOut     = init_agen_A_from_cfg(cfgs[6]);
    chess_protect_access agen_B rangeOut        = init_agen_B_from_cfg(cfgs[7]);
    chess_protect_access agen_C elevationOut    = init_agen_C_from_cfg(cfgs[8]);
    chess_protect_access agen_A targetXOut      = init_agen_A_from_cfg(cfgs[9]);
    chess_protect_access agen_B targetYOut      = init_agen_B_from_cfg(cfgs[10]);
    chess_protect_access agen_C targetZOut      = init_agen_C_from_cfg(cfgs[11]);

    int32_t numRemainPeaks = (isLastIter) ? params->GP->remainPeaks : PEAK_TILE_HEIGHT;

    int32_t *rindPtr     = (int32_t *)params->rindPtr;
    int32_t *dfoldindPtr = (int32_t *)params->dfoldindPtr;
    int32_t nciQbits     = params->GP->fftQbits;

    cupvaSamplerWait();
    chess_separator_scheduler();

    for (int i = 0; i < params->subBinIters2; i++)
    {
        int32_t numNciNodes = numRemainPeaks > NUM_NCI_NODES ? NUM_NCI_NODES : numRemainPeaks;
        cupvaGSDFUpdateTiles(params->nciHdl, dfoldindPtr, rindPtr, numNciNodes);
        cupvaGSDFTrig(params->nciHdl);
        cupvaGSDFSync(params->nciHdl);

        for (int j = 0; j < params->subBinIters1; j++)
        {
            DVTYPEX v11 = DVULOAD_TRANSP(nciLdIn);
            DVTYPEX v12 = DVULOAD_TRANSP(nciLdIn);
            DVTYPEX v13 = DVULOAD_TRANSP(nciLdIn);

            DVTYPEX v21 = DVULOAD_TRANSP(nciLdIn);
            DVTYPEX v22 = DVULOAD_TRANSP(nciLdIn);
            DVTYPEX v23 = DVULOAD_TRANSP(nciLdIn);

            DVTYPEX v31 = DVULOAD_TRANSP(nciLdIn);
            DVTYPEX v32 = DVULOAD_TRANSP(nciLdIn);
            DVTYPEX v33 = DVULOAD_TRANSP(nciLdIn);

            dvfloatx v21Float, v22Float, v23Float, v12Float, v32Float;

            v21Float.lo = vint_vfp(v21.lo);
            v21Float.hi = vint_vfp(v21.hi);
            v22Float.lo = vint_vfp(v22.lo);
            v22Float.hi = vint_vfp(v22.hi);
            v23Float.lo = vint_vfp(v23.lo);
            v23Float.hi = vint_vfp(v23.hi);
            v12Float.lo = vint_vfp(v12.lo);
            v12Float.hi = vint_vfp(v12.hi);
            v32Float.lo = vint_vfp(v32.lo);
            v32Float.hi = vint_vfp(v32.hi);

            dvfloatx pr, pd;
            quadraticInterpolation(v12Float, v22Float, v32Float, pr);
            quadraticInterpolation(v21Float, v22Float, v23Float, pd);

            DVTYPEX vRind = DVLOAD(rindIn);
            dvfloatx vRindFloat;
            vRindFloat.lo = vint_vfp(vRind.lo);
            vRindFloat.hi = vint_vfp(vRind.hi);

            DVTYPEX vDind = DVLOAD(dindIn);
            dvfloatx vDindFloat;
            vDindFloat.lo = vint_vfp(vDind.lo);
            vDindFloat.hi = vint_vfp(vDind.hi);

            dvfloatx vIxr = vRindFloat + pr;
            dvfloatx vIxd = vDindFloat + pd;

            dvfloatx vFast  = vIxr * params->GP->f_fast_delta;
            dvfloatx vSlow  = vIxd * params->GP->f_slow_delta;
            dvfloatx vTempR = (vFast - vSlow) * params->GP->contR;

            dvfloatx vTempVBase = vFast * params->GP->contVFast - vSlow * params->GP->contVSlow;
            dvfloatx vTempV;
            calcVelocity(vTempVBase, vTempV, params->GP);

            dvfloatx velocity = vTempV * 3.6f;
            dvfloatx range    = vTempR;

            dvfloatx azimuthFloat = DVLOADF(azimuthIn);
            dvfloatx x            = range * dvfsin(azimuthFloat * (1.0f / 360.0f));
            dvfloatx y            = range * dvfcos(azimuthFloat * (1.0f / 360.0f));

            dvfloatx vElevationFloat;
            DVTYPEX vElevationInt     = DVLOAD(elevationIn);
            DVTYPEX vElevationIntSign = DVLOAD(elevationSignIn);
            vElevationInt             = dvmux(vElevationIntSign, -vElevationInt, vElevationInt);
            vElevationFloat.lo        = vint_vfp(vElevationInt.lo) * (1.0f / (1 << 15));
            vElevationFloat.hi        = vint_vfp(vElevationInt.hi) * (1.0f / (1 << 15));
            dvfloatx z                = range * dvfsin(vElevationFloat * (1.0f / 360.0f));

            vstore(velocity, velocityOut);
            vstore(range, rangeOut);
            vstore(vElevationFloat, elevationOut);
            vstore(x, targetXOut);
            vstore(y, targetYOut);
            vstore(z, targetZOut);
        }

        rindPtr += NUM_NCI_NODES;
        dfoldindPtr += NUM_NCI_NODES;
        numRemainPeaks -= NUM_NCI_NODES;
    }

    // No need to increment cfg_idx here as this is the last stage
}

void DOAKnl(DOAParams *params, bool isLastIter)
{
    int32_t cfg_idx = 41;
    processSubBinInit(cfg_idx, params, isLastIter);

    int32_t niters = (isLastIter) ? params->GP->remainPeakIters : PEAK_ITERS;
    for (int i = 0; i < niters; i++)
    {
        cfg_idx = 0;
        clearFFTInput(cfg_idx, params);
        processSnapData(cfg_idx, params, i);
        processFFT(cfg_idx, params);
        calcAzimuth(cfg_idx, i, params);
    }

    processSubBin(cfg_idx, params, isLastIter);
}