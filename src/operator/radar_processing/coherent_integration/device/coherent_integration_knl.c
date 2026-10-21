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

#include "coherent_integration_knl.h"

#include <stddef.h>

#ifndef PVASOL_COHERENT_INTEGRATION_RX
#    error "PVASOL_COHERENT_INTEGRATION_RX must select 4, 8, 12, or 16 receivers"
#endif

#if (PVASOL_COHERENT_INTEGRATION_RX != CI_RX4) && (PVASOL_COHERENT_INTEGRATION_RX != CI_RX8) && \
    (PVASOL_COHERENT_INTEGRATION_RX != CI_RX12) && (PVASOL_COHERENT_INTEGRATION_RX != CI_RX16)
#    error "PVASOL_COHERENT_INTEGRATION_RX must be 4, 8, 12, or 16"
#endif

static inline dvintx roundInputByTwo(dvintx value)
{
    return dvround(value, 1);
}

static inline dvintx zeroIntVector(void)
{
    dvintx zero;
    zero.lo = zero.hi = replicatew(0);
    return zero;
}

/* Rounding is monotone, so rounding the extrema also gives the extrema of the rounded values. */
static inline dvintx deriveLimbShift(dvintx minimum, dvintx maximum)
{
    const dvintx roundedMinimum = roundInputByTwo(minimum);
    const dvintx roundedMaximum = roundInputByTwo(maximum);
    const dvintx maxMagnitude   = dvmax(dvabsdif(roundedMinimum, 0), dvabsdif(roundedMaximum, 0));

    /* VMSBD(0) is -1, so max(maxMagnitude - 1, 0) also covers magnitudes 0 and 1. */
    const dvintx magnitudeBits = dvmsbd(dvmax(maxMagnitude - 1, 0)) + 1;
    return dvmax(magnitudeBits - 15, 0) + 1;
}

static inline void storeCellShift(CoherentIntegrationPreprocessAgenConfig const *agenConfig, int8_t *cellShift,
                                  uint32_t firstCell, dvintx limbShift)
{
    AgenCFG shiftConfig = agenConfig->shift;
    cupvaModifyAgenCfgBase(&shiftConfig, cellShift + firstCell);
    agen_A shiftAgen = init_agen_A_from_cfg(shiftConfig);
    vstore_wb_plround(replicateh(0), -limbShift, shiftAgen);
}

static inline void scanRawBlock(CoherentIntegrationPreprocessAgenConfig const *agenConfig, int32_t *rawCells,
                                dvintx *minimum, dvintx *maximum)
{
    *minimum = zeroIntVector();
    *maximum = zeroIntVector();

    AgenCFG rawConfig = agenConfig->raw;
    cupvaModifyAgenCfgBase(&rawConfig, rawCells);
    agen_A rawAgen = init_agen_A_from_cfg(rawConfig);
    for (uint32_t rx = 0U; rx < PVASOL_COHERENT_INTEGRATION_RX; rx++) chess_prepare_for_pipelining
    chess_loop_range(4, 16)
    {
        const dvintx cells0 = dvint_load_di(rawAgen);
        const dvintx cells1 = dvint_load_di(rawAgen);

        /*
         * The deinterleaving loads produce [real.lo, imag.lo] and
         * [real.hi, imag.hi].  Reduce those halves in place so the VPU does
         * not have to assemble temporary full-width real/imaginary vectors.
         */
        minimum->lo = vmin3(minimum->lo, cells0.lo, cells0.hi);
        minimum->hi = vmin3(minimum->hi, cells1.lo, cells1.hi);
        maximum->lo = vmax3(maximum->lo, cells0.lo, cells0.hi);
        maximum->hi = vmax3(maximum->hi, cells1.lo, cells1.hi);
    }
}

/* Split one block while inspecting the next block with independent min/max chains. */
static inline void splitRawBlockAndScanNext(CoherentIntegrationPreprocessAgenConfig const *agenConfig,
                                            int32_t *rawCells, int32_t *nextRawCells, dvintx limbShift,
                                            int8_t *cellShift, int16_t *highLimbs, int16_t *residualLimbs,
                                            uint32_t firstCell, dvintx *nextMinimum, dvintx *nextMaximum)
{
    storeCellShift(agenConfig, cellShift, firstCell, limbShift);

    AgenCFG rawConfig = agenConfig->raw;
    cupvaModifyAgenCfgBase(&rawConfig, rawCells);
    agen_A rawAgen = init_agen_A_from_cfg(rawConfig);
    cupvaModifyAgenCfgBase(&rawConfig, nextRawCells);
    agen_A nextRawAgen = init_agen_A_from_cfg(rawConfig);

    const uint32_t firstLimb = firstCell * PVASOL_COHERENT_INTEGRATION_RX * 2U;
    AgenCFG limbConfig       = agenConfig->limb;
    cupvaModifyAgenCfgBase(&limbConfig, highLimbs + firstLimb);
    agen_B highAgen = init_agen_B_from_cfg(limbConfig);
    cupvaModifyAgenCfgBase(&limbConfig, residualLimbs + firstLimb);
    agen_C residualAgen = init_agen_C_from_cfg(limbConfig);

    dvintx groupedShift0;
    dvintx groupedShift1;
    groupedShift0.lo = groupedShift0.hi = limbShift.lo;
    groupedShift1.lo = groupedShift1.hi = limbShift.hi;

    dvintx one;
    one.lo = one.hi    = replicatew(1);
    const dvintx bias0 = dvshift(one, groupedShift0 - 1);
    const dvintx bias1 = dvshift(one, groupedShift1 - 1);

    *nextMinimum = zeroIntVector();
    *nextMaximum = zeroIntVector();

    for (uint32_t rx = 0U; rx < PVASOL_COHERENT_INTEGRATION_RX; rx++) chess_prepare_for_pipelining
    chess_loop_range(4, 16)
    {
        dvintx cells0           = dvint_load_di(rawAgen);
        dvintx cells1           = dvint_load_di(rawAgen);
        const dvintx nextCells0 = dvint_load_di(nextRawAgen);
        const dvintx nextCells1 = dvint_load_di(nextRawAgen);

        cells0 = roundInputByTwo(cells0);
        cells1 = roundInputByTwo(cells1);

        nextMinimum->lo = vmin3(nextMinimum->lo, nextCells0.lo, nextCells0.hi);
        nextMinimum->hi = vmin3(nextMinimum->hi, nextCells1.lo, nextCells1.hi);
        nextMaximum->lo = vmax3(nextMaximum->lo, nextCells0.lo, nextCells0.hi);
        nextMaximum->hi = vmax3(nextMaximum->hi, nextCells1.lo, nextCells1.hi);

        dvintx highCells0;
        dvintx highCells1;
        dvintx biasedLowCells0;
        dvintx biasedLowCells1;
        dvsplitbits(cells0 + bias0, groupedShift0, highCells0, biasedLowCells0);
        dvsplitbits(cells1 + bias1, groupedShift1, highCells1, biasedLowCells1);
        const dvintx lowCells0 = biasedLowCells0 - bias0;
        const dvintx lowCells1 = biasedLowCells1 - bias1;

        vstore_i2(highCells0, highCells1, highAgen);
        vstore_i2(lowCells0, lowCells1, residualAgen);
    }
}

/* Split the final block without accessing a nonexistent next block. */
static inline void splitFinalRawBlock(CoherentIntegrationPreprocessAgenConfig const *agenConfig, int32_t *rawCells,
                                      dvintx limbShift, int8_t *cellShift, int16_t *highLimbs, int16_t *residualLimbs,
                                      uint32_t firstCell)
{
    storeCellShift(agenConfig, cellShift, firstCell, limbShift);

    AgenCFG rawConfig = agenConfig->raw;
    cupvaModifyAgenCfgBase(&rawConfig, rawCells);
    agen_A rawAgen = init_agen_A_from_cfg(rawConfig);

    const uint32_t firstLimb = firstCell * PVASOL_COHERENT_INTEGRATION_RX * 2U;
    AgenCFG limbConfig       = agenConfig->limb;
    cupvaModifyAgenCfgBase(&limbConfig, highLimbs + firstLimb);
    agen_B highAgen = init_agen_B_from_cfg(limbConfig);
    cupvaModifyAgenCfgBase(&limbConfig, residualLimbs + firstLimb);
    agen_C residualAgen = init_agen_C_from_cfg(limbConfig);

    dvintx groupedShift0;
    dvintx groupedShift1;
    groupedShift0.lo = groupedShift0.hi = limbShift.lo;
    groupedShift1.lo = groupedShift1.hi = limbShift.hi;

    dvintx one;
    one.lo = one.hi    = replicatew(1);
    const dvintx bias0 = dvshift(one, groupedShift0 - 1);
    const dvintx bias1 = dvshift(one, groupedShift1 - 1);

    for (uint32_t rx = 0U; rx < PVASOL_COHERENT_INTEGRATION_RX; rx++) chess_prepare_for_pipelining
    chess_loop_range(4, 16)
    {
        dvintx cells0 = roundInputByTwo(dvint_load_di(rawAgen));
        dvintx cells1 = roundInputByTwo(dvint_load_di(rawAgen));

        dvintx highCells0;
        dvintx highCells1;
        dvintx biasedLowCells0;
        dvintx biasedLowCells1;
        dvsplitbits(cells0 + bias0, groupedShift0, highCells0, biasedLowCells0);
        dvsplitbits(cells1 + bias1, groupedShift1, highCells1, biasedLowCells1);
        const dvintx lowCells0 = biasedLowCells0 - bias0;
        const dvintx lowCells1 = biasedLowCells1 - bias1;

        vstore_i2(highCells0, highCells1, highAgen);
        vstore_i2(lowCells0, lowCells1, residualAgen);
    }
}

void coherentInitializePreprocessAgenConfig(CoherentIntegrationPreprocessAgenConfig *config, uint32_t rawCellCapacity)
{
    AgenWrapper rawWrapper;
    rawWrapper.size = sizeof(int32_t);
    rawWrapper.n1   = 2U;
    rawWrapper.s1   = CI_CELL_VECTOR_LANES;
    rawWrapper.n2   = PVASOL_COHERENT_INTEGRATION_RX;
    rawWrapper.s2   = rawCellCapacity * 2U;
    agen rawAgen    = init(NULL);
    INIT_AGEN2(rawAgen, rawWrapper);
    config->raw = extract_agen_cfg(rawAgen);

    AgenWrapper limbWrapper;
    limbWrapper.size = sizeof(int16_t);
    limbWrapper.n1   = PVASOL_COHERENT_INTEGRATION_RX;
    limbWrapper.s1   = 2U * CI_CELL_VECTOR_LANES;
    agen limbAgen    = init(NULL);
    INIT_AGEN1(limbAgen, limbWrapper);
    config->limb = extract_agen_cfg(limbAgen);

    AgenWrapper shiftWrapper;
    shiftWrapper.size = sizeof(int8_t);
    shiftWrapper.n1   = 1U;
    shiftWrapper.s1   = CI_CELL_VECTOR_LANES;
    agen shiftAgen    = init(NULL);
    INIT_AGEN1(shiftAgen, shiftWrapper);
    config->shift = extract_agen_cfg(shiftAgen);
}

void coherentPreprocessRawTileImpl(CoherentIntegrationPreprocessAgenConfig const *agenConfig, uint8_t *rawInput,
                                   int8_t *cellShift, int16_t *highLimbs, int16_t *residualLimbs,
                                   uint32_t validCellCount)
{
    int32_t *raw              = (int32_t *)rawInput;
    const uint32_t blockCount = validCellCount / CI_CELL_VECTOR_LANES;

    if (blockCount == 0U)
    {
        return;
    }

    dvintx minimum;
    dvintx maximum;
    scanRawBlock(agenConfig, raw, &minimum, &maximum);
    dvintx limbShift = deriveLimbShift(minimum, maximum);

    if (blockCount > 1U)
    {
        for (uint32_t block = 0U; block + 1U < blockCount; block++)
        {
            const uint32_t firstCell = block * CI_CELL_VECTOR_LANES;
            int32_t *rawCells        = raw + firstCell * 2U;
            int32_t *nextRawCells    = rawCells + CI_CELL_VECTOR_LANES * 2U;

            dvintx nextMinimum;
            dvintx nextMaximum;
            splitRawBlockAndScanNext(agenConfig, rawCells, nextRawCells, limbShift, cellShift, highLimbs, residualLimbs,
                                     firstCell, &nextMinimum, &nextMaximum);
            limbShift = deriveLimbShift(nextMinimum, nextMaximum);
        }
    }

    const uint32_t lastCell = (blockCount - 1U) * CI_CELL_VECTOR_LANES;
    splitFinalRawBlock(agenConfig, raw + lastCell * 2U, limbShift, cellShift, highLimbs, residualLimbs, lastCell);
}

static inline float pow2fExact(int32_t exponent)
{
    union
    {
        uint32_t bits;
        float value;
    } result;
    result.bits = (uint32_t)(exponent + 127) << 23;
    return result.value;
}

/*
 * Match sqrtf's IEEE round-to-nearest result for the nonnegative, normal
 * FP32 values produced by converting the signed-48 coherent peak power.
 *
 * Express the approximation as a 24-bit significand in the exact root binade,
 * then compare the scaled input against neighboring square midpoints to select
 * the correctly rounded FP32 encoding. One exact 33x33-to-48 VMulW product
 * produces T(M)=M*(M-1)/2; the four thresholds for M-2 through M+2 follow by
 * addition.
 *
 * For input significand X, root significand M, and biased input exponent E,
 * round upward across M iff
 *
 *     X << (23 - (E & 1)) > M * (M + 1) / 2.
 *
 * Equality selects M.  A true halfway tie is impossible because its square
 * contains a one-quarter term while the scaled input is integral.  Clamping
 * the approximate significand to [2^23+1, 2^24-2] keeps both two-neighbor
 * tests in-binade; the correction can still produce either endpoint.
 */
static inline dvfloatx correctlyRoundedSqrtNonnegative(dvfloatx input)
{
    enum
    {
        hiddenBit    = 1 << 23,
        fractionMask = hiddenBit - 1,
    };

    dvintx zeroVector;
    zeroVector.lo = zeroVector.hi = replicatew(0);
    dvintx hiddenBitVector;
    hiddenBitVector.lo = hiddenBitVector.hi = replicatew(hiddenBit);

    const dvintx inputBits       = as_dvintx(input);
    const dvintx approximateBits = as_dvintx(dvfsqrt(input));
    const dvintx biasedExponent  = dvshift(inputBits, -23);
    const dvintx rootExponent    = dvshift(biasedExponent + 127, -1);
    const dvintx rootBase        = dvshift(rootExponent, 23);

    dvintx candidate = dvadd2sub(approximateBits, hiddenBitVector, rootBase);
    candidate        = dvmin(dvmax(candidate, hiddenBit + 1), 2 * hiddenBit - 2);

    const dvintx inputSignificand = (inputBits & (uint32_t)fractionMask) | (uint32_t)hiddenBit;
    /* E parity is opposite the unbiased-exponent parity. */
    const dvintx inputShift    = (biasedExponent & 1U) ^ 23U;
    const dvintx scaledInput   = dvshift(inputSignificand, inputShift);
    const dvintx candidateHalf = dvshift(candidate, -1);
    const dvintx oddFactor     = (candidate - 1) | 1U;
    const dvintx lower         = dvmulw(candidateHalf, oddFactor, 0);         /* T(M) */
    const dvintx upper         = lower + candidate;                           /* T(M+1) */
    const dvintx lower2        = dvadd2sub(lower, zeroVector + 1, candidate); /* T(M-1) */
    const dvintx upper2        = dvadd2sub(upper, candidate, -1);             /* T(M+2) */

    const dvintx lowCount   = (scaledInput <= lower) + (scaledInput <= lower2);
    const dvintx highCount  = (scaledInput > upper) + (scaledInput > upper2);
    const dvintx corrected  = candidate + highCount - lowCount;
    const dvintx resultBits = dvadd2sub(rootBase, corrected, hiddenBit);
    return as_dvfloatx(dvmux(inputBits == 0, zeroVector, resultBits));
}

static inline dvintx encodeMap(dvfloatx magnitude, uint32_t shift)
{
    const dvfloatx scaled = magnitude * pow2fExact(-(int32_t)shift);
    dvintx encoded        = dvfp_dvintx_trunc(scaled);
    encoded               = dvmux(scaled >= 2147483648.0F, INT32_MAX, encoded);
    dvintx zero;
    zero.lo = zero.hi = replicatew(0);
    return dvmux(scaled > 0.0F, encoded, zero);
}

void coherentFinishMagnitudes(CoherentIntegrationParams const *params, int8_t const *cellShift, int32_t *peakPower,
                              float *foldAccumulator, uint8_t *outputStage, uint32_t validCellCount,
                              bool initializeFold)
{
    const uint32_t blockCount = validCellCount / CI_CELL_VECTOR_LANES;

    AgenWrapper peakWrapper;
    peakWrapper.size = sizeof(int32_t);
    peakWrapper.n1   = blockCount * 2U;
    peakWrapper.s1   = CI_CELL_VECTOR_LANES;
    agen_A peakAgen  = init_A((dvint *)peakPower);
    INIT_AGEN1(peakAgen, peakWrapper);

    AgenWrapper shiftWrapper;
    shiftWrapper.size = sizeof(int8_t);
    shiftWrapper.n1   = blockCount;
    shiftWrapper.s1   = CI_CELL_VECTOR_LANES;
    agen_A shiftAgen  = init_A((vchar *)cellShift);
    INIT_AGEN1(shiftAgen, shiftWrapper);

    AgenWrapper foldWrapper;
    foldWrapper.size     = sizeof(float);
    foldWrapper.n1       = blockCount;
    foldWrapper.s1       = CI_CELL_VECTOR_LANES;
    agen_A foldReadAgen  = init_A((dvfloat *)foldAccumulator);
    agen_A foldWriteAgen = init_A((dvfloat *)foldAccumulator);
    INIT_AGEN1(foldReadAgen, foldWrapper);
    INIT_AGEN1(foldWriteAgen, foldWrapper);

    AgenWrapper outputWrapper;
    outputWrapper.size = sizeof(uint32_t);
    outputWrapper.n1   = blockCount;
    outputWrapper.s1   = CI_CELL_VECTOR_LANES;
    agen_C outputAgen  = init_C((dvint *)outputStage);
    INIT_AGEN1(outputAgen, outputWrapper);

    /* Dot-product power is Q14; sqrt produces Q7. */
    const int32_t restoreBias = (int32_t)params->termShift - 7;
    dvintx restoreBiasVector;
    restoreBiasVector.lo = restoreBiasVector.hi = replicatew(restoreBias);

    dvfloatx previousFold;
    previousFold.lo = previousFold.hi = replicatef(0.0F);

    for (uint32_t block = 0U; block < blockCount; block++) chess_prepare_for_pipelining
    chess_loop_range(1, CI_MAX_PROCESSING_CELL_BLOCKS)
    {
        const dvintx peakLow     = dvuint_load_di(peakAgen);
        const dvintx peakHigh    = dvint_load_di(peakAgen);
        const dvintx peak        = dvshiftadd(peakHigh, 32, peakLow);
        const dvfloatx magnitude = correctlyRoundedSqrtNonnegative(dvintx_dvfp(peak));

        const dvintx reconstructionShift = vchar_dvintx_load_di(shiftAgen);
        const dvfloatx scale             = dvfexp2(dvintx_dvfp(restoreBiasVector - reconstructionShift));
        const dvfloatx restoredMagnitude = magnitude * scale;

        dvfloat_load_di(previousFold, foldReadAgen, !initializeFold);
        vstore_i(previousFold + restoredMagnitude, foldWriteAgen);
        vstore_i(encodeMap(restoredMagnitude, params->coherentIntegrationRxShift), outputAgen);
    }
}

void coherentEncodeFolded(CoherentIntegrationParams const *params, float const *foldAccumulator, uint8_t *outputStage,
                          uint32_t cellCount)
{
    const uint32_t blockCount = cellCount / CI_CELL_VECTOR_LANES;

    AgenWrapper inputWrapper;
    inputWrapper.size = sizeof(float);
    inputWrapper.n1   = blockCount;
    inputWrapper.s1   = CI_CELL_VECTOR_LANES;
    agen_A inputAgen  = init_A((dvfloat *)foldAccumulator);
    INIT_AGEN1(inputAgen, inputWrapper);

    AgenWrapper outputWrapper;
    outputWrapper.size = sizeof(uint32_t);
    outputWrapper.n1   = blockCount;
    outputWrapper.s1   = CI_CELL_VECTOR_LANES;
    agen_C outputAgen  = init_C((dvint *)outputStage);
    INIT_AGEN1(outputAgen, outputWrapper);

    for (uint32_t block = 0U; block < blockCount; block++) chess_prepare_for_pipelining
    chess_loop_range(1, CI_MAX_PROCESSING_CELL_BLOCKS)
    {
        vstore_i(encodeMap(dvfloat_load_di(inputAgen), params->coherentIntegrationFinalShift), outputAgen);
    }
}

void coherentPreprocessRawTile(CoherentIntegrationPreprocessAgenConfig const *agenConfig, uint8_t *rawInput,
                               int8_t *cellShift, int16_t *highLimbs, int16_t *residualLimbs, uint32_t validCellCount)
{
    /* The public contract guarantees complete vector blocks. */
    coherentPreprocessRawTileImpl(agenConfig, rawInput, cellShift, highLimbs, residualLimbs, validCellCount);
}
