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

/*
 * Score, for output (x, y), template T of size w x h and image window I:
 *
 *     C(x,y)  = sum_ij T(i,j) * I(y+i, x+j)          2D correlation
 *     EI(x,y) = sum_ij I(y+i, x+j)^2                 box sum of I^2
 *     D(x,y)  = ET + EI(x,y) - 2*C(x,y)              == sum_ij (T-I)^2
 *     out     = D / sqrt(ET * EI)
 *
 * The identity is exact in int32: the largest template puts ET, EI and 2C all
 * inside 2^31, so D matches the reference's int64 diffSum bit for bit and only
 * the final divide is floating point.
 *
 * LANE ORDER.  VFilt4x2x2 and VSumSq both split their results by column
 * parity, .lo holding the even columns and .hi the odd ones, so every scratch
 * buffer stores a 32-column chunk as
 *
 *     [ 16 even columns 0,2,..,30 ] [ 16 odd columns 1,3,..,31 ]
 *
 * and no stage needs a permute.  The interleaving F32 store undoes the split
 * at the end.
 */

#include "template_matching_knl.h"

#include <cupva_device.h>

void template_matching_init_agens(const TemplateMatchingParams *params, int32_t imgLinePitch, int32_t outLinePitch,
                                  const uint8_t *imgCbStartA, const uint8_t *imgCbStartB, int32_t imgCbLen,
                                  TemplateMatchingConfig *cfg)
{
    const int32_t nChunk  = params->nChunk;
    const int32_t sqPairs = params->sqPairs;

    cfg->nChunk      = nChunk;
    cfg->groupTaps   = params->groupTaps;
    cfg->numGroups   = params->numGroups;
    cfg->sqLoopPairs = sqPairs - 1;
    // The peeled final VSumSq pair reads template column 2*sqPairs-1, which is
    // past the template when w is odd. Masking the operand to zero is branch
    // free and loop invariant, so no zero-filled scratch row is needed.
    cfg->lastTapMask = ((params->templateWidth & 1) != 0) ? 0 : 255;

    AgenWrapper w;

    // box sum: horizontal sum of squares over w taps. Pairs (0,1), (2,3), ...
    // are consumed by VSumSq; the even tap comes from bank A and the odd from
    // bank B so the two reads issue in the same cycle.
    w.size       = (int32_t)sizeof(uint8_t);
    w.n1         = (uint16_t)cfg->sqLoopPairs;
    w.s1         = 2;
    w.n2         = (uint16_t)nChunk;
    w.s2         = TM_CHUNK;
    agen aSqEven = init((vuchar *)NULL);
    INIT_AGEN2(aSqEven, w);
    aSqEven           = update_agen_cb_start(aSqEven, (intptr_t)imgCbStartA);
    aSqEven           = update_agen_cb_size(aSqEven, imgCbLen);
    cfg->sqEvenTapCfg = extract_agen_cfg(aSqEven);
    agen aSqOdd       = init((vuchar *)NULL);
    INIT_AGEN2(aSqOdd, w);
    aSqOdd           = update_agen_cb_start(aSqOdd, (intptr_t)imgCbStartB);
    aSqOdd           = update_agen_cb_size(aSqOdd, imgCbLen);
    cfg->sqOddTapCfg = extract_agen_cfg(aSqOdd);

    w.n1          = (uint16_t)nChunk;
    w.s1          = TM_CHUNK;
    w.n2          = 1;
    w.s2          = TM_CHUNK;
    agen aSqLastA = init((vuchar *)NULL);
    INIT_AGEN2(aSqLastA, w);
    aSqLastA        = update_agen_cb_start(aSqLastA, (intptr_t)imgCbStartA);
    aSqLastA        = update_agen_cb_size(aSqLastA, imgCbLen);
    cfg->sqLastACfg = extract_agen_cfg(aSqLastA);
    agen aSqLastB   = init((vuchar *)NULL);
    INIT_AGEN2(aSqLastB, w);
    aSqLastB        = update_agen_cb_start(aSqLastB, (intptr_t)imgCbStartB);
    aSqLastB        = update_agen_cb_size(aSqLastB, imgCbLen);
    cfg->sqLastBCfg = extract_agen_cfg(aSqLastB);

    // box sum: vertical prefix sum over image rows. V(r) = V(r-1) + HS(r).
    // The de-interleaving load matches the lane order VPromote_DI produces, and
    // the interleaving store puts the row back into even-then-odd column order.
    w.size      = (int32_t)sizeof(int32_t);
    w.n1        = 2;
    w.s1        = TM_CHUNK / 2;
    w.n2        = (uint16_t)nChunk;
    w.s2        = TM_CHUNK;
    agen aVPrev = init((vint *)NULL);
    INIT_AGEN2(aVPrev, w);
    cfg->vPrevCfg = extract_agen_cfg(aVPrev);
    agen aVStore  = init((vint *)NULL);
    INIT_AGEN2(aVStore, w);
    cfg->vStoreCfg = extract_agen_cfg(aVStore);

    // zero fill for V(-1)
    w.n1        = (uint16_t)(TM_SCRATCH_ROW / (TM_CHUNK / 2));
    w.s1        = TM_CHUNK / 2;
    w.n2        = 1;
    w.s2        = TM_CHUNK / 2;
    agen aVZero = init((vint *)NULL);
    INIT_AGEN2(aVZero, w);
    cfg->vZeroCfg = extract_agen_cfg(aVZero);

    // correlation
    // One flat tap walk per chunk: dim1 steps the 4-tap group (+4 bytes),
    // dim2 steps the vertical tap pair (+2 rows), dim3 steps the chunk.
    w.size         = (int32_t)sizeof(uint8_t);
    w.n1           = (uint16_t)params->tapGroupsX;
    w.s1           = 4;
    w.n2           = (uint16_t)params->tapPairsY;
    w.s2           = 2 * imgLinePitch;
    w.n3           = (uint16_t)nChunk;
    w.s3           = TM_CHUNK;
    agen aCorrEven = init((vuchar *)NULL);
    INIT_AGEN3(aCorrEven, w);
    aCorrEven           = update_agen_cb_start(aCorrEven, (intptr_t)imgCbStartA);
    aCorrEven           = update_agen_cb_size(aCorrEven, imgCbLen);
    cfg->corrEvenRowCfg = extract_agen_cfg(aCorrEven);
    agen aCorrOdd       = init((vuchar *)NULL);
    INIT_AGEN3(aCorrOdd, w);
    aCorrOdd           = update_agen_cb_start(aCorrOdd, (intptr_t)imgCbStartB);
    aCorrOdd           = update_agen_cb_size(aCorrOdd, imgCbLen);
    cfg->corrOddRowCfg = extract_agen_cfg(aCorrOdd);

    // 16-byte coefficient record per tap, rewound for every chunk.
    w.n1       = (uint16_t)params->numTaps;
    w.s1       = TM_COEF_BLOCK_BYTES;
    w.n2       = (uint16_t)nChunk;
    w.s2       = 0;
    agen aCoef = init((vuchar *)NULL);
    INIT_AGEN2(aCoef, w);
    cfg->corrCoefCfg = extract_agen_cfg(aCoef);

    // Scratch: [even 16][odd 16] per chunk, two output rows.
    w.size          = (int32_t)sizeof(int32_t);
    w.n1            = 2;
    w.s1            = TM_CHUNK / 2;
    w.n2            = 2;
    w.s2            = TM_SCRATCH_ROW;
    w.n3            = (uint16_t)nChunk;
    w.s3            = TM_CHUNK;
    agen aCorrStore = init((vint *)NULL);
    INIT_AGEN3(aCorrStore, w);
    cfg->corrStoreCfg = extract_agen_cfg(aCorrStore);

    // score epilogue
    agen aEpiCorr = init((vint *)NULL);
    INIT_AGEN3(aEpiCorr, w);
    cfg->epiCorrCfg = extract_agen_cfg(aEpiCorr);

    w.n1       = 2;
    w.s1       = TM_CHUNK / 2;
    w.n2       = (uint16_t)nChunk;
    w.s2       = TM_CHUNK;
    agen aVNew = init((vint *)NULL);
    INIT_AGEN2(aVNew, w);
    cfg->epiVNewCfg = extract_agen_cfg(aVNew);
    agen aVOld      = init((vint *)NULL);
    INIT_AGEN2(aVOld, w);
    cfg->epiVOldCfg = extract_agen_cfg(aVOld);

    // Output: two 16-float halves per chunk, two rows.
    w.size    = (int32_t)sizeof(float);
    w.n1      = 2;
    w.s1      = TM_CHUNK / 2;
    w.n2      = 2;
    w.s2      = outLinePitch;
    w.n3      = (uint16_t)nChunk;
    w.s3      = TM_CHUNK;
    agen aOut = init((vfloat *)NULL);
    INIT_AGEN3(aOut, w);
    cfg->epiOutCfg = extract_agen_cfg(aOut);

    // Two 32-entry DVLDPermHBU pattern vectors: image window, then
    // coefficient replication.
    w.size     = (int32_t)sizeof(uint8_t);
    w.n1       = 2;
    w.s1       = TM_PERM_PATTERN;
    w.n2       = 1;
    w.s2       = TM_PERM_PATTERN;
    agen aPerm = init((vuchar *)NULL);
    INIT_AGEN2(aPerm, w);
    cfg->permCfg = extract_agen_cfg(aPerm);
}

// ET = sum of squared template samples. The pad around the transferred template
// is zero (see the tmplStage declaration), so the whole buffer is swept without
// a mask and the template size is not needed.
// The result is exact in int32 -- at most 31*31*255^2 -- because the score's
// D = EI - 2C + ET identity is an integer one.
int32_t template_matching_template_energy(const uint8_t *stage)
{
    AgenWrapper w;
    w.size = (int32_t)sizeof(uint8_t);
    w.n1   = (uint16_t)TM_STAGE_ROWS;
    w.s1   = TM_STAGE_PITCH;
    w.n2   = 1;
    w.s2   = TM_STAGE_PITCH;

    agen aStage = init((vuchar *)NULL);
    INIT_AGEN2(aStage, w);
    AgenCFG stageCfg = extract_agen_cfg(aStage);
    cupvaModifyAgenCfgBase(&stageCfg, (void *)stage);
    agen_A aRow = init_agen_A_from_cfg(stageCfg);

    const vcharx zero = replicateb(0);
    int32_t energy    = 0;

    for (int32_t r = 0; r < TM_STAGE_ROWS; r++) chess_loop_range(TM_STAGE_ROWS, TM_STAGE_ROWS)
    chess_prepare_for_pipelining
    {
        // One row per VSumSq; the second operand is zeroed so the lane sums hold
        // that row's squares only.
        const dvshortx sq = vsumsq(vuchar_load(aRow), zero);
        energy += vsumr_s(sq.lo) + vsumr_s(sq.hi);
    }

    return energy;
}

// Pack the staged template into the VFilt4x2x2 coefficient blocks: one 16-byte
// record per (vertical tap pair i, horizontal tap group j), bytes 0..7 set 0
// with template rows (2i, 2i+1) and bytes 8..15 set 1 with rows (2i-1, 2i),
// four columns each. Records run in address order, so the 4-byte groups are
// copied with scalar accesses and the store agen never rewinds.
void template_matching_pack_coefficients(const uint8_t *stage, uint8_t *coef, const TemplateMatchingParams *params)
{
    const int32_t ib = params->tapPairsY;
    const int32_t jb = params->tapGroupsX;

    AgenWrapper w;

    // One 4-byte tap group per access, jb groups along a template row.
    w.size = (int32_t)sizeof(uint32_t);
    w.n1   = (uint16_t)jb;
    w.s1   = 1;
    w.n2   = 1;
    w.s2   = 1;

    agen aGroup = init((vuchar *)NULL);
    INIT_AGEN2(aGroup, w);
    AgenCFG groupCfg = extract_agen_cfg(aGroup);

    // Four groups make one record and the records are contiguous.
    w.n1 = 4;
    w.s1 = 1;
    w.n2 = (uint16_t)(ib * jb);
    w.s2 = 4;

    agen aRecord = init((vuchar *)NULL);
    INIT_AGEN2(aRecord, w);
    AgenCFG recordCfg = extract_agen_cfg(aRecord);
    cupvaModifyAgenCfgBase(&recordCfg, coef);
    agen_C aStore = init_agen_C_from_cfg(recordCfg);

    for (int32_t i = 0; i < ib; i++) chess_loop_range(1, TM_MAX_IB)
    {
        // Row 2i-1 is only out of range for i == 0, where the zero row stands
        // in. The transfer never writes that row, so it reads as zero.
        const int32_t prevRow = (i == 0) ? TM_STAGE_ZERO_ROW : ((2 * i) - 1);

        cupvaModifyAgenCfgBase(&groupCfg, (void *)&stage[(2 * i) * TM_STAGE_PITCH]);
        agen_A aEven = init_agen_A_from_cfg(groupCfg);
        cupvaModifyAgenCfgBase(&groupCfg, (void *)&stage[((2 * i) + 1) * TM_STAGE_PITCH]);
        agen_A aOdd = init_agen_A_from_cfg(groupCfg);
        cupvaModifyAgenCfgBase(&groupCfg, (void *)&stage[prevRow * TM_STAGE_PITCH]);
        agen_A aPrev = init_agen_A_from_cfg(groupCfg);

        for (int32_t j = 0; j < jb; j++) chess_loop_range(1, TM_MAX_JB)
        chess_prepare_for_pipelining
        {
            const uint32_t evenGroup = uint_load(aEven);

            uint_store(evenGroup, aStore);
            uint_store(uint_load(aOdd), aStore);
            uint_store(uint_load(aPrev), aStore);
            uint_store(evenGroup, aStore);
        }
    }
}

// Zero fill: V(-1) = 0
void template_matching_zero_v_row(int32_t *vRow, TemplateMatchingConfig *cfg)
{
    cupvaModifyAgenCfgBase(&cfg->vZeroCfg, vRow);
    agen_B aZero = init_agen_B_from_cfg(cfg->vZeroCfg);

    dvintx zero;
    zero.lo = replicatew(0);
    zero.hi = replicatew(0);

    const int32_t zeroIters = TM_SCRATCH_ROW / (TM_CHUNK / 2);
    for (int32_t i = 0; i < zeroIters; i++) chess_loop_range(16, 16)
    chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        vstore(zero, aZero);
    }
}

// The final pair of taps, peeled out of the tap loop. Its odd operand is masked
// off when w is odd.
static inline dvshortx template_matching_peel_pair(agen_A &aLastA, agen_B &aLastB, vcharx lastMask)
{
    const vcharx lastEvenTap = vuchar_load(aLastA);
    const vcharx lastOddRaw  = vuchar_load(aLastB);

    return vsumsq(lastEvenTap, vand(lastOddRaw, lastMask));
}

// Widen the window sums to the int32 prefix row, add V(r-1), store V(r).
static inline void template_matching_accumulate_v(dvshortx hs, agen_B &aVPrev, agen_B &aVStore)
{
    dvintx pe, po;
    vpromote_di(hs.lo, pe.lo, pe.hi);
    vpromote_di(hs.hi, po.lo, po.hi);

    pe = dvadd(pe, dvint_load_di(aVPrev));
    po = dvadd(po, dvint_load_di(aVPrev));

    vstore_i(pe, aVStore);
    vstore_i(po, aVStore);
}

// Box sum: HS(r,x) = sum_{j<w} I(r, x+j)^2, then V(r) = V(r-1) + HS(r).
//
// VSumSq squares two columns per lane. HS fits the 24-bit lane for the widest
// template; VPromote_DI widens it to 48-bit lanes for the int32 prefix row.
void template_matching_box_row(const uint8_t *imgARow, const uint8_t *imgBRow, const uint8_t *lastARow,
                               const uint8_t *lastBRow, const int32_t *vPrevRow, int32_t *vCurRow,
                               TemplateMatchingConfig *cfg)
{
    cupvaModifyAgenCfgBase(&cfg->sqEvenTapCfg, (void *)imgARow);
    cupvaModifyAgenCfgBase(&cfg->sqOddTapCfg, (void *)imgBRow);
    cupvaModifyAgenCfgBase(&cfg->sqLastACfg, (void *)lastARow);
    cupvaModifyAgenCfgBase(&cfg->sqLastBCfg, (void *)lastBRow);
    cupvaModifyAgenCfgBase(&cfg->vPrevCfg, (void *)vPrevRow);
    cupvaModifyAgenCfgBase(&cfg->vStoreCfg, vCurRow);

    agen_A aLastA  = init_agen_A_from_cfg(cfg->sqLastACfg);
    agen_B aLastB  = init_agen_B_from_cfg(cfg->sqLastBCfg);
    agen_B aVPrev  = init_agen_B_from_cfg(cfg->vPrevCfg);
    agen_B aVStore = init_agen_B_from_cfg(cfg->vStoreCfg);

    const int32_t nPairLoop = cfg->sqLoopPairs;
    const vcharx lastMask   = replicateb(cfg->lastTapMask);

    // w <= 2 leaves nothing beyond the peeled pair. Selecting the case here, once
    // per row rather than once per chunk, lets the tap loop below declare a
    // minimum trip count of 1 and pipeline without a zero-trip guard.
    if (nPairLoop == 0)
    {
        for (int32_t u = 0; u < cfg->nChunk; u++) chess_loop_range(TM_MIN_CHUNKS_PER_TILE, TM_CHUNKS_PER_TILE)
        chess_prepare_for_pipelining
        {
            const dvshortx hs = template_matching_peel_pair(aLastA, aLastB, lastMask);

            template_matching_accumulate_v(hs, aVPrev, aVStore);
        }
    }
    else
    {
        agen_A aEven = init_agen_A_from_cfg(cfg->sqEvenTapCfg);
        agen_B aOdd  = init_agen_B_from_cfg(cfg->sqOddTapCfg);

        for (int32_t u = 0; u < cfg->nChunk; u++) chess_loop_range(TM_MIN_CHUNKS_PER_TILE, TM_CHUNKS_PER_TILE)
        chess_prepare_for_pipelining
        {
            dvshortx hs = template_matching_peel_pair(aLastA, aLastB, lastMask);

            for (int32_t t = 0; t < nPairLoop; t++) chess_loop_range(1, 15)
            chess_prepare_for_pipelining
            {
                // Template taps 2t and 2t+1, from the bank A and bank B copies.
                const vcharx evenTap = vuchar_load(aEven);
                const vcharx oddTap  = vuchar_load(aOdd);

                hs = dvadd(hs, vsumsq(evenTap, oddTap));
            }

            template_matching_accumulate_v(hs, aVPrev, aVStore);
        }
    }
}

// One tap is a single VFilt4x2x2: 4 horizontal by 2 vertical template samples
// against the even and odd image rows, updating both output rows' accumulators.
// pred == 0 replaces the accumulators, which starts an accumulator group;
// pred == -1 adds into them. All three agens advance by one tap.
static inline void template_matching_one_tap(agen_A &aEvenRow, agen_B &aOddRow, agen_C &aCoef, vcharx srcPerm,
                                             vcharx coefPerm, int32_t pred, dvshortx &acc0, dvshortx &acc1)
{
    const dvcharx d = dvuchar_load_perm(aEvenRow, srcPerm);
    const dvcharx e = dvuchar_load_perm(aOddRow, srcPerm);
    const dvcharx k = dvuchar_load_perm(aCoef, coefPerm);
    vfilt4x2x2_bbh(d, e, k, acc0, acc1, pred, acc0, acc1);
}

// Correlation: C(x,y) for two output rows.
//
// Coefficient set 0 carries template rows (2i, 2i+1) and lands on output row y;
// set 1 carries rows (2i-1, 2i) and lands on row y+1.
//
// The BBH accumulator is 24 bits and holds at most TM_ACC_MAX_SAMPLES template
// samples, so larger templates are split into `numGroups` groups of `groupTaps`
// instructions, each widened to 48-bit lanes and added in registers. The host
// sizes the groups; when one covers the template the group loop is not entered.
void template_matching_correlate(const uint8_t *imgARow, const uint8_t *imgBRow, const uint8_t *coef,
                                 const uint8_t *permTable, int32_t *corrScratch, TemplateMatchingConfig *cfg)
{
    cupvaModifyAgenCfgBase(&cfg->corrEvenRowCfg, (void *)imgARow);
    cupvaModifyAgenCfgBase(&cfg->corrOddRowCfg, (void *)imgBRow);
    cupvaModifyAgenCfgBase(&cfg->corrCoefCfg, (void *)coef);
    cupvaModifyAgenCfgBase(&cfg->corrStoreCfg, corrScratch);
    cupvaModifyAgenCfgBase(&cfg->permCfg, (void *)permTable);

    agen_A aEvenRow = init_agen_A_from_cfg(cfg->corrEvenRowCfg);
    agen_B aOddRow  = init_agen_B_from_cfg(cfg->corrOddRowCfg);
    agen_C aCoef    = init_agen_C_from_cfg(cfg->corrCoefCfg);
    agen_B aStore   = init_agen_B_from_cfg(cfg->corrStoreCfg);

    // DVLDPermHBU pattern: halfwords 0..15 -> .lo (bytes 0..31),
    // halfwords 4..19 -> .hi (bytes 8..39) == D[0..15] per 8-lane group.
    agen_A aPerm    = init_agen_A_from_cfg(cfg->permCfg);
    vcharx srcPerm  = vuchar_load(aPerm);
    vcharx coefPerm = vuchar_load(aPerm);

    const int32_t tapLoop   = cfg->groupTaps - 1;
    const int32_t groupLoop = cfg->numGroups - 1;

    for (int32_t c = 0; c < cfg->nChunk; c++) chess_loop_range(TM_MIN_CHUNKS_PER_TILE, TM_CHUNKS_PER_TILE)
    {
        dvshortx acc0, acc1;
        dvintx sumEven0, sumOdd0, sumEven1, sumOdd1;

        // First tap of the group replaces the accumulators, the rest add.
        template_matching_one_tap(aEvenRow, aOddRow, aCoef, srcPerm, coefPerm, 0, acc0, acc1);
        // groupTaps = groupPairs * ceil(w/4) is at least 2: w >= 5 makes the
        // second factor at least 2, and w <= 4 makes w*h fit one accumulator
        // group, so groupPairs is the full tapPairs, which the host pads to 2
        // for the one shape class that would otherwise give 1. So this loop
        // always runs, and saying so lets chess pipeline it without a
        // zero-trip guard.
        for (int32_t t = 0; t < tapLoop; t++) chess_loop_range(1, 127)
        chess_prepare_for_pipelining
        {
            template_matching_one_tap(aEvenRow, aOddRow, aCoef, srcPerm, coefPerm, -1, acc0, acc1);
        }

        vpromote_di(acc0.lo, sumEven0.lo, sumEven0.hi);
        vpromote_di(acc0.hi, sumOdd0.lo, sumOdd0.hi);
        vpromote_di(acc1.lo, sumEven1.lo, sumEven1.hi);
        vpromote_di(acc1.hi, sumOdd1.lo, sumOdd1.hi);

        // Most template sizes fit one group, leaving this loop a zero trip
        // count, which nvvpu rejects as a minimum. Hoisting the zero case into
        // a branch costs chess nothing: it folds the branch away.
        if (groupLoop > 0)
        {
            for (int32_t g = 0; g < groupLoop; g++) chess_loop_range(1, 15)
            {
                template_matching_one_tap(aEvenRow, aOddRow, aCoef, srcPerm, coefPerm, 0, acc0, acc1);
                for (int32_t t = 0; t < tapLoop; t++) chess_loop_range(1, 127)
                chess_prepare_for_pipelining
                {
                    template_matching_one_tap(aEvenRow, aOddRow, aCoef, srcPerm, coefPerm, -1, acc0, acc1);
                }

                sumEven0 = dvadd(sumEven0, vpromote_di(acc0.lo));
                sumOdd0  = dvadd(sumOdd0, vpromote_di(acc0.hi));
                sumEven1 = dvadd(sumEven1, vpromote_di(acc1.lo));
                sumOdd1  = dvadd(sumOdd1, vpromote_di(acc1.hi));
            }
        }

        vstore_i(sumEven0, aStore);
        vstore_i(sumOdd0, aStore);
        vstore_i(sumEven1, aStore);
        vstore_i(sumOdd1, aStore);
    }
}

// Score: out = (ET + EI - 2C) * rsqrt(ET * EI) for two output rows.
//
// EI(y) = V(y+h-1) - V(y-1) from the vertical prefix rows. VRSQ's relative
// error is far below the reference's 1e-4 absolute tolerance for the score
// range this operator produces, and it reproduces the reference's IEEE Inf/NaN
// behaviour when ET*EI is zero.
//
// D stays on double vectors; the floating point tail runs on single vectors so
// the even and odd halves land in one register pair for the interleaving store.
static inline vfloatx template_matching_one_score(vintx diff, vintx imageEnergy, float templateEnergyF)
{
    const vfloatx rsq = vfrsq(vmulf(vintx_vfp(imageEnergy), templateEnergyF));
    return vmulf(vintx_vfp(diff), rsq);
}

void template_matching_score(const int32_t *corrScratch, const int32_t *vNewRow0, const int32_t *vOldRow0,
                             const int32_t *vNewRow1, const int32_t *vOldRow1, float *outTile,
                             TemplateMatchingConfig *cfg)
{
    cupvaModifyAgenCfgBase(&cfg->epiCorrCfg, (void *)corrScratch);
    cupvaModifyAgenCfgBase(&cfg->epiOutCfg, outTile);

    agen_B aCorr = init_agen_B_from_cfg(cfg->epiCorrCfg);
    agen_C aOut  = init_agen_C_from_cfg(cfg->epiOutCfg);

    cupvaModifyAgenCfgBase(&cfg->epiVNewCfg, (void *)vNewRow0);
    agen_B aNew0 = init_agen_B_from_cfg(cfg->epiVNewCfg);
    cupvaModifyAgenCfgBase(&cfg->epiVOldCfg, (void *)vOldRow0);
    agen_B aOld0 = init_agen_B_from_cfg(cfg->epiVOldCfg);
    cupvaModifyAgenCfgBase(&cfg->epiVNewCfg, (void *)vNewRow1);
    agen_B aNew1 = init_agen_B_from_cfg(cfg->epiVNewCfg);
    cupvaModifyAgenCfgBase(&cfg->epiVOldCfg, (void *)vOldRow1);
    agen_B aOld1 = init_agen_B_from_cfg(cfg->epiVOldCfg);

    const int32_t energy      = cfg->templateEnergy;
    const float energyAsFloat = (float)energy;

    /* nChunk is 4 or 8, so the unroll always divides the trip count. */
    for (int32_t c = 0; c < cfg->nChunk; c++) chess_loop_range(TM_MIN_CHUNKS_PER_TILE, TM_CHUNKS_PER_TILE)
    chess_unroll_loop(TM_SCORE_UNROLL)
    chess_prepare_for_pipelining
    {
        /* Output row y. */
        const dvintx cEven0 = dvint_load(aCorr);
        const dvintx cOdd0  = dvint_load(aCorr);
        const dvintx eEven0 = dvsub(dvint_load(aNew0), dvint_load(aOld0));
        const dvintx eOdd0  = dvsub(dvint_load(aNew0), dvint_load(aOld0));

        const dvintx dEven0 = dvadd(dvsub(eEven0, dvadd(cEven0, cEven0)), energy);
        const dvintx dOdd0  = dvadd(dvsub(eOdd0, dvadd(cOdd0, cOdd0)), energy);

        dvfloatx low0, high0;
        low0.lo  = template_matching_one_score(dEven0.lo, eEven0.lo, energyAsFloat);
        low0.hi  = template_matching_one_score(dOdd0.lo, eOdd0.lo, energyAsFloat);
        high0.lo = template_matching_one_score(dEven0.hi, eEven0.hi, energyAsFloat);
        high0.hi = template_matching_one_score(dOdd0.hi, eOdd0.hi, energyAsFloat);

        /* Output row y+1. */
        const dvintx cEven1 = dvint_load(aCorr);
        const dvintx cOdd1  = dvint_load(aCorr);
        const dvintx eEven1 = dvsub(dvint_load(aNew1), dvint_load(aOld1));
        const dvintx eOdd1  = dvsub(dvint_load(aNew1), dvint_load(aOld1));

        const dvintx dEven1 = dvadd(dvsub(eEven1, dvadd(cEven1, cEven1)), energy);
        const dvintx dOdd1  = dvadd(dvsub(eOdd1, dvadd(cOdd1, cOdd1)), energy);

        dvfloatx low1, high1;
        low1.lo  = template_matching_one_score(dEven1.lo, eEven1.lo, energyAsFloat);
        low1.hi  = template_matching_one_score(dOdd1.lo, eOdd1.lo, energyAsFloat);
        high1.lo = template_matching_one_score(dEven1.hi, eEven1.hi, energyAsFloat);
        high1.hi = template_matching_one_score(dOdd1.hi, eOdd1.hi, energyAsFloat);

        vstore_i(low0, aOut);
        vstore_i(high0, aOut);
        vstore_i(low1, aOut);
        vstore_i(high1, aOut);
    }
}
