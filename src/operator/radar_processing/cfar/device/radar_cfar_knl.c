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

#include "radar_cfar_knl.h"

/*
 * chess_loop_range bounds below are literals rather than expressions over the CFAR_*
 * macros because the hint parser only accepts digits. They correspond to:
 *   65   = (CFAR_MAX_DIM + CFAR_LANES) / CFAR_LANES         vectors per widest row
 *   16   = (CFAR_MAX_WINDOW + CFAR_LANES) / CFAR_LANES, and CFAR_LANES (rows per strip)
 *   97   = (CFAR_MAX_PADDED + 2 * CFAR_LANES) / CFAR_LANES  vectors per padded row
 *   255  = CFAR_MAX_WINDOW                                  training cells per side
 *   254  = CFAR_MAX_WINDOW - 1                              seed iterations (train - 1)
 *   1024 = CFAR_MAX_DIM                                     marched extent
 * Keep them in sync if the corresponding macro changes.
 */

/*
 * Fixed-point noise and threshold arithmetic.
 *
 * The CPU reference computes, in int64:
 *     recip = 65536 / count
 *     noise = (windowSum * recip) / 65536
 *     thr   = (noise * factorQ14) / 16384
 * Every operand is non-negative, so each division is an arithmetic right shift.
 *
 * A word lane is extended to 48 bits in the register file (Programmer's Guide, Vector
 * Register File), and
 * VMulW multiplies 33-bit x 33-bit into that 48-bit lane (Programmer's Guide, VMulW).
 * Neither product fits a 33-bit operand directly, so each is evaluated by splitting the
 * left operand at the shift point. For any non-negative s and r,
 *     floor(((sHi << k) + sLo) * r / 2^k) == sHi * r + floor(sLo * r / 2^k)
 * because the first term is an exact multiple of 2^k. Both halves stay inside the 48-bit
 * lane, so the result is bit-identical to the reference's int64 value.
 */

/** Broadcast a scalar to all 16 lanes of a double vector. */
static inline dvintx cfarSplat(int32_t v)
{
    dvintx r;
    r.lo = replicatew(v);
    r.hi = r.lo;
    return r;
}

/**
 * Noise estimate: floor(sum * recip / 2^16).
 *
 * numTrain + numGuard <= 255 bounds @p sum by 510 * (2^32 - 1) < 2^41, and @p recip by
 * 2^16. The split gives sHi * recip <= 2^41 and (sLo * recip) >> 16 <= 2^16. Since
 * recip <= 2^16 / count, the result is bounded by sum / count, i.e. by the largest input
 * magnitude, so it always fits in 32 bits unsigned.
 */
static inline dvintx cfarNoise(dvintx sum, dvintx recip)
{
    dvintx const sHi = dvsra(sum, CFAR_RECIP_QBITS);
    dvintx const sLo = dvand(sum, (unsigned)(CFAR_RECIP_ONE - 1));
    return dvadd(dvmulw(sHi, recip, VPU_TRUNC_0), dvmulw(sLo, recip, VPU_TRUNC_16));
}

/**
 * Detection threshold: floor(noise * factorQ14 / 2^14).
 *
 * @p noise is bounded by 2^32 and the factor by 2^28 (enforced at Create), so the split
 * gives nHi * factor <= 2^18 * 2^28 = 2^46 and nLo * factor <= 2^14 * 2^28 = 2^42, both
 * inside the 48-bit lane. The threshold itself may exceed 2^32, which is why it is kept
 * in a 48-bit lane rather than saturated: comparing it against a 32-bit magnitude then
 * yields the correct "not detected" without any clamp. VMulW offers no truncate-by-14,
 * so the low half is shifted separately.
 */
static inline dvintx cfarThreshold(dvintx noise, dvintx factorQ14)
{
    dvintx const nHi = dvsra(noise, CFAR_QBITS);
    dvintx const nLo = dvand(noise, (unsigned)(CFAR_FIXED_POINT_SCALE - 1));
    return dvadd(dvmulw(nHi, factorQ14, VPU_TRUNC_0), dvsra(dvmulw(nLo, factorQ14, VPU_TRUNC_0), CFAR_QBITS));
}

/**
 * Combine the trailing and leading noise estimates for the configured algorithm.
 *
 * Selected at build time so the inner loop carries no branch and each variant pays only
 * for the reciprocal multiplies it needs: CA divides one combined sum, GO and SO divide
 * the two sums separately and then reduce.
 */
#if defined(CFAR_ALGO_CA)
static inline dvintx cfarCombine(dvintx trailSum, dvintx leadSum, dvintx recipTotal, dvintx recipTrail,
                                 dvintx recipLead)
{
    (void)recipTrail;
    (void)recipLead;
    return cfarNoise(dvadd(trailSum, leadSum), recipTotal);
}
#elif defined(CFAR_ALGO_GO)
static inline dvintx cfarCombine(dvintx trailSum, dvintx leadSum, dvintx recipTotal, dvintx recipTrail,
                                 dvintx recipLead)
{
    (void)recipTotal;
    return dvmax(cfarNoise(trailSum, recipTrail), cfarNoise(leadSum, recipLead));
}
#elif defined(CFAR_ALGO_SO)
static inline dvintx cfarCombine(dvintx trailSum, dvintx leadSum, dvintx recipTotal, dvintx recipTrail,
                                 dvintx recipLead)
{
    (void)recipTotal;
    return dvmin(cfarNoise(trailSum, recipTrail), cfarNoise(leadSum, recipLead));
}
#else
#    error "One of CFAR_ALGO_CA, CFAR_ALGO_GO or CFAR_ALGO_SO must be defined"
#endif

void cfarInitLaneIdx(int32_t *restrict laneIdx)
{
    /*
     * vcollate_idx_bits on an all-lanes-set predicate returns the identity permutation,
     * i.e. {0, 1, ... 7} per single vector. The high half is that plus 8.
     */
    dvintx const ones = cfarSplat(-1);

    int32_t bits;
    vintx ramp;
    vcollate_idx_bits(ones.lo, ramp, bits);

    dvintx idx;
    idx.lo = ramp;
    idx.hi = vadd(ramp, CFAR_LANES / 2);

    agen aSt;
    aSt      = init((vint *)laneIdx);
    aSt.n1   = 1;
    aSt.mod1 = CFAR_LANES * (int32_t)sizeof(int32_t);
    vstore(idx, aSt);
}

void cfarPadRowStrip(uint32_t const *restrict tile, int32_t tileLp, uint32_t *restrict pad, int32_t width,
                     int32_t window, int32_t rows, int32_t cyclic)
{
    /*
     * Store order is left halo, then body, then right halo, then the slop word. Every
     * store is a whole vector, and neither the halo width nor the width is guaranteed to
     * be a multiple of CFAR_LANES (width is only a multiple of 8), so each region's final
     * store can overrun forward by up to CFAR_LANES-1 elements. This order makes every
     * overrun harmless because the next region rewrites exactly what was clobbered:
     *   - the left halo's tail spills into the start of the body, which the body then writes;
     *   - the body's tail spills into the start of the right halo, which the right halo then
     *     writes;
     *   - the right halo's tail spills into the trailing pitch slop that CFAR_H_PAD_LP
     *     reserves, and the slop word is written last.
     * The order is load-bearing: writing the halos before the body leaves the right halo
     * clobbered for every width where width % CFAR_LANES == 8 (e.g. 1016, 24), which no
     * width in either test matrix exercises.
     */
    int32_t const nVec    = (width + CFAR_LANES - 1) / CFAR_LANES;
    int32_t const haloVec = (window + CFAR_LANES - 1) / CFAR_LANES;
    int32_t const total   = width + 2 * window + 1;
    int32_t const step    = CFAR_LANES * (int32_t)sizeof(uint32_t);
    dvintx const zero     = cfarSplat(0);

    int32_t const haloVecEven  = (haloVec + 1) & ~1;
    int32_t const nVecEven     = (nVec + 1) & ~1;
    int32_t const totalVecEven = ((total + CFAR_LANES - 1) / CFAR_LANES + 1) & ~1;

    for (int32_t r = 0; r < rows; r++) chess_loop_range(1, 16)
    {
        uint32_t const *restrict src = tile + (int32_t)r * tileLp;
        uint32_t *restrict dst       = pad + (int32_t)r * CFAR_H_PAD_LP;

        /*
         * Left halo: dst[k*L ..] <- the row's last `window` elements. Sourced from the DMA
         * tile rather than from dst, because the body has not been written yet at this
         * point. src + width - window + haloVec*L - 1 <= src + width + CFAR_LANES - 2 stays
         * inside the tile row: tileLp >= width and the tile buffer holds CFAR_LANES rows of
         * it, so the read lands in the next row at worst, never past the buffer (row
         * CFAR_LANES-1 is the last, and inBuf is sized CFAR_LANES * CFAR_MAX_DIM with
         * width <= CFAR_MAX_DIM).
         */
        agen aHaloL, aSrcL;
        aHaloL      = init((vuint *)dst);
        aHaloL.n1   = haloVec;
        aHaloL.mod1 = step;
        aSrcL       = init((vuint *)(src + width - window));
        aSrcL.n1    = haloVec;
        aSrcL.mod1  = step;

        if (cyclic != 0)
        {
            for (int32_t i = 0; i < haloVecEven; i++) chess_prepare_for_pipelining
            chess_unroll_loop(2)
            chess_loop_range(1, 16)
            {
                vstore(dvuint_load(aSrcL), aHaloL);
            }
        }
        else
        {
            for (int32_t i = 0; i < haloVecEven; i++) chess_prepare_for_pipelining
            chess_unroll_loop(2)
            chess_loop_range(1, 16)
            {
                vstore(zero, aHaloL);
            }
        }

        /* Body: the image row lands at dst[window .. window+width). */
        agen aLd, aSt;
        aLd      = init((vuint *)src);
        aLd.n1   = nVec;
        aLd.mod1 = step;
        aSt      = init((vuint *)(dst + window));
        aSt.n1   = nVec;
        aSt.mod1 = step;

        for (int32_t i = 0; i < nVecEven; i++) chess_prepare_for_pipelining
        chess_unroll_loop(2)
        chess_loop_range(1, 65)
        {
            vstore(dvuint_load(aLd), aSt);
        }

        /* Right halo: dst[window+width ..] <- the row's first `window` elements, now
         * available from the body at dst[window ..]. */
        agen aHaloR, aSrcR;
        aHaloR      = init((vuint *)(dst + window + width));
        aHaloR.n1   = haloVec;
        aHaloR.mod1 = step;
        aSrcR       = init((vuint *)(dst + window));
        aSrcR.n1    = haloVec;
        aSrcR.mod1  = step;

        if (cyclic != 0)
        {
            for (int32_t i = 0; i < haloVecEven; i++) chess_prepare_for_pipelining
            chess_unroll_loop(2)
            chess_loop_range(1, 16)
            {
                vstore(dvuint_load(aSrcR), aHaloR);
            }
        }
        else
        {
            for (int32_t i = 0; i < haloVecEven; i++) chess_prepare_for_pipelining
            chess_unroll_loop(2)
            chess_loop_range(1, 16)
            {
                vstore(zero, aHaloR);
            }
        }

        /* One element of read-ahead slop for the leading-window entry head. */
        dst[total - 1] = 0u;
    }

    /* Zero the lanes past the last valid row so they cannot manufacture detections. */
    for (int32_t r = rows; r < CFAR_LANES; r++)
    {
        uint32_t *restrict dst = pad + (int32_t)r * CFAR_H_PAD_LP;

        agen aSt;
        aSt      = init((vuint *)dst);
        aSt.n1   = (total + CFAR_LANES - 1) / CFAR_LANES;
        aSt.mod1 = CFAR_LANES * (int32_t)sizeof(uint32_t);

        for (int32_t i = 0; i < totalVecEven; i++) chess_prepare_for_pipelining
        chess_unroll_loop(2)
        chess_loop_range(1, 97)
        {
            vstore(cfarSplat(0), aSt);
        }
    }
}

void cfarPadColStrip(uint32_t const *restrict tile, int32_t tileLp, uint32_t *restrict pad, int32_t height,
                     int32_t window, int32_t cyclic)
{
    /* A row of a column strip is CFAR_LANES contiguous elements, i.e. exactly one double
     * vector, so both the body copy and the halo wrap are straight vector copies. */
    agen aLd;
    aLd      = init((vuint *)tile);
    aLd.n1   = height;
    aLd.mod1 = tileLp * (int32_t)sizeof(uint32_t);

    agen aSt;
    aSt      = init((vuint *)(pad + (int32_t)window * CFAR_LANES));
    aSt.n1   = height;
    aSt.mod1 = CFAR_LANES * (int32_t)sizeof(uint32_t);

    for (int32_t i = 0; i < height; i++) chess_prepare_for_pipelining
    chess_unroll_loop(2)
    chess_loop_range(8, 1024)
    {
        vstore(dvuint_load(aLd), aSt);
    }

    /* Top halo reads the last `window` image rows, bottom halo the first `window`. */
    agen aLdTop;
    aLdTop      = init((vuint *)(pad + (int32_t)height * CFAR_LANES));
    aLdTop.n1   = window;
    aLdTop.mod1 = CFAR_LANES * (int32_t)sizeof(uint32_t);

    agen aStTop;
    aStTop      = init((vuint *)pad);
    aStTop.n1   = window;
    aStTop.mod1 = CFAR_LANES * (int32_t)sizeof(uint32_t);

    agen aLdBot;
    aLdBot      = init((vuint *)(pad + (int32_t)window * CFAR_LANES));
    aLdBot.n1   = window;
    aLdBot.mod1 = CFAR_LANES * (int32_t)sizeof(uint32_t);

    agen aStBot;
    aStBot      = init((vuint *)(pad + (int32_t)(window + height) * CFAR_LANES));
    aStBot.n1   = window;
    aStBot.mod1 = CFAR_LANES * (int32_t)sizeof(uint32_t);

    dvintx const zero = cfarSplat(0);

    int32_t const windowEven = (window + 1) & ~1;

    /* The branch is hoisted out of the loop: a conditional load inside it would both add
     * a branch to the pipelined body and leave the agen advanced inconsistently. */
    if (cyclic != 0)
    {
        for (int32_t i = 0; i < windowEven; i++) chess_prepare_for_pipelining
        chess_unroll_loop(2)
        chess_loop_range(1, 256)
        {
            vstore(dvuint_load(aLdTop), aStTop);
            vstore(dvuint_load(aLdBot), aStBot);
        }
    }
    else
    {
        for (int32_t i = 0; i < windowEven; i++) chess_prepare_for_pipelining
        chess_unroll_loop(2)
        chess_loop_range(1, 256)
        {
            vstore(zero, aStTop);
            vstore(zero, aStBot);
        }
    }

    /* One row of read-ahead slop for the leading-window entry head. */
    agen aStSlop;
    aStSlop      = init((vuint *)(pad + (int32_t)(2 * window + height) * CFAR_LANES));
    aStSlop.n1   = 1;
    aStSlop.mod1 = CFAR_LANES * (int32_t)sizeof(uint32_t);
    vstore(zero, aStSlop);
}

void cfarHorizontalStrip(uint32_t const *restrict pad, int32_t *restrict mask, int32_t maskLp, int32_t maskLaneOfst,
                         int32_t const *restrict recipTotal, int32_t const *restrict recipTrail,
                         int32_t const *restrict recipLead, int32_t const *restrict factorQ14,
                         CfarHost2DevParam const *restrict p)
{
    int32_t const width  = p->width;
    int32_t const window = p->horWindow;
    int32_t const train  = p->numHorTrain;
    int32_t const guard  = p->numHorGuard;

    /* Per-row threshold factors: the lanes hold CFAR_LANES consecutive image rows, so
     * the horizontal factor is a per-lane vector here (the vertical one is a scalar). */
    agen aFactor;
    aFactor              = init((vint *)factorQ14);
    aFactor.n1           = 1;
    aFactor.mod1         = CFAR_LANES * (int32_t)sizeof(int32_t);
    dvintx const factorV = dvint_load(aFactor);

    /* Grouping-off mask: makes the neighbour test pass without a branch in the loop. */
    dvintx const groupOff = cfarSplat(p->peakGrouping != 0 ? 0 : -1);

    /*
     * Seed the running sums for x == 0. With px = x + window the trailing training cells
     * are [x, x + train) and the leading ones are [x + window + guard + 1, x + 2*window].
     */
    dvintx trailSum = cfarSplat(0);
    dvintx leadSum  = cfarSplat(0);
    {
        agen aT;
        aT           = init((vuint *)pad);
        aT.n1        = train;
        aT.mod1      = (int32_t)sizeof(uint32_t);
        aT.lane_ofst = CFAR_H_PAD_LANE_OFST;

        agen aL;
        aL           = init((vuint *)(pad + window + guard + 1));
        aL.n1        = train;
        aL.mod1      = (int32_t)sizeof(uint32_t);
        aL.lane_ofst = CFAR_H_PAD_LANE_OFST;

        /* train == 1 makes this a zero-trip loop, so it is guarded to keep the hint's lower
         * bound of 1 truthful. */
        if (train > 1)
        {
            for (int32_t i = 0; i < train - 1; i++) chess_prepare_for_pipelining
            chess_loop_range(1, 254)
            {
                trailSum = dvadd(trailSum, dvuint_load_transp(aT));
                leadSum  = dvadd(leadSum, dvuint_load_transp(aL));
            }
        }
        /* final element accumulated explicitly */
        trailSum = dvadd(trailSum, dvuint_load_transp(aT));
        leadSum  = dvadd(leadSum, dvuint_load_transp(aL));
    }

    /*
     * Read heads. The VPU has 8 address generators; using more forces the compiler to spill
     * and restore agen state every iteration (AgenCfgST/AgenCfgLD), which costs more than the
     * loop body itself. Two economies keep the count at 6:
     *
     *  - The {left, cur, right} neighbour taps are three consecutive x positions, so they
     *    are carried in a 3-deep register history advanced once per iteration instead of
     *    three separate agens reading the same row at fixed offsets.
     *  - The two transposed mask stores share one 2D agen: dimension 1 walks the two
     *    8-row halves (stride 8 * maskLp), dimension 2 advances one column.
     */
    agen aNext, aSubT, aAddT, aSubL, aAddL;

#define CFAR_H_HEAD(agenVar, startCol, count)                    \
    do                                                           \
    {                                                            \
        (agenVar)           = init((vuint *)(pad + (startCol))); \
        (agenVar).n1        = (count);                           \
        (agenVar).mod1      = (int32_t)sizeof(uint32_t);         \
        (agenVar).lane_ofst = CFAR_H_PAD_LANE_OFST;              \
    } while (0)

    /* One column ahead of the cell under test, to feed the rolling history. */
    CFAR_H_HEAD(aNext, window + 1, width);
    CFAR_H_HEAD(aSubT, 0, width);
    CFAR_H_HEAD(aAddT, train, width);
    CFAR_H_HEAD(aSubL, window + guard + 1, width);
    CFAR_H_HEAD(aAddL, 2 * window + 1, width);
#undef CFAR_H_HEAD

    /* Prime the history: left = x-1 (padded), cur = x=0. */
    dvintx prev;
    dvintx cur;
    {
        agen aSeed;
        aSeed           = init((vuint *)(pad + window - 1));
        aSeed.n1        = 2;
        aSeed.mod1      = (int32_t)sizeof(uint32_t);
        aSeed.lane_ofst = CFAR_H_PAD_LANE_OFST;
        prev            = dvuint_load_transp(aSeed);
        cur             = dvuint_load_transp(aSeed);
    }

    /*
     * The decision is written transposed so the output RasterDataFlow lands it in natural
     * row-major order in DRAM. vstore_transp takes a single vector (8 lanes, one per row),
     * so one 2D agen emits the low half then the high half of each predicate.
     */
    agen aMask;
    {
        AgenWrapper w;
        w.size          = sizeof(int32_t);
        w.n1            = 2;
        w.s1            = (CFAR_LANES / 2) * maskLp;
        w.n2            = width;
        w.s2            = 1;
        aMask           = init((vint *)mask);
        aMask.lane_ofst = maskLaneOfst;
        INIT_AGEN2(aMask, w);
    }

    /* The reciprocal tables vary along the marched axis in "full" mode and are constant
     * in cyclic mode; either way they are scalar per iteration, not per lane. */
    for (int32_t x = 0; x < width; x++) chess_prepare_for_pipelining
    chess_unroll_loop(2)
    chess_loop_range(8, 1024)
    {
        dvintx const next = dvuint_load_transp(aNext);

        dvintx const noise =
            cfarCombine(trailSum, leadSum, cfarSplat(recipTotal[x]), cfarSplat(recipTrail[x]), cfarSplat(recipLead[x]));
        dvintx const thr = cfarThreshold(noise, factorV);

        dvintx const peak = dvand(dvCmpGE(cur, prev), dvCmpGE(cur, next));
        dvintx const det  = dvand(dvCmpGE(cur, thr), dvor(peak, groupOff));

        vstore_transp(det.lo, aMask);
        vstore_transp(det.hi, aMask);

        prev = cur;
        cur  = next;

        trailSum = dvadd(trailSum, dvsub(dvuint_load_transp(aAddT), dvuint_load_transp(aSubT)));
        leadSum  = dvadd(leadSum, dvsub(dvuint_load_transp(aAddL), dvuint_load_transp(aSubL)));
    }
}

int32_t cfarVerticalStrip(uint32_t const *restrict pad, int32_t const *restrict mask, int32_t maskLp,
                          int32_t const *restrict laneIdx, int32_t const *restrict recipTotal,
                          int32_t const *restrict recipTrail, int32_t const *restrict recipLead, int32_t x0, int32_t y0,
                          int32_t rows, int32_t *restrict stageY, int32_t *restrict stageX, int32_t room,
                          CfarHost2DevParam const *restrict p)
{
    int32_t const window = p->verWindow;
    int32_t const train  = p->numVerTrain;
    int32_t const guard  = p->numVerGuard;

    dvintx const factorV  = cfarSplat(p->verFactorQ14);
    dvintx const groupOff = cfarSplat(p->peakGrouping != 0 ? 0 : -1);

    /* Image x of each lane, plus the mask that drops lanes past the right edge. */
    agen aLane;
    aLane                = init((vint *)laneIdx);
    aLane.n1             = 1;
    aLane.mod1           = CFAR_LANES * (int32_t)sizeof(int32_t);
    dvintx const xVec    = dvadd(dvint_load(aLane), x0);
    dvintx const inImage = dvCmpLT(xVec, cfarSplat(p->width));

    dvintx trailSum = cfarSplat(0);
    dvintx leadSum  = cfarSplat(0);
    {
        agen aT;
        aT      = init((vuint *)(pad + (int32_t)y0 * CFAR_LANES));
        aT.n1   = train;
        aT.mod1 = CFAR_LANES * (int32_t)sizeof(uint32_t);

        agen aL;
        aL      = init((vuint *)(pad + (int32_t)(y0 + window + guard + 1) * CFAR_LANES));
        aL.n1   = train;
        aL.mod1 = CFAR_LANES * (int32_t)sizeof(uint32_t);

        for (int32_t i = 0; i < train; i++) chess_prepare_for_pipelining
        chess_loop_range(1, 255)
        {
            trailSum = dvadd(trailSum, dvuint_load(aT));
            leadSum  = dvadd(leadSum, dvuint_load(aL));
        }
    }

    /* Same agen economy as the horizontal pass: the {up, cur, down} taps become a 3-deep
     * register history, leaving 6 agens and no agen spill in the loop body. */
    agen aNext, aSubT, aAddT, aSubL, aAddL;

#define CFAR_V_HEAD(agenVar, startRow, count)                                   \
    do                                                                          \
    {                                                                           \
        (agenVar)      = init((vuint *)(pad + (int32_t)(startRow)*CFAR_LANES)); \
        (agenVar).n1   = (count);                                               \
        (agenVar).mod1 = CFAR_LANES * (int32_t)sizeof(uint32_t);                \
    } while (0)

    CFAR_V_HEAD(aNext, y0 + window + 1, rows);
    CFAR_V_HEAD(aSubT, y0, rows);
    CFAR_V_HEAD(aAddT, y0 + train, rows);
    CFAR_V_HEAD(aSubL, y0 + window + guard + 1, rows);
    CFAR_V_HEAD(aAddL, y0 + 2 * window + 1, rows);
#undef CFAR_V_HEAD

    dvintx prev;
    dvintx cur;
    {
        agen aSeed;
        aSeed      = init((vuint *)(pad + (int32_t)(y0 + window - 1) * CFAR_LANES));
        aSeed.n1   = 2;
        aSeed.mod1 = CFAR_LANES * (int32_t)sizeof(uint32_t);
        prev       = dvuint_load(aSeed);
        cur        = dvuint_load(aSeed);
    }

    agen aMask;
    aMask      = init((vint *)(mask + (int32_t)y0 * maskLp));
    aMask.n1   = rows;
    aMask.mod1 = maskLp * (int32_t)sizeof(int32_t);

    int32_t count = 0;

    for (int32_t y = y0; y < y0 + rows; y++) chess_prepare_for_pipelining
    chess_unroll_loop(2)
    chess_loop_range(8, 1024)
    {
        dvintx const next = dvuint_load(aNext);
        dvintx const hDet = dvint_load(aMask);

        dvintx const noise =
            cfarCombine(trailSum, leadSum, cfarSplat(recipTotal[y]), cfarSplat(recipTrail[y]), cfarSplat(recipLead[y]));
        dvintx const thr = cfarThreshold(noise, factorV);

        dvintx const peak = dvand(dvCmpGE(cur, prev), dvCmpGE(cur, next));
        dvintx const det  = dvand(dvand(dvCmpGE(cur, thr), hDet), dvand(dvor(peak, groupOff), inImage));

        /*
         * Compaction: vcollate_idx_bits yields a permutation gathering the passing lanes
         * to the front plus a bitmask of which passed. The full vector is stored
         * unconditionally and the cursor advances by the popcount, so slots past the
         * count are overwritten next iteration. vcollate_idx_bits has no double-vector
         * form, so the halves are collated separately.
         *
         * The write cursor is clamped rather than branched on: once the caller's budget
         * is used up the same slots are rewritten and the count stops growing, which is
         * the documented saturation at the detection-list capacity.
         */
        int32_t bits;
        vintx idxLo;
        vcollate_idx_bits(det.lo, idxLo, bits);
        int32_t const nLo = bitcount(bits);

        vintx idxHi;
        vcollate_idx_bits(det.hi, idxHi, bits);
        int32_t const nHi = bitcount(bits);

        int32_t const pos = count < room ? count : room;

        *(vint *)(stageX + pos)       = extract(vpermute(xVec.lo, idxLo));
        *(vint *)(stageX + pos + nLo) = extract(vpermute(xVec.hi, idxHi));
        /* y is lane-invariant, so it needs no permutation. */
        *(dvint *)(stageY + pos) = extract(cfarSplat(y));

        int32_t const advanced = count + nLo + nHi;
        count                  = advanced < room ? advanced : room;

        prev = cur;
        cur  = next;

        trailSum = dvadd(trailSum, dvsub(dvuint_load(aAddT), dvuint_load(aSubT)));
        leadSum  = dvadd(leadSum, dvsub(dvuint_load(aAddL), dvuint_load(aSubL)));
    }

    return count;
}
