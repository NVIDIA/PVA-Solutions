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
 * @file histogram_equalization_knl.c
 *
 * @brief The seven VPU kernel stages, in execution order:
 *
 *   knl_hist       count active samples into the 32-way halfword histogram
 *   knl_drain      fold the 32 ways of each bin into 32-bit counts, clear ways
 *   knl_scan       inclusive cumulative counts across the 256 bins
 *   knl_delta      delta = min{ accum[i] : accum[i] > 0 }
 *   knl_lut        lut[i] = (max(accum[i] - delta, 0) * scale) >> 15
 *   knl_replicate  broadcast each lut entry across its 32 lookup ways
 *   knl_apply      dst = lut32[src], 32 pixels per lookup
 *
 * knl_hist and knl_apply are the hot stages, each bound at one M0 issue per 32
 * pixels; the other five run over 256 bins once per frame.
 */

#include "histogram_equalization_knl.h"

/**
 * Count the tile's active samples into the 32-way halfword histogram.
 *
 * One vhist_simple_32h issue handles 32 pixels, one per sub-histogram, so the updates cannot
 * collide. The mask doubles as the update vector: a zero-weight lane is a no-op, so row
 * padding needs no predication.
 */
void histogram_equalization_knl_hist(HistogramEqualizationConfig *cfg, uint8_t *inTile, int16_t *maskRow)
{
    int16_t *hist = cfg->hist;
    int32_t niter = cfg->histNiter;

    agen_A in   = init_agen_A_from_cfg(cfg->tileCfg);
    agen_C mask = init_agen_C_from_cfg(cfg->histMaskCfg);

    in.a   = (void chess_storage(RAM_Ab) *)inTile;
    mask.a = (void chess_storage(RAM_Cb) *)maskRow;

    const int32_t quotient  = (niter / 8) * 8;
    const int32_t remainder = niter - quotient;

    /* Same split as the apply loop below, and for the same reason: chess_unroll_loop(8) needs a
     * trip count divisible by 8. Rounding up is additionally unsafe here, since the histogram
     * accumulates and an over-run would inject counts no later stage could subtract. */
    if (quotient > 0)
    {
        for (int32_t i = 0; i < quotient; i++) chess_loop_range(8, 900)
        chess_unroll_loop(8)
        chess_prepare_for_pipelining
        {
            dvshortx dvpix = vuchar_dvshortx_load(in);
            dvshortx dvwgt = dvshort_load(mask);

            vhist_simple_32h(hist, dvpix, dvwgt);
        }
    }

    /* The 0 to 7 iterations the unrolled loop could not take. */
    if (remainder > 0)
    {
        for (int32_t i = 0; i < remainder; i++) chess_loop_range(1, 7)
        chess_unroll_loop(2)
        chess_prepare_for_pipelining
        {
            dvshortx dvpix = vuchar_dvshortx_load(in);
            dvshortx dvwgt = dvshort_load(mask);

            vhist_simple_32h(hist, dvpix, dvwgt);
        }
    }
}

/**
 * Fold the 32 ways of every bin into the 32-bit cumulative-count array, then
 * clear the ways so the halfword bins cannot overflow.
 *
 * Called whenever the issue count since the last drain reaches
 * HEQ_DRAIN_ISSUES, and once more after the last tile.
 */
void histogram_equalization_knl_drain(HistogramEqualizationConfig *cfg)
{
    /* binLineCfg is stored based on hist; the replicate stage rebases the same shape to lut32. */
    agen_B in = init_agen_B_from_cfg(cfg->binLineCfg);

    agen_B accLd = init_agen_B_from_cfg(cfg->accAgenCfg);
    agen_B accSt = init_agen_B_from_cfg(cfg->accAgenCfg);

    for (int32_t i = 0; i < HEQ_NBINS; i++) chess_unroll_loop(8)
    chess_prepare_for_pipelining
    {
        // All 32 ways of one bin, contiguous in a single 64-byte line.
        dvshortx dvways = dvshort_load(in);

        const int32_t ways = vsumr_s(dvways.lo) + vsumr_s(dvways.hi);
        const int32_t prev = int_load(accLd);

        int_store(prev + ways, accSt);
    }

    heq_bzero(cfg->hist, HEQ_TABLE_ELEMS * (int32_t)sizeof(int16_t));
}

/**
 * Inclusive cumulative sum over the 256 bin counts.
 *
 * Hillis-Steele doubling, 8 passes. The shifted operand is taken at an address offset rather
 * than a lane permute, so each pass is one load pair plus one add. Both buffers carry
 * HEQ_SCAN_PAD zero words in front so src[i - k] always reads real zeros. The pass count is
 * even, so the result lands back in accum.
 */
void histogram_equalization_knl_scan(HistogramEqualizationConfig *cfg)
{
    const int32_t niter = HEQ_NBINS / (int32_t)chess_elementsof(dvint);

    agen_B aCur = init_agen_B_from_cfg(cfg->scanCurCfg);
    agen_B aOut = init_agen_B_from_cfg(cfg->scanOutCfg);

    int32_t *scanSrc = cfg->accum;

    for (int32_t p = 0; p < HEQ_SCAN_PASSES; p++)
    {
        /* Same shape every pass; only the -2^p base moves, so rebase the one stored config. */
        AgenCFG shfCfg = cfg->binVecCfg;
        cupvaModifyAgenCfgBase(&shfCfg, &scanSrc[HEQ_SCAN_PAD - (1 << p)]);

        agen_B aShf = init_agen_B_from_cfg(shfCfg);

        for (int32_t i = 0; i < niter; i++) chess_unroll_loop(2)
        chess_prepare_for_pipelining
        {
            dvintx dvcur = dvint_load(aCur);
            dvintx dvshf = dvint_load(aShf);

            vstore(dvadd(dvcur, dvshf), aOut);
        }

        scanSrc = (scanSrc == cfg->accum) ? cfg->accumAlt : cfg->accum;
    }
}

/**
 * delta = min{ accum[i] : accum[i] > 0 }.
 *
 * The reference takes the raw count of the lowest occupied intensity. Every bin below it is
 * empty, so that bin's cumulative value equals its raw count and is the smallest positive value
 * in the array -- which turns a first-nonzero search into a min-reduce.
 *
 * @return delta, or 0 when the histogram is empty.
 */
int32_t histogram_equalization_knl_delta(HistogramEqualizationConfig *cfg)
{
    agen_B in = init_agen_B_from_cfg(cfg->binVecCfg);

    const int32_t vecw  = chess_elementsof(dvint);
    const int32_t niter = HEQ_NBINS / vecw;

    // Empty bins must not win the min, so park them above every real count.
    dvintx dvbig;
    dvbig.lo = dvbig.hi = replicatew(0x7FFFFFFF);

    dvintx dvacc = dvbig;

    for (int32_t i = 0; i < niter; i++) chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        dvintx dvcum = dvint_load(in);

        dvintx dvcand = dvmux(dvCmpGT(dvcum, 0), dvcum, dvbig);

        dvacc = dvmin(dvacc, dvcand);
    }

    const int32_t lo    = vminr_s(dvacc.lo);
    const int32_t hi    = vminr_s(dvacc.hi);
    const int32_t delta = (lo < hi) ? lo : hi;

    return (delta == 0x7FFFFFFF) ? 0 : delta;
}

/**
 * lut[i] = (max(accum[i] - delta, 0) * scale) >> 15.
 *
 * Bins below the lowest occupied intensity must come out 0: the reference evaluates them while
 * delta is still zero. Re-deriving them with the final delta gives a negative numerator, which
 * the reference's uint16_t cast wraps into a large value, so the numerator is clamped at zero.
 * The product stays at or below 255 << 15, inside the 48-bit lane. VPU_TRUNC_15 is not
 * available for word multiplies, hence the explicit shift.
 */
void histogram_equalization_knl_lut(HistogramEqualizationConfig *cfg)
{
    agen_B in  = init_agen_B_from_cfg(cfg->binVecCfg);
    agen_C out = init_agen_C_from_cfg(cfg->lutStoreAgenCfg);

    const int32_t vecw  = chess_elementsof(dvint);
    const int32_t niter = HEQ_NBINS / vecw;

    const int32_t delta = cfg->delta;
    const int32_t scale = cfg->scale;

    dvintx dvscale;
    dvscale.lo = dvscale.hi = replicatew(scale);

    for (int32_t i = 0; i < niter; i++) chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        dvintx dvcum = dvint_load(in);

        dvintx dvnum = dvmax(dvsub(dvcum, delta), 0);
        dvintx dvval = dvmulw(dvnum, dvscale, VPU_TRUNC_0) >> HEQ_QBITS;

        vstore_wh(dvval, out);
    }
}

/**
 * Broadcast each lookup value across the 32 ways of the parallel lookup table.
 *
 * Lane i reads only sub-table i, so the logical table must be present in all 32. Entry b of way
 * i sits at byte 2*i + 64*b: one 64-byte line per bin, exactly one dvshortx store.
 */
void histogram_equalization_knl_replicate(HistogramEqualizationConfig *cfg)
{
    agen_C in = init_agen_C_from_cfg(cfg->replicateInCfg);

    /* Same shape as the drain's hist sweep, rebased onto the replicated table. */
    AgenCFG outCfg = cfg->binLineCfg;
    cupvaModifyAgenCfgBase(&outCfg, (void *)cfg->lutTable);
    agen_C out = init_agen_C_from_cfg(outCfg);

    for (int32_t i = 0; i < HEQ_NBINS; i++) chess_unroll_loop(4)
    chess_prepare_for_pipelining
    {
        const int32_t val = ushort_load(in);

        dvshortx dvrep;
        dvrep.lo = dvrep.hi = replicateh(val);

        vstore(dvrep, out);
    }
}

/**
 * Equalize the tile: dst = lut32[src], 32 pixels per lookup issue.
 *
 * No weight mask and no predication: the output RasterDataFlow transfers only
 * the columns and rows inside the image, so a group running past the valid
 * width writes tile-buffer bytes the DMA never reads.
 */
void histogram_equalization_knl_apply(HistogramEqualizationConfig *cfg, uint8_t *inTile, uint8_t *outTile)
{
    int32_t niter = cfg->applyNiter;

    /* One stored shape, two registers; each takes its own base here rather than in VMEM. */
    agen_A in  = init_agen_A_from_cfg(cfg->tileCfg);
    agen_B out = init_agen_B_from_cfg(cfg->tileCfg);

    in.a  = (void chess_storage(RAM_Ab) *)inTile;
    out.a = (void chess_storage(RAM_Bb) *)outTile;

    /* The storage qualifier is what selects the bank-C overload of vlookup_32hu. */
    const uint16_t chess_storage(RAM_Cb) *lut = (const uint16_t chess_storage(RAM_Cb) *)cfg->lutTable;

    const int32_t quotient  = (niter / 8) * 8;
    const int32_t remainder = niter - quotient;

    /* chess_unroll_loop(8) requires a trip count divisible by 8, hence the split: a single
     * loop pipelines but cannot unroll. Rounding the count up instead would read and write up
     * to 448 bytes into the ping-pong twin, since a full tile is exactly HEQ_TILE_BYTES.
     */
    if (quotient > 0)
    {
        for (int32_t i = 0; i < quotient; i++) chess_loop_range(8, 900)
        chess_unroll_loop(8)
        chess_prepare_for_pipelining
        {
            dvshortx dvpix = vuchar_dvshortx_load(in);

            dvshortx dveq = dvmin(vlookup_32hu(lut, dvpix), 255);

            vstore_hb(dveq, out);
        }
    }

    /* The 0 to 7 iterations the unrolled loop could not take. */
    if (remainder > 0)
    {
        for (int32_t i = 0; i < remainder; i++) chess_loop_range(1, 7)
        chess_unroll_loop(2)
        chess_prepare_for_pipelining
        {
            dvshortx dvpix = vuchar_dvshortx_load(in);

            dvshortx dveq = dvmin(vlookup_32hu(lut, dvpix), 255);

            vstore_hb(dveq, out);
        }
    }
}
