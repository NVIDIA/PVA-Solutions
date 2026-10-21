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

/**
 * @file histogram_equalization_common.c
 *
 * @brief One-time agen configuration and mask generation. Called once from
 *        CUPVA_VPU_MAIN before the tile loops; the loop bodies only rebase.
 */

#include "histogram_equalization_knl.h"

/**
 * Configure every agen once. Only the base pointer moves per tile.
 *
 * The three tile walks (pass-1 source, pass-2 source, pass-2 destination) have identical
 * shape, so one INIT_AGEN2 result is built and stored once. Each use loads it into an agen
 * register and sets that register's base, so the two cursors pass 2 needs cost one stored
 * config, not two.
 *
 * @param src       shared input tile buffer, both passes (superbank A)
 * @param mask      per-column weight mask (superbank C)
 * @param hist      32-way halfword histogram scratch (superbank B)
 * @param accum     cumulative counts, scan ping buffer (superbank B)
 * @param accumAlt  scan pong buffer (superbank B)
 * @param lut       dense halfword lookup values (superbank C)
 * @param lut32     32-way replicated lookup table (superbank C)
 * @param tw        tile width in pixels; a multiple of HEQ_WAYS
 * @param th        rows in a full tile
 * @param thLast    rows in the last (possibly short) tile row
 * @param linePitch tile buffer row pitch in pixels
 */
void histogram_equalizationInitAgens(uint8_t *src, int16_t *mask, int16_t *hist, int32_t *accum, int32_t *accumAlt,
                                     uint16_t *lut, uint16_t *lut32, int32_t tw, int32_t th, int32_t thLast,
                                     int32_t linePitch, HistogramEqualizationConfig *cfg)
{
    AgenWrapper wrapper;

    /*
     * Both hot stages consume HEQ_WAYS pixels per issue. vuchar_dvshortx_load
     * widens 32 unsigned bytes into the 32 halfword lanes of a dvshortx, so a
     * pixel agen advances 32 *bytes* per issue even though the register is 96
     * bytes wide -- a widening load consumes the source width.
     */
    const int32_t groupsPerRow = tw / HEQ_WAYS;

    // The shared tile walk: one group of 32 pixels per issue, row by row.
    wrapper.size = sizeof(uint8_t);
    wrapper.n1   = groupsPerRow;
    wrapper.s1   = HEQ_WAYS;
    wrapper.n2   = th;
    wrapper.s2   = linePitch;

    agen agen_tile = init(src);
    INIT_AGEN2(agen_tile, wrapper);

    const AgenCFG tileCfg = extract_agen_cfg(agen_tile);

    cfg->tileCfg = tileCfg;

    /*
     * Pass 1, weight mask: the same column pattern on every row, so the outer
     * stride is 0 and one tw-long mask row is re-read for free. The mask is the
     * histogram's update vector, which makes padding columns add zero -- the
     * branch-free way to keep row padding out of the counts.
     */
    wrapper.size = sizeof(int16_t);
    wrapper.n1   = groupsPerRow;
    wrapper.s1   = HEQ_WAYS;
    wrapper.n2   = th;
    wrapper.s2   = 0;

    agen agen_hist_mask = init(mask);
    INIT_AGEN2(agen_hist_mask, wrapper);
    cfg->histMaskCfg = extract_agen_cfg(agen_hist_mask);

    cfg->histIssues     = groupsPerRow * th;
    cfg->histIssuesLast = groupsPerRow * thLast;

    cfg->applyIssues     = groupsPerRow * th;
    cfg->applyIssuesLast = groupsPerRow * thLast;

    /*
     * Drain: bin b of way i lives at halfword HEQ_LINE_ELEMS*b + i, so one
     * dvshort_load at that stride reads all 32 ways of exactly one bin.
     */
    wrapper.size = sizeof(int16_t);
    wrapper.n1   = HEQ_NBINS;
    wrapper.s1   = HEQ_LINE_ELEMS;

    /*
     * Scan: HEQ_SCAN_PASSES doubling passes ping-ponging between accum and accumAlt.
     *
     * An agen address is affine in its loop counters -- constant stride per dimension -- so the
     * current-read and the destination-write fold into one agen each spanning every pass:
     * dimension 1 walks the 16 vectors of a pass, dimension 2 is the two-buffer ping-pong, and
     * dimension 3 repeats that pair HEQ_SCAN_PASSES / 2 times. The shifted read cannot fold: its
     * offset is -2^p, which doubles per pass, and no constant stride generates that.
     */
    const int32_t scanVecs  = HEQ_NBINS / (int32_t)chess_elementsof(dvint);
    const int32_t scanDelta = (int32_t)(accumAlt - accum);

    wrapper.size = sizeof(int32_t);
    wrapper.n1   = scanVecs;
    wrapper.s1   = (int32_t)chess_elementsof(dvint);
    wrapper.n2   = 2;
    wrapper.s2   = scanDelta;
    wrapper.n3   = HEQ_SCAN_PASSES / 2;
    wrapper.s3   = 0;

    agen agen_scan_cur = init(&accum[HEQ_SCAN_PAD]);
    INIT_AGEN3(agen_scan_cur, wrapper);
    cfg->scanCurCfg = extract_agen_cfg(agen_scan_cur);

    wrapper.s2 = -scanDelta;

    agen agen_scan_out = init(&accumAlt[HEQ_SCAN_PAD]);
    INIT_AGEN3(agen_scan_out, wrapper);
    cfg->scanOutCfg = extract_agen_cfg(agen_scan_out);

    wrapper.n1 = scanVecs;
    wrapper.s1 = (int32_t)chess_elementsof(dvint);

    wrapper.size = sizeof(int16_t);
    wrapper.n1   = HEQ_NBINS;
    wrapper.s1   = HEQ_LINE_ELEMS;

    agen agen_drain_hist = init(hist);
    INIT_AGEN1(agen_drain_hist, wrapper);
    cfg->binLineCfg = extract_agen_cfg(agen_drain_hist);

    /*
     * The drain's 32-bit accumulators. This is a separate config from binVecCfg
     * even though both address accum[]: the drain reads and writes one bin per
     * issue, so it needs two independent cursors advancing in lockstep, whereas
     * the bin sweeps walk 16 bins per issue.
     */
    wrapper.size = sizeof(int32_t);
    wrapper.n1   = HEQ_NBINS;
    wrapper.s1   = 1;

    agen agen_drain_acc = init(&accum[HEQ_SCAN_PAD]);
    INIT_AGEN1(agen_drain_acc, wrapper);
    cfg->accAgenCfg = extract_agen_cfg(agen_drain_acc);

    /*
     * Bin sweeps over the cumulative counts. A dvintx covers 16 words, so 16
     * issues span the 256 bins. The delta min-reduce and the LUT build walk the
     * same buffer the same way, so they share one configuration.
     */
    const int32_t binVecs = HEQ_NBINS / chess_elementsof(dvint);

    wrapper.size = sizeof(int32_t);
    wrapper.n1   = binVecs;
    wrapper.s1   = chess_elementsof(dvint);

    agen agen_accum = init(&accum[HEQ_SCAN_PAD]);
    INIT_AGEN1(agen_accum, wrapper);
    cfg->binVecCfg = extract_agen_cfg(agen_accum);

    /*
     * LUT store: vstore_wh narrows the 16 word lanes to 16 halfwords, 32 bytes
     * per issue. The values are provably inside [0, 255] (see knl_lut), but the
     * store path is armed with unsigned saturation to [0, 255] anyway, so the
     * narrowing store cannot wrap even if a later change widens the numerator.
     * Saturation rides the store path and costs no issue slot.
     */
    wrapper.size = sizeof(uint16_t);
    wrapper.n1   = binVecs;
    wrapper.s1   = chess_elementsof(dvint);

    agen agen_lut_out       = init(lut);
    agen_lut_out.sat_opt    = 3; // enabled, unsigned
    agen_lut_out.sat_lim_lo = 0;
    agen_lut_out.sat_val_lo = 0;
    agen_lut_out.sat_lim_hi = 255;
    agen_lut_out.sat_val_hi = 255;
    INIT_AGEN1(agen_lut_out, wrapper);
    cfg->lutStoreAgenCfg = extract_agen_cfg(agen_lut_out);

    // Replicate: read one dense lut entry, write its 64-byte lookup line.
    wrapper.size = sizeof(uint16_t);
    wrapper.n1   = HEQ_NBINS;
    wrapper.s1   = 1;

    agen agen_rep_in = init(lut);
    INIT_AGEN1(agen_rep_in, wrapper);
    cfg->replicateInCfg = extract_agen_cfg(agen_rep_in);

    cfg->hist     = hist;
    cfg->accum    = accum;
    cfg->accumAlt = accumAlt;
    cfg->lutTable = lut32;
}

/**
 * Build the two per-column weight masks.
 *
 * maskFull is all ones, for every tile column fully inside the image. maskLast
 * has ones for the first validWidth columns and zeros beyond, for the rightmost
 * tile column when the width is not a whole number of tiles. Past validWidth
 * the tile buffer still holds whatever the previous tile left there, so a zero
 * weight is what keeps those samples out of the counts.
 */
void histogram_equalization_gen_mask(int16_t *maskFull, int16_t *maskLast, int32_t tw, int32_t validWidth)
{

    dvshortx vone;
    vone.lo = vone.hi = replicateh(1);

    const int32_t vecw  = chess_elementsof(dvshort);
    const int32_t niter = tw / vecw;

    agen_C outFull = init_C(maskFull);
    outFull.n1     = niter;
    outFull.mod1   = sizeof(dvshort);

    agen_C outLast = init_C(maskLast);
    outLast.n1     = niter;
    outLast.mod1   = sizeof(dvshort);

    for (int32_t i = 0; i < niter; i++) chess_loop_range(1, 60)
    chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        /*
         * Lane j of group i covers column i*32 + j and is active while that
         * column is inside the image. The predicate is a lane bitmask, so the
         * partially valid group falls out of the same store -- no separate tail.
         */
        const int32_t remaining = validWidth - i * vecw;
        // Unsigned shift: remaining can be 31 (e.g. width 255 with tw 224), and
        // 1 << 31 on a signed int is undefined behavior.
        const int32_t pred =
            (remaining >= vecw) ? ~0 : ((remaining > 0) ? (int32_t)((1u << (unsigned)remaining) - 1u) : 0);

        vstore(vone, outFull);
        vstore(vone, outLast, pred);
    }
}
