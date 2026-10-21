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
 * @file histogram_equalization_knl.h
 *
 * @brief Kernel config struct and entry points for the histogram equalization
 *        VPU kernels.
 */

#ifndef HISTOGRAM_EQUALIZATION_KNL_H
#define HISTOGRAM_EQUALIZATION_KNL_H

#include "histogram_equalization_param.h"

#include <cupva_device.h>

/**
 * Agen configurations and scalars shared by the kernel stages.
 *
 * Every agen is configured once, before the tile loops, and only its base
 * pointer is patched per tile (the RDF ping-pong rotates the tile buffer). A
 * config can be materialized into more than one live agen -- the drain does that
 * to get independent load and store cursors over one buffer.
 */
typedef struct HistogramEqualizationConfig
{
    /**
     * The tile walk. Both passes read a source tile with this shape and pass 2 writes a
     * destination tile with it, so one copy of the shape is stored and loaded into whichever
     * agen registers a pass needs. Each register then takes its own base at the point of use,
     * which is what CfgAgen Base is for, so a second stored copy would only duplicate the shape.
     */
    AgenCFG tileCfg;
    /** Per-column weight mask, pass 1: the same walk with a zero row stride. */
    AgenCFG histMaskCfg;
    /**
     * One bin per issue, HEQ_LINE_ELEMS apart: the drain's 32-way sweep over hist, and the
     * replicate's write across the 32 ways of lut32. Same shape, so one config, rebased.
     */
    AgenCFG binLineCfg;
    /** Dense 32-bit counts, one bin per issue. */
    AgenCFG accAgenCfg;
    /**
     * 16 bins per issue over the cumulative counts: the delta min-reduce, the LUT build, and the
     * scan's shifted read, which rebases it to -2^p each pass.
     */
    AgenCFG binVecCfg;
    /** Halfword lookup values, 16 bins per issue. */
    AgenCFG lutStoreAgenCfg;
    /** One halfword lookup value per issue; the replicated line uses binLineCfg. */
    AgenCFG replicateInCfg;
    /** Scan current-read and destination-write, each spanning all passes (see common.c). */
    AgenCFG scanCurCfg;
    AgenCFG scanOutCfg;

    /** 32-way halfword histogram scratch, superbank B. */
    int16_t *hist;
    /** Inclusive cumulative counts (ping half of the scan), superbank B. */
    int32_t *accum;
    /** Pong half of the scan; the shifted read rebases between the two each pass. */
    int32_t *accumAlt;
    /** 32-way replicated halfword lookup table, superbank C. */
    uint16_t *lutTable;

    /** vhist_simple_32h issues per full tile, and per short last tile row. */
    int32_t histIssues;
    int32_t histIssuesLast;
    /** vlookup_32hu issues per full tile, and per short last tile row. */
    int32_t applyIssues;
    int32_t applyIssuesLast;

    /** Active issue counts for the tile row being processed. */
    int32_t histNiter;
    int32_t applyNiter;

    /** Fixed-point LUT scale, (255 << 15) / (width * height - delta). */
    int32_t scale;
    /** Count of the lowest occupied intensity. */
    int32_t delta;
    /** Active image samples, width * height. */
    int32_t numSamples;
} HistogramEqualizationConfig;

/*
 * Contract for heq_bzero, enforced at compile time: the region it clears must be a whole number
 * of dvchar stores, or the trailing bytes would keep stale contents.
 */
/* Negative-array-size typedefs, not static_assert: not every supported device compiler has it. */
typedef char heq_knl_contract_1
    [(HEQ_TABLE_ELEMS * sizeof(int16_t) % 64 == 0)
         ? 1
         : -1]; /* CONTRACT: the histogram table must be a whole number of dvchar stores for heq_bzero */

/**
 * Vectorized zero fill of a VMEM region.
 *
 * @param mem  Region base. Must be 64-byte aligned (VMEM() arrays are).
 * @param size Region length in bytes. Must be a multiple of sizeof(dvchar) -- the contract
 *             above pins this for its call site.
 */
static inline void heq_bzero(void *mem, int32_t size)
{
    dvcharx vzero;
    vzero.lo = vzero.hi = replicateb(0);

    const int32_t vecw  = chess_elementsof(dvchar);
    const int32_t niter = size / vecw;

    agen out = init(mem);
    out.n1   = niter;
    out.mod1 = sizeof(dvchar);

    for (int32_t i = 0; i < niter; i++) chess_loop_range(1, 256)
    chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        vstore(vzero, out);
    }
}

void histogram_equalizationInitAgens(uint8_t *src, int16_t *mask, int16_t *hist, int32_t *accum, int32_t *accumAlt,
                                     uint16_t *lut, uint16_t *lut32, int32_t tw, int32_t th, int32_t thLast,
                                     int32_t linePitch, HistogramEqualizationConfig *cfg);

void histogram_equalization_gen_mask(int16_t *maskFull, int16_t *maskLast, int32_t tw, int32_t validWidth);

void histogram_equalization_knl_hist(HistogramEqualizationConfig *cfg, uint8_t *inTile, int16_t *mask);

void histogram_equalization_knl_drain(HistogramEqualizationConfig *cfg);

void histogram_equalization_knl_scan(HistogramEqualizationConfig *cfg);

int32_t histogram_equalization_knl_delta(HistogramEqualizationConfig *cfg);

void histogram_equalization_knl_lut(HistogramEqualizationConfig *cfg);

void histogram_equalization_knl_replicate(HistogramEqualizationConfig *cfg);

void histogram_equalization_knl_apply(HistogramEqualizationConfig *cfg, uint8_t *inTile, uint8_t *outTile);

#endif // HISTOGRAM_EQUALIZATION_KNL_H
