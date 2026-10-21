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

#include "image_histogram_knl_u8.h"

#include "image_histogram_param.h"

void generate_mask_u8(uint8_t *mask, int tw, int mw)
{
    bzero_opt(mask, tw * sizeof(uint8_t));

    int niter = mw;

    agen out = init(mask);
    out.n1   = niter;
    out.mod1 = sizeof(uint8_t);

    for (int i = 0; i < niter; i++)
    {
        uchar_store(0x1, out);
    }

    mem_fence();
}

/**
 *  Init agens for each kernels
 */
void image_histogram_agen_init_u8(uint8_t *src, int tw, int th, int th_last, int lp, int start, int end, int nbins,
                                  int16_t *hist, uint32_t *dst, int32_t *bucketAcc, uint8_t *mask,
                                  ImageHistogramConfig *cfg)
{
    AgenWrapper wrapper;

    int niter2      = th;
    int niter2_last = th_last;

    cfg->start = start;
    cfg->end   = end;

    const int a = (1 << QBITS) * nbins / (end - start);
    cfg->a      = a;
    cfg->b      = -start * a;

    // init agen for exec kernel

    int vecw_exec   = chess_elementsof(dvshort);
    int niter1_exec = tw / vecw_exec;

    agen agen_exec_src = init(src);
    wrapper.size       = sizeof(uint8_t);
    wrapper.n1         = niter1_exec;
    wrapper.n2         = niter2;
    wrapper.s1         = vecw_exec;
    wrapper.s2         = lp;
    INIT_AGEN2(agen_exec_src, wrapper);

    cfg->execAgenCfgs[0] = extract_agen_cfg(agen_exec_src);

    agen agen_exec_mask = init(mask);
    wrapper.size        = sizeof(uint8_t);
    wrapper.n1          = niter1_exec;
    wrapper.n2          = niter2;
    wrapper.s1          = vecw_exec;
    wrapper.s2          = 0; // the column mask repeats every row
    INIT_AGEN2(agen_exec_mask, wrapper);

    cfg->execAgenCfgs[1] = extract_agen_cfg(agen_exec_mask);

    cfg->niter_exec      = niter1_exec * niter2;
    cfg->niter_exec_last = niter1_exec * niter2_last;

    cfg->hist      = hist;
    cfg->dst       = dst;
    cfg->bucketAcc = bucketAcc;

    // init agen for drain kernel

    agen agen_drain_hist = init(hist);
    wrapper.size         = sizeof(int16_t);
    wrapper.n1           = HIST_U8_BUCKETS;
    wrapper.s1           = HIST_U8_WAYS;
    INIT_AGEN1(agen_drain_hist, wrapper);

    cfg->reduceAgenCfgs[0] = extract_agen_cfg(agen_drain_hist);

    cfg->nbins = nbins;
}

/*
 * Total each bucket's per-way counters and reset them.
 *
 * An int16 counter cannot hold a whole image, so this runs every HIST_U8_DRAIN_TILES tiles and at
 * the end.
 */
void image_histogram_knl_fold_u8(ImageHistogramConfig *cfg)
{
    int32_t *acc = cfg->bucketAcc;

    agen in = init_agen_from_cfg(cfg->reduceAgenCfgs[0]);

    for (int v = 0; v < HIST_U8_BUCKETS; v++) chess_unroll_loop(8)
    chess_prepare_for_pipelining
    {
        dvshortx sub = dvshort_load(in);

        /* Each half reduces on its own: two int16 counters can exceed an int16 together. */
        vintx sum = vsumr(sub.lo) + vsumr(sub.hi);

        vhist_simple_1w(acc, replicatew(v), sum);
    }

    bzero_opt(cfg->hist, HIST_U8_BUCKETS * HIST_U8_WAYS * (int)sizeof(int16_t));
}

/* One issue: 32 samples counted at the weight the caller supplies. */
static inline void execStepU8(int16_t *hist, agen &in, agen &mask, int masked, dvshortx dvone)
{
    dvshortx dv0 = vuchar_dvshortx_load(in);

    vhist_simple_32h(hist, dv0, masked ? vuchar_dvshortx_load(mask) : dvone);
}

/*
 * Count every sample at weight one.
 *
 * The fold reads only the buckets in [start, end), so an out-of-range sample lands in a bucket
 * nothing looks at and needs no weight of its own. The last tile column does, to drop the columns
 * past the image edge.
 */
static inline void execRunU8(ImageHistogramConfig *cfg, int masked)
{
    int16_t *hist = (int16_t *)cfg->hist;
    int niter     = cfg->execNiter;

    agen in   = init_agen_from_cfg(cfg->execAgenCfgs[0]);
    agen mask = init_agen_from_cfg(cfg->execAgenCfgs[1]);

    dvshortx dvone;
    dvone.lo = replicateh(1);
    dvone.hi = replicateh(1);

    if (niter >= 24)
    {
        int quotient  = (niter / 8) * 8;
        int remainder = niter % 8;

        for (int i = 0; i < quotient; i++) chess_loop_range(24, )
        chess_unroll_loop(8)
        chess_prepare_for_pipelining
        {
            execStepU8(hist, in, mask, masked, dvone);
        }

        mem_fence();

        for (int i = 0; i < remainder; i++) chess_prepare_for_pipelining
        {
            execStepU8(hist, in, mask, masked, dvone);
        }
    }
    else
    {
        for (int i = 0; i < niter; i++) chess_prepare_for_pipelining
        {
            execStepU8(hist, in, mask, masked, dvone);
        }
    }
}

void image_histogram_knl_exec_u8(ImageHistogramConfig *cfg)
{
    execRunU8(cfg, 0);
}

void image_histogram_knl_exec_masked_u8(ImageHistogramConfig *cfg)
{
    execRunU8(cfg, 1);
}

/*
 * Project the bucket totals onto the output bins. Runs once for the frame, which is what keeps the
 * drain's cost independent of numBins.
 */
void image_histogram_knl_emit_u8(ImageHistogramConfig *cfg)
{
    int32_t const *acc = cfg->bucketAcc;
    uint32_t *dst      = cfg->dst;

    for (int v = cfg->start; v < cfg->end; v++)
    {
        dst[(v * cfg->a + cfg->b) >> QBITS] += (uint32_t)acc[v];
    }
}
