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

#include "image_histogram_knl_u16.h"

/*
 * The way count picks both the histogram instruction and the counter layout. Sub-16-way issues
 * take the low half of the vector; the rest of the index vector is ignored by the instruction.
 */
#if HIST_WAYS == 16
#    define HIST_ISSUE(h, idx, wt) vhist_simple_16w((h), (idx), (wt))
#elif HIST_WAYS == 8
#    define HIST_ISSUE(h, idx, wt) vhist_simple_8w((h), (idx).lo, (wt).lo)
#elif HIST_WAYS == 4
#    define HIST_ISSUE(h, idx, wt) vhist_simple_4w((h), (idx).lo, (wt).lo)
#elif HIST_WAYS == 2
#    define HIST_ISSUE(h, idx, wt) vhist_simple_2w((h), (idx).lo, (wt).lo)
#else
#    error "HIST_WAYS must be 16, 8, 4 or 2"
#endif

void generate_mask_u16(uint16_t *mask, int tw, int mw)
{
    bzero_opt(mask, tw * sizeof(uint16_t));

    int niter = mw;

    agen out = init(mask);
    out.n1   = niter;
    out.mod1 = sizeof(uint16_t);

    for (int i = 0; i < niter; i++)
    {
        ushort_store(0x1, out);
    }

    mem_fence();
}

/**
 *  Init agens for each kernels
 */
void image_histogram_agen_init_u16(uint16_t *src, int tw, int th, int th_last, int lp, int start, int end, int nbins,
                                   int a, int qbits, uint16_t *valid, int32_t *hist, uint32_t *dst, uint16_t *mask,
                                   int ways, ImageHistogramConfig *cfg)
{
    AgenWrapper wrapper;

    // init agen for valid kernel

    int vecw_valid   = chess_elementsof(dvshort);
    int niter1_valid = tw / vecw_valid;
    int niter2       = th;
    int niter2_last  = th_last;

    agen agen_valid_src = init(src);
    wrapper.size        = sizeof(uint16_t);
    wrapper.n1          = niter1_valid;
    wrapper.n2          = niter2;
    wrapper.s1          = vecw_valid;
    wrapper.s2          = lp;
    INIT_AGEN2(agen_valid_src, wrapper);

    cfg->validAgenCfgs[0] = extract_agen_cfg(agen_valid_src);

    agen agen_valid_valid = init(valid);
    wrapper.size          = sizeof(uint16_t);
    wrapper.n1            = niter1_valid;
    wrapper.n2            = niter2;
    wrapper.s1            = vecw_valid;
    wrapper.s2            = lp;
    INIT_AGEN2(agen_valid_valid, wrapper);

    cfg->validAgenCfgs[1] = extract_agen_cfg(agen_valid_valid);

    cfg->niter_valid      = niter1_valid * niter2;
    cfg->niter_valid_last = niter1_valid * niter2_last;

    cfg->start = start;
    cfg->end   = end;

    // init agen for mask kernel

    if (cfg->maskWidth)
    {
        int vecw_mask   = chess_elementsof(dvshort);
        int niter1_mask = tw / vecw_mask;
        int niter2      = th;
        int niter2_last = th_last;

        agen agen_mask_src = init(valid);
        wrapper.size       = sizeof(uint16_t);
        wrapper.n1         = niter1_mask;
        wrapper.n2         = niter2;
        wrapper.s1         = vecw_mask;
        wrapper.s2         = lp;
        INIT_AGEN2(agen_mask_src, wrapper);

        cfg->maskAgenCfgs[0] = extract_agen_cfg(agen_mask_src);

        agen agen_mask_mask = init(mask);
        wrapper.size        = sizeof(uint16_t);
        wrapper.n1          = niter1_mask;
        wrapper.n2          = niter2;
        wrapper.s1          = vecw_mask;
        wrapper.s2          = 0;
        INIT_AGEN2(agen_mask_mask, wrapper);

        cfg->maskAgenCfgs[1] = extract_agen_cfg(agen_mask_mask);

        agen agen_mask_dst = init(valid);
        wrapper.size       = sizeof(uint16_t);
        wrapper.n1         = niter1_mask;
        wrapper.n2         = niter2;
        wrapper.s1         = vecw_mask;
        wrapper.s2         = lp;
        INIT_AGEN2(agen_mask_dst, wrapper);

        cfg->maskAgenCfgs[2] = extract_agen_cfg(agen_mask_dst);

        cfg->niter_mask      = niter1_mask * niter2;
        cfg->niter_mask_last = niter1_mask * niter2_last;
    }

    // init agen for index kernel

    int vecw_index   = chess_elementsof(dvint);
    int niter1_index = tw / vecw_index;

    agen agen_index_in = init(src);
    wrapper.size       = sizeof(uint16_t);
    wrapper.n1         = niter1_index;
    wrapper.n2         = niter2;
    wrapper.s1         = vecw_index;
    wrapper.s2         = lp;
    INIT_AGEN2(agen_index_in, wrapper);

    cfg->indexAgenCfgs[0] = extract_agen_cfg(agen_index_in);

    agen agen_index_out = init(src);
    wrapper.size        = sizeof(uint16_t);
    wrapper.n1          = niter1_index;
    wrapper.n2          = niter2;
    wrapper.s1          = vecw_index;
    wrapper.s2          = lp;
    INIT_AGEN2(agen_index_out, wrapper);
    agen_index_out.round = 0x80 | qbits; // truncate

    cfg->indexAgenCfgs[1] = extract_agen_cfg(agen_index_out);

    cfg->niter_index      = niter1_index * niter2;
    cfg->niter_index_last = niter1_index * niter2_last;

    /*
     * f(x) = (x - start) * a >> qbits, the shift riding the index agen's round field below.
     *
     * Subtracting start rather than folding -start * a into an addend is what frees the shift: the
     * addend had to fit int32, which capped it at 16 and left the scale quantized on wide ranges.
     */
    cfg->a = a;

    // init agen for exec kernel

    int vecw_exec   = ways;
    int niter1_exec = tw / vecw_exec;

    agen agen_exec_src = init(src);
    wrapper.size       = sizeof(uint16_t);
    wrapper.n1         = niter1_exec;
    wrapper.n2         = niter2;
    wrapper.s1         = vecw_exec;
    wrapper.s2         = lp;
    INIT_AGEN2(agen_exec_src, wrapper);

    cfg->execAgenCfgs[0] = extract_agen_cfg(agen_exec_src);

    agen agen_exec_valid = init(valid);
    wrapper.size         = sizeof(uint16_t);
    wrapper.n1           = niter1_exec;
    wrapper.n2           = niter2;
    wrapper.s1           = vecw_exec;
    wrapper.s2           = lp;
    INIT_AGEN2(agen_exec_valid, wrapper);

    cfg->execAgenCfgs[1] = extract_agen_cfg(agen_exec_valid);

    cfg->niter_exec      = niter1_exec * niter2;
    cfg->niter_exec_last = niter1_exec * niter2_last;

    cfg->hist = hist;

    // init agen for reduce kernel

    int vecw_reduce   = chess_elementsof(dvint);
    int nbins_per_vec = chess_elementsof(dvint) / ways;

    agen agen_reduce_hist = init(hist);
    wrapper.size          = sizeof(int32_t);
    wrapper.n1            = (nbins + nbins_per_vec - 1) / nbins_per_vec;
    wrapper.s1            = vecw_reduce;
    INIT_AGEN1(agen_reduce_hist, wrapper);

    cfg->reduceAgenCfgs[0] = extract_agen_cfg(agen_reduce_hist);

    agen agen_reduce_dst = init(dst);
    wrapper.size         = sizeof(uint32_t);
    if (ways == 2)
    {
        wrapper.n1 = (nbins + chess_elementsof(vint) - 1) / chess_elementsof(vint);
        wrapper.s1 = chess_elementsof(vint);
    }
    else
    {
        wrapper.n1 = nbins;
        wrapper.s1 = 1;
    }
    INIT_AGEN1(agen_reduce_dst, wrapper);

    cfg->reduceAgenCfgs[1] = extract_agen_cfg(agen_reduce_dst);

    cfg->nbins = nbins;
}

/**
 *  Check pixel value in range of [start, end)
 */
void image_histogram_knl_valid_u16(ImageHistogramConfig *cfg)
{
    int niter = cfg->validNiter;
    int start = cfg->start;
    int end   = cfg->end;

    agen in    = init_agen_from_cfg(cfg->validAgenCfgs[0]);
    agen valid = init_agen_from_cfg(cfg->validAgenCfgs[1]);

    dvshortx dvend;
    dvend.lo = replicateh(end);
    dvend.hi = replicateh(end);

    niter = max(niter, 24);
    niter = (niter + 7) / 8 * 8;

    for (int i = 0; i < niter; i++) chess_loop_range(24, )
    chess_unroll_loop(8)
    chess_prepare_for_pipelining
    {
        dvshortx dv0 = dvushort_load(in);

        dvshortx dv1 = dvCmpLT_andL(dv0, dvend, (dv0 >= start));

        vstore(dv1, valid);
    }
}

/**
 *  Mask off invalid pixel value
 */
void image_histogram_knl_mask_u16(ImageHistogramConfig *cfg)
{
    int niter = cfg->maskNiter;

    agen in   = init_agen_from_cfg(cfg->maskAgenCfgs[0]);
    agen mask = init_agen_from_cfg(cfg->maskAgenCfgs[1]);
    agen out  = init_agen_from_cfg(cfg->maskAgenCfgs[2]);

    niter = max(niter, 24);
    niter = (niter + 7) / 8 * 8;

    for (int i = 0; i < niter; i++) chess_loop_range(24, )
    chess_unroll_loop(8)
    chess_prepare_for_pipelining
    {
        dvshortx dv0 = dvushort_load(in);
        dvshortx dv1 = dvushort_load(mask);

        dv0 = dv0 && dv1;

        vstore(dv0, out);
    }
}

/**
 *  Calculate index of each pixel value for histogram operation
 */
void image_histogram_knl_index_u16(ImageHistogramConfig *cfg)
{
    int niter = cfg->indexNiter;
    int a     = cfg->a;
    int start = cfg->start;

    agen in  = init_agen_from_cfg(cfg->indexAgenCfgs[0]);
    agen out = init_agen_from_cfg(cfg->indexAgenCfgs[1]);

    dvintx dva;
    dva.lo = replicatew(a);
    dva.hi = replicatew(a);

    niter = max(niter, 24);
    niter = (niter + 7) / 8 * 8;

    for (int i = 0; i < niter; i++) chess_loop_range(24, )
    chess_unroll_loop(8)
    chess_prepare_for_pipelining
    {
        dvintx dv0 = vushort_dvintx_load(in);

        dv0 = dvmulwhw(dva, dv0 - start);

        vstore_wh(dv0, out);
    }
}

/**
 *  Count number of each pixel value by HIST_WAYS-way histogram operation
 */
void image_histogram_knl_exec_u16(ImageHistogramConfig *cfg)
{
    int *hist = (int *)cfg->hist;
    int niter = cfg->execNiter;

    agen in    = init_agen_from_cfg(cfg->execAgenCfgs[0]);
    agen valid = init_agen_from_cfg(cfg->execAgenCfgs[1]);

    dvintx dvzero;
    dvzero.lo = replicatew(0);
    dvzero.hi = replicatew(0);

    if (niter >= 24)
    {
        int quotient  = (niter / 8) * 8;
        int remainder = niter % 8;

        for (int i = 0; i < quotient; i++) chess_loop_range(24, )
        chess_unroll_loop(8)
        chess_prepare_for_pipelining
        {
            dvintx dv0 = vushort_dvintx_load(in);
            dvintx dv1 = vushort_dvintx_load(valid);

            dv0 = dvmux(dv1, dv0, dvzero);

            HIST_ISSUE(hist, dv0, dv1);
        }

        mem_fence();

        for (int i = 0; i < remainder; i++) chess_prepare_for_pipelining
        {
            dvintx dv0 = vushort_dvintx_load(in);
            dvintx dv1 = vushort_dvintx_load(valid);

            dv0 = dvmux(dv1, dv0, dvzero);

            HIST_ISSUE(hist, dv0, dv1);
        }
    }
    else
    {
        for (int i = 0; i < niter; i++) chess_prepare_for_pipelining
        {
            dvintx dv0 = vushort_dvintx_load(in);
            dvintx dv1 = vushort_dvintx_load(valid);

            dv0 = dvmux(dv1, dv0, dvzero);

            HIST_ISSUE(hist, dv0, dv1);
        }
    }
}

/**
 *  Count total number of each pixel value after HIST_WAYS-way histogram operation
 *
 *  A loaded vector holds 16 counters, so it spans 16 / HIST_WAYS bins and the way groups must be
 *  separated before they are summed. At 2 ways each lane pair is already a whole bin, so the
 *  vector reduces with one add and stores as a vector.
 */
void image_histogram_knl_reduce_u16(ImageHistogramConfig *cfg)
{
    int niter = cfg->nbins;

    agen in  = init_agen_from_cfg(cfg->reduceAgenCfgs[0]);
    agen out = init_agen_from_cfg(cfg->reduceAgenCfgs[1]);

#if HIST_WAYS == 4
    vintx voff   = replicatew(0);
    vintx von    = replicatew(-1);
    vintx vseq   = vcollate_idx(replicatew(0));
    vintx vmask0 = vmux(vseq & 1, voff, von);
    vintx vmask1 = vmux(vseq & 1, von, voff);
#endif

    niter = max(niter, 24);
    niter = (niter + 7) / 8 * 8;

    for (int i = 0; i < niter; i++) chess_loop_range(24, )
    chess_unroll_loop(8)
    chess_prepare_for_pipelining
    {
        /*
         * At 8 and 4 ways a vector spans several bins with their ways interleaved across the
         * lanes, so the load de-interleaves to bring one bin's counters together. At 16 ways the
         * vector is a single bin, and at 2 ways lo and hi pair up lane-wise; neither needs it.
         */
#if HIST_WAYS == 8 || HIST_WAYS == 4
        dvintx dv0 = dvuint_load_di(in);
#else
        dvintx dv0 = dvuint_load(in);
#endif

#if HIST_WAYS == 16
        int_store(vsumr_s(dv0.lo) + vsumr_s(dv0.hi), out);
#elif HIST_WAYS == 8
        int_store(vsumr_s(dv0.lo), out);
        int_store(vsumr_s(dv0.hi), out);
#elif HIST_WAYS == 4
        int_store(vsumr_s(dv0.lo & vmask0), out);
        int_store(vsumr_s(dv0.lo & vmask1), out);
        int_store(vsumr_s(dv0.hi & vmask0), out);
        int_store(vsumr_s(dv0.hi & vmask1), out);
#elif HIST_WAYS == 2
        vstore(dv0.lo + dv0.hi, out);
#endif
    }
}
