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

#include "capon_roi_preprocess_fft.h"

#include <fft_knl.h>
#include <fft_radix4.h>

/* Working buffers + engine state; defined in capon_roi_preprocess_top.c with every other buffer. */
EXTERN_VMEM(B, int32_t, crpFftPing, CRP_FFT_BUF_ELEMS);
EXTERN_VMEM(B, int32_t, crpFftPong, CRP_FFT_BUF_ELEMS);
EXTERN_VMEM(B, int32_t, crpFftTmp, CRP_FFT_BUF_ELEMS);
EXTERN_VMEM(A, FFTFactorization, crpFftFactorization);
EXTERN_VMEM(A, AgenCFG, crpFftCfgs, CRP_FFT_MAX_CFGS);

void caponInitFft64Batch(const int32_t *twiddleQ30)
{
    /* Built host-side by priv::FactorizeFFTSize and bound as a program parameter, like range_fft. */
    FFTFactorization *f = &crpFftFactorization;

    int32_t cfg_idx = 0;
    int32_t it      = 0;
    /* Twiddle-multiplied outputs carry an extra Q(TWIDDLE_QBITS-TRUNCATE_QBITS) scale after the
     * engine's internal >>16, so their store-side shift is 2 + that vs >>2 for the untwiddled X0. */
    it = func(fft_batched_radix4_init)((int32_t *)crpFftPing, (int32_t *)crpFftPong, (int32_t *)crpFftTmp,
                                       (int32_t *)twiddleQ30, TWIDDLE_QBITS - TRUNCATE_QBITS, crpFftCfgs, cfg_idx, f,
                                       it, CRP_FFT_PITCH, CRP_FFT_GROUPS, CRP_FFT_CHUNK, 2);
    /* need_transpose=false / o_lane_ofst=0: keep the natural bin-major, lane-minor layout. Each bin
     * row holds the CRP_FFT_GROUPS chunks side by side, CRP_FFT_CHUNK int32 apart. */
    it = func(fft_batched_digit_reverse_transpose_init)((int32_t *)crpFftTmp, (int32_t *)crpFftPong, crpFftCfgs,
                                                        cfg_idx, false, 0, f, it, CRP_FFT_PITCH, CRP_FFT_GROUPS,
                                                        CRP_FFT_CHUNK, CRP_FFT_CHUNK, -1);
}

static void caponCoarseDoAAzFft64Batch(void)
{
    FFTFactorization *f               = &crpFftFactorization;
    const int32_t totalPowerOf2Stages = f->groups[RADIX4_GROUP_IDX].numStages + f->groups[RADIX4_GROUP_IDX].isOdd;
    int32_t cfg_idx                   = 0;

    for (int32_t i = 0; i < totalPowerOf2Stages - 1; i++)
    {
        func(fft_batched_radix4)(crpFftCfgs, cfg_idx, CRP_FFT_N, CRP_FFT_GROUPS);
    }
    func(fft_batched_radix4_digit_reverse_transpose)(crpFftCfgs, cfg_idx, f, CRP_FFT_GROUPS);
}

void caponInitFftBatchAgensFx(CaponKnl *k)
{
    const int32_t vwi  = chess_elementsof(vintx);
    const int32_t nCpy = CRP_FFT_N / vwi;

    AgenWrapper wDbSt      = {.size = sizeof(int32_t), .n1 = CRP_FFT_N, .n2 = 1, .s1 = 1, .s2 = 0};
    k->fftDbStRe           = init_B((void *)crpFftPing);
    k->fftDbStRe.lane_ofst = CRP_FFT_SPEC_LANE_OFST;
    INIT_AGEN2(k->fftDbStRe, wDbSt);
    k->fftDbStIm           = init_C((void *)crpFftTmp);
    k->fftDbStIm.lane_ofst = CRP_FFT_SPEC_LANE_OFST;
    INIT_AGEN2(k->fftDbStIm, wDbSt);

    AgenWrapper wCpySrc = {.size = sizeof(int32_t),
                           .n1   = (uint16_t)nCpy,
                           .n2   = CRP_FFT_LANES,
                           .s1   = vwi,
                           .s2   = CRP_FFT_SPEC_STRIDE};
    k->fftCpySrcRe      = init_A((void *)crpFftPing);
    INIT_AGEN2(k->fftCpySrcRe, wCpySrc);
    k->fftCpySrcIm = init_B((void *)crpFftTmp);
    INIT_AGEN2(k->fftCpySrcIm, wCpySrc);

    AgenWrapper wCpyDst = {.size = sizeof(int32_t), .n1 = (uint16_t)nCpy, .n2 = CRP_FFT_LANES, .s1 = vwi, .s2 = 0};
    k->fftCpyDstRe      = init_C((void *)k->coarseTmpReBuf);
    INIT_AGEN2(k->fftCpyDstRe, wCpyDst);
    k->fftCpyDstIm = init_A((void *)k->coarseTmpImBuf);
    INIT_AGEN2(k->fftCpyDstIm, wCpyDst);
}

void caponCoarseDoAFftBatchPrepFx(CaponKnl *k, const float *elem, int32_t lane, int32_t groupSize)
{
    const int32_t n2 = k->nvc * 2;

    const int32_t chunk  = lane / CRP_FFT_LANES;
    const int32_t l      = lane - chunk * CRP_FFT_LANES;
    const int32_t nLanes = (lane == groupSize - 1) ? (CRP_FFT_LANES - l) : 1;

    const int32_t vw     = chess_elementsof(vfloatx);
    const int32_t nv     = n2 / vw;
    const int32_t rem    = n2 - nv * vw;
    const int32_t nvCeil = nv + (rem ? 1 : 0);

    AgenWrapper ws = {.size = sizeof(float), .n1 = (uint16_t)nvCeil, .n2 = 1, .s1 = vw, .s2 = 0};
    agen_A sa      = init_A((void *)elem);
    INIT_AGEN2(sa, ws);
    AgenWrapper wd = {.size = sizeof(float),
                      .n1   = (uint16_t)nLanes,
                      .n2   = (uint16_t)nvCeil,
                      .s1   = CRP_FFT_ELEM_STRIDE,
                      .s2   = vw};
    agen_A da      = init_A((void *)(k->fftBatchElem[chunk] + l * CRP_FFT_ELEM_STRIDE));
    INIT_AGEN2(da, wd);

    for (int32_t v = 0; v < nvCeil; v++) chess_prepare_for_pipelining
    {
        const int32_t pred = (v < nv) ? -1 : CRP_LOW_LANES(rem);
        const vfloatx x    = vfloat_load(sa);
        for (int32_t t = 0; t < nLanes; t++) vstore(x, da, pred);
    }
}

static void crpFftBatchQuantize(CaponKnl *k, int32_t chunk)
{
    const float scale = (float)(1 << CRP_COARSE_CONJ_QBITS);
    /* Conjugate + range scale. A whole vector here is all-real or all-imaginary, so this is one
     * sign per vector -- a register broadcast, with no VMEM scratch, agens or fill loop. */
    const vfloatx vPos = replicatef(scale);
    const vfloatx vNeg = replicatef(-scale);

    AgenWrapper wLd = {.size = sizeof(float), .n1 = CRP_FFT_N, .n2 = 1, .s1 = 2, .s2 = 0};
    agen_A ldRe     = init_A((void *)k->fftBatchElem[chunk]);
    ldRe.lane_ofst  = CRP_FFT_ELEM_LANE_OFST;
    INIT_AGEN2(ldRe, wLd);
    agen_A ldIm    = init_A((void *)(k->fftBatchElem[chunk] + 1));
    ldIm.lane_ofst = CRP_FFT_ELEM_LANE_OFST;
    INIT_AGEN2(ldIm, wLd);

    /* Store: this chunk's 16 contiguous int32 per bin (8 lanes x re,im) in its band of the bin row,
     * advancing one bin per call. */
    AgenWrapper wSt = {.size = sizeof(int32_t), .n1 = CRP_FFT_N, .n2 = 1, .s1 = CRP_FFT_PITCH, .s2 = 0};
    agen_B stA      = init_B((void *)(k->fftBatchIn + chunk * CRP_FFT_CHUNK));
    INIT_AGEN2(stA, wSt);

    for (int32_t i = 0; i < CRP_FFT_N; i++) chess_loop_range(CRP_FFT_N, )
    chess_prepare_for_pipelining
    {
        dvintx d;
        d.lo = vfp_vintx_round(vmulf(vfloat_load_transp(ldRe), vPos));
        d.hi = vfp_vintx_round(vmulf(vfloat_load_transp(ldIm), vNeg));
        vstore_i(d, stA); /* interleaves lo/hi -> lane0.re, lane0.im, lane1.re, ... */
    }
    CRP_SEPARATOR();
}

void caponCoarseDoAFftBatchRankFx(CaponKnl *k, int32_t groupSize, int32_t *roiAzStartOut, int32_t *roiElStartOut)
{
    /* Only the chunks holding real detections are quantized and ranked. The engine still runs over
     * all CRP_FFT_GROUPS bands; an unused band holds stale data whose spectrum is never read. */
    const int32_t nChunks = (groupSize + CRP_FFT_LANES - 1) / CRP_FFT_LANES;
    for (int32_t g = 0; g < nChunks; g++)
    {
        crpFftBatchQuantize(k, g); /* this chunk's 8 lanes of staged elem -> its band of the engine input */
    }
    caponCoarseDoAAzFft64Batch();

    /* caponCoarseDoARankAzFx ranks coarseTmpReBuf/ImBuf, so each lane's spectrum is copied there. */
    const int32_t vwi  = chess_elementsof(vintx);
    const int32_t nCpy = CRP_FFT_N / vwi; /* 64/8 = 8 whole vectors */

    AgenWrapper wDbLd = {.size = sizeof(int32_t), .n1 = CRP_FFT_N, .n2 = 1, .s1 = CRP_FFT_PITCH, .s2 = 0};

    for (int32_t g = 0; g < nChunks; g++)
    {
        /* De-batch this chunk's band into per-lane contiguous spectra -- the mirror of
         * crpFftBatchQuantize, inverted for the mirror reason, so the strided access lands on the
         * STORE into a buffer that is ours. Scratches crpFftPing/crpFftTmp, both dead once the
         * transform finishes, so it costs no VMEM. */
        agen_A ldA = init_A((void *)(k->fftBatchOut + g * CRP_FFT_CHUNK));
        INIT_AGEN2(ldA, wDbLd);
        agen_B stRe = CRP_AG_RELOAD(k, fftDbStRe); /* hoisted; see caponInitFftBatchAgensFx */
        agen_C stIm = CRP_AG_RELOAD(k, fftDbStIm);

        for (int32_t i = 0; i < CRP_FFT_N; i++) chess_loop_range(CRP_FFT_N, )
        chess_prepare_for_pipelining
        {
            dvintx d = dvint_load_di(ldA); /* .lo = 8 lanes' re, .hi = 8 lanes' im */
            vstore_transp(d.lo, stRe);
            vstore_transp(d.hi, stIm);
        }
        CRP_SEPARATOR();

        /* Reloaded per chunk: the source steps lanes, the destination rewinds. */
        agen_A cpySrcRe = CRP_AG_RELOAD(k, fftCpySrcRe);
        agen_B cpySrcIm = CRP_AG_RELOAD(k, fftCpySrcIm);
        agen_C cpyDstRe = CRP_AG_RELOAD(k, fftCpyDstRe);
        agen_A cpyDstIm = CRP_AG_RELOAD(k, fftCpyDstIm);

        const int32_t remain = groupSize - g * CRP_FFT_LANES;
        const int32_t nLanes = (remain < CRP_FFT_LANES) ? remain : CRP_FFT_LANES;
        for (int32_t lane = 0; lane < nLanes; lane++)
        {
            for (int32_t c = 0; c < nCpy; c++) chess_loop_range(CRP_FFT_N / CRP_FFT_LANES, )
            chess_prepare_for_pipelining
            {
                vstore(vint_load(cpySrcRe), cpyDstRe);
                vstore(vint_load(cpySrcIm), cpyDstIm);
            }
            CRP_SEPARATOR(); /* order the spectrum writes before RankAzFx's vector reads */
            const float *elemLane = k->fftBatchElem[g] + lane * CRP_FFT_ELEM_STRIDE;
            int32_t peakAz, peakEl;
            caponCoarseDoARankAzFx(k, elemLane, &peakAz, &peakEl);
            caponCenterRoi(k, peakAz, peakEl, &roiAzStartOut[g * CRP_FFT_LANES + lane],
                           &roiElStartOut[g * CRP_FFT_LANES + lane]);
        }
    }
}
