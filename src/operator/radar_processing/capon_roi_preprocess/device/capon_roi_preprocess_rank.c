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
 * @file capon_roi_preprocess_rank.c
 * @brief caponCoarseDoARankAzFx, the azimuth-only coarse-DoA ranking both coarse-DoA methods share.
 */

#include "capon_roi_preprocess_agens.h"
#include "capon_roi_preprocess_knl.h"
#include "capon_roi_preprocess_params.h"

#define CRP_RANK_KC 4

void caponCoarseDoARankAzFx(CaponKnl *k, const float *elem, int32_t *peakAzOut, int32_t *peakElOut)
{
    const int32_t numAzVirt = k->numAzVirt, beamAz = k->beamAz, shAz = k->shAz, shEl = k->shEl, beamEl = k->beamEl;

    const int32_t vwi  = chess_elementsof(vintx); /* 8 */
    const int32_t nKa  = (beamAz + vwi - 1) / vwi;
    const int32_t nKaF = beamAz / vwi;
    const int32_t kaR  = beamAz - nKaF * vwi;
    int32_t *pScr      = (int32_t *)k->tmpRoiReBuf; /* [beamAz] int32 power scratch (free here) */
    agen_A trA         = CRP_AG_RELOAD(k, rankTrAgen);
    agen_A tiA         = CRP_AG_RELOAD(k, rankTiAgen);
    agen_A pA          = CRP_AG_RELOAD(k, rankPAgen);
    for (int32_t kc = 0; kc < nKa; kc++)
    {
        vintx tr           = vint_load(trA);
        vintx ti           = vint_load(tiA);
        vintx p            = (vmulw(tr, tr, 0) + vmulw(ti, ti, 0)) >> 8; /* s48 power -> int32 (coarse) */
        const int32_t pred = (kc < nKaF) ? -1 : CRP_LOW_LANES(kaR);
        vstore(p, pA, pred);
    }
    CRP_SEPARATOR();
    const int32_t KC = CRP_RANK_KC;
    int32_t candKa[CRP_RANK_KC];
    bool candValid[CRP_RANK_KC]; /* false for a slot left empty when fewer than KC bins are in the FoV */

    const int32_t vwKc   = chess_elementsof(vintx);
    const int32_t nVecKc = (beamAz + vwKc - 1) / vwKc;
    const int32_t IMAX   = 0x7FFFFFFF;
    for (int32_t i = beamAz; i < nVecKc * vwKc; i++) pScr[i] = -1;

#if CRP_GENERAL_GEOMETRY
    /* Coarse-DoA FoV window: disable out-of-window bins before the top-KC selection, so both the
     * int16 ranking and the fp32 refine below only ever see candidates inside it. pScr is indexed
     * by raw ka; the window is in output-bin space. numElVirt==1 here, so elevation needs no mask. */
    if (k->azLo != 0 || k->azHi != beamAz - 1)
    {
        for (int32_t ka = 0; ka < beamAz; ka++)
        {
            const int32_t o = (ka + shAz) % beamAz;
            if (o < k->azLo || o > k->azHi)
                pScr[ka] = -1;
        }
    }
#endif

    const vintx vlane = vcollate_idx(replicatew(0)); /* lane index, 0..vwKc-1 */
    const vintx vimax = replicatew(IMAX);
    AgenWrapper wKc   = {.size = sizeof(int32_t), .n1 = (uint16_t)nVecKc, .n2 = 1, .s1 = vwKc, .s2 = 0};

    for (int32_t j = 0; j < KC; j++)
    {
        agen_A mxAg = init_A((void *)pScr);
        INIT_AGEN2(mxAg, wKc);
        vintx vbest = replicatew(-1);
        for (int32_t v = 0; v < nVecKc; v++) vbest = vmax(vbest, vint_load(mxAg));
        int32_t bestP, ignored;
        vmaxrid_s(vbest, bestP, ignored);

        agen_A ixAg = init_A((void *)pScr);
        INIT_AGEN2(ixAg, wKc);
        const vintx vbp = replicatew(bestP);
        vintx vka       = vimax;
        for (int32_t v = 0; v < nVecKc; v++)
        {
            const vintx pv  = vint_load(ixAg);
            const vintx idx = vlane + replicatew(v * vwKc);
            vka             = vmin(vka, vmux(pv == vbp, idx, vimax));
        }
        int32_t bestKa;
        vminrid_s(vka, bestKa, ignored);

        const int32_t mk = (bestP < 0) ? 0 : bestKa; /* in-bounds, and a no-op store when empty */
        candKa[j]        = mk;
        candValid[j]     = (bestP >= 0);
        pScr[mk]         = -1;
    }

    const int32_t vwf   = chess_elementsof(vfloatx); /* complex taps per dvfloatx_di load */
    const int32_t nCh   = numAzVirt / vwf;           /* full vwf-complex chunks */
    const int32_t aTail = nCh * vwf;
    AgenWrapper wdi     = {.size = sizeof(float), .n1 = (uint16_t)nCh, .n2 = 1, .s1 = 16, .s2 = 0};
    agen_A eAg          = init_A((void *)elem);
    INIT_AGEN2(eAg, wdi);
    agen_A w0 = init_A((void *)(k->waz + (int32_t)(candKa[0] * numAzVirt * 2)));
    INIT_AGEN2(w0, wdi);
    agen_A w1 = init_A((void *)(k->waz + (int32_t)(candKa[1] * numAzVirt * 2)));
    INIT_AGEN2(w1, wdi);
    agen_A w2 = init_A((void *)(k->waz + (int32_t)(candKa[2] * numAzVirt * 2)));
    INIT_AGEN2(w2, wdi);
    agen_A w3 = init_A((void *)(k->waz + (int32_t)(candKa[3] * numAzVirt * 2)));
    INIT_AGEN2(w3, wdi);
    const vfloatx zf = replicatef(0.0f);
    vfloatx r0 = zf, r1 = zf, r2 = zf, r3 = zf, i0 = zf, i1 = zf, i2 = zf, i3 = zf;
#define CRP_REF_ACC(R, I, E, W)                                                \
    do                                                                         \
    {                                                                          \
        (R) = vaddf((R), vaddf(vmulf((E).lo, (W).lo), vmulf((E).hi, (W).hi))); \
        (I) = vaddf((I), vsubf(vmulf((E).lo, (W).hi), vmulf((E).hi, (W).lo))); \
    } while (0)
    for (int32_t c = 0; c < nCh; c++) chess_prepare_for_pipelining
    {
        const dvfloatx eD = dvfloat_load_di(eAg); /* eD.lo = re, eD.hi = im (deinterleaved) */
        const dvfloatx d0 = dvfloat_load_di(w0), d1 = dvfloat_load_di(w1);
        const dvfloatx d2 = dvfloat_load_di(w2), d3 = dvfloat_load_di(w3);
        CRP_REF_ACC(r0, i0, eD, d0); /* er*wr + eim*wi, er*wi - eim*wr */
        CRP_REF_ACC(r1, i1, eD, d1);
        CRP_REF_ACC(r2, i2, eD, d2);
        CRP_REF_ACC(r3, i3, eD, d3);
    }
#if CRP_GENERAL_GEOMETRY
    /* Tail (numAzVirt % vwf taps). vfloat_load's predicate is a whole-vector bool, so this loads at FULL width --
     * in bounds for both elem and waz -- and vmux zeroes the spare lanes. */
    const int32_t nTailCpx = numAzVirt - aTail;
    if (nTailCpx > 0)
    {
        AgenWrapper wt = {.size = sizeof(float), .n1 = 1, .n2 = 1, .s1 = 16, .s2 = 0};
        agen_A eTl     = init_A((void *)(elem + (int32_t)aTail * 2));
        INIT_AGEN2(eTl, wt);
        const vintx keep = vcollate_idx(replicatew(0)) < replicatew(nTailCpx); /* lane < nTailCpx */
        dvfloatx eD      = dvfloat_load_di(eTl);
        eD.lo            = vmux(keep, eD.lo, zf);
        eD.hi            = vmux(keep, eD.hi, zf);
#    define CRP_REF_TAIL(R, I, J)                                                                     \
        do                                                                                            \
        {                                                                                             \
            agen_A wTl = init_A((void *)(k->waz + (int32_t)(candKa[J] * numAzVirt * 2) + aTail * 2)); \
            INIT_AGEN2(wTl, wt);                                                                      \
            dvfloatx wD = dvfloat_load_di(wTl);                                                       \
            wD.lo       = vmux(keep, wD.lo, zf);                                                      \
            wD.hi       = vmux(keep, wD.hi, zf);                                                      \
            CRP_REF_ACC(R, I, eD, wD);                                                                \
        } while (0)
        CRP_REF_TAIL(r0, i0, 0);
        CRP_REF_TAIL(r1, i1, 1);
        CRP_REF_TAIL(r2, i2, 2);
        CRP_REF_TAIL(r3, i3, 3);
#    undef CRP_REF_TAIL
    }
#else
    (void)aTail; /* numAzVirt = 64: no partial vector */
#endif
#undef CRP_REF_ACC
    /* Rows 0..3: re of candidates 0..3, rows 4..7: im. tmpRoiReBuf past the power scratch is free here. */
    float *hs  = (float *)k->tmpRoiReBuf + 2 * CRP_MAX_BEAM_AZ;
    agen_A hsW = initAgenA2_i((int32_t *)hs, 2 * KC, 17, 1, 0);
    vstore(r0, hsW);
    vstore(r1, hsW);
    vstore(r2, hsW);
    vstore(r3, hsW);
    vstore(i0, hsW);
    vstore(i1, hsW);
    vstore(i2, hsW);
    vstore(i3, hsW);
    CRP_SEPARATOR();
    AgenWrapper wtr = {.size = sizeof(float), .n1 = (uint16_t)vwf, .n2 = 1, .s1 = 1, .s2 = 0};
    agen_A hsT      = init_A((void *)hs);
    hsT.lane_ofst   = 1; /* lane l reads row l: pitch 16 * 1 + 1 = 17 */
    INIT_AGEN2(hsT, wtr);
    vfloatx sum = vfloat_load_transp(hsT);
    for (int32_t c = 1; c < vwf; c++) sum = vaddf(sum, vfloat_load_transp(hsT));
    const vfloatx pw = vmulf(sum, sum); /* lane j: re_j^2, lane j+4: im_j^2 */
    float pF[CRP_RANK_KC];
    pF[0]         = vselectlane(pw, 0) + vselectlane(pw, 4);
    pF[1]         = vselectlane(pw, 1) + vselectlane(pw, 5);
    pF[2]         = vselectlane(pw, 2) + vselectlane(pw, 6);
    pF[3]         = vselectlane(pw, 3) + vselectlane(pw, 7);
    float bestPF  = -1.0f;
    int32_t winKa = candKa[0];
    for (int32_t j = 0; j < KC; j++)
    {
        /* Branchless keep-the-max. Strict '>' keeps the earliest candidate slot on a tie, matching the reference;
         * the select form preserves that because both assignments share one predicate. */
        const bool better = candValid[j] && (pF[j] > bestPF); /* an empty slot never wins */
        bestPF            = better ? pF[j] : bestPF;
        winKa             = better ? candKa[j] : winKa;
    }
    *peakAzOut = (winKa + shAz) % beamAz;
    /* numElVirt == 1: every elevation row of the spectrum is the same, so take the 0-degree row, or the
     * nearest field-of-view bin when the window excludes it. */
    int32_t peakEl = shEl % beamEl;
#if CRP_GENERAL_GEOMETRY
    peakEl = (peakEl < k->elLo) ? k->elLo : ((peakEl > k->elHi) ? k->elHi : peakEl);
#endif
    *peakElOut = peakEl;
}
