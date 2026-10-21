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
 * @file capon_roi_preprocess_dft2d.c
 * @brief The 2-D (elevation) coarse-DoA argmax.
 */

#include "capon_roi_preprocess_agens.h"
#include "capon_roi_preprocess_knl.h"
#include "capon_roi_preprocess_params.h"

#if !(defined(CRP_COARSE_DOA_FFT) && CRP_COARSE_DOA_FFT)
/* The 2-D (elevation) part of caponCoarseDoAFx, out of line so the azimuth-only code block stays compact. */
void caponCoarseDoA2DFx(CaponKnl *k, int32_t *peakAzOut, int32_t *peakElOut)
{
    const int32_t numElVirt = k->coarseElVirt;
    const int32_t beamAz = k->beamAz, beamEl = k->beamEl, shAz = k->shAz, shEl = k->shEl;
    int32_t *tmpRe = (int32_t *)k->coarseTmpReBuf;
    int32_t *tmpIm = (int32_t *)k->coarseTmpImBuf;
    int32_t peakAz = 0, peakEl = 0;
    const vshortx lane = vcollate_idx(replicateh(0));
    const vshortx e2   = (lane & 1) * 2;
    const vshortx up   = vmux(lane >= 8, replicateh(1), replicateh(0));
    vcharx idx         = chess_dont_care(vcharx);
    vdemote_i(e2 + up, e2 + 1 - up, idx); /* lanes 0-15: (0,1,2,3) x 4; lanes 16-31: (1,0,3,2) x 4 */
    const vshortx oddL  = lane & 1;
    const vshortx zero  = replicateh(0);
    const vshortx first = (lane & 3) < 2;

    /* El DFT + 2D (ka,ke) power argmax over a per-lane running max, tie-broken smallest-ke then
     * smallest-ka to match the reference. Keep the accumulators in registers and reduce once. */
    const int16_t *wel16 = k->wel16Buf;
    const int32_t vwi    = chess_elementsof(vintx); /* 8 */
    const int32_t nKa    = (beamAz + vwi - 1) / vwi;
    /* Lane index 0..vwi-1 straight from the ISA. A stack array is NOT a valid agen target --
     * agens address bank-typed VMEM, and the failure mode is silent garbage. */
    vintx laneId = vcollate_idx(replicatew(0));
    vintx bestP = replicatew(-1), bestKa = replicatew(0), bestKe = replicatew(0);
    const int32_t azLo = k->azLo, azHi = k->azHi, elLo = k->elLo, elHi = k->elHi;

    const int32_t npE   = (numElVirt + 1) >> 1;
    const int32_t tailE = numElVirt & 1;
    /* Per 8-bin ka block, once per detection: the ka index of each lane and the lanes that are real bins
     * inside the azimuth FoV (tmpRoiImBuf is free during coarse DoA). */
    int32_t *kaTab = (int32_t *)k->tmpRoiImBuf;             /* [nKa][8] ka index */
    int32_t *okTab = (int32_t *)k->tmpRoiImBuf + nKa * vwi; /* [nKa][8] -1 = candidate, 0 = not */
    {
        agen_A ks = initAgenA2_i(kaTab, nKa, vwi, 1, 0);
        agen_A os = initAgenA2_i(okTab, nKa, vwi, 1, 0);
        for (int32_t kc = 0; kc < nKa; kc++)
        {
            const vintx idx8  = laneId + kc * vwi;
            const vintx sh    = idx8 + replicatew(shAz);
            const vintx idxO  = vmux(sh < replicatew(beamAz), sh, sh - replicatew(beamAz));
            const vintx inFov = (idx8 < replicatew(beamAz)) & (idxO >= replicatew(azLo)) & (idxO <= replicatew(azHi));
            vstore(idx8, ks);
            vstore(vmux(inFov, replicatew(-1), replicatew(0)), os);
        }
        CRP_SEPARATOR();
    }
    for (int32_t ke = 0; ke < beamEl; ke++)
    {
        /* ke is a raw (unshifted) bin; the FoV window is expressed in output bins. */
        const int32_t keOut = (ke + shEl) % beamEl;
        if (keOut < elLo || keOut > elHi)
            continue;
        const vintx keV = replicatew(ke);
        AgenWrapper wce = {.size = sizeof(int16_t), .n1 = (uint16_t)npE, .n2 = (uint16_t)nKa, .s1 = 4, .s2 = 0};
        agen_A ce       = init_A((void *)(wel16 + (int32_t)(ke * numElVirt * 2)));
        INIT_AGEN2(ce, wce);
        /* e, e+1 (n1), pair (n2), 8-bin ka block (n3) */
        agen_A trAg = initAgenA3_i(tmpRe, 2, beamAz, npE, 2 * beamAz, nKa, vwi);
        agen_A tiAg = initAgenA3_i(tmpIm, 2, beamAz, npE, 2 * beamAz, nKa, vwi);
        agen_A kaAg = initAgenA2_i(kaTab, nKa, vwi, 1, 0);
        agen_A okAg = initAgenA2_i(okTab, nKa, vwi, 1, 0);
#    define CRP_EL_COEF(C)                                                 \
        do                                                                 \
        {                                                                  \
            (C)    = dvshort_load_perm(ce, idx);                           \
            (C).lo = vmux(oddL != 0, zero - (C).lo, (C).lo); /* -wi odd */ \
        } while (0)
#    define CRP_EL_UPDATE(ACCR, ACCI)                                                       \
        do                                                                                  \
        {                                                                                   \
            const vintx r_  = ((ACCR) + (1 << 15)) >> 16;                                   \
            const vintx i_  = ((ACCI) + (1 << 15)) >> 16;                                   \
            vintx p_        = vmulw(r_, r_, 0) + vmulw(i_, i_, 0); /* 48-bit exact power */ \
            const vintx ka_ = vint_load(kaAg);                                              \
            p_              = vmux(vint_load(okAg) != 0, p_, replicatew(-1));               \
            const vintx gt_ = p_ > bestP; /* strict: keeps first (ke,ka) */                 \
            bestKa          = vmux(gt_, ka_, bestKa);                                       \
            bestKe          = vmux(gt_, keV, bestKe);                                       \
            bestP           = vmax(p_, bestP);                                              \
        } while (0)
        if (npE == 1)
        {
            /* One element pair (numElVirt <= 2): the coefficients are fixed for this ke. */
            dvshortx c;
            CRP_EL_COEF(c);
            if (tailE)
            {
                c.lo = vmux(first, c.lo, zero);
                c.hi = vmux(first, c.hi, zero);
            }
            for (int32_t kc = 0; kc < nKa; kc++) chess_prepare_for_pipelining
            {
                dvintx d0, d1;
                d0.lo            = vint_load(trAg);
                d1.lo            = vint_load(trAg);
                d0.hi            = vint_load(tiAg);
                d1.hi            = vint_load(tiAg);
                const vintx accR = vdotp4_whw(d0, d1, c.lo, replicatew(0), 0);
                const vintx accI = vdotp4_whw(d0, d1, c.hi, replicatew(0), 0);
                CRP_EL_UPDATE(accR, accI);
            }
        }
        else
        {
            for (int32_t kc = 0; kc < nKa; kc++)
            {
                vintx accR = replicatew(0), accI = replicatew(0);
                for (int32_t ep = 0; ep < npE; ep++)
                {
                    dvshortx c;
                    CRP_EL_COEF(c);
                    if (tailE && ep == npE - 1)
                    {
                        /* element numElVirt-1 alone: zero the second element's coefficients */
                        c.lo = vmux(first, c.lo, zero);
                        c.hi = vmux(first, c.hi, zero);
                    }
                    dvintx d0, d1;
                    d0.lo = vint_load(trAg);
                    d1.lo = vint_load(trAg);
                    d0.hi = vint_load(tiAg);
                    d1.hi = vint_load(tiAg);
                    accR  = vdotp4_whw(d0, d1, c.lo, accR, -1);
                    accI  = vdotp4_whw(d0, d1, c.hi, accI, -1);
                }
                CRP_EL_UPDATE(accR, accI);
            }
        }
#    undef CRP_EL_COEF
#    undef CRP_EL_UPDATE
    }
    /* Horizontal argmax: vmaxrid_s's internal compare is full s48, so the winning LANE is picked
         * on the exact power (its returned scalar truncates to s32 and is unused). bestKa/bestKe are
         * dumped once and indexed by that lane, a single non-looped read the separator orders. */
    int32_t maxVal, maxLane;
    vmaxrid_s(bestP, maxVal, maxLane);
    (void)maxVal;
    int32_t *sc = (int32_t *)k->tmpRoiReBuf;
    agen_A aKa  = initAgenA2_i(sc + 0, 1, 0, 1, 0);
    agen_A aKe  = initAgenA2_i(sc + vwi, 1, 0, 1, 0);
    vstore(bestKa, aKa);
    vstore(bestKe, aKe);
    CRP_SEPARATOR(); /* order the vector stores before the scalar reads below */
    int32_t peakKa    = sc[maxLane];
    int32_t peakKeRaw = sc[vwi + maxLane];
    peakAz            = (peakKa + shAz) % beamAz;
    peakEl            = (peakKeRaw + shEl) % beamEl;
    *peakAzOut        = peakAz;
    *peakElOut        = peakEl;
}
#endif
