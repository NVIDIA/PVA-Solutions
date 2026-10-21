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
 * @file capon_roi_preprocess_elproj.c
 * @brief Elevation projection of the dot-product path (numElVirt == 2), in the general direct-DFT kernel.
 */

#include "capon_roi_preprocess_agens.h"
#include "capon_roi_preprocess_knl.h"
#include "capon_roi_preprocess_params.h"

#if CRP_PROJ_GENERAL

void caponProjectElDotP4Fx(CaponKnl *k, int32_t roiAzStart, int32_t roiElStart)
{
    (void)roiAzStart;
    const int32_t roiAz = k->roiAz, roiEl = k->roiEl, H = k->H;
    const int32_t beamEl = k->beamEl, shEl = k->shEl;
    const int32_t fbaSym = k->fbaSym, numSel = k->numSel;
    const int32_t nC     = H / CRP_VECW;
    const int32_t nCf    = fbaSym ? (numSel + CRP_VECW - 1) / CRP_VECW : nC;
    const int32_t Hb     = (fbaSym ? 2 : 1) * nCf * CRP_VECW; /* tmpRoi row pitch (crpProjectAzDotP4) */
    const int32_t shift  = (k->vdQBits - 2) - k->inputQBits;
    const int32_t rshift = (shift > 0) ? shift : 0;
    const int32_t lshift = (shift < 0) ? -shift : 0;
    const int32_t rnd    = (shift > 0) ? (1 << (shift - 1)) : 0;
    int32_t *tRe = (int32_t *)k->tmpRoiReBuf, *tIm = (int32_t *)k->tmpRoiImBuf;
    int32_t *pRe = (int32_t *)k->patchReBuf, *pIm = (int32_t *)k->patchImBuf;

    /* Permute indices: forward (0,1,2,3) x 4 | (1,0,3,2) x 4, reversed elements (2,3,0,1) x 4 | (3,2,1,0) x 4.
     * vdemote_i interleaves its two sources, a the even lanes and b the odd ones. */
    const vshortx lane = vcollate_idx(replicateh(0));
    const vshortx up   = vmux(lane >= 8, replicateh(1), replicateh(0));
    const vshortx eF   = (lane & 1) * 2;
    const vshortx eR   = replicateh(2) - eF;
    vcharx idxF = chess_dont_care(vcharx), idxR = chess_dont_care(vcharx);
    vdemote_i(eF + up, eF + 1 - up, idxF);
    vdemote_i(eR + up, eR + 1 - up, idxR);
    const vshortx oddL = lane & 1;
    const vshortx zero = replicateh(0);
    CRP_SEPARATOR(); /* order the azimuth pass's tmpRoi stores before the loads below */

    if (nCf < nC)
    {
        /* Rows past the forward and backward blocks are zero; the backward stores below overwrite the rest. */
        const int32_t nz = nC - nCf;
        agen_A zr        = initAgenA2_i(pRe + nCf * CRP_VECW, nz, CRP_VECW, roiEl * roiAz, H);
        agen_B zi        = initAgenB2_i(pIm + nCf * CRP_VECW, nz, CRP_VECW, roiEl * roiAz, H);
        const vintx z    = replicatew(0);
        for (int32_t i = 0; i < nz * roiEl * roiAz; i++) chess_prepare_for_pipelining
        chess_loop_range(1, )
        {
            vstore(z, zr);
            vstore(z, zi);
        }
    }

    const int32_t nIt      = roiAz * nCf;                       /* (Doppler block, beam) per elevation slot */
    const int32_t left     = H - numSel - (nCf - 1) * CRP_VECW; /* rows of the last backward block below H */
    const int32_t predLast = (left >= CRP_VECW) ? -1 : CRP_LOW_LANES(left);
    for (int32_t eLoc = 0; eLoc < roiEl; eLoc++)
    {
        int32_t ke = (roiElStart + eLoc - shEl) % beamEl;
        if (ke < 0)
            ke += beamEl;
        AgenWrapper wce = {.size = sizeof(int16_t), .n1 = 1, .n2 = 1, .s1 = 0, .s2 = 0};
        agen_A ce       = init_A((void *)(k->wel16Buf + ke * 4));
        INIT_AGEN2(ce, wce);
        dvshortx cf = dvshort_load_perm(ce, idxF);
        cf.lo       = vmux(oddL != 0, zero - cf.lo, cf.lo); /* -wi in odd lanes (17-bit in the WRF) */
        int32_t *oR = pRe + eLoc * roiAz * H, *oI = pIm + eLoc * roiAz * H;
        for (int32_t half = 0; half < (fbaSym ? 2 : 1); half++)
        {
            if (half)
            {
                agen_A cr = init_A((void *)(k->wel16Buf + ke * 4));
                INIT_AGEN2(cr, wce);
                cf    = dvshort_load_perm(cr, idxR);
                cf.lo = vmux(oddL != 0, zero - cf.lo, cf.lo);
            }
            const int32_t src = half * nCf * CRP_VECW; /* backward blocks follow the forward ones in tmpRoi */
            const int32_t dst = half ? numSel : 0;
            /* elements 0 and 1: block c (n1), beam (n2) */
            agen_A a0r = initAgenA2_i(tRe + src, nCf, CRP_VECW, roiAz, Hb);
            agen_A a1r = initAgenA2_i(tRe + roiAz * Hb + src, nCf, CRP_VECW, roiAz, Hb);
            agen_A a0i = initAgenA2_i(tIm + src, nCf, CRP_VECW, roiAz, Hb);
            agen_A a1i = initAgenA2_i(tIm + roiAz * Hb + src, nCf, CRP_VECW, roiAz, Hb);
            agen_A sr  = initAgenA2_i(oR + dst, nCf, CRP_VECW, roiAz, H);
            agen_B si  = initAgenB2_i(oI + dst, nCf, CRP_VECW, roiAz, H);
            int32_t c  = 0;
            for (int32_t i = 0; i < nIt; i++) chess_prepare_for_pipelining
            chess_loop_range(1, )
            {
                dvintx d0, d1;
                d0.lo              = vint_load(a0r);
                d0.hi              = vint_load(a0i);
                d1.lo              = vint_load(a1r);
                d1.hi              = vint_load(a1i);
                const vintx accR   = vdotp4_whw(d0, d1, cf.lo, replicatew(0), 0);
                const vintx accI   = vdotp4_whw(d0, d1, cf.hi, replicatew(0), 0);
                const vintx t_r    = (accR + (1 << 15)) >> 16;
                const vintx t_i    = (accI + (1 << 15)) >> 16;
                const int32_t pred = (half && c == nCf - 1) ? predLast : -1; /* backward rows clipped at H */
                vstore(((t_r + rnd) >> rshift) << lshift, sr, pred);
                vstore(((t_i + rnd) >> rshift) << lshift, si, pred);
                c = (c == nCf - 1) ? 0 : c + 1;
            }
        }
    }
}

#endif
