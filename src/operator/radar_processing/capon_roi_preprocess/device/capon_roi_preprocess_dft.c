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

#include "capon_roi_preprocess_agens.h"
#include "capon_roi_preprocess_knl.h"

/* Agen builders (size in bytes, strides in elements), file-static so each TU gets its own copy. */

static inline agen_B initAgenB3_s(int16_t *base, int32_t n1, int32_t s1, int32_t n2, int32_t s2, int32_t n3, int32_t s3)
{
    AgenWrapper w = {.size = sizeof(int16_t),
                     .n1   = (uint16_t)n1,
                     .n2   = (uint16_t)n2,
                     .n3   = (uint16_t)n3,
                     .s1   = s1,
                     .s2   = s2,
                     .s3   = s3};
    agen_B a      = init_B((void *)base);
    INIT_AGEN3(a, w);
    return a;
}

void caponCoarseDoAFx(CaponKnl *k, const float *elem, int32_t *roiAzStart, int32_t *roiElStart)
{
    const int32_t nvc = k->nvc, numAzVirt = k->numAzVirt, numElVirt = k->coarseElVirt;
    const int32_t beamAz = k->beamAz, beamEl = k->beamEl, shAz = k->shAz, shEl = k->shEl;
    int16_t *conjX16 = k->conjXBuf;

    /* conj(elem) -> int16 at a FIXED Q13, which fills int16 for a unit peak and does not clip until
     * |elem| ~= 4. Callers MUST feed a bounded snapshot -- a raw large-magnitude input would clip
     * and flip the ROI. */
    const float scale = (float)(1 << CRP_COARSE_CONJ_QBITS);
    const int32_t vwf = chess_elementsof(vfloatx); /* 8 */
    const int32_t vws = chess_elementsof(vshortx); /* 16 */
    const int32_t n2  = nvc * 2;                   /* interleaved re/im floats */
    const int32_t nv  = n2 / vws;                  /* full 16-wide (8-complex) blocks */
    /* n2 may not be a multiple of vws: predicate the last store to its valid lanes, no scalar tail. */
    const int32_t rem    = n2 - nv * vws;      /* leftover interleaved floats (0..vws-1) */
    const int32_t nvCeil = nv + (rem ? 1 : 0); /* ceil; the last block's store is predicated */

    /* Alternating conj+scale vector [+s,-s,...]; the interleaved layout keeps the parity aligned
     * across both vfloatx halves of a block. Built in coarseTmpReBuf, free until the az DFT. */
    float *sc = (float *)k->coarseTmpReBuf;
    for (int32_t j = 0; j < vwf; j++) sc[j] = (j & 1) ? -scale : scale;
    CRP_SEPARATOR(); /* order the scalar pattern writes before the vector load below */
    AgenWrapper ws = {.size = sizeof(float), .n1 = 1, .n2 = 1, .s1 = 0, .s2 = 0};
    agen sa        = init((void *)sc);
    INIT_AGEN2(sa, ws);
    const vfloatx vscale = vfloat_load(sa);

    AgenWrapper we = {.size = sizeof(float), .n1 = (uint16_t)(2 * nvCeil), .n2 = 1, .s1 = vwf, .s2 = 0};
    agen ea        = init((void *)elem);
    INIT_AGEN2(ea, we);
    AgenWrapper wc = {.size = sizeof(int16_t), .n1 = (uint16_t)nvCeil, .n2 = 1, .s1 = vws, .s2 = 0};
    agen ca        = init((void *)conjX16);
    INIT_AGEN2(ca, wc);

    for (int32_t v = 0; v < nvCeil; v++) chess_prepare_for_pipelining
    {
        vintx iv0 = vfp_vintx_round(vmulf(vfloat_load(ea), vscale));
        vintx iv1 = vfp_vintx_round(vmulf(vfloat_load(ea), vscale));
        dvintx d;
        d.lo = iv0;
        d.hi = iv1;
        /* Predicate the last (partial) block to its valid int16 lanes; -1 stores all vws lanes. */
        const int32_t pred = (v < nv) ? -1 : CRP_LOW_LANES(rem);
        vstore(vdemote(d), ca, pred); /* 16 int16, saturating narrow */
    }

    const int32_t vw      = chess_elementsof(vshortx);
    const int32_t nKaFull = beamAz / vw;               /* full vw-wide ka vectors */
    const int32_t kaRem   = beamAz - nKaFull * vw;     /* remaining ka bins in the partial vector (0..vw-1) */
    const int32_t nKaVec  = nKaFull + (kaRem ? 1 : 0); /* ceil; the last vector's store is predicated */
    int32_t *tmpRe        = (int32_t *)k->coarseTmpReBuf;
    int32_t *tmpIm        = (int32_t *)k->coarseTmpImBuf;

    const vshortx lane = vcollate_idx(replicateh(0));
    const vshortx e2   = (lane & 1) * 2;
    const vshortx up   = vmux(lane >= 8, replicateh(1), replicateh(0));
    vcharx idx         = chess_dont_care(vcharx);
    vdemote_i(e2 + up, e2 + 1 - up, idx); /* lanes 0-15: (0,1,2,3) x 4; lanes 16-31: (1,0,3,2) x 4 */
    const vshortx oddL  = lane & 1;
    const vshortx zero  = replicateh(0);
    const vshortx first = (lane & 3) < 2; /* the first tap's two coefficients of each group */
    const int32_t np    = numAzVirt >> 1; /* whole tap pairs */
    const int32_t tail  = numAzVirt & 1;
    const int32_t row2  = 2 * beamAz; /* interleaved table row (one tap) in int16 */
    agen_A rAgen        = initAgenA2_i(tmpRe, nKaVec, vw, numElVirt, beamAz);
    agen_A iAgen        = initAgenA2_i(tmpIm, nKaVec, vw, numElVirt, beamAz);
#define CRP_COEF(C, CX)                                                         \
    do                                                                          \
    {                                                                           \
        (C)    = dvshort_load_perm(CX, idx);                                    \
        (C).lo = vmux(oddL != 0, zero - (C).lo, (C).lo); /* -xi in odd lanes */ \
    } while (0)
#define CRP_DP4(C, TW)                                               \
    do                                                               \
    {                                                                \
        dvshortx d0 = dvshort_load_di(TW), d1 = dvshort_load_di(TW); \
        vdotp4x2_hhw(d0, d1, (C), aR0, aI0, -1, aR0, aI0);           \
        d0 = dvshort_load_di(TW), d1 = dvshort_load_di(TW);          \
        vdotp4x2_hhw(d0, d1, (C), aR1, aI1, -1, aR1, aI1);           \
        d0 = dvshort_load_di(TW), d1 = dvshort_load_di(TW);          \
        vdotp4x2_hhw(d0, d1, (C), aR2, aI2, -1, aR2, aI2);           \
        d0 = dvshort_load_di(TW), d1 = dvshort_load_di(TW);          \
        vdotp4x2_hhw(d0, d1, (C), aR3, aI3, -1, aR3, aI3);           \
    } while (0)
#define CRP_DFT_OUT(AR, AI, J)                                                     \
    do                                                                             \
    {                                                                              \
        if (kb + (J) < nKaVec)                                                     \
        {                                                                          \
            dvintx r_, i_;                                                         \
            r_.lo              = ((AR).lo + (1 << 15)) >> 16;                      \
            r_.hi              = ((AR).hi + (1 << 15)) >> 16;                      \
            i_.lo              = ((AI).lo + (1 << 15)) >> 16;                      \
            i_.hi              = ((AI).hi + (1 << 15)) >> 16;                      \
            const int32_t pred = (kb + (J) < nKaFull) ? -1 : CRP_LOW_LANES(kaRem); \
            vstore_i(r_, rAgen, pred);                                             \
            vstore_i(i_, iAgen, pred);                                             \
        }                                                                          \
    } while (0)
    for (int32_t e = 0; e < numElVirt; e++)
    {
        const int16_t *xcol = conjX16 + (int32_t)(e * numAzVirt * 2);
        for (int32_t kb = 0; kb < nKaVec; kb += 4)
        {
            /* Use named accumulators, not a dvintx array. */
            dvintx aR0, aR1, aR2, aR3, aI0, aI1, aI2, aI3;
            aR0.lo = replicatew(0), aR0.hi = replicatew(0), aI0 = aR0;
            aR1 = aR0, aR2 = aR0, aR3 = aR0, aI1 = aR0, aI2 = aR0, aI3 = aR0;
            /* Coefficient pairs: 4 int16 per pair. Twiddles: tap a, a+1 (n1), 16-beam block (n2), pair (n3). */
            AgenWrapper wcx = {.size = sizeof(int16_t), .n1 = (uint16_t)(np + tail), .n2 = 1, .s1 = 4, .s2 = 0};
            agen_A cx       = init_A((void *)xcol);
            INIT_AGEN2(cx, wcx);
            agen_B tw = initAgenB3_s(k->wazTBuf + kb * 2 * vw, 2, row2, 4, 2 * vw, np + tail, 2 * row2);
            for (int32_t p = 0; p < np; p++) chess_prepare_for_pipelining
            {
                dvshortx c;
                CRP_COEF(c, cx);
                CRP_DP4(c, tw);
            }
            if (tail)
            {
                /* Tap numAzVirt-1 alone: zero the second tap's coefficients (its data is read past the row). */
                dvshortx c;
                CRP_COEF(c, cx);
                c.lo = vmux(first, c.lo, zero);
                c.hi = vmux(first, c.hi, zero);
                CRP_DP4(c, tw);
            }
            CRP_DFT_OUT(aR0, aI0, 0);
            CRP_DFT_OUT(aR1, aI1, 1);
            CRP_DFT_OUT(aR2, aI2, 2);
            CRP_DFT_OUT(aR3, aI3, 3);
        }
    }
#undef CRP_COEF
#undef CRP_DP4
#undef CRP_DFT_OUT
    CRP_SEPARATOR(); /* order the az DFT's vector stores of tmpRe/tmpIm before the argmax reads */

    int32_t peakAz = 0, peakEl = 0;
    if (numElVirt == 1)
    {
        caponCoarseDoARankAzFx(k, elem, &peakAz, &peakEl);
    }
    else
    {
        caponCoarseDoA2DFx(k, &peakAz, &peakEl); /* capon_roi_preprocess_dft2d.c */
    }

    caponCenterRoi(k, peakAz, peakEl, roiAzStart, roiElStart);
}
