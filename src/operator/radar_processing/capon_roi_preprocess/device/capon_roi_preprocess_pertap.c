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
 * @file capon_roi_preprocess_pertap.c
 * @brief The per-tap MAC projection paths of the general kernel: the azimuth projection (caponProjectAzPerTapFx)
 *        for geometries the dot-product path does not take (more than two elevation rows, odd numAzVirt, custom
 *        tables with FBA mirror rows off an 8-row boundary) and the elevation projection (caponProjectElPerTapFx).
 */

#include "capon_roi_preprocess_agens.h"
#include "capon_roi_preprocess_knl.h"
#include "capon_roi_preprocess_params.h"

#if CRP_PROJ_GENERAL
#    define CRP_CMAC_D(ACC_R, ACC_I, XR_AG, XI_AG, W)            \
        do                                                       \
        {                                                        \
            const dvintx xr_  = dvint_load(XR_AG);               \
            const dvintx xi_  = dvint_load(XI_AG);               \
            const int32_t wr_ = (W)[0], wi_ = (W)[1];            \
            ACC_R = dvmaddwhw(xr_, wr_, ACC_R, VPU_ROUND_16, 1); \
            ACC_R = dvmsubwhw(xi_, wi_, ACC_R, VPU_ROUND_16, 1); \
            ACC_I = dvmaddwhw(xr_, wi_, ACC_I, VPU_ROUND_16, 1); \
            ACC_I = dvmaddwhw(xi_, wr_, ACC_I, VPU_ROUND_16, 1); \
        } while (0)
#    define CRP_CMAC_V(ACC_R, ACC_I, XR_AG, XI_AG, W)           \
        do                                                      \
        {                                                       \
            const vintx xr_   = vint_load(XR_AG);               \
            const vintx xi_   = vint_load(XI_AG);               \
            const int32_t wr_ = (W)[0], wi_ = (W)[1];           \
            ACC_R = vmaddwhw(xr_, wr_, ACC_R, VPU_ROUND_16, 1); \
            ACC_R = vmsubwhw(xi_, wi_, ACC_R, VPU_ROUND_16, 1); \
            ACC_I = vmaddwhw(xr_, wi_, ACC_I, VPU_ROUND_16, 1); \
            ACC_I = vmaddwhw(xi_, wr_, ACC_I, VPU_ROUND_16, 1); \
        } while (0)

void caponProjectAzPerTapFx(CaponKnl *k, int32_t roiAzStart)
{
    const int32_t roiAz = k->roiAz, numElVirt = k->numElVirt, numAzVirt = k->numAzVirt;
    const int32_t beamAz = k->beamAz, shAz = k->shAz, H = k->H;
    const int32_t DVECW = chess_elementsof(dvintx); /* 16 */

    if (numElVirt == 1)
    {
        /* Azimuth-only: the elevation projection's identity narrow is fused in here, keeping accR/accI
         * in registers. Keep it fused: do not hand off through tmpRoi. */
        const int32_t roiEl  = k->roiEl;
        const int32_t shift  = (k->vdQBits - 1) - k->inputQBits;
        const int32_t rshift = (shift > 0) ? shift : 0;
        const int32_t lshift = (shift < 0) ? -shift : 0;
        const int32_t rnd    = (shift > 0) ? (1 << (shift - 1)) : 0;

        if (H % DVECW == 0)
        {
            const int32_t nHd = H / DVECW;
            agen_C xr_agen    = initAgenC4_i(k->vdFxReBuf, numAzVirt, H, nHd, DVECW, 1, numAzVirt * H, roiAz, 0);
            agen_C xi_agen    = initAgenC4_i(k->vdFxImBuf, numAzVirt, H, nHd, DVECW, 1, numAzVirt * H, roiAz, 0);
            agen_A prAg       = CRP_AG_REBUILD(k, prAgD);
            agen_B piAg       = CRP_AG_REBUILD(k, piAgD);
            for (int32_t aLoc = 0; aLoc < roiAz; aLoc++)
            {
                int32_t ka = (roiAzStart + aLoc - shAz) % beamAz;
                if (ka < 0)
                    ka += beamAz;
                const int16_t *wrow = k->waz16Buf + (int32_t)(ka * numAzVirt * 2);
                for (int32_t hc = 0; hc < nHd; hc++)
                {
                    dvintx accR, accI;
                    accR.lo = replicatew(0), accR.hi = replicatew(0);
                    accI.lo = replicatew(0), accI.hi = replicatew(0);
                    for (int32_t a = 0; a < numAzVirt; a++)
                    {
                        CRP_CMAC_D(accR, accI, xr_agen, xi_agen, wrow + a * 2);
                    }
                    const dvintx rr = ((accR + rnd) >> rshift) << lshift; /* identity el narrow */
                    const dvintx ri = ((accI + rnd) >> rshift) << lshift;
                    for (int32_t eLoc = 0; eLoc < roiEl; eLoc++) chess_prepare_for_pipelining
                    chess_loop_range(1, )
                    {
                        vstore(rr, prAg); /* replicate to patch[eLoc*roiAz+aLoc][h] */
                        vstore(ri, piAg);
                    }
                }
            }
            return;
        }

        /* vintx fallback (H not a multiple of DVECW, e.g. H=8). */
        const int32_t nHc = H / CRP_VECW;
        agen_C xr_agen    = initAgenC4_i(k->vdFxReBuf, numAzVirt, H, nHc, CRP_VECW, 1, numAzVirt * H, roiAz, 0);
        agen_C xi_agen    = initAgenC4_i(k->vdFxImBuf, numAzVirt, H, nHc, CRP_VECW, 1, numAzVirt * H, roiAz, 0);
        agen_A prAg       = CRP_AG_RELOAD(k, prAgV);
        agen_B piAg       = CRP_AG_RELOAD(k, piAgV);
        for (int32_t aLoc = 0; aLoc < roiAz; aLoc++)
        {
            int32_t ka = (roiAzStart + aLoc - shAz) % beamAz;
            if (ka < 0)
                ka += beamAz;
            const int16_t *wrow = k->waz16Buf + (int32_t)(ka * numAzVirt * 2);
            for (int32_t hc = 0; hc < nHc; hc++)
            {
                vintx accR = replicatew(0), accI = replicatew(0);
                for (int32_t a = 0; a < numAzVirt; a++)
                {
                    CRP_CMAC_V(accR, accI, xr_agen, xi_agen, wrow + a * 2);
                }
                const vintx rr = ((accR + rnd) >> rshift) << lshift;
                const vintx ri = ((accI + rnd) >> rshift) << lshift;
                for (int32_t eLoc = 0; eLoc < roiEl; eLoc++) chess_prepare_for_pipelining
                chess_loop_range(1, )
                {
                    vstore(rr, prAg);
                    vstore(ri, piAg);
                }
            }
        }
        return;
    }

#    if CRP_GENERAL_GEOMETRY
    /* numElVirt > 1: write tmpRoi; the elevation projection does the el-DFT off it. */
    if (H % DVECW == 0)
    {
        const int32_t nHd = H / DVECW;
        agen_A accR_agen  = CRP_AG_REBUILD(k, accAgReD);
        agen_A accI_agen  = CRP_AG_REBUILD(k, accAgImD);
        agen_C xr_agen    = initAgenC4_i(k->vdFxReBuf, numAzVirt, H, nHd, DVECW, numElVirt, numAzVirt * H, roiAz, 0);
        agen_C xi_agen    = initAgenC4_i(k->vdFxImBuf, numAzVirt, H, nHd, DVECW, numElVirt, numAzVirt * H, roiAz, 0);
        for (int32_t aLoc = 0; aLoc < roiAz; aLoc++)
        {
            int32_t ka = (roiAzStart + aLoc - shAz) % beamAz;
            if (ka < 0)
                ka += beamAz;
            const int16_t *wrow = k->waz16Buf + (int32_t)(ka * numAzVirt * 2);
            for (int32_t e = 0; e < numElVirt; e++)
            {
                for (int32_t hc = 0; hc < nHd; hc++)
                {
                    dvintx accR, accI;
                    accR.lo = replicatew(0), accR.hi = replicatew(0);
                    accI.lo = replicatew(0), accI.hi = replicatew(0);
                    for (int32_t a = 0; a < numAzVirt; a++)
                    {
                        CRP_CMAC_D(accR, accI, xr_agen, xi_agen, wrow + a * 2);
                    }
                    vstore(accR, accR_agen);
                    vstore(accI, accI_agen);
                }
            }
        }
        return;
    }

    const int32_t nHc = H / CRP_VECW;
    agen_A accR_agen  = CRP_AG_REBUILD(k, accAgReV);
    agen_A accI_agen  = CRP_AG_REBUILD(k, accAgImV);
    agen_C xr_agen    = initAgenC4_i(k->vdFxReBuf, numAzVirt, H, nHc, CRP_VECW, numElVirt, numAzVirt * H, roiAz, 0);
    agen_C xi_agen    = initAgenC4_i(k->vdFxImBuf, numAzVirt, H, nHc, CRP_VECW, numElVirt, numAzVirt * H, roiAz, 0);
    for (int32_t aLoc = 0; aLoc < roiAz; aLoc++)
    {
        int32_t ka = (roiAzStart + aLoc - shAz) % beamAz;
        if (ka < 0)
            ka += beamAz;
        const int16_t *wrow = k->waz16Buf + (int32_t)(ka * numAzVirt * 2);
        for (int32_t e = 0; e < numElVirt; e++)
        {
            for (int32_t hc = 0; hc < nHc; hc++)
            {
                vintx accR = replicatew(0), accI = replicatew(0);
                for (int32_t a = 0; a < numAzVirt; a++)
                {
                    CRP_CMAC_V(accR, accI, xr_agen, xi_agen, wrow + a * 2);
                }
                vstore(accR, accR_agen);
                vstore(accI, accI_agen);
            }
        }
    }
#    endif /* CRP_GENERAL_GEOMETRY */
}

#    if CRP_GENERAL_GEOMETRY
/* Elevation projection: el DFT of tmpRoi -> patch Q(inputQBits), h-vectorized; a no-op for
 * numElVirt==1, where Wel is the identity and the narrow is fused into caponProjectAzFx instead. */
void caponProjectElPerTapFx(CaponKnl *k, int32_t roiElStart)
{
    const int32_t roiEl = k->roiEl, roiAz = k->roiAz, H = k->H;
    const int32_t DVECW = chess_elementsof(dvintx); /* 16 */

    if (k->numElVirt == 1)
    {
        /* Fused into caponProjectAzFx; see the race note there. */
        (void)roiElStart;
        return;
    }

    /* Store 3D (eLoc, aLoc, hc); load 4D (eLoc, aLoc, hc, e) with eLoc-stride 0, as tmpRoi is
     * eLoc-independent. */
    const int32_t numElVirt = k->numElVirt, beamEl = k->beamEl, shEl = k->shEl;
    const int16_t *wel16 = k->wel16Buf;
    const int32_t shift  = (k->vdQBits - 2) - k->inputQBits;
    const int32_t rshift = (shift > 0) ? shift : 0;
    const int32_t lshift = (shift < 0) ? -shift : 0;
    const int32_t rnd    = (shift > 0) ? (1 << (shift - 1)) : 0;

    if (H % DVECW == 0)
    {
        const int32_t nHc = H / DVECW; /* 16-wide el DFT: half the h-blocks / MAC instructions */
        agen_A pR_agen    = CRP_AG_REBUILD(k, elPRAgD);
        agen_B pI_agen    = CRP_AG_REBUILD(k, elPIAgD);
        agen_A tr_agen    = CRP_AG_REBUILD(k, elTrAgD);
        agen_A ti_agen    = CRP_AG_REBUILD(k, elTiAgD);
        for (int32_t eLoc = 0; eLoc < roiEl; eLoc++)
        {
            int32_t ke = (roiElStart + eLoc - shEl) % beamEl;
            if (ke < 0)
                ke += beamEl;
            const int16_t *wrow = wel16 + (int32_t)(ke * numElVirt * 2);
            for (int32_t aLoc = 0; aLoc < roiAz; aLoc++)
            {
                for (int32_t hc = 0; hc < nHc; hc++)
                {
                    dvintx pR, pI;
                    pR.lo = replicatew(0), pR.hi = replicatew(0);
                    pI.lo = replicatew(0), pI.hi = replicatew(0);
                    for (int32_t e = 0; e < numElVirt; e++)
                    {
                        CRP_CMAC_D(pR, pI, tr_agen, ti_agen, wrow + e * 2);
                    }
                    const dvintx outR = ((pR + rnd) >> rshift) << lshift;
                    const dvintx outI = ((pI + rnd) >> rshift) << lshift;
                    vstore(outR, pR_agen);
                    vstore(outI, pI_agen);
                }
            }
        }
        return;
    }

    const int32_t nHc = H / CRP_VECW; /* fallback (H not a multiple of DVECW): 8-wide */
    agen_A pR_agen    = CRP_AG_REBUILD(k, elPRAgV);
    agen_B pI_agen    = CRP_AG_REBUILD(k, elPIAgV);
    agen_A tr_agen    = CRP_AG_REBUILD(k, elTrAgV);
    agen_A ti_agen    = CRP_AG_REBUILD(k, elTiAgV);
    for (int32_t eLoc = 0; eLoc < roiEl; eLoc++)
    {
        int32_t ke = (roiElStart + eLoc - shEl) % beamEl;
        if (ke < 0)
            ke += beamEl;
        const int16_t *wrow = wel16 + (int32_t)(ke * numElVirt * 2);
        for (int32_t aLoc = 0; aLoc < roiAz; aLoc++)
        {
            for (int32_t hc = 0; hc < nHc; hc++)
            {
                vintx pR = replicatew(0), pI = replicatew(0);
                for (int32_t e = 0; e < numElVirt; e++)
                {
                    CRP_CMAC_V(pR, pI, tr_agen, ti_agen, wrow + e * 2);
                }
                const vintx outR = ((pR + rnd) >> rshift) << lshift;
                const vintx outI = ((pI + rnd) >> rshift) << lshift;
                vstore(outR, pR_agen);
                vstore(outI, pI_agen);
            }
        }
    }
}
#    endif /* CRP_GENERAL_GEOMETRY */

#endif /* CRP_PROJ_GENERAL */
