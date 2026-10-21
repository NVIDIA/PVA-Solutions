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
 * @file capon_roi_preprocess_init.c
 * @brief Run-once setup of the per-detection stages. Kept apart from them, and linked first, so the
 *        per-detection code is one contiguous block that fits the 16 KB, 2-way VPU instruction cache.
 */

#include "capon_roi_preprocess_agens.h"
#include "capon_roi_preprocess_knl.h"

/* Build the invariant patch agens once; both the DVECW and vintx-fallback pairs, only one is read. */
void caponInitProjectAgensFx(CaponKnl *k)
{
    k->prAgD     = crpBuild_prAgD(k);
    k->piAgD     = crpBuild_piAgD(k);
    k->prAgV     = crpBuild_prAgV(k);
    k->piAgV     = crpBuild_piAgV(k);
    k->patchRdRe = crpBuild_patchRdRe(k);
    k->patchRdIm = crpBuild_patchRdIm(k);
#if CRP_GENERAL_GEOMETRY
    k->accAgReD = crpBuild_accAgReD(k);
    k->accAgImD = crpBuild_accAgImD(k);
    k->accAgReV = crpBuild_accAgReV(k);
    k->accAgImV = crpBuild_accAgImV(k);
    k->elPRAgD  = crpBuild_elPRAgD(k);
    k->elPIAgD  = crpBuild_elPIAgD(k);
    k->elTrAgD  = crpBuild_elTrAgD(k);
    k->elTiAgD  = crpBuild_elTiAgD(k);
    k->elPRAgV  = crpBuild_elPRAgV(k);
    k->elPIAgV  = crpBuild_elPIAgV(k);
    k->elTrAgV  = crpBuild_elTrAgV(k);
    k->elTiAgV  = crpBuild_elTiAgV(k);
#endif
}

void caponInitVdFx(CaponKnl *k)
{
    const int32_t nVec = 2 * k->nvcProj * k->H / CRP_VECW; /* H is a multiple of CRP_VECW */
    AgenWrapper w      = {.size = sizeof(int32_t), .n1 = (uint16_t)nVec, .n2 = 1, .s1 = CRP_VECW, .s2 = 0};
    agen_C z           = init_C((void *)k->vdBufInt);
    INIT_AGEN2(z, w);
    const vintx zero = replicatew(0);
    for (int32_t i = 0; i < nVec; i++) chess_prepare_for_pipelining
    {
        vstore(zero, z);
    }
    CRP_SEPARATOR();
}

/* caponCoarseDoARankAzFx's power-spectrum agens, built once (see the CaponKnl field comment). Both the
 * direct-DFT and the FFT kernel rank through them. */
void caponInitRankAgensFx(CaponKnl *k)
{
    const int32_t vwi = chess_elementsof(vintx);
    const int32_t nKa = (k->beamAz + vwi - 1) / vwi;
    k->rankTrAgen     = initAgenA2_i((int32_t *)k->coarseTmpReBuf, nKa, vwi, 1, 0);
    k->rankTiAgen     = initAgenA2_i((int32_t *)k->coarseTmpImBuf, nKa, vwi, 1, 0);
    k->rankPAgen      = initAgenA2_i((int32_t *)k->tmpRoiReBuf, nKa, vwi, 1, 0);
}
