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
 * @file capon_roi_preprocess_agens.h
 * @brief Agen builders shared by the run-once setup (capon_roi_preprocess_init.c) and the per-detection
 *        stages (capon_roi_preprocess_knl.c), which rebuild some agens in place on NVVPU (CRP_AG_REBUILD).
 */

#ifndef PVA_SOLUTIONS_PRIV_CAPON_ROI_PREPROCESS_AGENS_H
#define PVA_SOLUTIONS_PRIV_CAPON_ROI_PREPROCESS_AGENS_H

#include "capon_roi_preprocess_knl.h"

/* Agen builders (size in bytes, strides in elements), file-static so each TU gets its own copy. */
static inline agen_C initAgenC4_i(int32_t *base, int32_t n1, int32_t s1, int32_t n2, int32_t s2, int32_t n3, int32_t s3,
                                  int32_t n4, int32_t s4)
{
    AgenWrapper w = {.size = sizeof(int32_t),
                     .n1   = (uint16_t)n1,
                     .n2   = (uint16_t)n2,
                     .n3   = (uint16_t)n3,
                     .n4   = (uint16_t)n4,
                     .s1   = s1,
                     .s2   = s2,
                     .s3   = s3,
                     .s4   = s4};
    agen_C a      = init_C((void *)base);
    INIT_AGEN4(a, w);
    return a;
}

static inline agen_A initAgenA3_i(int32_t *base, int32_t n1, int32_t s1, int32_t n2, int32_t s2, int32_t n3, int32_t s3)
{
    AgenWrapper w = {.size = sizeof(int32_t),
                     .n1   = (uint16_t)n1,
                     .n2   = (uint16_t)n2,
                     .n3   = (uint16_t)n3,
                     .s1   = s1,
                     .s2   = s2,
                     .s3   = s3};
    agen_A a      = init_A((void *)base);
    INIT_AGEN3(a, w);
    return a;
}
static inline agen_B initAgenB3_i(int32_t *base, int32_t n1, int32_t s1, int32_t n2, int32_t s2, int32_t n3, int32_t s3)
{
    AgenWrapper w = {.size = sizeof(int32_t),
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

static inline agen_A initAgenA4_i(int32_t *base, int32_t n1, int32_t s1, int32_t n2, int32_t s2, int32_t n3, int32_t s3,
                                  int32_t n4, int32_t s4)
{
    AgenWrapper w = {.size = sizeof(int32_t),
                     .n1   = (uint16_t)n1,
                     .n2   = (uint16_t)n2,
                     .n3   = (uint16_t)n3,
                     .n4   = (uint16_t)n4,
                     .s1   = s1,
                     .s2   = s2,
                     .s3   = s3,
                     .s4   = s4};
    agen_A a      = init_A((void *)base);
    INIT_AGEN4(a, w);
    return a;
}

/* Unroll for caponQuantizeSnapshotsFxT's transpose-store loop. */
#define CRP_QUANT_TRANSP_UNROLL 4

static inline agen_A crpBuild_prAgD(const CaponKnl *k)
{
    const int32_t dvw = chess_elementsof(dvintx);
    return initAgenA3_i((int32_t *)k->patchReBuf, k->roiEl, k->roiAz * k->H, k->H / dvw, dvw, k->roiAz, k->H);
}
static inline agen_B crpBuild_piAgD(const CaponKnl *k)
{
    const int32_t dvw = chess_elementsof(dvintx);
    return initAgenB3_i((int32_t *)k->patchImBuf, k->roiEl, k->roiAz * k->H, k->H / dvw, dvw, k->roiAz, k->H);
}
static inline agen_A crpBuild_prAgV(const CaponKnl *k)
{
    return initAgenA3_i((int32_t *)k->patchReBuf, k->roiEl, k->roiAz * k->H, k->H / CRP_VECW, CRP_VECW, k->roiAz, k->H);
}
static inline agen_B crpBuild_piAgV(const CaponKnl *k)
{
    return initAgenB3_i((int32_t *)k->patchImBuf, k->roiEl, k->roiAz * k->H, k->H / CRP_VECW, CRP_VECW, k->roiAz, k->H);
}
/* Quantize reads of patch[svDim][H]. The dot-product kernels store one row per ROI beam (an azimuth-only array
 * makes every elevation slot of a beam identical), so the elevation dimension rewinds (stride 0). */
#if CRP_PROJ_GENERAL
static inline agen_A crpBuild_patchRdRe(const CaponKnl *k)
{
    return initAgenA2_i((int32_t *)k->patchReBuf, k->H / CRP_VECW, CRP_VECW, k->svDim, k->H);
}
static inline agen_B crpBuild_patchRdIm(const CaponKnl *k)
{
    return initAgenB2_i((int32_t *)k->patchImBuf, k->H / CRP_VECW, CRP_VECW, k->svDim, k->H);
}
#else
static inline agen_A crpBuild_patchRdRe(const CaponKnl *k)
{
    return initAgenA3_i((int32_t *)k->patchReBuf, k->H / CRP_VECW, CRP_VECW, k->roiAz, k->H, k->roiEl, 0);
}
static inline agen_B crpBuild_patchRdIm(const CaponKnl *k)
{
    return initAgenB3_i((int32_t *)k->patchImBuf, k->H / CRP_VECW, CRP_VECW, k->roiAz, k->H, k->roiEl, 0);
}
#endif
#if CRP_GENERAL_GEOMETRY
/* Elevation agens; built unconditionally, read only when numElVirt > 1. */
static inline agen_A crpBuildAcc(const CaponKnl *k, float *buf, int32_t w)
{
    const int32_t H = k->H, numElVirt = k->numElVirt;
    return initAgenA3_i((int32_t *)buf, H / w, w, numElVirt, H, k->roiAz, numElVirt * H);
}
static inline agen_A crpBuild_accAgReD(const CaponKnl *k)
{
    return crpBuildAcc(k, k->tmpRoiReBuf, chess_elementsof(dvintx));
}
static inline agen_A crpBuild_accAgImD(const CaponKnl *k)
{
    return crpBuildAcc(k, k->tmpRoiImBuf, chess_elementsof(dvintx));
}
static inline agen_A crpBuild_accAgReV(const CaponKnl *k)
{
    return crpBuildAcc(k, k->tmpRoiReBuf, CRP_VECW);
}
static inline agen_A crpBuild_accAgImV(const CaponKnl *k)
{
    return crpBuildAcc(k, k->tmpRoiImBuf, CRP_VECW);
}
static inline agen_A crpBuildElPR(const CaponKnl *k, int32_t w)
{
    const int32_t H = k->H, roiAz = k->roiAz;
    return initAgenA3_i((int32_t *)k->patchReBuf, H / w, w, roiAz, H, k->roiEl, roiAz * H);
}
static inline agen_B crpBuildElPI(const CaponKnl *k, int32_t w)
{
    const int32_t H = k->H, roiAz = k->roiAz;
    return initAgenB3_i((int32_t *)k->patchImBuf, H / w, w, roiAz, H, k->roiEl, roiAz * H);
}
static inline agen_A crpBuildElT(const CaponKnl *k, float *buf, int32_t w)
{
    const int32_t H = k->H, numElVirt = k->numElVirt;
    return initAgenA4_i((int32_t *)buf, numElVirt, H, H / w, w, k->roiAz, numElVirt * H, k->roiEl, 0);
}
static inline agen_A crpBuild_elPRAgD(const CaponKnl *k)
{
    return crpBuildElPR(k, chess_elementsof(dvintx));
}
static inline agen_B crpBuild_elPIAgD(const CaponKnl *k)
{
    return crpBuildElPI(k, chess_elementsof(dvintx));
}
static inline agen_A crpBuild_elTrAgD(const CaponKnl *k)
{
    return crpBuildElT(k, k->tmpRoiReBuf, chess_elementsof(dvintx));
}
static inline agen_A crpBuild_elTiAgD(const CaponKnl *k)
{
    return crpBuildElT(k, k->tmpRoiImBuf, chess_elementsof(dvintx));
}
static inline agen_A crpBuild_elPRAgV(const CaponKnl *k)
{
    return crpBuildElPR(k, CRP_VECW);
}
static inline agen_B crpBuild_elPIAgV(const CaponKnl *k)
{
    return crpBuildElPI(k, CRP_VECW);
}
static inline agen_A crpBuild_elTrAgV(const CaponKnl *k)
{
    return crpBuildElT(k, k->tmpRoiReBuf, CRP_VECW);
}
static inline agen_A crpBuild_elTiAgV(const CaponKnl *k)
{
    return crpBuildElT(k, k->tmpRoiImBuf, CRP_VECW);
}
#endif /* CRP_GENERAL_GEOMETRY */

#endif /* PVA_SOLUTIONS_PRIV_CAPON_ROI_PREPROCESS_AGENS_H */
