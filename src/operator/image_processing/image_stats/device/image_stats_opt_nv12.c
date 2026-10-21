/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "image_stats_opt_nv12.h"

void init_image_stats_nv12_cfg(IMAGE_STATS_CFG &cfg, uint8_t *srcPtrs[MAX_PLANE_COUNT], uint8_t *mask,
                               uint16_t *tileWidth, uint16_t *tileHeight, uint16_t *tileLp, uint16_t maskLp)
{
    int vecw = pva_elementsof(dvcharx);

    // load 2 rows of Y and mask, 1 row of U and V
    int niter2 = tileWidth[0] / vecw;
    int niter3 = tileHeight[0] / 2;

    AgenWrapper w0;
    w0.size = sizeof(uint8_t);
    w0.n1   = 2;
    w0.s1   = tileLp[0];
    w0.n2   = niter2;
    w0.s2   = vecw;
    w0.n3   = niter3;
    w0.s3   = tileLp[0] * 2;

    agen srcYAgen = init((dvuchar *)srcPtrs[0]);
    INIT_AGEN3(srcYAgen, w0);
    cfg._src0_agen_cfg = extract_agen_cfg(srcYAgen);

#if defined(COV_IO)
    // auxiliary agens for vdotp2
    agen srcY1Agen = init((dvuchar *)srcPtrs[0]);
    INIT_AGEN3(srcY1Agen, w0);
    cfg._srcC0_aux_agen_cfg = extract_agen_cfg(srcY1Agen);
#endif

#if defined(MASK_IO)
    w0.s1         = maskLp;
    w0.s3         = 2 * maskLp;
    agen maskAgen = init((dvuchar *)mask);
    INIT_AGEN3(maskAgen, w0);
    cfg._mask_agen_cfg = extract_agen_cfg(maskAgen);

#    if defined(COV_IO)
    // auxiliary agens for vdotp2
    agen mask1Agen = init((dvuchar *)mask);
    INIT_AGEN3(mask1Agen, w0);
    cfg._mask_aux_agen_cfg = extract_agen_cfg(mask1Agen);
#    endif

#endif

    AgenWrapper w1;
    w1.size        = sizeof(uint8_t);
    w1.n1          = niter2;
    w1.s1          = vecw;
    w1.n2          = niter3;
    w1.s2          = tileLp[1];
    agen srcUVAgen = init((dvuchar *)srcPtrs[1]);
    INIT_AGEN2(srcUVAgen, w1);
    cfg._src1_agen_cfg = extract_agen_cfg(srcUVAgen);

#if defined(COV_IO)
    agen srcUV1Agen = init((dvuchar *)srcPtrs[1]);
    INIT_AGEN2(srcUV1Agen, w1);
    cfg._srcC1_aux_agen_cfg = extract_agen_cfg(srcUV1Agen);
#endif

    cfg._niter = niter2 * niter3;
}

void image_stats_opt_nv12_exec(IMAGE_STATS_CFG &cfg, uint8_t *srcPtrs[MAX_PLANE_COUNT])
{
    cupvaModifyAgenCfgBase(&cfg._src0_agen_cfg, srcPtrs[0]);
    cupvaModifyAgenCfgBase(&cfg._src1_agen_cfg, srcPtrs[1]);

    agen_B a_y  = init_agen_B_from_cfg(cfg._src0_agen_cfg);
    agen_C a_uv = init_agen_C_from_cfg(cfg._src1_agen_cfg);

#if defined(COV_IO)
    cupvaModifyAgenCfgBase(&cfg._srcC0_aux_agen_cfg, srcPtrs[0]);
    cupvaModifyAgenCfgBase(&cfg._srcC1_aux_agen_cfg, srcPtrs[1]);
    agen_B a_y_aux  = init_agen_B_from_cfg(cfg._srcC0_aux_agen_cfg);
    agen_C a_uv_aux = init_agen_C_from_cfg(cfg._srcC1_aux_agen_cfg);
#endif

#if defined(SUM_IO)
    int32_t sum_y = 0;
    int32_t sum_u = 0;
    int32_t sum_v = 0;
#endif

#if defined(VAR_IO)
    int32_t varSum_y = 0;
    int32_t varSum_u = 0;
    int32_t varSum_v = 0;
#endif

#if defined(COV_IO)
    int32_t covSum_yu = 0;
    int32_t covSum_yv = 0;
    int32_t covSum_uv = 0;
#endif

    int niter = cfg._niter;
    niter     = max((niter + 7) / 8 * 8, 24);

    chess_separator_scheduler();

    for (int i = 0; i < niter; i++) chess_loop_range(24, )
#if !defined(VAR_IO)
    chess_unroll_loop(4)
#endif
    chess_prepare_for_pipelining
    {

#if defined(SUM_IO)
        dvcharx chess_storage(DV) dvdata_y1 = dvuchar_load_di(a_y);
        dvcharx chess_storage(DV) dvdata_y2 = dvuchar_load_di(a_y);

        // sum of pixel values for Y channel
        dvcharx dvdata_y = dvdata_y1 + dvdata_y2;

        vcharx bsum_y = dvdata_y.lo + dvdata_y.hi;
        sum_y += vsumr_s(bsum_y);

        dvcharx dvdata_uv = dvuchar_load_di(a_uv);

        // sum of pixel values for U and V channels
        sum_u += vsumr_s(dvdata_uv.lo);
        sum_v += vsumr_s(dvdata_uv.hi);
#endif

#if defined(VAR_IO)
        // sum of pixel square for Y channel
        dvshortx dhprod_yy1 = vsumsq(dvdata_y1.lo, dvdata_y1.hi);
        dvshortx dhprod_yy2 = vsumsq(dvdata_y2.lo, dvdata_y2.hi);
        dvshortx dhprod_yy  = dhprod_yy1 + dhprod_yy2;

        vshortx hprod_yy = dhprod_yy.lo + dhprod_yy.hi;
        varSum_y += vsumr_s(hprod_yy);

        // sum of pixel square for variance
        dvshortx dhprod_uu = vmulbh(dvdata_uv.lo, dvdata_uv.lo, 0);
        vshortx hprod_uu   = dhprod_uu.lo + dhprod_uu.hi;
        varSum_u += vsumr_s(hprod_uu);

        dvshortx dhprod_vv = vmulbh(dvdata_uv.hi, dvdata_uv.hi, 0);
        vshortx hprod_vv   = dhprod_vv.lo + dhprod_vv.hi;
        varSum_v += vsumr_s(hprod_vv);
#endif

#if defined(COV_IO)
        // cov for uv
        dvshortx dhprod_uv = vmulbh(dvdata_uv.lo, dvdata_uv.hi, 0);
        vshortx hprod_uv   = dhprod_uv.lo + dhprod_uv.hi;
        covSum_uv += vsumr_s(hprod_uv);

        // cov for yu and yv
        dvcharx dvdata_y1_aux = dvuchar_load(a_y_aux);
        dvcharx dvdata_y2_aux = dvuchar_load(a_y_aux);
        dvcharx dvdata_uv_aux = dvuchar_load(a_uv_aux);

        dvshortx dhprod_yuyv;
        dhprod_yuyv = vdotp2_bbh(dvdata_uv_aux.lo, dvdata_uv_aux.lo, dvdata_y1_aux.lo, dhprod_yuyv, 0);
        dhprod_yuyv = vdotp2_bbh(dvdata_uv_aux.lo, dvdata_uv_aux.lo, dvdata_y2_aux.lo, dhprod_yuyv, -1);
        dhprod_yuyv = vdotp2_bbh(dvdata_uv_aux.hi, dvdata_uv_aux.hi, dvdata_y1_aux.hi, dhprod_yuyv, -1);
        dhprod_yuyv = vdotp2_bbh(dvdata_uv_aux.hi, dvdata_uv_aux.hi, dvdata_y2_aux.hi, dhprod_yuyv, -1);

        covSum_yu += vsumr_s(dhprod_yuyv.lo);
        covSum_yv += vsumr_s(dhprod_yuyv.hi);
#endif
    }

#if defined(SUM_IO)
    cfg._perChannelSumU64Buf[0] += sum_y;
    cfg._perChannelSumU64Buf[1] += sum_u * 4;
    cfg._perChannelSumU64Buf[2] += sum_v * 4;
#endif

#if defined(VAR_IO)
    cfg._covarianceMatU64Buf[0] += varSum_y;
    cfg._covarianceMatU64Buf[5] += varSum_u * 4;
    cfg._covarianceMatU64Buf[10] += varSum_v * 4;
#endif

#if defined(COV_IO)
    // only fill the upper triangular part of the covariance matrix
    cfg._covarianceMatU64Buf[1] += covSum_yu;
    cfg._covarianceMatU64Buf[2] += covSum_yv;
    cfg._covarianceMatU64Buf[6] += covSum_uv * 4;
#endif
}

void image_stats_opt_mask_nv12_exec(IMAGE_STATS_CFG &cfg, uint8_t *srcPtrs[MAX_PLANE_COUNT], uint8_t *mask)
{
    cupvaModifyAgenCfgBase(&cfg._src0_agen_cfg, srcPtrs[0]);
    cupvaModifyAgenCfgBase(&cfg._src1_agen_cfg, srcPtrs[1]);
    cupvaModifyAgenCfgBase(&cfg._mask_agen_cfg, mask);

    agen_B a_y    = init_agen_B_from_cfg(cfg._src0_agen_cfg);
    agen_C a_uv   = init_agen_C_from_cfg(cfg._src1_agen_cfg);
    agen_A a_mask = init_agen_A_from_cfg(cfg._mask_agen_cfg);

#if defined(COV_IO)
    cupvaModifyAgenCfgBase(&cfg._srcC0_aux_agen_cfg, srcPtrs[0]);
    cupvaModifyAgenCfgBase(&cfg._srcC1_aux_agen_cfg, srcPtrs[1]);
    cupvaModifyAgenCfgBase(&cfg._mask_aux_agen_cfg, mask);
    agen_B a_y_aux    = init_agen_B_from_cfg(cfg._srcC0_aux_agen_cfg);
    agen_C a_uv_aux   = init_agen_C_from_cfg(cfg._srcC1_aux_agen_cfg);
    agen_A a_mask_aux = init_agen_A_from_cfg(cfg._mask_aux_agen_cfg);
#endif

#if defined(CNT_IO)
    int32_t pixelCount_y = 0;
#endif

#if defined(SUM_IO)
    int32_t sum_y = 0;
    int32_t sum_u = 0;
    int32_t sum_v = 0;
#endif

#if defined(VAR_IO)
    int32_t varSum_y = 0;
    int32_t varSum_u = 0;
    int32_t varSum_v = 0;
#endif

#if defined(COV_IO)
    int32_t covSum_yu = 0;
    int32_t covSum_yv = 0;
    int32_t covSum_uv = 0;
#endif

    int niter = cfg._niter;
    niter     = max((niter + 7) / 8 * 8, 24);

    chess_separator_scheduler();

    for (int i = 0; i < niter; i++) chess_loop_range(24, )
#if !defined(SUM_IO)
    chess_unroll_loop(8)
#elif !defined(COV_IO)
    chess_unroll_loop(2)
#endif
    chess_prepare_for_pipelining
    {
        dvcharx dvmask1    = dvuchar_load_di(a_mask);
        dvcharx dvmask2    = dvuchar_load_di(a_mask);
        dvcharx dvmask_sum = dvmask1 + dvmask2;
        vcharx vmask_sum   = dvmask_sum.lo + dvmask_sum.hi;

#if defined(CNT_IO)
        // pixel count
        pixelCount_y += vsumr_s(vmask_sum);
#endif

#if defined(SUM_IO)
        dvcharx dvdata_y1 = dvuchar_load_di(a_y);
        dvcharx dvdata_y2 = dvuchar_load_di(a_y);

        // apply the binary mask
        // dvmulb: double vector pseudo intrinsics
        dvcharx dvdata_y1_nz = dvmulb(dvdata_y1, dvmask1, 0);
        dvcharx dvdata_y2_nz = dvmulb(dvdata_y2, dvmask2, 0);
        // sum of pixel values for Y channel
        dvcharx dvdata_y_nz_sum = dvdata_y1_nz + dvdata_y2_nz;
        vcharx bsum_y           = dvdata_y_nz_sum.lo + dvdata_y_nz_sum.hi;
        sum_y += vsumr_s(bsum_y);

        dvcharx dvdata_uv = dvuchar_load_di(a_uv);

        // apply sum of mask to U and V channels
        dvshortx dh_u_nz_sum = vmulbh(dvdata_uv.lo, vmask_sum, 0);
        dvshortx dh_v_nz_sum = vmulbh(dvdata_uv.hi, vmask_sum, 0);

        // sum of pixel values for U and V channels
        vshortx bsum_u = dh_u_nz_sum.lo + dh_u_nz_sum.hi;
        sum_u += vsumr_s(bsum_u);

        vshortx bsum_v = dh_v_nz_sum.lo + dh_v_nz_sum.hi;
        sum_v += vsumr_s(bsum_v);
#endif

#if defined(VAR_IO)
        // sum of pixel square for Y channel
        dvshortx dhprod_yy1 = vsumsq(dvdata_y1_nz.lo, dvdata_y1_nz.hi);
        dvshortx dhprod_yy2 = vsumsq(dvdata_y2_nz.lo, dvdata_y2_nz.hi);
        dvshortx dhprod_yy  = dhprod_yy1 + dhprod_yy2;
        vshortx hprod_yy    = dhprod_yy.lo + dhprod_yy.hi;
        varSum_y += vsumr_s(hprod_yy);

        // sum of pixel square for U and V channels
        dvshortx dh_u = vpromote_di(dvdata_uv.lo);
        dvshortx dh_v = vpromote_di(dvdata_uv.hi);

        // sum of pixel square for U and V channels
        // pseudo double vector intrinsics
        dvshortx dhprod_uu = dvmulh(dh_u_nz_sum, dh_u, 0);
        vshortx hprod_uu   = dhprod_uu.lo + dhprod_uu.hi;
        varSum_u += vsumr_s(hprod_uu);

        dvshortx dhprod_vv = dvmulh(dh_v_nz_sum, dh_v, 0);
        vshortx hprod_vv   = dhprod_vv.lo + dhprod_vv.hi;
        varSum_v += vsumr_s(hprod_vv);
#endif

#if defined(COV_IO)
        // cov for uv
        // pseudo double vector intrinsics
        dvshortx dhprod_uv = dvmulh(dh_u_nz_sum, dh_v, 0);
        vshortx hprod_uv   = dhprod_uv.lo + dhprod_uv.hi;
        covSum_uv += vsumr_s(hprod_uv);

        dvcharx dvdata_y1_aux = dvuchar_load(a_y_aux);
        dvcharx dvdata_y2_aux = dvuchar_load(a_y_aux);
        dvcharx dvmask1_aux   = dvuchar_load(a_mask_aux);
        dvcharx dvmask2_aux   = dvuchar_load(a_mask_aux);
        dvcharx dvdata_uv_aux = dvuchar_load(a_uv_aux);

        // apply the binary mask
        // dvmulb: double vector pseudo intrinsics
        dvcharx dvdata_y1_aux_nz = dvmulb(dvdata_y1_aux, dvmask1_aux, 0);
        dvcharx dvdata_y2_aux_nz = dvmulb(dvdata_y2_aux, dvmask2_aux, 0);

        // cov for yu and yv
        // yu at even lanes, yv at odd lanes
        dvshortx dhprod_yuyv;
        dhprod_yuyv = vdotp2_bbh(dvdata_uv_aux.lo, dvdata_uv_aux.lo, dvdata_y1_aux_nz.lo, dhprod_yuyv, 0);
        dhprod_yuyv = vdotp2_bbh(dvdata_uv_aux.lo, dvdata_uv_aux.lo, dvdata_y2_aux_nz.lo, dhprod_yuyv, -1);
        dhprod_yuyv = vdotp2_bbh(dvdata_uv_aux.hi, dvdata_uv_aux.hi, dvdata_y1_aux_nz.hi, dhprod_yuyv, -1);
        dhprod_yuyv = vdotp2_bbh(dvdata_uv_aux.hi, dvdata_uv_aux.hi, dvdata_y2_aux_nz.hi, dhprod_yuyv, -1);
        covSum_yu += vsumr_s(dhprod_yuyv.lo);
        covSum_yv += vsumr_s(dhprod_yuyv.hi);
#endif
    }

#if defined(CNT_IO)
    cfg._planePixelCount[0] += pixelCount_y;
#endif

#if defined(SUM_IO)
    cfg._perChannelSumU64Buf[0] += sum_y;
    cfg._perChannelSumU64Buf[1] += sum_u;
    cfg._perChannelSumU64Buf[2] += sum_v;
#endif

#if defined(VAR_IO)
    cfg._covarianceMatU64Buf[0] += varSum_y;
    cfg._covarianceMatU64Buf[5] += varSum_u;
    cfg._covarianceMatU64Buf[10] += varSum_v;
#endif

#if defined(COV_IO)
    // only fill the upper triangular part of the covariance matrix
    cfg._covarianceMatU64Buf[1] += covSum_yu;
    cfg._covarianceMatU64Buf[2] += covSum_yv;
    cfg._covarianceMatU64Buf[6] += covSum_uv;
#endif
}
