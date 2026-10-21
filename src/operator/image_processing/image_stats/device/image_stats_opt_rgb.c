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

#include "image_stats_opt_rgb.h"

void init_image_stats_rgb_cfg(IMAGE_STATS_CFG &cfg, uint8_t *srcPtrs[MAX_PLANE_COUNT], uint8_t *mask,
                              uint16_t *tileWidth, uint16_t *tileHeight, uint16_t *tileLp, uint16_t maskLp)
{
    int vecw = pva_elementsof(vcharx);
    // transpose load for 3 channels
    AgenWrapper w0;
    w0.size = sizeof(uint8_t);
    w0.n1   = 3;
    w0.s1   = 1;
    w0.n2   = tileHeight[0] / vecw;
    w0.s2   = tileLp[0] * vecw;
    w0.n3   = tileWidth[0];
    w0.s3   = 3;

    agen srcC0Agen = init((vuchar *)srcPtrs[0]);
    INIT_AGEN3(srcC0Agen, w0);
    int srcLaneOfst     = (tileLp[0] - 2) / 64;
    srcC0Agen.lane_ofst = srcLaneOfst;

    cfg._src0_agen_cfg = extract_agen_cfg(srcC0Agen);

    cfg._niter = w0.n2 * w0.n3;

#if defined(MASK_IO)
    AgenWrapper w1;
    w1.size       = sizeof(uint8_t);
    w1.n1         = tileHeight[0] / vecw;
    w1.s1         = maskLp * vecw;
    w1.n2         = tileWidth[0];
    w1.s2         = 1;
    agen maskAgen = init((vuchar *)mask);
    INIT_AGEN2(maskAgen, w1);
    int maskLaneOfst   = (maskLp - 2) / 64;
    maskAgen.lane_ofst = maskLaneOfst;
    cfg._mask_agen_cfg = extract_agen_cfg(maskAgen);
#endif
}

void image_stats_opt_rgb_exec(IMAGE_STATS_CFG &cfg, uint8_t *srcPtrs[MAX_PLANE_COUNT])
{
    cupvaModifyAgenCfgBase(&cfg._src0_agen_cfg, srcPtrs[0]);

    agen a_rgb = init_agen_from_cfg(cfg._src0_agen_cfg);

    int niter = cfg._niter;
    niter     = max((niter + 7) / 8 * 8, 24);

#if defined(SUM_IO)
    int32_t sum_r = 0;
    int32_t sum_g = 0;
    int32_t sum_b = 0;
#endif

#if defined(VAR_IO)
    int32_t varSum_r = 0;
    int32_t varSum_g = 0;
    int32_t varSum_b = 0;
#endif

#if defined(COV_IO)
    int32_t covSum_rg = 0;
    int32_t covSum_rb = 0;
    int32_t covSum_gb = 0;
#endif

    chess_separator_scheduler();

    for (int i = 0; i < niter; i++) chess_loop_range(24, )
#if defined(VAR_IO)
    chess_unroll_loop(2)
#else
    chess_unroll_loop(8)
#endif
    chess_prepare_for_pipelining
    {

#if defined(SUM_IO)
        vcharx vdata_r = vuchar_load_transp(a_rgb);
        vcharx vdata_g = vuchar_load_transp(a_rgb);
        vcharx vdata_b = vuchar_load_transp(a_rgb);
        // per channel sum
        sum_r += vsumr_s(vdata_r);
        sum_g += vsumr_s(vdata_g);
        sum_b += vsumr_s(vdata_b);
#endif

#if defined(VAR_IO)
        // sum of pixel square for variance
        dvshortx dhprod_rr = vmulbh(vdata_r, vdata_r, 0);
        varSum_r += vsumr_s(dhprod_rr.lo) + vsumr_s(dhprod_rr.hi);
        dvshortx dhprod_gg = vmulbh(vdata_g, vdata_g, 0);
        varSum_g += vsumr_s(dhprod_gg.lo) + vsumr_s(dhprod_gg.hi);
        dvshortx dhprod_bb = vmulbh(vdata_b, vdata_b, 0);
        varSum_b += vsumr_s(dhprod_bb.lo) + vsumr_s(dhprod_bb.hi);
#endif

#if defined(COV_IO)
        // only fill the upper triangular part of the covariance matrix
        dvshortx dhprod_rg = vmulbh(vdata_r, vdata_g, 0);
        covSum_rg += vsumr_s(dhprod_rg.lo) + vsumr_s(dhprod_rg.hi);
        dvshortx dhprod_rb = vmulbh(vdata_r, vdata_b, 0);
        covSum_rb += vsumr_s(dhprod_rb.lo) + vsumr_s(dhprod_rb.hi);
        dvshortx dhprod_gb = vmulbh(vdata_g, vdata_b, 0);
        covSum_gb += vsumr_s(dhprod_gb.lo) + vsumr_s(dhprod_gb.hi);
#endif
    }

#if defined(SUM_IO)
    cfg._perChannelSumU64Buf[0] += sum_r;
    cfg._perChannelSumU64Buf[1] += sum_g;
    cfg._perChannelSumU64Buf[2] += sum_b;
#endif

#if defined(VAR_IO)
    cfg._covarianceMatU64Buf[0] += varSum_r;
    cfg._covarianceMatU64Buf[5] += varSum_g;
    cfg._covarianceMatU64Buf[10] += varSum_b;
#endif

#if defined(COV_IO)
    cfg._covarianceMatU64Buf[1] += covSum_rg;
    cfg._covarianceMatU64Buf[2] += covSum_rb;
    cfg._covarianceMatU64Buf[6] += covSum_gb;
#endif
}

void image_stats_opt_mask_rgb_exec(IMAGE_STATS_CFG &cfg, uint8_t *srcPtrs[MAX_PLANE_COUNT], uint8_t *mask)
{
    cupvaModifyAgenCfgBase(&cfg._src0_agen_cfg, srcPtrs[0]);
    cupvaModifyAgenCfgBase(&cfg._mask_agen_cfg, mask);

    agen_B a_rgb  = init_agen_B_from_cfg(cfg._src0_agen_cfg);
    agen_C a_mask = init_agen_C_from_cfg(cfg._mask_agen_cfg);

    int niter = cfg._niter;
    niter     = max((niter + 7) / 8 * 8, 24);

#if defined(CNT_IO)
    int32_t pixelCount = 0;
#endif

#if defined(SUM_IO)
    int32_t sum_r = 0;
    int32_t sum_g = 0;
    int32_t sum_b = 0;
#endif

#if defined(VAR_IO)
    int32_t varSum_r = 0;
    int32_t varSum_g = 0;
    int32_t varSum_b = 0;
#endif

#if defined(COV_IO)
    int32_t covSum_rg = 0;
    int32_t covSum_rb = 0;
    int32_t covSum_gb = 0;
#endif

    chess_separator_scheduler();

    for (int i = 0; i < niter; i++) chess_loop_range(24, )
#if defined(VAR_IO)
    chess_unroll_loop(2)
#elif defined(SUM_IO)
    chess_unroll_loop(4)
#else
    chess_unroll_loop(8)
#endif
    chess_prepare_for_pipelining
    {
        vcharx vmask = vuchar_load_transp(a_mask);

#if defined(CNT_IO)
        // pixel count
        pixelCount += vsumr_s(vmask);
#endif

#if defined(SUM_IO)
        vcharx vdata_r = vuchar_load_transp(a_rgb);
        vcharx vdata_g = vuchar_load_transp(a_rgb);
        vcharx vdata_b = vuchar_load_transp(a_rgb);
        // apply binary mask
        vcharx v_nz_r = vmulb(vdata_r, vmask, 0);
        vcharx v_nz_g = vmulb(vdata_g, vmask, 0);
        vcharx v_nz_b = vmulb(vdata_b, vmask, 0);
        // per channel sum
        sum_r += vsumr_s(v_nz_r);
        sum_g += vsumr_s(v_nz_g);
        sum_b += vsumr_s(v_nz_b);
#endif

#if defined(VAR_IO)
        // sum of pixel square for variance
        dvshortx dhprod_rr = vmulbh(v_nz_r, v_nz_r, 0);
        varSum_r += vsumr_s(dhprod_rr.lo) + vsumr_s(dhprod_rr.hi);
        dvshortx dhprod_gg = vmulbh(v_nz_g, v_nz_g, 0);
        varSum_g += vsumr_s(dhprod_gg.lo) + vsumr_s(dhprod_gg.hi);
        dvshortx dhprod_bb = vmulbh(v_nz_b, v_nz_b, 0);
        varSum_b += vsumr_s(dhprod_bb.lo) + vsumr_s(dhprod_bb.hi);
#endif

#if defined(COV_IO)
        // only fill the upper triangular part of the covariance matrix
        dvshortx dhprod_rg = vmulbh(v_nz_r, v_nz_g, 0);
        covSum_rg += vsumr_s(dhprod_rg.lo) + vsumr_s(dhprod_rg.hi);
        dvshortx dhprod_rb = vmulbh(v_nz_r, v_nz_b, 0);
        covSum_rb += vsumr_s(dhprod_rb.lo) + vsumr_s(dhprod_rb.hi);
        dvshortx dhprod_gb = vmulbh(v_nz_g, v_nz_b, 0);
        covSum_gb += vsumr_s(dhprod_gb.lo) + vsumr_s(dhprod_gb.hi);
#endif
    }

#if defined(CNT_IO)
    cfg._planePixelCount[0] += pixelCount;
#endif

#if defined(SUM_IO)
    cfg._perChannelSumU64Buf[0] += sum_r;
    cfg._perChannelSumU64Buf[1] += sum_g;
    cfg._perChannelSumU64Buf[2] += sum_b;
#endif

#if defined(VAR_IO)
    cfg._covarianceMatU64Buf[0] += varSum_r;
    cfg._covarianceMatU64Buf[5] += varSum_g;
    cfg._covarianceMatU64Buf[10] += varSum_b;
#endif

#if defined(COV_IO)
    cfg._covarianceMatU64Buf[1] += covSum_rg;
    cfg._covarianceMatU64Buf[2] += covSum_rb;
    cfg._covarianceMatU64Buf[6] += covSum_gb;
#endif
}
