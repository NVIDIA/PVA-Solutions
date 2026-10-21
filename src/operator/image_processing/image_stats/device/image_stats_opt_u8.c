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

#include "image_stats_opt_u8.h"

void init_image_stats_u8_cfg(IMAGE_STATS_CFG &cfg, uint8_t *srcPtrs[MAX_PLANE_COUNT], uint8_t *mask,
                             uint16_t *tileWidth, uint16_t *tileHeight, uint16_t *tileLp, uint16_t maskLp)
{
    int vecw = pva_elementsof(dvcharx);

    AgenWrapper w0;
    w0.size = sizeof(uint8_t);
    w0.n1   = tileWidth[0] / vecw;
    w0.s1   = vecw;
    w0.n2   = tileHeight[0];
    w0.s2   = tileLp[0];

    agen srcAgen = init((dvuchar *)srcPtrs[0]);
    INIT_AGEN2(srcAgen, w0);
    cfg._src0_agen_cfg = extract_agen_cfg(srcAgen);

    cfg._niter = w0.n1 * w0.n2;

#if defined(MASK_IO)
    w0.s2         = maskLp;
    agen maskAgen = init((dvuchar *)mask);
    INIT_AGEN2(maskAgen, w0);
    cfg._mask_agen_cfg = extract_agen_cfg(maskAgen);
#endif
}

void image_stats_opt_u8_exec(IMAGE_STATS_CFG &cfg, uint8_t *srcPtrs[MAX_PLANE_COUNT])
{
    cupvaModifyAgenCfgBase(&cfg._src0_agen_cfg, srcPtrs[0]);
    agen a_src = init_agen_from_cfg(cfg._src0_agen_cfg);

    int niter = cfg._niter;
    niter     = max((niter + 7) / 8 * 8, 24);

    chess_separator_scheduler();

#if defined(SUM_IO)
    int32_t sum = 0;
#endif

#if defined(VAR_IO)
    int32_t varSum = 0;
#endif

    for (int i = 0; i < niter; i++) chess_loop_range(24, )
#if defined(VAR_IO)
    chess_unroll_loop(4)
#else
    chess_unroll_loop(8)
#endif
    chess_prepare_for_pipelining
    {

#if defined(SUM_IO)
        dvcharx dvdata = dvuchar_load(a_src);
        // per channel sum
        vcharx b_sum = dvdata.lo + dvdata.hi;
        sum += vsumr_s(b_sum);
#endif

#if defined(VAR_IO)
        // sum of pixel square for variance
        dvshortx dhprod = vsumsq(dvdata.lo, dvdata.hi);
        vshortx hprod   = dhprod.lo + dhprod.hi;
        varSum += vsumr_s(hprod);
#endif
    }

#if defined(SUM_IO)
    cfg._perChannelSumU64Buf[0] += sum;
#endif

#if defined(VAR_IO)
    cfg._covarianceMatU64Buf[0] += varSum;
#endif
}

void image_stats_opt_mask_u8_exec(IMAGE_STATS_CFG &cfg, uint8_t *srcPtrs[MAX_PLANE_COUNT], uint8_t *mask)
{
    cupvaModifyAgenCfgBase(&cfg._src0_agen_cfg, srcPtrs[0]);
    cupvaModifyAgenCfgBase(&cfg._mask_agen_cfg, mask);
    agen_B a_src  = init_agen_B_from_cfg(cfg._src0_agen_cfg);
    agen_C a_mask = init_agen_C_from_cfg(cfg._mask_agen_cfg);

    int niter = cfg._niter;
    niter     = max((niter + 7) / 8 * 8, 24);

    chess_separator_scheduler();

#if defined(CNT_IO)
    int32_t pixelCount = 0;
#endif

#if defined(SUM_IO)
    int32_t sum = 0;
#endif

#if defined(VAR_IO)
    int32_t varSum = 0;
#endif

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
        dvcharx dvmask = dvuchar_load(a_mask);

#if defined(CNT_IO)
        // pixel count
        vcharx b_mask = dvmask.lo + dvmask.hi;
        pixelCount += vsumr_s(b_mask);
#endif

#if defined(SUM_IO)
        dvcharx dvdata = dvuchar_load(a_src);
        // apply binary mask
        //double vector pseudo intrinsics
        dvcharx dv_nz_data = dvmulb(dvdata, dvmask, 0);
        // per channel sum
        vcharx b_sum = dv_nz_data.lo + dv_nz_data.hi;
        sum += vsumr_s(b_sum);
#endif

#if defined(VAR_IO)
        // sum of pixel square for variance
        dvshortx dhprod = vsumsq(dv_nz_data.lo, dv_nz_data.hi);
        vshortx hprod   = dhprod.lo + dhprod.hi;
        varSum += vsumr_s(hprod);
#endif
    }

#if defined(CNT_IO)
    cfg._planePixelCount[0] += pixelCount;
#endif

#if defined(SUM_IO)
    cfg._perChannelSumU64Buf[0] += sum;
#endif

#if defined(VAR_IO)
    cfg._covarianceMatU64Buf[0] += varSum;
#endif
}
