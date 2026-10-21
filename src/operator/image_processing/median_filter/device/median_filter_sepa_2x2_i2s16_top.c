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

/*
 * Separable 2x2 median filter for channel-interleaved 2S16 input.
 *
 * The 2x2 reuse scheme needs an even/odd-pixel split from the deinterleaving
 * double-vector loads, but for 2S16 the stride-2 deinterleave is consumed by
 * the channel split. Instead of doubling register pressure in the sorting
 * loops, the tile is first deinterleaved into two flat single-channel planes
 * (one small memory-bound copy loop), the unmodified single-channel separable
 * pipeline runs once per plane, and a second small copy loop re-interleaves
 * the two planar outputs into the 2S16 destination tile.
 *
 * The plane geometry macros (TILE_H_DF, PLANE_*) come from
 * median_filter_param.h; the copy-pass agen setup and loops come from
 * median_filter_common_sepa_2x2.{h,c}.
 */

#include "median_filter_common_sepa_2x2.h"
#include "median_filter_param.h"
#include "median_filter_sepa_2x2_i2s16.h"

#if defined(KNL_7x7)
#    include "median_filter_knl_sepa_2x2_7x7.h"

#elif defined(KNL_5x7)
#    include "median_filter_knl_sepa_2x2_5x7.h"

#elif defined(KNL_5x5)
#    include "median_filter_knl_sepa_2x2_5x5.h"

#endif

VMEM(A, BASE_TYPE, planeCh0, PLANE_LP *PLANE_ROWS);
VMEM(C, BASE_TYPE, planeCh1, PLANE_LP *PLANE_ROWS);

VMEM(B, BASE_TYPE, outPlaneCh0, CH_TILE_W *TILE_H_DF);
VMEM(A, BASE_TYPE, outPlaneCh1, CH_TILE_W *TILE_H_DF);

VMEM(A, BASE_TYPE, tmpA, INTERM_TILE_SIZE *SEPA_2X2_NUM_ELEMENTS_A);
VMEM(B, BASE_TYPE, tmpB, INTERM_TILE_SIZE *SEPA_2X2_NUM_ELEMENTS_B);
VMEM(C, BASE_TYPE, tmpC, INTERM_TILE_SIZE *SEPA_2X2_NUM_ELEMENTS_C);

VMEM(A, BASE_TYPE, kept_idx_1, KEEP_IDX_1_SIZE *GROUP_NUM_PASS_2);
VMEM(B, BASE_TYPE, kept_idx_2_top, KEEP_IDX_2_SIZE *GROUP_NUM_PASS_2);
VMEM(C, BASE_TYPE, kept_idx_2_bottom, KEEP_IDX_2_SIZE *GROUP_NUM_PASS_2);
VMEM(A, BASE_TYPE, st_sorted_left_col, (KNL_H - 1) * GROUP_NUM_PASS_2);
VMEM(C, BASE_TYPE, st_sorted_right_col, (KNL_H - 1) * GROUP_NUM_PASS_2);

VMEM(B, BASE_TYPE, srcBuf, RDF_CIRCULAR(BASE_TYPE, TILE_W, TILE_H_DF, HALO_X, HALO_Y));
VMEM(C, BASE_TYPE, dstBuf, RDF_DOUBLE(BASE_TYPE, TILE_W, TILE_H_DF));

// Handlers for cupva DMA operations
VMEM_RDF_UNIFIED(A, srcHdl);
VMEM_RDF_UNIFIED(A, dstHdl);

VMEM(A, int32_t, nTileX);
VMEM(A, int32_t, nTileY);

VMEM(A, MedianFilterSepa2x2Config, cfgCh0);
VMEM(A, MedianFilterSepa2x2Config, cfgCh1);
VMEM(A, MedianFilterSepa2x2PlanarConfig, cfgPlanar);

CUPVA_VPU_MAIN()
{
    cupvaRasterDataFlowOpen(srcHdl, srcBuf);
    cupvaRasterDataFlowOpen(dstHdl, dstBuf);

    // The source data flow is configured in whole-pixel (2-element) units;
    // scale its geometry to BASE_TYPE elements. The destination flow is
    // already element-based (width * CH).
    int32_t const srcCbLen = cupvaRasterDataFlowGetCbLen(srcHdl) * CHANNEL_NUM;
    int32_t const srcLp    = cupvaRasterDataFlowGetLinePitch(srcHdl) * CHANNEL_NUM;
    int32_t const dstLp    = cupvaRasterDataFlowGetLinePitch(dstHdl);

    // Per-channel configs on the flat planes: line pitch PLANE_LP, no circular
    // wrap (the plane itself is passed as the "circular" region so offsets
    // never wrap). Plane bases are constant across tiles, so the bases are set
    // once here.
    KNL_INIT_FUNC(planeCh0, outPlaneCh0, tmpA, tmpB, tmpC, SEPA_2X2_NUM_ELEMENTS_A, SEPA_2X2_NUM_ELEMENTS_B,
                  SEPA_2X2_NUM_ELEMENTS_C, CH_TILE_W, TILE_H_DF, PLANE_LP, CH_TILE_W,
                  (int)sizeof(BASE_TYPE) * PLANE_LP * PLANE_ROWS, planeCh0, kept_idx_1, kept_idx_2_top,
                  kept_idx_2_bottom, st_sorted_left_col, st_sorted_right_col, cfgCh0);
    median_filter_sepa_2x2_update_agen_bases(planeCh0, outPlaneCh0, planeCh0, PLANE_LP * PLANE_ROWS, PLANE_LP,
                                             CH_TILE_W, cfgCh0);

    KNL_INIT_FUNC(planeCh1, outPlaneCh1, tmpA, tmpB, tmpC, SEPA_2X2_NUM_ELEMENTS_A, SEPA_2X2_NUM_ELEMENTS_B,
                  SEPA_2X2_NUM_ELEMENTS_C, CH_TILE_W, TILE_H_DF, PLANE_LP, CH_TILE_W,
                  (int)sizeof(BASE_TYPE) * PLANE_LP * PLANE_ROWS, planeCh1, kept_idx_1, kept_idx_2_top,
                  kept_idx_2_bottom, st_sorted_left_col, st_sorted_right_col, cfgCh1);
    median_filter_sepa_2x2_update_agen_bases(planeCh1, outPlaneCh1, planeCh1, PLANE_LP * PLANE_ROWS, PLANE_LP,
                                             CH_TILE_W, cfgCh1);

    // pre/post copy-pass agens (both channels in one pass, per-tile state)
    median_filter_sepa_2x2_planar_init(srcBuf, srcLp, srcCbLen * (int)sizeof(BASE_TYPE), planeCh0, planeCh1,
                                       outPlaneCh0, outPlaneCh1, dstBuf, dstLp, cfgPlanar);

    for (int32_t i = 0; i < nTileX * nTileY; i++)
    {
        BASE_TYPE *src = (BASE_TYPE *)cupvaRasterDataFlowAcquire(srcHdl);
        BASE_TYPE *dst = (BASE_TYPE *)cupvaRasterDataFlowAcquire(dstHdl);

        cupvaModifyAgenCfgBase(&cfgPlanar.pre_src_agen_cfg, src);
        cupvaModifyAgenCfgBase(&cfgPlanar.post_dst_agen_cfg, dst);

        i2s16_deinterleave_src_exec(cfgPlanar);
        cupvaRasterDataFlowRelease(srcHdl);

        KNL_FUNC_EXEC(cfgCh0);
        KNL_FUNC_EXEC(cfgCh1);

        i2s16_interleave_dst_exec(cfgPlanar);
        cupvaRasterDataFlowRelease(dstHdl);
    }

    cupvaRasterDataFlowClose(srcHdl);
    cupvaRasterDataFlowClose(dstHdl);

    return 0;
}
