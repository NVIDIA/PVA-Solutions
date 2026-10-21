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

#include "median_filter_common_sepa_2x2.h"
#include "median_filter_param.h"

#if defined(KNL_7x7)
#    include "median_filter_knl_sepa_2x2_7x7.h"

#elif defined(KNL_5x7)
#    include "median_filter_knl_sepa_2x2_5x7.h"

#elif defined(KNL_5x5)
#    include "median_filter_knl_sepa_2x2_5x5.h"

#endif

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

VMEM(A, MedianFilterSepa2x2Config, cfg);

CUPVA_VPU_MAIN()
{
    cupvaRasterDataFlowOpen(srcHdl, srcBuf);
    cupvaRasterDataFlowOpen(dstHdl, dstBuf);

    int32_t const srcCbLen = cupvaRasterDataFlowGetCbLen(srcHdl);
    int32_t const srcLp    = cupvaRasterDataFlowGetLinePitch(srcHdl);
    int32_t const dstLp    = cupvaRasterDataFlowGetLinePitch(dstHdl);

    // init median filter configurations
    KNL_INIT_FUNC(srcBuf, dstBuf, tmpA, tmpB, tmpC, SEPA_2X2_NUM_ELEMENTS_A, SEPA_2X2_NUM_ELEMENTS_B,
                  SEPA_2X2_NUM_ELEMENTS_C, TILE_W, TILE_H_DF, srcLp, dstLp, srcCbLen * sizeof(BASE_TYPE), srcBuf,
                  kept_idx_1, kept_idx_2_top, kept_idx_2_bottom, st_sorted_left_col, st_sorted_right_col, cfg);

    for (int32_t i = 0; i < nTileX * nTileY; i++)
    {
        BASE_TYPE *src = (BASE_TYPE *)cupvaRasterDataFlowAcquire(srcHdl);
        BASE_TYPE *dst = (BASE_TYPE *)cupvaRasterDataFlowAcquire(dstHdl);

        median_filter_sepa_2x2_update_agen_bases(src, dst, srcBuf, srcCbLen, srcLp, dstLp, cfg);

        // execute median filter kernel function
        KNL_FUNC_EXEC(cfg);

        cupvaRasterDataFlowRelease(srcHdl);

        cupvaRasterDataFlowRelease(dstHdl);
    }

    cupvaRasterDataFlowClose(srcHdl);
    cupvaRasterDataFlowClose(dstHdl);

    return 0;
}
