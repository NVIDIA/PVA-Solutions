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

#include "median_filter_common.h"
#include "median_filter_knl_3x3.h"
#include "median_filter_param.h"
#define KNL_FUNC_EXEC median_filter_3x3_exec
VMEM(A, BASE_TYPE, tile_min, INTERM_TILE_SIZE);
VMEM(B, BASE_TYPE, tile_mid, INTERM_TILE_SIZE);
VMEM(C, BASE_TYPE, tile_max, INTERM_TILE_SIZE);

VMEM(B, BASE_TYPE, srcBuf, RDF_CIRCULAR(BASE_TYPE, TILE_W, TILE_H_PIX, HALO_X, HALO_Y));
VMEM(C, BASE_TYPE, dstBuf, RDF_DOUBLE(BASE_TYPE, TILE_W, TILE_H_PIX));

// Handlers for cupva DMA operations
VMEM_RDF_UNIFIED(A, srcHdl);
VMEM_RDF_UNIFIED(A, dstHdl);

VMEM(A, int32_t, nTileX);
VMEM(A, int32_t, nTileY);

VMEM(A, MedianFilterConfig, cfg);

CUPVA_VPU_MAIN()
{
    int32_t srcCbLen = cupvaRasterDataFlowGetCbLen(srcHdl);
    int32_t srcLp    = cupvaRasterDataFlowGetLinePitch(srcHdl);
    int32_t dstLp    = cupvaRasterDataFlowGetLinePitch(dstHdl);

    // for interleaved data type, need to apply channel number to the srcCbLen and srcLinePitch
    srcCbLen *= CHANNEL_NUM;
    srcLp *= CHANNEL_NUM;

    // init tmp pointers
    BASE_TYPE *tmpPtr[KNL_W];
    tmpPtr[0] = &tile_min[0];
    tmpPtr[1] = &tile_mid[0];
    tmpPtr[2] = &tile_max[0];

    cupvaRasterDataFlowOpen(srcHdl, srcBuf);
    cupvaRasterDataFlowOpen(dstHdl, dstBuf);

    // init median filter configurations
    median_filter_nxm_init(&srcBuf[0], &dstBuf[0], tmpPtr, TILE_W, TILE_H_PIX, srcLp, dstLp,
                           srcCbLen * sizeof(BASE_TYPE), &srcBuf[0], &cfg);

    for (int32_t i = 0; i < nTileX * nTileY; i++)
    {
        void *src = cupvaRasterDataFlowAcquire(srcHdl);
        void *dst = cupvaRasterDataFlowAcquire(dstHdl);

        // modify agen base
        cupvaModifyAgenCfgBase(&cfg._in_agen_cfg, src);
        cupvaModifyAgenCfgBase(&cfg._out_agen_cfg, dst);

        // execute median filter kernel function
        KNL_FUNC_EXEC(&cfg);

        cupvaRasterDataFlowRelease(srcHdl);
        cupvaRasterDataFlowRelease(dstHdl);
    }

    cupvaRasterDataFlowClose(srcHdl);
    cupvaRasterDataFlowClose(dstHdl);

    return 0;
}
