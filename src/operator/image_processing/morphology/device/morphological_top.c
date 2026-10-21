/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morphological_knl.h"
#include "morphological_param.h"

#include <cupva_device.h>

// Handlers for DMA operations
VMEM_RDF_UNIFIED(A, srcHdl);
VMEM_RDF_UNIFIED(A, dstHdl);

// Agen Config and tile loop
VMEM(A, int32_t, niterOuter);
VMEM(A, AgenCFG, agenCfg0);
VMEM(A, AgenCFG, agenCfg1);

// Tile buffers
VMEM(B, BASE_TYPE, src_v, RDF_CIRCULAR_ROWMAJOR(BASE_TYPE, TILE_W, TILE_H, HALO_X, HALO_Y));
VMEM(C, BASE_TYPE, dst_v, RDF_CIRCULAR_ROWMAJOR(BASE_TYPE, TILE_W, TILE_H, 0, 0));

CUPVA_VPU_MAIN()
{
    int32_t srcCbLen     = cupvaRasterDataFlowGetCbLen(srcHdl);
    int32_t srcLinePitch = cupvaRasterDataFlowGetLinePitch(srcHdl);
    int32_t dstLinePitch = cupvaRasterDataFlowGetLinePitch(dstHdl);
    int32_t srcCbSize    = srcCbLen * sizeof(BASE_TYPE);

    // initialize IO Agens
    morphological_filter_init(src_v, srcLinePitch, dst_v, dstLinePitch, TILE_W, TILE_H, KWIDTH, KHEIGHT, src_v,
                              srcCbSize, &agenCfg0, &agenCfg1);

    //init RDF unified
    cupvaRasterDataFlowOpen(srcHdl, src_v);
    cupvaRasterDataFlowOpen(dstHdl, dst_v);

    for (int32_t i = 0; i < niterOuter; i++)
    {
        // acquire tile for Read/Write, blocking call
        void *pSrcTile = cupvaRasterDataFlowAcquire(srcHdl);
        void *pDstTile = cupvaRasterDataFlowAcquire(dstHdl);

        // call vector kernel
        cupvaModifyAgenCfgBase(&agenCfg0, pSrcTile);
        cupvaModifyAgenCfgBase(&agenCfg1, pDstTile);

        // run appropriate morphological filter kernel
        morphological_filter_exec(&agenCfg0, &agenCfg1);

        // release tile for write and fetch next tile
        cupvaRasterDataFlowRelease(srcHdl);
        cupvaRasterDataFlowRelease(dstHdl);
    }
    cupvaRasterDataFlowClose(srcHdl);
    cupvaRasterDataFlowClose(dstHdl);

    return 0;
}
