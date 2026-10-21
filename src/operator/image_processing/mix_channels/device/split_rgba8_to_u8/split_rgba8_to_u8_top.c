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

#include "split_rgba8_to_u8_knl.h"

VMEM(A, uint32_t, niter);
VMEM_RDF_UNIFIED(A, srcDFHandler);
VMEM_RDF_UNIFIED(A, dstDFHandler);

VMEM(B, uint8_t, srcBuf, RDF_DOUBLE(uint8_t, TILE_W * 4, TILE_H, 0, 0, TRANS_MODE_1));
VMEM(C, uint8_t, dst0Buf, RDF_DOUBLE(uint8_t, TILE_W, TILE_H, 0, 0, TRANS_MODE_1));
VMEM(C, uint8_t, dst1Buf, RDF_DOUBLE(uint8_t, TILE_W, TILE_H, 0, 0, TRANS_MODE_1));
VMEM(C, uint8_t, dst2Buf, RDF_DOUBLE(uint8_t, TILE_W, TILE_H, 0, 0, TRANS_MODE_1));
VMEM(C, uint8_t, dst3Buf, RDF_DOUBLE(uint8_t, TILE_W, TILE_H, 0, 0, TRANS_MODE_1));

CUPVA_VPU_MAIN()
{
    cupvaRasterDataFlowOpen(srcDFHandler, &srcBuf[0]);
    cupvaRasterDataFlowOpen(dstDFHandler, &dst0Buf[0]);
    AgenCFG agenCfgs[CFG_NUM];
    uint32_t srcTileLP = cupvaRasterDataFlowGetLinePitch(srcDFHandler);
    uint32_t dstTileLP = cupvaRasterDataFlowGetLinePitch(dstDFHandler);
    split_rgba8_to_u8_init(agenCfgs, srcTileLP, dstTileLP);

    for (uint32_t i = 0; i < niter; ++i)
    {
        uint8_t *src     = (uint8_t *)cupvaRasterDataFlowAcquire(srcDFHandler);
        uint8_t *dst0    = (uint8_t *)cupvaRasterDataFlowAcquire(dstDFHandler);
        ptrdiff_t offset = dst0 - &dst0Buf[0];

        cupvaModifyAgenCfgBase(&agenCfgs[SRC], src);
        cupvaModifyAgenCfgBase(&agenCfgs[DST_0], dst0);
        cupvaModifyAgenCfgBase(&agenCfgs[DST_1], dst1Buf + offset);
        cupvaModifyAgenCfgBase(&agenCfgs[DST_2], dst2Buf + offset);
        cupvaModifyAgenCfgBase(&agenCfgs[DST_3], dst3Buf + offset);
        split_rgba8_to_u8_exec(agenCfgs);

        cupvaRasterDataFlowRelease(srcDFHandler);
        cupvaRasterDataFlowRelease(dstDFHandler);
    }
    cupvaRasterDataFlowClose(srcDFHandler);
    cupvaRasterDataFlowClose(dstDFHandler);

    return 0;
}
