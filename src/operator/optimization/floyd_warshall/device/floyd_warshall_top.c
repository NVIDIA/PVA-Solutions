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

#include "floyd_warshall_knl.h"
#include "floyd_warshall_param.h"

#include <cupva_device.h>
#include <stdint.h>

// Handlers for DMA operations
VMEM_RDF_UNIFIED(A, inLenHdl);
VMEM_RDF_UNIFIED(A, inNextHdl);
VMEM_RDF_UNIFIED(A, outLenHdl);
VMEM_RDF_UNIFIED(A, outNextHdl);

VMEM(A, int32_t, num);
VMEM(A, AgenCFG, cfgs, CFG_NUM);

// Two buffers for lenTensor to avoid bank conflicts
VMEM(A, uint16_t, len_v, RDF_SINGLE(uint16_t, MAX_WIDTH, MAX_HEIGHT));
VMEM(B, uint16_t, len_cp, RDF_SINGLE(uint16_t, MAX_WIDTH, MAX_HEIGHT));
VMEM(C, uint16_t, next_v, RDF_SINGLE(uint16_t, MAX_WIDTH, MAX_HEIGHT));

CUPVA_VPU_MAIN()
{
    cupvaRasterDataFlowOpen(inLenHdl, &len_v[0]);
    cupvaRasterDataFlowOpen(inNextHdl, &next_v[0]);
    cupvaRasterDataFlowOpen(outLenHdl, &len_v[0]);
    cupvaRasterDataFlowOpen(outNextHdl, &next_v[0]);

    floyd_warshall_knl_init(cfgs, num);
    cupvaModifyAgenCfgBase(&cfgs[IN_LEN_CP_IJ], &len_cp[0]);
    cupvaModifyAgenCfgBase(&cfgs[OUT_LEN_IJ], &len_v[0]);
    cupvaModifyAgenCfgBase(&cfgs[OUT_LEN_CP_IJ], &len_cp[0]);
    cupvaModifyAgenCfgBase(&cfgs[OUT_NEXT_IJ], &next_v[0]);

    cupvaRasterDataFlowAcquire(inLenHdl);
    cupvaRasterDataFlowAcquire(inNextHdl);
    cupvaRasterDataFlowAcquire(outLenHdl);
    cupvaRasterDataFlowAcquire(outNextHdl);

    for (int32_t k = 0; k < num; k++)
    {
        cupvaModifyAgenCfgBase(&cfgs[IN_LEN_IK], &len_v[k]);
        cupvaModifyAgenCfgBase(&cfgs[IN_LEN_KJ], &len_v[k * num]);
        cupvaModifyAgenCfgBase(&cfgs[IN_NEXT_IK], &next_v[k]);
        floyd_warshall_knl_exec(cfgs, num);
    }

    cupvaRasterDataFlowRelease(inLenHdl);
    cupvaRasterDataFlowRelease(inNextHdl);
    cupvaRasterDataFlowRelease(outLenHdl);
    cupvaRasterDataFlowRelease(outNextHdl);

    cupvaRasterDataFlowClose(inLenHdl);
    cupvaRasterDataFlowClose(inNextHdl);
    cupvaRasterDataFlowClose(outLenHdl);
    cupvaRasterDataFlowClose(outNextHdl);

    return 0;
}
