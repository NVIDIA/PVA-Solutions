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

#include "image_flip_param.h"
#include "image_flip_vertical_knl.h"

#include <cupva_device.h>
#include <cupva_device_debug.h>

VMEM(C, int32_t, niterW);
VMEM(C, int32_t, niterH);
VMEM(A, int32_t, IMW_param); // Runtime image width parameter
VMEM(A, int32_t, IMH_param); // Runtime image height parameter
VMEM(A, int32_t, TW_param);  // Runtime tile width parameter
VMEM(A, int32_t, TH_param);  // Runtime tile height parameter
VMEM(A, int32_t, flip_mode); // Runtime flip mode parameter

// Use compile-time constants for VMEM buffer allocation
VMEM(A, uint8_t, input_v, RDF_DOUBLE(uint8_t, TW, TH));
VMEM_RDF_UNIFIED(A, srcHdl1);

VMEM(B, uint8_t, output_v, RDF_DOUBLE(uint8_t, TW, TH));
VMEM_RDF_UNIFIED(B, dstHdl1);

VMEM(C, param_vertical_flip_t, param_vflip, 1);

CUPVA_VPU_MAIN()
{

    // Start data flow processing
    cupvaRasterDataFlowOpen(srcHdl1, &input_v[0]);
    cupvaRasterDataFlowOpen(dstHdl1, &output_v[0]);

    // Initialize vertical flip parameters
    param_vertical_flip_t param_vflip;
    vertical_flip_init(&input_v[0], &output_v[0], TW_param, TH_param, TW, &param_vflip);

    for (int i = 0; i < niterH; i++)
    {
        for (int j = 0; j < niterW; j++)
        {

            uint8_t *inTile  = (uint8_t *)cupvaRasterDataFlowAcquire(srcHdl1);
            uint8_t *outTile = (uint8_t *)cupvaRasterDataFlowAcquire(dstHdl1);

            cupvaModifyAgenCfgBase(&param_vflip.cfgs[0], inTile); // Input configuration
            // Configure output AGEN with offset for vertical flip
            // Offset accounts for bottom row access
            uint8_t *outTileWithOffset = (uint8_t *)outTile + (TH_param - 1) * TW;
            cupvaModifyAgenCfgBase(&param_vflip.cfgs[1], outTileWithOffset); // Output configuration
            vertical_flip_exec(&param_vflip);

            cupvaRasterDataFlowRelease(srcHdl1);
            cupvaRasterDataFlowRelease(dstHdl1);
        }
    }

    cupvaRasterDataFlowClose(srcHdl1);
    cupvaRasterDataFlowClose(dstHdl1);

    return 0;
}