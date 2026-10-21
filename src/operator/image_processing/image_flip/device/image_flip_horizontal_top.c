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

#include "image_flip_horizontal_knl.h"
#include "image_flip_param.h"

#include <cupva_device.h>
#include <cupva_device_debug.h>

VMEM(C, int32_t, niterW);
VMEM(C, int32_t, niterH);
VMEM(B, int32_t, IMW_param); // Runtime image width parameter
VMEM(B, int32_t, IMH_param); // Runtime image height parameter
VMEM(A, int32_t, TW_param);  // Runtime tile width parameter
VMEM(A, int32_t, TH_param);  // Runtime tile height parameter

// Use compile-time constants for VMEM buffer allocation
VMEM(A, uint8_t, input_v, RDF_DOUBLE(uint8_t, TW, TH));
VMEM_RDF_UNIFIED(A, srcHdl);

VMEM(B, uint8_t, output_v, RDF_DOUBLE(uint8_t, TW, TH));
VMEM_RDF_UNIFIED(B, dstHdl);
VMEM(B, param_horizontal_flip_t, param_hflip, 1);

CUPVA_VPU_MAIN()
{

    // Start data flow processing
    cupvaRasterDataFlowOpen(srcHdl, &input_v[0]);
    cupvaRasterDataFlowOpen(dstHdl, &output_v[0]);

    // Initialize horizontal flip parameters
    param_horizontal_flip_t param_hflip;
    horizontal_flip_init(&input_v[0], &output_v[0], TW_param, TH_param, TW, &param_hflip);

    // Process tiles: iterate through height and width
    for (int i = 0; i < niterH; i++)
    {
        for (int j = 0; j < niterW; j++)
        {
            uint8_t *inTile  = (uint8_t *)cupvaRasterDataFlowAcquire(srcHdl);
            uint8_t *outTile = (uint8_t *)cupvaRasterDataFlowAcquire(dstHdl);

            // Configure input AGEN with current tile
            cupvaModifyAgenCfgBase(&param_hflip.cfgs[0], inTile);

            // Configure output AGEN with offset for horizontal flip
            // Offset accounts for vector width alignment for rightmost column access
            uint8_t *outTileWithOffset = (uint8_t *)outTile + TW - chess_elementsof(vcharx);
            cupvaModifyAgenCfgBase(&param_hflip.cfgs[1], outTileWithOffset);

            horizontal_flip_exec(&param_hflip);

            cupvaRasterDataFlowRelease(srcHdl);
            cupvaRasterDataFlowRelease(dstHdl);
        }
    }

    cupvaRasterDataFlowClose(srcHdl);
    cupvaRasterDataFlowClose(dstHdl);

    return 0;
}
