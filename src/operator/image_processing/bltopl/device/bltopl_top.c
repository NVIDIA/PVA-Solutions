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

#include "bltopl_param.h"

#include <cupva_device.h>

VMEM(A, uint8_t, inoutTileBufferVMEM, RDF_DOUBLE(uint8_t, TILE_X, TILE_Y));

VMEM(B, bl_to_pl_host2dev_param_t, vpu_param);

VMEM(A, RasterDataFlowHandler, srcHdlY);
VMEM(B, RasterDataFlowHandler, dstHdlY);

VMEM(A, RasterDataFlowHandler, srcHdlUv);
VMEM(B, RasterDataFlowHandler, dstHdlUv);

CUPVA_VPU_MAIN()
{
    // Y plane
    cupvaRasterDataFlowTrig(srcHdlY);
    for (int32_t i = 0; i < vpu_param.num_tiles_y; i++)
    {
        cupvaRasterDataFlowSync(srcHdlY);
        // no need to consider an extra trigger when i == (vpu_param.num_tiles_y-1),
        // since the API itself can handle this.
        cupvaRasterDataFlowTrig(srcHdlY);

        // cannot use cupvaRasterDataFlowAcquire and cupvaRasterDataFlowRelease
        // since DMA dataflow's fine control is needed, otherwise will need
        // distinguished input and output VMEM buffer and VPU needs extra memcpy
        cupvaRasterDataFlowTrig(dstHdlY);
        cupvaRasterDataFlowSync(dstHdlY);
    }

    // UV plane, same as to the Y plane except for the number of tiles
    cupvaRasterDataFlowTrig(srcHdlUv);
    for (int32_t i = 0; i < vpu_param.num_tiles_uv; i++)
    {
        cupvaRasterDataFlowSync(srcHdlUv);
        cupvaRasterDataFlowTrig(srcHdlUv);

        cupvaRasterDataFlowTrig(dstHdlUv);
        cupvaRasterDataFlowSync(dstHdlUv);
    }

    return 0;
}
