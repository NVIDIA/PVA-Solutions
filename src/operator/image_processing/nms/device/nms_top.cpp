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

#include "nms_param.h"

#include <pvaAplNmsVpu.hpp>

#include <cupva_device.h>

// Bank assignment is dictated by the primitive's agen ports: it reads the input through port A, the
// scratch through port C and writes the output through port B.
//
// The source uses a circular layout so the horizontal halo is shared between neighboring tiles
// instead of being re-fetched per tile. Tiles then sit at varying offsets in the ring and their rows
// can straddle its end, so the primitive is given the ring base and length and wraps both of its
// source agens.
VMEM(A, int32_t, srcBuffer, RDF_CIRCULAR(int32_t, TILE_WIDTH, TILE_HEIGHT, NMS_RADIUS, NMS_RADIUS));
VMEM(B, int32_t, dstBuffer, RDF_DOUBLE(int32_t, TILE_WIDTH, TILE_HEIGHT));
VMEM(C, int32_t, scratchBuffer, NMS_SCRATCH_ELEMS);

// NMS_SCRATCH_ELEMS restates the primitive's sizing rule as a macro, so pin the two together here:
// if ScratchElements changes, an under-sized scratchBuffer would be written past at run time.
static_assert(NMS_SCRATCH_ELEMS >= pvaApl::NmsVpu<NMS_SIZE, int32_t>::ScratchElements(SRC_TILE_WIDTH, SRC_TILE_HEIGHT),
              "NMS_SCRATCH_ELEMS is smaller than pvaApl::NmsVpu requires for this tile geometry");

VMEM(C, int32_t, niterX);
VMEM(C, int32_t, niterY);

VMEM_RDF_UNIFIED(C, srcHandler);
VMEM_RDF_UNIFIED(C, dstHandler);

CUPVA_VPU_MAIN()
{
    int32_t srcLinePitch = cupvaRasterDataFlowGetLinePitch(srcHandler);
    int32_t dstLinePitch = cupvaRasterDataFlowGetLinePitch(dstHandler);
    int32_t srcCbLen     = cupvaRasterDataFlowGetCbLen(srcHandler);

    // Agen shapes depend only on the tile geometry, so Init is hoisted out of the tile loop and only
    // the base pointers are re-bound per tile.
    pvaApl::NmsVpu<NMS_SIZE, int32_t> nms;
    nms.Init(&srcBuffer[0], SRC_TILE_WIDTH, SRC_TILE_HEIGHT, srcLinePitch, dstLinePitch, &dstBuffer[0], &srcBuffer[0],
             srcCbLen, &scratchBuffer[0]);

    cupvaRasterDataFlowOpen(srcHandler, &srcBuffer[0]);
    cupvaRasterDataFlowOpen(dstHandler, &dstBuffer[0]);

    for (int32_t i = 0; i < niterY; i++)
    {
        for (int32_t j = 0; j < niterX; j++)
        {
            // The acquired source pointer addresses the top-left of the padded tile somewhere inside
            // the ring, which is exactly what the primitive expects.
            int32_t *srcTile = static_cast<int32_t *>(cupvaRasterDataFlowAcquire(srcHandler));
            int32_t *dstTile = static_cast<int32_t *>(cupvaRasterDataFlowAcquire(dstHandler));

            nms.Update(srcTile, dstTile);
            nms.Execute();

            cupvaRasterDataFlowRelease(srcHandler);
            cupvaRasterDataFlowRelease(dstHandler);
        }
    }

    cupvaRasterDataFlowClose(srcHandler);
    cupvaRasterDataFlowClose(dstHandler);

    return 0;
}
