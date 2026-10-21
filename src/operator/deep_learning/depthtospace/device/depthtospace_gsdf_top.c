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

#include "depthtospace_gsdf_param.h"

#include <cupva_device.h>

#define MAX_BLK_SZ (4) // only 2,3,4 are supported
#define MAX_GSDF_HDL_SZ (4)

// the maximum output tile depends on (128KiB-max(pre-allocated global/stack space, separate variables like trigger, vpu_param))/2
VMEM(A, char, scratch_a, SCRATCH_SIZE);
VMEM(B, char, scratch_b, SCRATCH_SIZE);
VMEM(C, char, scratch_c, SCRATCH_SIZE);

VMEM_GSDF_HANDLER(C, srcHdlT16, MAX_BLK_SZ *MAX_BLK_SZ);
VMEM_GSDF_HANDLER(B, srcHdlT8, 8);
VMEM_GSDF_HANDLER(B, srcHdlT4_0, 4);
VMEM_GSDF_HANDLER(B, srcHdlT4_1, 4);

VMEM_GSDF_HANDLER(C, dstHdl, 1);

VMEM_RDF_UNIFIED(A, tblHdl);

VMEM(B, depth_to_space_host2dev_param_t, vpu_param);
VMEM(C, depth_to_space_buffer_info_t, buffer_info);

// Generate blk_sz^2 intput/output offsets array, each corresponds to the upper left point of an input/output tile
// iter: 0~num_tiles-1, num_tiles = oc * ((iw/tile_x) * (ih/tile_y))
// offsets1DIn size = blk_sz^2
// offsets1DOut size = 1
//
// iw/ih are LOGICAL extents (what the tile grid counts); in_row_stride/in_chan_stride
// are ADDRESS strides (what an offset accumulates from). They coincide only when tight.
static void updateTilesWithOffsets(depth_to_space_host2dev_param_t const *vpu_param, int iter, int32_t *offsets1DIn,
                                   int32_t *offsets1DOut)
{
    const int32_t tiles_per_hw = vpu_param->iw / vpu_param->tile_x * vpu_param->ih / vpu_param->tile_y;
    int32_t xCoords            = (iter % tiles_per_hw) % (vpu_param->iw / vpu_param->tile_x) * vpu_param->tile_x;
    int32_t yCoords            = (iter % tiles_per_hw) / (vpu_param->iw / vpu_param->tile_x) * vpu_param->tile_y;
    int32_t ic_start           = iter / tiles_per_hw;
    int32_t ic_stride          = vpu_param->in_chan_stride;

    int32_t xCoordsOut = xCoords * vpu_param->blk_sz;
    int32_t yCoordsOut = yCoords * vpu_param->blk_sz;
    int32_t offsetsOut = ic_start * (vpu_param->ow * vpu_param->oh);
    offsets1DOut[0]    = offsetsOut + yCoordsOut * vpu_param->ow + xCoordsOut;

    for (int i = 0; i < vpu_param->blk_sz * vpu_param->blk_sz; i++)
    {
        int32_t offsetsIn = ic_start * ic_stride + i * vpu_param->oc * ic_stride;
        offsets1DIn[i]    = offsetsIn + yCoords * vpu_param->in_row_stride + xCoords;
    }
}

CUPVA_VPU_MAIN()
{
    char *scratch_mems[3] = {scratch_a, scratch_b, scratch_c};

    cupvaRasterDataFlowOpen(tblHdl, scratch_mems[buffer_info.tbl_id] + buffer_info.tbl_ofst);
    uint16_t *tbl = (uint16_t *)cupvaRasterDataFlowAcquire(tblHdl);
    cupvaRasterDataFlowRelease(tblHdl);
    cupvaRasterDataFlowClose(tblHdl);

    uint8_t *dut_in_base    = (uint8_t *)(scratch_mems[buffer_info.dut_in_id] + buffer_info.dut_in_ofst);
    uint8_t *dut_in_arr[2]  = {dut_in_base, dut_in_base + buffer_info.dut_in_sz / 2};
    uint8_t *dut_out_base   = (uint8_t *)(scratch_mems[buffer_info.dut_out_id] + buffer_info.dut_out_ofst);
    uint8_t *dut_out_arr[2] = {dut_out_base, dut_out_base + buffer_info.dut_out_sz / 2};

    const int num_tiles  = vpu_param.num_tiles;
    const int tile_x     = vpu_param.tile_x;
    const int tile_y     = vpu_param.tile_y;
    const int blk_sz     = vpu_param.blk_sz;
    const int numGsdfHdl = vpu_param.numGsdfHdl;

    int blksPerGsdf;
    GSDFHandler *srcHdlArr[MAX_GSDF_HDL_SZ];
    if (numGsdfHdl == 4)
    {
        srcHdlArr[0] = srcHdlT16;
        srcHdlArr[1] = srcHdlT8;
        srcHdlArr[2] = srcHdlT4_0;
        srcHdlArr[3] = srcHdlT4_1;
        blksPerGsdf  = 1;
    }
    else if (numGsdfHdl == 3)
    {
        srcHdlArr[0] = srcHdlT8;
        srcHdlArr[1] = srcHdlT4_0;
        srcHdlArr[2] = srcHdlT4_1;
        blksPerGsdf  = 3;
    }
    else
    {
        srcHdlArr[0] = srcHdlT16;
        blksPerGsdf  = blk_sz * blk_sz;
    }

    // initial pointer values, choose between pingpong input/output
    uint8_t *dut_in      = dut_in_arr[0];
    uint8_t *dut_in_nxt  = dut_in_arr[1];
    uint8_t *dut_out     = dut_out_arr[0];
    uint8_t *dut_out_nxt = dut_out_arr[1];

    int32_t offsets_input[MAX_BLK_SZ * MAX_BLK_SZ];
    int32_t offsets_output[1];
    int32_t offsets_output_nxt[1];
    updateTilesWithOffsets(&vpu_param, 0, offsets_input, offsets_output);
    for (int j = 0; j < numGsdfHdl; j++)
    {
        cupvaGSDFOpen(srcHdlArr[j], dut_in + j * blksPerGsdf * tile_x * tile_y, tile_x, tile_x * tile_y);

        // update the VMEM address for the pong buffer
        cupvaGSDFCycleBuffers(srcHdlArr[j]);
        cupvaGSDFUpdateVmemAddr(srcHdlArr[j], dut_in_nxt + j * blksPerGsdf * tile_x * tile_y, tile_x, tile_x * tile_y);
        cupvaGSDFCycleBuffers(srcHdlArr[j]);

        cupvaGSDFUpdateOffsets1D(srcHdlArr[j], &offsets_input[blksPerGsdf * j]);
        cupvaGSDFTrig(srcHdlArr[j]);
    }

    cupvaGSDFOpen(dstHdl, dut_out, blk_sz * tile_x, blk_sz * blk_sz * tile_x * tile_y);
    // update the VMEM address for the pong buffer
    cupvaGSDFCycleBuffers(dstHdl);
    cupvaGSDFUpdateVmemAddr(dstHdl, dut_out_nxt, blk_sz * tile_x, blk_sz * blk_sz * tile_x * tile_y);
    cupvaGSDFCycleBuffers(dstHdl);

    CupvaSampler sampler;
    // DLUT's table related fields
    CupvaSamplerInput1D input = {
        .data           = dut_in,
        .type           = SAMPLER_INPUT_TYPE_U8,
        .length         = (uint16_t)(blk_sz * blk_sz * tile_x * tile_y),
        .outOfRangeMode = SAMPLER_OUT_OF_RANGE_CONSTANT,
    };
    // DLUT's index related fields
    CupvaSamplerIndices1D indices = {
        .data               = tbl,
        .type               = SAMPLER_INDEX_TYPE_U16,
        .width              = (uint16_t)(blk_sz * blk_sz * tile_x * tile_y),
        .height             = 1U,
        .fractionalHandling = SAMPLER_FRAC_HANDLING_TRUNCATE,
    };
    // DLUT's output related fields
    CupvaSamplerOutput output = {
        .data      = dut_out,
        .transMode = TRANS_MODE_NONE,
    };
    cupvaSamplerSetup(&sampler, &input, &indices, &output);

    for (int i = 0; i < num_tiles; i++)
    {
        for (int j = 0; j < numGsdfHdl; j++)
        {
            cupvaGSDFSync(srcHdlArr[j]);
        }

        // If it is not the last tile, need to fetch the next tile from DRAM to VMEM
        if (i != (num_tiles - 1))
        {
            updateTilesWithOffsets(&vpu_param, i + 1, offsets_input, offsets_output_nxt);
            for (int j = 0; j < numGsdfHdl; j++)
            {
                cupvaGSDFUpdateOffsets1D(srcHdlArr[j], &offsets_input[blksPerGsdf * j]);
                cupvaGSDFTrig(srcHdlArr[j]);
            }
        }

        cupvaSamplerUpdateAddrs(&sampler, dut_in, tbl, dut_out);
        cupvaSamplerStart(&sampler);
        cupvaSamplerWait();

        // If it is not the first tile, need to wait until the last tile has been fetched from VMEM to DRAM
        if (i != 0)
        {
            cupvaGSDFSync(dstHdl);
        }

        cupvaGSDFUpdateOffsets1D(dstHdl, offsets_output);
        cupvaGSDFTrig(dstHdl);

        // Update for the next iteration's use
        dut_in            = dut_in_arr[(i + 1) % 2];
        dut_out           = dut_out_arr[(i + 1) % 2];
        offsets_output[0] = offsets_output_nxt[0];
    }

    cupvaGSDFSync(dstHdl);
    cupvaGSDFClose(dstHdl);

    for (int j = 0; j < numGsdfHdl; j++)
    {
        cupvaGSDFClose(srcHdlArr[j]);
    }

    return 0;
}
