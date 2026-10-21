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

#include "distance_transform_dev_param.h"
#include "distance_transform_knl.h"

#include <cupva_device.h>
#include <cupva_device/gatherscatter_dataflow.h>

#define EVEN 0
#define ODD 1

// overlay vars
struct Stage2BankC
{
    uint16_t dist[TW * MAX_IMAGE_HEIGHT / 2];
    uint16_t label[TW * MAX_IMAGE_HEIGHT / 2];
};

union BankB
{
    uint16_t buf_b[(MAX_IMAGE_WIDTH + 1) * TH];
    uint16_t v_arr[TW * MAX_IMAGE_HEIGHT];
};

union BankC
{
    uint16_t buf_c[(MAX_IMAGE_WIDTH + 1) * TH];
    Stage2BankC stage2BankC;
};

VMEM(B, BankB, bankBData);
VMEM(C, BankC, bankCData);
CUPVA_EXPORT(bankBData.buf_b, buf_b);
CUPVA_EXPORT(bankCData.buf_c, buf_c);
CUPVA_EXPORT(bankCData.stage2BankC.dist, dist);
CUPVA_EXPORT(bankCData.stage2BankC.label, label);

// stage 1 vars
VMEM(A, int32_t, n_tile_row);
VMEM_GSDF_HANDLER(A, read_hdl, 1);
VMEM_GSDF_HANDLER(A, write_hdl, 1);

// stage 2 vars
VMEM(A, uint16_t, hd_in, RDF_SINGLE(uint16_t, TW, MAX_IMAGE_HEIGHT));
VMEM(B, uint16_t, sqrt_tbl, 65 * 32);
VMEM(B, int32_t, iw);
VMEM(B, int32_t, ih);
VMEM(B, int32_t, n_tile_col);
VMEM_RDF_UNIFIED(B, read_trigger);
VMEM_RDF_UNIFIED(B, write_trigger0);
VMEM_RDF_UNIFIED(B, write_trigger1);
VMEM(B, uint16_t, max_dist);
VMEM(B, int32_t, transpose_lp);
VMEM(B, int32_t, height_rounded);
VMEM(B, int32_t, width_rounded);

CUPVA_VPU_MAIN()
{
    uint16_t *buf_b = bankBData.buf_b;
    uint16_t *buf_c = bankCData.buf_c;
    uint16_t *v_arr = bankBData.v_arr;
    uint16_t *dist  = bankCData.stage2BankC.dist;
    uint16_t *label = bankCData.stage2BankC.label;

    // ------------------------------------------------------------------------------------------------------------- //
    // STAGE1                                                                                                        //
    // ------------------------------------------------------------------------------------------------------------- //
    VX_USHORT_TILE tile[2];
    param_dt_stage1_t param[2];

    // ping buffer
    tile[0].tp = buf_b;
    tile[0].tw = iw;
    tile[0].th = TH;
    tile[0].lp = transpose_lp;

    // pong buffer
    tile[1].tp = buf_c;
    tile[1].tw = iw;
    tile[1].th = TH;
    tile[1].lp = transpose_lp;

    // ----------------------------------+-------------------------------------+------------------------------------ //
    // read even tile row                | initial kernel                      |                                     //
    // ----------------------------------+-------------------------------------+------------------------------------ //
    int32_t x         = 0;
    int32_t y_read    = 0;
    int32_t y_write   = 0;
    int32_t ping_pong = 0;
    uint16_t *dma_tile_ptr;

    cupvaGSDFOpen(read_hdl, buf_b, transpose_lp, transpose_lp * TH);
    // the padding is need when the image width is not multiple of 4
    // have to use below API to deal with padding calculation correctly from device side
    cupvaGSDFUpdateTiles(read_hdl, &x, &y_read, 1);
    cupvaGSDFTrig(read_hdl);
    y_read += TH;
    cupvaGSDFOpen(write_hdl, buf_b, transpose_lp, transpose_lp * TH);

    distance_transform_stage1_init(&tile[EVEN], width_rounded, iw, &param[EVEN]);
    distance_transform_stage1_init(&tile[ODD], width_rounded, iw, &param[ODD]);

    cupvaGSDFSync(read_hdl);

    for (int32_t i = 0; i < n_tile_row; i++)
    {
        // dma buffer pointer reverts the current ping_pong index which is marked for compute
        dma_tile_ptr = ping_pong ? buf_b : buf_c;

        if (i)
        {
            cupvaGSDFUpdateOffsets2D(write_hdl, &x, &y_write);
            cupvaGSDFUpdateVmemAddr(write_hdl, dma_tile_ptr, transpose_lp, transpose_lp * TH);
            cupvaGSDFTrig(write_hdl);
            y_write += TH;
        }

        distance_transform_stage1_l2r(&param[ping_pong]);

        if (i)
        {
            cupvaGSDFSync(write_hdl);
        }

        if (i < n_tile_row - 1)
        {
            cupvaGSDFUpdateTiles(read_hdl, &x, &y_read, 1);
            cupvaGSDFUpdateVmemAddr(read_hdl, dma_tile_ptr, transpose_lp, transpose_lp * TH);
            cupvaGSDFTrig(read_hdl);
            y_read += TH;
        }

        distance_transform_stage1_r2l(&param[ping_pong]);

        if (i < n_tile_row - 1)
        {
            cupvaGSDFSync(read_hdl);
        }

        ping_pong ^= 1;
    }

    dma_tile_ptr = ping_pong ? buf_b : buf_c;
    cupvaGSDFUpdateOffsets2D(write_hdl, &x, &y_write);
    cupvaGSDFUpdateVmemAddr(write_hdl, dma_tile_ptr, transpose_lp, transpose_lp * TH);
    cupvaGSDFTrig(write_hdl);
    cupvaGSDFSync(write_hdl);
    cupvaGSDFClose(read_hdl);
    cupvaGSDFClose(write_hdl);

    // ------------------------------------------------------------------------------------------------------------- //
    // STAGE2                                                                                                        //
    // ------------------------------------------------------------------------------------------------------------- //
    dvshortx k, k2;
    dvshortx start1, end1, start2, end2;

    // ----------------------------------+-------------------------------------+------------------------------------ //
    // read tile col                     |                                     |                                     //
    // ----------------------------------+-------------------------------------+------------------------------------ //
    cupvaRasterDataFlowOpen(read_trigger, hd_in);
    cupvaRasterDataFlowOpen(write_trigger0, dist);
    cupvaRasterDataFlowOpen(write_trigger1, label);

    k2.lo     = replicateh(0);
    k2.hi     = replicateh(0);
    start1.lo = replicateh(0);
    start1.hi = replicateh(0);
    end1.lo   = replicateh(height_rounded / 2 - 1);
    end1.hi   = replicateh(height_rounded / 2 - 1);
    start2.lo = replicateh(height_rounded / 2);
    start2.hi = replicateh(height_rounded / 2);
    end2.lo   = replicateh(ih - 1);
    end2.hi   = replicateh(ih - 1);

    for (int32_t i = 0; i < n_tile_col; i++)
    {
        // ----------------------------------+-------------------------------------+------------------------------------ //
        //                                   | sweep1 build v_arr                  |                                     //
        // ----------------------------------+-------------------------------------+------------------------------------ //
        uint16_t *hd_in_tile = (uint16_t *)cupvaRasterDataFlowAcquire(read_trigger);
        uint16_t *dist_tile  = (uint16_t *)cupvaRasterDataFlowAcquire(write_trigger0);
        uint16_t *label_tile = (uint16_t *)cupvaRasterDataFlowAcquire(write_trigger1);
        distance_transform_stage2_sweep1(TW, ih, iw, ih, max_dist, hd_in_tile, v_arr, dist_tile, label_tile, sqrt_tbl,
                                         &k);

        // ----------------------------------+-------------------------------------+------------------------------------ //
        //                                   | sweep2 generate first half          |                                     //
        // ----------------------------------+-------------------------------------+------------------------------------ //
        distance_transform_stage2_sweep2(TW, ih, iw, ih, max_dist, hd_in_tile, v_arr, dist_tile, label_tile, sqrt_tbl,
                                         &k, &k2, &start1, &end1);

        // ----------------------------------+-------------------------------------+------------------------------------ //
        //                                   |                                     | write first half dist and label     //
        // ----------------------------------+-------------------------------------+------------------------------------ //
        cupvaRasterDataFlowRelease(write_trigger0);
        cupvaRasterDataFlowRelease(write_trigger1);
        dist_tile  = (uint16_t *)cupvaRasterDataFlowAcquire(write_trigger0);
        label_tile = (uint16_t *)cupvaRasterDataFlowAcquire(write_trigger1);

        // ----------------------------------+-------------------------------------+------------------------------------ //
        //                                   | sweep2 generate second half         |                                     //
        // ----------------------------------+-------------------------------------+------------------------------------ //
        distance_transform_stage2_sweep2(TW, ih, iw, ih, max_dist, hd_in_tile, v_arr, dist_tile, label_tile, sqrt_tbl,
                                         &k, &k2, &start2, &end2);
        k2.lo = replicateh(0);
        k2.hi = replicateh(0);

        // ----------------------------------+-------------------------------------+------------------------------------ //
        // read next tile col                |                                     | write second half dist and label    //
        // ----------------------------------+-------------------------------------+------------------------------------ //
        cupvaRasterDataFlowRelease(read_trigger);
        cupvaRasterDataFlowRelease(write_trigger0);
        cupvaRasterDataFlowRelease(write_trigger1);
    }

    cupvaRasterDataFlowClose(read_trigger);
    cupvaRasterDataFlowClose(write_trigger0);
    cupvaRasterDataFlowClose(write_trigger1);

    return 0;
}
