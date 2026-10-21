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

#include "canny_edge_detector_ccl_funcs.h"
#include "canny_edge_detector_filters.h"
#include "canny_edge_detector_funcs.h"
#include "canny_edge_detector_knls.h"
#include "canny_edge_detector_params.h"

#include <cupva_device.h>
#include <cupva_device_debug.h>

struct IntensityBankA
{
    int8_t edge_dir[RDF_DOUBLE(int8_t, TW, TH)];
    int16_t gx_tmp[LINEPITCH * BLKH];
    int16_t gx[LINEPITCH_T * TW];
    struct Flt4x2Para flt4x2_para;
    struct Flt4Para flt4_para[2];
    struct EdgePara edge_para[2];
    struct Stage1BufPtrs stage1_buf_ptrs;
};

struct IntensityBankB
{
    int8_t coef_r[128];
    int32_t intensity[RDF_DOUBLE(int32_t, TW, TH)];
    int16_t gy[LINEPITCH_T * TW];
};

struct IntensityBankC
{
    uint8_t image[RDF_CIRCULAR_ROWMAJOR(uint8_t, TW, TH, MAX_HALO, MAX_HALO)];

    int16_t gy_tmp[LINEPITCH * BLKH];
};

struct NMSBankA
{
    uint8_t edge_type[RDF_DOUBLE(uint8_t, TW_NMS, TH_NMS)];
    struct EdgePara edge_para;
};

struct NMSBankB
{
    int8_t edge_dir[RDF_DOUBLE(int8_t, TW_NMS, TH_NMS)];
};

struct NMSBankC
{
    int32_t intensity[RDF_CIRCULAR_ROWMAJOR(int32_t, TW_NMS, TH_NMS, 1, 1)];
};

struct CCLBankA
{
    uint8_t inBuffer[RDF_CIRCULAR(uint8_t, CCL_TILEW, CCL_TILEH, CCL_HALOX, 0)];
    uint16_t inIntermediateBuffer[RDF_CIRCULAR(uint16_t, CCL_TILEW, CCL_TILEH, CCL_HALOX, 0)];
    // Buffer to hold equivalent pairs for a single tile. Sized to accommodate the worst-case scenario
    // where every pixel could potentially have an equivalent pair, though this is theoretically impossible.
    uint16_t equivalentPairs[CCL_TILEW * CCL_TILEH * 4];
    // Buffer for tracking labels between tile rows
    uint16_t previousRowLabels[CCL_MAX_IMG_WIDTH + 2 * CCL_HALOX];
    // Reformatted labels - i.e. skew removed
    uint16_t reformattedLabels[CCL_TILEW * CCL_TILEH];
};

struct CCLBankB
{
    uint16_t outIntermediateBuffer[RDF_DOUBLE(uint16_t, CCL_TILEW, CCL_TILEH, 0, 0, TRANS_MODE_1)];
    uint8_t outEdgeMap[RDF_DOUBLE(uint8_t, CCL_TILEW, CCL_TILEH)];
    // Reformatted input to support transposed parallelogram access with a skew of -2 per row
    uint8_t reformatInput[CCL_REFORMAT_TILE_LP * CCL_REFORMAT_TILE_H];
    // Temporary buffer for collapsing the label tree
    uint16_t tempCollapseBuffer[CCL_COLLAPSE_TILESIZE];
    // Final labels - i.e. labels with root labels
    uint16_t outLabels[CCL_TILEW * CCL_TILEH];
    struct Stage3BufPtrs bufPtrs;
};

struct CCLBankC
{
    uint16_t labelLut[CCL_MAX_LABELS];
};

union BankA
{
    IntensityBankA stage1;
    NMSBankA stage2;
    CCLBankA stage3;
};
VMEM(A, BankA, bankAData);

union BankB
{
    IntensityBankB stage1;
    NMSBankB stage2;
    CCLBankB stage3;
};
VMEM(B, BankB, bankBData);

union BankC
{
    IntensityBankC stage1;
    NMSBankC stage2;
    CCLBankC stage3;
};
VMEM(C, BankC, bankCData);

CUPVA_EXPORT(bankAData.stage1.edge_dir, edge_dir_out);
CUPVA_EXPORT(bankBData.stage1.intensity, intensity_out);
CUPVA_EXPORT(bankCData.stage1.image, image);
CUPVA_EXPORT(bankAData.stage2.edge_type, edge_type);
CUPVA_EXPORT(bankBData.stage2.edge_dir, edge_dir_in);
CUPVA_EXPORT(bankCData.stage2.intensity, intensity_in);
CUPVA_EXPORT(bankAData.stage3.inBuffer, in_edge_type);
CUPVA_EXPORT(bankAData.stage3.inIntermediateBuffer, in_intermediate_buffer);
CUPVA_EXPORT(bankBData.stage3.outIntermediateBuffer, out_intermediate_buffer);
CUPVA_EXPORT(bankBData.stage3.outEdgeMap, out_edge_map);

VMEM(A, int32_t, img_w);
VMEM(A, int32_t, img_h);
VMEM(A, int32_t, gradient_size);
VMEM(A, int32_t, threshold_weak);
VMEM(A, int32_t, threshold_strong);

VMEM_RDF_UNIFIED(B, image_handler);
VMEM_RDF_UNIFIED(B, intensity_out_handler);
VMEM_RDF_UNIFIED(B, edge_dir_out_handler);
VMEM_RDF_UNIFIED(B, intensity_in_handler);
VMEM_RDF_UNIFIED(B, edge_dir_in_handler);
VMEM_RDF_UNIFIED(B, edge_type_out_handler);

CUPVA_VPU_MAIN()
{
    int32_t ntiles_stage1 = ((img_w + TW - 1) / TW) * ((img_h + TH - 1) / TH);
    int32_t ntiles_stage2 = ((img_w + TW_NMS - 1) / TW_NMS) * ((img_h + TH_NMS - 1) / TH_NMS);

    // Stage 1
    cupvaRasterDataFlowOpen(image_handler, &bankCData.stage1.image[0]);
    cupvaRasterDataFlowOpen(intensity_out_handler, &bankBData.stage1.intensity[0]);
    cupvaRasterDataFlowOpen(edge_dir_out_handler, &bankAData.stage1.edge_dir[0]);

    bankAData.stage1.stage1_buf_ptrs.coef_r = &bankBData.stage1.coef_r[0];
    bankAData.stage1.stage1_buf_ptrs.coef1  = gradient_size == 3   ? &coef_Sobel3_h[0]
                                              : gradient_size == 5 ? &coef_Sobel5_h[0]
                                                                   : &coef_Sobel7_h[0];
    bankAData.stage1.stage1_buf_ptrs.coef2  = gradient_size == 3   ? &coef_Sobel3_v[0]
                                              : gradient_size == 5 ? &coef_Sobel5_v[0]
                                                                   : &coef_Sobel7_v[0];
    bankAData.stage1.stage1_buf_ptrs.gx_tmp = &bankAData.stage1.gx_tmp[0];
    bankAData.stage1.stage1_buf_ptrs.gy_tmp = &bankCData.stage1.gy_tmp[0];
    bankAData.stage1.stage1_buf_ptrs.gx     = &bankAData.stage1.gx[0];
    bankAData.stage1.stage1_buf_ptrs.gy     = &bankBData.stage1.gy[0];

    int32_t image_lp         = cupvaRasterDataFlowGetLinePitch(image_handler);
    int32_t intensity_out_lp = cupvaRasterDataFlowGetLinePitch(intensity_out_handler);
    int32_t edge_dir_out_lp  = cupvaRasterDataFlowGetLinePitch(edge_dir_out_handler);

    int32_t image_cb_len = cupvaRasterDataFlowGetCbLen(image_handler);

    canny_edge_detector_stage1_init(&bankCData.stage1.image[0], &bankBData.stage1.intensity[0],
                                    &bankAData.stage1.edge_dir[0], &bankAData.stage1.stage1_buf_ptrs,
                                    &bankAData.stage1.flt4x2_para, &bankAData.stage1.flt4_para[0],
                                    &bankAData.stage1.edge_para[0], gradient_size, image_lp, intensity_out_lp,
                                    edge_dir_out_lp, image_cb_len * sizeof(uint8_t));

    cupvaICachePrefetch((uintptr_t)canny_edge_detector_stage1_exec, 8192);

    for (int32_t i = 0; i < ntiles_stage1; i++)
    {
        uint8_t *image     = (uint8_t *)cupvaRasterDataFlowAcquire(image_handler);
        int32_t *intensity = (int32_t *)cupvaRasterDataFlowAcquire(intensity_out_handler);
        int8_t *edge_dir   = (int8_t *)cupvaRasterDataFlowAcquire(edge_dir_out_handler);

        cupvaModifyAgenCfgBase(&bankAData.stage1.flt4x2_para.c[0], image);
        cupvaModifyAgenCfgBase(&bankAData.stage1.flt4x2_para.c[1], image + 4);
        cupvaModifyAgenCfgBase(&bankAData.stage1.edge_para[0].c[2], intensity);
        cupvaModifyAgenCfgBase(&bankAData.stage1.edge_para[1].c[2], edge_dir);

        canny_edge_detector_stage1_exec(&bankAData.stage1.flt4x2_para, &bankAData.stage1.flt4_para[0],
                                        &bankAData.stage1.edge_para[0]);

        cupvaRasterDataFlowRelease(image_handler);
        cupvaRasterDataFlowRelease(intensity_out_handler);
        cupvaRasterDataFlowRelease(edge_dir_out_handler);
    }
    cupvaRasterDataFlowClose(image_handler);
    cupvaRasterDataFlowClose(intensity_out_handler);
    cupvaRasterDataFlowClose(edge_dir_out_handler);

    // Stage 2
    cupvaRasterDataFlowOpen(edge_dir_in_handler, &bankBData.stage2.edge_dir[0]);
    cupvaRasterDataFlowOpen(intensity_in_handler, &bankCData.stage2.intensity[0]);
    cupvaRasterDataFlowOpen(edge_type_out_handler, &bankAData.stage2.edge_type[0]);

    int32_t intensity_in_lp = cupvaRasterDataFlowGetLinePitch(intensity_in_handler);
    int32_t edge_dir_in_lp  = cupvaRasterDataFlowGetLinePitch(edge_dir_in_handler);
    int32_t edge_type_lp    = cupvaRasterDataFlowGetLinePitch(edge_type_out_handler);

    int32_t intensity_cb_len = cupvaRasterDataFlowGetCbLen(intensity_in_handler);

    nms_double_thresholding_init(&bankBData.stage2.edge_dir[0], &bankCData.stage2.intensity[0],
                                 &bankAData.stage2.edge_type[0], TW_NMS, TH_NMS, edge_dir_in_lp, intensity_in_lp,
                                 edge_type_lp, intensity_cb_len * sizeof(int32_t), &bankAData.stage2.edge_para);

    for (int32_t i = 0; i < ntiles_stage2; i++)
    {
        int8_t *edge_dir   = (int8_t *)cupvaRasterDataFlowAcquire(edge_dir_in_handler);
        int32_t *intensity = (int32_t *)cupvaRasterDataFlowAcquire(intensity_in_handler);
        uint8_t *edge_type = (uint8_t *)cupvaRasterDataFlowAcquire(edge_type_out_handler);

        cupvaModifyAgenCfgBase(&bankAData.stage2.edge_para.c[0], edge_dir);
        cupvaModifyAgenCfgBase(&bankAData.stage2.edge_para.c[1], intensity);
        cupvaModifyAgenCfgBase(&bankAData.stage2.edge_para.c[2], edge_type);

        nms_double_thresholding_exec(&bankAData.stage2.edge_para, threshold_strong, threshold_weak);

        cupvaRasterDataFlowRelease(edge_dir_in_handler);
        cupvaRasterDataFlowRelease(intensity_in_handler);
        cupvaRasterDataFlowRelease(edge_type_out_handler);
    }
    cupvaRasterDataFlowClose(edge_dir_in_handler);
    cupvaRasterDataFlowClose(intensity_in_handler);
    cupvaRasterDataFlowClose(edge_type_out_handler);

    // Stage 3
    bankBData.stage3.bufPtrs.inBuffer              = &bankAData.stage3.inBuffer[0];
    bankBData.stage3.bufPtrs.inIntermediateBuffer  = &bankAData.stage3.inIntermediateBuffer[0];
    bankBData.stage3.bufPtrs.equivalentPairs       = &bankAData.stage3.equivalentPairs[0];
    bankBData.stage3.bufPtrs.previousRowLabels     = &bankAData.stage3.previousRowLabels[0];
    bankBData.stage3.bufPtrs.reformattedLabels     = &bankAData.stage3.reformattedLabels[0];
    bankBData.stage3.bufPtrs.outIntermediateBuffer = &bankBData.stage3.outIntermediateBuffer[0];
    bankBData.stage3.bufPtrs.outEdgeMap            = &bankBData.stage3.outEdgeMap[0];
    bankBData.stage3.bufPtrs.reformatInput         = &bankBData.stage3.reformatInput[0];
    bankBData.stage3.bufPtrs.tempCollapseBuffer    = &bankBData.stage3.tempCollapseBuffer[0];
    bankBData.stage3.bufPtrs.outLabels             = &bankBData.stage3.outLabels[0];
    bankBData.stage3.bufPtrs.labelLut              = &bankCData.stage3.labelLut[0];

    int32_t nIterX = (img_w + CCL_TILEW - 1) / CCL_TILEW;
    int32_t nIterY = (img_h + CCL_TILEH - 1) / CCL_TILEH;

    cupvaICachePrefetch((uintptr_t)canny_edge_detector_ccl_exec, 8192);

    canny_edge_detector_ccl_exec(&bankBData.stage3.bufPtrs, nIterX, nIterY);
    return 0;
}
