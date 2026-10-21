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

#include "nhwc_to_nchw_letterbox_normalize_knl.h"
#include "nhwc_to_nchw_letterbox_normalize_param.h"

#include <cupva_device.h>

/*
 * VMEM layout
 * -----------
 * Bank A: srcBuf (uint8 HWC, TRANS_MODE_1, double buffered) + srcHdl + vpu_param
 * Bank B: dstC0Buf, dstC1Buf (float32 planar, TRANS_MODE_1, double buffered)
 * Bank C: dstC2Buf (float32 planar, TRANS_MODE_1, double buffered) + dstHdl
 *
 * Input DMA tiles over the IMAGE REGION (imgH x imgW), not the full output.
 * Output DMA tiles write to the image region within each channel plane,
 * with DRAM pitch = outW to skip over letterbox padding columns.
 * Padding is not written by the kernel.
 */

/* Input: uint8 HWC with TRANS_MODE_1 for transposed byte access */
VMEM(A, uint8_t, srcBuf, RDF_DOUBLE(uint8_t, NCHW_LN_TILE_W * 3, NCHW_LN_TILE_H, 0, 0, TRANS_MODE_1));

/* Output: float32 per-channel with TRANS_MODE_1 for transposed float stores */
VMEM(B, float, dstC0Buf, RDF_DOUBLE(float, NCHW_LN_TILE_W, NCHW_LN_TILE_H, 0, 0, TRANS_MODE_1));
VMEM(B, float, dstC1Buf, RDF_DOUBLE(float, NCHW_LN_TILE_W, NCHW_LN_TILE_H, 0, 0, TRANS_MODE_1));
VMEM(C, float, dstC2Buf, RDF_DOUBLE(float, NCHW_LN_TILE_W, NCHW_LN_TILE_H, 0, 0, TRANS_MODE_1));

/* RDF handlers */
VMEM_RDF_UNIFIED(A, srcHdl); /* Input handler */
VMEM_RDF_UNIFIED(C, dstHdl); /* Output handler (shared by 3 RDFs) */

/* Parameters */
VMEM(A, nhwc_to_nchw_letterbox_normalize_host2dev_param_t, vpu_param);

CUPVA_VPU_MAIN()
{
    const int tile_w  = vpu_param.tile_width;
    const int tile_h  = vpu_param.tile_height;
    const int niter_x = vpu_param.niter_x;
    const int niter_y = vpu_param.niter_y;
    const int img_w   = vpu_param.img_width;
    const int img_h   = vpu_param.img_height;
    const float scale = vpu_param.scale;

    /* Open dataflows */
    cupvaRasterDataFlowOpen(srcHdl, &srcBuf[0]);
    cupvaRasterDataFlowOpen(dstHdl, &dstC0Buf[0]);

    /* Compute lane offsets for TRANS_MODE_1 access */
    int src_lane_ofst = cupvaTransposeLaneOfst(srcHdl, TRANS_MODE_1, sizeof(srcBuf[0]));
    int dst_lane_ofst = cupvaTransposeLaneOfst(dstHdl, TRANS_MODE_1, sizeof(dstC0Buf[0]));

    /* Line pitches are invariant for the lifetime of the open dataflows. */
    int src_pitch = cupvaRasterDataFlowGetLinePitch(srcHdl); /* uint8 elements */
    int dst_pitch = cupvaRasterDataFlowGetLinePitch(dstHdl); /* float elements */

    /* Process all tiles */
    for (int tile_idx = 0; tile_idx < niter_x * niter_y; tile_idx++)
    {
        /* Acquire input tile (uint8 HWC, TRANS_MODE_1) */
        unsigned char *src_tile = (unsigned char *)cupvaRasterDataFlowAcquire(srcHdl);

        /* Acquire output tiles (3 channels via shared handler) */
        float *dstC0_tile = (float *)cupvaRasterDataFlowAcquire(dstHdl);

        /* Derive other channel buffer pointers from the shared handler offset */
        ptrdiff_t offset  = dstC0_tile - &dstC0Buf[0];
        float *dstC1_tile = &dstC1Buf[offset];
        float *dstC2_tile = &dstC2Buf[offset];

        /* Actual tile dimensions (handle right/bottom edge tiles) */
        int tile_x   = tile_idx % niter_x;
        int tile_y   = tile_idx / niter_x;
        int actual_w = (tile_x == niter_x - 1) ? (img_w - tile_x * tile_w) : tile_w;
        int actual_h = (tile_y == niter_y - 1) ? (img_h - tile_y * tile_h) : tile_h;

        /* Fused transpose + normalize */
        nhwc_to_nchw_letterbox_normalize_tile_kernel(src_tile, dstC0_tile, dstC1_tile, dstC2_tile, actual_w, actual_h,
                                                     src_pitch, dst_pitch, src_lane_ofst, dst_lane_ofst, scale);

        /* Release tiles */
        cupvaRasterDataFlowRelease(srcHdl);
        cupvaRasterDataFlowRelease(dstHdl);
    }

    /* Close dataflows */
    cupvaRasterDataFlowClose(srcHdl);
    cupvaRasterDataFlowClose(dstHdl);

    return 0;
}
