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

#include <cupva_device.h>

void nhwc_to_nchw_letterbox_normalize_tile_kernel(const unsigned char *src_nhwc, float *dst_c0, float *dst_c1,
                                                  float *dst_c2, int tile_width, int tile_height, int src_pitch,
                                                  int dst_pitch, int src_lane_ofst, int dst_lane_ofst, float scale)
{
    /*
     * Fused transpose + normalize using vuchar_dvintx_load_transp / vstore_transp.
     *
     * vuchar_dvintx_load_transp reads 1 uint8 per lane from 16 consecutive rows
     * (TRANS_MODE_1 VMEM), zero-extends each to int32 -> dvintx (16 x int32).
     *   dvintx.lo = rows 0-7,  dvintx.hi = rows 8-15
     *
     * Channel deinterleave is FREE: in HWC layout, bytes at offset x*3+0 are R,
     * x*3+1 are G, x*3+2 are B. Three sequential transposed loads naturally
     * separate the channels.
     *
     * vintx_vfp converts int32 -> float32; vmulf applies normalization scale.
     * vstore_transp writes 8 float32 values to 8 rows of the TRANS_MODE_1 output.
     *
     * Per pixel column: 3 loads (R,G,B from 16 rows) + 6 stores (3 ch x 2 halves).
     */

    for (int y_block = 0; y_block < tile_height; y_block += 16)
    {
        const unsigned char *src_block = src_nhwc + y_block * src_pitch;
        float *dst_block_c0            = dst_c0 + y_block * dst_pitch;
        float *dst_block_c1            = dst_c1 + y_block * dst_pitch;
        float *dst_block_c2            = dst_c2 + y_block * dst_pitch;

        /*
         * Source agen: uint8 transposed access, steps through R,G,B bytes.
         *   n1 = tile_width * 3: covers all channel bytes across all pixel columns
         *   mod1 = 1 byte: advance to next channel/pixel byte
         *   lane_ofst: T1 row pitch for 16-lane transposed access
         */
        agen a_src      = init((vuchar *)src_block);
        a_src.lane_ofst = src_lane_ofst;
        a_src.n1        = tile_width * 3;
        a_src.mod1      = sizeof(unsigned char);

        /* Output agens for rows 0-7 (dvintx.lo half) */
        agen a_c0_lo      = init((vfloat *)dst_block_c0);
        a_c0_lo.lane_ofst = dst_lane_ofst;
        a_c0_lo.n1        = tile_width;
        a_c0_lo.mod1      = sizeof(float);

        agen a_c1_lo      = init((vfloat *)dst_block_c1);
        a_c1_lo.lane_ofst = dst_lane_ofst;
        a_c1_lo.n1        = tile_width;
        a_c1_lo.mod1      = sizeof(float);

        agen a_c2_lo      = init((vfloat *)dst_block_c2);
        a_c2_lo.lane_ofst = dst_lane_ofst;
        a_c2_lo.n1        = tile_width;
        a_c2_lo.mod1      = sizeof(float);

        /* Output agens for rows 8-15 (dvintx.hi half) */
        agen a_c0_hi      = init((vfloat *)(dst_block_c0 + 8 * dst_pitch));
        a_c0_hi.lane_ofst = dst_lane_ofst;
        a_c0_hi.n1        = tile_width;
        a_c0_hi.mod1      = sizeof(float);

        agen a_c1_hi      = init((vfloat *)(dst_block_c1 + 8 * dst_pitch));
        a_c1_hi.lane_ofst = dst_lane_ofst;
        a_c1_hi.n1        = tile_width;
        a_c1_hi.mod1      = sizeof(float);

        agen a_c2_hi      = init((vfloat *)(dst_block_c2 + 8 * dst_pitch));
        a_c2_hi.lane_ofst = dst_lane_ofst;
        a_c2_hi.n1        = tile_width;
        a_c2_hi.mod1      = sizeof(float);

        for (int x = 0; x < tile_width; x++)
        {
            /* Transposed byte load: R, G, B from 16 rows each */
            dvintx ri = vuchar_dvintx_load_transp(a_src);
            dvintx gi = vuchar_dvintx_load_transp(a_src);
            dvintx bi = vuchar_dvintx_load_transp(a_src);

            /* Rows 0-7: convert to float, normalize, store */
            vfloatx r_lo = vmulf(vintx_vfp(ri.lo), scale);
            vfloatx g_lo = vmulf(vintx_vfp(gi.lo), scale);
            vfloatx b_lo = vmulf(vintx_vfp(bi.lo), scale);
            vstore_transp(r_lo, a_c0_lo);
            vstore_transp(g_lo, a_c1_lo);
            vstore_transp(b_lo, a_c2_lo);

            /* Rows 8-15: convert to float, normalize, store */
            vfloatx r_hi = vmulf(vintx_vfp(ri.hi), scale);
            vfloatx g_hi = vmulf(vintx_vfp(gi.hi), scale);
            vfloatx b_hi = vmulf(vintx_vfp(bi.hi), scale);
            vstore_transp(r_hi, a_c0_hi);
            vstore_transp(g_hi, a_c1_hi);
            vstore_transp(b_hi, a_c2_hi);
        }
    }
}
