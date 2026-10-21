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

#ifndef MINMAXLOC_COMMON_H
#define MINMAXLOC_COMMON_H

#include <cupva_device.h>

typedef struct
{
    int _niter_w;
    int _niter_h;
    AgenCFG _src_agen_cfg;
} MINMAX_VAL_CFG;

typedef struct
{
    int _blk_w;
    int _niter_w;
    int _niter_h;
    AgenCFG _src_agen_cfg;

    unsigned short *_p_lane_id;
} MINMAX_LOC_CFG;

// common
static inline int collate_nz_pos_vshort(vshortx v_flag, vshortx v_pix_pos_x, vshortx v_pix_pos_y,
                                        unsigned short *&p_loc_x, unsigned short *&p_loc_y)
{
    vshortx v_nz_idx;
    int bit_nz_idx;
    vcollate_idx_bits(v_flag, v_nz_idx, bit_nz_idx);
    int count = bitcount(bit_nz_idx);

    // extract nonzero indices and write to p_loc_x/y
    vshortx v_data_compact                     = vpermute(v_pix_pos_x, v_nz_idx);
    *(vshort chess_storage(RAM_Bb) *)(p_loc_x) = extract(v_data_compact);
    *(vshort chess_storage(RAM_Cb) *)(p_loc_y) = extract(v_pix_pos_y);

    p_loc_x += count;
    p_loc_y += count;
    return count;
}

static inline void interleave_loc_xy_vshort(unsigned short *p_loc_x, unsigned short *p_loc_y, int count,
                                            unsigned short *p_loc_xy)
{
    // init the agen
    AgenWrapper w0;
    w0.size   = sizeof(unsigned short);
    int vec_w = pva_elementsof(vshortx);
    int niter = (count + vec_w - 1) / vec_w;
    w0.n1     = niter;
    w0.s1     = vec_w;
    agen a_x  = init((vshort *)p_loc_x);
    agen a_y  = init((vshort *)p_loc_y);
    INIT_AGEN1(a_x, w0);
    INIT_AGEN1(a_y, w0);

    AgenWrapper w1;
    w1.size     = sizeof(unsigned short);
    w1.n1       = niter;
    w1.s1       = 2 * vec_w;
    agen a_xy_i = init((dvshort *)p_loc_xy);
    INIT_AGEN1(a_xy_i, w1);

    niter = (niter + 7) / 8 * 8;
    for (int i = 0; i < niter; i++) chess_loop_range(24, )
    chess_unroll_loop(8)
    chess_prepare_for_pipelining
    {
        dvshortx dv_before;
        dv_before.lo = vushort_load(a_x);
        dv_before.hi = vushort_load(a_y);
        vstore_i(dv_before, a_xy_i);
    }
}

// SQDF write
void write_minmax_buf(int &write_iter, SequenceDataFlowHandler &sqdf_hdl, ExtMemPointer &extLocPtr, int &pingPong,
                      unsigned short *LocBuf, int &buf_start, int &buf_end, int &glb_write_cnt, unsigned short *p_loc_x,
                      unsigned short *p_loc_y, int pos_tile_size, int loc_capacity);

#endif // MINMAXLOC_COMMON_H
