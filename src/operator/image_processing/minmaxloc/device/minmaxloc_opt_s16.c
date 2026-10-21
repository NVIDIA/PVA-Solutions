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

#include "minmaxloc_opt_s16.h"

// main functions
void find_minmax_values_short(MINMAX_VAL_CFG &cfg, short *p_min_val, short *p_max_val)
{
    agen a_src = init_agen_from_cfg(cfg._src_agen_cfg);

    dvshortx dv_min, dv_max;
    dv_min.lo = replicateh(*p_min_val);
    dv_min.hi = replicateh(*p_min_val);
    dv_max.lo = replicateh(*p_max_val);
    dv_max.hi = replicateh(*p_max_val);

    chess_separator_scheduler();

    int niter = cfg._niter_w * cfg._niter_h;
    niter     = max(((niter + 8 - 1) / 8) * 8, 24);
    for (int i = 0; i < niter; i++) chess_loop_range(24, )
    chess_unroll_loop(8)
    chess_prepare_for_pipelining
    {
        dvshortx dv_data = dvshort_load(a_src);
        dv_min           = dvmin(dv_data, dv_min);
        dv_max           = dvmax(dv_data, dv_max);
    }

    short min_val_lo = vminr_s(dv_min.lo);
    short min_val_hi = vminr_s(dv_min.hi);
    *p_min_val       = (min_val_lo > min_val_hi) ? min_val_hi : min_val_lo;

    short max_val_lo = vmaxr_s(dv_max.lo);
    short max_val_hi = vmaxr_s(dv_max.hi);
    *p_max_val       = (max_val_lo > max_val_hi) ? max_val_lo : max_val_hi;
}

int find_value_locations_short(MINMAX_LOC_CFG &cfg, short value, unsigned short *p_val_loc_x,
                               unsigned short *p_val_loc_y, unsigned short tile_offset_x, unsigned short tile_offset_y)
{
    agen a_src = init_agen_from_cfg(cfg._src_agen_cfg);

    unsigned short *p_lane_id = cfg._p_lane_id;
    dvshortx dv_lane_id       = zero_extend(*(dvushort *)p_lane_id);
    unsigned short stride_x   = cfg._blk_w;

    int niter_w = cfg._niter_w;
    int niter_h = cfg._niter_h;
    int niter   = niter_w * niter_h;

    dvshortx dv_pix_pos_x_row_start = dv_lane_id + tile_offset_x;
    dvshortx dv_pix_pos_x_row_end   = dv_pix_pos_x_row_start + (niter_w - 1) * stride_x;
    dvshortx dv_pix_pos_x           = dv_pix_pos_x_row_start;
    vshortx v_pix_pos_y             = replicateh(tile_offset_y);

    chess_separator_scheduler();

    int total_cnt = 0;

    niter = ((niter + 3) / 4) * 4;
    for (int i = 0; i < niter; i++) chess_loop_range(24, )
    chess_unroll_loop(4)
    chess_prepare_for_pipelining
    {
        dvshortx dv_data = dvshort_load(a_src);

        // 1. get locations equal to min/max
        dvshortx dv_flag_eq = (dv_data == value);

        // 2. extract min/max location coordinates
        int val_cnt = collate_nz_pos_vshort(dv_flag_eq.lo, dv_pix_pos_x.lo, v_pix_pos_y, p_val_loc_x, p_val_loc_y);
        val_cnt += collate_nz_pos_vshort(dv_flag_eq.hi, dv_pix_pos_x.hi, v_pix_pos_y, p_val_loc_x, p_val_loc_y);

        // increase counter
        total_cnt += val_cnt;

        // update coordinates: check if starts a new row
        dvshortx dflag_start_new_row = (dv_pix_pos_x == dv_pix_pos_x_row_end);
        dv_pix_pos_x                 = dvmux(dflag_start_new_row, dv_pix_pos_x_row_start, dv_pix_pos_x + stride_x);
        v_pix_pos_y                  = vmux(dflag_start_new_row.lo, v_pix_pos_y + 1, v_pix_pos_y);
    }
    return total_cnt;
}

void find_minmax_locations_short(MINMAX_LOC_CFG &cfg, short min_val, short max_val, int &restrict min_cnt,
                                 int &restrict max_cnt, unsigned short *p_min_loc_x, unsigned short *p_min_loc_y,
                                 unsigned short *p_max_loc_x, unsigned short *p_max_loc_y, unsigned short tile_offset_x,
                                 unsigned short tile_offset_y)
{
    agen a_src = init_agen_from_cfg(cfg._src_agen_cfg);

    unsigned short *p_lane_id = cfg._p_lane_id;
    dvshortx dv_lane_id       = zero_extend(*(dvushort *)p_lane_id);
    unsigned short stride_x   = cfg._blk_w;

    int niter_w = cfg._niter_w;
    int niter_h = cfg._niter_h;
    int niter   = niter_w * niter_h;

    dvshortx dv_pix_pos_x_row_start = dv_lane_id + tile_offset_x;
    dvshortx dv_pix_pos_x_row_end   = dv_pix_pos_x_row_start + (niter_w - 1) * stride_x;
    dvshortx dv_pix_pos_x           = dv_pix_pos_x_row_start;
    vshortx v_pix_pos_y             = replicateh(tile_offset_y);

    chess_separator_scheduler();

    min_cnt = 0;
    max_cnt = 0;
    for (int i = 0; i < niter; i++) chess_prepare_for_pipelining
    {
        dvshortx dv_data = dvshort_load(a_src);

        // 1. get locations equal to min/max
        dvshortx dv_flag_min = (dv_data == min_val);
        dvshortx dv_flag_max = (dv_data == max_val);

        // 2. extract min/max location coordinates
        int step_min_cnt =
            collate_nz_pos_vshort(dv_flag_min.lo, dv_pix_pos_x.lo, v_pix_pos_y, p_min_loc_x, p_min_loc_y);
        step_min_cnt += collate_nz_pos_vshort(dv_flag_min.hi, dv_pix_pos_x.hi, v_pix_pos_y, p_min_loc_x, p_min_loc_y);
        int step_max_cnt =
            collate_nz_pos_vshort(dv_flag_max.lo, dv_pix_pos_x.lo, v_pix_pos_y, p_max_loc_x, p_max_loc_y);
        step_max_cnt += collate_nz_pos_vshort(dv_flag_max.hi, dv_pix_pos_x.hi, v_pix_pos_y, p_max_loc_x, p_max_loc_y);

        // increase counter
        min_cnt += step_min_cnt;
        max_cnt += step_max_cnt;

        // update coordinates: check if starts a new row
        dvshortx dflag_start_new_row = (dv_pix_pos_x == dv_pix_pos_x_row_end);
        dv_pix_pos_x                 = dvmux(dflag_start_new_row, dv_pix_pos_x_row_start, dv_pix_pos_x + stride_x);
        v_pix_pos_y                  = vmux(dflag_start_new_row.lo, v_pix_pos_y + 1, v_pix_pos_y);
    }
}

// special kernels
int find_value_locations_short_last_column(MINMAX_LOC_CFG &cfg, short value, unsigned short *p_val_loc_x,
                                           unsigned short *p_val_loc_y, unsigned short tile_offset_x,
                                           unsigned short tile_offset_y, unsigned short image_width)
{
    agen a_src = init_agen_from_cfg(cfg._src_agen_cfg);

    unsigned short *p_lane_id = cfg._p_lane_id;
    dvshortx dv_lane_id       = zero_extend(*(dvushort *)p_lane_id);
    unsigned short stride_x   = cfg._blk_w;

    int niter_w = cfg._niter_w;
    int niter_h = cfg._niter_h;
    int niter   = niter_w * niter_h;

    dvshortx dv_pix_pos_x_row_start = dv_lane_id + tile_offset_x;
    dvshortx dv_pix_pos_x_row_end   = dv_pix_pos_x_row_start + (niter_w - 1) * stride_x;
    dvshortx dv_pix_pos_x           = dv_pix_pos_x_row_start;
    vshortx v_pix_pos_y             = replicateh(tile_offset_y);

    chess_separator_scheduler();

    int total_cnt = 0;
    for (int i = 0; i < niter; i++) chess_prepare_for_pipelining
    {
        dvshortx dv_data = dvshort_load(a_src);

        // 1. get locations equal to min/max
        dvshortx dv_flag_eq = (dv_data == value) && (dv_pix_pos_x < image_width);

        // 2. extract min/max location coordinates
        int val_cnt = collate_nz_pos_vshort(dv_flag_eq.lo, dv_pix_pos_x.lo, v_pix_pos_y, p_val_loc_x, p_val_loc_y);
        val_cnt += collate_nz_pos_vshort(dv_flag_eq.hi, dv_pix_pos_x.hi, v_pix_pos_y, p_val_loc_x, p_val_loc_y);

        // increase counter
        total_cnt += val_cnt;

        // update coordinates: check if starts a new row
        dvshortx dflag_start_new_row = (dv_pix_pos_x == dv_pix_pos_x_row_end);
        dv_pix_pos_x                 = dvmux(dflag_start_new_row, dv_pix_pos_x_row_start, dv_pix_pos_x + stride_x);
        v_pix_pos_y                  = vmux(dflag_start_new_row.lo, v_pix_pos_y + 1, v_pix_pos_y);
    }
    return total_cnt;
}

void find_minmax_locations_short_last_column(MINMAX_LOC_CFG &cfg, short min_val, short max_val, int &restrict min_cnt,
                                             int &restrict max_cnt, unsigned short *p_min_loc_x,
                                             unsigned short *p_min_loc_y, unsigned short *p_max_loc_x,
                                             unsigned short *p_max_loc_y, unsigned short tile_offset_x,
                                             unsigned short tile_offset_y, unsigned short image_width)
{
    agen a_src = init_agen_from_cfg(cfg._src_agen_cfg);

    unsigned short *p_lane_id = cfg._p_lane_id;
    dvshortx dv_lane_id       = zero_extend(*(dvushort *)p_lane_id);
    unsigned short stride_x   = cfg._blk_w;

    int niter_w = cfg._niter_w;
    int niter_h = cfg._niter_h;
    int niter   = niter_w * niter_h;

    dvshortx dv_pix_pos_x_row_start = dv_lane_id + tile_offset_x;
    dvshortx dv_pix_pos_x_row_end   = dv_pix_pos_x_row_start + (niter_w - 1) * stride_x;
    dvshortx dv_pix_pos_x           = dv_pix_pos_x_row_start;
    vshortx v_pix_pos_y             = replicateh(tile_offset_y);

    chess_separator_scheduler();

    min_cnt = 0;
    max_cnt = 0;
    for (int i = 0; i < niter; i++) chess_prepare_for_pipelining
    {
        dvshortx dv_data = dvshort_load(a_src);

        // 1. get locations equal to min/max
        dvshortx dv_flag_min = (dv_data == min_val) && (dv_pix_pos_x < image_width);
        dvshortx dv_flag_max = (dv_data == max_val) && (dv_pix_pos_x < image_width);

        // 2. extract min/max location coordinates
        int step_min_cnt =
            collate_nz_pos_vshort(dv_flag_min.lo, dv_pix_pos_x.lo, v_pix_pos_y, p_min_loc_x, p_min_loc_y);
        step_min_cnt += collate_nz_pos_vshort(dv_flag_min.hi, dv_pix_pos_x.hi, v_pix_pos_y, p_min_loc_x, p_min_loc_y);
        int step_max_cnt =
            collate_nz_pos_vshort(dv_flag_max.lo, dv_pix_pos_x.lo, v_pix_pos_y, p_max_loc_x, p_max_loc_y);
        step_max_cnt += collate_nz_pos_vshort(dv_flag_max.hi, dv_pix_pos_x.hi, v_pix_pos_y, p_max_loc_x, p_max_loc_y);

        // increase counter
        min_cnt += step_min_cnt;
        max_cnt += step_max_cnt;

        // update coordinates: check if starts a new row
        dvshortx dflag_start_new_row = (dv_pix_pos_x == dv_pix_pos_x_row_end);
        dv_pix_pos_x                 = dvmux(dflag_start_new_row, dv_pix_pos_x_row_start, dv_pix_pos_x + stride_x);
        v_pix_pos_y                  = vmux(dflag_start_new_row.lo, v_pix_pos_y + 1, v_pix_pos_y);
    }
}

int find_value_locations_short_no_unroll(MINMAX_LOC_CFG &cfg, short value, unsigned short *p_val_loc_x,
                                         unsigned short *p_val_loc_y, unsigned short tile_offset_x,
                                         unsigned short tile_offset_y)
{
    agen a_src = init_agen_from_cfg(cfg._src_agen_cfg);

    unsigned short *p_lane_id = cfg._p_lane_id;
    dvshortx dv_lane_id       = zero_extend(*(dvushort *)p_lane_id);
    unsigned short stride_x   = cfg._blk_w;

    int niter_w = cfg._niter_w;
    int niter_h = cfg._niter_h;
    int niter   = niter_w * niter_h;

    dvshortx dv_pix_pos_x_row_start = dv_lane_id + tile_offset_x;
    dvshortx dv_pix_pos_x_row_end   = dv_pix_pos_x_row_start + (niter_w - 1) * stride_x;
    dvshortx dv_pix_pos_x           = dv_pix_pos_x_row_start;
    vshortx v_pix_pos_y             = replicateh(tile_offset_y);

    chess_separator_scheduler();

    int total_cnt = 0;
    for (int i = 0; i < niter; i++) chess_prepare_for_pipelining
    {
        dvshortx dv_data = dvshort_load(a_src);

        // 1. get locations equal to min/max
        dvshortx dv_flag_eq = (dv_data == value);

        // 2. extract min/max location coordinates
        int val_cnt = collate_nz_pos_vshort(dv_flag_eq.lo, dv_pix_pos_x.lo, v_pix_pos_y, p_val_loc_x, p_val_loc_y);
        val_cnt += collate_nz_pos_vshort(dv_flag_eq.hi, dv_pix_pos_x.hi, v_pix_pos_y, p_val_loc_x, p_val_loc_y);

        // increase counter
        total_cnt += val_cnt;

        // update coordinates: check if starts a new row
        dvshortx dflag_start_new_row = (dv_pix_pos_x == dv_pix_pos_x_row_end);
        dv_pix_pos_x                 = dvmux(dflag_start_new_row, dv_pix_pos_x_row_start, dv_pix_pos_x + stride_x);
        v_pix_pos_y                  = vmux(dflag_start_new_row.lo, v_pix_pos_y + 1, v_pix_pos_y);
    }
    return total_cnt;
}
