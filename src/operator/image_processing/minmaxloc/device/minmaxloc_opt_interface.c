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

#include "minmaxloc_opt_interface.h"

void minmax_val_cfg_init(const STYPE *src, unsigned short tile_width, unsigned short tile_height,
                         unsigned short line_pitch, MINMAX_VAL_CFG &cfg)
{

    int blk_w    = pva_elementsof(DVTYPEX);
    int niter_w  = tile_width / blk_w;
    int niter_h  = tile_height;
    cfg._niter_w = niter_w;
    cfg._niter_h = niter_h;

    // agen wrapper
    AgenWrapper w0;
    w0.size = sizeof(STYPE);
    w0.n1   = niter_w;
    w0.s1   = blk_w;
    w0.n2   = niter_h;
    w0.s2   = line_pitch;
    agen a0 = init((DVTYPE *)src);
    INIT_AGEN2(a0, w0);
    cfg._src_agen_cfg = extract_agen_cfg(a0);
}

void minmax_loc_cfg_init(const STYPE_LOC *src, unsigned short tile_width, unsigned short tile_height,
                         unsigned short line_pitch, unsigned short *p_lane_id, MINMAX_LOC_CFG &cfg)
{

    int blk_w    = pva_elementsof(DVTYPEX_LOC);
    cfg._blk_w   = blk_w;
    int niter_w  = tile_width / blk_w;
    int niter_h  = tile_height;
    cfg._niter_w = niter_w;
    cfg._niter_h = niter_h;

    // lane ids for vcharx
    cfg._p_lane_id = p_lane_id;

    // agen wrapper
    AgenWrapper w0;
    w0.size = sizeof(STYPE_LOC);
    w0.n1   = niter_w;
    w0.s1   = blk_w;
    w0.n2   = niter_h;
    w0.s2   = line_pitch;
    agen a0 = init((DVTYPE_LOC *)src);
    INIT_AGEN2(a0, w0);
    cfg._src_agen_cfg = extract_agen_cfg(a0);
}

int find_value_locations_check_last(MINMAX_LOC_CFG &cfg_loc, STYPE_LOC value, unsigned short *p_val_loc_x,
                                    unsigned short *p_val_loc_y, unsigned short tile_offset_x,
                                    unsigned short tile_offset_y, unsigned short image_width, int ntile_w, int ntile_h,
                                    int row_idx, int col_idx, bool has_last_column, bool last_row_loc_no_unroll)
{
    bool last_column_flag = has_last_column && (col_idx == ntile_w - 1);
    bool no_unroll_flag   = last_row_loc_no_unroll && (row_idx == ntile_h - 1);

    int count;
    if (last_column_flag)
    {
        count = FIND_VALUE_LOCATIONS_LAST_COLUMN(cfg_loc, value, p_val_loc_x, p_val_loc_y, tile_offset_x, tile_offset_y,
                                                 image_width);
    }
    else if (no_unroll_flag)
    {
        count = FIND_VALUE_LOCATIONS_NO_UNROLL(cfg_loc, value, p_val_loc_x, p_val_loc_y, tile_offset_x, tile_offset_y);
    }
    else
    {
        count = FIND_VALUE_LOCATIONS(cfg_loc, value, p_val_loc_x, p_val_loc_y, tile_offset_x, tile_offset_y);
    }
    return count;
}
