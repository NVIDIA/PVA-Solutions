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

#ifndef MINMAXLOC_OPT_INTERFACE_H
#define MINMAXLOC_OPT_INTERFACE_H

#if defined(DATA_TYPE_U8) && (DATA_TYPE_U8 == 1)
#    include "minmaxloc_opt_s8.h"
#    include "minmaxloc_opt_u8.h"
#    define STYPE unsigned char
#    define DVTYPE dvuchar
#    define DVTYPEX dvcharx
#    define STYPE_LOC char
#    define DVTYPEX_LOC dvshortx
#    define DVTYPE_LOC dvshort
#    define FIND_MINMAX_VALUES find_minmax_values_uchar
#    define FIND_VALUE_LOCATIONS find_value_locations_char
#    define FIND_MINMAX_LOCATIONS find_minmax_locations_char
#    define FIND_VALUE_LOCATIONS_LAST_COLUMN find_value_locations_char_last_column
#    define FIND_MINMAX_LOCATIONS_LAST_COLUMN find_minmax_locations_char_last_column
#    define FIND_VALUE_LOCATIONS_NO_UNROLL find_value_locations_char_no_unroll
#elif defined(DATA_TYPE_S8) && (DATA_TYPE_S8 == 1)
#    include "minmaxloc_opt_s8.h"
#    define STYPE char
#    define DVTYPEX dvcharx
#    define DVTYPE dvchar
#    define STYPE_LOC char
#    define DVTYPEX_LOC dvshortx
#    define DVTYPE_LOC dvshort
#    define FIND_MINMAX_VALUES find_minmax_values_char
#    define FIND_VALUE_LOCATIONS find_value_locations_char
#    define FIND_MINMAX_LOCATIONS find_minmax_locations_char
#    define FIND_VALUE_LOCATIONS_LAST_COLUMN find_value_locations_char_last_column
#    define FIND_MINMAX_LOCATIONS_LAST_COLUMN find_minmax_locations_char_last_column
#    define FIND_VALUE_LOCATIONS_NO_UNROLL find_value_locations_char_no_unroll
#elif defined(DATA_TYPE_U16) && (DATA_TYPE_U16 == 1)
#    include "minmaxloc_opt_s16.h"
#    include "minmaxloc_opt_u16.h"
#    define STYPE unsigned short
#    define DVTYPEX dvshortx
#    define DVTYPE dvushort
#    define STYPE_LOC short
#    define DVTYPEX_LOC dvshortx
#    define DVTYPE_LOC dvshort
#    define FIND_MINMAX_VALUES find_minmax_values_ushort
#    define FIND_VALUE_LOCATIONS find_value_locations_short
#    define FIND_MINMAX_LOCATIONS find_minmax_locations_short
#    define FIND_VALUE_LOCATIONS_LAST_COLUMN find_value_locations_short_last_column
#    define FIND_MINMAX_LOCATIONS_LAST_COLUMN find_minmax_locations_short_last_column
#    define FIND_VALUE_LOCATIONS_NO_UNROLL find_value_locations_short_no_unroll
#elif defined(DATA_TYPE_S16) && (DATA_TYPE_S16 == 1)
#    include "minmaxloc_opt_s16.h"
#    define STYPE short
#    define DVTYPEX dvshortx
#    define DVTYPE dvshort
#    define STYPE_LOC short
#    define DVTYPEX_LOC dvshortx
#    define DVTYPE_LOC dvshort
#    define FIND_MINMAX_VALUES find_minmax_values_short
#    define FIND_VALUE_LOCATIONS find_value_locations_short
#    define FIND_MINMAX_LOCATIONS find_minmax_locations_short
#    define FIND_VALUE_LOCATIONS_LAST_COLUMN find_value_locations_short_last_column
#    define FIND_MINMAX_LOCATIONS_LAST_COLUMN find_minmax_locations_short_last_column
#    define FIND_VALUE_LOCATIONS_NO_UNROLL find_value_locations_short_no_unroll
#elif defined(DATA_TYPE_U32) && (DATA_TYPE_U32 == 1)
#    include "minmaxloc_opt_s32.h"
#    include "minmaxloc_opt_u32.h"
#    define STYPE unsigned int
#    define DVTYPEX dvintx
#    define DVTYPE dvuint
#    define STYPE_LOC int
#    define DVTYPEX_LOC dvintx
#    define DVTYPE_LOC dvint
#    define FIND_MINMAX_VALUES find_minmax_values_uint
#    define FIND_VALUE_LOCATIONS find_value_locations_int
#    define FIND_MINMAX_LOCATIONS find_minmax_locations_int
#    define FIND_VALUE_LOCATIONS_LAST_COLUMN find_value_locations_int_last_column
#    define FIND_MINMAX_LOCATIONS_LAST_COLUMN find_minmax_locations_int_last_column
#    define FIND_VALUE_LOCATIONS_NO_UNROLL find_value_locations_int_no_unroll
#elif defined(DATA_TYPE_S32) && (DATA_TYPE_S32 == 1)
#    include "minmaxloc_opt_s32.h"
#    define STYPE int
#    define DVTYPEX dvintx
#    define DVTYPE dvint
#    define STYPE_LOC int
#    define DVTYPEX_LOC dvintx
#    define DVTYPE_LOC dvint
#    define FIND_MINMAX_VALUES find_minmax_values_int
#    define FIND_VALUE_LOCATIONS find_value_locations_int
#    define FIND_MINMAX_LOCATIONS find_minmax_locations_int
#    define FIND_VALUE_LOCATIONS_LAST_COLUMN find_value_locations_int_last_column
#    define FIND_MINMAX_LOCATIONS_LAST_COLUMN find_minmax_locations_int_last_column
#    define FIND_VALUE_LOCATIONS_NO_UNROLL find_value_locations_int_no_unroll
#elif defined(DATA_TYPE_F32) && (DATA_TYPE_F32 == 1)
#    include "minmaxloc_opt_f32.h"
#    include "minmaxloc_opt_s32.h"
#    define STYPE float
#    define DVTYPEX dvfloatx
#    define DVTYPE dvfloat
#    define STYPE_LOC int
#    define DVTYPEX_LOC dvintx
#    define DVTYPE_LOC dvint
#    define FIND_VALUE_LOCATIONS find_value_locations_int
#    define FIND_MINMAX_LOCATIONS find_minmax_locations_int
#    define FIND_VALUE_LOCATIONS_LAST_COLUMN find_value_locations_int_last_column
#    define FIND_MINMAX_LOCATIONS_LAST_COLUMN find_minmax_locations_int_last_column
#    define FIND_VALUE_LOCATIONS_NO_UNROLL find_value_locations_int_no_unroll
#endif

#include "minmaxloc_common.h"

// init configs
void minmax_val_cfg_init(const STYPE *src, unsigned short tile_width, unsigned short tile_height,
                         unsigned short line_pitch, MINMAX_VAL_CFG &cfg);

void minmax_loc_cfg_init(const STYPE_LOC *src, unsigned short tile_width, unsigned short tile_height,
                         unsigned short line_pitch, unsigned short *p_lane_id, MINMAX_LOC_CFG &cfg);

// unified interface functions for all data types and corner cases(with check for last column/row tiles)

static inline void find_minmax_locations_check_last(MINMAX_LOC_CFG &cfg_loc, STYPE_LOC min_val, STYPE_LOC max_val,
                                                    int &tile_min_cnt, int &tile_max_cnt, unsigned short *p_min_loc_x,
                                                    unsigned short *p_min_loc_y, unsigned short *p_max_loc_x,
                                                    unsigned short *p_max_loc_y, unsigned short tile_offset_x,
                                                    unsigned short tile_offset_y, unsigned short image_width,
                                                    int ntile_w, int col_idx, bool has_last_column)
{
    bool last_column_flag = has_last_column && col_idx == ntile_w - 1;

    if (last_column_flag)
    {
        FIND_MINMAX_LOCATIONS_LAST_COLUMN(cfg_loc, min_val, max_val, tile_min_cnt, tile_max_cnt, p_min_loc_x,
                                          p_min_loc_y, p_max_loc_x, p_max_loc_y, tile_offset_x, tile_offset_y,
                                          image_width);
    }
    else
    {
        FIND_MINMAX_LOCATIONS(cfg_loc, min_val, max_val, tile_min_cnt, tile_max_cnt, p_min_loc_x, p_min_loc_y,
                              p_max_loc_x, p_max_loc_y, tile_offset_x, tile_offset_y);
    }
}

int find_value_locations_check_last(MINMAX_LOC_CFG &cfg_loc, STYPE_LOC value, unsigned short *p_val_loc_x,
                                    unsigned short *p_val_loc_y, unsigned short tile_offset_x,
                                    unsigned short tile_offset_y, unsigned short image_width, int ntile_w, int ntile_h,
                                    int row_idx, int col_idx, bool has_last_column, bool last_row_loc_no_unroll);

#endif // MINMAXLOC_OPT_INTERFACE_H
