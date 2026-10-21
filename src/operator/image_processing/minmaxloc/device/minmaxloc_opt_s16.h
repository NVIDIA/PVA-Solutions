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

#ifndef MINMAXLOC_OPT_S16_H
#define MINMAXLOC_OPT_S16_H

#include "minmaxloc_common.h"

// main functions
void find_minmax_values_short(MINMAX_VAL_CFG &cfg, short *p_min_val, short *p_max_val);

int find_value_locations_short(MINMAX_LOC_CFG &cfg, short value, unsigned short *p_val_loc_x,
                               unsigned short *p_val_loc_y, unsigned short tile_offset_x, unsigned short tile_offset_y);

void find_minmax_locations_short(MINMAX_LOC_CFG &cfg, short min_val, short max_val, int &restrict min_cnt,
                                 int &restrict max_cnt, unsigned short *p_min_loc_x, unsigned short *p_min_loc_y,
                                 unsigned short *p_max_loc_x, unsigned short *p_max_loc_y, unsigned short tile_offset_x,
                                 unsigned short tile_offset_y);

// special kernels
int find_value_locations_short_last_column(MINMAX_LOC_CFG &cfg, short value, unsigned short *p_val_loc_x,
                                           unsigned short *p_val_loc_y, unsigned short tile_offset_x,
                                           unsigned short tile_offset_y, unsigned short image_width);

void find_minmax_locations_short_last_column(MINMAX_LOC_CFG &cfg, short min_val, short max_val, int &restrict min_cnt,
                                             int &restrict max_cnt, unsigned short *p_min_loc_x,
                                             unsigned short *p_min_loc_y, unsigned short *p_max_loc_x,
                                             unsigned short *p_max_loc_y, unsigned short tile_offset_x,
                                             unsigned short tile_offset_y, unsigned short image_width);

int find_value_locations_short_no_unroll(MINMAX_LOC_CFG &cfg, short value, unsigned short *p_val_loc_x,
                                         unsigned short *p_val_loc_y, unsigned short tile_offset_x,
                                         unsigned short tile_offset_y);

#endif // MINMAXLOC_OPT_S16_H