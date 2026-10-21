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

#ifndef DISTANCE_TRANSFORM_KNL_H
#define DISTANCE_TRANSFORM_KNL_H
#include <cupva_device.h>

typedef struct
{
    uint16_t *tp;
    int32_t tw;
    int32_t th;
    int32_t lp;
} VX_USHORT_TILE;

typedef struct param_dt_stage1
{
    AgenCFG cfgs[4];
    int niter;
    int max_dist;
    int w;
} param_dt_stage1_t;

int distance_transform_stage1_init(VX_USHORT_TILE *tile, int niter, int max_dist, param_dt_stage1_t *param);

int distance_transform_stage1_l2r(param_dt_stage1_t *param);

int distance_transform_stage1_r2l(param_dt_stage1_t *param);

int distance_transform_stage2_sweep1(int tw, int th, int iw, int ih, int max_dist, unsigned short *horz_dist,
                                     unsigned short *v_arr, unsigned short *output_dist, unsigned short *output_label,
                                     unsigned short *sqrt_tbl, dvshortx *k_out);

int distance_transform_stage2_sweep2(int tw, int th, int iw, int ih, int max_dist, unsigned short *horz_dist,
                                     unsigned short *v_arr, unsigned short *output_dist, unsigned short *output_label,
                                     unsigned short *sqrt_tbl, dvshortx *k_in, dvshortx *k2_in, dvshortx *q_start,
                                     dvshortx *q_end);

#endif
