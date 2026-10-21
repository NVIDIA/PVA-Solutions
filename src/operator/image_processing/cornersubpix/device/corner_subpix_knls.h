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

#ifndef _CORNER_SUBPIX_KNLS_H_
#define _CORNER_SUBPIX_KNLS_H_

#include "corner_subpix_params.h"

#include <cupva_device.h>

typedef struct
{
    float *corners;
    int32_t *corner_x;
    int32_t *corner_y;
    int32_t *gxx;
    int32_t *gxy;
    int32_t *gyy;
    int32_t *mask;
    float *matrix;
    float *bb;
    int32_t win_size;
    int32_t tile_size;
    int32_t tile_pitch;
    int32_t tol;
    GSDFHandler *gsdf_handler;
    SequenceDataFlowHandler *in_handler;
    SequenceDataFlowHandler *out_handler;
    AgenCFG hessian_cfgs[5];
    int32_t image_niter;
} param_corner_subpix_t;

typedef struct
{
    uint8_t *img_u8;
    uint16_t *img_u16;
    uint16_t *subpix_img;
    AgenCFG image_cfgs[5];
    CupvaSampler *dlut_cfg;
    int32_t *dlut_matrix;
    int32_t *x_new;
    int32_t *y_new;
    int32_t *x_sat_lo;
    int32_t *y_sat_lo;
    int32_t *x_sat_hi;
    int32_t *y_sat_hi;
    int32_t *x_start;
    int32_t *y_start;
    int32_t *x_ofst;
    int32_t *y_ofst;
    int32_t all_converged[2];
    int32_t status[BATCH];
    int32_t converge[BATCH];
} buf_corner_subpix_t;

void image_u8_u16_init(param_corner_subpix_t *param, buf_corner_subpix_t *buf);
void bilinear_interpolation_index_init(buf_corner_subpix_t *buf);
void hessian_matrix_init(param_corner_subpix_t *param, buf_corner_subpix_t *buf);

int32_t image_u8_u16_exec(param_corner_subpix_t *param, buf_corner_subpix_t *buf);
int32_t bilinear_interpolation_index_exec(buf_corner_subpix_t *buf);
int32_t hessian_matrix_exec(param_corner_subpix_t *param, buf_corner_subpix_t *buf);
int32_t compute_motion_exec(param_corner_subpix_t *param, buf_corner_subpix_t *buf, const int32_t img_w,
                            const int32_t img_h, const float eps);
int32_t corner_conversion_preprocess(param_corner_subpix_t *param, const int32_t num_corners,
                                     const int32_t num_corners_input);
int32_t corner_conversion_postprocess(param_corner_subpix_t *param, const int32_t num_corners);

#endif