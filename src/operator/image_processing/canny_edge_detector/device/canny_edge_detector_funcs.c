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

#include "canny_edge_detector_funcs.h"

#include "canny_edge_detector_params.h"

void canny_edge_detector_stage1_init(const uint8_t *image, int32_t *intensity, int8_t *edge_dir,
                                     struct Stage1BufPtrs *stage1_buf_ptrs, struct Flt4x2Para *flt4x2_para,
                                     struct Flt4Para *flt4_para, struct EdgePara *edge_para, int32_t gradient_size,
                                     int32_t image_lp, int32_t intensity_lp, int32_t edge_dir_lp,
                                     uint32_t image_cb_bytes)
{

    int8_t *coef_r      = stage1_buf_ptrs->coef_r;
    const int8_t *coef1 = stage1_buf_ptrs->coef1;
    const int8_t *coef2 = stage1_buf_ptrs->coef2;
    int16_t *gx_tmp     = stage1_buf_ptrs->gx_tmp;
    int16_t *gy_tmp     = stage1_buf_ptrs->gy_tmp;
    int16_t *gx         = stage1_buf_ptrs->gx;
    int16_t *gy         = stage1_buf_ptrs->gy;
    int32_t qbits       = gradient_size == 7 ? QBITS_7 : 0;

    reformat_coef(coef_r, coef1, coef2, gradient_size);

    filter4x2_hor_short_init(image, coef_r, gx_tmp, gy_tmp, KH_ALIGNED(gradient_size), 0, 0, TW, TH + gradient_size - 1,
                             image_lp, LINEPITCH, image_cb_bytes, flt4x2_para);

    filter4_ver_short_init(gx_tmp, coef_r + 32, gx, KH_ALIGNED(gradient_size), qbits, TW, TH, LINEPITCH, LINEPITCH_T,
                           flt4_para);
    filter4_ver_short_init(gy_tmp, coef_r, gy, KH_ALIGNED(gradient_size), qbits, TW, TH, LINEPITCH, LINEPITCH_T,
                           &flt4_para[1]);

    calc_intensity_L2_init(gx, gy, intensity, TH, TW, LINEPITCH_T, intensity_lp, edge_para);

    edge_direction_init(gx, gy, edge_dir, TH, TW, LINEPITCH_T, edge_dir_lp, &edge_para[1]);
}

int32_t canny_edge_detector_stage1_exec(struct Flt4x2Para *flt4x2_para, struct Flt4Para *flt4_para,
                                        struct EdgePara *edge_para)
{
    filter4x2_hor_short_exec(flt4x2_para);
    filter4_ver_short_exec(flt4_para);
    filter4_ver_short_exec(&flt4_para[1]);
    calc_intensity_L2_exec(edge_para);
    edge_direction_exec(&edge_para[1]);
    return 0;
}
