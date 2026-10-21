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

#ifndef CANNY_EDGE_DETECTOR_FUNCS_H
#define CANNY_EDGE_DETECTOR_FUNCS_H

#include "canny_edge_detector_knls.h"

#include <cupva_device.h>

struct Stage1BufPtrs
{
    int8_t *coef_r;
    const int8_t *coef1;
    const int8_t *coef2;
    int16_t *gx_tmp;
    int16_t *gy_tmp;
    int16_t *gx;
    int16_t *gy;
};

void canny_edge_detector_stage1_init(const uint8_t *image, int32_t *intensity, int8_t *edge_dir,
                                     struct Stage1BufPtrs *stage1_buf_ptrs, struct Flt4x2Para *flt4x2_para,
                                     struct Flt4Para *flt4_para, struct EdgePara *edge_para, int32_t gradient_size,
                                     int32_t image_lp, int32_t intensity_lp, int32_t edge_dir_lp,
                                     uint32_t image_cb_bytes);

int32_t canny_edge_detector_stage1_exec(struct Flt4x2Para *flt4x2_para, struct Flt4Para *flt4_para,
                                        struct EdgePara *edge_para);

#endif // CANNY_EDGE_DETECTOR_FUNCS_H