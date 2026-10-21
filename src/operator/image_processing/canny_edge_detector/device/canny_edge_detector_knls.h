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

#ifndef CANNY_EDGE_DETECTOR_KNLS_H
#define CANNY_EDGE_DETECTOR_KNLS_H

#include "canny_edge_detector_params.h"

#include <cupva_device.h>
#include <stdint.h>

struct Flt4x2Para
{
    AgenCFG c[5];
    int niter;
    int niter1;
};

struct Flt4Para
{
    AgenCFG c[4];
    int niter;
    int niter1;
};

struct EdgePara
{
    AgenCFG c[3];
    int niter;
};

void reformat_coef(int8_t *coef, const int8_t *coef1, const int8_t *coef2, int32_t kh);

void filter4x2_hor_short_init(const uint8_t *data, const int8_t *coef, int16_t *restrict out_x, int16_t *restrict out_y,
                              int32_t kh, int32_t qbits_x, int32_t qbits_y, int32_t blkw, int32_t blkh,
                              int32_t lofst_data, int32_t lofst_out, uint32_t cb_size_bytes, struct Flt4x2Para *para);

int32_t filter4x2_hor_short_exec(struct Flt4x2Para *para);

void filter4_ver_short_init(const int16_t *data, const int8_t *coef, int16_t *restrict out, int32_t kh, int32_t qbits,
                            int32_t blkw, int32_t blkh, int32_t lofst_data, int32_t lofst_out, struct Flt4Para *para);

int32_t filter4_ver_short_exec(struct Flt4Para *para);

void calc_intensity_L2_init(const int16_t *gx, const int16_t *gy, int32_t *restrict out, int32_t blkw, int32_t blkh,
                            int32_t lofst_data, int32_t lofst_out, struct EdgePara *para);

int32_t calc_intensity_L2_exec(struct EdgePara *para);

void edge_direction_init(const int16_t *gx, const int16_t *gy, int8_t *restrict out, int32_t blkw, int32_t blkh,
                         int32_t lofst_data, int32_t lofst_out, struct EdgePara *para);

int32_t edge_direction_exec(struct EdgePara *para);

void nms_double_thresholding_init(const int8_t *dir, const int32_t *intensity, uint8_t *restrict edge_type,
                                  int32_t blkw, int32_t blkh, int32_t lofst_dir, int32_t lofst_intensity,
                                  int32_t lofst_out, uint32_t cb_size_bytes, struct EdgePara *para);

int32_t nms_double_thresholding_exec(struct EdgePara *para, int32_t threshold_strong, int32_t threshold_weak);

#endif // CANNY_EDGE_DETECTOR_KNLS_H