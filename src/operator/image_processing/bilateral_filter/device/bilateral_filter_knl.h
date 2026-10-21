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

#ifndef BILATERAL_FILTER_KNL_H
#define BILATERAL_FILTER_KNL_H

#include "bilateral_filter_param.h"

#include <cupva_device.h>
#include <stdint.h>

typedef struct BilateralFilterConfig
{
    AgenCFG cfg_ld_img_corner;
    AgenCFG cfg_ld_img_square;
    AgenCFG cfg_ld_img_central;
    AgenCFG cfg_st_value;
    AgenCFG cfg_st_weight;
    AgenCFG cfg_ld_value;
    AgenCFG cfg_ld_weight;
    AgenCFG cfg_st_img;

    int32_t *lookupOffsets;
    int32_t niter;
} BilateralFilterConfig;

void replicate_table_char_32(uint8_t *tbl_out, uint8_t *tbl_in, int32_t size);

void bilateral_filter_init(int32_t *lookupOffsets, uint32_t *value_sum, uint32_t *weight_sum, int32_t cb_line_pitch,
                           int32_t cb_size, uint8_t *cb_start_addr, int32_t lastTileMaskValid,
                           BilateralFilterConfig *cfg);

void bilateral_filter_diff_stage(BilateralFilterConfig *cfg);

void bilateral_filter_accumulate_stage(uint8_t *weight_tbl, BilateralFilterConfig *cfg);

void bilateral_filter_division_stage(BilateralFilterConfig *cfg);

#endif