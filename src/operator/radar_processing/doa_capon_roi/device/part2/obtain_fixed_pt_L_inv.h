/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef OBTAIN_FIXED_PT_L_INV_HEADER_GUARD
#define OBTAIN_FIXED_PT_L_INV_HEADER_GUARD

#include <cupva_device.h> /* defines `restrict` macro so part-2 prototypes parse under native g++ */
#include <stdint.h>

struct Para_capon_roi_part2_config;

/**
 * @brief Initializes parameters for obtain_fixed_pt_L_inv.
 *
 * Reads L_inv in three-block order and writes [real, imag, -imag] to L_inv_fixed_pt_out in fixed-point
 * SQ1.28 using frcp(max per ROI)*2^28 as scale.
 *
 * This is specialized for a 16x16 lower-triangular matrix and one 16-ROI
 * group. The current L_inv input bases are passed directly to stage 0, and
 * the implementation privately owns the fixed offset traversal table.
 *
 * @param scale_out           Output scale values (frcp(max per ROI)*2^28), the C layout is float[roi_num].
 *                            It is required for get_top3_peaks kernel.
 * @param L_inv_fixed_pt_out  Output fixed point L_inv, the C layout is int32_t[roi_num][line_pitch].
 *                            line_pitch = smallest chess_elementsof(dvfloat)*K+1 >= 3L, for transposition.
 * @param para                Output parameter struct to be initialized.
 */
void init_obtain_fixed_pt_L_inv(float *const restrict scale_out, int32_t *const restrict L_inv_fixed_pt_out,
                                Para_capon_roi_part2_config &para);

static inline void obtain_fixed_pt_L_inv(const float *const restrict L_inv_real, const float *const restrict L_inv_imag,
                                         const Para_capon_roi_part2_config &para)
{
    void obtain_fixed_pt_L_inv_stage0(const float *const restrict, const float *const restrict,
                                      const Para_capon_roi_part2_config &);
    void obtain_fixed_pt_L_inv_stage1(const Para_capon_roi_part2_config &);
    obtain_fixed_pt_L_inv_stage0(L_inv_real, L_inv_imag, para);
    obtain_fixed_pt_L_inv_stage1(para);
}

#endif
