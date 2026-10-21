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

#ifndef COMPUTE_L_INV_TIMES_STEERING_MATRIX_HEADER_GUARD
#define COMPUTE_L_INV_TIMES_STEERING_MATRIX_HEADER_GUARD

#include <cupva_device.h> /* defines `restrict` macro so part-2 prototypes parse under native g++ */
#include <stdint.h>

struct Para_capon_roi_part2_config;

/**
 * @brief Initializes parameters for compute_L_inv_times_steering_matrix.
 *
 * The steering matrix is overwritten in-place with the result X = L_inv @ A.
 * Process the output from obtain_fixed_pt_L_inv and shift_fixed_pt_steering_matrix kernels.
 *
 * This is specialized for a 16x16 L_inv, 128 steering positions, and one
 * eight-ROI steering tile. L_inv and steering bases are initialized as dummy
 * bases and patched by update_part2_sv_agen_base before first use. The stable
 * L_inv base is recovered from obtain_fixed_pt_L_inv's output config.
 *
 * @param para      Output parameter struct to be initialized.
 */
void init_compute_L_inv_times_steering_matrix(Para_capon_roi_part2_config &para);

static inline void compute_L_inv_times_steering_matrix(const Para_capon_roi_part2_config &para)
{
    void compute_L_inv_times_steering_matrix_triangular_mat_mul(const Para_capon_roi_part2_config &para,
                                                                const bool is_upper_left_lower_triangular);

    void compute_L_inv_times_steering_matrix_rectangular_mat_mul(const Para_capon_roi_part2_config &para);

    // Step 1: lower-right triangle (in-place on lower half), clip 2^30 to 2^30 - 1 to avoid overflow later
    compute_L_inv_times_steering_matrix_triangular_mat_mul(para, false);
    // Step 2: lower-left rectangle (accumulate into lower half)
    compute_L_inv_times_steering_matrix_rectangular_mat_mul(para);
    // Step 3: upper-left triangle (in-place on upper half)
    compute_L_inv_times_steering_matrix_triangular_mat_mul(para, true);
}

#endif
