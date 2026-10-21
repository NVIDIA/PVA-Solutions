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

#ifndef COV_MAT_COMMON_H
#define COV_MAT_COMMON_H

#include "capon_roi_part1_config_types.h"

#include <cupva_device.h>

/* Select the covariance implementation for this device target. */
#if CAPON_ROI_USE_FP_COVMAT
#    include "cov_mat/capon_utils_knl.h"
#    define COMPUTE_COVMAT_DIAG_IMPL compute_covariance_matrix_knl_fp_impl_diag
#    define COMPUTE_COVMAT_OFF_DIAG_IMPL compute_covariance_matrix_knl_fp_impl_off_diag
#else
#    include "cov_mat/capon_utils_knl_s32_nhw.h"
#    define COMPUTE_COVMAT_DIAG_IMPL cov_mat_diag_s32_nhw
#    define COMPUTE_COVMAT_OFF_DIAG_IMPL cov_mat_off_diag_s32_nhw
#endif

#ifdef __cplusplus
extern "C" {
#endif

void compute_covariance_matrix_knl_nhw_init(Para_capon_roi_part1_config *cfg, int32_t n_samples_per_tile,
                                            float inv_n_samples,
#if !CAPON_ROI_USE_FP_COVMAT
                                            int32_t n_samples_bits,
#endif
                                            int32_t input_qbits, int32_t last_tile_doppler_bins);

void add_diagonal_loading_knl_fp_impl(Para_capon_roi_part1_config *cfg);

#ifdef __cplusplus
}
#endif

#endif /* COV_MAT_COMMON_H */
