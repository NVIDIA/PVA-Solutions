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

#ifndef CAPON_ROI_PART1_CONFIG_TYPES_H
#define CAPON_ROI_PART1_CONFIG_TYPES_H

#include "capon_roi_vmem_layout.h" /* BUF_MAX_SV_DIM */

#include <cupva_device.h>
#include <stdint.h>

// Part-1 VMEM placement. The *_BANK macros are complete storage qualifiers,
// used as `dvfloat COV_REAL_BANK *p`; the *_BANK_ID macros carry the bare bank
// letter for the places that need it.
#define SNAPSHOT_BANK chess_storage(RAM_Cb)
#define SNAPSHOT_BANK_ID C
#define SNAPSHOT_AGEN agen_C
#define INIT_SNAPSHOT_AGEN init_agen_C_from_cfg

#define COV_REAL_BANK chess_storage(RAM_Ab)
#define COV_REAL_BANK_ID A
#define COV_IMAG_BANK chess_storage(RAM_Cb)
#define COV_IMAG_BANK_ID C
#define LINV_REAL_BANK chess_storage(RAM_Ab)
#define LINV_REAL_BANK_ID A
#define LINV_IMAG_BANK chess_storage(RAM_Bb)
#define LINV_IMAG_BANK_ID B

// Shared covariance, Cholesky, and inverse-Cholesky configuration.
struct Para_capon_roi_part1_config
{
#if CAPON_ROI_USE_FP_COVMAT || (defined(CUPVA_PVA_GEN_NUMBER) && CUPVA_PVA_GEN_NUMBER >= 3)
    float inv_n_samples;
#endif
    int n_samples_per_tile;
    float *L_real_buf_ptr;
    float *L_imag_buf_ptr;
    float *Linv_real_buf_ptr;
    float *Linv_imag_buf_ptr;
#if CAPON_ROI_USE_FP_COVMAT && defined(CUPVA_PVA_GEN_NUMBER) && CUPVA_PVA_GEN_NUMBER >= 3
    int input_qbits;
#endif

    AgenCFG agen_cfg_cov_x;

    // last tile agen
    int n_samples_last_tile;
    AgenCFG agen_cfg_cov_x_last_tile;

    float adp_diag_const;

#if !CAPON_ROI_USE_FP_COVMAT
    int cov_sum_shift_num;
#    if defined(CUPVA_PVA_GEN_NUMBER) && CUPVA_PVA_GEN_NUMBER >= 3
    int cov_sum_qbits;
#    endif
#endif

    // GEN2 Q-format scales; GEN3 uses native Q-format conversion intrinsics.
#if !defined(CUPVA_PVA_GEN_NUMBER) || (CUPVA_PVA_GEN_NUMBER < 3)
#    if CAPON_ROI_USE_FP_COVMAT
    float scale_input_qbits;
#    else
    float scale_cov_sum_qbits_inv_n;
#    endif
#endif
};

typedef struct Para_capon_roi_part1_config Para_capon_roi_part1_config;

#endif /* CAPON_ROI_PART1_CONFIG_TYPES_H */
