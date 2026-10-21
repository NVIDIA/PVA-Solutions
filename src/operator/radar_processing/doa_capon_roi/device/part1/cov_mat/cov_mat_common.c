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

#include "cov_mat/cov_mat_common.h"

#include "block_indices.h"

void compute_covariance_matrix_knl_nhw_init(Para_capon_roi_part1_config *cfg, int32_t n_samples_per_tile,
                                            float inv_n_samples,
#if !CAPON_ROI_USE_FP_COVMAT
                                            int32_t n_samples_bits,
#endif
                                            int32_t input_qbits, int32_t last_tile_doppler_bins)
{
    cfg->n_samples_per_tile  = n_samples_per_tile;
    cfg->n_samples_last_tile = last_tile_doppler_bins;

#if CAPON_ROI_USE_FP_COVMAT
    cfg->inv_n_samples = inv_n_samples;
#    if CUPVA_PVA_GEN_NUMBER < 3
    cfg->scale_input_qbits = as_float((127 - input_qbits) << 23);
#    else
    cfg->input_qbits = input_qbits;
#    endif
#else
    // Account for truncated complex multiplication and sample normalization.
    cfg->cov_sum_shift_num         = -n_samples_bits;
    int32_t cov_sum_qbits          = input_qbits * 2 - 16 + cfg->cov_sum_shift_num;
#    if CUPVA_PVA_GEN_NUMBER < 3
    cfg->scale_cov_sum_qbits_inv_n = inv_n_samples * as_float((127 - cov_sum_qbits) << 23);
#    else
    cfg->inv_n_samples = inv_n_samples;
    cfg->cov_sum_qbits = cov_sum_qbits;
#    endif
#endif

    AgenWrapper w0;
    w0.size = sizeof(int32_t);
    w0.n1   = 2; /* re/im parts */
    w0.s1   = 1;
    w0.n2   = n_samples_per_tile;
    w0.s2   = 2 * BUF_MAX_SV_DIM;
    // The off-diagonal traversal count also serves the shorter diagonal
    // kernel: it constructs a fresh AGEN and stops before this dimension wraps.
    w0.n3 = BUF_MAX_SV_DIM * (BUF_MAX_SV_DIM - 1) / 2;
    w0.s3 = 2;

    // The selected normal/tail configurations receive their live base before
    // every covariance call, including the first tile.
    agen a_x      = init((dvint *)0);
    a_x.lane_ofst = 2 * n_samples_per_tile;
    INIT_AGEN3(a_x, w0);
    cfg->agen_cfg_cov_x = extract_agen_cfg(a_x);

    if (last_tile_doppler_bins != 0)
    {
        w0.n2 = cfg->n_samples_last_tile;
        INIT_AGEN3(a_x, w0);
        cfg->agen_cfg_cov_x_last_tile = extract_agen_cfg(a_x);
    }
}

void add_diagonal_loading_knl_fp_impl(Para_capon_roi_part1_config *cfg)
{
    // svDim is fixed at 16, so scale trace by exactly 2^-4.
    float scale_const = cfg->adp_diag_const * 0x1p-4f;

    dvfloat COV_REAL_BANK *restrict L_batch_real_buf_ptr = (dvfloat COV_REAL_BANK *)cfg->L_real_buf_ptr;
    dvfloat COV_REAL_BANK *restrict L_real_element_ptr;

    dvfloatx dv_diag_loading;
    dv_diag_loading.lo = replicatef(0.0f);
    dv_diag_loading.hi = replicatef(0.0f);

    for (int32_t j = 0; j < BUF_MAX_SV_DIM; j++) chess_unroll_loop(4)
    chess_loop_range(12, )
    chess_prepare_for_pipelining
    {
        LOAD_L_RE(j, j, A_jj_res);
        dv_diag_loading += A_jj_res;
    }

    chess_separator_scheduler();

    dv_diag_loading = dvmulf(dv_diag_loading, scale_const);

    dvfloatx dv_one;
    dv_one.lo = replicatef(1.0f);
    dv_one.hi = replicatef(1.0f);
    // Set padded lanes to one to avoid reciprocal-square-root traps.
    dv_diag_loading = dvmux(dv_diag_loading > 0, dv_diag_loading, dv_one);

    // Explicit groups expose four independent diagonal loads to the scheduler.
    for (int32_t j = 0; j < BUF_MAX_SV_DIM; j += 4) chess_loop_range(3, )
    chess_prepare_for_pipelining
    {
        dvfloat COV_REAL_BANK *p0 = L_batch_real_buf_ptr + L_REAL_IDX(j + 0, j + 0);
        dvfloat COV_REAL_BANK *p1 = L_batch_real_buf_ptr + L_REAL_IDX(j + 1, j + 1);
        dvfloat COV_REAL_BANK *p2 = L_batch_real_buf_ptr + L_REAL_IDX(j + 2, j + 2);
        dvfloat COV_REAL_BANK *p3 = L_batch_real_buf_ptr + L_REAL_IDX(j + 3, j + 3);

        dvfloatx A_j0_j0_res = sign_extend(*p0);
        dvfloatx A_j1_j1_res = sign_extend(*p1);
        dvfloatx A_j2_j2_res = sign_extend(*p2);
        dvfloatx A_j3_j3_res = sign_extend(*p3);

        A_j0_j0_res += dv_diag_loading;
        A_j1_j1_res += dv_diag_loading;
        A_j2_j2_res += dv_diag_loading;
        A_j3_j3_res += dv_diag_loading;

        *p0 = extract(A_j0_j0_res);
        *p1 = extract(A_j1_j1_res);
        *p2 = extract(A_j2_j2_res);
        *p3 = extract(A_j3_j3_res);
    }
}
