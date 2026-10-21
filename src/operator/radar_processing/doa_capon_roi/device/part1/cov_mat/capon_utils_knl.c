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

#include "capon_utils_knl.h"

#include "block_indices.h"

/* Convert one snapshot lane to float. GEN2 scales a plain cast by a precomputed
 * 2^-input_qbits; GEN3 uses the Q-aware cast. The INTERM MAC variants expose
 * independent products to the GEN3 scheduler. */
#if CUPVA_PVA_GEN_NUMBER < 3
#    define COV_FP_LOAD_SCALES(cfg) float scale_input_qbits = (cfg)->scale_input_qbits
#    define COV_FP_INPUT_TO_FP(x) dvmulf(dvint_dvfp(x), scale_input_qbits)
#    define COV_FP_CMAD_MAGSQ CMAD_MAGSQ
#    define COV_FP_CMAD_CONJ CMAD_CONJ
#else
#    define COV_FP_LOAD_SCALES(cfg) int input_qbits = (cfg)->input_qbits
#    define COV_FP_INPUT_TO_FP(x) dvint_dvfp(x, input_qbits)
#    define COV_FP_CMAD_MAGSQ CMAD_MAGSQ_INTERM
#    define COV_FP_CMAD_CONJ CMAD_CONJ_INTERM
#endif

/* Full and tail tile sizes are multiples of 4, so the directly unrolled
 * loops are valid for both AGEN configurations selected by is_last_tile. */

void compute_covariance_matrix_knl_fp_impl_diag(Para_capon_roi_part1_config *cfg, int32_t ntile, int32_t is_last_tile)
{
    float inv_n_samples = cfg->inv_n_samples;
    COV_FP_LOAD_SCALES(cfg);

    int n_samples_per_tile = is_last_tile ? cfg->n_samples_last_tile : cfg->n_samples_per_tile;
    SNAPSHOT_AGEN a_x_jk   = INIT_SNAPSHOT_AGEN(is_last_tile ? cfg->agen_cfg_cov_x_last_tile : cfg->agen_cfg_cov_x);

    // compute diagonal elements on 1 batch
    dvfloat COV_REAL_BANK *restrict L_batch_real_buf_ptr = (dvfloat COV_REAL_BANK *)cfg->L_real_buf_ptr;
    dvfloat COV_REAL_BANK *restrict L_real_element_ptr;

    dvintx dv_flag;
    dv_flag.lo = replicatew(ntile == 0);
    dv_flag.hi = replicatew(ntile == 0);

    chess_separator_scheduler();

    // compute the diagonals
    for (int j = 0; j < BUF_MAX_SV_DIM; j++)
    {
        LOAD_L_RE(j, j, A_jj_prev);
        dvfloatx A_jj_res = dvmux(dv_flag, 0, A_jj_prev);
        dvfloatx A_jj_inc;
        A_jj_inc.lo = replicatef(0.0f);
        A_jj_inc.hi = replicatef(0.0f);

        // A[j, j] = sum_{k=0}^{n_samples-1} X[j, k] * conj(X[j, k]) / n_samples
        for (int k = 0; k < n_samples_per_tile; k++)
#if CUPVA_PVA_GEN_NUMBER < 3
            chess_unroll_loop(2)
        chess_loop_range(6, )
#else
            chess_unroll_loop(4)
        chess_loop_range(12, )
#endif
        chess_prepare_for_pipelining
        {
            dvintx x_jk_re      = dvint_load_transp(a_x_jk);
            dvintx x_jk_im      = dvint_load_transp(a_x_jk);
            dvfloatx x_jk_fp_re = COV_FP_INPUT_TO_FP(x_jk_re);
            dvfloatx x_jk_fp_im = COV_FP_INPUT_TO_FP(x_jk_im);

            COV_FP_CMAD_MAGSQ(x_jk_fp_re, x_jk_fp_im, A_jj_inc);
        }

        A_jj_res += dvmulf(A_jj_inc, inv_n_samples);
        STORE_L_RE_AFTER_LOAD(A_jj_res);
    }
}

void compute_covariance_matrix_knl_fp_impl_off_diag(Para_capon_roi_part1_config *cfg, int32_t ntile,
                                                    int32_t is_last_tile)
{
    float inv_n_samples = cfg->inv_n_samples;
    COV_FP_LOAD_SCALES(cfg);

    int n_samples_per_tile   = is_last_tile ? cfg->n_samples_last_tile : cfg->n_samples_per_tile;
    AgenCFG &off_cfg         = is_last_tile ? cfg->agen_cfg_cov_x_last_tile : cfg->agen_cfg_cov_x;
    SNAPSHOT_AGEN a_x_jk_off = INIT_SNAPSHOT_AGEN(off_cfg);
    SNAPSHOT_AGEN a_x_ik_off = INIT_SNAPSHOT_AGEN(off_cfg);

    dvfloat COV_REAL_BANK *restrict L_real_element_ptr;
    dvfloat COV_IMAG_BANK *restrict L_imag_element_ptr;
    dvfloat COV_REAL_BANK *restrict L_batch_real_buf_ptr = (dvfloat COV_REAL_BANK *)cfg->L_real_buf_ptr;
    dvfloat COV_IMAG_BANK *restrict L_batch_imag_buf_ptr = (dvfloat COV_IMAG_BANK *)cfg->L_imag_buf_ptr;

    int32_t *x_batch_start_ptr = (int32_t *)cupvaGetAgenCfgBase(&off_cfg);

    dvintx dv_flag;
    dv_flag.lo = replicatew(ntile == 0);
    dv_flag.hi = replicatew(ntile == 0);

    chess_separator_scheduler();

    // compute off-diagonal elements
    for (int j = 0; j < BUF_MAX_SV_DIM; j++)
    {
        int element_offset_idx_j = j * 2;
        int32_t *x_jk_start_ptr  = x_batch_start_ptr + element_offset_idx_j;

        // idx i starts from j+1
        int element_offset_idx_i = element_offset_idx_j + 2;
        int32_t *x_ik_start_ptr  = x_batch_start_ptr + element_offset_idx_i;
        a_x_ik_off.a             = (dvint SNAPSHOT_BANK *restrict)x_ik_start_ptr;

        for (int i = j + 1; i < BUF_MAX_SV_DIM; i++)
        {
            a_x_jk_off.a = (dvint SNAPSHOT_BANK *restrict)x_jk_start_ptr;

            LOAD_L_RE(i, j, A_ij_prev_re);
            LOAD_L_IM(i, j, A_ij_prev_im);
            dvfloatx A_ij_res_re = dvmux(dv_flag, 0, A_ij_prev_re);
            dvfloatx A_ij_res_im = dvmux(dv_flag, 0, A_ij_prev_im);
            dvfloatx A_ij_inc_re;
            A_ij_inc_re.lo       = replicatef(0.0f);
            A_ij_inc_re.hi       = replicatef(0.0f);
            dvfloatx A_ij_inc_im = A_ij_inc_re;

            // A[i, j] = sum_{k=0}^{n_samples-1} X[i, k] * conj(X[j, k]) / n_samples
            for (int k = 0; k < n_samples_per_tile; k++)
#if CUPVA_PVA_GEN_NUMBER >= 3
                chess_unroll_loop(2)
#endif
            chess_loop_range(6, )
            chess_prepare_for_pipelining
            {
                dvintx x_jk_re = dvint_load_transp(a_x_jk_off);
                dvintx x_jk_im = dvint_load_transp(a_x_jk_off);
                dvintx x_ik_re = dvint_load_transp(a_x_ik_off);
                dvintx x_ik_im = dvint_load_transp(a_x_ik_off);

                dvfloatx x_jk_fp_re = COV_FP_INPUT_TO_FP(x_jk_re);
                dvfloatx x_jk_fp_im = COV_FP_INPUT_TO_FP(x_jk_im);
                dvfloatx x_ik_fp_re = COV_FP_INPUT_TO_FP(x_ik_re);
                dvfloatx x_ik_fp_im = COV_FP_INPUT_TO_FP(x_ik_im);

                // A[i,j] += X[i,k] * conj(X[j,k])
                COV_FP_CMAD_CONJ(x_ik_fp_re, x_ik_fp_im, x_jk_fp_re, x_jk_fp_im, A_ij_inc_re, A_ij_inc_im);
            }

            A_ij_res_re = dvmaddf(A_ij_inc_re, inv_n_samples, A_ij_res_re);
            A_ij_res_im = dvmaddf(A_ij_inc_im, inv_n_samples, A_ij_res_im);
            STORE_L_RE_AFTER_LOAD(A_ij_res_re);
            STORE_L_IM_AFTER_LOAD(A_ij_res_im);
        }
    }
}
