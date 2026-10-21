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

#include "capon_utils_knl_s32_nhw.h"

#include "block_indices.h"

/* Dequantize an accumulated fixed-point sum, normalize by 1/n_samples, and add it
 * onto res. GEN2 folds both factors into one precomputed scale; GEN3 uses the
 * Q-aware int-to-float cast followed by a multiply-add. */
#if CUPVA_PVA_GEN_NUMBER < 3
#    define COV_S32_LOAD_SCALES(cfg) float cov_scale_inv_n = (cfg)->scale_cov_sum_qbits_inv_n
#    define COV_S32_ACCUMULATE(acc, res) (res) += dvmulf(dvintx_dvfp(acc), cov_scale_inv_n)
#else
#    define COV_S32_LOAD_SCALES(cfg)                \
        int cov_sum_qbits   = (cfg)->cov_sum_qbits; \
        float inv_n_samples = (cfg)->inv_n_samples
#    define COV_S32_ACCUMULATE(acc, res) (res) = dvmaddf(dvintx_dvfp(acc, cov_sum_qbits), inv_n_samples, (res))
#endif

/* Full and tail tile widths are multiples of 4. */

void cov_mat_diag_s32_nhw(Para_capon_roi_part1_config *cfg, int32_t ntile, int32_t is_last_tile)
{
    int n_samples_per_tile = is_last_tile ? cfg->n_samples_last_tile : cfg->n_samples_per_tile;
    int cov_sum_shift_num  = cfg->cov_sum_shift_num;
    COV_S32_LOAD_SCALES(cfg);

    SNAPSHOT_AGEN a_x_jk = INIT_SNAPSHOT_AGEN(is_last_tile ? cfg->agen_cfg_cov_x_last_tile : cfg->agen_cfg_cov_x);

    dvfloat COV_REAL_BANK *restrict L_batch_real_buf_ptr = (dvfloat COV_REAL_BANK *)cfg->L_real_buf_ptr;
    dvfloat COV_REAL_BANK *restrict L_real_element_ptr;

    dvintx dv_flag;
    dv_flag.lo = replicatew(ntile == 0);
    dv_flag.hi = replicatew(ntile == 0);

    chess_separator_scheduler();

    for (int j = 0; j < BUF_MAX_SV_DIM; j++)
    {
        dvintx A_jj = chess_dont_care(dvintx) & 0;

        // A[j, j] = sum_{k=0}^{n_samples-1} X[j, k] * conj(X[j, k]) / n_samples
        for (int k = 0; k < n_samples_per_tile; k++) chess_unroll_loop(4)
        chess_loop_range(12, )
        chess_prepare_for_pipelining
        {
            dvintx x_jk_re = dvint_load_transp(a_x_jk);
            dvintx x_jk_im = dvint_load_transp(a_x_jk);

            dvintx inc = dvsumsq_t16(x_jk_re, x_jk_im);
            A_jj       = dvshiftadd(inc, cov_sum_shift_num, A_jj);
        }

        LOAD_L_RE(j, j, A_jj_prev);
        dvfloatx A_jj_res = dvmux(dv_flag, 0, A_jj_prev);

        COV_S32_ACCUMULATE(A_jj, A_jj_res);

        STORE_L_RE_AFTER_LOAD(A_jj_res);
    }
}

void cov_mat_off_diag_s32_nhw(Para_capon_roi_part1_config *cfg, int32_t ntile, int32_t is_last_tile)
{
    int n_samples_per_tile = is_last_tile ? cfg->n_samples_last_tile : cfg->n_samples_per_tile;
    int cov_sum_shift_num  = cfg->cov_sum_shift_num;
    COV_S32_LOAD_SCALES(cfg);

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

    for (int j = 0; j < BUF_MAX_SV_DIM; j++)
    {
        int element_offset_idx_j = j * 2;
        int32_t *x_jk_start_ptr  = x_batch_start_ptr + element_offset_idx_j;

        int element_offset_idx_i = element_offset_idx_j + 2;
        int32_t *x_ik_start_ptr  = x_batch_start_ptr + element_offset_idx_i;
        a_x_ik_off.a             = (dvint SNAPSHOT_BANK *restrict)x_ik_start_ptr;

        for (int i = j + 1; i < BUF_MAX_SV_DIM; i++)
        {
            a_x_jk_off.a = (dvint SNAPSHOT_BANK *restrict)x_jk_start_ptr;

            dvintx A_ij_re = chess_dont_care(dvintx) & 0;
            dvintx A_ij_im = chess_dont_care(dvintx) & 0;

            // A[i, j] = sum_k X[i, k] * conj(X[j, k]) / n_samples
            for (int k = 0; k < n_samples_per_tile; k++) chess_unroll_loop(2)
            chess_loop_range(6, )
            chess_prepare_for_pipelining
            {
                dvintx x_jk_re = dvint_load_transp(a_x_jk_off);
                dvintx x_jk_im = dvint_load_transp(a_x_jk_off);
                dvintx x_ik_re = dvint_load_transp(a_x_ik_off);
                dvintx x_ik_im = dvint_load_transp(a_x_ik_off);

                // (a + ib) * (c - id) = (ac + bd) + i(bc - ad)

                dvintx tmp_re_1 = dvmulw_t16(x_ik_re, x_jk_re);
                dvintx tmp_re_2 = dvmulw_t16(x_ik_im, x_jk_im);
                dvintx res_re   = tmp_re_1 + tmp_re_2;
                dvintx tmp_im_1 = dvmulw_t16(x_ik_im, x_jk_re);
                dvintx tmp_im_2 = dvmulw_t16(x_ik_re, x_jk_im);
                dvintx res_im   = tmp_im_1 - tmp_im_2;

                A_ij_re = dvshiftadd(res_re, cov_sum_shift_num, A_ij_re);
                A_ij_im = dvshiftadd(res_im, cov_sum_shift_num, A_ij_im);
            }

            /* Accumulate onto earlier tiles. */
            LOAD_L_RE(i, j, A_ij_prev_re);
            LOAD_L_IM(i, j, A_ij_prev_im);
            dvfloatx A_ij_res_re = dvmux(dv_flag, 0, A_ij_prev_re);
            dvfloatx A_ij_res_im = dvmux(dv_flag, 0, A_ij_prev_im);

            COV_S32_ACCUMULATE(A_ij_re, A_ij_res_re);
            COV_S32_ACCUMULATE(A_ij_im, A_ij_res_im);
            STORE_L_RE_AFTER_LOAD(A_ij_res_re);
            STORE_L_IM_AFTER_LOAD(A_ij_res_im);
        }
    }
}
