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

#include "block_cholesky_linv_bs4_knl.h"
#include "block_indices.h"

#define BS4_CMSUB_INTERM(a_re, a_im, b_re, b_im, acc_re, acc_im) \
    dvfloatx _t_re_1 = dvmulf(a_re, b_re);                       \
    dvfloatx _t_re_2 = dvmulf(a_im, b_im);                       \
    acc_re -= _t_re_1 - _t_re_2;                                 \
    dvfloatx _t_im_1 = dvmulf(a_re, b_im);                       \
    dvfloatx _t_im_2 = dvmulf(a_im, b_re);                       \
    acc_im -= _t_im_1 + _t_im_2;

#if CUPVA_PVA_GEN_NUMBER < 3
#    define BS4_LINV_DEP_CMAD_0 CMAD_INTERM
#    define BS4_LINV_DEP_CMAD_1 CMAD
#    define BS4_LINV_DEP_CMAD_2 CMAD
#    define BS4_LINV_DEP_CMAD_3 CMAD_INTERM
#    define BS4_LINV_COMPUTE_CMSUB_R3 CMSUB
#    define BS4_LINV_COMPUTE_CMSUB_R2 CMSUB
#    define BS4_LINV_COMPUTE_CMSUB_R1 CMSUB
#    define BS4_LINV_RELOAD_I3_I01                                                             \
        dvfloatx P_i3_i0_re = sign_extend(*(Linv_compute_reload_real_ptr + LINV_IDX(i3, i0))); \
        dvfloatx P_i3_i0_im = sign_extend(*(Linv_compute_reload_imag_ptr + LINV_IDX(i3, i0))); \
        dvfloatx P_i3_i1_re = sign_extend(*(Linv_compute_reload_real_ptr + LINV_IDX(i3, i1))); \
        dvfloatx P_i3_i1_im = sign_extend(*(Linv_compute_reload_imag_ptr + LINV_IDX(i3, i1)));
#else
#    define BS4_LINV_DEP_CMAD_0 CMAD_INTERM
#    define BS4_LINV_DEP_CMAD_1 CMAD_INTERM
#    define BS4_LINV_DEP_CMAD_2 CMAD_INTERM
#    define BS4_LINV_DEP_CMAD_3 CMAD_INTERM
#    define BS4_LINV_COMPUTE_CMSUB_R3 BS4_CMSUB_INTERM
#    define BS4_LINV_COMPUTE_CMSUB_R2 CMSUB
#    define BS4_LINV_COMPUTE_CMSUB_R1 CMSUB
#    define BS4_LINV_RELOAD_I3_I01                                                             \
        dvfloatx P_i3_i0_re = sign_extend(*(Linv_compute_reload_real_ptr + LINV_IDX(i3, i0))); \
        dvfloatx P_i3_i0_im = sign_extend(*(Linv_compute_reload_imag_ptr + LINV_IDX(i3, i0))); \
        dvfloatx P_i3_i1_re = sign_extend(*(Linv_compute_reload_real_ptr + LINV_IDX(i3, i1))); \
        dvfloatx P_i3_i1_im = sign_extend(*(Linv_compute_reload_imag_ptr + LINV_IDX(i3, i1)));
#endif

/* Apply the diagonal-block contribution P_lj = L_lj * P_jj. */

#define UPDATE_DIAG_ROW(l)                                                    \
    LOAD_L_RE(l, j0, L_l_j0_re);                                              \
    LOAD_L_IM(l, j0, L_l_j0_im);                                              \
    LOAD_L_RE(l, j1, L_l_j1_re);                                              \
    LOAD_L_IM(l, j1, L_l_j1_im);                                              \
    LOAD_L_RE(l, j2, L_l_j2_re);                                              \
    LOAD_L_IM(l, j2, L_l_j2_im);                                              \
    LOAD_L_RE(l, j3, L_l_j3_re);                                              \
    LOAD_L_IM(l, j3, L_l_j3_im);                                              \
    CMUL_REAL(L_l_j0_re, L_l_j0_im, P_j0_j0_re, P_l_j0_re, P_l_j0_im);        \
    CMAD(L_l_j1_re, L_l_j1_im, P_j1_j0_re, P_j1_j0_im, P_l_j0_re, P_l_j0_im); \
    CMAD(L_l_j2_re, L_l_j2_im, P_j2_j0_re, P_j2_j0_im, P_l_j0_re, P_l_j0_im); \
    CMAD(L_l_j3_re, L_l_j3_im, P_j3_j0_re, P_j3_j0_im, P_l_j0_re, P_l_j0_im); \
    STORE_LINV_RE(l, j0, P_l_j0_re);                                          \
    STORE_LINV_IM(l, j0, P_l_j0_im);                                          \
    CMUL_REAL(L_l_j1_re, L_l_j1_im, P_j1_j1_re, P_l_j1_re, P_l_j1_im);        \
    CMAD(L_l_j2_re, L_l_j2_im, P_j2_j1_re, P_j2_j1_im, P_l_j1_re, P_l_j1_im); \
    CMAD(L_l_j3_re, L_l_j3_im, P_j3_j1_re, P_j3_j1_im, P_l_j1_re, P_l_j1_im); \
    STORE_LINV_RE(l, j1, P_l_j1_re);                                          \
    STORE_LINV_IM(l, j1, P_l_j1_im);                                          \
    CMUL_REAL(L_l_j2_re, L_l_j2_im, P_j2_j2_re, P_l_j2_re, P_l_j2_im);        \
    CMAD(L_l_j3_re, L_l_j3_im, P_j3_j2_re, P_j3_j2_im, P_l_j2_re, P_l_j2_im); \
    STORE_LINV_RE(l, j2, P_l_j2_re);                                          \
    STORE_LINV_IM(l, j2, P_l_j2_im);                                          \
    CMUL_REAL(L_l_j3_re, L_l_j3_im, P_j3_j3_re, P_l_j3_re, P_l_j3_im);        \
    STORE_LINV_RE(l, j3, P_l_j3_re);                                          \
    STORE_LINV_IM(l, j3, P_l_j3_im);

void update_diag_bs4(Para_capon_roi_part1_config *cfg, dvfloat COV_REAL_BANK *restrict L_batch_real_buf_ptr,
                     dvfloat COV_IMAG_BANK *restrict L_batch_imag_buf_ptr,
                     dvfloat LINV_REAL_BANK *restrict Linv_batch_real_buf_ptr,
                     dvfloat LINV_IMAG_BANK *restrict Linv_batch_imag_buf_ptr,
                     dvfloat COV_REAL_BANK *restrict L_real_element_ptr,
                     dvfloat COV_IMAG_BANK *restrict L_imag_element_ptr,
                     dvfloat LINV_REAL_BANK *restrict Linv_real_element_ptr,
                     dvfloat LINV_IMAG_BANK *restrict Linv_imag_element_ptr, int j, int n)
{
    int j0 = j, j1 = j + 1, j2 = j + 2, j3 = j + 3;

    LOAD_LINV_RE(j0, j0, P_j0_j0_re);
    LOAD_LINV_RE(j1, j0, P_j1_j0_re);
    LOAD_LINV_IM(j1, j0, P_j1_j0_im);
    LOAD_LINV_RE(j1, j1, P_j1_j1_re);
    LOAD_LINV_RE(j2, j0, P_j2_j0_re);
    LOAD_LINV_IM(j2, j0, P_j2_j0_im);
    LOAD_LINV_RE(j2, j1, P_j2_j1_re);
    LOAD_LINV_IM(j2, j1, P_j2_j1_im);
    LOAD_LINV_RE(j2, j2, P_j2_j2_re);
    LOAD_LINV_RE(j3, j0, P_j3_j0_re);
    LOAD_LINV_IM(j3, j0, P_j3_j0_im);
    LOAD_LINV_RE(j3, j1, P_j3_j1_re);
    LOAD_LINV_IM(j3, j1, P_j3_j1_im);
    LOAD_LINV_RE(j3, j2, P_j3_j2_re);
    LOAD_LINV_IM(j3, j2, P_j3_j2_im);
    LOAD_LINV_RE(j3, j3, P_j3_j3_re);

    for (int l = j + 4; l < n; l++) chess_loop_range(4, )
    chess_prepare_for_pipelining
    {
        UPDATE_DIAG_ROW(l);
    }
}

/* Apply the off-diagonal contribution P_lj += L_li * P_ij. */

#define UPDATE_OFFDIAG_ELEMENT(l, j)                                                         \
    LOAD_L_RE(l, i0, L_l_i0_re);                                                             \
    LOAD_L_IM(l, i0, L_l_i0_im);                                                             \
    LOAD_L_RE(l, i1, L_l_i1_re);                                                             \
    LOAD_L_IM(l, i1, L_l_i1_im);                                                             \
    LOAD_L_RE(l, i2, L_l_i2_re);                                                             \
    LOAD_L_IM(l, i2, L_l_i2_im);                                                             \
    LOAD_L_RE(l, i3, L_l_i3_re);                                                             \
    LOAD_L_IM(l, i3, L_l_i3_im);                                                             \
    LOAD_LINV_RE(l, j, P_l_j_re);                                                            \
    LOAD_LINV_IM(l, j, P_l_j_im);                                                            \
    {                                                                                        \
        BS4_LINV_DEP_CMAD_0(L_l_i0_re, L_l_i0_im, P_i0_j_re, P_i0_j_im, P_l_j_re, P_l_j_im); \
    }                                                                                        \
    {                                                                                        \
        BS4_LINV_DEP_CMAD_1(L_l_i1_re, L_l_i1_im, P_i1_j_re, P_i1_j_im, P_l_j_re, P_l_j_im); \
    }                                                                                        \
    {                                                                                        \
        BS4_LINV_DEP_CMAD_2(L_l_i2_re, L_l_i2_im, P_i2_j_re, P_i2_j_im, P_l_j_re, P_l_j_im); \
    }                                                                                        \
    {                                                                                        \
        BS4_LINV_DEP_CMAD_3(L_l_i3_re, L_l_i3_im, P_i3_j_re, P_i3_j_im, P_l_j_re, P_l_j_im); \
    }                                                                                        \
    STORE_LINV_RE_AFTER_LOAD(P_l_j_re);                                                      \
    STORE_LINV_IM_AFTER_LOAD(P_l_j_im);

void update_offdiag_bs4(Para_capon_roi_part1_config *cfg, dvfloat COV_REAL_BANK *restrict L_batch_real_buf_ptr,
                        dvfloat COV_IMAG_BANK *restrict L_batch_imag_buf_ptr,
                        dvfloat LINV_REAL_BANK *restrict Linv_batch_real_buf_ptr,
                        dvfloat LINV_IMAG_BANK *restrict Linv_batch_imag_buf_ptr,
                        dvfloat COV_REAL_BANK *restrict L_real_element_ptr,
                        dvfloat COV_IMAG_BANK *restrict L_imag_element_ptr,
                        dvfloat LINV_REAL_BANK *restrict Linv_real_element_ptr,
                        dvfloat LINV_IMAG_BANK *restrict Linv_imag_element_ptr, int i, int j, int n)
{
    int i0 = i, i1 = i + 1, i2 = i + 2, i3 = i + 3;

    for (int jj = j; jj < j + 4; jj++)
    {
        LOAD_LINV_RE(i0, jj, P_i0_j_re);
        LOAD_LINV_IM(i0, jj, P_i0_j_im);
        LOAD_LINV_RE(i1, jj, P_i1_j_re);
        LOAD_LINV_IM(i1, jj, P_i1_j_im);
        LOAD_LINV_RE(i2, jj, P_i2_j_re);
        LOAD_LINV_IM(i2, jj, P_i2_j_im);
        LOAD_LINV_RE(i3, jj, P_i3_j_re);
        LOAD_LINV_IM(i3, jj, P_i3_j_im);

        for (int l = i + 4; l < n; l++) chess_loop_range(4, )
        chess_prepare_for_pipelining
        {
            UPDATE_OFFDIAG_ELEMENT(l, jj);
        }
    }
}

/* Compute P_ij = -(invL_ii * P_ij) for one off-diagonal block. */

#define COMPUTE_BLOCK_COL_BS4(j)                                                                       \
    LOAD_LINV_RE(i2, j, P_i2_j_re);                                                                    \
    LOAD_LINV_IM(i2, j, P_i2_j_im);                                                                    \
    LOAD_LINV_RE(i3, j, P_i3_j_re);                                                                    \
    LOAD_LINV_IM(i3, j, P_i3_j_im);                                                                    \
    CMUL_REAL_INPLACE(P_i3_j_re, P_i3_j_im, -P_i3_i3_re);                                              \
    {                                                                                                  \
        BS4_LINV_COMPUTE_CMSUB_R3(P_i3_i2_re, P_i3_i2_im, P_i2_j_re, P_i2_j_im, P_i3_j_re, P_i3_j_im); \
    }                                                                                                  \
    CMUL_REAL_INPLACE(P_i2_j_re, P_i2_j_im, -P_i2_i2_re);                                              \
    LOAD_LINV_RE(i0, j, P_i0_j_re);                                                                    \
    LOAD_LINV_IM(i0, j, P_i0_j_im);                                                                    \
    LOAD_LINV_RE(i1, j, P_i1_j_re);                                                                    \
    LOAD_LINV_IM(i1, j, P_i1_j_im);                                                                    \
    BS4_LINV_RELOAD_I3_I01                                                                             \
    {                                                                                                  \
        BS4_LINV_COMPUTE_CMSUB_R3(P_i3_i0_re, P_i3_i0_im, P_i0_j_re, P_i0_j_im, P_i3_j_re, P_i3_j_im); \
    }                                                                                                  \
    {                                                                                                  \
        BS4_LINV_COMPUTE_CMSUB_R3(P_i3_i1_re, P_i3_i1_im, P_i1_j_re, P_i1_j_im, P_i3_j_re, P_i3_j_im); \
    }                                                                                                  \
    STORE_LINV_RE(i3, j, P_i3_j_re);                                                                   \
    STORE_LINV_IM(i3, j, P_i3_j_im);                                                                   \
    {                                                                                                  \
        BS4_LINV_COMPUTE_CMSUB_R2(P_i2_i0_re, P_i2_i0_im, P_i0_j_re, P_i0_j_im, P_i2_j_re, P_i2_j_im); \
    }                                                                                                  \
    {                                                                                                  \
        BS4_LINV_COMPUTE_CMSUB_R2(P_i2_i1_re, P_i2_i1_im, P_i1_j_re, P_i1_j_im, P_i2_j_re, P_i2_j_im); \
    }                                                                                                  \
    STORE_LINV_RE(i2, j, P_i2_j_re);                                                                   \
    STORE_LINV_IM(i2, j, P_i2_j_im);                                                                   \
    CMUL_REAL_INPLACE(P_i1_j_re, P_i1_j_im, -P_i1_i1_re);                                              \
    {                                                                                                  \
        BS4_LINV_COMPUTE_CMSUB_R1(P_i1_i0_re, P_i1_i0_im, P_i0_j_re, P_i0_j_im, P_i1_j_re, P_i1_j_im); \
    }                                                                                                  \
    STORE_LINV_RE(i1, j, P_i1_j_re);                                                                   \
    STORE_LINV_IM(i1, j, P_i1_j_im);                                                                   \
    CMUL_REAL_INPLACE(P_i0_j_re, P_i0_j_im, -P_i0_i0_re);                                              \
    STORE_LINV_RE(i0, j, P_i0_j_re);                                                                   \
    STORE_LINV_IM(i0, j, P_i0_j_im);

void compute_block_bs4(Para_capon_roi_part1_config *cfg, dvfloat LINV_REAL_BANK *restrict Linv_batch_real_buf_ptr,
                       dvfloat LINV_IMAG_BANK *restrict Linv_batch_imag_buf_ptr,
                       dvfloat LINV_REAL_BANK *restrict Linv_real_element_ptr,
                       dvfloat LINV_IMAG_BANK *restrict Linv_imag_element_ptr, int i, int j)
{
    int i0 = i, i1 = i + 1, i2 = i + 2, i3 = i + 3;
    int j0 = j, j1 = j + 1, j2 = j + 2, j3 = j + 3;

    /* Reload two row-3 coefficients per column to cap full-unroll register pressure. */
    chess_protect_access dvfloat LINV_REAL_BANK *restrict Linv_compute_reload_real_ptr = Linv_batch_real_buf_ptr;
    chess_protect_access dvfloat LINV_IMAG_BANK *restrict Linv_compute_reload_imag_ptr = Linv_batch_imag_buf_ptr;

    LOAD_LINV_RE(i0, i0, P_i0_i0_re);
    LOAD_LINV_RE(i1, i0, P_i1_i0_re);
    LOAD_LINV_IM(i1, i0, P_i1_i0_im);
    LOAD_LINV_RE(i1, i1, P_i1_i1_re);
    LOAD_LINV_RE(i2, i0, P_i2_i0_re);
    LOAD_LINV_IM(i2, i0, P_i2_i0_im);
    LOAD_LINV_RE(i2, i1, P_i2_i1_re);
    LOAD_LINV_IM(i2, i1, P_i2_i1_im);
    LOAD_LINV_RE(i2, i2, P_i2_i2_re);
    LOAD_LINV_RE(i3, i2, P_i3_i2_re);
    LOAD_LINV_IM(i3, i2, P_i3_i2_im);
    LOAD_LINV_RE(i3, i3, P_i3_i3_re);

    for (int jj = j0; jj <= j3; jj++) chess_loop_count(4)
    chess_unroll_loop(*)
    {
        COMPUTE_BLOCK_COL_BS4(jj);
    }
}

/* Full block L-inverse driver. */
void linv_opt_full_driver_bs4(Para_capon_roi_part1_config *cfg)
{
    int n                                        = BUF_MAX_SV_DIM;
    dvfloat COV_REAL_BANK *restrict L_re_ptr     = (dvfloat COV_REAL_BANK *)cfg->L_real_buf_ptr;
    dvfloat COV_IMAG_BANK *restrict L_im_ptr     = (dvfloat COV_IMAG_BANK *)cfg->L_imag_buf_ptr;
    dvfloat LINV_REAL_BANK *restrict Linv_re_ptr = (dvfloat LINV_REAL_BANK *)cfg->Linv_real_buf_ptr;
    dvfloat LINV_IMAG_BANK *restrict Linv_im_ptr = (dvfloat LINV_IMAG_BANK *)cfg->Linv_imag_buf_ptr;
    dvfloat COV_REAL_BANK *restrict L_real_element_ptr;
    dvfloat COV_IMAG_BANK *restrict L_imag_element_ptr;
    dvfloat LINV_REAL_BANK *restrict Linv_real_element_ptr;
    dvfloat LINV_IMAG_BANK *restrict Linv_imag_element_ptr;

    chess_separator_scheduler();

    for (int j = 0; j < n - 4; j += 4)
    {
        update_diag_bs4(cfg, L_re_ptr, L_im_ptr, Linv_re_ptr, Linv_im_ptr, L_real_element_ptr, L_imag_element_ptr,
                        Linv_real_element_ptr, Linv_imag_element_ptr, j, n);

        for (int i = j + 4; i < n - 4; i += 4)
        {
            compute_block_bs4(cfg, Linv_re_ptr, Linv_im_ptr, Linv_real_element_ptr, Linv_imag_element_ptr, i, j);
            update_offdiag_bs4(cfg, L_re_ptr, L_im_ptr, Linv_re_ptr, Linv_im_ptr, L_real_element_ptr,
                               L_imag_element_ptr, Linv_real_element_ptr, Linv_imag_element_ptr, i, j, n);
        }
        compute_block_bs4(cfg, Linv_re_ptr, Linv_im_ptr, Linv_real_element_ptr, Linv_imag_element_ptr, n - 4, j);
    }
}
