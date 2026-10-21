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

/* Cholesky and L-inverse for one 4x4 diagonal block. */
#if CUPVA_PVA_GEN_NUMBER < 3
#    define BS4_DIAG_CMSUB_MAGSQ CMSUB_MAGSQ
#else
#    define BS4_DIAG_CMSUB_MAGSQ CMSUB_MAGSQ_INTERM
#endif

void diag_block_cholesky_bs4(Para_capon_roi_part1_config *cfg, dvfloat COV_REAL_BANK *restrict L_batch_real_buf_ptr,
                             dvfloat COV_IMAG_BANK *restrict L_batch_imag_buf_ptr,
                             dvfloat LINV_REAL_BANK *restrict Linv_batch_real_buf_ptr,
                             dvfloat LINV_IMAG_BANK *restrict Linv_batch_imag_buf_ptr,
                             dvfloat COV_REAL_BANK *restrict L_real_element_ptr,
                             dvfloat COV_IMAG_BANK *restrict L_imag_element_ptr,
                             dvfloat LINV_REAL_BANK *restrict Linv_real_element_ptr,
                             dvfloat LINV_IMAG_BANK *restrict Linv_imag_element_ptr, int32_t s)
{
    int32_t j0 = s, j1 = s + 1, j2 = s + 2, j3 = s + 3;

    LOAD_L_RE(j0, j0, A_j0_j0_re);
    LOAD_L_RE(j1, j0, A_j1_j0_re);
    LOAD_L_IM(j1, j0, A_j1_j0_im);
    LOAD_L_RE(j1, j1, A_j1_j1_re);

    dvfloatx inv_L_j0_j0_re = dvfrsq(A_j0_j0_re);
    STORE_LINV_RE(j0, j0, inv_L_j0_j0_re);

    LOAD_L_RE(j2, j0, A_j2_j0_re);
    LOAD_L_IM(j2, j0, A_j2_j0_im);

    CMUL_REAL(A_j1_j0_re, A_j1_j0_im, inv_L_j0_j0_re, L_j1_j0_re, L_j1_j0_im);
    STORE_L_RE(j1, j0, L_j1_j0_re);
    STORE_L_IM(j1, j0, L_j1_j0_im);

    LOAD_L_RE(j3, j0, A_j3_j0_re);
    LOAD_L_IM(j3, j0, A_j3_j0_im);

    {
        BS4_DIAG_CMSUB_MAGSQ(L_j1_j0_re, L_j1_j0_im, A_j1_j1_re);
    }
    dvfloatx inv_L_j1_j1_re = dvfrsq(A_j1_j1_re);
    STORE_LINV_RE(j1, j1, inv_L_j1_j1_re);

    LOAD_L_RE(j2, j1, A_j2_j1_re);
    LOAD_L_IM(j2, j1, A_j2_j1_im);

    CMUL_REAL(A_j2_j0_re, A_j2_j0_im, inv_L_j0_j0_re, L_j2_j0_re, L_j2_j0_im);
    STORE_L_RE(j2, j0, L_j2_j0_re);
    STORE_L_IM(j2, j0, L_j2_j0_im);

    LOAD_L_RE(j3, j1, A_j3_j1_re);
    LOAD_L_IM(j3, j1, A_j3_j1_im);

    CMUL_REAL(A_j3_j0_re, A_j3_j0_im, inv_L_j0_j0_re, L_j3_j0_re, L_j3_j0_im);
    STORE_L_RE(j3, j0, L_j3_j0_re);
    STORE_L_IM(j3, j0, L_j3_j0_im);

    CMSUB_CONJ(L_j2_j0_re, L_j2_j0_im, L_j1_j0_re, L_j1_j0_im, A_j2_j1_re, A_j2_j1_im);
    CMUL_REAL(A_j2_j1_re, A_j2_j1_im, inv_L_j1_j1_re, L_j2_j1_re, L_j2_j1_im);
    STORE_L_RE(j2, j1, L_j2_j1_re);
    STORE_L_IM(j2, j1, L_j2_j1_im);

    LOAD_L_RE(j2, j2, A_j2_j2_re);
    LOAD_L_RE(j3, j2, A_j3_j2_re);
    LOAD_L_IM(j3, j2, A_j3_j2_im);
    LOAD_L_RE(j3, j3, A_j3_j3_re);

    CMSUB_CONJ(L_j3_j0_re, L_j3_j0_im, L_j1_j0_re, L_j1_j0_im, A_j3_j1_re, A_j3_j1_im);
    CMUL_REAL(A_j3_j1_re, A_j3_j1_im, inv_L_j1_j1_re, L_j3_j1_re, L_j3_j1_im);
    STORE_L_RE(j3, j1, L_j3_j1_re);
    STORE_L_IM(j3, j1, L_j3_j1_im);

    /* Update the lower-right 2x2 Schur block. */
    {
        BS4_DIAG_CMSUB_MAGSQ(L_j2_j0_re, L_j2_j0_im, A_j2_j2_re);
    }
    CMSUB_CONJ(L_j3_j0_re, L_j3_j0_im, L_j2_j0_re, L_j2_j0_im, A_j3_j2_re, A_j3_j2_im);
    {
        BS4_DIAG_CMSUB_MAGSQ(L_j3_j0_re, L_j3_j0_im, A_j3_j3_re);
    }
    {
        BS4_DIAG_CMSUB_MAGSQ(L_j2_j1_re, L_j2_j1_im, A_j2_j2_re);
    }
    CMSUB_CONJ(L_j3_j1_re, L_j3_j1_im, L_j2_j1_re, L_j2_j1_im, A_j3_j2_re, A_j3_j2_im);
    {
        BS4_DIAG_CMSUB_MAGSQ(L_j3_j1_re, L_j3_j1_im, A_j3_j3_re);
    }
    dvfloatx inv_L_j2_j2_re = dvfrsq(A_j2_j2_re);
    STORE_LINV_RE(j2, j2, inv_L_j2_j2_re);

    CMUL_REAL(A_j3_j2_re, A_j3_j2_im, inv_L_j2_j2_re, L_j3_j2_re, L_j3_j2_im);
    STORE_L_RE(j3, j2, L_j3_j2_re);
    STORE_L_IM(j3, j2, L_j3_j2_im);

    {
        BS4_DIAG_CMSUB_MAGSQ(L_j3_j2_re, L_j3_j2_im, A_j3_j3_re);
    }
    dvfloatx inv_L_j3_j3_re = dvfrsq(A_j3_j3_re);
    STORE_LINV_RE(j3, j3, inv_L_j3_j3_re);
}

#if CUPVA_PVA_GEN_NUMBER < 3
#    define BS4_DIAG_LINV_DEPENDENT_CMAD CMAD
#else
#    define BS4_DIAG_LINV_DEPENDENT_CMAD CMAD_INTERM
#endif

void diag_block_linv_bs4(Para_capon_roi_part1_config *cfg, dvfloat COV_REAL_BANK *restrict L_batch_real_buf_ptr,
                         dvfloat COV_IMAG_BANK *restrict L_batch_imag_buf_ptr,
                         dvfloat LINV_REAL_BANK *restrict Linv_batch_real_buf_ptr,
                         dvfloat LINV_IMAG_BANK *restrict Linv_batch_imag_buf_ptr,
                         dvfloat COV_REAL_BANK *restrict L_real_element_ptr,
                         dvfloat COV_IMAG_BANK *restrict L_imag_element_ptr,
                         dvfloat LINV_REAL_BANK *restrict Linv_real_element_ptr,
                         dvfloat LINV_IMAG_BANK *restrict Linv_imag_element_ptr, int32_t s)
{
    int32_t j0 = s, j1 = s + 1, j2 = s + 2, j3 = s + 3;

    LOAD_L_RE(j1, j0, L_j1_j0_re);
    LOAD_L_IM(j1, j0, L_j1_j0_im);
    LOAD_LINV_RE(j0, j0, Linv_j0_j0_re);
    LOAD_LINV_RE(j1, j1, Linv_j1_j1_re);

    dvfloatx dv_zero;
    dv_zero.lo = replicatef(0.0f);
    dv_zero.hi = replicatef(0.0f);
    /* The inverse diagonal is real. */
    STORE_LINV_IM(j0, j0, dv_zero);
    STORE_LINV_IM(j1, j1, dv_zero);
    STORE_LINV_IM(j2, j2, dv_zero);
    STORE_LINV_IM(j3, j3, dv_zero);

    LOAD_L_RE(j2, j0, L_j2_j0_re);
    LOAD_L_IM(j2, j0, L_j2_j0_im);
    LOAD_L_RE(j2, j1, L_j2_j1_re);
    LOAD_L_IM(j2, j1, L_j2_j1_im);
    LOAD_LINV_RE(j2, j2, Linv_j2_j2_re);

    dvfloatx scale = -dvmulf(Linv_j0_j0_re, Linv_j1_j1_re);
    CMUL_REAL(L_j1_j0_re, L_j1_j0_im, scale, Linv_j1_j0_re, Linv_j1_j0_im);
    STORE_LINV_RE(j1, j0, Linv_j1_j0_re);
    STORE_LINV_IM(j1, j0, Linv_j1_j0_im);

    LOAD_L_RE(j3, j0, L_j3_j0_re);
    LOAD_L_IM(j3, j0, L_j3_j0_im);
    LOAD_L_RE(j3, j1, L_j3_j1_re);
    LOAD_L_IM(j3, j1, L_j3_j1_im);

    CMUL_REAL(L_j2_j0_re, L_j2_j0_im, Linv_j0_j0_re, Linv_j2_j0_re, Linv_j2_j0_im);
    CMAD(Linv_j1_j0_re, Linv_j1_j0_im, L_j2_j1_re, L_j2_j1_im, Linv_j2_j0_re, Linv_j2_j0_im);
    CMUL_REAL_INPLACE(Linv_j2_j0_re, Linv_j2_j0_im, -Linv_j2_j2_re);
    STORE_LINV_RE(j2, j0, Linv_j2_j0_re);
    STORE_LINV_IM(j2, j0, Linv_j2_j0_im);

    LOAD_L_RE(j3, j2, L_j3_j2_re);
    LOAD_L_IM(j3, j2, L_j3_j2_im);
    LOAD_LINV_RE(j3, j3, Linv_j3_j3_re);

    CMUL_REAL(L_j3_j0_re, L_j3_j0_im, Linv_j0_j0_re, Linv_j3_j0_re, Linv_j3_j0_im);
    CMAD(Linv_j1_j0_re, Linv_j1_j0_im, L_j3_j1_re, L_j3_j1_im, Linv_j3_j0_re, Linv_j3_j0_im);
    {
        BS4_DIAG_LINV_DEPENDENT_CMAD(Linv_j2_j0_re, Linv_j2_j0_im, L_j3_j2_re, L_j3_j2_im, Linv_j3_j0_re,
                                     Linv_j3_j0_im);
    }
    CMUL_REAL_INPLACE(Linv_j3_j0_re, Linv_j3_j0_im, -Linv_j3_j3_re);
    STORE_LINV_RE(j3, j0, Linv_j3_j0_re);
    STORE_LINV_IM(j3, j0, Linv_j3_j0_im);

    CMUL_REAL(L_j2_j1_re, L_j2_j1_im, Linv_j1_j1_re, Linv_j2_j1_re, Linv_j2_j1_im);
    CMUL_REAL_INPLACE(Linv_j2_j1_re, Linv_j2_j1_im, -Linv_j2_j2_re);
    STORE_LINV_RE(j2, j1, Linv_j2_j1_re);
    STORE_LINV_IM(j2, j1, Linv_j2_j1_im);

    CMUL_REAL(L_j3_j1_re, L_j3_j1_im, Linv_j1_j1_re, Linv_j3_j1_re, Linv_j3_j1_im);
    {
        BS4_DIAG_LINV_DEPENDENT_CMAD(Linv_j2_j1_re, Linv_j2_j1_im, L_j3_j2_re, L_j3_j2_im, Linv_j3_j1_re,
                                     Linv_j3_j1_im);
    }
    CMUL_REAL_INPLACE(Linv_j3_j1_re, Linv_j3_j1_im, -Linv_j3_j3_re);
    STORE_LINV_RE(j3, j1, Linv_j3_j1_re);
    STORE_LINV_IM(j3, j1, Linv_j3_j1_im);

    CMUL_REAL(L_j3_j2_re, L_j3_j2_im, Linv_j2_j2_re, Linv_j3_j2_re, Linv_j3_j2_im);
    CMUL_REAL_INPLACE(Linv_j3_j2_re, Linv_j3_j2_im, -Linv_j3_j3_re);
    STORE_LINV_RE(j3, j2, Linv_j3_j2_re);
    STORE_LINV_IM(j3, j2, Linv_j3_j2_im);
}

/* Off-diagonal block: L_xj = A_xj * Linv_jj^H. */

#if CUPVA_PVA_GEN_NUMBER >= 3
#    define BS4_OFFDIAG_PTR_ACCESS chess_protect_access
#    define BS4_OFFDIAG_J3_CMAD_CONJ CMAD_CONJ_INTERM
#else
#    define BS4_OFFDIAG_PTR_ACCESS
#    define BS4_OFFDIAG_J3_CMAD_CONJ CMAD_CONJ
#endif

void off_diag_bs4(Para_capon_roi_part1_config *cfg, dvfloat COV_REAL_BANK *restrict L_batch_real_buf_ptr,
                  dvfloat COV_IMAG_BANK *restrict L_batch_imag_buf_ptr,
                  dvfloat LINV_REAL_BANK *restrict Linv_batch_real_buf_ptr,
                  dvfloat LINV_IMAG_BANK *restrict Linv_batch_imag_buf_ptr,
                  dvfloat COV_REAL_BANK *restrict L_real_element_ptr,
                  dvfloat COV_IMAG_BANK *restrict L_imag_element_ptr,
                  dvfloat LINV_REAL_BANK *restrict Linv_real_element_ptr,
                  dvfloat LINV_IMAG_BANK *restrict Linv_imag_element_ptr, int32_t s, int32_t n)
{
    int32_t j0 = s, j1 = s + 1, j2 = s + 2, j3 = s + 3;

    LOAD_LINV_RE(j0, j0, Linv_j0_j0_re);
    LOAD_LINV_RE(j1, j0, Linv_j1_j0_re);
    LOAD_LINV_IM(j1, j0, Linv_j1_j0_im);
    LOAD_LINV_RE(j1, j1, Linv_j1_j1_re);
    LOAD_LINV_RE(j2, j0, Linv_j2_j0_re);
    LOAD_LINV_IM(j2, j0, Linv_j2_j0_im);
    LOAD_LINV_RE(j2, j1, Linv_j2_j1_re);
    LOAD_LINV_IM(j2, j1, Linv_j2_j1_im);
    LOAD_LINV_RE(j2, j2, Linv_j2_j2_re);
    LOAD_LINV_RE(j3, j0, Linv_j3_j0_re);
    LOAD_LINV_IM(j3, j0, Linv_j3_j0_im);
    LOAD_LINV_RE(j3, j1, Linv_j3_j1_re);
    LOAD_LINV_IM(j3, j1, Linv_j3_j1_im);
    LOAD_LINV_RE(j3, j2, Linv_j3_j2_re);
    LOAD_LINV_IM(j3, j2, Linv_j3_j2_im);
    LOAD_LINV_RE(j3, j3, Linv_j3_j3_re);

    int32_t row_start                                               = s + 4;
    BS4_OFFDIAG_PTR_ACCESS dvfloat COV_REAL_BANK *restrict pL_re_j0 = L_batch_real_buf_ptr + L_REAL_IDX(row_start, j0);
    BS4_OFFDIAG_PTR_ACCESS dvfloat COV_IMAG_BANK *restrict pL_im_j0 = L_batch_imag_buf_ptr + L_IMAG_IDX(row_start, j0);
    BS4_OFFDIAG_PTR_ACCESS dvfloat COV_REAL_BANK *restrict pL_re_j1 = L_batch_real_buf_ptr + L_REAL_IDX(row_start, j1);
    BS4_OFFDIAG_PTR_ACCESS dvfloat COV_IMAG_BANK *restrict pL_im_j1 = L_batch_imag_buf_ptr + L_IMAG_IDX(row_start, j1);
    BS4_OFFDIAG_PTR_ACCESS dvfloat COV_REAL_BANK *restrict pL_re_j2 = L_batch_real_buf_ptr + L_REAL_IDX(row_start, j2);
    BS4_OFFDIAG_PTR_ACCESS dvfloat COV_IMAG_BANK *restrict pL_im_j2 = L_batch_imag_buf_ptr + L_IMAG_IDX(row_start, j2);
    BS4_OFFDIAG_PTR_ACCESS dvfloat COV_REAL_BANK *restrict pL_re_j3 = L_batch_real_buf_ptr + L_REAL_IDX(row_start, j3);
    BS4_OFFDIAG_PTR_ACCESS dvfloat COV_IMAG_BANK *restrict pL_im_j3 = L_batch_imag_buf_ptr + L_IMAG_IDX(row_start, j3);

    for (int32_t i = row_start; i < n; i++) chess_loop_range(4, )
    chess_prepare_for_pipelining
    {
        dvfloatx A_i_j0_re = sign_extend(*pL_re_j0);
        dvfloatx A_i_j0_im = sign_extend(*pL_im_j0);
        dvfloatx A_i_j1_re = sign_extend(*pL_re_j1);
        dvfloatx A_i_j1_im = sign_extend(*pL_im_j1);
        dvfloatx A_i_j2_re = sign_extend(*pL_re_j2);
        dvfloatx A_i_j2_im = sign_extend(*pL_im_j2);
        dvfloatx A_i_j3_re = sign_extend(*pL_re_j3);
        dvfloatx A_i_j3_im = sign_extend(*pL_im_j3);

        CMUL_REAL(A_i_j0_re, A_i_j0_im, Linv_j0_j0_re, L_i_j0_re, L_i_j0_im);
        *pL_re_j0 = extract(L_i_j0_re);
        *pL_im_j0 = extract(L_i_j0_im);

        CMUL_CONJ(A_i_j0_re, A_i_j0_im, Linv_j1_j0_re, Linv_j1_j0_im, L_i_j1_re, L_i_j1_im);
        CMAD_REAL(A_i_j1_re, A_i_j1_im, Linv_j1_j1_re, L_i_j1_re, L_i_j1_im);
        *pL_re_j1 = extract(L_i_j1_re);
        *pL_im_j1 = extract(L_i_j1_im);

        CMUL_CONJ(A_i_j0_re, A_i_j0_im, Linv_j2_j0_re, Linv_j2_j0_im, L_i_j2_re, L_i_j2_im);
        CMAD_CONJ(A_i_j1_re, A_i_j1_im, Linv_j2_j1_re, Linv_j2_j1_im, L_i_j2_re, L_i_j2_im);
        CMAD_REAL(A_i_j2_re, A_i_j2_im, Linv_j2_j2_re, L_i_j2_re, L_i_j2_im);
        *pL_re_j2 = extract(L_i_j2_re);
        *pL_im_j2 = extract(L_i_j2_im);

        CMUL_CONJ(A_i_j0_re, A_i_j0_im, Linv_j3_j0_re, Linv_j3_j0_im, L_i_j3_re, L_i_j3_im);
        {
            BS4_OFFDIAG_J3_CMAD_CONJ(A_i_j1_re, A_i_j1_im, Linv_j3_j1_re, Linv_j3_j1_im, L_i_j3_re, L_i_j3_im);
        }
        {
            BS4_OFFDIAG_J3_CMAD_CONJ(A_i_j2_re, A_i_j2_im, Linv_j3_j2_re, Linv_j3_j2_im, L_i_j3_re, L_i_j3_im);
        }
        CMAD_REAL(A_i_j3_re, A_i_j3_im, Linv_j3_j3_re, L_i_j3_re, L_i_j3_im);
        *pL_re_j3 = extract(L_i_j3_re);
        *pL_im_j3 = extract(L_i_j3_im);

        pL_re_j0++;
        pL_im_j0++;
        pL_re_j1++;
        pL_im_j1++;
        pL_re_j2++;
        pL_im_j2++;
        pL_re_j3++;
        pL_im_j3++;
    }
}

/* Update one Schur-complement column: A_il -= L_ij * L_lj^H. */
/* Two-stage concatenation expands nested arguments before token pasting. */
#define BS4_CAT_(a, b) a##b
#define BS4_CAT(a, b) BS4_CAT_(a, b)
#define BS4_L_RE(i, j) BS4_CAT(L_, BS4_CAT(i, BS4_CAT(_, BS4_CAT(j, _re))))
#define BS4_L_IM(i, j) BS4_CAT(L_, BS4_CAT(i, BS4_CAT(_, BS4_CAT(j, _im))))

#define LOAD_BS4_OFF_DIAG_ROW(i)       \
    LOAD_L_RE(i, j0, BS4_L_RE(i, j0)); \
    LOAD_L_IM(i, j0, BS4_L_IM(i, j0)); \
    LOAD_L_RE(i, j1, BS4_L_RE(i, j1)); \
    LOAD_L_IM(i, j1, BS4_L_IM(i, j1)); \
    LOAD_L_RE(i, j2, BS4_L_RE(i, j2)); \
    LOAD_L_IM(i, j2, BS4_L_IM(i, j2)); \
    LOAD_L_RE(i, j3, BS4_L_RE(i, j3)); \
    LOAD_L_IM(i, j3, BS4_L_IM(i, j3));

#if CUPVA_PVA_GEN_NUMBER < 3
#    define BS4_SCHUR_CMSUB_MAGSQ_0 CMSUB_MAGSQ
#    define BS4_SCHUR_CMSUB_MAGSQ_1 CMSUB_MAGSQ_INTERM
#    define BS4_SCHUR_CMSUB_MAGSQ_2 CMSUB_MAGSQ
#    define BS4_SCHUR_CMSUB_MAGSQ_3 CMSUB_MAGSQ_INTERM
#else
#    define BS4_SCHUR_CMSUB_MAGSQ_0 CMSUB_MAGSQ_INTERM
#    define BS4_SCHUR_CMSUB_MAGSQ_1 CMSUB_MAGSQ_INTERM
#    define BS4_SCHUR_CMSUB_MAGSQ_2 CMSUB_MAGSQ_INTERM
#    define BS4_SCHUR_CMSUB_MAGSQ_3 CMSUB_MAGSQ_INTERM
#endif

#define SCHUR_UPDATE_ELEMENT_DIAG(i)                                               \
    LOAD_L_RE(i, i, BS4_L_RE(i, i));                                               \
    {                                                                              \
        BS4_SCHUR_CMSUB_MAGSQ_0(BS4_L_RE(i, j0), BS4_L_IM(i, j0), BS4_L_RE(i, i)); \
    }                                                                              \
    {                                                                              \
        BS4_SCHUR_CMSUB_MAGSQ_1(BS4_L_RE(i, j1), BS4_L_IM(i, j1), BS4_L_RE(i, i)); \
    }                                                                              \
    {                                                                              \
        BS4_SCHUR_CMSUB_MAGSQ_2(BS4_L_RE(i, j2), BS4_L_IM(i, j2), BS4_L_RE(i, i)); \
    }                                                                              \
    {                                                                              \
        BS4_SCHUR_CMSUB_MAGSQ_3(BS4_L_RE(i, j3), BS4_L_IM(i, j3), BS4_L_RE(i, i)); \
    }                                                                              \
    STORE_L_RE_AFTER_LOAD(BS4_L_RE(i, i));

/* GEN2 needs two dependency breaks to software-pipeline the pointer walk.
 * GEN3 can overlap all four independent multiply pairs. */
#if CUPVA_PVA_GEN_NUMBER < 3
#    define BS4_SCHUR_CMSUB_CONJ_0 CMSUB_CONJ
#    define BS4_SCHUR_CMSUB_CONJ_1 CMSUB_CONJ_INTERM
#    define BS4_SCHUR_CMSUB_CONJ_2 CMSUB_CONJ
#    define BS4_SCHUR_CMSUB_CONJ_3 CMSUB_CONJ_INTERM
#    define BS4_SCHUR_LAST2_CMSUB_MAGSQ CMSUB_MAGSQ
#    define BS4_SCHUR_LAST2_CMSUB_CONJ_0 CMSUB_CONJ
#    define BS4_SCHUR_LAST2_CMSUB_CONJ_1 CMSUB_CONJ
#    define BS4_SCHUR_LAST2_CMSUB_CONJ_2 CMSUB_CONJ
#    define BS4_SCHUR_LAST2_CMSUB_CONJ_3 CMSUB_CONJ
#else
#    define BS4_SCHUR_CMSUB_CONJ_0 CMSUB_CONJ_INTERM
#    define BS4_SCHUR_CMSUB_CONJ_1 CMSUB_CONJ_INTERM
#    define BS4_SCHUR_CMSUB_CONJ_2 CMSUB_CONJ_INTERM
#    define BS4_SCHUR_CMSUB_CONJ_3 CMSUB_CONJ_INTERM
#    define BS4_SCHUR_LAST2_CMSUB_MAGSQ CMSUB_MAGSQ_INTERM
#    define BS4_SCHUR_LAST2_CMSUB_CONJ_0 CMSUB_CONJ_INTERM
#    define BS4_SCHUR_LAST2_CMSUB_CONJ_1 CMSUB_CONJ_INTERM
#    define BS4_SCHUR_LAST2_CMSUB_CONJ_2 CMSUB_CONJ_INTERM
#    define BS4_SCHUR_LAST2_CMSUB_CONJ_3 CMSUB_CONJ_INTERM
#endif

#define SCHUR_UPDATE_ELEMENT_OFF_DIAG(i, l)                                                                        \
    LOAD_L_RE(i, l, BS4_L_RE(i, l));                                                                               \
    LOAD_L_IM(i, l, BS4_L_IM(i, l));                                                                               \
    {                                                                                                              \
        BS4_SCHUR_CMSUB_CONJ_0(BS4_L_RE(i, j0), BS4_L_IM(i, j0), BS4_L_RE(l, j0), BS4_L_IM(l, j0), BS4_L_RE(i, l), \
                               BS4_L_IM(i, l));                                                                    \
    }                                                                                                              \
    {                                                                                                              \
        BS4_SCHUR_CMSUB_CONJ_1(BS4_L_RE(i, j1), BS4_L_IM(i, j1), BS4_L_RE(l, j1), BS4_L_IM(l, j1), BS4_L_RE(i, l), \
                               BS4_L_IM(i, l));                                                                    \
    }                                                                                                              \
    {                                                                                                              \
        BS4_SCHUR_CMSUB_CONJ_2(BS4_L_RE(i, j2), BS4_L_IM(i, j2), BS4_L_RE(l, j2), BS4_L_IM(l, j2), BS4_L_RE(i, l), \
                               BS4_L_IM(i, l));                                                                    \
    }                                                                                                              \
    {                                                                                                              \
        BS4_SCHUR_CMSUB_CONJ_3(BS4_L_RE(i, j3), BS4_L_IM(i, j3), BS4_L_RE(l, j3), BS4_L_IM(l, j3), BS4_L_RE(i, l), \
                               BS4_L_IM(i, l));                                                                    \
    }                                                                                                              \
    STORE_L_RE_AFTER_LOAD(BS4_L_RE(i, l));                                                                         \
    STORE_L_IM_AFTER_LOAD(BS4_L_IM(i, l));

#define SCHUR_UPDATE_LAST_2_TERM(j, cmsub_op)                                         \
    LOAD_L_RE(i0, j, L_i0_j_re);                                                      \
    LOAD_L_IM(i0, j, L_i0_j_im);                                                      \
    LOAD_L_RE(i1, j, L_i1_j_re);                                                      \
    LOAD_L_IM(i1, j, L_i1_j_im);                                                      \
    {                                                                                 \
        BS4_SCHUR_LAST2_CMSUB_MAGSQ(L_i0_j_re, L_i0_j_im, A_i0_i0_re);                \
    }                                                                                 \
    {                                                                                 \
        BS4_SCHUR_LAST2_CMSUB_MAGSQ(L_i1_j_re, L_i1_j_im, A_i1_i1_re);                \
    }                                                                                 \
    {                                                                                 \
        cmsub_op(L_i1_j_re, L_i1_j_im, L_i0_j_re, L_i0_j_im, A_i1_i0_re, A_i1_i0_im); \
    }

void schur_update_col_last_2(Para_capon_roi_part1_config *cfg, dvfloat COV_REAL_BANK *restrict L_batch_real_buf_ptr,
                             dvfloat COV_IMAG_BANK *restrict L_batch_imag_buf_ptr,
                             dvfloat COV_REAL_BANK *restrict L_real_element_ptr,
                             dvfloat COV_IMAG_BANK *restrict L_imag_element_ptr, int32_t s, int32_t n)
{
    /* Update the final 2x2 Schur block. */
    int32_t j0 = s, j1 = s + 1, j2 = s + 2, j3 = s + 3;
    int32_t i0 = n - 2, i1 = n - 1;

    dvfloat COV_REAL_BANK *restrict pA_i0_i0_re = L_batch_real_buf_ptr + L_REAL_IDX(i0, i0);
    dvfloat COV_REAL_BANK *restrict pA_i1_i1_re = L_batch_real_buf_ptr + L_REAL_IDX(i1, i1);
    dvfloat COV_REAL_BANK *restrict pA_i1_i0_re = L_batch_real_buf_ptr + L_REAL_IDX(i1, i0);
    dvfloat COV_IMAG_BANK *restrict pA_i1_i0_im = L_batch_imag_buf_ptr + L_IMAG_IDX(i1, i0);

    dvfloatx A_i0_i0_re = sign_extend(*pA_i0_i0_re);
    dvfloatx A_i1_i1_re = sign_extend(*pA_i1_i1_re);
    dvfloatx A_i1_i0_re = sign_extend(*pA_i1_i0_re);
    dvfloatx A_i1_i0_im = sign_extend(*pA_i1_i0_im);

    {
        SCHUR_UPDATE_LAST_2_TERM(j0, BS4_SCHUR_LAST2_CMSUB_CONJ_0);
    }
    {
        SCHUR_UPDATE_LAST_2_TERM(j1, BS4_SCHUR_LAST2_CMSUB_CONJ_1);
    }
    {
        SCHUR_UPDATE_LAST_2_TERM(j2, BS4_SCHUR_LAST2_CMSUB_CONJ_2);
    }
    {
        SCHUR_UPDATE_LAST_2_TERM(j3, BS4_SCHUR_LAST2_CMSUB_CONJ_3);
    }

    *pA_i0_i0_re = extract(A_i0_i0_re);
    *pA_i1_i1_re = extract(A_i1_i1_re);
    *pA_i1_i0_re = extract(A_i1_i0_re);
    *pA_i1_i0_im = extract(A_i1_i0_im);
}

void schur_update_col_bs4(Para_capon_roi_part1_config *cfg, dvfloat COV_REAL_BANK *restrict L_batch_real_buf_ptr,
                          dvfloat COV_IMAG_BANK *restrict L_batch_imag_buf_ptr,
                          dvfloat COV_REAL_BANK *restrict L_real_element_ptr,
                          dvfloat COV_IMAG_BANK *restrict L_imag_element_ptr, int32_t s, int32_t i0, int32_t n)
{
    int32_t j0 = s, j1 = s + 1, j2 = s + 2, j3 = s + 3;

    LOAD_BS4_OFF_DIAG_ROW(i0);
    SCHUR_UPDATE_ELEMENT_DIAG(i0);

    int32_t row_start                        = i0 + 1;
    dvfloat COV_REAL_BANK *restrict pL_re_j0 = L_batch_real_buf_ptr + L_REAL_IDX(row_start, j0);
    dvfloat COV_IMAG_BANK *restrict pL_im_j0 = L_batch_imag_buf_ptr + L_IMAG_IDX(row_start, j0);
    dvfloat COV_REAL_BANK *restrict pL_re_j1 = L_batch_real_buf_ptr + L_REAL_IDX(row_start, j1);
    dvfloat COV_IMAG_BANK *restrict pL_im_j1 = L_batch_imag_buf_ptr + L_IMAG_IDX(row_start, j1);
    dvfloat COV_REAL_BANK *restrict pL_re_j2 = L_batch_real_buf_ptr + L_REAL_IDX(row_start, j2);
    dvfloat COV_IMAG_BANK *restrict pL_im_j2 = L_batch_imag_buf_ptr + L_IMAG_IDX(row_start, j2);
    dvfloat COV_REAL_BANK *restrict pL_re_j3 = L_batch_real_buf_ptr + L_REAL_IDX(row_start, j3);
    dvfloat COV_IMAG_BANK *restrict pL_im_j3 = L_batch_imag_buf_ptr + L_IMAG_IDX(row_start, j3);
    dvfloat COV_REAL_BANK *restrict pA_re_i0 = L_batch_real_buf_ptr + L_REAL_IDX(row_start, i0);
    dvfloat COV_IMAG_BANK *restrict pA_im_i0 = L_batch_imag_buf_ptr + L_IMAG_IDX(row_start, i0);

    for (int i = row_start; i < n; i++)
#if CUPVA_PVA_GEN_NUMBER < 3
        chess_loop_range(2, 11)
#else
        chess_loop_range(2, )
#endif
    chess_prepare_for_pipelining
    {
        dvfloatx L_i_j0_re = sign_extend(*pL_re_j0);
        dvfloatx L_i_j0_im = sign_extend(*pL_im_j0);
        dvfloatx L_i_j1_re = sign_extend(*pL_re_j1);
        dvfloatx L_i_j1_im = sign_extend(*pL_im_j1);
        dvfloatx L_i_j2_re = sign_extend(*pL_re_j2);
        dvfloatx L_i_j2_im = sign_extend(*pL_im_j2);
        dvfloatx L_i_j3_re = sign_extend(*pL_re_j3);
        dvfloatx L_i_j3_im = sign_extend(*pL_im_j3);
        dvfloatx A_i_i0_re = sign_extend(*pA_re_i0);
        dvfloatx A_i_i0_im = sign_extend(*pA_im_i0);

        {
            BS4_SCHUR_CMSUB_CONJ_0(L_i_j0_re, L_i_j0_im, BS4_L_RE(i0, j0), BS4_L_IM(i0, j0), A_i_i0_re, A_i_i0_im);
        }
        {
            BS4_SCHUR_CMSUB_CONJ_1(L_i_j1_re, L_i_j1_im, BS4_L_RE(i0, j1), BS4_L_IM(i0, j1), A_i_i0_re, A_i_i0_im);
        }
        {
            BS4_SCHUR_CMSUB_CONJ_2(L_i_j2_re, L_i_j2_im, BS4_L_RE(i0, j2), BS4_L_IM(i0, j2), A_i_i0_re, A_i_i0_im);
        }
        {
            BS4_SCHUR_CMSUB_CONJ_3(L_i_j3_re, L_i_j3_im, BS4_L_RE(i0, j3), BS4_L_IM(i0, j3), A_i_i0_re, A_i_i0_im);
        }
        *pA_re_i0 = extract(A_i_i0_re);
        *pA_im_i0 = extract(A_i_i0_im);

        pL_re_j0++;
        pL_im_j0++;
        pL_re_j1++;
        pL_im_j1++;
        pL_re_j2++;
        pL_im_j2++;
        pL_re_j3++;
        pL_im_j3++;
        pA_re_i0++;
        pA_im_i0++;
    }
}

/* Full BS4 Cholesky driver. */
void block_cholesky_opt_full_driver_bs4(Para_capon_roi_part1_config *cfg)
{
    int32_t n                                    = BUF_MAX_SV_DIM;
    dvfloat COV_REAL_BANK *restrict L_re_ptr     = (dvfloat COV_REAL_BANK *)cfg->L_real_buf_ptr;
    dvfloat COV_IMAG_BANK *restrict L_im_ptr     = (dvfloat COV_IMAG_BANK *)cfg->L_imag_buf_ptr;
    dvfloat LINV_REAL_BANK *restrict Linv_re_ptr = (dvfloat LINV_REAL_BANK *)cfg->Linv_real_buf_ptr;
    dvfloat LINV_IMAG_BANK *restrict Linv_im_ptr = (dvfloat LINV_IMAG_BANK *)cfg->Linv_imag_buf_ptr;
    dvfloat COV_REAL_BANK *restrict L_real_element_ptr;
    dvfloat COV_IMAG_BANK *restrict L_imag_element_ptr;
    dvfloat LINV_REAL_BANK *restrict Linv_real_element_ptr;
    dvfloat LINV_IMAG_BANK *restrict Linv_imag_element_ptr;

    chess_separator_scheduler();
    for (int32_t s = 0; s < n - 4; s += 4)
    {
        diag_block_cholesky_bs4(cfg, L_re_ptr, L_im_ptr, Linv_re_ptr, Linv_im_ptr, L_real_element_ptr,
                                L_imag_element_ptr, Linv_real_element_ptr, Linv_imag_element_ptr, s);
        diag_block_linv_bs4(cfg, L_re_ptr, L_im_ptr, Linv_re_ptr, Linv_im_ptr, L_real_element_ptr, L_imag_element_ptr,
                            Linv_real_element_ptr, Linv_imag_element_ptr, s);
        off_diag_bs4(cfg, L_re_ptr, L_im_ptr, Linv_re_ptr, Linv_im_ptr, L_real_element_ptr, L_imag_element_ptr,
                     Linv_real_element_ptr, Linv_imag_element_ptr, s, n);
        for (int32_t i = s + 4; i < n - 2; i++)
        {
            schur_update_col_bs4(cfg, L_re_ptr, L_im_ptr, L_real_element_ptr, L_imag_element_ptr, s, i, n);
        }
        schur_update_col_last_2(cfg, L_re_ptr, L_im_ptr, L_real_element_ptr, L_imag_element_ptr, s, n);
    }
    /* Factor the final diagonal block. */
    diag_block_cholesky_bs4(cfg, L_re_ptr, L_im_ptr, Linv_re_ptr, Linv_im_ptr, L_real_element_ptr, L_imag_element_ptr,
                            Linv_real_element_ptr, Linv_imag_element_ptr, n - 4);
    diag_block_linv_bs4(cfg, L_re_ptr, L_im_ptr, Linv_re_ptr, Linv_im_ptr, L_real_element_ptr, L_imag_element_ptr,
                        Linv_real_element_ptr, Linv_imag_element_ptr, n - 4);
}
