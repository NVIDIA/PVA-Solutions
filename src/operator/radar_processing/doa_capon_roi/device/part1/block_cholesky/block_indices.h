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

#ifndef BLOCK_INDICES_H
#define BLOCK_INDICES_H

#include "capon_roi_part1_config_types.h" /* Para_capon_roi_part1_config */

/* L real and L_inv include the diagonal; L imaginary excludes it. */

#if defined(CUPVA_PVA_GEN_NUMBER) && CUPVA_PVA_GEN_NUMBER >= 3
EXTERN_VMEM(D, int32_t, capon_roi_tri_col_bias, BUF_MAX_SV_DIM)
#else
EXTERN_VMEM(A, int32_t, capon_roi_tri_col_bias, BUF_MAX_SV_DIM)
#endif

/* Load/store helpers use caller-provided cfg, base, and scratch pointers. */

#define L_REAL_IDX(i, j) (capon_roi_tri_col_bias[(j)] + (i))
#define L_IMAG_IDX(i, j) (capon_roi_tri_col_bias[(j)] + (i) - (j)-1)
#define LINV_IDX(i, j) L_REAL_IDX(i, j)

/* L loads (from mat buffer, direct index) */
#define LOAD_L_RE(i, j, dst)                                      \
    L_real_element_ptr = L_batch_real_buf_ptr + L_REAL_IDX(i, j); \
    dvfloatx dst       = sign_extend(*L_real_element_ptr)

#define LOAD_L_IM(i, j, dst)                                      \
    L_imag_element_ptr = L_batch_imag_buf_ptr + L_IMAG_IDX(i, j); \
    dvfloatx dst       = sign_extend(*L_imag_element_ptr)

#define STORE_L_RE(i, j, val)                                      \
    L_real_element_ptr  = L_batch_real_buf_ptr + L_REAL_IDX(i, j); \
    *L_real_element_ptr = extract(val)

#define STORE_L_IM(i, j, val)                                      \
    L_imag_element_ptr  = L_batch_imag_buf_ptr + L_IMAG_IDX(i, j); \
    *L_imag_element_ptr = extract(val)

#define STORE_L_RE_AFTER_LOAD(val) *L_real_element_ptr = extract(val)
#define STORE_L_IM_AFTER_LOAD(val) *L_imag_element_ptr = extract(val)

#define LOAD_LINV_RE(i, j, dst)                                       \
    Linv_real_element_ptr = Linv_batch_real_buf_ptr + LINV_IDX(i, j); \
    dvfloatx dst          = sign_extend(*Linv_real_element_ptr)

#define LOAD_LINV_IM(i, j, dst)                                       \
    Linv_imag_element_ptr = Linv_batch_imag_buf_ptr + LINV_IDX(i, j); \
    dvfloatx dst          = sign_extend(*Linv_imag_element_ptr)

#define STORE_LINV_RE(i, j, val)                                       \
    Linv_real_element_ptr  = Linv_batch_real_buf_ptr + LINV_IDX(i, j); \
    *Linv_real_element_ptr = extract(val)

#define STORE_LINV_IM(i, j, val)                                       \
    Linv_imag_element_ptr  = Linv_batch_imag_buf_ptr + LINV_IDX(i, j); \
    *Linv_imag_element_ptr = extract(val)

#define STORE_LINV_RE_AFTER_LOAD(val) *Linv_real_element_ptr = extract(val)
#define STORE_LINV_IM_AFTER_LOAD(val) *Linv_imag_element_ptr = extract(val)

/* Complex arithmetic helpers; INTERM variants expose independent products. */

/* dst = a * b_re */
#define CMUL_REAL(a_re, a_im, b_re, dst_re, dst_im) \
    dvfloatx dst_re = dvmulf(a_re, b_re);           \
    dvfloatx dst_im = dvmulf(a_im, b_re)

/* a = a * b_re */
#define CMUL_REAL_INPLACE(a_re, a_im, b_re) \
    a_re = dvmulf(a_re, b_re);              \
    a_im = dvmulf(a_im, b_re)

/* acc += a * b_re */
#define CMAD_REAL(a_re, a_im, b_re, acc_re, acc_im) \
    acc_re = dvmaddf(a_re, b_re, acc_re);           \
    acc_im = dvmaddf(a_im, b_re, acc_im)

/* dst = a * conj(b) */
#define CMUL_CONJ(a_re, a_im, b_re, b_im, dst_re, dst_im) \
    dvfloatx dst_re = dvmulf(a_re, b_re);                 \
    dst_re          = dvmaddf(a_im, b_im, dst_re);        \
    dvfloatx dst_im = dvmulf(a_im, b_re);                 \
    dst_im          = dvmsubf(a_re, b_im, dst_im)

/* dst = a * conj(b), with all products exposed before either combine. */
#define CMUL_CONJ_INTERM(a_re, a_im, b_re, b_im, dst_re, dst_im) \
    dvfloatx _t_re_1 = dvmulf(a_re, b_re);                       \
    dvfloatx _t_re_2 = dvmulf(a_im, b_im);                       \
    dvfloatx _t_im_1 = dvmulf(a_im, b_re);                       \
    dvfloatx _t_im_2 = dvmulf(a_re, b_im);                       \
    dvfloatx dst_re  = _t_re_1 + _t_re_2;                        \
    dvfloatx dst_im  = _t_im_1 - _t_im_2;

/* acc += a * b */
#define CMAD(a_re, a_im, b_re, b_im, acc_re, acc_im) \
    acc_re = dvmaddf(a_re, b_re, acc_re);            \
    acc_re = dvmsubf(a_im, b_im, acc_re);            \
    acc_im = dvmaddf(a_re, b_im, acc_im);            \
    acc_im = dvmaddf(a_im, b_re, acc_im)

/* acc += a * b_re, intermediate (breaks dep chain). Use inside { }. */
#define CMAD_REAL_INTERM(a_re, a_im, b_re, acc_re, acc_im) \
    dvfloatx _t_re = dvmulf(a_re, b_re);                   \
    dvfloatx _t_im = dvmulf(a_im, b_re);                   \
    acc_re         = acc_re + _t_re;                       \
    acc_im         = acc_im + _t_im

/* acc += a * b with four independent products. Use inside { }. */
#define CMAD_INTERM(a_re, a_im, b_re, b_im, acc_re, acc_im) \
    dvfloatx _t_re_1 = dvmulf(a_re, b_re);                  \
    dvfloatx _t_re_2 = dvmulf(a_im, b_im);                  \
    acc_re += _t_re_1 - _t_re_2;                            \
    dvfloatx _t_im_1 = dvmulf(a_re, b_im);                  \
    dvfloatx _t_im_2 = dvmulf(a_im, b_re);                  \
    acc_im += _t_im_1 + _t_im_2;

/* acc -= a * b, intermediate (breaks dep chain). Use inside { }. */
#define CMSUB_INTERM(a_re, a_im, b_re, b_im, acc_re, acc_im) \
    dvfloatx _t_re = dvmulf(a_re, b_re);                     \
    _t_re          = dvmsubf(a_im, b_im, _t_re);             \
    acc_re         = acc_re - _t_re;                         \
    dvfloatx _t_im = dvmulf(a_re, b_im);                     \
    _t_im          = dvmaddf(a_im, b_re, _t_im);             \
    acc_im         = acc_im - _t_im

/* acc += a * conj(b) */
#define CMAD_CONJ(a_re, a_im, b_re, b_im, acc_re, acc_im) \
    acc_re = dvmaddf(a_re, b_re, acc_re);                 \
    acc_re = dvmaddf(a_im, b_im, acc_re);                 \
    acc_im = dvmaddf(a_im, b_re, acc_im);                 \
    acc_im = dvmsubf(a_re, b_im, acc_im)

/* acc -= a * b */
#define CMSUB(a_re, a_im, b_re, b_im, acc_re, acc_im) \
    acc_re = dvmsubf(a_re, b_re, acc_re);             \
    acc_re = dvmaddf(a_im, b_im, acc_re);             \
    acc_im = dvmsubf(a_re, b_im, acc_im);             \
    acc_im = dvmsubf(a_im, b_re, acc_im)

/* acc -= a * conj(b) */
#define CMSUB_CONJ(a_re, a_im, b_re, b_im, acc_re, acc_im) \
    acc_re = dvmsubf(a_re, b_re, acc_re);                  \
    acc_re = dvmsubf(a_im, b_im, acc_re);                  \
    acc_im = dvmsubf(a_im, b_re, acc_im);                  \
    acc_im = dvmaddf(a_re, b_im, acc_im)

/* acc -= a * conj(b) */
#define CMSUB_CONJ_INTERM(a_re, a_im, b_re, b_im, acc_re, acc_im) \
    dvfloatx _t_re_1 = dvmulf(a_re, b_re);                        \
    dvfloatx _t_re_2 = dvmulf(a_im, b_im);                        \
    acc_re -= _t_re_1 + _t_re_2;                                  \
    dvfloatx _t_im_1 = dvmulf(a_im, b_re);                        \
    dvfloatx _t_im_2 = dvmulf(a_re, b_im);                        \
    acc_im -= _t_im_1 - _t_im_2;

/* acc_re -= |a|^2 */
#define CMSUB_MAGSQ(a_re, a_im, acc_re)   \
    acc_re = dvmsubf(a_re, a_re, acc_re); \
    acc_re = dvmsubf(a_im, a_im, acc_re);

/* acc_re -= |a|^2 */
#define CMSUB_MAGSQ_INTERM(a_re, a_im, acc_re) \
    dvfloatx _t_1 = dvmulf(a_re, a_re);        \
    dvfloatx _t_2 = dvmulf(a_im, a_im);        \
    acc_re -= _t_1 + _t_2;

/* acc += |a|^2 */
#define CMAD_MAGSQ(a_re, a_im, acc_re)    \
    acc_re = dvmaddf(a_re, a_re, acc_re); \
    acc_re = dvmaddf(a_im, a_im, acc_re);

/* acc += |a|^2 */
#define CMAD_MAGSQ_INTERM(a_re, a_im, acc_re) \
    dvfloatx _t_1 = dvmulf(a_re, a_re);       \
    dvfloatx _t_2 = dvmulf(a_im, a_im);       \
    acc_re += _t_1 + _t_2;

/* acc += a * conj(b), intermediate (breaks dep chain). Use inside { }. */
#define CMAD_CONJ_INTERM(a_re, a_im, b_re, b_im, acc_re, acc_im) \
    dvfloatx _t_re_1 = dvmulf(a_re, b_re);                       \
    dvfloatx _t_re_2 = dvmulf(a_im, b_im);                       \
    acc_re += _t_re_1 + _t_re_2;                                 \
    dvfloatx _t_im_1 = dvmulf(a_im, b_re);                       \
    dvfloatx _t_im_2 = dvmulf(a_re, b_im);                       \
    acc_im += _t_im_1 - _t_im_2;

#endif /* BLOCK_INDICES_H */
