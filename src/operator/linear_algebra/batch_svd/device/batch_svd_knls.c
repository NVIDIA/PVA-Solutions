/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "batch_svd_knls.h"

#include <float.h>

/**
 * m x n (height x width) matrix, m >= n
 * 16 matrices are interleaved
 * A0, B0, C0, ...
 * A1, B1, C1, ...
 * A2, B2, C2, ...
 * A3, B3, C3, ...
 * A4, B4, C4, ...
 * ...
 * ...
 * ...
 */

void set_eye_matrix(float *v, const int32_t n, const int32_t pitch)
{
    int32_t niter = n * n;
    agen a        = init(v);
    a.n1          = niter;
    a.mod1        = pitch * sizeof(float);

    dvfloatx v0;
    v0.lo = replicatef(0.f);
    v0.hi = replicatef(0.f);

    chess_separator_scheduler();

    for (int32_t i = 0; i < niter; i++) vstore(v0, a);

    dvfloatx v1;
    v1.lo = replicatef(1.f);
    v1.hi = replicatef(1.f);

    chess_separator_scheduler();

    for (int32_t i = 0; i < n; i++)
    {
        *(dvfloat *)(v + (i * n + i) * pitch) = extract(v1);
    }
}

/**
 * Calculate alpha, beta, gamma =
 * sigma Uki^2, sigma Ukj^2, sigma Uki*Ukj
 * General kernel for matrices with a height > 4
 */
void calculate_alpha_beta_gamma(float *alpha, float *beta, float *gamma, const float *u, const int32_t m,
                                const int32_t n, const int32_t pitch, const int32_t i, const int32_t j)
{
    agen_C a_ki = init_C((float *)(u + i * pitch));
    a_ki.n1     = m;
    a_ki.mod1   = n * pitch * sizeof(float);

    agen_C a_kj = init_C((float *)(u + j * pitch));
    a_kj.n1     = m;
    a_kj.mod1   = n * pitch * sizeof(float);

    dvfloatx v_alpha  = as_dvfloatx(chess_dont_care(dvintx) & 0);
    dvfloatx v_beta   = as_dvfloatx(chess_dont_care(dvintx) & 0);
    dvfloatx v_gamma  = as_dvfloatx(chess_dont_care(dvintx) & 0);
    int32_t niter     = m / 2 * 2;
    int32_t niter_rem = m - niter;

    chess_separator_scheduler();

    for (int32_t t = 0; t < niter; t++) chess_loop_range(4, )
    chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        dvfloatx u_ki = dvfloat_load(a_ki);
        dvfloatx u_kj = dvfloat_load(a_kj);

        v_alpha = dvmaddf(u_ki, u_ki, v_alpha);
        v_beta  = dvmaddf(u_kj, u_kj, v_beta);
        v_gamma = dvmaddf(u_ki, u_kj, v_gamma);
    }

    for (int32_t t = 0; t < niter_rem; t++) chess_keep_sw_loop
    {
        dvfloatx u_ki = dvfloat_load(a_ki);
        dvfloatx u_kj = dvfloat_load(a_kj);

        v_alpha = dvmaddf(u_ki, u_ki, v_alpha);
        v_beta  = dvmaddf(u_kj, u_kj, v_beta);
        v_gamma = dvmaddf(u_ki, u_kj, v_gamma);
    }

    *((dvfloat *)alpha) = extract(v_alpha);
    *((dvfloat *)beta)  = extract(v_beta);
    *((dvfloat *)gamma) = extract(v_gamma);
}

/**
 * Calculate alpha, beta, gamma =
 * sigma Uki^2, sigma Ukj^2, sigma Uki*Ukj
 * Specialized kernel for matrices with a height = 2
 */
void calculate_alpha_beta_gamma_2(float *alpha, float *beta, float *gamma, const float *u, const int32_t m,
                                  const int32_t n, const int32_t pitch, const int32_t i, const int32_t j)
{
    agen_C a_ki = init_C((float *)(u + i * pitch));
    a_ki.n1     = m;
    a_ki.mod1   = n * pitch * sizeof(float);

    agen_C a_kj = init_C((float *)(u + j * pitch));
    a_kj.n1     = m;
    a_kj.mod1   = n * pitch * sizeof(float);

    dvfloatx v_alpha = as_dvfloatx(chess_dont_care(dvintx) & 0);
    dvfloatx v_beta  = as_dvfloatx(chess_dont_care(dvintx) & 0);
    dvfloatx v_gamma = as_dvfloatx(chess_dont_care(dvintx) & 0);
    chess_separator_scheduler();

    for (int32_t t = 0; t < 2; t++) chess_unroll_loop(*)
    chess_prepare_for_pipelining
    {
        dvfloatx u_ki = dvfloat_load(a_ki);
        dvfloatx u_kj = dvfloat_load(a_kj);

        v_alpha = dvmaddf(u_ki, u_ki, v_alpha);
        v_beta  = dvmaddf(u_kj, u_kj, v_beta);
        v_gamma = dvmaddf(u_ki, u_kj, v_gamma);
    }

    *((dvfloat *)alpha) = extract(v_alpha);
    *((dvfloat *)beta)  = extract(v_beta);
    *((dvfloat *)gamma) = extract(v_gamma);
}

/**
 * Calculate alpha, beta, gamma =
 * sigma Uki^2, sigma Ukj^2, sigma Uki*Ukj
 * Specialized kernel for matrices with a height = 3
 */
void calculate_alpha_beta_gamma_3(float *alpha, float *beta, float *gamma, const float *u, const int32_t m,
                                  const int32_t n, const int32_t pitch, const int32_t i, const int32_t j)
{
    agen_C a_ki = init_C((float *)(u + i * pitch));
    a_ki.n1     = m;
    a_ki.mod1   = n * pitch * sizeof(float);

    agen_C a_kj = init_C((float *)(u + j * pitch));
    a_kj.n1     = m;
    a_kj.mod1   = n * pitch * sizeof(float);

    dvfloatx v_alpha = as_dvfloatx(chess_dont_care(dvintx) & 0);
    dvfloatx v_beta  = as_dvfloatx(chess_dont_care(dvintx) & 0);
    dvfloatx v_gamma = as_dvfloatx(chess_dont_care(dvintx) & 0);
    chess_separator_scheduler();

    for (int32_t t = 0; t < 3; t++) chess_unroll_loop(*)
    chess_prepare_for_pipelining
    {
        dvfloatx u_ki = dvfloat_load(a_ki);
        dvfloatx u_kj = dvfloat_load(a_kj);

        v_alpha = dvmaddf(u_ki, u_ki, v_alpha);
        v_beta  = dvmaddf(u_kj, u_kj, v_beta);
        v_gamma = dvmaddf(u_ki, u_kj, v_gamma);
    }

    *((dvfloat *)alpha) = extract(v_alpha);
    *((dvfloat *)beta)  = extract(v_beta);
    *((dvfloat *)gamma) = extract(v_gamma);
}

/**
 * Calculate alpha, beta, gamma =
 * sigma Uki^2, sigma Ukj^2, sigma Uki*Ukj
 * Specialized kernel for matrices with a height = 4
 */
void calculate_alpha_beta_gamma_4(float *alpha, float *beta, float *gamma, const float *u, const int32_t m,
                                  const int32_t n, const int32_t pitch, const int32_t i, const int32_t j)
{
    agen_C a_ki = init_C((float *)(u + i * pitch));
    a_ki.n1     = m;
    a_ki.mod1   = n * pitch * sizeof(float);

    agen_C a_kj = init_C((float *)(u + j * pitch));
    a_kj.n1     = m;
    a_kj.mod1   = n * pitch * sizeof(float);

    dvfloatx v_alpha = as_dvfloatx(chess_dont_care(dvintx) & 0);
    dvfloatx v_beta  = as_dvfloatx(chess_dont_care(dvintx) & 0);
    dvfloatx v_gamma = as_dvfloatx(chess_dont_care(dvintx) & 0);
    chess_separator_scheduler();

    for (int32_t t = 0; t < 4; t++) chess_unroll_loop(*)
    chess_prepare_for_pipelining
    {
        dvfloatx u_ki = dvfloat_load(a_ki);
        dvfloatx u_kj = dvfloat_load(a_kj);

        v_alpha = dvmaddf(u_ki, u_ki, v_alpha);
        v_beta  = dvmaddf(u_kj, u_kj, v_beta);
        v_gamma = dvmaddf(u_ki, u_kj, v_gamma);
    }

    *((dvfloat *)alpha) = extract(v_alpha);
    *((dvfloat *)beta)  = extract(v_beta);
    *((dvfloat *)gamma) = extract(v_gamma);
}

void check_converge(float *e, int16_t *p, const float *alpha, const float *beta, const float *gamma, const float tol)
{
    const dvfloatx v_alpha = sign_extend(*(const dvfloat *)alpha);
    const dvfloatx v_beta  = sign_extend(*(const dvfloat *)beta);
    dvfloatx v_gamma       = sign_extend(*(const dvfloat *)gamma);
    dvfloatx v_e           = sign_extend(*(dvfloat *)e);

    dvfloatx v_denom = dvfsqrt(v_alpha * v_beta);

    dvfloatx v_abs_alpha = as_dvfloatx(as_dvintx(v_alpha) & 0x7fffffffu);
    dvfloatx v_abs_beta  = as_dvfloatx(as_dvintx(v_beta) & 0x7fffffffu);
    dvfloatx v_abs_gamma = as_dvfloatx(as_dvintx(v_gamma) & 0x7fffffffu);

    dvintx v_pred = v_denom * tol >= v_abs_gamma; // when both are 0

    // If abs(gamma) is very close to 0, the two columns are nearly orthogonal
    // and if abs(alpha) or abs(beta) is very close to 0, one of the columns has become negligible
    // Comparing results are either 1 or 0, bitwise & and | are functionally equivalent to logical && and ||
    v_pred = v_pred | (v_abs_gamma < FLT_EPSILON & (v_abs_alpha < FLT_EPSILON | v_abs_beta < FLT_EPSILON));

    dvfloatx v_tmp_e = dvmux(v_pred, 0.f, v_abs_gamma * dvfrcp(v_denom)); // to avoid nan

    v_e = as_dvfloatx(dvmax(as_dvintx(v_e), as_dvintx(v_tmp_e)));

    vshortx v_npred = vdemote(!v_pred);

    *(vshort *)p = extract(v_npred);

    *(dvfloat *)e = extract(v_e);
}

void calculate_sin_cos(float *s, float *c, const float *alpha, const float *beta, const float *gamma)
{
    const dvfloatx v_alpha = sign_extend(*(const dvfloat *)alpha);
    const dvfloatx v_beta  = sign_extend(*(const dvfloat *)beta);
    const dvfloatx v_gamma = sign_extend(*(const dvfloat *)gamma);

    dvfloatx v0 = v_beta - v_alpha;

    dvfloatx v1 = dvfrcp(v_gamma * 2); // rcp(+-0) = +-inf

    dvfloatx v_cot_2r = v0 * v1; // (+-inf) * (+-0) = nan

    dvfloatx v_one;
    v_one.lo = replicatef(1.f);
    v_one.hi = replicatef(1.f);

    dvintx v_cot_2r_sign = as_dvintx(v_cot_2r) & 0x80000000;

    dvfloatx v_cot_2r_abs = as_dvfloatx(as_dvintx(v_cot_2r) & 0x7fffffffu);

    dvfloatx v_cot_r_abs = v_cot_2r_abs + dvfsqrt(dvmaddf(v_cot_2r, v_cot_2r, v_one)); // inf or nan

    dvfloatx v_tan_r_abs = dvfrcp(v_cot_r_abs); // 0 or nan

    dvfloatx v_cos_r_abs = dvfrsq(dvmaddf(v_tan_r_abs, v_tan_r_abs, v_one));

    // Let |angle| <= PI/4
    // If cos(r) is nan, set cos(r) to 1
    dvintx v_p           = as_dvintx(v_cos_r_abs) > as_int32(1.f);
    dvfloatx v_cos_r     = dvmux(v_p, 1.f, v_cos_r_abs);
    dvfloatx v_sin_r_abs = v_cos_r_abs * v_tan_r_abs;
    dvfloatx v_sin_r     = dvmux(v_p, 0.f, as_dvfloatx(v_cot_2r_sign | as_dvintx(v_sin_r_abs)));

    *(dvfloat *)c = extract(v_cos_r);
    *(dvfloat *)s = extract(v_sin_r);
}

/**
 * Update matrix U and V
 * General kernel for matrices with a height >= 4
 */
void update_matrix(float *u, const float *s, const float *c, const int16_t *p, const int32_t m, const int32_t n,
                   const int32_t pitch, const int32_t i, const int32_t j)
{
    const dvfloatx v_s = sign_extend(*(const dvfloat *)s);
    const dvfloatx v_c = sign_extend(*(const dvfloat *)c);
    const vshortx pred = sign_extend(*(const vshort *)p);

    agen a_ki_r = init(u + i * pitch);
    a_ki_r.n1   = m;
    a_ki_r.mod1 = n * pitch * sizeof(float);

    agen a_kj_r = init(u + j * pitch);
    a_kj_r.n1   = m;
    a_kj_r.mod1 = n * pitch * sizeof(float);

    agen a_ki_w = init(u + i * pitch);
    a_ki_w.n1   = m;
    a_ki_w.mod1 = n * pitch * sizeof(float);

    agen a_kj_w = init(u + j * pitch);
    a_kj_w.n1   = m;
    a_kj_w.mod1 = n * pitch * sizeof(float);

    int32_t niter = (m + 1) / 2 * 2;

    chess_separator_scheduler();

    for (int32_t t = 0; t < niter; t++) chess_loop_range(4, )
    chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        dvfloatx u_ki = dvfloat_load(a_ki_r);
        dvfloatx u_kj = dvfloat_load(a_kj_r);

        dvfloatx v0 = v_c * u_ki;
        dvfloatx v1 = v_s * u_ki;

        // below matches C reference check better than vmadd and vmsub
        // dvfloatx v2 = dvmsubf(v_s, u_kj, v0);
        // dvfloatx v3 = dvmaddf(v_c, u_kj, v1);
        dvfloatx v2 = v_s * u_kj;
        dvfloatx v3 = v_c * u_kj;
        dvfloatx v4 = v0 - v2;
        dvfloatx v5 = v1 + v3;

        vstore(v4, a_ki_w, pred);
        vstore(v5, a_kj_w, pred);
    }
}

/**
 * Update matrix U and V
 * Specialized kernel for matrices with a height = 2 or 3
 */
void update_matrix_2_3(float *u, const float *s, const float *c, const int16_t *p, const int32_t m, const int32_t n,
                       const int32_t pitch, const int32_t i, const int32_t j)
{
    const dvfloatx v_s = sign_extend(*(const dvfloat *)s);
    const dvfloatx v_c = sign_extend(*(const dvfloat *)c);
    const vshortx pred = sign_extend(*(const vshort *)p);

    agen a_ki_r = init(u + i * pitch);
    a_ki_r.n1   = m;
    a_ki_r.mod1 = n * pitch * sizeof(float);

    agen a_kj_r = init(u + j * pitch);
    a_kj_r.n1   = m;
    a_kj_r.mod1 = n * pitch * sizeof(float);

    agen a_ki_w = init(u + i * pitch);
    a_ki_w.n1   = m;
    a_ki_w.mod1 = n * pitch * sizeof(float);

    agen a_kj_w = init(u + j * pitch);
    a_kj_w.n1   = m;
    a_kj_w.mod1 = n * pitch * sizeof(float);

    chess_separator_scheduler();

    for (int32_t t = 0; t < m; t++) chess_loop_range(2, )
    chess_prepare_for_pipelining
    {
        dvfloatx u_ki = dvfloat_load(a_ki_r);
        dvfloatx u_kj = dvfloat_load(a_kj_r);

        dvfloatx v0 = v_c * u_ki;
        dvfloatx v1 = v_s * u_ki;

        // below matches C reference check better than vmadd and vmsub
        // dvfloatx v2 = dvmsubf(v_s, u_kj, v0);
        // dvfloatx v3 = dvmaddf(v_c, u_kj, v1);
        dvfloatx v2 = v_s * u_kj;
        dvfloatx v3 = v_c * u_kj;
        dvfloatx v4 = v0 - v2;
        dvfloatx v5 = v1 + v3;

        vstore(v4, a_ki_w, pred);
        vstore(v5, a_kj_w, pred);
    }
}

void calculate_singular_values(float *s, const float *u, const int32_t m, const int32_t n, const int32_t pitch)
{
    agen a_u = init((float *)u);
    a_u.n1   = m;
    a_u.n2   = n;
    a_u.mod1 = n * pitch * sizeof(float);
    a_u.mod2 = (1 - (m - 1) * n) * pitch * sizeof(float);

    agen a_s = init(s);
    a_s.n1   = m;
    a_s.n2   = n;
    a_s.mod1 = 0;
    a_s.mod2 = pitch * sizeof(float);

    dvfloatx sum        = as_dvfloatx(chess_dont_care(dvintx) & 0);
    int32_t pred        = 0;
    int32_t count       = 0;
    int32_t pred_store  = 0;
    int32_t count_store = 0;

    int32_t niter = (m * n + 2) / 3 * 3;
    chess_separator_scheduler();

    for (int32_t i = 0; i < niter; i++) chess_loop_range(6, )
    chess_unroll_loop(3)
    chess_prepare_for_pipelining
    {
        dvfloatx in = dvfloat_load(a_u);

        sum = dvmaddf(in, in, sum, pred);

        count       = mod_inc_pred_nz(count, m - 1, pred);
        count_store = mod_inc_pred_z(count_store, m - 1, pred_store);

        vstore(sum, a_s, pred_store);
    }

    agen a_s_r = init(s);
    a_s_r.n1   = n;
    a_s_r.mod1 = pitch * sizeof(float);

    agen a_s_w = init(s);
    a_s_w.n1   = n;
    a_s_w.mod1 = pitch * sizeof(float);
    chess_separator_scheduler();

    for (int32_t i = 0; i < n; i++) chess_loop_range(2, )
    chess_prepare_for_pipelining
    {
        dvfloatx in = dvfloat_load(a_s_r);

        dvfloatx out = dvfsqrt(in);

        vstore(out, a_s_w);
    }
}

void normalize_matrix(float *u, const float *s, const int32_t m, const int32_t n, const int32_t pitch)
{
    agen_C a_u_r = init_C(u);
    a_u_r.n1     = m;
    a_u_r.n2     = n;
    a_u_r.mod1   = n * pitch * sizeof(float);
    a_u_r.mod2   = (1 - (m - 1) * n) * pitch * sizeof(float);

    agen_C a_u_w = init_C(u);
    a_u_w.n1     = m;
    a_u_w.n2     = n;
    a_u_w.mod1   = n * pitch * sizeof(float);
    a_u_w.mod2   = (1 - (m - 1) * n) * pitch * sizeof(float);

    agen_A a_s = init_A((float *)s);
    a_s.n1     = m;
    a_s.n2     = n;
    a_s.mod1   = 0;
    a_s.mod2   = pitch * sizeof(float);

    int32_t niter = (m * n + 2) / 3 * 3;
    chess_separator_scheduler();

    for (int32_t i = 0; i < niter; i++) chess_loop_range(6, )
    chess_unroll_loop(3)
    chess_prepare_for_pipelining
    {
        dvfloatx in_u = dvfloat_load(a_u_r);
        dvfloatx in_s = dvfloat_load(a_s);

        dvfloatx out_u = in_u * dvfrcp(in_s);

        vstore(out_u, a_u_w);
    }
}

void sort_singular_values(int32_t *t, float *sorted_s, float *sorted_u, float *sorted_v, const float *s, const float *u,
                          const float *v, const int32_t m, const int32_t n, const int32_t pitch)
{
    int32_t vecw = pva_elementsof(dvintx);
    agen a_i     = init((float *)s);
    a_i.n1       = n;
    a_i.mod1     = pitch * sizeof(float);

    agen a_j = init((float *)s);
    a_j.n1   = n;
    a_j.n2   = n;
    a_j.mod1 = pitch * sizeof(float);
    a_j.mod2 = (1 - n) * pitch * sizeof(float);

    dvintx vzero;
    vzero.lo = replicatew(0);
    vzero.hi = replicatew(0);

    chess_separator_scheduler();

    for (int32_t i = 0; i < n; i++)
    {
        dvintx t_i   = vzero;
        dvfloatx s_i = dvfloat_load(a_i);

        for (int32_t j = 0; j < n; j++) chess_loop_range(2, )
        chess_prepare_for_pipelining
        {
            dvfloatx s_j = dvfloat_load(a_j);

            dvintx p0 = s_i < s_j;

            t_i += p0;

            dvintx p1 = (s_i == s_j) && (j < i);

            t_i += p1;
        }

        *(dvint *)(t + i * vecw) = extract(t_i);
    }

    agen a_s = init((float *)s);
    a_s.n1   = n;
    a_s.mod1 = pitch * sizeof(float);

    agen a_u = init((float *)u);
    a_u.n1   = m;
    a_u.n2   = n;
    a_u.mod1 = n * pitch * sizeof(float);
    a_u.mod2 = (1 - (m - 1) * n) * pitch * sizeof(float);

    agen a_v = init((float *)v);
    a_v.n1   = n;
    a_v.n2   = n;
    a_v.mod1 = n * pitch * sizeof(float);
    a_v.mod2 = (1 - (n - 1) * n) * pitch * sizeof(float);

    chess_separator_scheduler();

    for (int32_t i = 0; i < n; i++)
    {
        dvintx col_idx = sign_extend(*(dvint *)(t + i * vecw));
        // sort U
        for (int32_t j = 0; j < m; j++) chess_loop_range(2, )
        chess_prepare_for_pipelining
        {
            dvfloatx u_i = dvfloat_load(a_u);

            dvintx idx = col_idx + j * n;

            vast_16f(sorted_u, idx, u_i, 0xffffffffu);
        }
        // sort V
        for (int32_t j = 0; j < n; j++) chess_loop_range(2, )
        chess_prepare_for_pipelining
        {
            dvfloatx v_i = dvfloat_load(a_v);

            dvintx idx = col_idx + j * n;

            vast_16f(sorted_v, idx, v_i, 0xffffffffu);
        }
        // sort S
        {
            dvfloatx s_i = dvfloat_load(a_s);
            vast_16f(sorted_s, col_idx, s_i, 0xffffffffu);
        }
    }
}