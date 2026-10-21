/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "canny_edge_detector_knls.h"

#include "canny_edge_detector_params.h"

// if kh = 3
// coef1 = [a0, a1, a2], coef2 = [a3, a4, a5]
// coef = [a0, a1, a2, 0, a0, a1, a2, 0, a0, a1, a2, 0, a0, a1, a2, 0, ...
//         a3, a4, a5, 0, a3, a4, a5, 0, a3, a4, a5, 0, a3, a4, a5, 0, ...]
void reformat_coef(int8_t *coef, const int8_t *coef1, const int8_t *coef2, int32_t kh)
{
    int32_t n = (kh + 3) / 4;
    int32_t d = kh % 4;

    for (int32_t i = 0; i < n; i++)
    {
        if (i < (n - 1))
        {
            for (int32_t j = 0; j < 8; j++)
            {
                coef[i * 64 + j * 4]      = coef1[i * 4];
                coef[i * 64 + j * 4 + 1]  = coef1[i * 4 + 1];
                coef[i * 64 + j * 4 + 2]  = coef1[i * 4 + 2];
                coef[i * 64 + j * 4 + 3]  = coef1[i * 4 + 3];
                coef[i * 64 + j * 4 + 32] = coef2[i * 4];
                coef[i * 64 + j * 4 + 33] = coef2[i * 4 + 1];
                coef[i * 64 + j * 4 + 34] = coef2[i * 4 + 2];
                coef[i * 64 + j * 4 + 35] = coef2[i * 4 + 3];
            }
        }
        else
        {
            if (d == 0)
            {
                d = 4;
            }
            for (int32_t j = 0; j < d; j++)
            {
                for (int32_t t = 0; t < 8; t++)
                {
                    coef[i * 64 + j + 4 * t]      = coef1[i * 4 + j];
                    coef[i * 64 + j + 4 * t + 32] = coef2[i * 4 + j];
                }
            }
            for (int32_t j = d; j < 4; j++)
            {
                for (int32_t t = 0; t < 16; t++)
                {
                    coef[i * 64 + j + 4 * t] = 0;
                }
            }
        }
    }
}

void filter4x2_hor_short_init(const uint8_t *data, const int8_t *coef, int16_t *restrict out_x, int16_t *restrict out_y,
                              int32_t kh, int32_t qbits_x, int32_t qbits_y, int32_t blkw, int32_t blkh,
                              int32_t lofst_data, int32_t lofst_out, uint32_t cb_size_bytes, struct Flt4x2Para *para)

{
    int32_t vecw = pva_elementsof(dvshortx); // 32 for 16-bit data, double vector width
    // requirement: kh % 4 == 0
    // requirement: blkw % (vecw * 2) == 0

    int32_t niter1 = kh / 4;
    int32_t niter2 = blkw / (vecw * 2);
    int32_t niter3 = blkh;
    int32_t niter  = niter1 * niter2 * niter3;

    agen a0    = init((dvuchar *)data);
    a0.n1      = niter1;
    a0.mod1    = 4 * sizeof(uint8_t);
    a0.n2      = niter2;
    a0.mod2    = ((1 - niter1) * 4 + vecw * 2) * sizeof(uint8_t);
    a0.n3      = niter3;
    a0.mod3    = ((1 - niter1) * 4 + (1 - niter2) * vecw * 2 + lofst_data) * sizeof(uint8_t);
    a0         = update_agen_cb_start(a0, (uintptr_t)(void *)data);
    a0         = update_agen_cb_size(a0, cb_size_bytes);
    para->c[0] = extract_agen_cfg(a0);

    agen a1    = init((dvuchar *)(data + 4));
    a1.n1      = niter1;
    a1.mod1    = 4 * sizeof(uint8_t);
    a1.n2      = niter2;
    a1.mod2    = ((1 - niter1) * 4 + vecw * 2) * sizeof(uint8_t);
    a1.n3      = niter3;
    a1.mod3    = ((1 - niter1) * 4 + (1 - niter2) * vecw * 2 + lofst_data) * sizeof(uint8_t);
    a1         = update_agen_cb_start(a1, (uintptr_t)(void *)data);
    a1         = update_agen_cb_size(a1, cb_size_bytes);
    para->c[1] = extract_agen_cfg(a1);

    agen a2    = init((dvchar *)coef);
    a2.n1      = niter1;
    a2.mod1    = vecw * 2 * sizeof(int8_t);
    a2.n2      = niter2 * niter3;
    a2.mod2    = (1 - niter1) * vecw * 2 * sizeof(int8_t);
    para->c[2] = extract_agen_cfg(a2);

    agen a3    = init(out_x);
    a3.n1      = 2;
    a3.n2      = niter1;
    a3.n3      = niter2;
    a3.n4      = niter3;
    a3.mod1    = vecw * sizeof(int16_t);
    a3.mod2    = -vecw * sizeof(int16_t);
    a3.mod3    = vecw * sizeof(int16_t);
    a3.mod4    = (-vecw + (1 - niter2) * vecw * 2 + lofst_out) * sizeof(int16_t);
    a3.round   = qbits_x;
    para->c[3] = extract_agen_cfg(a3);

    agen a4    = init(out_y);
    a4.n1      = 2;
    a4.n2      = niter1;
    a4.n3      = niter2;
    a4.n4      = niter3;
    a4.mod1    = vecw * sizeof(int16_t);
    a4.mod2    = -vecw * sizeof(int16_t);
    a4.mod3    = vecw * sizeof(int16_t);
    a4.mod4    = (-vecw + (1 - niter2) * vecw * 2 + lofst_out) * sizeof(int16_t);
    a4.round   = qbits_y;
    para->c[4] = extract_agen_cfg(a4);

    para->niter1 = niter1;
    para->niter  = niter;
}

int32_t filter4x2_hor_short_exec(struct Flt4x2Para *para)
{
    int32_t niter  = para->niter;
    int32_t niter1 = para->niter1;
    agen a0        = init_agen_from_cfg(para->c[0]);
    agen a1        = init_agen_from_cfg(para->c[1]);
    agen a2        = init_agen_from_cfg(para->c[2]);
    agen a3        = init_agen_from_cfg(para->c[3]);
    agen a4        = init_agen_from_cfg(para->c[4]);

    // niter_ceil should be >= 15 to satisfy chess_loop_range(15, ) requirement
    int32_t niter_ceil  = (niter + 4) / 5 * 5;
    int32_t count_madd  = 0;
    int32_t count_store = 0;
    int32_t pred_madd   = 0;
    int32_t pred_store  = 0;
    dvshortx vacc0      = chess_dont_care(dvshortx) & 0;
    dvshortx vacc1      = chess_dont_care(dvshortx) & 0;
    dvshortx vacc2      = chess_dont_care(dvshortx) & 0;
    dvshortx vacc3      = chess_dont_care(dvshortx) & 0;

    for (int32_t i = 0; i < niter_ceil; i++) chess_prepare_for_pipelining
    chess_unroll_loop(5)
    chess_loop_range(15, )
    {
        dvcharx vdata0 = dvuchar_load(a0);
        dvcharx vdata1 = dvuchar_load(a1);
        dvcharx vcoef  = dvchar_load(a2);

        count_store = mod_inc_pred_z(count_store, niter1 - 1, pred_store);

        vfilt4x2_bbh(vdata0.lo, vdata1.lo, vcoef, vacc0, vacc1, pred_madd, vacc0, vacc1);
        vfilt4x2_bbh(vdata0.hi, vdata1.hi, vcoef, vacc2, vacc3, pred_madd, vacc2, vacc3);

        vstore_i(vacc0, a3, pred_store);
        vstore_i(vacc1, a4, pred_store);
        vstore_i(vacc2, a3, pred_store);
        vstore_i(vacc3, a4, pred_store);

        count_madd = mod_inc_pred_nz(count_madd, niter1 - 1, pred_madd);
    }
    return 0;
}

void filter4_ver_short_init(const int16_t *data, const int8_t *coef, int16_t *restrict out, int32_t kh, int32_t qbits,
                            int32_t blkw, int32_t blkh, int32_t lofst_data, int32_t lofst_out, struct Flt4Para *para)
{
    int32_t vecw = pva_elementsof(dvshortx); // 32 for 16-bit data, double vector width
    // requirement: kh % 4 == 0
    // requirement: blkh % vecw == 0

    int32_t niter1 = kh / 4;
    int32_t niter2 = blkh / vecw;
    int32_t niter3 = blkw;
    int32_t niter  = niter1 * niter2 * niter3;

    agen a0      = init((dvshort *)data);
    a0.n1        = niter1;
    a0.n2        = niter2;
    a0.n3        = niter3;
    a0.mod1      = 4 * lofst_data * sizeof(int16_t);
    a0.mod2      = ((1 - niter1) * 4 + vecw) * lofst_data * sizeof(int16_t);
    a0.mod3      = (((1 - niter1) * 4 + (1 - niter2) * vecw) * lofst_data + 1) * sizeof(int16_t);
    a0.lane_ofst = lofst_data / 32;
    para->c[0]   = extract_agen_cfg(a0);

    agen a1      = init((dvshort *)(data + 4 * lofst_data));
    a1.n1        = niter1;
    a1.n2        = niter2;
    a1.n3        = niter3;
    a1.mod1      = 4 * lofst_data * sizeof(int16_t);
    a1.mod2      = ((1 - niter1) * 4 + vecw) * lofst_data * sizeof(int16_t);
    a1.mod3      = (((1 - niter1) * 4 + (1 - niter2) * vecw) * lofst_data + 1) * sizeof(int16_t);
    a1.lane_ofst = lofst_data / 32;
    para->c[1]   = extract_agen_cfg(a1);

    agen a2    = init((vchar *)(coef));
    a2.n1      = niter1;
    a2.n2      = niter2 * niter3;
    a2.mod1    = 2 * vecw * sizeof(int8_t);
    a2.mod2    = (1 - niter1) * 2 * vecw * sizeof(int8_t);
    para->c[2] = extract_agen_cfg(a2);

    agen a3    = init((dvshort *restrict)out);
    a3.n1      = niter1;
    a3.n2      = niter2;
    a3.n3      = niter3;
    a3.mod1    = 0;
    a3.mod2    = vecw * sizeof(int16_t);
    a3.mod3    = ((1 - niter2) * vecw + lofst_out) * sizeof(int16_t);
    a3.round   = qbits;
    para->c[3] = extract_agen_cfg(a3);

    para->niter1 = niter1;
    para->niter  = niter;
}

int32_t filter4_ver_short_exec(struct Flt4Para *para)
{
    int32_t niter  = para->niter;
    int32_t niter1 = para->niter1;
    agen a0        = init_agen_from_cfg(para->c[0]);
    agen a1        = init_agen_from_cfg(para->c[1]);
    agen a2        = init_agen_from_cfg(para->c[2]);
    agen a3        = init_agen_from_cfg(para->c[3]);

    // niter_ceil should be >= 15 to satisfy chess_loop_range(15, ) requirement
    int32_t niter_ceil  = (niter + 4) / 5 * 5;
    int32_t count_madd  = 0;
    int32_t count_store = 0;
    int32_t pred_madd   = 0;
    int32_t pred_store  = 0;
    dvintx vacc0        = chess_dont_care(dvintx) & 0;
    dvintx vacc1        = chess_dont_care(dvintx) & 0;

    for (int32_t i = 0; i < niter_ceil; i++) chess_prepare_for_pipelining
    chess_unroll_loop(5)
    chess_loop_range(15, )
    {
        dvshortx vdata0 = dvshort_load_transp(a0);
        dvshortx vdata1 = dvshort_load_transp(a1);
        dvshortx vcoef  = vchar_dvshortx_load(a2);

        count_store = mod_inc_pred_z(count_store, niter1 - 1, pred_store);

        vacc0 = vfilt4_hhw(vdata0.lo, vdata1.lo, vcoef.lo, vacc0, pred_madd);
        vacc1 = vfilt4_hhw(vdata0.hi, vdata1.hi, vcoef.hi, vacc1, pred_madd);

        vstore_i2(vacc0, vacc1, a3, pred_store);
        count_madd = mod_inc_pred_nz(count_madd, niter1 - 1, pred_madd);
    }
    return 0;
}

void calc_intensity_L2_init(const int16_t *gx, const int16_t *gy, int32_t *restrict out, int32_t blkw, int32_t blkh,
                            int32_t lofst_data, int32_t lofst_out, struct EdgePara *para)
{
    int32_t vecw = pva_elementsof(dvintx); // 16 for 32-bit data, double vector width
    // requirement: blkw % vecw == 0

    int32_t niter1 = blkh / vecw;
    int32_t niter2 = blkw;
    int32_t niter  = niter1 * niter2;

    agen a0      = init((vshort *)gx);
    a0.n1        = niter1;
    a0.n2        = niter2;
    a0.mod1      = vecw * lofst_data * sizeof(int16_t);
    a0.mod2      = ((1 - niter1) * vecw * lofst_data + 1) * sizeof(int16_t);
    a0.lane_ofst = lofst_data / 32;
    para->c[0]   = extract_agen_cfg(a0);

    agen a1      = init((vshort *)gy);
    a1.n1        = niter1;
    a1.n2        = niter2;
    a1.mod1      = vecw * lofst_data * sizeof(int16_t);
    a1.mod2      = ((1 - niter1) * vecw * lofst_data + 1) * sizeof(int16_t);
    a1.lane_ofst = lofst_data / 32;
    para->c[1]   = extract_agen_cfg(a1);

    agen a2    = init((dvint *)out);
    a2.n1      = niter1;
    a2.n2      = niter2;
    a2.mod1    = vecw * sizeof(int32_t);
    a2.mod2    = ((1 - niter1) * vecw + lofst_out) * sizeof(int32_t);
    para->c[2] = extract_agen_cfg(a2);

    para->niter = niter;
}

int32_t calc_intensity_L2_exec(struct EdgePara *para)
{
    int32_t niter = para->niter;

    agen_A a0 = init_agen_A_from_cfg(para->c[0]);
    agen_C a1 = init_agen_C_from_cfg(para->c[1]);
    agen_B a2 = init_agen_B_from_cfg(para->c[2]);

    // niter_ceil should be >= 12 to satisfy chess_loop_range(12, ) requirement
    int32_t niter_ceil = (niter + 3) & (-4);

    for (int32_t i = 0; i < niter_ceil; i++) chess_prepare_for_pipelining
    chess_unroll_loop(4)
    chess_loop_range(12, )
    {
        vshortx vgradx = vshort_load_transp(a0);
        vshortx vgrady = vshort_load_transp(a1);

        dvintx vsum = vsumsq(vgradx, vgrady);

        dvfloatx vsum_fp    = dvint_dvfp(vsum);
        dvfloatx vintens_fp = dvfsqrt(vsum_fp);
        dvintx vintens      = dvfp_dvint_trunc(vintens_fp);

        vstore_i(vintens, a2);
    }
    return 0;
}

void edge_direction_init(const int16_t *gx, const int16_t *gy, int8_t *restrict out, int32_t blkw, int32_t blkh,
                         int32_t lofst_data, int32_t lofst_out, struct EdgePara *para)
{
    int32_t vecw = pva_elementsof(dvshortx); // 32 for 16-bit data, double vector width
    // requirement: blkh % vecw == 0

    int32_t niter1 = blkh / vecw;
    int32_t niter2 = blkw;
    int32_t niter  = niter1 * niter2;

    agen a0      = init((dvshort *)gx);
    a0.n1        = niter1;
    a0.n2        = niter2;
    a0.mod1      = vecw * lofst_data * sizeof(int16_t);
    a0.mod2      = ((1 - niter1) * vecw * lofst_data + 1) * sizeof(int16_t);
    a0.lane_ofst = lofst_data / 32;
    para->c[0]   = extract_agen_cfg(a0);

    agen a1      = init((dvshort *)gy);
    a1.n1        = niter1;
    a1.n2        = niter2;
    a1.mod1      = vecw * lofst_data * sizeof(int16_t);
    a1.mod2      = ((1 - niter1) * vecw * lofst_data + 1) * sizeof(int16_t);
    a1.lane_ofst = lofst_data / 32;
    para->c[1]   = extract_agen_cfg(a1);

    agen a2     = init((vchar *)out);
    a2.n1       = niter1;
    a2.n2       = niter2;
    a2.mod1     = vecw * sizeof(int8_t);
    a2.mod2     = ((1 - niter1) * vecw + lofst_out) * sizeof(int8_t);
    para->c[2]  = extract_agen_cfg(a2);
    para->niter = niter;
}

int32_t edge_direction_exec(struct EdgePara *para)
{
    int32_t niter = para->niter;

    agen_A a0 = init_agen_A_from_cfg(para->c[0]);
    agen_C a1 = init_agen_C_from_cfg(para->c[1]);
    agen_B a2 = init_agen_B_from_cfg(para->c[2]);

    // niter_ceil should be >= 6 to satisfy chess_loop_range(6, ) requirement
    int32_t niter_ceil = (niter + 1) & (-2);

    dvshortx v_dir_diag0;
    v_dir_diag0.hi = v_dir_diag0.lo = replicateh(DIR_DIAG0);

    for (int32_t i = 0; i < niter_ceil; i++) chess_prepare_for_pipelining
    chess_unroll_loop(2)
    chess_loop_range(6, )
    {
        dvshortx gx = dvshort_load_transp(a0);
        dvshortx gy = dvshort_load_transp(a1);

        dvshortx gx_sign, gy_sign;
        dvshortx gx_abs, gy_abs;

        dvsignmag(gx, gx_sign, gx_abs);
        dvsignmag(gy, gy_sign, gy_abs);

        dvshortx tg22_x = dvmulh(gx_abs, TG22, 0);
        dvshortx tg67_x = dvmulh(gx_abs, TG67, 0);

        dvshortx gy_scaled = dvmulh(gy_abs, (1 << CANNY_SHIFT), 0);

        dvshortx pred_hor = gy_scaled < tg22_x;
        dvshortx pred_ver = gy_scaled > tg67_x;

        dvshortx sign = gx_sign ^ gy_sign;

        dvshortx dir = dvmux(sign, DIR_DIAG1, v_dir_diag0);
        dir          = dvmux(pred_hor, DIR_HORI, dir);
        dir          = dvmux(pred_ver, DIR_VERT, dir);

        vstore_hb(dir, a2);
    }
    return 0;
}

void nms_double_thresholding_init(const int8_t *dir, const int32_t *intensity, uint8_t *restrict edge_type,
                                  int32_t blkw, int32_t blkh, int32_t lofst_dir, int32_t lofst_intensity,
                                  int32_t lofst_out, uint32_t cb_size_bytes, struct EdgePara *para)
{
    int32_t vecw = pva_elementsof(dvintx); // 16 for 32-bit data, double vector width
    // requirement: blkw % vecw == 0

    int32_t niter1 = blkw / vecw;
    int32_t niter2 = blkh;
    int32_t niter  = niter1 * niter2;

    agen a0    = init((vchar *)dir);
    a0.n1      = niter1;
    a0.mod1    = vecw * sizeof(int8_t);
    a0.n2      = niter2;
    a0.mod2    = ((1 - niter1) * vecw + lofst_dir) * sizeof(int8_t);
    para->c[0] = extract_agen_cfg(a0);

    agen a1    = init((dvint *)intensity);
    a1.n1      = 3;
    a1.n2      = 3;
    a1.n3      = niter1;
    a1.n4      = niter2;
    a1.mod1    = sizeof(int32_t);
    a1.mod2    = (-2 + lofst_intensity) * sizeof(int32_t);
    a1.mod3    = (-2 - 2 * lofst_intensity + vecw) * sizeof(int32_t);
    a1.mod4    = (-2 - 2 * lofst_intensity + (1 - niter1) * vecw + lofst_intensity) * sizeof(int32_t);
    a1         = update_agen_cb_start(a1, (uintptr_t)(void *)intensity);
    a1         = update_agen_cb_size(a1, cb_size_bytes);
    para->c[1] = extract_agen_cfg(a1);

    agen a2    = init((vuchar *)edge_type);
    a2.n1      = niter1;
    a2.n2      = niter2;
    a2.mod1    = vecw * sizeof(uint8_t);
    a2.mod2    = ((1 - niter1) * vecw + lofst_out) * sizeof(uint8_t);
    para->c[2] = extract_agen_cfg(a2);

    para->niter = niter;
}

int32_t nms_double_thresholding_exec(struct EdgePara *para, int32_t threshold_strong, int32_t threshold_weak)
{
    int32_t niter = para->niter;
    // niter_ceil should be >= 6 to satisfy chess_loop_range(6, ) requirement
    int32_t niter_ceil = (niter + 1) & (-2);

    vshortx vzero = replicateh(0);
    dvintx v_edge_no;
    v_edge_no.hi = v_edge_no.lo = replicatew(EDGE_NO);

    agen a0 = init_agen_from_cfg(para->c[0]);
    agen a1 = init_agen_from_cfg(para->c[1]);
    agen a2 = init_agen_from_cfg(para->c[2]);

    for (int32_t i = 0; i < niter_ceil; i++) chess_prepare_for_pipelining
    chess_unroll_loop(2)
    chess_loop_range(6, )
    {
        dvintx dir = vchar_dvintx_load(a0);

        dvintx v0 = dvint_load(a1);
        dvintx v1 = dvint_load(a1);
        dvintx v2 = dvint_load(a1);
        dvintx v3 = dvint_load(a1);
        dvintx v4 = dvint_load(a1);
        dvintx v5 = dvint_load(a1);
        dvintx v6 = dvint_load(a1);
        dvintx v7 = dvint_load(a1);
        dvintx v8 = dvint_load(a1);

        // default horizontal
        // vertical
        dvintx pred_ver = dir == DIR_VERT;
        dvintx prev     = dvmux(pred_ver, v1, v3);
        dvintx next     = dvmux(pred_ver, v7, v5);

        // 45 degree
        dvintx pred_diag0 = dir == DIR_DIAG0;
        prev              = dvmux(pred_diag0, v0, prev);
        next              = dvmux(pred_diag0, v8, next);

        // -45 degree
        dvintx pred_diag1 = dir == DIR_DIAG1;
        prev              = dvmux(pred_diag1, v6, prev);
        next              = dvmux(pred_diag1, v2, next);

        dvintx pred = v4 > prev;
        pred        = dvCmpLE_andL(next, v4, pred);

        dvintx edge_type = dvmux(v4 > threshold_weak, EDGE_WEAK, v_edge_no);
        edge_type        = dvmux(v4 > threshold_strong, EDGE_STRONG, edge_type);
        edge_type        = dvmux(pred, edge_type, v_edge_no);

        vstore_wb_plround(vzero, edge_type, a2);
    }
    return 0;
}
