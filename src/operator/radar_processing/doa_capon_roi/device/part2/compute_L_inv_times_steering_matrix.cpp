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

#include "compute_L_inv_times_steering_matrix.h"

#include "capon_roi_part2_config.h"

#include <cupva_device.h>

/*
 * Computes L_inv * steering_matrix in-place using three-step fixed-point integer arithmetic:
 *   Step 1: lower-right 8x8 triangle (rows H..sv_dim-1, cols H..row), clipped to [-2^30, 2^30-1]
 *   Step 2: lower-left 8x8 rectangle (rows H..sv_dim-1, cols 0..H-1)
 *   Step 3: upper-left 8x8 triangle  (rows 0..H-1, cols 0..row)
 * where H = sv_dim / 2 = 8.
 *
 * Step 1 is clipped because step 2 accumulates its results in S48; without the clip, step 2 could
 * overflow. Only a step-1 output of exactly 2^30 (= 2^28 * 2^30 / 2^16 / 2^16 * 2^3 * 2) is clipped,
 * to 2^30 - 1. The bound uses L_inv real/imag in [-2^28, 2^28], steering real/imag in
 * [-2^30, 2^30], W.T16 and AGEN rounding by 16, 8 complex products, and 2 multiplies per real or
 * imaginary part.
 *
 * The kernel is specialized for sv_dim = 16. The 8x8 block decomposition extends to any multiple
 * of 8, but other sizes would need new clipping values, fixed-point formats and AGEN settings.
 */
void init_compute_L_inv_times_steering_matrix(Para_capon_roi_part2_config &para)
{
    // ---- Permute pattern for dvint_load_perm ----
    // Produces { 0, 2, 0, 2, ..., 1, 0, 1, 0, ... } to map [re, im, -im] triplets
    // into the dvintx layout expected by vdotp2x2_w_t16 for complex multiply.
    {
        const vintx t  = vcollate_idx(replicatew(0)); // 0, 1, 2, 3, ...
        const vintx tl = (t & 1) << 1;                // 0, 2, 0, 2, ...
        const vintx th = (t & 1) == 0;                // 1, 0, 1, 0, ...
        vshortx pat    = chess_dont_care(vshortx);
        vdemote(tl, th, pat);
        para.permute_pattern = extract(pat);
    }

    // ---- L_inv agen configs (three blocks) ----
    // L_inv data layout: [roi_num][line_pitch] of int32, triplets [re, im, -im].
    // Traversal: sequential triplets within a block, wrap for each sv group, advance per ROI.
    {
        AgenWrapper w;
        w.size = sizeof(int32_t);
        w.s1   = 3; // consecutive triplets
        w.n2   = CAPON_PART2_SV_GROUPS;
        w.s2   = 0; // wrap back for each sv group
        w.n3   = SV_ROI_NUM;
        w.s3   = BUF_LINE_PITCH_LINV_FIXED;

        // Upper-left triangle: H_tri elements starting at offset 0
        w.n1   = CAPON_PART2_HALF_TRI_LEN;
        agen a = init((int32_t *)0); // patched before first use
        INIT_AGEN3(a, w);
        para.cfg_L_inv_upper_left_triangular = extract_agen_cfg(a);

        // Lower-right triangle: H_tri elements starting after upper-left block
        a.a                                   = (int32_t *)0; // patched before first use
        para.cfg_L_inv_lower_right_triangular = extract_agen_cfg(a);

        // Lower-left rectangle: H*H elements starting after both triangles
        w.n1 = CAPON_PART2_HALF_SV_DIM * CAPON_PART2_HALF_SV_DIM;
        a.a  = (int32_t *)0; // patched before first use
        INIT_AGEN3(a, w);
        para.cfg_L_inv_rectangular = extract_agen_cfg(a);
    }

    // ---- sv_mat agen configs for triangular functions ----
    // Layout per channel: [roi_num][sv_dim][sv_num / 2][2] of int32.
    // Reads H rows per iteration, advances by 1 dvfloat per sv group, then per ROI.
    {
        // the 16x16 L_inv decomposed into 8x8 matrices
        // thus the 16 x N SV matrix is decomposed into the top part and the bottom part.
        AgenWrapper w;
        w.size = sizeof(dvfloat);
        w.n1   = CAPON_PART2_HALF_SV_DIM;
        w.s1   = CAPON_PART2_SV_GROUPS;
        w.n2   = CAPON_PART2_SV_GROUPS;
        w.s2   = 1;
        w.n3   = SV_ROI_NUM;
        w.s3   = BUF_MAX_SV_DIM * CAPON_PART2_SV_GROUPS;

        // Upper-left: rows 0..H-1
        agen a  = init((int32_t *)0); // patched before first use
        a.round = 16;
        INIT_AGEN3(a, w);
        para.cfg_sv_mat_0_upper_left_triangular = extract_agen_cfg(a);
        a.a                                     = (int32_t *)0; // patched before first use
        para.cfg_sv_mat_1_upper_left_triangular = extract_agen_cfg(a);

        // Lower-right: rows H..sv_dim-1
        a.a          = (int32_t *)0; // patched before first use
        a.sat_opt    = 2;            // treat 32-bit comparison values as signed
        a.sat_lim_lo = -(1 << 30);
        a.sat_val_lo = -(1 << 30);
        a.sat_lim_hi = (1 << 30) - 1;
        a.sat_val_hi = (1 << 30) - 1;
        INIT_AGEN3(a, w);
        para.cfg_sv_mat_0_lower_right_triangular = extract_agen_cfg(a);
        a.a                                      = (int32_t *)0; // patched before first use
        para.cfg_sv_mat_1_lower_right_triangular = extract_agen_cfg(a);
    }

    // ---- sv_mat agen configs for rectangular function ----
    // Reads upper-half rows (0..H-1) with predicated reload every H/2 iterations.
    {
        AgenWrapper w;
        w.size  = sizeof(dvfloat);
        w.n1    = CAPON_PART2_HALF_SV_DIM;
        w.s1    = CAPON_PART2_SV_GROUPS;
        w.n2    = CAPON_PART2_HALF_SV_DIM / 2; // cnt cycle: reload every H/2 iterations
        w.s2    = 0;                           // wrap back to same rows
        w.n3    = CAPON_PART2_SV_GROUPS;
        w.s3    = 1;
        w.n4    = SV_ROI_NUM;
        w.s4    = BUF_MAX_SV_DIM * CAPON_PART2_SV_GROUPS;
        agen a  = init((int32_t *)0); // patched before first use
        a.round = 16;
        INIT_AGEN4(a, w);
        para.cfg_sv_mat_0_rectangular = extract_agen_cfg(a);
        a.a                           = (int32_t *)0; // patched before first use
        para.cfg_sv_mat_1_rectangular = extract_agen_cfg(a);
    }
}

static inline void my_cmadd(const dvintx v, const dvintx w, dvintx &acc, const int pred)
{
    acc = vdotp2x2_w_t16(v.lo, v.hi, w, acc, pred);
}

static inline void my_cmadd(const dvintx v0, const dvintx v1, const dvintx w, dvintx &acc0, dvintx &acc1,
                            const int pred)
{
    my_cmadd(v0, w, acc0, pred);
    my_cmadd(v1, w, acc1, pred);
}

static inline void my_cmadd(const dvintx v0, const dvintx v1, agen_A &a_w, const vshortx pat, dvintx &acc0,
                            dvintx &acc1, const int pred)
{
    const dvintx chess_storage(DW) w = dvint_load_perm(a_w, pat);
    my_cmadd(v0, v1, w, acc0, acc1, pred);
}

/*
 * Row r of the 8x8 lower-triangular product, out_r = sum_{c <= r} L_inv_r_c * v_c, expanded from
 * the (row, col)-named L_inv registers and the v_c pairs in scope. TRI_TERMS_0_TO_c(r) emits the
 * terms for columns 0..c: the first overwrites the accumulator pair (pred 0), the rest accumulate.
 */
// clang-format off
#define TRI_TERM(r, c) my_cmadd(v##c##_0, v##c##_1, L_inv_##r##_##c, acc0, acc1, -1)
#define TRI_TERMS_0_TO_0(r) my_cmadd(v0_0, v0_1, L_inv_##r##_0, acc0, acc1, 0)
#define TRI_TERMS_0_TO_1(r) TRI_TERMS_0_TO_0(r); TRI_TERM(r, 1)
#define TRI_TERMS_0_TO_2(r) TRI_TERMS_0_TO_1(r); TRI_TERM(r, 2)
#define TRI_TERMS_0_TO_3(r) TRI_TERMS_0_TO_2(r); TRI_TERM(r, 3)
#define TRI_TERMS_0_TO_4(r) TRI_TERMS_0_TO_3(r); TRI_TERM(r, 4)
#define TRI_TERMS_0_TO_5(r) TRI_TERMS_0_TO_4(r); TRI_TERM(r, 5)
#define TRI_TERMS_0_TO_6(r) TRI_TERMS_0_TO_5(r); TRI_TERM(r, 6)
#define TRI_TERMS_0_TO_7(r) TRI_TERMS_0_TO_6(r); TRI_TERM(r, 7)
// clang-format on

/* Loads the two halves of steering vector r, used by rows r..7. */
#define TRI_LOAD_V(r)                                                    \
    const dvintx chess_storage(DV) v##r##_0 = dvint_load_di(a_sv_mat_0); \
    const dvintx chess_storage(DV) v##r##_1 = dvint_load_di(a_sv_mat_1)

/*
 * Row r into a fresh accumulator pair, stored in place through the output AGENs.
 * lane_ofst = 0 is used intentionally to avoid unnecessary memory stalls.
 */
#define TRI_ROW(r)                                                \
    {                                                             \
        dvintx chess_storage(DAC) acc0 = chess_dont_care(dvintx); \
        dvintx chess_storage(DAC) acc1 = chess_dont_care(dvintx); \
        TRI_TERMS_0_TO_##r(r);                                    \
        vstore_transp_i(acc0, a_out0);                            \
        vstore_transp_i(acc1, a_out1);                            \
    }

void compute_L_inv_times_steering_matrix_triangular_mat_mul(const Para_capon_roi_part2_config &para,
                                                            const bool is_upper_left_lower_triangular)
{
    agen_B a_sv_mat_0 = init_agen_B_from_cfg(is_upper_left_lower_triangular ? para.cfg_sv_mat_0_upper_left_triangular
                                                                            : para.cfg_sv_mat_0_lower_right_triangular);
    agen_C a_sv_mat_1 = init_agen_C_from_cfg(is_upper_left_lower_triangular ? para.cfg_sv_mat_1_upper_left_triangular
                                                                            : para.cfg_sv_mat_1_lower_right_triangular);
    agen_A a_L_inv    = init_agen_A_from_cfg(is_upper_left_lower_triangular ? para.cfg_L_inv_upper_left_triangular
                                                                            : para.cfg_L_inv_lower_right_triangular);
    agen_B a_out0     = a_sv_mat_0; // in-place
    agen_C a_out1     = a_sv_mat_1; // in-place

    const vshortx chess_storage(WB15) pat = sign_extend(para.permute_pattern);

    const uint32_t niter = CAPON_PART2_MUL_TRI_NITER;

    for (uint32_t i = 0; i < niter; ++i) chess_loop_range(2, )
    chess_prepare_for_pipelining
    {
        // Packed lower triangle of the 8x8 L_inv block, row by row, rotating through DW0..DW5.
        const dvintx chess_storage(DW0) L_inv_0_0 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW1) L_inv_1_0 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW2) L_inv_1_1 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW3) L_inv_2_0 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW4) L_inv_2_1 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW5) L_inv_2_2 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW0) L_inv_3_0 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW1) L_inv_3_1 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW2) L_inv_3_2 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW3) L_inv_3_3 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW4) L_inv_4_0 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW5) L_inv_4_1 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW0) L_inv_4_2 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW1) L_inv_4_3 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW2) L_inv_4_4 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW3) L_inv_5_0 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW4) L_inv_5_1 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW5) L_inv_5_2 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW0) L_inv_5_3 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW1) L_inv_5_4 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW2) L_inv_5_5 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW3) L_inv_6_0 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW4) L_inv_6_1 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW5) L_inv_6_2 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW0) L_inv_6_3 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW1) L_inv_6_4 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW2) L_inv_6_5 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW3) L_inv_6_6 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW4) L_inv_7_0 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW5) L_inv_7_1 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW0) L_inv_7_2 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW1) L_inv_7_3 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW2) L_inv_7_4 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW3) L_inv_7_5 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW4) L_inv_7_6 = dvint_load_perm(a_L_inv, pat);
        const dvintx chess_storage(DW5) L_inv_7_7 = dvint_load_perm(a_L_inv, pat);

        // out_r = sum_{c <= r} L_inv_r_c * v_c for the eight rows; v_r is loaded right before its first use.
        TRI_LOAD_V(0);
        TRI_ROW(0);
        TRI_LOAD_V(1);
        TRI_ROW(1);
        TRI_LOAD_V(2);
        TRI_ROW(2);
        TRI_LOAD_V(3);
        TRI_ROW(3);
        TRI_LOAD_V(4);
        TRI_ROW(4);
        TRI_LOAD_V(5);
        TRI_ROW(5);
        TRI_LOAD_V(6);
        TRI_ROW(6);
        TRI_LOAD_V(7);
        TRI_ROW(7);
    }
}

static inline void rect_mac8_and_store(dvintx &acc0, dvintx &acc1, const dvintx v0_0, const dvintx v0_1,
                                       const dvintx v1_0, const dvintx v1_1, const dvintx v2_0, const dvintx v2_1,
                                       const dvintx v3_0, const dvintx v3_1, const dvintx v4_0, const dvintx v4_1,
                                       const dvintx v5_0, const dvintx v5_1, const dvintx v6_0, const dvintx v6_1,
                                       const dvintx v7_0, const dvintx v7_1, agen_A &a_L_inv, const vshortx pat,
                                       agen_B &a_out0, agen_C &a_out1)
{
    my_cmadd(v0_0, v0_1, a_L_inv, pat, acc0, acc1, -1);
    my_cmadd(v1_0, v1_1, a_L_inv, pat, acc0, acc1, -1);
    my_cmadd(v2_0, v2_1, a_L_inv, pat, acc0, acc1, -1);
    my_cmadd(v3_0, v3_1, a_L_inv, pat, acc0, acc1, -1);
    my_cmadd(v4_0, v4_1, a_L_inv, pat, acc0, acc1, -1);
    my_cmadd(v5_0, v5_1, a_L_inv, pat, acc0, acc1, -1);
    my_cmadd(v6_0, v6_1, a_L_inv, pat, acc0, acc1, -1);
    my_cmadd(v7_0, v7_1, a_L_inv, pat, acc0, acc1, -1);
    // lane_ofst = 0 is used intentionally to avoid unnecessary memory stalls
    vstore_transp_i(acc0, a_out0);
    vstore_transp_i(acc1, a_out1);
}

void compute_L_inv_times_steering_matrix_rectangular_mat_mul(const Para_capon_roi_part2_config &para)
{
    agen_B a_sv_mat_0  = init_agen_B_from_cfg(para.cfg_sv_mat_0_rectangular);
    agen_C a_sv_mat_1  = init_agen_C_from_cfg(para.cfg_sv_mat_1_rectangular);
    agen_A a_L_inv     = init_agen_A_from_cfg(para.cfg_L_inv_rectangular);
    agen_B a_out0      = init_agen_B_from_cfg(para.cfg_sv_mat_0_lower_right_triangular);
    agen_C a_out1      = init_agen_C_from_cfg(para.cfg_sv_mat_1_lower_right_triangular);
    a_out0.sat_opt     = 0;      // No need to do clipping.
    a_out1.sat_opt     = 0;      // No need to do clipping.
    agen_B a_pre_acc_0 = a_out0; // in-place
    agen_C a_pre_acc_1 = a_out1; // in-place

    dvintx chess_storage(DV0) v0_0 = chess_dont_care(dvintx);
    dvintx chess_storage(DV1) v0_1 = chess_dont_care(dvintx);
    dvintx chess_storage(DV2) v1_0 = chess_dont_care(dvintx);
    dvintx chess_storage(DV3) v1_1 = chess_dont_care(dvintx);
    dvintx chess_storage(DV4) v2_0 = chess_dont_care(dvintx);
    dvintx chess_storage(DV5) v2_1 = chess_dont_care(dvintx);
    dvintx chess_storage(DV6) v3_0 = chess_dont_care(dvintx);
    dvintx chess_storage(DV7) v3_1 = chess_dont_care(dvintx);

    const vshortx chess_storage(V) pat = sign_extend(para.permute_pattern);

    const uint32_t cnt_max = CAPON_PART2_MUL_CNTMAX;
    const uint32_t niter   = CAPON_PART2_MUL_RECT_NITER;
    uint32_t cnt           = cnt_max;

    for (uint32_t i = 0; i < niter; ++i) chess_loop_range(3, )
    chess_prepare_for_pipelining
    {
        int32_t pred = chess_dont_care(int32_t);
        cnt          = mod_inc_pred_z(cnt, cnt_max, pred);

        // Predicated load for saving power.
        dvint_load_di(v0_0, a_sv_mat_0, pred);
        dvint_load_di(v0_1, a_sv_mat_1, pred);
        dvint_load_di(v1_0, a_sv_mat_0, pred);
        dvint_load_di(v1_1, a_sv_mat_1, pred);
        dvint_load_di(v2_0, a_sv_mat_0, pred);
        dvint_load_di(v2_1, a_sv_mat_1, pred);
        dvint_load_di(v3_0, a_sv_mat_0, pred);
        dvint_load_di(v3_1, a_sv_mat_1, pred);

        // Repeated load for releasing register pressure.
        const dvintx chess_storage(DV) v4_0 = dvint_load_di(a_sv_mat_0);
        const dvintx chess_storage(DV) v4_1 = dvint_load_di(a_sv_mat_1);
        const dvintx chess_storage(DV) v5_0 = dvint_load_di(a_sv_mat_0);
        const dvintx chess_storage(DV) v5_1 = dvint_load_di(a_sv_mat_1);
        const dvintx chess_storage(DV) v6_0 = dvint_load_di(a_sv_mat_0);
        const dvintx chess_storage(DV) v6_1 = dvint_load_di(a_sv_mat_1);
        const dvintx chess_storage(DV) v7_0 = dvint_load_di(a_sv_mat_0);
        const dvintx chess_storage(DV) v7_1 = dvint_load_di(a_sv_mat_1);

        {
            dvintx chess_storage(DAC0) acc0 =
                dvmsubwhw(dvint_load_di(a_pre_acc_0), -(1 << 16), chess_dont_care(dvintx), VPU_ROUND_0, 0);
            dvintx chess_storage(DAC1) acc1 =
                dvmsubwhw(dvint_load_di(a_pre_acc_1), -(1 << 16), chess_dont_care(dvintx), VPU_ROUND_0, 0);
            rect_mac8_and_store(acc0, acc1, v0_0, v0_1, v1_0, v1_1, v2_0, v2_1, v3_0, v3_1, v4_0, v4_1, v5_0, v5_1,
                                v6_0, v6_1, v7_0, v7_1, a_L_inv, pat, a_out0, a_out1);
        }
        {
            dvintx chess_storage(DAC2) acc0 =
                dvmsubwhw(dvint_load_di(a_pre_acc_0), -(1 << 16), chess_dont_care(dvintx), VPU_ROUND_0, 0);
            dvintx chess_storage(DAC3) acc1 =
                dvmsubwhw(dvint_load_di(a_pre_acc_1), -(1 << 16), chess_dont_care(dvintx), VPU_ROUND_0, 0);
            rect_mac8_and_store(acc0, acc1, v0_0, v0_1, v1_0, v1_1, v2_0, v2_1, v3_0, v3_1, v4_0, v4_1, v5_0, v5_1,
                                v6_0, v6_1, v7_0, v7_1, a_L_inv, pat, a_out0, a_out1);
        }
    }
}
