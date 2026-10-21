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

#include "median_filter_knl_sepa_2x2_5x5.h"

void sort_row_4_exec(const MedianFilterSepa2x2Config &cfg)
{
    agen_B a_in = init_agen_B_from_cfg(cfg.in_agen_cfg);

    // tmp agens in different superbanks

    agen_A a_tmpA = init_agen_A_from_cfg(cfg.tmp_st_agen_cfg_A);
    agen_B a_tmpB = init_agen_B_from_cfg(cfg.tmp_st_agen_cfg_B);
    agen_C a_tmpC = init_agen_C_from_cfg(cfg.tmp_st_agen_cfg_C);

    int niter = max(cfg.first_pass_niter, 6);
    niter     = (niter + 1) / 2 * 2;

    for (int i = 0; i < niter; i++) chess_loop_range(6, )
    chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        // use from the 2nd element
        VECX_TYPE x0 = DVLOAD_DI(a_in).hi;

        // hw only support even address for dv
        DVECX_TYPE dv1 = DVLOAD_DI(a_in);
        VECX_TYPE x1   = dv1.lo;
        VECX_TYPE x2   = dv1.hi;
        DVECX_TYPE dv2 = DVLOAD_DI(a_in);
        VECX_TYPE x3   = dv2.lo;

        SORT_4;

        // 0, 3 use C
        // 1 uses B
        // 2 uses A
        vstore(x0, a_tmpC);
        vstore(x1, a_tmpB);
        vstore(x2, a_tmpA);
        vstore(x3, a_tmpC);
    }
}

// step 1: sort the mutual block, one x-group pair per iteration.
// dvsort2/dvmin/dvmax expand to a pair of single-vector instructions per
// double vector; the win over the single-vector version is halved
// load/store/loop overhead and a larger body for the scheduler.
void sort_mutual_block_16_dv_exec(const MedianFilterSepa2x2Config &cfg)
{
    agen_A a_tmpA = init_agen_A_from_cfg(cfg.ld_mutual_block_agen_cfg_A);
    agen_B a_tmpB = init_agen_B_from_cfg(cfg.ld_mutual_block_agen_cfg_B);
    agen_C a_tmpC = init_agen_C_from_cfg(cfg.ld_mutual_block_agen_cfg_C);

    agen_A a_kept_idx_1 = init_agen_A_from_cfg(cfg.st_kept_idx_1_agen_cfg);

    int niter = cfg.second_pass_niter / 2;

    for (int i = 0; i < niter; i++) chess_loop_range(6, )
    chess_prepare_for_pipelining
    {
        // load all elements in the mutual block, of size 4x4, for both
        // x-groups of the pair
        // 0, 3 use C
        // 1 uses B
        // 2 uses A

        DVECX_TYPE x0 = DVLOAD(a_tmpC);
        DVECX_TYPE x1 = DVLOAD(a_tmpB);
        DVECX_TYPE x2 = DVLOAD(a_tmpA);
        DVECX_TYPE x3 = DVLOAD(a_tmpC);

        DVECX_TYPE x4 = DVLOAD(a_tmpC);
        DVECX_TYPE x5 = DVLOAD(a_tmpB);
        DVECX_TYPE x6 = DVLOAD(a_tmpA);
        DVECX_TYPE x7 = DVLOAD(a_tmpC);

        DVECX_TYPE x8  = DVLOAD(a_tmpC);
        DVECX_TYPE x9  = DVLOAD(a_tmpB);
        DVECX_TYPE x10 = DVLOAD(a_tmpA);
        DVECX_TYPE x11 = DVLOAD(a_tmpC);

        DVECX_TYPE x12 = DVLOAD(a_tmpC);
        DVECX_TYPE x13 = DVLOAD(a_tmpB);
        DVECX_TYPE x14 = DVLOAD(a_tmpA);
        DVECX_TYPE x15 = DVLOAD(a_tmpC);

        // step 1: sort the mutual block (same comparator network as the
        // single-vector version, double width)
        // [[0, 4], [2, 6], [2, 4], [1, 5], [3, 7], [3, 5], [1, 2], [3, 4],
        // [5, 6], [8, 12], [10, 14], [10, 12], [9, 13], [11, 15], [11, 13],
        // [9, 10], [11, 12], [13, 14], [0, 8], [4, 12], [4, 8], [2, 10], [6, 14],
        // [6, 10], [2, 4], [6, 8], [10, 12], [1, 9], [5, 13], [5, 9], [3, 11],
        // [7, 15], [7, 11], [3, 5], [7, 9], [11, 13], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]

        dvsort2(x0, x4, x0, x4);
        dvsort2(x2, x6, x2, x6);
        dvsort2(x2, x4, x2, x4);
        dvsort2(x1, x5, x1, x5);
        dvsort2(x3, x7, x3, x7);
        dvsort2(x3, x5, x3, x5);
        dvsort2(x1, x2, x1, x2);
        dvsort2(x3, x4, x3, x4);
        dvsort2(x5, x6, x5, x6);
        dvsort2(x8, x12, x8, x12);
        dvsort2(x10, x14, x10, x14);
        dvsort2(x10, x12, x10, x12);
        dvsort2(x9, x13, x9, x13);
        dvsort2(x11, x15, x11, x15);
        dvsort2(x11, x13, x11, x13);
        dvsort2(x9, x10, x9, x10);
        dvsort2(x11, x12, x11, x12);
        dvsort2(x13, x14, x13, x14);
        x8 = dvmax(x0, x8);
        dvsort2(x4, x12, x4, x12);
        dvsort2(x4, x8, x4, x8);
        dvsort2(x2, x10, x2, x10);
        x6 = dvmin(x6, x14);
        dvsort2(x6, x10, x6, x10);
        x4 = dvmax(x2, x4);
        dvsort2(x6, x8, x6, x8);
        dvsort2(x10, x12, x10, x12);
        x9 = dvmax(x1, x9);
        dvsort2(x5, x13, x5, x13);
        dvsort2(x5, x9, x5, x9);
        dvsort2(x3, x11, x3, x11);
        x7 = dvmin(x7, x15);
        dvsort2(x7, x11, x7, x11);
        dvsort2(x3, x5, x3, x5);
        dvsort2(x7, x9, x7, x9);
        x11 = dvmin(x11, x13);
        dvsort2(x3, x4, x3, x4);
        dvsort2(x5, x6, x5, x6);
        dvsort2(x7, x8, x7, x8);
        dvsort2(x9, x10, x9, x10);
        dvsort2(x11, x12, x11, x12);

        // kept indices: [3, 4, 5, 6, 7, 8, 9, 10, 11, 12], value-major pairs
        vstore(x3, a_kept_idx_1);
        vstore(x4, a_kept_idx_1);
        vstore(x5, a_kept_idx_1);
        vstore(x6, a_kept_idx_1);
        vstore(x7, a_kept_idx_1);
        vstore(x8, a_kept_idx_1);
        vstore(x9, a_kept_idx_1);
        vstore(x10, a_kept_idx_1);
        vstore(x11, a_kept_idx_1);
        vstore(x12, a_kept_idx_1);
    }
}

// step 2: merge kept list 1 with the top/bottom row (dv). Shared body,
// instantiated as overloads per kept-idx-2 agen bank.
#define MERGE_KEPT_LIST_1_WITH_ROW_10_4_DV_BODY                                       \
    for (int i = 0; i < niter; i++) chess_loop_range(6, )                             \
    chess_prepare_for_pipelining                                                      \
    {                                                                                 \
        /* row values for the x-group pair                                          \
           0, 3 use C; 1 uses B; 2 uses A */ \
        DVECX_TYPE x0 = DVLOAD(a_rowC);                                               \
        DVECX_TYPE x1 = DVLOAD(a_rowB);                                               \
        DVECX_TYPE x2 = DVLOAD(a_rowA);                                               \
        DVECX_TYPE x3 = DVLOAD(a_rowC);                                               \
                                                                                      \
        /* kept indices 1, one dv per value */                                        \
        DVECX_TYPE x4  = DVLOAD(a_kept_idx_1);                                        \
        DVECX_TYPE x5  = DVLOAD(a_kept_idx_1);                                        \
        DVECX_TYPE x6  = DVLOAD(a_kept_idx_1);                                        \
        DVECX_TYPE x7  = DVLOAD(a_kept_idx_1);                                        \
        DVECX_TYPE x8  = DVLOAD(a_kept_idx_1);                                        \
        DVECX_TYPE x9  = DVLOAD(a_kept_idx_1);                                        \
        DVECX_TYPE x10 = DVLOAD(a_kept_idx_1);                                        \
        DVECX_TYPE x11 = DVLOAD(a_kept_idx_1);                                        \
        DVECX_TYPE x12 = DVLOAD(a_kept_idx_1);                                        \
        DVECX_TYPE x13 = DVLOAD(a_kept_idx_1);                                        \
                                                                                      \
        /* merge with the row, new row first                                        \
           [[0, 4], [4, 12], [4, 8], [2, 6], [6, 10], [2, 4], [6, 8], [10, 12],    \
           [1, 5], [5, 13], [5, 9], [3, 7], [7, 11], [3, 5], [7, 9], [3, 4],       \
           [5, 6], [7, 8], [9, 10]]                                                 \
           Same fold choices as the single-vector version (see its notes). */ \
                                                                                      \
        x4 = dvmax(x0, x4);                                                           \
        dvsort2(x2, x6, x2, x6);                                                      \
        x5 = dvmax(x1, x5);                                                           \
        x5 = dvmin(x5, x13);                                                          \
        dvsort2(x3, x7, x3, x7);                                                      \
        dvsort2(x4, x12, x4, x12);                                                    \
        dvsort2(x6, x10, x6, x10);                                                    \
        dvsort2(x5, x9, x5, x9);                                                      \
        x7 = dvmin(x7, x11);                                                          \
        dvsort2(x4, x8, x4, x8);                                                      \
        dvsort2(x3, x5, x3, x5);                                                      \
        dvsort2(x7, x9, x7, x9);                                                      \
        dvsort2(x6, x8, x6, x8);                                                      \
        x4 = dvmax3(x2, x4, x3);                                                      \
        dvsort2(x5, x6, x5, x6);                                                      \
        dvsort2(x7, x8, x7, x8);                                                      \
        x9 = dvmin3(x9, x10, x12);                                                    \
                                                                                      \
        /* kept indices 2: [4, 5, 6, 7, 8, 9], value-major pairs */                   \
        vstore(x4, a_kept_idx_2);                                                     \
        vstore(x5, a_kept_idx_2);                                                     \
        vstore(x6, a_kept_idx_2);                                                     \
        vstore(x7, a_kept_idx_2);                                                     \
        vstore(x8, a_kept_idx_2);                                                     \
        vstore(x9, a_kept_idx_2);                                                     \
    }

// step 2.1: merge with top row
void merge_with_top_row_10_4_dv_exec(const MedianFilterSepa2x2Config &cfg)
{
    agen_A a_rowA = init_agen_A_from_cfg(cfg.ld_first_row_agen_cfg_A);
    agen_B a_rowB = init_agen_B_from_cfg(cfg.ld_first_row_agen_cfg_B);
    agen_C a_rowC = init_agen_C_from_cfg(cfg.ld_first_row_agen_cfg_C);

    // load kept indices 1
    agen_A a_kept_idx_1 = init_agen_A_from_cfg(cfg.ld_kept_idx_1_agen_cfg);

    // store kept indices 2
    agen_B a_kept_idx_2 = init_agen_B_from_cfg(cfg.st_kept_idx_2_top_agen_cfg);

    int niter = cfg.second_pass_niter / 2;

    MERGE_KEPT_LIST_1_WITH_ROW_10_4_DV_BODY
}

// step 2.2: merge with bottom row
void merge_with_bottom_row_10_4_dv_exec(const MedianFilterSepa2x2Config &cfg)
{
    agen_A a_rowA = init_agen_A_from_cfg(cfg.ld_last_row_agen_cfg_A);
    agen_B a_rowB = init_agen_B_from_cfg(cfg.ld_last_row_agen_cfg_B);
    agen_C a_rowC = init_agen_C_from_cfg(cfg.ld_last_row_agen_cfg_C);

    // load kept indices 1
    agen_A a_kept_idx_1 = init_agen_A_from_cfg(cfg.ld_kept_idx_1_agen_cfg);

    // store kept indices 2
    agen_C a_kept_idx_2 = init_agen_C_from_cfg(cfg.st_kept_idx_2_bottom_agen_cfg);

    int niter = cfg.second_pass_niter / 2;

    MERGE_KEPT_LIST_1_WITH_ROW_10_4_DV_BODY
}

// steps 4 and 5: merge with the side columns and corners (dv). The kept
// indices 2 are loaded once per body and shared by the left and right paths
// (the single-vector version loads them twice through two agens).
// Statement macro rather than a helper function: older chess-clang releases
// do not inline a helper with six by-value dvcharx arguments, producing a
// call inside the pipelined loop and W-register spill/restore (VST/VLDWX_P)
// traffic around it.
#define MERGE_WITH_CORNER_6_4_1_DV_PATH(x_out, k4_, k5_, k6_, k7_, k8_, k9_, a_ld_sorted_col, x_cnr) \
    do                                                                                               \
    {                                                                                                \
        /* sorted column, one dv per value */                                                        \
        DVECX_TYPE x0 = DVLOAD(a_ld_sorted_col);                                                     \
        DVECX_TYPE x1 = DVLOAD(a_ld_sorted_col);                                                     \
        DVECX_TYPE x2 = DVLOAD(a_ld_sorted_col);                                                     \
        DVECX_TYPE x3 = DVLOAD(a_ld_sorted_col);                                                     \
                                                                                                     \
        /* kept values are read, never written (shared with the other side) */                       \
        DVECX_TYPE x4 = (k4_);                                                                       \
        DVECX_TYPE x5 = (k5_);                                                                       \
        DVECX_TYPE x6 = (k6_);                                                                       \
        DVECX_TYPE x7 = (k7_);                                                                       \
        DVECX_TYPE x8 = (k8_);                                                                       \
        DVECX_TYPE x9 = (k9_);                                                                       \
                                                                                                     \
        /* [[0, 4], [4, 8], [2, 6], [2, 4], [6, 8], [1, 5],                         \
            [5, 9], [3, 7], [3, 5], [3, 4], [5, 6]]                                 \
           vmid3 folds as in the single-vector version (bit-exact, 0-1 proven). */                \
        x5 = dvmid3(x1, x5, x9);                                                                     \
        x3 = dvmin(x3, x7);                                                                          \
        x4 = dvmax(x0, x4);                                                                          \
        dvsort2(x2, x6, x2, x6);                                                                     \
        x6 = dvmid3(x4, x8, x6);                                                                     \
        x4 = dvmid3(x4, x8, x2);                                                                     \
        dvsort2(x3, x5, x3, x5);                                                                     \
                                                                                                     \
        /* kept indices 3: [4, 5]; merge with the corner [[0, 1], [1, 2]] */                         \
        x4      = dvmax3(x3, x4, (x_cnr));                                                           \
        (x_out) = dvmin3(x4, x5, x6);                                                                \
    } while (0)

#define MERGE_WITH_CORNERS_6_4_1_DV_BODY                                                            \
    for (int i = 0; i < niter; i++) chess_loop_range(6, )                                           \
    chess_prepare_for_pipelining                                                                    \
    {                                                                                               \
        /* kept indices 2, shared by the left and right paths */                                    \
        DVECX_TYPE k4 = DVLOAD(a_kept_idx_2);                                                       \
        DVECX_TYPE k5 = DVLOAD(a_kept_idx_2);                                                       \
        DVECX_TYPE k6 = DVLOAD(a_kept_idx_2);                                                       \
        DVECX_TYPE k7 = DVLOAD(a_kept_idx_2);                                                       \
        DVECX_TYPE k8 = DVLOAD(a_kept_idx_2);                                                       \
        DVECX_TYPE k9 = DVLOAD(a_kept_idx_2);                                                       \
                                                                                                    \
        /* corner pixels for both x-groups of the pair (lo halves = left/even       \
           columns, hi halves = right/odd columns) */               \
        DVECX_TYPE cl0 = DVLOAD_DI(a_left_corner);                                                  \
        DVECX_TYPE cl1 = DVLOAD_DI(a_left_corner);                                                  \
        DVECX_TYPE x_cl;                                                                            \
        x_cl.lo        = cl0.lo;                                                                    \
        x_cl.hi        = cl1.lo;                                                                    \
        DVECX_TYPE cr0 = DVLOAD_DI(a_right_corner);                                                 \
        DVECX_TYPE cr1 = DVLOAD_DI(a_right_corner);                                                 \
        DVECX_TYPE x_cr;                                                                            \
        x_cr.lo = cr0.hi;                                                                           \
        x_cr.hi = cr1.hi;                                                                           \
                                                                                                    \
        DVECX_TYPE outL;                                                                            \
        MERGE_WITH_CORNER_6_4_1_DV_PATH(outL, k4, k5, k6, k7, k8, k9, a_ld_sorted_left_col, x_cl);  \
        DVECX_TYPE outR;                                                                            \
        MERGE_WITH_CORNER_6_4_1_DV_PATH(outR, k4, k5, k6, k7, k8, k9, a_ld_sorted_right_col, x_cr); \
                                                                                                    \
        /* re-pair {left|right} per x-group for the interleaved stores */                           \
        DVECX_TYPE o0;                                                                              \
        o0.lo = outL.lo;                                                                            \
        o0.hi = outR.lo;                                                                            \
        DVECX_TYPE o1;                                                                              \
        o1.lo = outL.hi;                                                                            \
        o1.hi = outR.hi;                                                                            \
        vstore_i(o0, a_out);                                                                        \
        vstore_i(o1, a_out);                                                                        \
    }

void merge_with_top_corners_6_4_1_dv_exec(const MedianFilterSepa2x2Config &cfg)
{
    agen_B a_kept_idx_2 = init_agen_B_from_cfg(cfg.ld_kept_idx_2_top_agen_cfg);

    agen_A a_ld_sorted_left_col  = init_agen_A_from_cfg(cfg.ld_sorted_left_col_agen_cfg);
    agen_C a_ld_sorted_right_col = init_agen_C_from_cfg(cfg.ld_sorted_right_col_agen_cfg);

    agen_B a_left_corner  = init_agen_B_from_cfg(cfg.ld_top_left_corner_agen_cfg);
    agen_B a_right_corner = init_agen_B_from_cfg(cfg.ld_top_right_corner_agen_cfg);

    agen_C a_out = init_agen_C_from_cfg(cfg.out_agen_cfg_top);

    int niter = cfg.second_pass_niter / 2;

    MERGE_WITH_CORNERS_6_4_1_DV_BODY
}

void merge_with_bottom_corners_6_4_1_dv_exec(const MedianFilterSepa2x2Config &cfg)
{
    agen_C a_kept_idx_2 = init_agen_C_from_cfg(cfg.ld_kept_idx_2_bottom_agen_cfg);

    agen_A a_ld_sorted_left_col  = init_agen_A_from_cfg(cfg.ld_sorted_left_col_agen_cfg);
    agen_C a_ld_sorted_right_col = init_agen_C_from_cfg(cfg.ld_sorted_right_col_agen_cfg);

    agen_B a_left_corner  = init_agen_B_from_cfg(cfg.ld_bottom_left_corner_agen_cfg);
    agen_B a_right_corner = init_agen_B_from_cfg(cfg.ld_bottom_right_corner_agen_cfg);

    agen_C a_out = init_agen_C_from_cfg(cfg.out_agen_cfg_bottom);

    int niter = cfg.second_pass_niter / 2;

    MERGE_WITH_CORNERS_6_4_1_DV_BODY
}

void serial_run_sepa_2x2_5x5_exec(const MedianFilterSepa2x2Config &cfg)
{
    // 1st pass: sort horizontal rows (single-vector; floored by the
    // deinterleaving loads)
    sort_row_4_exec(cfg);

    // 2nd pass:
    // step 1: sort mutual block (dv)
    sort_mutual_block_16_dv_exec(cfg);

    // step 2: merge with top and bottom rows (dv)
    merge_with_top_row_10_4_dv_exec(cfg);
    merge_with_bottom_row_10_4_dv_exec(cfg);

    // step 3: sort left and right columns (shared single-vector kernels;
    // the store agens carry the x-group-paired layout)
    sort_left_column_4_exec(cfg);
    sort_right_column_4_exec(cfg);

    // step 4: merge with left and right columns
    // step 5: merge with corners (dv)
    merge_with_top_corners_6_4_1_dv_exec(cfg);
    merge_with_bottom_corners_6_4_1_dv_exec(cfg);
}
