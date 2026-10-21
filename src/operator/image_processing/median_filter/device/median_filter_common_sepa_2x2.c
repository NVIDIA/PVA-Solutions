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

#include "median_filter_common_sepa_2x2.h"

// shared codes between SEPA_2x2 kernels

// 1st pass: sort rows of 6 (shared by 5x7 and 7x7)
void sort_row_6_exec(const MedianFilterSepa2x2Config &cfg)
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
        // hw only support even address for dv
        VECX_TYPE x0   = DVLOAD_DI(a_in).hi;
        DVECX_TYPE dv1 = DVLOAD_DI(a_in);
        VECX_TYPE x1   = dv1.lo;
        VECX_TYPE x2   = dv1.hi;
        DVECX_TYPE dv2 = DVLOAD_DI(a_in);
        VECX_TYPE x3   = dv2.lo;
        VECX_TYPE x4   = dv2.hi;
        DVECX_TYPE dv3 = DVLOAD_DI(a_in);
        VECX_TYPE x5   = dv3.lo;

        SORT_6;

        // 0, 3 use A
        // 1, 4 use B
        // 2, 5 use C
        vstore(x0, a_tmpA);
        vstore(x1, a_tmpB);
        vstore(x2, a_tmpC);
        vstore(x3, a_tmpA);
        vstore(x4, a_tmpB);
        vstore(x5, a_tmpC);
    }
}

// step 3: sort the side columns (shared by 5x5 and 5x7)
void sort_left_column_4_exec(const MedianFilterSepa2x2Config &cfg)
{
    agen_B a_ld_left_col        = init_agen_B_from_cfg(cfg.ld_left_col_sort_agen_cfg);
    agen_A a_st_sorted_left_col = init_agen_A_from_cfg(cfg.st_sorted_left_col_agen_cfg);

    int niter = max(cfg.second_pass_niter, 6);
    niter     = (niter + 1) / 2 * 2;

    for (int i = 0; i < niter; i++) chess_loop_range(6, )
    chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        // load the left column KNL_H - 1 elements to be sorted
        // start from the 2nd row
        VECX_TYPE x0 = DVLOAD_DI(a_ld_left_col).lo;
        VECX_TYPE x1 = DVLOAD_DI(a_ld_left_col).lo;
        VECX_TYPE x2 = DVLOAD_DI(a_ld_left_col).lo;
        VECX_TYPE x3 = DVLOAD_DI(a_ld_left_col).lo;

        // sort the left column
        SORT_4;

        // store the sorted left column
        vstore(x0, a_st_sorted_left_col);
        vstore(x1, a_st_sorted_left_col);
        vstore(x2, a_st_sorted_left_col);
        vstore(x3, a_st_sorted_left_col);
    }
}

void sort_right_column_4_exec(const MedianFilterSepa2x2Config &cfg)
{
    agen_B a_ld_right_col        = init_agen_B_from_cfg(cfg.ld_right_col_sort_agen_cfg);
    agen_C a_st_sorted_right_col = init_agen_C_from_cfg(cfg.st_sorted_right_col_agen_cfg);

    int niter = max(cfg.second_pass_niter, 6);
    niter     = (niter + 1) / 2 * 2;

    for (int i = 0; i < niter; i++) chess_loop_range(6, )
    chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        // load the right column KNL_H - 1 elements to be sorted
        // start from the 2nd row
        // hw only supports even addresses for double-vector loads
        VECX_TYPE x0 = DVLOAD_DI(a_ld_right_col).hi;
        VECX_TYPE x1 = DVLOAD_DI(a_ld_right_col).hi;
        VECX_TYPE x2 = DVLOAD_DI(a_ld_right_col).hi;
        VECX_TYPE x3 = DVLOAD_DI(a_ld_right_col).hi;

        // sort the right column
        SORT_4;

        // store the sorted right column
        vstore(x0, a_st_sorted_right_col);
        vstore(x1, a_st_sorted_right_col);
        vstore(x2, a_st_sorted_right_col);
        vstore(x3, a_st_sorted_right_col);
    }
}

// step 4: merge with side columns
// step 5: merge with corners

// The 5x5 kernel provides its own double-vector corner merges; keep the
// single-vector versions out of its program image.
#if !defined(KNL_5x5)

// Statement macro rather than a helper function: the kept-idx-2 and
// sorted-column agens have different register classes per call site
#    define MERGE_WITH_CORNER_6_4_1_PATH(x_out, a_kept_idx_2, a_ld_sorted_col, x_cnr)   \
        do                                                                              \
        {                                                                               \
            /* load sorted column */                                                    \
            DVECX_TYPE dv0_1 = DVLOAD(a_ld_sorted_col);                                 \
            VECX_TYPE x0     = dv0_1.lo;                                                \
            VECX_TYPE x1     = dv0_1.hi;                                                \
            DVECX_TYPE dv2_3 = DVLOAD(a_ld_sorted_col);                                 \
            VECX_TYPE x2     = dv2_3.lo;                                                \
            VECX_TYPE x3     = dv2_3.hi;                                                \
                                                                                        \
            /* load kept indices 2 */                                                   \
            DVECX_TYPE dv4_5 = DVLOAD(a_kept_idx_2);                                    \
            VECX_TYPE x4     = dv4_5.lo;                                                \
            VECX_TYPE x5     = dv4_5.hi;                                                \
            DVECX_TYPE dv6_7 = DVLOAD(a_kept_idx_2);                                    \
            VECX_TYPE x6     = dv6_7.lo;                                                \
            VECX_TYPE x7     = dv6_7.hi;                                                \
            DVECX_TYPE dv8_9 = DVLOAD(a_kept_idx_2);                                    \
            VECX_TYPE x8     = dv8_9.lo;                                                \
            VECX_TYPE x9     = dv8_9.hi;                                                \
                                                                                        \
            /* [[0, 4], [4, 8], [2, 6], [2, 4], [6, 8], [1, 5],                       \
            [5, 9], [3, 7], [3, 5], [3, 4], [5, 6]]                               \
           vmid3 folds below deviate from the comparator list per-wire but keep  \
           the final median bit-exact; proven by exhaustive 0-1 enumeration. */ \
            x5 = vmid3(x1, x5, x9);                                                     \
            x3 = vmin(x3, x7);                                                          \
            x4 = vmax(x0, x4);                                                          \
            vsort2(x2, x6, x2, x6);                                                     \
            x6 = vmid3(x4, x8, x6);                                                     \
            x4 = vmid3(x4, x8, x2);                                                     \
            vsort2(x3, x5, x3, x5);                                                     \
                                                                                        \
            /* kept indices 3: [4, 5]                                                 \
           merge with the selected corner                                         \
           [[0, 1], [1, 2]] */ \
            x4      = vmax3(x3, x4, x_cnr);                                             \
            (x_out) = vmin3(x4, x5, x6);                                                \
        } while (0)

void merge_with_top_corners_6_4_1_exec(const MedianFilterSepa2x2Config &cfg)
{
    agen_B a_kept_idx_2_top_left  = init_agen_B_from_cfg(cfg.ld_kept_idx_2_top_agen_cfg);
    agen_B a_kept_idx_2_top_right = init_agen_B_from_cfg(cfg.ld_kept_idx_2_top_agen_cfg);

    agen_A a_ld_sorted_left_col  = init_agen_A_from_cfg(cfg.ld_sorted_left_col_agen_cfg);
    agen_C a_ld_sorted_right_col = init_agen_C_from_cfg(cfg.ld_sorted_right_col_agen_cfg);

    agen_B a_top_left_corner  = init_agen_B_from_cfg(cfg.ld_top_left_corner_agen_cfg);
    agen_B a_top_right_corner = init_agen_B_from_cfg(cfg.ld_top_right_corner_agen_cfg);

    agen_C a_out_top = init_agen_C_from_cfg(cfg.out_agen_cfg_top);

    int niter = cfg.second_pass_niter;

    for (int i = 0; i < niter; i++) chess_loop_range(6, )
    chess_prepare_for_pipelining
    {
        DVECX_TYPE dv_out;
        MERGE_WITH_CORNER_6_4_1_PATH(dv_out.lo, a_kept_idx_2_top_left, a_ld_sorted_left_col,
                                     DVLOAD_DI(a_top_left_corner).lo);
        MERGE_WITH_CORNER_6_4_1_PATH(dv_out.hi, a_kept_idx_2_top_right, a_ld_sorted_right_col,
                                     DVLOAD_DI(a_top_right_corner).hi);
        vstore_i(dv_out, a_out_top);
    }
}

void merge_with_bottom_corners_6_4_1_exec(const MedianFilterSepa2x2Config &cfg)
{
    agen_C a_kept_idx_2_bottom_left  = init_agen_C_from_cfg(cfg.ld_kept_idx_2_bottom_agen_cfg);
    agen_C a_kept_idx_2_bottom_right = init_agen_C_from_cfg(cfg.ld_kept_idx_2_bottom_agen_cfg);

    agen_A a_ld_sorted_left_col  = init_agen_A_from_cfg(cfg.ld_sorted_left_col_agen_cfg);
    agen_C a_ld_sorted_right_col = init_agen_C_from_cfg(cfg.ld_sorted_right_col_agen_cfg);

    agen_B a_bottom_left_corner  = init_agen_B_from_cfg(cfg.ld_bottom_left_corner_agen_cfg);
    agen_B a_bottom_right_corner = init_agen_B_from_cfg(cfg.ld_bottom_right_corner_agen_cfg);

    agen_C a_out_bottom = init_agen_C_from_cfg(cfg.out_agen_cfg_bottom);

    int niter = cfg.second_pass_niter;

    for (int i = 0; i < niter; i++) chess_loop_range(6, )
    chess_prepare_for_pipelining
    {
        DVECX_TYPE dv_out;
        MERGE_WITH_CORNER_6_4_1_PATH(dv_out.lo, a_kept_idx_2_bottom_left, a_ld_sorted_left_col,
                                     DVLOAD_DI(a_bottom_left_corner).lo);
        MERGE_WITH_CORNER_6_4_1_PATH(dv_out.hi, a_kept_idx_2_bottom_right, a_ld_sorted_right_col,
                                     DVLOAD_DI(a_bottom_right_corner).hi);
        vstore_i(dv_out, a_out_bottom);
    }
}

#endif // !KNL_5x5
