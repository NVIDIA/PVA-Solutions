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

#include "median_filter_knl_sepa_2x2_7x7.h"

// step 1: sort the mutual block
void sort_mutual_block_36_exec(const MedianFilterSepa2x2Config &cfg)
{
    agen_A a_tmpA = init_agen_A_from_cfg(cfg.ld_mutual_block_agen_cfg_A);
    agen_B a_tmpB = init_agen_B_from_cfg(cfg.ld_mutual_block_agen_cfg_B);
    agen_C a_tmpC = init_agen_C_from_cfg(cfg.ld_mutual_block_agen_cfg_C);

    agen_A a_kept_idx_1 = init_agen_A_from_cfg(cfg.st_kept_idx_1_agen_cfg);

    int niter = cfg.second_pass_niter;

    for (int i = 0; i < niter; i++) chess_loop_range(6, )
    chess_prepare_for_pipelining
    {
        // load all elements in the mutual block, of size 6x6

        // 0, 3 use A
        // 1, 4 uses B
        // 2, 5 uses C

        VECX_TYPE x0 = VLOAD(a_tmpA);
        VECX_TYPE x1 = VLOAD(a_tmpB);
        VECX_TYPE x2 = VLOAD(a_tmpC);
        VECX_TYPE x3 = VLOAD(a_tmpA);
        VECX_TYPE x4 = VLOAD(a_tmpB);
        VECX_TYPE x5 = VLOAD(a_tmpC);

        VECX_TYPE x6  = VLOAD(a_tmpA);
        VECX_TYPE x7  = VLOAD(a_tmpB);
        VECX_TYPE x8  = VLOAD(a_tmpC);
        VECX_TYPE x9  = VLOAD(a_tmpA);
        VECX_TYPE x10 = VLOAD(a_tmpB);
        VECX_TYPE x11 = VLOAD(a_tmpC);

        VECX_TYPE x12 = VLOAD(a_tmpA);
        VECX_TYPE x13 = VLOAD(a_tmpB);
        VECX_TYPE x14 = VLOAD(a_tmpC);
        VECX_TYPE x15 = VLOAD(a_tmpA);
        VECX_TYPE x16 = VLOAD(a_tmpB);
        VECX_TYPE x17 = VLOAD(a_tmpC);

        VECX_TYPE x18 = VLOAD(a_tmpA);
        VECX_TYPE x19 = VLOAD(a_tmpB);
        VECX_TYPE x20 = VLOAD(a_tmpC);
        VECX_TYPE x21 = VLOAD(a_tmpA);
        VECX_TYPE x22 = VLOAD(a_tmpB);
        VECX_TYPE x23 = VLOAD(a_tmpC);

        VECX_TYPE x24 = VLOAD(a_tmpA);
        VECX_TYPE x25 = VLOAD(a_tmpB);
        VECX_TYPE x26 = VLOAD(a_tmpC);
        VECX_TYPE x27 = VLOAD(a_tmpA);
        VECX_TYPE x28 = VLOAD(a_tmpB);
        VECX_TYPE x29 = VLOAD(a_tmpC);

        VECX_TYPE x30 = VLOAD(a_tmpA);
        VECX_TYPE x31 = VLOAD(a_tmpB);
        VECX_TYPE x32 = VLOAD(a_tmpC);
        VECX_TYPE x33 = VLOAD(a_tmpA);
        VECX_TYPE x34 = VLOAD(a_tmpB);
        VECX_TYPE x35 = VLOAD(a_tmpC);

        // step 1: merge the six sorted rows
        // Tree: (((row 0, row 1), row 2), ((row 3, row 4), row 5)).
        // Global rank pruning reduces the 175-comparator merge to 153 operations.
        // Dead outputs use vmin/vmax; one admitted same-direction fold uses vmin3,
        // leaving 152 vector-math operations for retained ranks [11, 24].
        // NOTE: three more op-saving folds exist and are functionally exact
        // (0-1 verified): x10/x12 vmax pair -> vmax3, and vmid3 pairs for the
        // vsort2+vmax+vmin triples around (x16,x18) and (x17,x19). All of them
        // push this register-limit-critical loop into VMov/AC0 spills and
        // regress II (78 -> 79..84 measured on GEN3). Keep the baseline forms.

        vsort2(x0, x6, x0, x6);
        vsort2(x4, x10, x4, x10);
        vsort2(x4, x6, x4, x6);
        vsort2(x2, x8, x2, x8);
        vsort2(x2, x4, x2, x4);
        vsort2(x6, x8, x6, x8);
        vsort2(x1, x7, x1, x7);
        vsort2(x5, x11, x5, x11);
        vsort2(x5, x7, x5, x7);
        vsort2(x3, x9, x3, x9);
        vsort2(x3, x5, x3, x5);
        vsort2(x7, x9, x7, x9);
        vsort2(x1, x2, x1, x2);
        vsort2(x3, x4, x3, x4);
        vsort2(x5, x6, x5, x6);
        vsort2(x7, x8, x7, x8);
        vsort2(x9, x10, x9, x10);
        vsort2(x0, x12, x0, x12);
        vsort2(x8, x12, x8, x12);
        vsort2(x4, x16, x4, x16);
        vsort2(x4, x8, x4, x8);
        vsort2(x12, x16, x12, x16);
        vsort2(x2, x14, x2, x14);
        vsort2(x10, x14, x10, x14);
        vsort2(x6, x10, x6, x10);
        vsort2(x2, x4, x2, x4);
        vsort2(x6, x8, x6, x8);
        vsort2(x10, x12, x10, x12);
        vsort2(x14, x16, x14, x16);
        vsort2(x1, x13, x1, x13);
        vsort2(x9, x13, x9, x13);
        vsort2(x5, x17, x5, x17);
        vsort2(x5, x9, x5, x9);
        vsort2(x13, x17, x13, x17);
        vsort2(x3, x15, x3, x15);
        vsort2(x11, x15, x11, x15);
        vsort2(x7, x11, x7, x11);
        vsort2(x3, x5, x3, x5);
        vsort2(x7, x9, x7, x9);
        vsort2(x11, x13, x11, x13);
        vsort2(x15, x17, x15, x17);
        vsort2(x1, x2, x1, x2);
        vsort2(x3, x4, x3, x4);
        vsort2(x5, x6, x5, x6);
        vsort2(x7, x8, x7, x8);
        vsort2(x9, x10, x9, x10);
        vsort2(x11, x12, x11, x12);
        vsort2(x13, x14, x13, x14);
        vsort2(x15, x16, x15, x16);
        vsort2(x18, x24, x18, x24);
        vsort2(x22, x28, x22, x28);
        vsort2(x22, x24, x22, x24);
        vsort2(x20, x26, x20, x26);
        vsort2(x20, x22, x20, x22);
        vsort2(x24, x26, x24, x26);
        vsort2(x19, x25, x19, x25);
        vsort2(x23, x29, x23, x29);
        vsort2(x23, x25, x23, x25);
        vsort2(x21, x27, x21, x27);
        vsort2(x21, x23, x21, x23);
        vsort2(x25, x27, x25, x27);
        vsort2(x19, x20, x19, x20);
        vsort2(x21, x22, x21, x22);
        vsort2(x23, x24, x23, x24);
        vsort2(x25, x26, x25, x26);
        vsort2(x27, x28, x27, x28);
        vsort2(x18, x30, x18, x30);
        vsort2(x26, x30, x26, x30);
        vsort2(x22, x34, x22, x34);
        vsort2(x22, x26, x22, x26);
        vsort2(x30, x34, x30, x34);
        vsort2(x20, x32, x20, x32);
        vsort2(x28, x32, x28, x32);
        vsort2(x24, x28, x24, x28);
        vsort2(x20, x22, x20, x22);
        vsort2(x24, x26, x24, x26);
        vsort2(x28, x30, x28, x30);
        vsort2(x32, x34, x32, x34);
        vsort2(x19, x31, x19, x31);
        vsort2(x27, x31, x27, x31);
        vsort2(x23, x35, x23, x35);
        vsort2(x23, x27, x23, x27);
        vsort2(x31, x35, x31, x35);
        vsort2(x21, x33, x21, x33);
        vsort2(x29, x33, x29, x33);
        vsort2(x25, x29, x25, x29);
        vsort2(x21, x23, x21, x23);
        vsort2(x25, x27, x25, x27);
        vsort2(x29, x31, x29, x31);
        vsort2(x33, x35, x33, x35);
        vsort2(x19, x20, x19, x20);
        vsort2(x21, x22, x21, x22);
        vsort2(x23, x24, x23, x24);
        vsort2(x25, x26, x25, x26);
        vsort2(x27, x28, x27, x28);
        vsort2(x29, x30, x29, x30);
        vsort2(x31, x32, x31, x32);
        vsort2(x33, x34, x33, x34);
        x18 = vmax(x0, x18);
        x16 = vmin(x16, x34);
        vsort2(x16, x18, x16, x18);
        vsort2(x8, x26, x8, x26);
        x16 = vmax(x8, x16);
        x18 = vmin(x18, x26);
        x22 = vmax(x4, x22);
        x12 = vmin(x12, x30);
        vsort2(x12, x22, x12, x22);
        vsort2(x12, x16, x12, x16);
        vsort2(x18, x22, x18, x22);
        x20 = vmax(x2, x20);
        vsort2(x10, x28, x10, x28);
        vsort2(x10, x20, x10, x20);
        vsort2(x6, x24, x6, x24);
        x14 = vmin(x14, x32);
        vsort2(x14, x24, x14, x24);
        x10 = vmax(x6, x10);
        vsort2(x14, x20, x14, x20);
        x24 = vmin(x24, x28);
        x12 = vmax(x10, x12);
        vsort2(x14, x16, x14, x16);
        vsort2(x18, x20, x18, x20);
        vsort2(x22, x24, x22, x24);
        x19 = vmax(x1, x19);
        x17 = vmin(x17, x35);
        vsort2(x17, x19, x17, x19);
        vsort2(x9, x27, x9, x27);
        x17 = vmax(x9, x17);
        x19 = vmin(x19, x27);
        x23 = vmax(x5, x23);
        x13 = vmin(x13, x31);
        vsort2(x13, x23, x13, x23);
        vsort2(x13, x17, x13, x17);
        vsort2(x19, x23, x19, x23);
        x21 = vmax(x3, x21);
        vsort2(x11, x29, x11, x29);
        vsort2(x11, x21, x11, x21);
        vsort2(x7, x25, x7, x25);
        x15 = vmin(x15, x33);
        vsort2(x15, x25, x15, x25);
        x11 = vmax(x7, x11);
        vsort2(x15, x21, x15, x21);
        vsort2(x11, x13, x11, x13);
        vsort2(x15, x17, x15, x17);
        vsort2(x19, x21, x19, x21);
        x23 = vmin3(x23, x25, x29);
        vsort2(x11, x12, x11, x12);
        vsort2(x13, x14, x13, x14);
        vsort2(x15, x16, x15, x16);
        vsort2(x17, x18, x17, x18);
        vsort2(x19, x20, x19, x20);
        vsort2(x21, x22, x21, x22);
        vsort2(x23, x24, x23, x24);

        // kept indices: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        vstore(x11, a_kept_idx_1);
        vstore(x12, a_kept_idx_1);
        vstore(x13, a_kept_idx_1);
        vstore(x14, a_kept_idx_1);
        vstore(x15, a_kept_idx_1);
        vstore(x16, a_kept_idx_1);
        vstore(x17, a_kept_idx_1);
        vstore(x18, a_kept_idx_1);
        vstore(x19, a_kept_idx_1);
        vstore(x20, a_kept_idx_1);
        vstore(x21, a_kept_idx_1);
        vstore(x22, a_kept_idx_1);
        vstore(x23, a_kept_idx_1);
        vstore(x24, a_kept_idx_1);
    }
}

// Body macro rather than a helper function: the kept-idx-2 agen has a
// different register class per call site (B for top, C for bottom), and the
// chess front end (C only) has no templates or overloading to abstract that.
// Expects locals a_rowA/a_rowB/a_rowC, a_kept_idx_1, a_kept_idx_2 and niter.
#define MERGE_KEPT_LIST_1_WITH_ROW_14_6_BODY                                              \
    for (int i = 0; i < niter; i++) chess_loop_range(6, )                                 \
    chess_prepare_for_pipelining                                                          \
    {                                                                                     \
        /* load row, #row = 6                                                           \
           0, 3 use A                                                                   \
           1, 4 uses B                                                                  \
           2, 5 uses C */ \
        VECX_TYPE x0 = VLOAD(a_rowA);                                                     \
        VECX_TYPE x1 = VLOAD(a_rowB);                                                     \
        VECX_TYPE x2 = VLOAD(a_rowC);                                                     \
        VECX_TYPE x3 = VLOAD(a_rowA);                                                     \
        VECX_TYPE x4 = VLOAD(a_rowB);                                                     \
        VECX_TYPE x5 = VLOAD(a_rowC);                                                     \
                                                                                          \
        /* load kept indices by double vectors, #kept_idx_1 = 14 */                       \
        DVECX_TYPE dv6_7   = DVLOAD(a_kept_idx_1);                                        \
        VECX_TYPE x6       = dv6_7.lo;                                                    \
        VECX_TYPE x7       = dv6_7.hi;                                                    \
        DVECX_TYPE dv8_9   = DVLOAD(a_kept_idx_1);                                        \
        VECX_TYPE x8       = dv8_9.lo;                                                    \
        VECX_TYPE x9       = dv8_9.hi;                                                    \
        DVECX_TYPE dv10_11 = DVLOAD(a_kept_idx_1);                                        \
        VECX_TYPE x10      = dv10_11.lo;                                                  \
        VECX_TYPE x11      = dv10_11.hi;                                                  \
        DVECX_TYPE dv12_13 = DVLOAD(a_kept_idx_1);                                        \
        VECX_TYPE x12      = dv12_13.lo;                                                  \
        VECX_TYPE x13      = dv12_13.hi;                                                  \
        DVECX_TYPE dv14_15 = DVLOAD(a_kept_idx_1);                                        \
        VECX_TYPE x14      = dv14_15.lo;                                                  \
        VECX_TYPE x15      = dv14_15.hi;                                                  \
        DVECX_TYPE dv16_17 = DVLOAD(a_kept_idx_1);                                        \
        VECX_TYPE x16      = dv16_17.lo;                                                  \
        VECX_TYPE x17      = dv16_17.hi;                                                  \
        DVECX_TYPE dv18_19 = DVLOAD(a_kept_idx_1);                                        \
        VECX_TYPE x18      = dv18_19.lo;                                                  \
        VECX_TYPE x19      = dv18_19.hi;                                                  \
                                                                                          \
        /* merge with the row, new row first                                            \
           [[0, 6], [6, 14], [4, 10], [10, 18], [4, 6], [10, 14], [2, 8], [8, 16],     \
           [8, 12], [6, 8], [10, 12], [14, 16], [1, 7], [7, 15], [5, 11], [11, 19],    \
           [5, 7], [11, 15], [3, 9], [9, 17], [9, 13], [3, 5], [7, 9], [11, 13],       \
           [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]]                                 \
           vmid3 folds below deviate from the comparator list per-wire but keep         \
           the retained ranks bit-exact; proven by exhaustive 0-1 enumeration. */ \
                                                                                          \
        x6 = vmax(x0, x6);                                                                \
        vsort2(x6, x14, x6, x14);                                                         \
        x6  = vmid3(x4, x10, x6);                                                         \
        x10 = vmid3(x4, x10, x18);                                                        \
        vsort2(x10, x14, x10, x14);                                                       \
        x8 = vmax(x2, x8);                                                                \
        vsort2(x8, x16, x8, x16);                                                         \
        vsort2(x8, x12, x8, x12);                                                         \
        vsort2(x6, x8, x6, x8);                                                           \
        vsort2(x10, x12, x10, x12);                                                       \
        x7 = vmax(x1, x7);                                                                \
        vsort2(x7, x15, x7, x15);                                                         \
        vsort2(x5, x11, x5, x11);                                                         \
        vsort2(x5, x7, x5, x7);                                                           \
        x11 = vmin3(x11, x19, x15);                                                       \
        vsort2(x3, x9, x3, x9);                                                           \
        x9 = vmin(x9, x17);                                                               \
        vsort2(x9, x13, x9, x13);                                                         \
        vsort2(x7, x9, x7, x9);                                                           \
        vsort2(x11, x13, x11, x13);                                                       \
        x6 = vmax3(x3, x5, x6);                                                           \
        vsort2(x7, x8, x7, x8);                                                           \
        vsort2(x9, x10, x9, x10);                                                         \
        vsort2(x11, x12, x11, x12);                                                       \
        x13 = vmin3(x13, x14, x16);                                                       \
                                                                                          \
        /* kept indices 2: [6, 7, 8, 9, 10, 11, 12, 13] */                                \
        vstore(x6, a_kept_idx_2);                                                         \
        vstore(x7, a_kept_idx_2);                                                         \
        vstore(x8, a_kept_idx_2);                                                         \
        vstore(x9, a_kept_idx_2);                                                         \
        vstore(x10, a_kept_idx_2);                                                        \
        vstore(x11, a_kept_idx_2);                                                        \
        vstore(x12, a_kept_idx_2);                                                        \
        vstore(x13, a_kept_idx_2);                                                        \
    }

// step 2.1: merge with top row
void merge_with_top_row_14_6_exec(const MedianFilterSepa2x2Config &cfg)
{
    agen_A a_rowA = init_agen_A_from_cfg(cfg.ld_first_row_agen_cfg_A);
    agen_B a_rowB = init_agen_B_from_cfg(cfg.ld_first_row_agen_cfg_B);
    agen_C a_rowC = init_agen_C_from_cfg(cfg.ld_first_row_agen_cfg_C);

    // load kept indices 1
    agen_A a_kept_idx_1 = init_agen_A_from_cfg(cfg.ld_kept_idx_1_agen_cfg);

    // store kept indices 2
    agen_B a_kept_idx_2 = init_agen_B_from_cfg(cfg.st_kept_idx_2_top_agen_cfg);

    int niter = cfg.second_pass_niter;

    MERGE_KEPT_LIST_1_WITH_ROW_14_6_BODY
}

// step 2.2: merge with bottom row
void merge_with_bottom_row_14_6_exec(const MedianFilterSepa2x2Config &cfg)
{
    agen_A a_rowA = init_agen_A_from_cfg(cfg.ld_last_row_agen_cfg_A);
    agen_B a_rowB = init_agen_B_from_cfg(cfg.ld_last_row_agen_cfg_B);
    agen_C a_rowC = init_agen_C_from_cfg(cfg.ld_last_row_agen_cfg_C);

    // load kept indices 1
    agen_A a_kept_idx_1 = init_agen_A_from_cfg(cfg.ld_kept_idx_1_agen_cfg);

    // store kept indices 2
    agen_C a_kept_idx_2 = init_agen_C_from_cfg(cfg.st_kept_idx_2_bottom_agen_cfg);

    int niter = cfg.second_pass_niter;

    MERGE_KEPT_LIST_1_WITH_ROW_14_6_BODY
}

// step 3: sort the side columns
void sort_left_column_6_exec(const MedianFilterSepa2x2Config &cfg)
{
    agen_B a_ld_left_col        = init_agen_B_from_cfg(cfg.ld_left_col_sort_agen_cfg);
    agen_A a_st_sorted_left_col = init_agen_A_from_cfg(cfg.st_sorted_left_col_agen_cfg);

    int niter = max((cfg.second_pass_niter + 1) / 2 * 2, 6);

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
        VECX_TYPE x4 = DVLOAD_DI(a_ld_left_col).lo;
        VECX_TYPE x5 = DVLOAD_DI(a_ld_left_col).lo;

        // sort the left column
        SORT_6;

        // store the sorted left column
        vstore(x0, a_st_sorted_left_col);
        vstore(x1, a_st_sorted_left_col);
        vstore(x2, a_st_sorted_left_col);
        vstore(x3, a_st_sorted_left_col);
        vstore(x4, a_st_sorted_left_col);
        vstore(x5, a_st_sorted_left_col);
    }
}

void sort_right_column_6_exec(const MedianFilterSepa2x2Config &cfg)
{
    agen_B a_ld_right_col        = init_agen_B_from_cfg(cfg.ld_right_col_sort_agen_cfg);
    agen_C a_st_sorted_right_col = init_agen_C_from_cfg(cfg.st_sorted_right_col_agen_cfg);

    int niter = max((cfg.second_pass_niter + 1) / 2 * 2, 6);

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
        VECX_TYPE x4 = DVLOAD_DI(a_ld_right_col).hi;
        VECX_TYPE x5 = DVLOAD_DI(a_ld_right_col).hi;

        SORT_6;

        // store the sorted right column
        vstore(x0, a_st_sorted_right_col);
        vstore(x1, a_st_sorted_right_col);
        vstore(x2, a_st_sorted_right_col);
        vstore(x3, a_st_sorted_right_col);
        vstore(x4, a_st_sorted_right_col);
        vstore(x5, a_st_sorted_right_col);
    }
}

// step 4: merge with side columns
// step 5: merge with corners

// Statement macro rather than a helper function: the kept-idx-2 and
// sorted-column agens have different register classes per call site, and the
// chess front end (C only) has no templates or overloading to abstract that.
#define MERGE_WITH_CORNER_8_6_1_PATH(x_out, a_kept_idx_2, a_ld_sorted_col, x_cnr)       \
    do                                                                                  \
    {                                                                                   \
        /* load sorted column */                                                        \
        DVECX_TYPE dv0_1 = DVLOAD(a_ld_sorted_col);                                     \
        VECX_TYPE x0     = dv0_1.lo;                                                    \
        VECX_TYPE x1     = dv0_1.hi;                                                    \
        DVECX_TYPE dv2_3 = DVLOAD(a_ld_sorted_col);                                     \
        VECX_TYPE x2     = dv2_3.lo;                                                    \
        VECX_TYPE x3     = dv2_3.hi;                                                    \
        DVECX_TYPE dv4_5 = DVLOAD(a_ld_sorted_col);                                     \
        VECX_TYPE x4     = dv4_5.lo;                                                    \
        VECX_TYPE x5     = dv4_5.hi;                                                    \
                                                                                        \
        /* load kept indices 2 */                                                       \
        DVECX_TYPE dv6_7   = DVLOAD(a_kept_idx_2);                                      \
        VECX_TYPE x6       = dv6_7.lo;                                                  \
        VECX_TYPE x7       = dv6_7.hi;                                                  \
        DVECX_TYPE dv8_9   = DVLOAD(a_kept_idx_2);                                      \
        VECX_TYPE x8       = dv8_9.lo;                                                  \
        VECX_TYPE x9       = dv8_9.hi;                                                  \
        DVECX_TYPE dv10_11 = DVLOAD(a_kept_idx_2);                                      \
        VECX_TYPE x10      = dv10_11.lo;                                                \
        VECX_TYPE x11      = dv10_11.hi;                                                \
        DVECX_TYPE dv12_13 = DVLOAD(a_kept_idx_2);                                      \
        VECX_TYPE x12      = dv12_13.lo;                                                \
        VECX_TYPE x13      = dv12_13.hi;                                                \
                                                                                        \
        /* [[0, 6], [4, 10], [4, 6], [2, 8], [8, 12], [6, 8], [1, 7], [5, 11],       \
           [5, 7], [3, 9], [9, 13], [3, 5], [7, 9], [5, 6], [7, 8]]                   \
           The vmid3 fold below deviates from the comparator list per-wire but        \
           keeps the final median bit-exact; proven by exhaustive 0-1 enumeration. */  \
        x4 = vmin(x4, x10);                                                             \
        x6 = vmax3(x0, x6, x4);                                                         \
        x8 = vmid3(x2, x8, x12);                                                        \
        vsort2(x6, x8, x6, x8);                                                         \
        x7 = vmax(x1, x7);                                                              \
        x5 = vmin(x5, x11);                                                             \
        vsort2(x5, x7, x5, x7);                                                         \
        vsort2(x3, x9, x3, x9);                                                         \
        x7 = vmin3(x7, x9, x13);                                                        \
        x6 = vmax3(x3, x5, x6);                                                         \
                                                                                        \
        /* kept indices 3: [6, 7]                                                     \
           merge with the selected corner                                             \
           [[0, 1], [1, 2]] */ \
        x6      = vmax(x_cnr, x6);                                                      \
        (x_out) = vmin3(x6, x7, x8);                                                    \
    } while (0)

void merge_with_top_corners_8_6_1_exec(const MedianFilterSepa2x2Config &cfg)
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
        MERGE_WITH_CORNER_8_6_1_PATH(dv_out.lo, a_kept_idx_2_top_left, a_ld_sorted_left_col,
                                     DVLOAD_DI(a_top_left_corner).lo);
        MERGE_WITH_CORNER_8_6_1_PATH(dv_out.hi, a_kept_idx_2_top_right, a_ld_sorted_right_col,
                                     DVLOAD_DI(a_top_right_corner).hi);
        vstore_i(dv_out, a_out_top);
    }
}

void merge_with_bottom_corners_8_6_1_exec(const MedianFilterSepa2x2Config &cfg)
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
        MERGE_WITH_CORNER_8_6_1_PATH(dv_out.lo, a_kept_idx_2_bottom_left, a_ld_sorted_left_col,
                                     DVLOAD_DI(a_bottom_left_corner).lo);
        MERGE_WITH_CORNER_8_6_1_PATH(dv_out.hi, a_kept_idx_2_bottom_right, a_ld_sorted_right_col,
                                     DVLOAD_DI(a_bottom_right_corner).hi);
        vstore_i(dv_out, a_out_bottom);
    }
}

void serial_run_sepa_2x2_7x7_exec(const MedianFilterSepa2x2Config &cfg)
{
    // 1st pass: sort horizontal rows
    sort_row_6_exec(cfg);

    // 2nd pass:
    // step 1: sort mutual block
    sort_mutual_block_36_exec(cfg);

    // step 2: merge with top and bottom rows
    merge_with_top_row_14_6_exec(cfg);
    merge_with_bottom_row_14_6_exec(cfg);

    // step 3: sort left and right columns
    sort_left_column_6_exec(cfg);
    sort_right_column_6_exec(cfg);

    // step 4: merge with left and right columns
    // step 5: merge with corners
    merge_with_top_corners_8_6_1_exec(cfg);
    merge_with_bottom_corners_8_6_1_exec(cfg);
}
