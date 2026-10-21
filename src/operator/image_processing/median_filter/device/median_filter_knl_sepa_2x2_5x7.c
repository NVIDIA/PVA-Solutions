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

#include "median_filter_knl_sepa_2x2_5x7.h"

// step 1: sort the mutual block
void sort_mutual_block_24_exec(const MedianFilterSepa2x2Config &cfg)
{
    agen_A a_tmpA = init_agen_A_from_cfg(cfg.ld_mutual_block_agen_cfg_A);
    agen_B a_tmpB = init_agen_B_from_cfg(cfg.ld_mutual_block_agen_cfg_B);
    agen_C a_tmpC = init_agen_C_from_cfg(cfg.ld_mutual_block_agen_cfg_C);

    agen_A a_kept_idx_1 = init_agen_A_from_cfg(cfg.st_kept_idx_1_agen_cfg);

    int niter = cfg.second_pass_niter;

    for (int i = 0; i < niter; i++) chess_loop_range(6, )
    chess_prepare_for_pipelining
    {
        // load all elements in the mutual block, of size (KNL_W-1) * (KNL_H-1)
        // only use the hi part: start from the 2nd element

        VECX_TYPE x9  = VLOAD(a_tmpA);
        VECX_TYPE x10 = VLOAD(a_tmpB);
        VECX_TYPE x11 = VLOAD(a_tmpC);
        VECX_TYPE x12 = VLOAD(a_tmpA);
        VECX_TYPE x13 = VLOAD(a_tmpB);
        VECX_TYPE x14 = VLOAD(a_tmpC);

        VECX_TYPE x17 = VLOAD(a_tmpA);
        VECX_TYPE x18 = VLOAD(a_tmpB);
        VECX_TYPE x19 = VLOAD(a_tmpC);
        VECX_TYPE x20 = VLOAD(a_tmpA);
        VECX_TYPE x21 = VLOAD(a_tmpB);
        VECX_TYPE x22 = VLOAD(a_tmpC);

        VECX_TYPE x25 = VLOAD(a_tmpA);
        VECX_TYPE x26 = VLOAD(a_tmpB);
        VECX_TYPE x27 = VLOAD(a_tmpC);
        VECX_TYPE x28 = VLOAD(a_tmpA);
        VECX_TYPE x29 = VLOAD(a_tmpB);
        VECX_TYPE x30 = VLOAD(a_tmpC);

        VECX_TYPE x33 = VLOAD(a_tmpA);
        VECX_TYPE x34 = VLOAD(a_tmpB);
        VECX_TYPE x35 = VLOAD(a_tmpC);
        VECX_TYPE x36 = VLOAD(a_tmpA);
        VECX_TYPE x37 = VLOAD(a_tmpB);
        VECX_TYPE x38 = VLOAD(a_tmpC);

        // step 1: sort the mutual block
        // [[9, 17], [13, 21], [13, 17], [11, 19], [11, 13], [17, 19], [10, 18], [14, 22],
        // [14, 18], [12, 20], [12, 14], [18, 20], [10, 11], [12, 13], [14, 17], [18, 19],
        // [20, 21], [25, 33], [29, 37], [29, 33], [27, 35], [27, 29], [33, 35], [26, 34],
        // [30, 38], [30, 34], [28, 36], [28, 30], [34, 36], [26, 27], [28, 29], [30, 33],
        // [34, 35], [36, 37], [9, 25], [19, 35], [19, 25], [13, 29], [13, 19], [25, 29], [11, 27],
        // [21, 37], [21, 27], [17, 33], [17, 21], [27, 33], [17, 19], [21, 25], [27, 29], [33, 35],
        // [10, 26], [20, 36], [20, 26], [14, 30], [14, 20], [26, 30], [12, 28], [22, 38], [22, 28],
        // [18, 34], [18, 22], [28, 34], [12, 14], [18, 20], [22, 26], [28, 30], [14, 17], [18, 19],
        // [20, 21], [22, 25], [26, 27], [28, 29], [30, 33]]

        vsort2(x9, x17, x9, x17);
        vsort2(x13, x21, x13, x21);
        vsort2(x13, x17, x13, x17);
        vsort2(x11, x19, x11, x19);
        vsort2(x11, x13, x11, x13);
        vsort2(x17, x19, x17, x19);
        vsort2(x10, x18, x10, x18);
        vsort2(x14, x22, x14, x22);
        vsort2(x14, x18, x14, x18);
        vsort2(x12, x20, x12, x20);
        vsort2(x12, x14, x12, x14);
        vsort2(x18, x20, x18, x20);
        vsort2(x10, x11, x10, x11);
        vsort2(x12, x13, x12, x13);
        vsort2(x14, x17, x14, x17);
        vsort2(x18, x19, x18, x19);
        vsort2(x20, x21, x20, x21);
        vsort2(x25, x33, x25, x33);
        vsort2(x29, x37, x29, x37);
        vsort2(x29, x33, x29, x33);
        vsort2(x27, x35, x27, x35);
        vsort2(x27, x29, x27, x29);
        vsort2(x33, x35, x33, x35);
        vsort2(x26, x34, x26, x34);
        vsort2(x30, x38, x30, x38);
        vsort2(x30, x34, x30, x34);
        vsort2(x28, x36, x28, x36);
        vsort2(x28, x30, x28, x30);
        vsort2(x34, x36, x34, x36);
        vsort2(x26, x27, x26, x27);
        vsort2(x28, x29, x28, x29);
        vsort2(x30, x33, x30, x33);
        vsort2(x34, x35, x34, x35);
        vsort2(x36, x37, x36, x37);
        x25 = vmax(x9, x25);
        vsort2(x19, x35, x19, x35);
        vsort2(x19, x25, x19, x25);
        vsort2(x13, x29, x13, x29);
        x19 = vmax(x13, x19);
        vsort2(x25, x29, x25, x29);
        x27 = vmax(x11, x27);
        x21 = vmin(x21, x37);
        vsort2(x21, x27, x21, x27);
        vsort2(x17, x33, x17, x33);
        vsort2(x17, x21, x17, x21);
        vsort2(x27, x33, x27, x33);
        vsort2(x17, x19, x17, x19);
        vsort2(x21, x25, x21, x25);
        vsort2(x27, x29, x27, x29);
        x26 = vmax(x10, x26);
        x20 = vmin(x20, x36);
        vsort2(x20, x26, x20, x26);
        vsort2(x14, x30, x14, x30);
        vsort2(x14, x20, x14, x20);
        vsort2(x26, x30, x26, x30);
        vsort2(x12, x28, x12, x28);
        x22 = vmin(x22, x38);
        vsort2(x22, x28, x22, x28);
        vsort2(x18, x34, x18, x34);
        vsort2(x18, x22, x18, x22);
        x28 = vmin(x28, x34);
        vsort2(x18, x20, x18, x20);
        vsort2(x22, x26, x22, x26);
        vsort2(x28, x30, x28, x30);
        x17 = vmax3(x12, x14, x17);
        vsort2(x18, x19, x18, x19);
        vsort2(x20, x21, x20, x21);
        vsort2(x22, x25, x22, x25);
        vsort2(x26, x27, x26, x27);
        vsort2(x28, x29, x28, x29);
        x30 = vmin3(x30, x33, x35);

        // kept indices: [17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30]
        vstore(x17, a_kept_idx_1);
        vstore(x18, a_kept_idx_1);
        vstore(x19, a_kept_idx_1);
        vstore(x20, a_kept_idx_1);
        vstore(x21, a_kept_idx_1);
        vstore(x22, a_kept_idx_1);
        vstore(x25, a_kept_idx_1);
        vstore(x26, a_kept_idx_1);
        vstore(x27, a_kept_idx_1);
        vstore(x28, a_kept_idx_1);
        vstore(x29, a_kept_idx_1);
        vstore(x30, a_kept_idx_1);
    }
}

// Body macro rather than a helper function: the kept-idx-2 agen has a
// different register class per call site (B for top, C for bottom), and the
// chess front end (C only) has no templates or overloading to abstract that.
// Expects locals a_rowA/a_rowB/a_rowC, a_kept_idx_1, a_kept_idx_2 and niter.
#define MERGE_KEPT_LIST_1_WITH_ROW_12_6_BODY                                                                    \
                                                                                                                \
    for (int i = 0; i < niter; i++) chess_loop_range(6, )                                                       \
    chess_prepare_for_pipelining                                                                                \
    {                                                                                                           \
        /* load row */                                                                                          \
        VECX_TYPE x0 = VLOAD(a_rowA);                                                                           \
        VECX_TYPE x1 = VLOAD(a_rowB);                                                                           \
        VECX_TYPE x2 = VLOAD(a_rowC);                                                                           \
        VECX_TYPE x3 = VLOAD(a_rowA);                                                                           \
        VECX_TYPE x4 = VLOAD(a_rowB);                                                                           \
        VECX_TYPE x5 = VLOAD(a_rowC);                                                                           \
                                                                                                                \
        /* load kept indices by double vectors */                                                               \
        DVECX_TYPE dv6_7   = DVLOAD(a_kept_idx_1);                                                              \
        VECX_TYPE x6       = dv6_7.lo;                                                                          \
        VECX_TYPE x7       = dv6_7.hi;                                                                          \
        DVECX_TYPE dv8_9   = DVLOAD(a_kept_idx_1);                                                              \
        VECX_TYPE x8       = dv8_9.lo;                                                                          \
        VECX_TYPE x9       = dv8_9.hi;                                                                          \
        DVECX_TYPE dv10_11 = DVLOAD(a_kept_idx_1);                                                              \
        VECX_TYPE x10      = dv10_11.lo;                                                                        \
        VECX_TYPE x11      = dv10_11.hi;                                                                        \
        DVECX_TYPE dv12_13 = DVLOAD(a_kept_idx_1);                                                              \
        VECX_TYPE x12      = dv12_13.lo;                                                                        \
        VECX_TYPE x13      = dv12_13.hi;                                                                        \
        DVECX_TYPE dv14_15 = DVLOAD(a_kept_idx_1);                                                              \
        VECX_TYPE x14      = dv14_15.lo;                                                                        \
        VECX_TYPE x15      = dv14_15.hi;                                                                        \
        DVECX_TYPE dv16_17 = DVLOAD(a_kept_idx_1);                                                              \
        VECX_TYPE x16      = dv16_17.lo;                                                                        \
        VECX_TYPE x17      = dv16_17.hi;                                                                        \
                                                                                                                \
        /* merge with the row, new row first                                                                  \
           [[0, 6], [6, 14], [4, 10], [4, 6], [10, 14], [2, 8], [8, 16], [8, 12], [6, 8], [10, 12], [1, 7],  \
           [7, 15], [5, 11], [5, 7], [11, 15], [3, 9], [9, 17], [9, 13], [3, 5], [7, 9], [11, 13], [5, 6],   \
           [7, 8], [9, 10], [11, 12]]                                                                         \
           vmid3/vmin3 folds below deviate from the comparator list per-wire but                              \
           keep the final median bit-exact; proven by exhaustive 0-1 enumeration. */ \
        x6 = vmax(x0, x6);                                                                                      \
        vsort2(x4, x10, x4, x10);                                                                               \
        x10 = vmid3(x6, x14, x10);                                                                              \
        x6  = vmid3(x6, x14, x4);                                                                               \
        x8  = vmid3(x2, x8, x16);                                                                               \
        vsort2(x8, x12, x8, x12);                                                                               \
        vsort2(x6, x8, x6, x8);                                                                                 \
        vsort2(x10, x12, x10, x12);                                                                             \
        x7 = vmax(x1, x7);                                                                                      \
        vsort2(x7, x15, x7, x15);                                                                               \
        vsort2(x5, x11, x5, x11);                                                                               \
        vsort2(x5, x7, x5, x7);                                                                                 \
        vsort2(x3, x9, x3, x9);                                                                                 \
        x9 = vmin(x9, x17);                                                                                     \
        vsort2(x9, x13, x9, x13);                                                                               \
        vsort2(x7, x9, x7, x9);                                                                                 \
        x6 = vmax3(x3, x5, x6);                                                                                 \
        vsort2(x7, x8, x7, x8);                                                                                 \
        vsort2(x9, x10, x9, x10);                                                                               \
        x11 = vmin3(x11, x15, x13);                                                                             \
        x11 = vmin(x11, x12);                                                                                   \
                                                                                                                \
        /* kept indices 2: [6, 7, 8, 9, 10, 11] */                                                              \
        vstore(x6, a_kept_idx_2);                                                                               \
        vstore(x7, a_kept_idx_2);                                                                               \
        vstore(x8, a_kept_idx_2);                                                                               \
        vstore(x9, a_kept_idx_2);                                                                               \
        vstore(x10, a_kept_idx_2);                                                                              \
        vstore(x11, a_kept_idx_2);                                                                              \
    }

// step 2.1: merge with top row
void merge_with_top_row_12_6_exec(const MedianFilterSepa2x2Config &cfg)
{
    agen_A a_rowA = init_agen_A_from_cfg(cfg.ld_first_row_agen_cfg_A);
    agen_B a_rowB = init_agen_B_from_cfg(cfg.ld_first_row_agen_cfg_B);
    agen_C a_rowC = init_agen_C_from_cfg(cfg.ld_first_row_agen_cfg_C);

    // load kept indices 1
    agen_A a_kept_idx_1 = init_agen_A_from_cfg(cfg.ld_kept_idx_1_agen_cfg);

    // store kept indices 2
    agen_B a_kept_idx_2 = init_agen_B_from_cfg(cfg.st_kept_idx_2_top_agen_cfg);

    int niter = cfg.second_pass_niter;

    MERGE_KEPT_LIST_1_WITH_ROW_12_6_BODY
}

// step 2.2: merge with bottom row
void merge_with_bottom_row_12_6_exec(const MedianFilterSepa2x2Config &cfg)
{
    agen_A a_rowA = init_agen_A_from_cfg(cfg.ld_last_row_agen_cfg_A);
    agen_B a_rowB = init_agen_B_from_cfg(cfg.ld_last_row_agen_cfg_B);
    agen_C a_rowC = init_agen_C_from_cfg(cfg.ld_last_row_agen_cfg_C);

    // load kept indices 1
    agen_A a_kept_idx_1 = init_agen_A_from_cfg(cfg.ld_kept_idx_1_agen_cfg);

    // store kept indices 2
    agen_C a_kept_idx_2 = init_agen_C_from_cfg(cfg.st_kept_idx_2_bottom_agen_cfg);

    int niter = cfg.second_pass_niter;

    MERGE_KEPT_LIST_1_WITH_ROW_12_6_BODY
}

void serial_run_sepa_2x2_5x7_exec(const MedianFilterSepa2x2Config &cfg)
{
    // 1st pass: sort horizontal rows
    sort_row_6_exec(cfg);

    // 2nd pass:
    // step 1: sort mutual block
    sort_mutual_block_24_exec(cfg);

    // step 2: merge with top and bottom rows
    merge_with_top_row_12_6_exec(cfg);
    merge_with_bottom_row_12_6_exec(cfg);

    // step 3: sort left and right columns
    sort_left_column_4_exec(cfg);
    sort_right_column_4_exec(cfg);

    // step 4: merge with left and right columns
    // step 5: merge with corners
    merge_with_top_corners_6_4_1_exec(cfg);
    merge_with_bottom_corners_6_4_1_exec(cfg);
}
