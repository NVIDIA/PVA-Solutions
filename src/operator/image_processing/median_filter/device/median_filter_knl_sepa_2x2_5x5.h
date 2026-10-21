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

#ifndef MEDIAN_FILTER_KNL_SEPA_2X2_5X5_H
#define MEDIAN_FILTER_KNL_SEPA_2X2_5X5_H

#include "median_filter_common_sepa_2x2.h"
#include "median_filter_param.h"

#include <cupva_device.h>

// Double-vector 2nd pass: each loop body covers a pair of adjacent x-groups (dv loads/ops on
// value-major paired tmp layouts); pass 1 and the column sorts stay single-vector.
// Requires an even number of x-group iterations per row, which holds at TILE_W = 128.
void serial_run_sepa_2x2_5x5_exec(const MedianFilterSepa2x2Config &cfg);

void sort_mutual_block_16_dv_exec(const MedianFilterSepa2x2Config &cfg);
void merge_with_top_row_10_4_dv_exec(const MedianFilterSepa2x2Config &cfg);
void merge_with_bottom_row_10_4_dv_exec(const MedianFilterSepa2x2Config &cfg);
void merge_with_top_corners_6_4_1_dv_exec(const MedianFilterSepa2x2Config &cfg);
void merge_with_bottom_corners_6_4_1_dv_exec(const MedianFilterSepa2x2Config &cfg);

// Same signature as median_filter_nxm_sepa_2x2_init, with the second-pass agen
// configs describing the double-vector access patterns above.
static inline void median_filter_5x5_sepa_2x2_init(BASE_TYPE *in, BASE_TYPE *out, BASE_TYPE *tmpA, BASE_TYPE *tmpB,
                                                   BASE_TYPE *tmpC, int numElementsA, int numElementsB,
                                                   int numElementsC, int tileW, int tileH, int inputLp, int outputLp,
                                                   int cbSizeInBytes, const void *const cbStartAddr,
                                                   BASE_TYPE *_kept_idx_1, BASE_TYPE *_kept_idx_2_top,
                                                   BASE_TYPE *_kept_idx_2_bottom, BASE_TYPE *_st_sorted_left_col,
                                                   BASE_TYPE *_st_sorted_right_col, MedianFilterSepa2x2Config &cfg)
{
    // 1st pass: sort horizontal rows, with vertical traversal (as nxm init)
    int vecw = pva_elementsof(VECX_TYPE);

    // NOTE: use dvload_di to get elements in stride of 2
    int niterX = tileW / (2 * vecw);
    int niterY = tileH + KNL_H - 1;

    AgenWrapper w0;
    w0.size = sizeof(BASE_TYPE);
    w0.n1   = (KNL_W + 1) / 2;
    w0.s1   = 2;
    w0.n2   = niterX;
    w0.s2   = 2 * vecw;
    w0.n3   = niterY;
    w0.s3   = inputLp;

    agen a_in = init((DVECX_TYPE *)in);
    INIT_AGEN3(a_in, w0);
    a_in            = update_agen_cb_start(a_in, (intptr_t)cbStartAddr);
    a_in            = update_agen_cb_size(a_in, cbSizeInBytes);
    cfg.in_agen_cfg = extract_agen_cfg(a_in);

    // store intermediate outputs
    int tmpLp = niterX * vecw;

    int buf_size = INTERM_TILE_SIZE;

    AgenWrapper w1;
    w1.size = sizeof(BASE_TYPE);
    w1.n1   = numElementsA;
    w1.s1   = buf_size;
    w1.n2   = niterX;
    w1.s2   = vecw;
    w1.n3   = niterY;
    w1.s3   = tmpLp;

    agen a_tmp_st_A = init((VECX_TYPE *)tmpA);
    INIT_AGEN3(a_tmp_st_A, w1);
    cfg.tmp_st_agen_cfg_A = extract_agen_cfg(a_tmp_st_A);

    w1.n1           = numElementsB;
    agen a_tmp_st_B = init((VECX_TYPE *)tmpB);
    INIT_AGEN3(a_tmp_st_B, w1);
    cfg.tmp_st_agen_cfg_B = extract_agen_cfg(a_tmp_st_B);

    w1.n1           = numElementsC;
    agen a_tmp_st_C = init((VECX_TYPE *)tmpC);
    INIT_AGEN3(a_tmp_st_C, w1);
    cfg.tmp_st_agen_cfg_C = extract_agen_cfg(a_tmp_st_C);

    cfg.first_pass_niter = niterX * niterY;

    // 2nd pass: double-vector, one body per pair of adjacent x-groups

    niterX     = tileW / (2 * vecw);
    niterY     = tileH / 2;
    int nPairX = niterX / 2;

    // step 1: sort the mutual block (dv loads span an x-group pair)
    AgenWrapper w_step_1;
    w_step_1.size = sizeof(BASE_TYPE);

    w_step_1.n1 = numElementsA;
    w_step_1.s1 = buf_size;
    w_step_1.n2 = KNL_H - 1;
    w_step_1.s2 = tmpLp;
    w_step_1.n3 = nPairX;
    w_step_1.s3 = 2 * vecw;
    w_step_1.n4 = niterY;
    w_step_1.s4 = 2 * tmpLp;

    // skip the first row
    agen a_ld_mutual_block_A = init((DVECX_TYPE *)(tmpA + tmpLp));
    INIT_AGEN4(a_ld_mutual_block_A, w_step_1);
    cfg.ld_mutual_block_agen_cfg_A = extract_agen_cfg(a_ld_mutual_block_A);

    w_step_1.n1              = numElementsB;
    agen a_ld_mutual_block_B = init((DVECX_TYPE *)(tmpB + tmpLp));
    INIT_AGEN4(a_ld_mutual_block_B, w_step_1);
    cfg.ld_mutual_block_agen_cfg_B = extract_agen_cfg(a_ld_mutual_block_B);

    w_step_1.n1              = numElementsC;
    agen a_ld_mutual_block_C = init((DVECX_TYPE *)(tmpC + tmpLp));
    INIT_AGEN4(a_ld_mutual_block_C, w_step_1);
    cfg.ld_mutual_block_agen_cfg_C = extract_agen_cfg(a_ld_mutual_block_C);

    // kept indices 1: value-major x-group pairs, one dv per value
    AgenWrapper w_kept_idx_1;
    w_kept_idx_1.size = sizeof(BASE_TYPE);
    w_kept_idx_1.n1   = KEEP_IDX_1_SIZE * nPairX * niterY;
    w_kept_idx_1.s1   = 2 * vecw;

    agen a_st_kept_idx_1 = init((DVECX_TYPE *)_kept_idx_1);
    INIT_AGEN1(a_st_kept_idx_1, w_kept_idx_1);
    cfg.st_kept_idx_1_agen_cfg = extract_agen_cfg(a_st_kept_idx_1);

    agen a_ld_kept_idx_1 = init((DVECX_TYPE *)_kept_idx_1);
    INIT_AGEN1(a_ld_kept_idx_1, w_kept_idx_1);
    cfg.ld_kept_idx_1_agen_cfg = extract_agen_cfg(a_ld_kept_idx_1);

    // step 2: merge with the upper and lower rows (dv row loads)
    AgenWrapper w3;
    w3.size = sizeof(BASE_TYPE);
    w3.n1   = numElementsA;
    w3.s1   = buf_size;
    w3.n2   = nPairX;
    w3.s2   = 2 * vecw;
    w3.n3   = niterY;
    w3.s3   = 2 * tmpLp;

    agen a_ld_first_row_A = init((DVECX_TYPE *)(tmpA));
    INIT_AGEN3(a_ld_first_row_A, w3);
    cfg.ld_first_row_agen_cfg_A = extract_agen_cfg(a_ld_first_row_A);

    agen a_ld_last_row_A = init((DVECX_TYPE *)(tmpA + KNL_H * tmpLp));
    INIT_AGEN3(a_ld_last_row_A, w3);
    cfg.ld_last_row_agen_cfg_A = extract_agen_cfg(a_ld_last_row_A);

    w3.n1                 = numElementsB;
    agen a_ld_first_row_B = init((DVECX_TYPE *)(tmpB));
    INIT_AGEN3(a_ld_first_row_B, w3);
    cfg.ld_first_row_agen_cfg_B = extract_agen_cfg(a_ld_first_row_B);

    agen a_ld_last_row_B = init((DVECX_TYPE *)(tmpB + KNL_H * tmpLp));
    INIT_AGEN3(a_ld_last_row_B, w3);
    cfg.ld_last_row_agen_cfg_B = extract_agen_cfg(a_ld_last_row_B);

    w3.n1                 = numElementsC;
    agen a_ld_first_row_C = init((DVECX_TYPE *)(tmpC));
    INIT_AGEN3(a_ld_first_row_C, w3);
    cfg.ld_first_row_agen_cfg_C = extract_agen_cfg(a_ld_first_row_C);

    agen a_ld_last_row_C = init((DVECX_TYPE *)(tmpC + KNL_H * tmpLp));
    INIT_AGEN3(a_ld_last_row_C, w3);
    cfg.ld_last_row_agen_cfg_C = extract_agen_cfg(a_ld_last_row_C);

    // kept indices 2: value-major x-group pairs, one dv per value
    AgenWrapper w_kept_idx_2;
    w_kept_idx_2.size = sizeof(BASE_TYPE);
    w_kept_idx_2.n1   = KEEP_IDX_2_SIZE * nPairX * niterY;
    w_kept_idx_2.s1   = 2 * vecw;

    agen a_st_kept_idx_2_top = init((DVECX_TYPE *)_kept_idx_2_top);
    INIT_AGEN1(a_st_kept_idx_2_top, w_kept_idx_2);
    cfg.st_kept_idx_2_top_agen_cfg = extract_agen_cfg(a_st_kept_idx_2_top);

    agen a_st_kept_idx_2_bottom = init((DVECX_TYPE *)_kept_idx_2_bottom);
    INIT_AGEN1(a_st_kept_idx_2_bottom, w_kept_idx_2);
    cfg.st_kept_idx_2_bottom_agen_cfg = extract_agen_cfg(a_st_kept_idx_2_bottom);

    agen a_ld_kept_idx_2_top = init((DVECX_TYPE *)_kept_idx_2_top);
    INIT_AGEN1(a_ld_kept_idx_2_top, w_kept_idx_2);
    cfg.ld_kept_idx_2_top_agen_cfg = extract_agen_cfg(a_ld_kept_idx_2_top);

    agen a_ld_kept_idx_2_bottom = init((DVECX_TYPE *)_kept_idx_2_bottom);
    INIT_AGEN1(a_ld_kept_idx_2_bottom, w_kept_idx_2);
    cfg.ld_kept_idx_2_bottom_agen_cfg = extract_agen_cfg(a_ld_kept_idx_2_bottom);

    // step 3: sort the left and right columns (unchanged single-vector
    // kernels; loads identical to the nxm init)
    AgenWrapper w4;
    w4.size = sizeof(BASE_TYPE);
    w4.n1   = KNL_H - 1;
    w4.s1   = inputLp;
    w4.n2   = niterX;
    w4.s2   = 2 * vecw;
    w4.n3   = niterY;
    w4.s3   = 2 * inputLp;

    // Start from the second row and select the low half in the kernel.
    agen a_ld_left_col_sort = init((DVECX_TYPE *)(in + inputLp));
    INIT_AGEN3(a_ld_left_col_sort, w4);
    a_ld_left_col_sort            = update_agen_cb_start(a_ld_left_col_sort, (intptr_t)cbStartAddr);
    a_ld_left_col_sort            = update_agen_cb_size(a_ld_left_col_sort, cbSizeInBytes);
    cfg.ld_left_col_sort_agen_cfg = extract_agen_cfg(a_ld_left_col_sort);

    // Start from the second row on the preceding even column and select the high half in the kernel.
    agen a_ld_right_col_sort = init((DVECX_TYPE *)(in + inputLp + KNL_W - 1));
    INIT_AGEN3(a_ld_right_col_sort, w4);
    a_ld_right_col_sort            = update_agen_cb_start(a_ld_right_col_sort, (intptr_t)cbStartAddr);
    a_ld_right_col_sort            = update_agen_cb_size(a_ld_right_col_sort, cbSizeInBytes);
    cfg.ld_right_col_sort_agen_cfg = extract_agen_cfg(a_ld_right_col_sort);

    // sorted-column stores re-striped so {sl_k(xgA) | sl_k(xgB)} are adjacent
    // (store order in the kernel: k inner, then x-group, then row)
    AgenWrapper w_sorted_col_st;
    w_sorted_col_st.size = sizeof(BASE_TYPE);
    w_sorted_col_st.n1   = KNL_H - 1;
    w_sorted_col_st.s1   = 2 * vecw;
    w_sorted_col_st.n2   = 2;
    w_sorted_col_st.s2   = vecw;
    w_sorted_col_st.n3   = nPairX;
    w_sorted_col_st.s3   = (KNL_H - 1) * 2 * vecw;
    w_sorted_col_st.n4   = niterY;
    w_sorted_col_st.s4   = nPairX * (KNL_H - 1) * 2 * vecw;

    agen a_st_sorted_left_col = init((VECX_TYPE *)_st_sorted_left_col);
    INIT_AGEN4(a_st_sorted_left_col, w_sorted_col_st);
    cfg.st_sorted_left_col_agen_cfg = extract_agen_cfg(a_st_sorted_left_col);

    agen a_st_sorted_right_col = init((VECX_TYPE *)_st_sorted_right_col);
    INIT_AGEN4(a_st_sorted_right_col, w_sorted_col_st);
    cfg.st_sorted_right_col_agen_cfg = extract_agen_cfg(a_st_sorted_right_col);

    // load sorted columns: one dv per value, paired across the x-group pair
    AgenWrapper w_sorted_col_ld;
    w_sorted_col_ld.size = sizeof(BASE_TYPE);
    w_sorted_col_ld.n1   = (KNL_H - 1) * nPairX * niterY;
    w_sorted_col_ld.s1   = 2 * vecw;

    agen a_ld_sorted_left_col = init((DVECX_TYPE *)_st_sorted_left_col);
    INIT_AGEN1(a_ld_sorted_left_col, w_sorted_col_ld);
    cfg.ld_sorted_left_col_agen_cfg = extract_agen_cfg(a_ld_sorted_left_col);

    agen a_ld_sorted_right_col = init((DVECX_TYPE *)_st_sorted_right_col);
    INIT_AGEN1(a_ld_sorted_right_col, w_sorted_col_ld);
    cfg.ld_sorted_right_col_agen_cfg = extract_agen_cfg(a_ld_sorted_right_col);

    // load the 4 corners (as nxm init; the dv kernel advances n1 twice per body)
    AgenWrapper w_ld_corners;
    w_ld_corners.size = sizeof(BASE_TYPE);
    w_ld_corners.n1   = niterX;
    w_ld_corners.s1   = 2 * vecw;
    w_ld_corners.n2   = niterY;
    w_ld_corners.s2   = 2 * inputLp;

    agen a_ld_top_left_corner = init((DVECX_TYPE *)in);
    INIT_AGEN2(a_ld_top_left_corner, w_ld_corners);
    a_ld_top_left_corner            = update_agen_cb_start(a_ld_top_left_corner, (intptr_t)cbStartAddr);
    a_ld_top_left_corner            = update_agen_cb_size(a_ld_top_left_corner, cbSizeInBytes);
    cfg.ld_top_left_corner_agen_cfg = extract_agen_cfg(a_ld_top_left_corner);

    agen a_ld_top_right_corner = init((DVECX_TYPE *)(in + KNL_W - 1));
    INIT_AGEN2(a_ld_top_right_corner, w_ld_corners);
    a_ld_top_right_corner            = update_agen_cb_start(a_ld_top_right_corner, (intptr_t)cbStartAddr);
    a_ld_top_right_corner            = update_agen_cb_size(a_ld_top_right_corner, cbSizeInBytes);
    cfg.ld_top_right_corner_agen_cfg = extract_agen_cfg(a_ld_top_right_corner);

    agen a_ld_bottom_left_corner = init((DVECX_TYPE *)(in + KNL_H * inputLp));
    INIT_AGEN2(a_ld_bottom_left_corner, w_ld_corners);
    a_ld_bottom_left_corner            = update_agen_cb_start(a_ld_bottom_left_corner, (intptr_t)cbStartAddr);
    a_ld_bottom_left_corner            = update_agen_cb_size(a_ld_bottom_left_corner, cbSizeInBytes);
    cfg.ld_bottom_left_corner_agen_cfg = extract_agen_cfg(a_ld_bottom_left_corner);

    agen a_ld_bottom_right_corner = init((DVECX_TYPE *)(in + KNL_H * inputLp + KNL_W - 1));
    INIT_AGEN2(a_ld_bottom_right_corner, w_ld_corners);
    a_ld_bottom_right_corner            = update_agen_cb_start(a_ld_bottom_right_corner, (intptr_t)cbStartAddr);
    a_ld_bottom_right_corner            = update_agen_cb_size(a_ld_bottom_right_corner, cbSizeInBytes);
    cfg.ld_bottom_right_corner_agen_cfg = extract_agen_cfg(a_ld_bottom_right_corner);

    // interleave results before store (as nxm init; two stores per dv body)
    AgenWrapper w_out;
    w_out.size = sizeof(BASE_TYPE);
    w_out.n1   = niterX;
    w_out.s1   = 2 * vecw;
    w_out.n2   = niterY;
    w_out.s2   = 2 * outputLp;

    agen a_out_top = init((DVECX_TYPE *)out);
    INIT_AGEN2(a_out_top, w_out);
    cfg.out_agen_cfg_top = extract_agen_cfg(a_out_top);

    agen a_out_bottom = init((DVECX_TYPE *)(out + outputLp));
    INIT_AGEN2(a_out_bottom, w_out);
    cfg.out_agen_cfg_bottom = extract_agen_cfg(a_out_bottom);

    // column sorts iterate per x-group; dv kernels derive their pair count
    // as second_pass_niter / 2
    cfg.second_pass_niter = niterX * niterY;
}

#endif /* MEDIAN_FILTER_KNL_SEPA_2X2_5X5_H */
