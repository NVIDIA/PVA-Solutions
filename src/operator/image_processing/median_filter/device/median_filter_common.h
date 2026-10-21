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

#ifndef MEDIAN_FILTER_COMMON_H
#define MEDIAN_FILTER_COMMON_H

#include "median_filter_param.h"

#include <cupva_device.h>

typedef struct MedianFilterConfig
{
    // 1st pass: sort horizontal rows
    AgenCFG _in_agen_cfg;
    AgenCFG _tmp_st_agen_cfg[KNL_W];

    // 2nd pass: merge sorted rows
    AgenCFG _tmp_ld_agen_cfg[KNL_W];
    AgenCFG _out_agen_cfg;

    int _sort_horz_niter;
    int _merge_sorted_rows_niter;
} MedianFilterConfig;

static inline void median_filter_nxm_init(BASE_TYPE *in, BASE_TYPE *out, BASE_TYPE *tmp[KNL_W], int tileW, int tileH,
                                          int inputLp, int outputLp, int cbSizeInBytes, const void *const cbStartAddr,
                                          MedianFilterConfig *cfg)
{

    // 1st pass: sort horizontal rows
    int vecw   = pva_elementsof(KNL_VEC_TYPE_1);
    int niterX = tileW / vecw;
    int niterY = tileH + KNL_H - 1;

    // input: KNL_W elements in each row
    AgenWrapper w0;
    w0.size = sizeof(BASE_TYPE);

#if defined(INTERLEAVED2C)
    w0.n1 = KNL_W;
    w0.s1 = 2;
#else
#    if defined(KNL_3x3)
    // 2 vectors of single-channel input generate 2 single-channel outputs (2 values)
    w0.n1 = 2;
    w0.s1 = 2;
#    else
    w0.n1 = KNL_W;
    w0.s1 = 1;
#    endif
#endif

    w0.n2 = niterX;
    w0.s2 = vecw;
    w0.n3 = niterY;
    w0.s3 = inputLp;

    agen a_in = init((KNL_VEC_TYPE_1 *)in);
    INIT_AGEN3(a_in, w0);
    // setup circular buffer
    a_in              = update_agen_cb_start(a_in, (intptr_t)cbStartAddr);
    a_in              = update_agen_cb_size(a_in, cbSizeInBytes);
    cfg->_in_agen_cfg = extract_agen_cfg(a_in);

    // Intermediate output: KNL_W sorted elements in each row
    int tmpLp = tileW;
    AgenWrapper w1;
    w1.size = sizeof(BASE_TYPE);
    w1.n1   = niterX;
    w1.s1   = vecw;
    w1.n2   = niterY;
    w1.s2   = tmpLp;

    // store sorted elements in each row
    for (int i = 0; i < KNL_W; i++)
    {
        agen a_tmp_st_i = init((KNL_VEC_TYPE_1 *)tmp[i]);
        INIT_AGEN2(a_tmp_st_i, w1);
        cfg->_tmp_st_agen_cfg[i] = extract_agen_cfg(a_tmp_st_i);
    }

    cfg->_sort_horz_niter = niterX * niterY;

    // 2nd pass: merge sorted rows into 1 sorted array
    vecw   = pva_elementsof(KNL_VEC_TYPE_2);
    niterX = tileW / vecw;
    niterY = tileH;

    AgenWrapper w2;
    w2.size = sizeof(BASE_TYPE);
    w2.n1   = KNL_H;
    w2.s1   = tmpLp;
    w2.n2   = niterX;
    w2.s2   = vecw;
    w2.n3   = niterY;
    w2.s3   = tmpLp;

    // load sorted rows in kernel window
    for (int i = 0; i < KNL_W; i++)
    {
        agen a_tmp_ld_i = init((KNL_VEC_TYPE_2 *)tmp[i]);
        INIT_AGEN3(a_tmp_ld_i, w2);
        cfg->_tmp_ld_agen_cfg[i] = extract_agen_cfg(a_tmp_ld_i);
    }

    // output: tileW * tileH elements
    AgenWrapper w3;
    w3.size    = sizeof(BASE_TYPE);
    w3.n1      = niterX;
    w3.s1      = vecw;
    w3.n2      = niterY;
    w3.s2      = outputLp;
    agen a_out = init((KNL_VEC_TYPE_2 *)out);
    INIT_AGEN2(a_out, w3);
    cfg->_out_agen_cfg = extract_agen_cfg(a_out);

    cfg->_merge_sorted_rows_niter = niterX * niterY;
}

#endif // MEDIAN_FILTER_COMMON_H
