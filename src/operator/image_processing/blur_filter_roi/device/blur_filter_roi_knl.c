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

#include "blur_filter_roi_knl.h"

void vmemcpy_2d(void *dst, size_t dpitch, void *src, size_t spitch, size_t width, size_t height)
{
    int32_t vecw   = chess_elementsof(vchar);
    int32_t niter1 = (width + vecw - 1) / vecw;
    int32_t niter2 = height;
    int32_t niter  = max(24, niter1 * niter2);

    AgenWrapper wrapper;

    wrapper.size = sizeof(uint8_t);
    wrapper.n1   = niter1;
    wrapper.n2   = niter2;
    wrapper.s1   = vecw;
    wrapper.s2   = spitch;

    agen agen_in = init(src);
    INIT_AGEN2(agen_in, wrapper);

    wrapper.s2 = dpitch;

    agen agen_out = init(dst);
    INIT_AGEN2(agen_out, wrapper);

    int32_t col_cnt   = 0;
    int32_t pred_all  = -1;
    int32_t remain    = width % vecw;
    int32_t pred_last = remain ? (1 << remain) - 1 : -1;

    chess_separator_scheduler();

    for (int32_t i = 0; i < niter; i++) chess_loop_range(24, )
    chess_unroll_loop(8)
    chess_prepare_for_pipelining
    {
        vcharx vdata = vchar_load(agen_in);
        col_cnt      = mod_inc(col_cnt, niter1 - 1);
        int32_t pred = mux(col_cnt, pred_all, pred_last);
        vstore(vdata, agen_out, pred);
    }
}