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

#include "blur_filter_roi_knl_u8.h"

void blur_filter_roi_knl_exec(int32_t w, int32_t h, int32_t lp, uint8_t data, uint8_t *dst)
{
    int32_t vecw   = chess_elementsof(vchar);
    int32_t niter1 = (w + vecw - 1) / vecw;
    int32_t niter2 = h;
    int32_t niter  = max(6, niter1 * niter2);
    niter          = (niter + 1) / 2 * 2;

    AgenWrapper wrapper;
    wrapper.size = sizeof(uint8_t);
    wrapper.n1   = niter1;
    wrapper.n2   = niter2;
    wrapper.s1   = vecw;
    wrapper.s2   = lp;

    agen agen_out0 = init(dst);
    INIT_AGEN2(agen_out0, wrapper);

    vcharx vdata = replicateb(data);

    int32_t col_cnt   = 0;
    int32_t pred_all  = -1;
    int32_t remain    = w % vecw;
    int32_t pred_last = remain ? (1 << remain) - 1 : -1;

    chess_separator_scheduler();

    for (int32_t i = 0; i < niter; i++) chess_loop_range(6, )
    chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        col_cnt      = mod_inc(col_cnt, niter1 - 1);
        int32_t pred = mux(col_cnt, pred_all, pred_last);
        vstore(vdata, agen_out0, pred);
    }
}