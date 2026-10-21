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

#include "floyd_warshall_knl.h"

#include <cupva_device.h>

void floyd_warshall_knl_init(AgenCFG *cfgs, int32_t num)
{
    const int dvecw  = 32;
    const int niter1 = (num + dvecw - 1) / dvecw; // for loop j
    const int niter2 = num;                       // for loop i

    AgenWrapper w0; // for read len_ik and next_ik
    w0.size = sizeof(short);
    w0.n1   = niter1;
    w0.s1   = 0;
    w0.n2   = niter2;
    w0.s2   = num;

    AgenWrapper w1; // for read len_kj
    w1.size = sizeof(short);
    w1.n1   = niter1;
    w1.s1   = dvecw;
    w1.n2   = niter2;
    w1.s2   = 0;

    AgenWrapper w2; // for write len_ij and next_ij
    w2.size = sizeof(short);
    w2.n1   = niter1;
    w2.s1   = dvecw;
    w2.n2   = niter2;
    w2.s2   = num;

    agen_A in_len_ik     = init_A(NULL);
    agen_A in_len_kj     = init_A(NULL);
    agen_B in_len_cp_ij  = init_B(NULL);
    agen_A out_len_ij    = init_A(NULL);
    agen_B out_len_cp_ij = init_B(NULL);
    agen_C in_next_ik    = init_C(NULL);
    agen_C out_next_ij   = init_C(NULL);

    INIT_AGEN2(in_len_ik, w0);
    INIT_AGEN2(in_len_kj, w1);
    INIT_AGEN2(in_len_cp_ij, w2);
    INIT_AGEN2(out_len_ij, w2);
    INIT_AGEN2(out_len_cp_ij, w2);
    INIT_AGEN2(in_next_ik, w0);
    INIT_AGEN2(out_next_ij, w2);

    cfgs[IN_LEN_IK]     = extract_agen_A_cfg(in_len_ik);
    cfgs[IN_LEN_KJ]     = extract_agen_A_cfg(in_len_kj);
    cfgs[IN_LEN_CP_IJ]  = extract_agen_B_cfg(in_len_cp_ij);
    cfgs[OUT_LEN_IJ]    = extract_agen_A_cfg(out_len_ij);
    cfgs[OUT_LEN_CP_IJ] = extract_agen_B_cfg(out_len_cp_ij);
    cfgs[IN_NEXT_IK]    = extract_agen_C_cfg(in_next_ik);
    cfgs[OUT_NEXT_IJ]   = extract_agen_C_cfg(out_next_ij);
}

void floyd_warshall_knl_exec(AgenCFG *cfgs, int32_t num)
{
    int dvecw     = 32;
    int niter1    = (num + dvecw - 1) / dvecw;
    int lane_mask = (num % dvecw) ? (int)((1u << (num % dvecw)) - 1) : -1;

    agen_A in_len_ik     = init_agen_A_from_cfg(cfgs[IN_LEN_IK]);
    agen_A in_len_kj     = init_agen_A_from_cfg(cfgs[IN_LEN_KJ]);
    agen_B in_len_cp_ij  = init_agen_B_from_cfg(cfgs[IN_LEN_CP_IJ]);
    agen_A out_len_ij    = init_agen_A_from_cfg(cfgs[OUT_LEN_IJ]);
    agen_B out_len_cp_ij = init_agen_B_from_cfg(cfgs[OUT_LEN_CP_IJ]);
    agen_C in_next_ik    = init_agen_C_from_cfg(cfgs[IN_NEXT_IK]);
    agen_C out_next_ij   = init_agen_C_from_cfg(cfgs[OUT_NEXT_IJ]);

    int niter = niter1 * num;
    niter     = (niter > 18) ? niter : 18;
    niter     = (niter + 5) / 6 * 6;

    int count = 0;
    unsigned short len_ik;
    dvshortx len_kj, len_ij, next_ik;
    vcharx zero_perm = replicateb(0);

    for (int iter = 0; iter < niter; iter++) chess_loop_range(18, )
    chess_unroll_loop(6)
    chess_prepare_for_pipelining
    {
        len_ik  = ushort_load(in_len_ik);
        len_kj  = dvushort_load(in_len_kj);
        len_ij  = dvushort_load(in_len_cp_ij);
        next_ik = dvushort_load_perm(in_next_ik, zero_perm);

        dvshortx new_len = len_kj + len_ik;
        dvshortx pred_ij = new_len < len_ij;
        vcharx pred      = vdemote(pred_ij);
        int pred_store   = vbitcmp_s(pred, 1); // vector compare and bit-pack

        // at the last iteration of each row, mask off high bits to prevent writing garbage data
        count      = mod_inc(count, niter1 - 1);
        pred_store = mux(count, pred_store, pred_store & lane_mask);

        vstore(new_len, out_len_ij, pred_store);
        vstore(new_len, out_len_cp_ij, pred_store);
        vstore(next_ik, out_next_ij, pred_store);
    }
}