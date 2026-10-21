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

#include "obtain_fixed_pt_steering_matrix.h"

#include "capon_roi_part2_config.h"

#include <cupva_device.h>

/*
Steering vectors are split into two channels by groups of 8:
   sv_mat_0: even groups (indices 0-7, 16-23, ...)
   sv_mat_1: odd groups  (indices 8-15, 24-31, ...)
*/

void init_shift_fixed_pt_steering_matrix(const uint16_t steering_qbits, Para_capon_roi_part2_config &para)
{
    // Required internal steering format is Q22. shift_bits is the left-shift
    // that takes the incoming Q(steering_qbits) steering to Q22. When the input
    // is already Q22 the shift is the identity, so capon_roi_top skips the
    // kernel entirely. The configs are still built because their bases are
    // unconditionally patched together with the downstream steering configs.
    para.shift_bits = 22 - steering_qbits;

    {
        AgenWrapper w;
        w.size       = sizeof(dvint);
        w.n1         = CAPON_PART2_STEER_BASE_NITER;
        w.s1         = 1;
        agen a       = init((int32_t *)0); // patched before first use
        a.sat_opt    = 2;                  // treat 32-bit comparison values as signed
        a.sat_lim_lo = -(1 << 30);
        a.sat_val_lo = -(1 << 30);
        a.sat_lim_hi = 1 << 30;
        a.sat_val_hi = 1 << 30;
        INIT_AGEN1(a, w);
        para.cfg_sv_mat_0 = extract_agen_cfg(a);
        a.a               = (int32_t *)0; // patched before first use
        para.cfg_sv_mat_1 = extract_agen_cfg(a);
    }
}

void shift_fixed_pt_steering_matrix(const Para_capon_roi_part2_config &para)
{
    agen_B a_sv_mat_0 = init_agen_B_from_cfg(para.cfg_sv_mat_0);
    agen_C a_sv_mat_1 = init_agen_C_from_cfg(para.cfg_sv_mat_1);

    agen_B a_out0 = a_sv_mat_0; // in-place
    agen_C a_out1 = a_sv_mat_1; // in-place

    const uint32_t niter = CAPON_PART2_STEER_NITER;

    for (uint32_t i = 0; i < niter; ++i) chess_loop_range(18, )
#if CUPVA_PVA_GEN_NUMBER < 3
    chess_unroll_loop(2) // compute-bounded
#else
    chess_unroll_loop(6) // memory-slot-issue-bounded
#endif
    chess_prepare_for_pipelining
    {
        const dvintx v0 = dvint_load(a_sv_mat_0);
        const dvintx v1 = dvint_load(a_sv_mat_1);

        // double vector pseudo intrinsics
        dvintx n0 = dvshift(v0, para.shift_bits);
        dvintx n1 = dvshift(v1, para.shift_bits);

        // lane_ofst = 0 is used intentionally to avoid unnecessary memory stalls
        vstore_transp(n0, a_out0);
        vstore_transp(n1, a_out1);
    }
}
