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

#include "orb_desc_extr_knl_x8.h"

#include "orb_desc_extr_common.h"

void centroid_calc_x8(ORBDescExtractorConfig *cfg)
{
    agen_B a_in = init_agen_B_from_cfg(cfg->_centroid_calc_in_cfg);
    agen_A a_w  = init_agen_A_from_cfg(cfg->_centroid_calc_weights_cfg);
    agen_A a_mx = init_agen_A_from_cfg(cfg->_centroid_calc_out_mx_cfg);
    agen_A a_my = init_agen_A_from_cfg(cfg->_centroid_calc_out_my_cfg);

    vcharx weight_perm = sign_extend(*(vchar *)cfg->_weight_perm_array);

    dvshortx dh_mx, dh_my;
    dh_mx.lo = replicateh(0);
    dh_mx.hi = replicateh(0);
    dh_my.lo = replicateh(0);
    dh_my.hi = replicateh(0);

    chess_separator_scheduler();

    int niter = max(cfg->_centroid_calc_niter, 24);
    niter     = (niter + 7) / 8 * 8;

    for (int i = 0; i < niter; i++) chess_loop_range(24, )
    chess_unroll_loop(8)
    chess_prepare_for_pipelining
    {
        dvcharx src1, src2;
        src1.lo   = VLOAD_TRANSP(a_in);
        src1.hi   = VLOAD_TRANSP(a_in);
        src2.lo   = VLOAD_TRANSP(a_in);
        src2.hi   = VLOAD_TRANSP(a_in);
        dvcharx w = dvchar_load_perm(a_w, weight_perm);

        vdotp4x2_bbh(src1, src2, w, dh_mx, dh_my, -1, dh_mx, dh_my);
    }

    // promote and interleave before store
    dvintx dw_mx_lo = vpromote(dh_mx.lo);
    dvintx dw_mx_hi = vpromote(dh_mx.hi);

    dvintx dw_mx_st;
    dw_mx_st.lo = dw_mx_lo.lo;
    dw_mx_st.hi = dw_mx_hi.lo;
    vstore_i(dw_mx_st, a_mx);

    dw_mx_st.lo = dw_mx_lo.hi;
    dw_mx_st.hi = dw_mx_hi.hi;
    vstore_i(dw_mx_st, a_mx);

    dvintx dw_my_lo = vpromote(dh_my.lo);
    dvintx dw_my_hi = vpromote(dh_my.hi);

    dvintx dw_my_st;
    dw_my_st.lo = dw_my_lo.lo;
    dw_my_st.hi = dw_my_hi.lo;
    vstore_i(dw_my_st, a_my);

    dw_my_st.lo = dw_my_lo.hi;
    dw_my_st.hi = dw_my_hi.hi;
    vstore_i(dw_my_st, a_my);
}

void serial_run_x8(ORBDescExtractorConfig *cfg, uint8_t *descriptor_buffer)
{

#if defined(KNL_RBRIEF)
    centroid_calc_x8(cfg);
    get_rotation_angle(cfg);
    rotate_orb_patterns(cfg);
#else
    shift_orb_pattern_by_offsets(cfg);
#endif

    // execute 2D lookup
    cupvaSamplerStart(cfg->_sampler);
    cupvaSamplerWait();

    compare_pixels(cfg, descriptor_buffer);
}
