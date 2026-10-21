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

#include "orb_desc_extr_knl_x16.h"

#include "orb_desc_extr_common.h"
#include "orb_desc_extr_params.h"

void centroid_calc_x16(ORBDescExtractorConfig *cfg)
{
    agen_B a_in = init_agen_B_from_cfg(cfg->_centroid_calc_in_cfg);

    agen_A a_w  = init_agen_A_from_cfg(cfg->_centroid_calc_weights_cfg);
    agen_A a_mx = init_agen_A_from_cfg(cfg->_centroid_calc_out_mx_cfg);
    agen_A a_my = init_agen_A_from_cfg(cfg->_centroid_calc_out_my_cfg);

    vshortx weight_perm = sign_extend(*(vshort *)cfg->_weight_perm_array);

    // moments range estimate: pixel_value * weight * pixel_cnt, 16 + 4 + 8 = 28 bits
    dvintx dw_mx, dw_my;
    // initialize mx and my with zeros
    dw_mx.lo = replicatew(0);
    dw_mx.hi = replicatew(0);
    dw_my.lo = replicatew(0);
    dw_my.hi = replicatew(0);

    int niter = max(cfg->_centroid_calc_niter, 12);
    niter     = (niter + 3) / 4 * 4;

    chess_separator_scheduler();

    for (int i = 0; i < niter; i++) chess_loop_range(12, )
    chess_unroll_loop(4)
    chess_prepare_for_pipelining
    {
        // load the data and the corresponding weights and perform multiplication
        // for T2 dvushort transp load, lp ~ 32k+2
        dvshortx src1 = DVLOAD_TRANSP2(a_in);
        dvshortx src2 = DVLOAD_TRANSP2(a_in);
        vcharx wb     = vchar_load_perm(a_w, weight_perm);
        dvshortx wh   = vpromote(wb);

        vdotp4x2_hhw(src1, src2, wh, dw_mx, dw_my, -1, dw_mx, dw_my);
    }

    vstore_i(dw_mx, a_mx);
    vstore_i(dw_my, a_my);
}

void serial_run_x16(ORBDescExtractorConfig *cfg, uint8_t *descriptor_buffer)
{

#if defined(KNL_RBRIEF)
    centroid_calc_x16(cfg);
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
