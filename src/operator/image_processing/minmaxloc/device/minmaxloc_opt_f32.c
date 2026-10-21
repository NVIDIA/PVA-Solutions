/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "minmaxloc_opt_f32.h"

void find_minmax_values_float(MINMAX_VAL_CFG &cfg, float *p_min_val, float *p_max_val, float *p_tmp_dvmin,
                              float *p_tmp_dvmax)
{
    agen a_src = init_agen_from_cfg(cfg._src_agen_cfg);

    dvfloatx dv_min, dv_max;
    dv_min.lo = replicatef(*p_min_val);
    dv_min.hi = replicatef(*p_min_val);
    dv_max.lo = replicatef(*p_max_val);
    dv_max.hi = replicatef(*p_max_val);

    chess_separator_scheduler();

    int niter = cfg._niter_w * cfg._niter_h;
    niter     = max(((niter + 8 - 1) / 8) * 8, 24);
    for (int i = 0; i < niter; i++) chess_loop_range(24, )
    chess_unroll_loop(8)
    chess_prepare_for_pipelining
    {
        dvfloatx dv_data = dvfloat_load(a_src);
        dv_min           = dvmux(dv_data < dv_min, dv_data, dv_min);
        dv_max           = dvmux(dv_data > dv_max, dv_data, dv_max);
    }
    chess_separator_scheduler();
    mem_fence();

    // vector reduction for final results
    unsigned int niter_f32 = pva_elementsof(dvfloatx);
    AgenWrapper w0;
    w0.size          = sizeof(float);
    w0.n1            = 1;
    w0.s1            = niter_f32;
    agen a_store_min = init((dvfloat *)p_tmp_dvmin);
    agen a_store_max = init((dvfloat *)p_tmp_dvmax);
    INIT_AGEN1(a_store_min, w0);
    INIT_AGEN1(a_store_max, w0);
    vstore(dv_min, a_store_min);
    vstore(dv_max, a_store_max);

    chess_separator_scheduler();
    mem_fence();

    AgenWrapper w1;
    w1.size         = sizeof(float);
    w1.n1           = niter_f32 / 2;
    w1.s1           = 2;
    agen a_load_min = init((float *)p_tmp_dvmin);
    agen a_load_max = init((float *)p_tmp_dvmax);
    INIT_AGEN1(a_load_min, w1);
    INIT_AGEN1(a_load_max, w1);

    float final_min = *(float *)p_min_val;
    float final_max = *(float *)p_max_val;
    float min_i1, min_i2;
    float max_i1, max_i2;
    for (unsigned int i = 0; i < niter_f32 / 2; i++)
    {
        float_load(a_load_min, min_i1, min_i2);
        final_min = (min_i1 < final_min) ? min_i1 : final_min;
        final_min = (min_i2 < final_min) ? min_i2 : final_min;

        float_load(a_load_max, max_i1, max_i2);
        final_max = (max_i1 > final_max) ? max_i1 : final_max;
        final_max = (max_i2 > final_max) ? max_i2 : final_max;
    }

    *(float *)p_min_val = final_min;
    *(float *)p_max_val = final_max;
}
