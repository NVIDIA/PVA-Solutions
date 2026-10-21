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

#include "minmaxloc_opt_u16.h"

void find_minmax_values_ushort(MINMAX_VAL_CFG &cfg, unsigned short *p_min_val, unsigned short *p_max_val)
{
    agen a_src = init_agen_from_cfg(cfg._src_agen_cfg);

    dvshortx dv_min, dv_max;
    dv_min.lo = replicateh(*p_min_val);
    dv_min.hi = replicateh(*p_min_val);
    dv_max.lo = replicateh(*p_max_val);
    dv_max.hi = replicateh(*p_max_val);

    chess_separator_scheduler();

    int niter = cfg._niter_w * cfg._niter_h;
    niter     = max(((niter + 7) / 8) * 8, 24);
    for (int i = 0; i < niter; i++) chess_loop_range(24, )
    chess_unroll_loop(8)
    chess_prepare_for_pipelining
    {
        dvshortx dv_data = dvushort_load(a_src);
        dv_min           = dvmin(dv_data, dv_min);
        dv_max           = dvmax(dv_data, dv_max);
    }

    unsigned short min_val_lo = vminr_s(dv_min.lo);
    unsigned short min_val_hi = vminr_s(dv_min.hi);
    *p_min_val                = (min_val_lo > min_val_hi) ? min_val_hi : min_val_lo;

    unsigned short max_val_lo = vmaxr_s(dv_max.lo);
    unsigned short max_val_hi = vmaxr_s(dv_max.hi);
    *p_max_val                = (max_val_lo > max_val_hi) ? max_val_lo : max_val_hi;
}
