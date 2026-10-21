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

#include "mark_peak_candidates.h"

#include "capon_roi_part2_config.h"

#include <cupva_device.h>

void init_mark_peak_candidates(const uint16_t *const restrict fixed_pt_spectrum, uint16_t *const restrict candidate,
                               Para_capon_roi_part2_config &para)
{
    {
        AgenWrapper w;
        w.size = sizeof(dvushort);
        w.n1   = (NUM_AZ_BINS - 2) / 2;
        w.s1   = 1;
        w.n2   = NUM_EL_BINS - 2;
        w.s2   = NUM_AZ_BINS / 2;
        agen a = init((uint16_t *)fixed_pt_spectrum + (NUM_AZ_BINS + 1) * chess_elementsof(vushort));
        INIT_AGEN2(a, w);
        para.cfg_cen = extract_agen_cfg(a);

        w.s2 = (NUM_AZ_BINS - 2) / 2;
        a    = init(candidate);
        INIT_AGEN2(a, w);
        para.cand_cfg_candidate = extract_agen_cfg(a);
    }
}

void mark_peak_candidates(const Para_capon_roi_part2_config &para)
{
    agen_B a_cen          = init_agen_B_from_cfg(para.cfg_cen);
    agen_A a_local_max    = init_agen_A_from_cfg(para.cfg_local_max_stage1);
    agen_C a_local_sum_d8 = init_agen_C_from_cfg(para.cfg_local_sum_d8_stage1);
    agen_A a_candidate    = init_agen_A_from_cfg(para.cand_cfg_candidate);

    const int32_t k_nb   = CAPON_PART2_CANDIDATE_K;
    const uint32_t niter = CAPON_PART2_CANDIDATE_NITER;

    for (uint32_t i = 0; i < niter; ++i) chess_loop_range(12, )
    chess_unroll_loop(4)
    chess_prepare_for_pipelining
    {
        const dvshortx cen            = dvushort_load(a_cen);
        const dvshortx isnt_local_max = dvushort_load(a_local_max) != cen;
        const dvshortx th             = dvmulh(dvushort_load(a_local_sum_d8), k_nb, VPU_ROUND_7);
        dvshortx cond                 = chess_dont_care(dvshortx);
        cond.lo                       = cen.lo <= th.lo || isnt_local_max.lo;
        cond.hi                       = cen.hi <= th.hi || isnt_local_max.hi;
        vstore(dvmux(cond, 0, cen), a_candidate);
    }
}
