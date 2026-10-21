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

#include "extract_3x3filtered_spectrum_in16az8el_out14az6el.h"

#include "capon_roi_part2_config.h"

#include <cupva_device.h>

/* ============================================================================
 * Data flow: extract_3x3filtered_spectrum_in16az8el_out14az6el
 *
 *   fixed_pt_spectrum input (from obtain_spectrum output)
 *   uint16_t[n_roi_groups][8][16][chess_elementsof(vushort)]
 *
 *   Per roi_group:
 *
 *        az0  az1  az2  ...  az15          16 ROIs contiguous
 *   el0 [ . ][ . ][ . ] ... [ . ]         per point (each [ . ] = 16 uint16)
 *   el1 [ . ][ . ][ . ] ... [ . ]
 *   ...
 *   el7 [ . ][ . ][ . ] ... [ . ]
 *         8 (vertical) x 16 (horizontal) x 16 ROIs
 *
 *   Stage 0: vertical sliding window of 3 along elevation (height 8 -> 6)
 *            removes top (el0) and bottom (el7) edges
 *
 *        az0  az1  az2  ...  az15
 *   el1 [ . ][ . ][ . ] ... [ . ]
 *   el2 [ . ][ . ][ . ] ... [ . ]
 *   ...
 *   el6 [ . ][ . ][ . ] ... [ . ]
 *         6 (vertical) x 16 (horizontal) x 16 ROIs
 *
 *   Stage 1: horizontal sliding window of 3 along azimuth (width 16 -> 14)
 *            removes left (az0) and right (az15) edges
 *
 *         az1  az2  ...  az14
 *   el1 [ . ][ . ] ... [ . ]
 *   el2 [ . ][ . ] ... [ . ]
 *   ...
 *   el6 [ . ][ . ] ... [ . ]
 *         6 (vertical) x 14 (horizontal) x 16 ROIs
 *
 *   Intermediate buffer: uint16_t[16 * 6 * roi_num], per roi_group 48 dvushort:
 *     dvushort index = el_out * 8 + az_pair
 *     pos  0: (el_out0, az0-1)  ...  pos  7: (el_out0, az14-15)
 *     pos  8: (el_out1, az0-1)  ...  pos 15: (el_out1, az14-15)
 *     ...
 *     pos 40: (el_out5, az0-1)  ...  pos 47: (el_out5, az14-15)
 *
 *   Stage 0 writes with stride-8 (one az pair column across 6 el_out rows).
 *   Stage 1 reads one row (8 dvushort) sequentially, writes 7 dvushort tightly
 *   from the lowest position, overwriting stage 0 data in-place:
 *     iter k: reads pos [8k, 8k+7], writes pos [7k, 7k+6]
 *     total 42 dvushort (pos 0-41), pos 42-47 are don't-care.
 */
void init_extract_3x3filtered_spectrum_in16az8el_out14az6el(const uint16_t *const restrict fixed_pt_spectrum,
                                                            uint16_t *const restrict local_max,
                                                            uint16_t *const restrict local_sum_d8,
                                                            Para_capon_roi_part2_config &para)
{
    { // Stage0 input/output
        AgenWrapper w;
        w.size = sizeof(dvushort);
        w.n1   = NUM_EL_BINS;
        w.s1   = NUM_AZ_BINS / 2;
        w.n2   = NUM_AZ_BINS / 2;
        w.s2   = 1;
        agen a = init((uint16_t *)fixed_pt_spectrum);
        INIT_AGEN2(a, w);
        para.cfg_spectrum_stage0 = extract_agen_cfg(a);

        w.n1 = NUM_EL_BINS - 2;
        a    = init(local_max);
        INIT_AGEN2(a, w);
        para.cfg_v_max_out_stage0    = extract_agen_cfg(a);
        a.a                          = local_sum_d8;
        a.round                      = 1; // (sum + 1) / 2
        para.cfg_v_sum_d2_out_stage0 = extract_agen_cfg(a);
    }
    { // Stage1 input
        AgenWrapper w;
        w.size = sizeof(dvushort);
        w.n1   = NUM_AZ_BINS / 2;
        w.s1   = 1;
        w.n2   = NUM_EL_BINS - 2;
        w.s2   = NUM_AZ_BINS / 2;
        agen a = init(local_max);
        INIT_AGEN2(a, w);
        para.cfg_v_max_in_stage1    = extract_agen_cfg(a);
        a.a                         = local_sum_d8;
        para.cfg_v_sum_d2_in_stage1 = extract_agen_cfg(a);
    }
    { // Stage 1 output
        AgenWrapper w;
        w.size = sizeof(dvushort);
        w.n1   = NUM_AZ_BINS / 2 - 1;
        w.s1   = 1;
        w.n2   = NUM_EL_BINS - 2;
        w.s2   = NUM_AZ_BINS / 2 - 1;
        agen a = init(local_max);
        INIT_AGEN2(a, w);
        para.cfg_local_max_stage1    = extract_agen_cfg(a);
        a.a                          = local_sum_d8;
        a.round                      = 2; // (sum + 2) / 4
        a.sat_opt                    = 2; // treat 32-bit comparison values as signed
        a.sat_lim_lo                 = 0;
        a.sat_val_lo                 = 0;
        a.sat_lim_hi                 = (1 << 15) * 3 * 3 / 8;
        a.sat_val_hi                 = (1 << 15) * 3 * 3 / 8;
        para.cfg_local_sum_d8_stage1 = extract_agen_cfg(a);
    }
}

void extract_3x3filtered_spectrum_in16az8el_out14az6el_stage0(const Para_capon_roi_part2_config &para)
{
    agen_B a_spectrum     = init_agen_B_from_cfg(para.cfg_spectrum_stage0);
    agen_A a_v_max_out    = init_agen_A_from_cfg(para.cfg_v_max_out_stage0);    // reuse output buffer
    agen_C a_v_sum_d2_out = init_agen_C_from_cfg(para.cfg_v_sum_d2_out_stage0); // reuse output buffer

    const uint32_t niter = CAPON_PART2_FILTER_STAGE0_NITER;

    for (uint32_t i = 0; i < niter; ++i) chess_loop_range(3, )
    chess_prepare_for_pipelining
    {
        const dvshortx chess_storage(DV) p0 = dvushort_load(a_spectrum);
        const dvshortx chess_storage(DV) p1 = dvushort_load(a_spectrum);
        const dvshortx chess_storage(DV) p2 = dvushort_load(a_spectrum);
        const dvshortx chess_storage(DV) p3 = dvushort_load(a_spectrum);
        const dvshortx chess_storage(DV) p4 = dvushort_load(a_spectrum);
        const dvshortx chess_storage(DV) p5 = dvushort_load(a_spectrum);
        const dvshortx chess_storage(DV) p6 = dvushort_load(a_spectrum);
        const dvshortx chess_storage(DV) p7 = dvushort_load(a_spectrum);

        vstore(dvmax3(p0, p1, p2), a_v_max_out);
        vstore(dvadd3(p0, p1, p2), a_v_sum_d2_out);
        vstore(dvmax3(p1, p2, p3), a_v_max_out);
        vstore(dvadd3(p1, p2, p3), a_v_sum_d2_out);
        vstore(dvmax3(p2, p3, p4), a_v_max_out);
        vstore(dvadd3(p2, p3, p4), a_v_sum_d2_out);
        vstore(dvmax3(p3, p4, p5), a_v_max_out);
        vstore(dvadd3(p3, p4, p5), a_v_sum_d2_out);
        vstore(dvmax3(p4, p5, p6), a_v_max_out);
        vstore(dvadd3(p4, p5, p6), a_v_sum_d2_out);
        vstore(dvmax3(p5, p6, p7), a_v_max_out);
        vstore(dvadd3(p5, p6, p7), a_v_sum_d2_out);
    }
}

void extract_3x3filtered_spectrum_in16az8el_out14az6el_stage1(const Para_capon_roi_part2_config &para)
{
    agen_A a_v_max_in     = init_agen_A_from_cfg(para.cfg_v_max_in_stage1);     // reuse output buffer
    agen_C a_v_sum_d2_in  = init_agen_C_from_cfg(para.cfg_v_sum_d2_in_stage1);  // reuse output buffer
    agen_A a_local_max    = init_agen_A_from_cfg(para.cfg_local_max_stage1);    // output buffer
    agen_C a_local_sum_d8 = init_agen_C_from_cfg(para.cfg_local_sum_d8_stage1); // output buffer

    const uint32_t niter = CAPON_PART2_FILTER_STAGE1_NITER;

    for (uint32_t i = 0; i < niter; ++i) chess_loop_range(3, )
    chess_prepare_for_pipelining
    {
        const dvshortx chess_storage(DV) p0 = dvushort_load(a_v_max_in);
        const dvshortx chess_storage(DW) p1 = dvushort_load(a_v_max_in);
        const dvshortx chess_storage(DV) p2 = dvushort_load(a_v_max_in);
        const dvshortx chess_storage(DW) p3 = dvushort_load(a_v_max_in);
        const dvshortx chess_storage(DV) p4 = dvushort_load(a_v_max_in);
        const dvshortx chess_storage(DW) p5 = dvushort_load(a_v_max_in);
        const dvshortx chess_storage(DV) p6 = dvushort_load(a_v_max_in);
        const dvshortx chess_storage(DW) p7 = dvushort_load(a_v_max_in);

        const dvshortx chess_storage(DV) q0 = dvushort_load(a_v_sum_d2_in);
        const dvshortx chess_storage(DV) q1 = dvushort_load(a_v_sum_d2_in);
        const dvshortx chess_storage(DV) q2 = dvushort_load(a_v_sum_d2_in);
        const dvshortx chess_storage(DV) q3 = dvushort_load(a_v_sum_d2_in);
        const dvshortx chess_storage(DV) q4 = dvushort_load(a_v_sum_d2_in);
        const dvshortx chess_storage(DV) q5 = dvushort_load(a_v_sum_d2_in);
        const dvshortx chess_storage(DV) q6 = dvushort_load(a_v_sum_d2_in);
        const dvshortx chess_storage(DV) q7 = dvushort_load(a_v_sum_d2_in);

        dvshortx chess_storage(DV) m0 = chess_dont_care(dvshortx), chess_storage(DV) a0 = chess_dont_care(dvshortx);
        dvshortx chess_storage(DV) m1 = chess_dont_care(dvshortx), chess_storage(DV) a1 = chess_dont_care(dvshortx);
        dvshortx chess_storage(DV) m2 = chess_dont_care(dvshortx), chess_storage(DV) a2 = chess_dont_care(dvshortx);
        dvshortx chess_storage(DV) m3 = chess_dont_care(dvshortx), chess_storage(DV) a3 = chess_dont_care(dvshortx);
        dvshortx chess_storage(DV) m4 = chess_dont_care(dvshortx), chess_storage(DV) a4 = chess_dont_care(dvshortx);
        dvshortx chess_storage(DV) m5 = chess_dont_care(dvshortx), chess_storage(DV) a5 = chess_dont_care(dvshortx);
        dvshortx chess_storage(DV) m6 = chess_dont_care(dvshortx), chess_storage(DV) a6 = chess_dont_care(dvshortx);

        m0.lo = vmax3(p0.lo, p0.hi, p1.lo);
        a0.lo = vadd3(q0.lo, q0.hi, q1.lo);
        m0.hi = vmax3(p0.hi, p1.lo, p1.hi);
        a0.hi = vadd3(q0.hi, q1.lo, q1.hi);
        m1.lo = vmax3(p2.lo, p1.lo, p1.hi);
        a1.lo = vadd3(q1.lo, q1.hi, q2.lo);
        m1.hi = vmax3(p2.lo, p2.hi, p1.hi);
        a1.hi = vadd3(q1.hi, q2.lo, q2.hi);
        m2.lo = vmax3(p2.lo, p2.hi, p3.lo);
        a2.lo = vadd3(q2.lo, q2.hi, q3.lo);
        m2.hi = vmax3(p2.hi, p3.lo, p3.hi);
        a2.hi = vadd3(q2.hi, q3.lo, q3.hi);
        m3.lo = vmax3(p4.lo, p3.lo, p3.hi);
        a3.lo = vadd3(q3.lo, q3.hi, q4.lo);
        m3.hi = vmax3(p4.lo, p4.hi, p3.hi);
        a3.hi = vadd3(q3.hi, q4.lo, q4.hi);
        m4.lo = vmax3(p4.lo, p4.hi, p5.lo);
        a4.lo = vadd3(q4.lo, q4.hi, q5.lo);
        m4.hi = vmax3(p4.hi, p5.lo, p5.hi);
        a4.hi = vadd3(q4.hi, q5.lo, q5.hi);
        m5.lo = vmax3(p6.lo, p5.lo, p5.hi);
        a5.lo = vadd3(q5.lo, q5.hi, q6.lo);
        m5.hi = vmax3(p6.lo, p6.hi, p5.hi);
        a5.hi = vadd3(q5.hi, q6.lo, q6.hi);
        m6.lo = vmax3(p6.lo, p6.hi, p7.lo);
        a6.lo = vadd3(q6.lo, q6.hi, q7.lo);
        m6.hi = vmax3(p6.hi, p7.lo, p7.hi);
        a6.hi = vadd3(q6.hi, q7.lo, q7.hi);

        vstore(m0, a_local_max);
        vstore(a0, a_local_sum_d8);
        vstore(m1, a_local_max);
        vstore(a1, a_local_sum_d8);
        vstore(m2, a_local_max);
        vstore(a2, a_local_sum_d8);
        vstore(m3, a_local_max);
        vstore(a3, a_local_sum_d8);
        vstore(m4, a_local_max);
        vstore(a4, a_local_sum_d8);
        vstore(m5, a_local_max);
        vstore(a5, a_local_sum_d8);
        vstore(m6, a_local_max);
        vstore(a6, a_local_sum_d8);
    }
}
