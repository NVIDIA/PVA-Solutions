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

#include "compute_fp_rcp_spectrum.h"

#include "capon_roi_part2_config.h"

#include <cupva_device.h>

/*
 * FMA accumulation of ||x||^2 = sum(re^2) + sum(im^2) over sv_dim rows, followed by
 * vaddf reduction (re^2 + im^2), output stored to rcp_spectrum_out.
 *
 * The output cannot be infinity: max input |x| <= 2^31-1 (int32 range), so max x^2 < 2^62,
 * sv_dim * x^2 < 16 * 2^62 = 2^66, and re^2_sum + im^2_sum < 2^67, far below float32 max ~2^128.
 */
void init_compute_fp_rcp_spectrum(Para_capon_roi_part2_config &para)
{
    {
        AgenWrapper w;
        w.size = sizeof(dvfloat);
        w.n1   = 8 / 2;
        w.s1   = 1;
        w.n2   = BUF_MAX_SV_DIM; // all rows
        w.s2   = CAPON_PART2_SV_GROUPS;
        w.n3   = CAPON_PART2_SV_GROUPS / (8 / 2);
        w.s3   = 8 / 2;
        w.n4   = SV_ROI_NUM;
        w.s4   = BUF_MAX_SV_DIM * CAPON_PART2_SV_GROUPS;
        agen a = init((int32_t *)0); // patched before first use
        INIT_AGEN4(a, w);
        para.cfg_L_inv_a_0 = extract_agen_cfg(a);
        a.a                = (int32_t *)0; // patched before first use
        para.cfg_L_inv_a_1 = extract_agen_cfg(a);
    }

    // Output base is patched from cfg_rcp_p_stage0 before first use.
    {
        AgenWrapper w;
        w.size = sizeof(float);
        w.n1   = CAPON_PART2_SV_GROUPS;
        w.s1   = chess_elementsof(dvfloat);
        w.n2   = SV_ROI_NUM;
        w.s2   = BUF_LINE_PITCH_RCP;
        agen a = init((float *)0); // patched before first use
        INIT_AGEN2(a, w);
        para.cfg_out = extract_agen_cfg(a);
    }
}

static inline void my_store(const dvfloatx x, const dvfloatx y, agen_A &a_out)
{
    dvfloatx chess_storage(DV) r = chess_dont_care(dvfloatx);
    r.lo                         = vaddf(x.lo, x.hi);
    r.hi                         = vaddf(y.lo, y.hi);
    vstore(r, a_out);
}

void compute_fp_rcp_spectrum(const Para_capon_roi_part2_config &para)
{
    agen_B a_L_inv_a_0 = init_agen_B_from_cfg(para.cfg_L_inv_a_0);
    agen_C a_L_inv_a_1 = init_agen_C_from_cfg(para.cfg_L_inv_a_1);
    agen_A a_out       = init_agen_A_from_cfg(para.cfg_out);

    const uint32_t niter_outer = CAPON_PART2_RCP_OUTER_NITER;

    dvintx chess_storage(DW) v0 = dvint_load_di(a_L_inv_a_0);
    dvintx chess_storage(DW) v1 = dvint_load_di(a_L_inv_a_1);
    dvintx chess_storage(DW) v2 = dvint_load_di(a_L_inv_a_0);
    dvintx chess_storage(DW) v3 = dvint_load_di(a_L_inv_a_1);
    dvintx chess_storage(DW) v4 = dvint_load_di(a_L_inv_a_0);
    dvintx chess_storage(DW) v5 = dvint_load_di(a_L_inv_a_1);
    dvintx chess_storage(DW) v6 = dvint_load_di(a_L_inv_a_0);
    dvintx chess_storage(DW) v7 = dvint_load_di(a_L_inv_a_1);

    for (uint32_t r = 0; r < niter_outer; ++r) chess_loop_range(1, )
    {
#if CUPVA_PVA_GEN_NUMBER < 3
        dvfloatx chess_storage(DV8) acc0  = chess_dont_care(dvfloatx);
        dvfloatx chess_storage(DV9) acc1  = chess_dont_care(dvfloatx);
        dvfloatx chess_storage(DV10) acc2 = chess_dont_care(dvfloatx);
        dvfloatx chess_storage(DV11) acc3 = chess_dont_care(dvfloatx);
        dvfloatx chess_storage(DV12) acc4 = chess_dont_care(dvfloatx);
        dvfloatx chess_storage(DV13) acc5 = chess_dont_care(dvfloatx);
        dvfloatx chess_storage(DV14) acc6 = chess_dont_care(dvfloatx);
        dvfloatx chess_storage(DV15) acc7 = chess_dont_care(dvfloatx);
#else
        dvfloatx chess_storage(DW8) acc0  = chess_dont_care(dvfloatx);
        dvfloatx chess_storage(DW9) acc1  = chess_dont_care(dvfloatx);
        dvfloatx chess_storage(DW10) acc2 = chess_dont_care(dvfloatx);
        dvfloatx chess_storage(DW11) acc3 = chess_dont_care(dvfloatx);
        dvfloatx chess_storage(DW12) acc4 = chess_dont_care(dvfloatx);
        dvfloatx chess_storage(DW13) acc5 = chess_dont_care(dvfloatx);
        dvfloatx chess_storage(DW14) acc6 = chess_dont_care(dvfloatx);
        dvfloatx chess_storage(DW15) acc7 = chess_dont_care(dvfloatx);
#endif

        int pred = 0;

        for (uint32_t i = 0; i < BUF_MAX_SV_DIM; ++i) chess_loop_range(1, )
        {
            const dvfloatx chess_storage(DV) f0 = dvint_dvfp(v0);
            v0                                  = dvint_load_di(a_L_inv_a_0);
            const dvfloatx chess_storage(DV) f1 = dvint_dvfp(v1);
            v1                                  = dvint_load_di(a_L_inv_a_1);
            const dvfloatx chess_storage(DV) f2 = dvint_dvfp(v2);
            v2                                  = dvint_load_di(a_L_inv_a_0);
            const dvfloatx chess_storage(DV) f3 = dvint_dvfp(v3);
            v3                                  = dvint_load_di(a_L_inv_a_1);
            const dvfloatx chess_storage(DV) f4 = dvint_dvfp(v4);
            v4                                  = dvint_load_di(a_L_inv_a_0);
            const dvfloatx chess_storage(DV) f5 = dvint_dvfp(v5);
            v5                                  = dvint_load_di(a_L_inv_a_1);
            const dvfloatx chess_storage(DV) f6 = dvint_dvfp(v6);
            v6                                  = dvint_load_di(a_L_inv_a_0);
            const dvfloatx chess_storage(DV) f7 = dvint_dvfp(v7);
            v7                                  = dvint_load_di(a_L_inv_a_1);

            acc0 = dvmaddf(f0, f0, acc0, pred);
            acc1 = dvmaddf(f1, f1, acc1, pred);
            acc2 = dvmaddf(f2, f2, acc2, pred);
            acc3 = dvmaddf(f3, f3, acc3, pred);
            acc4 = dvmaddf(f4, f4, acc4, pred);
            acc5 = dvmaddf(f5, f5, acc5, pred);
            acc6 = dvmaddf(f6, f6, acc6, pred);
            acc7 = dvmaddf(f7, f7, acc7, pred);

            pred = -1;
        }
        my_store(acc0, acc1, a_out);
        my_store(acc2, acc3, a_out);
        my_store(acc4, acc5, a_out);
        my_store(acc6, acc7, a_out);
    }
}
