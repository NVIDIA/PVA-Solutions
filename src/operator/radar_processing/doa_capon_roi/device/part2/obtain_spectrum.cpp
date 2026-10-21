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

#include "obtain_spectrum.h"

#include "capon_roi_part2_config.h"

#include <cupva_device.h>

void init_obtain_spectrum(float *const restrict rcp_spectrum,         /* bank A */
                          float *const restrict min_buf,              /* bank A */
                          uint16_t *const restrict fixed_pt_spectrum, /* bank B */
                          float *const restrict float_pt_spectrum,    /* bank B */
                          Para_capon_roi_part2_config &para)
{
    // Keep the interior origin used by top-3 lookup setup. Computing this once
    // avoids adding pointer arithmetic to the hot top-3 kernel.
    para.p_float_pt_spectrum = float_pt_spectrum + (NUM_AZ_BINS + 1) * chess_elementsof(dvfloat);

    {
        AgenWrapper w;
        w.size      = sizeof(float);
        w.n1        = BUF_MAX_SV_NUM;
        w.s1        = 1;
        agen a      = init(rcp_spectrum);
        a.lane_ofst = (uint16_t)(BUF_LINE_PITCH_RCP * sizeof(float) / sizeof(dvfloat));
        INIT_AGEN1(a, w);
        para.cfg_rcp_p_stage0 = extract_agen_cfg(a);
    }
    {
        AgenWrapper w;
        w.size = sizeof(dvfloat);
        w.n1   = BUF_MAX_SV_NUM;
        w.s1   = 1;
        agen a = init(float_pt_spectrum);
        INIT_AGEN1(a, w);
        a.sat_opt                                = 2; // treat 32-bit comparison values as signed
        a.sat_lim_lo                             = 0;
        a.sat_val_lo                             = 0;
        a.sat_lim_hi                             = 0x7F7FFFFF; // FLT_MAX
        a.sat_val_hi                             = 0x7F7FFFFF; // FLT_MAX
        para.cfg_float_pt_spectrum_stage0_stage1 = extract_agen_cfg(a);
    }
    {
        AgenWrapper w;
        w.size = sizeof(dvfloat);
        w.n1   = CAPON_PART2_SPEC_STAGE0_NITER; // unroll 2 then 4
        w.s1   = 0;
        agen a = init(min_buf);
        INIT_AGEN1(a, w);
        para.cfg_min_stage0 = extract_agen_cfg(a);

        a.n1                = 2; // load and store
        a.sat_opt           = 2; // treat 32-bit comparison values as signed
        a.sat_lim_lo        = as_int32(1.f / 0x1p113f);
        a.sat_val_lo        = as_int32(1.f / 0x1p113f);
        a.sat_lim_hi        = 0x7F7FFFFF; // FLT_MAX
        a.sat_val_hi        = 0x7F7FFFFF; // FLT_MAX
        para.cfg_min_stage1 = extract_agen_cfg(a);
    }
    {
        AgenWrapper w;
        w.size       = sizeof(dvushort);
        w.n1         = BUF_MAX_SV_NUM / 2;
        w.s1         = 1;
        agen a       = init(fixed_pt_spectrum);
        a.sat_opt    = 2; // treat 32-bit comparison values as signed
        a.sat_lim_lo = 0;
        a.sat_val_lo = 0;
        a.sat_lim_hi = 1 << 15;
        a.sat_val_hi = 1 << 15;
        INIT_AGEN1(a, w);
        para.cfg_fixed_pt_spectrum_stage1 = extract_agen_cfg(a);
    }
}

void obtain_spectrum_stage0(const Para_capon_roi_part2_config &para)
{
    agen_A a_rcp_p = init_agen_A_from_cfg(para.cfg_rcp_p_stage0);
    agen_B a_p     = init_agen_B_from_cfg(para.cfg_float_pt_spectrum_stage0_stage1);
    agen_A a_min   = init_agen_A_from_cfg(para.cfg_min_stage0);

    const uint32_t niter   = CAPON_PART2_SPEC_STAGE0_NITER;
    const uint32_t cnt_max = CAPON_PART2_SPEC_STAGE0_CNTMAX;

    uint32_t cnt_m = cnt_max, cnt_w = 0;
    int32_t pred_m = 0, pred_w = 0;

    dvfloatx m = chess_dont_care(dvfloatx);

    for (uint32_t i = 0; i < niter; ++i) chess_loop_range(3, )
    chess_prepare_for_pipelining
    {
        cnt_m = mod_inc_pred_nz(cnt_m, cnt_max, pred_m);

        for (uint32_t j = 0; j < 4; ++j) chess_unroll_loop(*)
        {
            // Vectorized on ROIs.
            const dvfloatx r0 = dvfloat_load_transp(a_rcp_p);
            const dvfloatx r1 = dvfloat_load_transp(a_rcp_p);

            m = as_dvfloatx(dvmin3_ca(as_dvintx(r0), as_dvintx(r1), as_dvintx(m), j ? -1 : pred_m));

            vstore(dvfrcp(r0), a_p);
            vstore(dvfrcp(r1), a_p);
        }

        cnt_w = mod_inc_pred_z(cnt_w, cnt_max, pred_w);
        vstore(m, a_min, pred_w);
    }
}

void obtain_spectrum_stage1(const Para_capon_roi_part2_config &para)
{
    agen_A a_scale = init_agen_A_from_cfg(para.cfg_min_stage0);
    agen_B a_power = init_agen_B_from_cfg(para.cfg_float_pt_spectrum_stage0_stage1);
    agen_B a_out   = init_agen_B_from_cfg(para.cfg_fixed_pt_spectrum_stage1);
    {
        agen_A a_min2scale = init_agen_A_from_cfg(para.cfg_min_stage1);
        const dvfloatx v   = dvfloat_load(a_min2scale) * 0x1p15f;
        // lane_ofst = 0 is used intentionally to avoid unnecessary memory stalls
        vstore_transp(v, a_min2scale);
    }
    chess_separator_scheduler();

    const uint32_t niter   = CAPON_PART2_SPEC_STAGE1_NITER;
    const uint32_t cnt_max = CAPON_PART2_SPEC_STAGE1_CNTMAX;

    uint32_t cnt = cnt_max;
    int32_t pred = 0;

    dvfloatx scale0 = chess_dont_care(dvfloatx);
    dvfloatx scale1 = chess_dont_care(dvfloatx);

    for (uint32_t i = 0; i < niter; ++i) chess_loop_range(3, )
    chess_prepare_for_pipelining
    {
        cnt = mod_inc_pred_z(cnt, cnt_max, pred);

        for (uint32_t j = 0; j < 2; ++j) chess_unroll_loop(*)
        {
            // scale0/scale1 are two reads from the same location.
            // The main reason is that predicated loads require ping-pong in a iteration.
            // Predicated load is for power saving.
            if (j)
                dvfloat_load(scale1, a_scale, pred);
            else
                dvfloat_load(scale0, a_scale, pred);

            const dvfloatx scale = j ? scale1 : scale0;

#if CUPVA_PVA_GEN_NUMBER < 3
            const uint32_t k_unroll = 1;
#else
            const uint32_t k_unroll = 2;
#endif
            for (uint32_t k = 0; k < k_unroll; ++k) chess_unroll_loop(*)
            {
                // Vectorized on ROIs.
                const dvfloatx chess_storage(DV) p0 = dvfloat_load(a_power);
                const dvfloatx chess_storage(DV) p1 = dvfloat_load(a_power);

                vstore(dvfp_dvintx_round(dvmulf(scale, p0)), dvfp_dvintx_round(dvmulf(scale, p1)), a_out);
            }
        }
    }
}
