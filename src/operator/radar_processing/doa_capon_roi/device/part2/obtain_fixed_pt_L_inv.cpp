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

#include "obtain_fixed_pt_L_inv.h"

#include "capon_roi_part2_config.h"

#include <cupva_device.h>

/*
 * Fixed three-block traversal for the packed 16x16 L-inverse. This kernel
 * privately owns the immutable table in the same auxiliary bank used by its
 * offset AGEN and imaginary input.
 */
#if defined(CUPVA_PVA_GEN_NUMBER) && CUPVA_PVA_GEN_NUMBER >= 3
VMEM_NOEXPORT(D, int32_t, LinvOfstBuffer, BUF_SZ_LINV_OFST) = {
#else
VMEM_NOEXPORT(A, int32_t, LinvOfstBuffer, BUF_SZ_LINV_OFST) = {
#endif
    64,    960,   -896,  960,   896,  -1792, 960, 896,  832,   -2624, 960,   896,   832,   768, -3392, 960, 896,
    832,   768,   704,   -4096, 960,  896,   832, 768,  704,   640,   -4736, 960,   896,   832, 768,   704, 640,
    576,   576,   64,    448,   -384, 448,   384, -768, 448,   384,   320,   -1088, 448,   384, 320,   256, -1344,
    448,   384,   320,   256,   192,  -1536, 448, 384,  320,   256,   192,   128,   -1664, 448, 384,   320, 256,
    192,   128,   64,    -8128, 960,  896,   832, 768,  704,   640,   576,   -5312, 960,   896, 832,   768, 704,
    640,   576,   -5312, 960,   896,  832,   768, 704,  640,   576,   -5312, 960,   896,   832, 768,   704, 640,
    576,   -5312, 960,   896,   832,  768,   704, 640,  576,   -5312, 960,   896,   832,   768, 704,   640, 576,
    -5312, 960,   896,   832,   768,  704,   640, 576,  -5312, 960,   896,   832,   768,   704, 640,   576, -6272};

void init_obtain_fixed_pt_L_inv(float *const restrict scale_out, int32_t *const restrict L_inv_fixed_pt_out,
                                Para_capon_roi_part2_config &para)
{
    // ---- Stage 0 agen configs ----
    { // a_refmt: transposition store [real, imag, _] to L_inv_fixed_pt_out
        AgenWrapper w;
        w.size      = sizeof(float);
        w.n1        = 2; // real and imag per element
        w.s1        = 1;
        w.n2        = BUF_MAX_TRI_LEN; // elements per ROI group
        w.s2        = 3;               // each element occupies 3 slots: [re, im, _]
        agen a      = init(L_inv_fixed_pt_out);
        a.lane_ofst = (uint16_t)(BUF_LINE_PITCH_LINV_FIXED * sizeof(int32_t) / sizeof(dvfloat));
        INIT_AGEN2(a, w);
        para.cfg_refmt_stage0_stage1 = extract_agen_cfg(a);
    }
    { // a_ofst: offset table for pointer-based non-linear input access
        AgenWrapper w;
        w.size = 2 * sizeof(int32_t);
        w.n1   = BUF_MAX_TRI_LEN / 2;
        w.s1   = 1;
        agen a = init(LinvOfstBuffer);
        INIT_AGEN1(a, w);
        para.cfg_ofst_stage0 = extract_agen_cfg(a);
    }
    { // a_max: store max-abs per ROI group (predicated)
        AgenWrapper w;
        w.size = sizeof(dvfloat);
        w.n1   = CAPON_PART2_LINV_STAGE0_NITER;
        w.s1   = 0;
        agen a = init(scale_out);
        INIT_AGEN1(a, w);
        para.cfg_max_stage0 = extract_agen_cfg(a);
    }

    // ---- Stage 1 agen configs ----
    { // a_o: transposition store of output [re, im, -im]
        AgenWrapper w;
        w.size       = sizeof(int32_t);
        w.n1         = 3 * BUF_MAX_TRI_LEN;
        w.s1         = 1;
        agen a       = init(L_inv_fixed_pt_out);
        a.lane_ofst  = (uint16_t)(BUF_LINE_PITCH_LINV_FIXED * sizeof(int32_t) / sizeof(dvfloat));
        a.sat_opt    = 2; // treat 32-bit comparison values as signed
        a.sat_lim_lo = -(1 << 28);
        a.sat_val_lo = -(1 << 28);
        a.sat_lim_hi = 1 << 28;
        a.sat_val_hi = 1 << 28;
        INIT_AGEN1(a, w);
        para.cfg_output_stage1 = extract_agen_cfg(a);
    }
}

void obtain_fixed_pt_L_inv_stage0(const float *const restrict L_inv_real, const float *const restrict L_inv_imag,
                                  const Para_capon_roi_part2_config &para)
{
    agen_A a_refmt             = init_agen_A_from_cfg(para.cfg_refmt_stage0_stage1);
    PART2_LINV_AUX_AGEN a_ofst = INIT_PART2_LINV_AUX_AGEN(para.cfg_ofst_stage0);
    PART2_LINV_AUX_AGEN a_max  = INIT_PART2_LINV_AUX_AGEN(para.cfg_max_stage0);

    const char *restrict p_real = (const char *)L_inv_real;
    const char *restrict p_imag = (const char *)L_inv_imag;

    const uint32_t cnt_max = CAPON_PART2_LINV_STAGE0_CNTMAX;
    const uint32_t niter   = CAPON_PART2_LINV_STAGE0_NITER;

    uint32_t cnt_m = cnt_max, cnt_w = 0;

    dvfloatx m          = chess_dont_care(dvfloatx);
    const uint32_t mask = INT32_MAX;

    for (uint32_t i = 0; i < niter; ++i) chess_loop_range(3, )
    chess_prepare_for_pipelining
    {
        int32_t ofst0 = chess_dont_care(int32_t);
        int32_t ofst1 = chess_dont_care(int32_t);
        int32_t ofst2 = chess_dont_care(int32_t);
        int32_t ofst3 = chess_dont_care(int32_t);
        int_load(a_ofst, ofst0, ofst1);
        int_load(a_ofst, ofst2, ofst3);

        int32_t pred_m = chess_dont_care(int32_t);
        cnt_m          = mod_inc_pred_nz(cnt_m, cnt_max, pred_m);

        for (uint32_t j = 0; j < 4; ++j) chess_unroll_loop(*)
        {
            const int32_t ofst = j == 0 ? ofst0 : j == 1 ? ofst1 : j == 2 ? ofst2 : ofst3;

            const dvfloatx v0 = sign_extend(*(const dvfloat chess_storage(RAM_Ab) *)p_real);
            const dvfloatx v1 = sign_extend(*(const dvfloat PART2_LINV_AUX_BANK *)p_imag);

            p_real += ofst;
            p_imag += ofst;

            m = as_dvfloatx(dvmax3_ca(as_dvintx(v0) & mask, as_dvintx(v1) & mask, as_dvintx(m), j ? -1 : pred_m));

            vstore_transp(v0, a_refmt);
            vstore_transp(v1, a_refmt);
        }

        int32_t pred_w = chess_dont_care(int32_t);
        cnt_w          = mod_inc_pred_z(cnt_w, cnt_max, pred_w);
        vstore(m, a_max, pred_w);
    }
}

void obtain_fixed_pt_L_inv_stage1(const Para_capon_roi_part2_config &para)
{
    PART2_LINV_AUX_AGEN a_scale = INIT_PART2_LINV_AUX_AGEN(para.cfg_max_stage0);
    agen_A a_input              = init_agen_A_from_cfg(para.cfg_refmt_stage0_stage1);
    agen_A a_output             = init_agen_A_from_cfg(para.cfg_output_stage1);
    {
        dvfloat PART2_LINV_AUX_BANK *p = (dvfloat PART2_LINV_AUX_BANK *)a_scale.a;
        *p                             = extract(dvfrcp(sign_extend(*p)) * 0x1p28f);
    }
    chess_separator_scheduler();

    const uint32_t niter   = CAPON_PART2_LINV_STAGE1_NITER;
    const uint32_t cnt_max = CAPON_PART2_LINV_STAGE1_CNTMAX;

    uint32_t cnt = cnt_max;
    int32_t pred = 0;

    dvfloatx scale_0 = chess_dont_care(dvfloatx);
    dvfloatx scale_1 = chess_dont_care(dvfloatx);

    const dvintx zero = chess_dont_care(dvintx) & 0;

    for (uint32_t i = 0; i < niter; ++i) chess_loop_range(3, )
    chess_prepare_for_pipelining
    {
        cnt = mod_inc_pred_z(cnt, cnt_max, pred);

        for (uint32_t j = 0; j < 2; ++j) chess_unroll_loop(*)
        {
            // scale_0/scale_1 are two reads from the same location.
            // The main reason is that predicated loads require ping-pong register.
            // Using predicated load is for saving VMEM power.
            if (j)
                dvfloat_load(scale_1, a_scale, pred);
            else
                dvfloat_load(scale_0, a_scale, pred);

            const dvfloatx scale = j ? scale_1 : scale_0;

#if CUPVA_PVA_GEN_NUMBER < 3
            const uint32_t k_unroll = 1; // compute-bounded
#else
            const uint32_t k_unroll = 2; // store-bounded
#endif
            for (uint32_t k = 0; k < k_unroll; ++k) chess_unroll_loop(*)
            {
                const dvfloatx chess_storage(DV) f_real = dvfloat_load_transp(a_input);
                const dvfloatx chess_storage(DV) f_imag = dvfloat_load_transp(a_input);

                const dvintx i_real = dvfp_dvintx_round(scale * f_real);
                const dvintx i_imag = dvfp_dvintx_round(scale * f_imag);

                vstore_transp(i_real, a_output);
                vstore_transp(i_imag, a_output);
                vstore_transp(zero - i_imag, a_output);
            }
        }
    }
}
