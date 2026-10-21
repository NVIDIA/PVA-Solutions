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

#include "get_top3_peaks.h"

#include "capon_roi_part2_config.h"

#include <cupva_device.h>

/* Sub-bin quadratic interpolation toggle.
 * Controlled by the device CMake target (`-DCAPON_ROI_ENABLE_SUBBIN_INTERP=0/1`).
 * When 0, the kernel writes the integer (el, az) bin indices directly instead of
 * running quadratic_interp_1d on the 3x3 neighborhood. */
#ifndef CAPON_ROI_ENABLE_SUBBIN_INTERP
#    define CAPON_ROI_ENABLE_SUBBIN_INTERP 1
#endif

void init_get_top3_peaks(const float noise_floor_ratio, const uint16_t *const restrict candidate,
                         Para_capon_roi_part2_config &para)
{
    para.noise_floor_ratio = noise_floor_ratio;

    {
        // candidate agen: sequential read, one vushort (16 ROIs) per position
        // dvushort_load_perm reads a dvushort but pat only selects the first 16 values
        AgenWrapper w;
        w.size = sizeof(vushort);
        w.n1   = CAPON_PART2_TOP3_POINTS_PER_ROI;
        w.s1   = 1;
        agen a = init((uint16_t *)candidate);
        INIT_AGEN1(a, w);
        para.top3_cfg_candidate = extract_agen_cfg(a);
    }
    {
        // lp ~ 16k+1 for T1 transpose store
        // output agens: float[roi_num][3]
        AgenWrapper w;
        w.size = sizeof(float);
        w.n1   = 3; // number of peaks
        w.s1   = 1;

        agen a      = init((float *)0); // patched before first use
        a.lane_ofst = (PEAK_POWER_LINE_PITCH - 1) / CAPON_PART2_DVFLOAT_LANES;
        INIT_AGEN1(a, w);
        para.cfg_az = extract_agen_cfg(a);

        a.a         = (float *)0; // patched before first use
        para.cfg_el = extract_agen_cfg(a);

        // Store-side saturation on the float bit pattern, compared as signed int32:
        //   bits < 1          : +0, negative floats, and negative NaNs -> invalid
        //   bits > 0x7F800000 : positive NaNs (+Inf itself passes)    -> invalid
        // so P_abs is either a positive value or PVA_DOA_CAPON_ROI_INVALID_PEAK_POWER, never NaN.
        const union
        {
            float f;
            int32_t i;
        } invalid_power = {PVA_DOA_CAPON_ROI_INVALID_PEAK_POWER};

        a.a            = (float *)0; // patched before first use
        a.sat_opt      = 2;          // treat 32-bit comparison values as signed
        a.sat_lim_lo   = 1;
        a.sat_val_lo   = invalid_power.i;
        a.sat_lim_hi   = 0x7F800000; // +Inf
        a.sat_val_hi   = invalid_power.i;
        para.cfg_power = extract_agen_cfg(a);
    }
}

static inline dvfloatx quadratic_interp_1d(const dvintx i, const dvfloatx y_minus, const dvfloatx y0,
                                           const dvfloatx y_plus)
{
    dvfloatx scale = dvfrcp(dvmsubf(y_plus, 0.5f, dvmsubf(y_minus, 0.5f, y0)) * 4.f);
    // -0 will get -Inf, so trim sign bit is required.
    scale = dvmux((as_dvintx(scale) & INT32_MAX) == 0x7F800000 /*Inf*/, 0.f, scale); // collinear -> 0 (avoid 0*inf)
    const dvfloatx offset = (y_plus - y_minus) * scale;
    // Accept sub-bin offset only when center is a STRICT local max and |offset| <= 0.5.
    // Strict '>' is intentional: any tie falls back to the coarse bin by design.
    // NaN/inf safe: any failing compare clears cond, so we fall back to the coarse bin.
    const dvintx cond = (y0 > y_minus) & (y0 > y_plus) & (as_dvfloatx(as_dvintx(offset) & INT32_MAX) <= 0.5f);
    return dvintx_dvfp(i) + dvmux(cond, offset, as_dvfloatx(cond));
}

void get_top3_peaks(const Para_capon_roi_part2_config &para)
{
    vcharx pat        = chess_dont_care(vcharx);
    dvshortx odd_lane = chess_dont_care(dvshortx);
    {
        const vshortx seq_012toF = vcollate_idx(replicateh(0));
        vdemote_i(seq_012toF, seq_012toF, pat);
        odd_lane.hi = odd_lane.lo = seq_012toF & 1;
    }

    agen_A a_candidate   = init_agen_A_from_cfg(para.top3_cfg_candidate);
    agen_B a_noise_floor = init_agen_B_from_cfg(para.noise_cfg_noise_floor);
    agen_A a_power       = init_agen_A_from_cfg(para.cfg_power);
    agen_A a_az          = init_agen_A_from_cfg(para.cfg_az);
    agen_A a_el          = init_agen_A_from_cfg(para.cfg_el);

    const int32_t start_idx = CAPON_PART2_TOP3_START_IDX;
    const int32_t cnt_max   = CAPON_PART2_TOP3_CNTMAX;
    const int32_t j_nxt_row = CAPON_PART2_TOP3_NEXT_ROW;
    const int32_t num_az    = NUM_AZ_BINS;
    const int32_t e_az      = CAPON_PART2_TOP3_E_AZ;
    const uint32_t az_mask  = (1 << e_az) - 1;

    const float noise_floor_ratio = para.noise_floor_ratio;

    // p_center points to interior origin (1, 1).
    // All neighbor base pointers stay within the float_pt_spectrum buffer.
    const float chess_storage(RAM_Bb) *restrict p_center =
        (const float chess_storage(RAM_Bb) *)para.p_float_pt_spectrum;
#if CAPON_ROI_ENABLE_SUBBIN_INTERP
    const float chess_storage(RAM_Bb) *restrict p_left   = p_center - chess_elementsof(dvfloat);
    const float chess_storage(RAM_Bb) *restrict p_right  = p_center + chess_elementsof(dvfloat);
    const float chess_storage(RAM_Bb) *restrict p_top    = p_center - chess_elementsof(dvfloat) * num_az;
    const float chess_storage(RAM_Bb) *restrict p_bottom = p_center + chess_elementsof(dvfloat) * num_az;
#endif

    const dvfloat PART2_LINV_AUX_BANK *restrict p_scale =
        (const dvfloat PART2_LINV_AUX_BANK *)cupvaGetAgenCfgBase(const_cast<AgenCFG *>(&para.cfg_max_stage0));

    const uint32_t n_per_roi_group = CAPON_PART2_TOP3_POINTS_PER_ROI;

    {
        // Init to -1 (not 0): all interior candidates (including value=0) are > -1 in s24 signed
        // comparison, so they replace the initial slots via dvsort2pl. After the selection,
        // all slots carry idx from el_bin [1, num_el - 2], az_bin [1, num_az - 2].
        // This guarantees interpolation inputs never go outside float_pt_spectrum.
        // Tie-breaking: dvsort2pl does not swap on equal power, so the first candidate
        // encountered in scan order (elevation-major, ascending) is kept.
        dvshortx p0 = chess_dont_care(dvshortx) | -1 /*0xFFFF'FFFFu truncated to s24 0xFF'FFFF=-1*/, p1 = p0, p2 = p0;

        int32_t cnt = 0, idx = start_idx;

        for (uint32_t i = 0; i < n_per_roi_group; ++i) chess_loop_range(6, )
        chess_unroll_loop(2)
        chess_prepare_for_pipelining
        {
            dvshortx tmp = chess_dont_care(dvshortx);
            dvsort2pl(dvmux(odd_lane, idx, dvushort_load_perm(a_candidate, pat)), p2, tmp, p2);
            (void)tmp;
            dvsort2pl(p2, p1, p2, p1);
            dvsort2pl(p1, p0, p1, p0);

            cnt = mod_inc(cnt, cnt_max);
            idx += mux(cnt, 1, j_nxt_row);
        }

        // scale = frcp(max(|L_inv|)) * 2^28, same for all 3 peaks within a roi.
        // It's from obtain_fixed_pt_L_inv kernel.
        const dvfloatx scale = sign_extend(*p_scale);

        // P_abs = 1 / ||(L_inv @ a)||^2
        // float_pt_spectrum ~ 1 / ||(L_inv * scale @ a * 2^22) / 2^32||^2
        //             ~ P_abs / (scale * 2^-10)^2
        // P_abs ~ scale^2 * 2^-20 * float_pt_spectrum
        const dvfloatx p_alpha = scale * scale * (1.f / 0x1p20f);

        // noise floor threshold, same for all 3 peaks within a roi.
        // It's from estimate_noise_floor kernel.
        const dvfloatx noise_floor_th = dvint_dvfp(vushort_dvintx_load(a_noise_floor)) * noise_floor_ratio;

        for (uint32_t i = 0; i < 3; ++i) chess_unroll_loop(*)
        {
            const dvshortx p = i == 2 ? p2 : i == 1 ? p1 : p0;

            dvintx idx    = chess_dont_care(dvintx);
            dvintx c_norm = chess_dont_care(dvintx);
            vpromote_di(p.lo, c_norm.lo, idx.lo);
            vpromote_di(p.hi, c_norm.hi, idx.hi);

            const dvintx az_int = idx & az_mask;
            const dvintx el_int = idx >> e_az;

            // tbl_loc relative to interior origin (1, 1):
            //   (el-1)*num_az + (az-1) = el*num_az + az - (num_az + 1)
            // This keeps all vlookup base pointers (p_center, p_top, p_left, etc.)
            // within same superbank.  With base at (1,1), p_top points to (0, 1)
            // and p_left to (1, 0), both inside the float_pt_spectrum buffer.
            const dvintx tbl_loc = dvmaddwhw(el_int, num_az, az_int, VPU_ROUND_0, -1) - (num_az + 1);

            const dvfloatx C = vlookup_16f(p_center, tbl_loc);

            dvfloatx P_abs = p_alpha * C;

            // Mark peaks at or below the noise-floor threshold as invalid.
            P_abs = dvmux(dvint_dvfp(c_norm) <= noise_floor_th, PVA_DOA_CAPON_ROI_INVALID_PEAK_POWER, P_abs);

            vstore_transp(P_abs, a_power);
#if CAPON_ROI_ENABLE_SUBBIN_INTERP
            const dvfloatx L = vlookup_16f(p_left, tbl_loc);
            const dvfloatx R = vlookup_16f(p_right, tbl_loc);
            const dvfloatx T = vlookup_16f(p_top, tbl_loc);
            const dvfloatx B = vlookup_16f(p_bottom, tbl_loc);
            vstore_transp(quadratic_interp_1d(az_int, L, C, R), a_az);
            vstore_transp(quadratic_interp_1d(el_int, T, C, B), a_el);
#else
            vstore_transp(dvintx_dvfp(az_int), a_az);
            vstore_transp(dvintx_dvfp(el_int), a_el);
#endif
        }
    }
}
