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

#include "capon_roi_part2_config.h"

#include <cupva_device.h>

/*
 * Single entry point that initialises the unified part-2 parameter block by
 * forwarding to each kernel's own init routine. Every kernel writes only its
 * own (prefixed or shared) fields of Para_capon_roi_part2_config.
 */
void init_capon_roi_part2_config(const PVADoaCaponROIParams &params, float *const restrict scale,
                                 int32_t *const restrict L_inv_fixed_pt, float *const restrict rcp_spectrum,
                                 float *const restrict spectrum_min, uint16_t *const restrict fixed_pt_spectrum,
                                 float *const restrict float_pt_spectrum, void *const restrict noise_hist,
                                 void *const restrict noise_cumul, uint16_t *const restrict noise_floor,
                                 uint16_t *const restrict local_max, uint16_t *const restrict local_sum_d8,
                                 uint16_t *const restrict candidate, Para_capon_roi_part2_config &para)
{
    const uint16_t steering_qbits = (uint16_t)params.steeringQBits;
    const float noise_floor_ratio = params.noiseFloorRatio;

    // Phase 4: obtain_fixed_pt_L_inv
    init_obtain_fixed_pt_L_inv(scale, L_inv_fixed_pt, para);

    // Phase 5: shift_fixed_pt_steering_matrix. Tile bases are patched in the hot loop.
    init_shift_fixed_pt_steering_matrix(steering_qbits, para);

    // Phase 6: compute_L_inv_times_steering_matrix. Reuse the Phase-4 output base.
    init_compute_L_inv_times_steering_matrix(para);

    // Phase 7: compute_fp_rcp_spectrum. Reuse the Phase-8 reciprocal-spectrum base.
    init_compute_fp_rcp_spectrum(para);

    // Phase 8: obtain_spectrum
    init_obtain_spectrum(rcp_spectrum, spectrum_min, fixed_pt_spectrum, float_pt_spectrum, para);

    // Phase 9: estimate_noise_floor
    init_estimate_noise_floor(fixed_pt_spectrum, noise_hist, noise_cumul, noise_floor, para);

    // Phase 10: extract_3x3filtered_spectrum
    init_extract_3x3filtered_spectrum_in16az8el_out14az6el(fixed_pt_spectrum, local_max, local_sum_d8, para);

    // Phase 11: mark_peak_candidates. Reuse the Phase-10 local-sum input config.
    init_mark_peak_candidates(fixed_pt_spectrum, candidate, para);

    // Phase 12: get_top3_peaks. Output bases are patched in the hot loop.
    init_get_top3_peaks(noise_floor_ratio, candidate, para);
}

void update_part2_sv_agen_base(Para_capon_roi_part2_config &para, int32_t *sv_mat_0, int32_t *sv_mat_1,
                               int32_t pingPong)
{
    // update agen config address
    cupvaModifyAgenCfgBase(&para.cfg_sv_mat_0, sv_mat_0);
    cupvaModifyAgenCfgBase(&para.cfg_sv_mat_1, sv_mat_1);
    cupvaModifyAgenCfgBase(&para.cfg_sv_mat_0_upper_left_triangular, sv_mat_0);
    cupvaModifyAgenCfgBase(&para.cfg_sv_mat_1_upper_left_triangular, sv_mat_1);
    cupvaModifyAgenCfgBase(&para.cfg_sv_mat_0_lower_right_triangular,
                           sv_mat_0 + CAPON_PART2_SV_LOWER_HALF_OFFSET); // int32 elements
    cupvaModifyAgenCfgBase(&para.cfg_sv_mat_1_lower_right_triangular,
                           sv_mat_1 + CAPON_PART2_SV_LOWER_HALF_OFFSET); // int32 elements
    cupvaModifyAgenCfgBase(&para.cfg_sv_mat_0_rectangular, sv_mat_0);
    cupvaModifyAgenCfgBase(&para.cfg_sv_mat_1_rectangular, sv_mat_1);
    cupvaModifyAgenCfgBase(&para.cfg_L_inv_a_0, sv_mat_0);
    cupvaModifyAgenCfgBase(&para.cfg_L_inv_a_1, sv_mat_1);

    int32_t *const L_inv_fixed_pt_ptr = (int32_t *)cupvaGetAgenCfgBase(&para.cfg_output_stage1);
    int32_t *L_inv_base = pingPong ? L_inv_fixed_pt_ptr + CAPON_PART2_LINV_PONG_OFFSET : L_inv_fixed_pt_ptr;
    cupvaModifyAgenCfgBase(&para.cfg_L_inv_upper_left_triangular, L_inv_base);
    cupvaModifyAgenCfgBase(&para.cfg_L_inv_lower_right_triangular,
                           L_inv_base + CAPON_PART2_LINV_LOWER_RIGHT_OFFSET); // int32 elements
    cupvaModifyAgenCfgBase(&para.cfg_L_inv_rectangular,
                           L_inv_base + CAPON_PART2_LINV_RECT_OFFSET); // int32 elements

    float *const rcp_spectrum_ptr = (float *)cupvaGetAgenCfgBase(&para.cfg_rcp_p_stage0);
    float *const rcp_spectrum_pingpong_ptr =
        rcp_spectrum_ptr + pingPong * CAPON_PART2_RCP_PONG_OFFSET; // float elements
    cupvaModifyAgenCfgBase(&para.cfg_out, rcp_spectrum_pingpong_ptr);
}
