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

#ifndef CAPON_ROI_PART2_CONFIG_HEADER_GUARD
#define CAPON_ROI_PART2_CONFIG_HEADER_GUARD

#include "capon_roi_vmem_layout.h"

#include <PvaOperatorTypes.h>
#include <cupva_device.h> /* defines `restrict` macro so part-2 prototypes parse under native g++ */
#include <stdint.h>

/* PART2_LINV_AUX_BANK is a complete storage qualifier, used as `dvfloat PART2_LINV_AUX_BANK *p`. */
#if CUPVA_PVA_GEN_NUMBER >= 3
#    define PART2_LINV_AUX_BANK chess_storage(RAM_Db)
#    define PART2_LINV_AUX_AGEN agen_D
#    define INIT_PART2_LINV_AUX_AGEN init_agen_D_from_cfg
#else
#    define PART2_LINV_AUX_BANK chess_storage(RAM_Ab)
#    define PART2_LINV_AUX_AGEN agen_A
#    define INIT_PART2_LINV_AUX_AGEN init_agen_A_from_cfg
#endif

/*
 * Part 2 is deliberately specialized for one 16-ROI group, a 16-element
 * steering vector, and a 16-azimuth x 8-elevation steering grid. Keep the
 * fixed geometry and all derived loop bounds here so neither the one-time
 * setup code nor the hot kernels need to materialize them in the parameter
 * block.
 */
enum
{
    CAPON_PART2_VFLOAT_LANES            = 8,
    CAPON_PART2_DVFLOAT_LANES           = 16,
    CAPON_PART2_ROI_GROUPS              = DV_ROI_NUM / CAPON_PART2_DVFLOAT_LANES,
    CAPON_PART2_HALF_SV_DIM             = BUF_MAX_SV_DIM / 2,
    CAPON_PART2_HALF_TRI_LEN            = CAPON_PART2_HALF_SV_DIM * (CAPON_PART2_HALF_SV_DIM + 1) / 2,
    CAPON_PART2_SV_GROUPS               = BUF_MAX_SV_NUM / 2 / CAPON_PART2_VFLOAT_LANES,
    CAPON_PART2_LINV_STAGE0_NITER       = BUF_MAX_TRI_LEN / 4,
    CAPON_PART2_LINV_STAGE0_CNTMAX      = CAPON_PART2_LINV_STAGE0_NITER - 1,
    CAPON_PART2_STEER_BASE_NITER        = BUF_SV_SINGLE_TILE_PIX / CAPON_PART2_VFLOAT_LANES / 2,
    CAPON_PART2_MUL_TRI_NITER           = CAPON_PART2_SV_GROUPS * SV_ROI_NUM,
    CAPON_PART2_MUL_RECT_NITER          = (CAPON_PART2_HALF_SV_DIM / 2) * CAPON_PART2_SV_GROUPS * SV_ROI_NUM,
    CAPON_PART2_MUL_CNTMAX              = CAPON_PART2_HALF_SV_DIM / 2 - 1,
    CAPON_PART2_RCP_OUTER_NITER         = BUF_MAX_SV_NUM / (CAPON_PART2_VFLOAT_LANES * 8) * SV_ROI_NUM,
    CAPON_PART2_SPEC_STAGE0_NITER       = BUF_MAX_SV_NUM / 8,
    CAPON_PART2_SPEC_STAGE0_CNTMAX      = CAPON_PART2_SPEC_STAGE0_NITER - 1,
    CAPON_PART2_NOISE_ORDER             = BUF_MAX_SV_NUM / 2,
    CAPON_PART2_NOISE_LOADS             = BUF_MAX_SV_NUM / 2,
    CAPON_PART2_FILTER_STAGE0_NITER     = NUM_AZ_BINS / 2,
    CAPON_PART2_FILTER_STAGE1_NITER     = NUM_EL_BINS - 2,
    CAPON_PART2_CANDIDATE_NITER         = (((NUM_EL_BINS - 2) * ((NUM_AZ_BINS - 2) / 2) + 3) / 4) * 4,
    CAPON_PART2_CANDIDATE_K             = 139,
    CAPON_PART2_TOP3_E_AZ               = 4,
    CAPON_PART2_TOP3_START_IDX          = (1 << CAPON_PART2_TOP3_E_AZ) + 1,
    CAPON_PART2_TOP3_CNTMAX             = NUM_AZ_BINS - 3,
    CAPON_PART2_TOP3_NEXT_ROW           = CAPON_PART2_TOP3_START_IDX - (NUM_AZ_BINS - 2),
    CAPON_PART2_TOP3_POINTS_PER_ROI     = (NUM_EL_BINS - 2) * (NUM_AZ_BINS - 2),
    CAPON_PART2_SV_LOWER_HALF_OFFSET    = CAPON_PART2_HALF_SV_DIM * BUF_MAX_SV_NUM,
    CAPON_PART2_LINV_PONG_OFFSET        = SV_ROI_NUM * BUF_LINE_PITCH_LINV_FIXED,
    CAPON_PART2_LINV_LOWER_RIGHT_OFFSET = 3 * CAPON_PART2_HALF_TRI_LEN,
    CAPON_PART2_LINV_RECT_OFFSET        = 2 * CAPON_PART2_LINV_LOWER_RIGHT_OFFSET,
    CAPON_PART2_RCP_PONG_OFFSET         = SV_ROI_NUM * BUF_LINE_PITCH_RCP
};

#if CUPVA_PVA_GEN_NUMBER < 3
enum
{
    CAPON_PART2_LINV_STAGE1_NITER  = BUF_MAX_TRI_LEN / 2,
    CAPON_PART2_LINV_STAGE1_CNTMAX = CAPON_PART2_LINV_STAGE1_NITER - 1,
    CAPON_PART2_STEER_NITER        = BUF_SV_SINGLE_TILE_PIX / CAPON_PART2_VFLOAT_LANES / 2,
    CAPON_PART2_SPEC_STAGE1_NITER  = BUF_MAX_SV_NUM / 4,
    CAPON_PART2_SPEC_STAGE1_CNTMAX = CAPON_PART2_SPEC_STAGE1_NITER - 1
};
#else
enum
{
    CAPON_PART2_LINV_STAGE1_NITER  = BUF_MAX_TRI_LEN / 4,
    CAPON_PART2_LINV_STAGE1_CNTMAX = CAPON_PART2_LINV_STAGE1_NITER - 1,
    CAPON_PART2_STEER_NITER        = ((CAPON_PART2_STEER_BASE_NITER + 5) / 6) * 6,
    CAPON_PART2_SPEC_STAGE1_NITER  = BUF_MAX_SV_NUM / 8,
    CAPON_PART2_SPEC_STAGE1_CNTMAX = CAPON_PART2_SPEC_STAGE1_NITER - 1
};
#endif

/* Per-kernel function declarations for Part 2 (Phases 4-12).
 * Every kernel consumes the unified Para_capon_roi_part2_config defined below
 * (each kernel header only forward-declares it). */
#include "compute_L_inv_times_steering_matrix.h"
#include "compute_fp_rcp_spectrum.h"
#include "estimate_noise_floor.h"
#include "extract_3x3filtered_spectrum_in16az8el_out14az6el.h"
#include "get_top3_peaks.h"
#include "mark_peak_candidates.h"
#include "obtain_fixed_pt_L_inv.h"
#include "obtain_fixed_pt_steering_matrix.h"
#include "obtain_spectrum.h"

/**
 * @brief Unified parameter block for Part 2 (Phases 4-12).
 *
 * All nine part-2 kernels read their parameters from this single struct. Fields
 * are grouped below by the kernel that initializes them; some producer configs
 * are intentionally reused by downstream kernels.
 */
struct Para_capon_roi_part2_config
{
    /* ---- obtain_fixed_pt_L_inv ---- */
    AgenCFG cfg_refmt_stage0_stage1;
    AgenCFG cfg_ofst_stage0;
    AgenCFG cfg_max_stage0;
    AgenCFG cfg_output_stage1;

    /* ---- shift_fixed_pt_steering_matrix ---- */
    AgenCFG cfg_sv_mat_0;
    AgenCFG cfg_sv_mat_1;
    int32_t shift_bits;

    /* ---- compute_L_inv_times_steering_matrix ---- */
    AgenCFG cfg_sv_mat_0_upper_left_triangular;
    AgenCFG cfg_sv_mat_1_upper_left_triangular;
    AgenCFG cfg_sv_mat_0_lower_right_triangular;
    AgenCFG cfg_sv_mat_1_lower_right_triangular;
    AgenCFG cfg_L_inv_upper_left_triangular;
    AgenCFG cfg_L_inv_lower_right_triangular;
    AgenCFG cfg_sv_mat_0_rectangular;
    AgenCFG cfg_sv_mat_1_rectangular;
    AgenCFG cfg_L_inv_rectangular;
    vshort permute_pattern;

    /* ---- compute_fp_rcp_spectrum ---- */
    AgenCFG cfg_L_inv_a_0;
    AgenCFG cfg_L_inv_a_1;
    AgenCFG cfg_out;

    /* ---- obtain_spectrum ---- */
    AgenCFG cfg_rcp_p_stage0;
    AgenCFG cfg_float_pt_spectrum_stage0_stage1;
    AgenCFG cfg_min_stage0;
    AgenCFG cfg_min_stage1;
    AgenCFG cfg_fixed_pt_spectrum_stage1;

    /* ---- estimate_noise_floor ---- */
    AgenCFG cfg_hist;
    AgenCFG cfg_spectrum;
    AgenCFG noise_cfg_noise_floor;
    uint16_t noise_n_roi_groups;
    vshort chess_storage(RAM_Cb) *restrict cumul_buf;

    /* ---- extract_3x3filtered_spectrum_in16az8el_out14az6el ---- */
    AgenCFG cfg_spectrum_stage0;
    AgenCFG cfg_v_max_out_stage0;
    AgenCFG cfg_v_sum_d2_out_stage0;
    AgenCFG cfg_v_max_in_stage1;
    AgenCFG cfg_v_sum_d2_in_stage1;
    AgenCFG cfg_local_max_stage1;
    AgenCFG cfg_local_sum_d8_stage1;

    /* ---- mark_peak_candidates ---- */
    AgenCFG cfg_cen;
    AgenCFG cand_cfg_candidate;

    /* ---- get_top3_peaks ---- */
    AgenCFG top3_cfg_candidate;
    AgenCFG cfg_power;
    AgenCFG cfg_az;
    AgenCFG cfg_el;
    float noise_floor_ratio;
    const float *restrict p_float_pt_spectrum;
};

/**
 * @brief Initialises every part-2 kernel parameter in one call.
 *
 * Forwards to the individual kernel init routines; argument semantics, buffer
 * layouts and superbank-placement constraints are documented in each kernel's
 * own header.
 *
 * @param params             Capon ROI operator parameters.
 * @param scale              Per-ROI scale (obtain_fixed_pt_L_inv -> get_top3_peaks).
 * @param L_inv_fixed_pt     Fixed-point L_inv (obtain_fixed_pt_L_inv -> compute_L_inv_times_steering_matrix).
 * @param rcp_spectrum       Reciprocal spectrum (compute_fp_rcp_spectrum -> obtain_spectrum).
 * @param spectrum_min       obtain_spectrum: per-ROI min scratch buffer.
 * @param fixed_pt_spectrum  UQ1.15 spectrum (obtain_spectrum / noise / filt / candidates).
 * @param float_pt_spectrum  Float spectrum (obtain_spectrum -> get_top3_peaks); 64-byte aligned.
 * @param noise_hist         estimate_noise_floor: histogram scratch buffer.
 * @param noise_cumul        estimate_noise_floor: cumulative scratch buffer.
 * @param noise_floor        Per-ROI noise floor (estimate_noise_floor -> get_top3_peaks).
 * @param local_max          3x3 max filter output (extract -> candidates).
 * @param local_sum_d8       3x3 sum/8 output (extract -> candidates).
 * @param candidate          Peak candidates (mark_peak_candidates -> get_top3_peaks).
 * @param para               Output unified parameter block.
 *
 * The fixed 16x128 steering geometry processes one 16-ROI group. L_inv input,
 * steering-tile, and output config bases are intentionally left for the hot
 * phase to patch immediately before their first use.
 */
void init_capon_roi_part2_config(const PVADoaCaponROIParams &params, float *const restrict scale,
                                 int32_t *const restrict L_inv_fixed_pt, float *const restrict rcp_spectrum,
                                 float *const restrict spectrum_min, uint16_t *const restrict fixed_pt_spectrum,
                                 float *const restrict float_pt_spectrum, void *const restrict noise_hist,
                                 void *const restrict noise_cumul, uint16_t *const restrict noise_floor,
                                 uint16_t *const restrict local_max, uint16_t *const restrict local_sum_d8,
                                 uint16_t *const restrict candidate, Para_capon_roi_part2_config &para);

/* Patch every steering/L_inv/reciprocal-spectrum AGEN base for the current
 * eight-ROI steering tile. Stable L_inv and reciprocal-spectrum bases are
 * recovered from configs initialized by the producer/consumer phases. */
void update_part2_sv_agen_base(Para_capon_roi_part2_config &para, int32_t *sv_mat_0, int32_t *sv_mat_1,
                               int32_t pingPong);
#endif
