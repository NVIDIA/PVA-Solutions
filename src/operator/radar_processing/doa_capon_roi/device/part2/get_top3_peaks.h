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

#ifndef GET_TOP3_PEAKS_HEADER_GUARD
#define GET_TOP3_PEAKS_HEADER_GUARD

#include <cupva_device.h> /* defines `restrict` macro so part-2 prototypes parse under native g++ */
#include <stdint.h>

struct Para_capon_roi_part2_config;

/**
 * @brief Initializes parameters for get_top3_peaks.
 *
 * For one group of 16 ROIs, selects the top-3 peaks and output its absolute power (P_abs)
 * and azimuth/elevation sub-bins. Ties are broken by scan order (preferring upper-left).
 *
 * A peak with P_abs == PVA_DOA_CAPON_ROI_INVALID_PEAK_POWER (-1.0f) is invalid; its az_bin and
 * el_bin cannot be used.
 * P_abs is set to PVA_DOA_CAPON_ROI_INVALID_PEAK_POWER when the candidate's power doesn't exceed
 * noise_floor (estimated) * noise_floor_ratio, or when it's computed as +-zero, negative, or NaN.
 * P_abs is never NaN.
 * If invalid_flag is set, it usually means that the input already contains NaN or +/-Inf, or
 * that some value exceeded the range supported by the implementation during computation.
 *
 * This is specialized for the 16x8 steering grid. The noise-floor input config
 * is reused from estimate_noise_floor. Output configs are initialized with
 * dummy bases and patched in the hot phase before first use.
 *
 * @param noise_floor_ratio  Noise floor threshold ratio, e.g. precomputed 10^(db/10).
 * @param candidate          Input candidate from mark_peak_candidates kernel.
 *                           The C layout is uint16_t[n_roi_groups][num_el-2][num_az-2][chess_elementsof(vushort)],
 *                           where n_roi_groups = roi_num / chess_elementsof(vushort).
 * @param para               Output parameter struct to be initialized.
 *                           Float-spectrum and per-ROI L-inverse-scale bases
 *                           are reused from their producer state/configs.
 */
void init_get_top3_peaks(const float noise_floor_ratio, const uint16_t *const restrict candidate,
                         Para_capon_roi_part2_config &para);

void get_top3_peaks(const Para_capon_roi_part2_config &para);

#endif
