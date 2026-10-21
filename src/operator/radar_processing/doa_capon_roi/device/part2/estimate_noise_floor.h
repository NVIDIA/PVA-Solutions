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

#ifndef ESTIMATE_NOISE_FLOOR_HEADER_GUARD
#define ESTIMATE_NOISE_FLOOR_HEADER_GUARD

#include <cupva_device.h> /* defines `restrict` macro so part-2 prototypes parse under native g++ */
#include <stdint.h>

struct Para_capon_roi_part2_config;

/**
 * @brief Initializes parameters for noise floor estimation.
 *
 * For each ROI, finds the [sv_num / 2]-th smallest value (index sv_num / 2 - 1) among sv_num
 * spectrum values, but only estimates the 10 MSBs for saving cycles.
 *
 * This is specialized for 128 spectrum values and one 16-ROI group. The
 * noise-floor AGEN config is also reused by get_top3_peaks.
 *
 * @param fixed_pt_spectrum  Input spectrum buffer, UQ1.15, the C layout is uint16_t[roi_num / 16][sv_num][16].
 * @param hist_buf           64B-aligned 2KB intermediate buffer put at different superbank from fixed_pt_spectrum.
 * @param cumul_buf          64B-aligned 1056-byte intermediate buffer put at different superbank from hist_buf.
 *                           It's fine if fixed_pt_spectrum and cumul_buf are at same superbank.
 * @param noise_floor        Output buffer, the C layout is uint16_t[roi_num].
 * @param para               Output parameter struct to be initialized.
 */
void init_estimate_noise_floor(const uint16_t *const restrict fixed_pt_spectrum, void *const restrict hist_buf,
                               void *const restrict cumul_buf, uint16_t *const restrict noise_floor,
                               Para_capon_roi_part2_config &para);

void estimate_noise_floor(const Para_capon_roi_part2_config &para);

#endif
