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

#ifndef EXTRACT_3X3FILTERED_SPECTRUM_IN16AZ8EL_OUT14AZ6EL_HEADER_GUARD
#define EXTRACT_3X3FILTERED_SPECTRUM_IN16AZ8EL_OUT14AZ6EL_HEADER_GUARD

#include <cupva_device.h> /* defines `restrict` macro so part-2 prototypes parse under native g++ */
#include <stdint.h>

struct Para_capon_roi_part2_config;

/**
 * @brief Initializes parameters for extract_3x3filtered_spectrum_in16az8el_out14az6el.
 *
 * Computes separable 3x3 max and 3x3 sum/8 on a 16az x 8el UQ1.15 spectrum.
 * Output dimensions are 14az x 6el (outermost ring removed).
 * Two outputs: local_max (3x3 max value) and local_sum_d8 (3x3 sum / 8, rounded).
 *
 * It is specialized for a 16x8 steering grid and one 16-ROI group. The
 * local-max output config is reused by mark_peak_candidates.
 *
 * @param fixed_pt_spectrum  Input buffer, UQ1.15, the C layout is uint16_t[n_roi_groups][8][16][chess_elementsof(vushort)]
 *                           where n_roi_groups = roi_num / chess_elementsof(vushort).
 *                           The input range should be [0, 2^15].
 * @param local_max          3x3 max filter output, the C layout is uint16_t[16 * 6 * roi_num].
 *                           Only the first 14 * 6 * roi_num output are valid, range is [0, 2^15].
 * @param local_sum_d8       3x3 Sum/8 output, same layout with local_max.
 *                           Only the first 14 * 6 * roi_num output are valid, range is [0, 3*3*2^15/8=36864].
 *                           For the sake of performance, put at different superbank from local_max.
 * @param para               Output parameter struct to be initialized.
 */
void init_extract_3x3filtered_spectrum_in16az8el_out14az6el(const uint16_t *const restrict fixed_pt_spectrum,
                                                            uint16_t *const restrict local_max,
                                                            uint16_t *const restrict local_sum_d8,
                                                            Para_capon_roi_part2_config &para);

static inline void extract_3x3filtered_spectrum_in16az8el_out14az6el(const Para_capon_roi_part2_config &para)
{
    void extract_3x3filtered_spectrum_in16az8el_out14az6el_stage0(const Para_capon_roi_part2_config &);
    void extract_3x3filtered_spectrum_in16az8el_out14az6el_stage1(const Para_capon_roi_part2_config &);
    extract_3x3filtered_spectrum_in16az8el_out14az6el_stage0(para);
    extract_3x3filtered_spectrum_in16az8el_out14az6el_stage1(para);
}

#endif
