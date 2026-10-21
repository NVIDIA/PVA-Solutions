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

#ifndef MARK_PEAK_CANDIDATES_HEADER_GUARD
#define MARK_PEAK_CANDIDATES_HEADER_GUARD

#include <cupva_device.h> /* defines `restrict` macro so part-2 prototypes parse under native g++ */
#include <stdint.h>

struct Para_capon_roi_part2_config;

/**
 * @brief Initializes parameters for mark_peak_candidates.
 *
 * For each interior position (14az x 6el), outputs the center spectrum value if it
 * is the 3x3 local maximum AND exceeds the neighbor prominence threshold
 * (center > (k_nb * local_sum_d8 + 64) >> 7), otherwise outputs 0.
 *
 * Input: fixed_pt_spectrum from obtain_spectrum, local_max and local_sum_d8 from
 * extract_3x3filtered_spectrum_in16az8el_out14az6el.
 *
 * This is specialized for the 16x8 steering grid and one 16-ROI group.
 * Its local-max and local-sum inputs reuse the output configs produced by the
 * 3x3 filter initializer. Store rounding and saturation metadata in the
 * local-sum config do not affect candidate loads.
 *
 * @param fixed_pt_spectrum  Input spectrum buffer.
 *                           The C layout is uint16_t[n_roi_groups][num_el][num_az][chess_elementsof(vushort)],
 *                           where n_roi_groups = roi_num / chess_elementsof(vushort).
 * @param candidate          Output buffer, same layout as local_max. 0 if rejected, center value if accepted.
 *                           The kernel does not strictly require it, but it is safe if the output pointer has the
 *                           same value (head address) as any input pointer (aliasing).
 * @param para               Output parameter struct to be initialized.
 */
void init_mark_peak_candidates(const uint16_t *const restrict fixed_pt_spectrum, uint16_t *const restrict candidate,
                               Para_capon_roi_part2_config &para);

void mark_peak_candidates(const Para_capon_roi_part2_config &para);

#endif
