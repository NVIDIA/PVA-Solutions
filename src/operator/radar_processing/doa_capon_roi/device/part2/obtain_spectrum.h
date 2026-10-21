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

#ifndef OBTAIN_SPECTRUM_HEADER_GUARD
#define OBTAIN_SPECTRUM_HEADER_GUARD

#include <cupva_device.h> /* defines `restrict` macro so part-2 prototypes parse under native g++ */
#include <stdint.h>

struct Para_capon_roi_part2_config;

/**
 * @brief Initializes parameters for obtain_spectrum.
 *
 * Converts the reciprocal of floating-point spectrum to UQ1.15 fixed-point and floating-point spectrum.
 *
 * For fixed-point output:
 *     Input range is [0, +Inf]. Output range is [0, 2^15].
 *     If any input is 0, the corresponding output will be 2^15 without setting invalid flag.
 *     If any input is +Inf, the corresponding output will be 0 without setting invalid flag.
 *     If any input is negative or NaN, then there is a bug, see compute_fp_rcp_spectrum upstream for more details.
 *
 * For floating-point output:
 *     It's clipped to [0, FLT_MAX], the main reason that we avoid infinity mainly to reduce the chance of
 *     get_top3_peaks kernel producing NaN during interpolation of sub-bin.
 *
 * This is specialized for the 16x8 steering grid and one 16-ROI group.
 * cfg_rcp_p_stage0 retains the stable reciprocal-spectrum base that
 * update_part2_sv_agen_base reuses for the eight-ROI producer slices.
 *
 * @param rcp_spectrum       The reciprocal of spectrum, the C layout is float[roi_num][line_pitch],
 *                           where line_pitch = (sv_num - 1 + chess_elementsof(dvfloat) - 1)
 *                                                          / chess_elementsof(dvfloat) * chess_elementsof(dvfloat) + 1.
 * @param min_buf            Intermediate buffer, the C layout is float[roi_num].
 * @param fixed_pt_spectrum  Output buffer, UQ1.15, the C layout is uint16_t[n_roi_groups][sv_num][chess_elementsof(vushort)],
 *                           where n_roi_groups = roi_num / chess_elementsof(vushort).
 * @param float_pt_spectrum  Must be 64-byte aligned. Output buffer, floating-point spectrum saturated to [0, FLT_MAX],
 *                           the C layout is float[n_roi_groups][sv_num][chess_elementsof(dvfloat)],
 *                           where n_roi_groups = roi_num / chess_elementsof(dvfloat).
 *                           For the sake of performance, put it at different superbank from min_buf.
 * @param para               Output parameter struct to be initialized.
 */
void init_obtain_spectrum(float *const restrict rcp_spectrum, float *const restrict min_buf,
                          uint16_t *const restrict fixed_pt_spectrum, float *const restrict float_pt_spectrum,
                          Para_capon_roi_part2_config &para);

static inline void obtain_spectrum(const Para_capon_roi_part2_config &para)
{
    void obtain_spectrum_stage0(const Para_capon_roi_part2_config &);
    void obtain_spectrum_stage1(const Para_capon_roi_part2_config &);
    obtain_spectrum_stage0(para);
    obtain_spectrum_stage1(para);
}

#endif
