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

#ifndef OBTAIN_FIXED_PT_STEERING_MATRIX_HEADER_GUARD
#define OBTAIN_FIXED_PT_STEERING_MATRIX_HEADER_GUARD

#include <cupva_device.h> /* defines `restrict` macro so part-2 prototypes parse under native g++ */
#include <stdint.h>

struct Para_capon_roi_part2_config;

/**
 * @brief Initializes parameters for shifting fixed-point steering vectors to SQ9.22.
 *
 * Shifts the input fixed-point steering vectors in-place to the Q22 format used by Part 2.
 * Steering vectors are split into two channels by groups of 8:
 *   sv_mat_0: even groups (indices 0-7, 16-23, ...)
 *   sv_mat_1: odd groups  (indices 8-15, 24-31, ...)
 *
 * Although this kernel does not impose a specific ordering on sv_mat — only that the input and output
 * use the same layout — the downstream kernels including extract_3x3filtered_spectrum_in16az8el_out14az6el
 * and get_top3_peaks assume that the steering vector indices are in elevation-major order (azimuth varies
 * fastest, i.e., index = el * num_az + az).
 *
 * This is specialized for 16-element vectors, 128 steering positions, and an
 * eight-ROI channel tile. Both steering AGEN bases are initialized with dummy
 * bases and patched by update_part2_sv_agen_base before first use.
 *
 * @param steering_qbits Number of fractional bits in the input steering vectors.
 * @param para      Output parameter struct to be initialized.
 */

void init_shift_fixed_pt_steering_matrix(const uint16_t steering_qbits, Para_capon_roi_part2_config &para);
void shift_fixed_pt_steering_matrix(const Para_capon_roi_part2_config &para);

#endif
