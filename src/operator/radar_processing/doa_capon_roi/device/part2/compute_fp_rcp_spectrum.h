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

#ifndef COMPUTE_FP_RCP_SPECTRUM_HEADER_GUARD
#define COMPUTE_FP_RCP_SPECTRUM_HEADER_GUARD

#include <cupva_device.h> /* defines `restrict` macro so part-2 prototypes parse under native g++ */
#include <stdint.h>

struct Para_capon_roi_part2_config;

/**
 * @brief Initializes parameters for compute_fp_rcp_spectrum.
 *
 * Computes ||x||^2 = sum(x_re[i]^2) + sum(x_im[i]^2) for each steering vector using IEEE float32
 * FMA, where x = L_inv @ a is the output of compute_L_inv_times_steering_matrix.
 * This is the reciprocal of the Capon spectrum: 1/P_capon = a^H R^{-1} a = ||L^{-1} a||^2.
 *
 * This is specialized for 16-element vectors, 128 steering positions, and an
 * eight-ROI steering tile. Input and output configs use dummy bases; the hot
 * phase patches them before first use. The stable reciprocal-spectrum base is
 * recovered from obtain_spectrum's input config.
 *
 * @param para              Output parameter struct to be initialized.
 */
void init_compute_fp_rcp_spectrum(Para_capon_roi_part2_config &para);

void compute_fp_rcp_spectrum(const Para_capon_roi_part2_config &para);

#endif
