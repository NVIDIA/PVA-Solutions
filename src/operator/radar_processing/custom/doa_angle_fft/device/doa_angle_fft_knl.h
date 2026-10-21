/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DOA_ANGLE_FFT_KNL_H
#define DOA_ANGLE_FFT_KNL_H

// Include angle estimation header for doa_angle_fft_params_t definition and angle estimation functions
#include "doa_angle_estimation_knl.h"

/**
 * @brief Configure the 2D window and antenna re-mapping of one detection into its band of the packed grid
 *
 * @param src Input snapshot slot
 * @param dst Packed azimuth grid
 * @param win 2D window coefficients
 * @param cfgs [out] AGEN configurations (3)
 * @param param FFT sizes and shift
 * @param groups Detections packed into the grid (G)
 * @param slot Band of this detection, in [0, groups)
 * @return Next ping-pong iteration
 */
int32_t windowing_init_azimuth(int32_t *src, int32_t *dst, int32_t *win, AgenCFG *cfgs, doa_angle_fft_params_t *param,
                               int32_t groups, int32_t slot);

/**
 * @brief Configure the azimuth radix-4 stages and the transposing digit reverse over all packed bands
 *
 * @param ping, pong, tmp FFT working buffers; windowing leaves the packed grid in pong
 * @param twiddles Twiddle tables [azimuth, elevation]
 * @param cfgs [out] AGEN configurations (at most FFT_CFGS_PER_SET)
 * @param param FFT sizes, factorization and shifts
 * @param groups Detections packed into the grid (G)
 * @return 0
 */
int32_t fft_batched_init_azimuth(int32_t *ping, int32_t *pong, int32_t *tmp, int32_t **twiddles, AgenCFG *cfgs,
                                 doa_angle_fft_params_t *param, int32_t groups);

/**
 * @brief Initialize elevation FFT processing pipeline
 *
 * Configures AGEN for:
 * - 1D FFT in vertical direction (elevation)
 * - Power spectrum computation (magnitude squared), tracking its maximum
 *
 * @param ping, pong, tmp FFT working buffers, offset to this detection's band of the transposed grid
 * @param twiddles Twiddle tables, starting at the elevation table
 * @param cfgs [out] AGEN configurations (at most FFT_CFGS_PER_SET)
 * @param param FFT sizes, factorization and shifts
 * @return 0
 */
int32_t fft_batched_init_elevation(int32_t *ping, int32_t *pong, int32_t *tmp, int32_t **twiddles, AgenCFG *cfgs,
                                   doa_angle_fft_params_t *param);

/**
 * @brief Run the windowing configured by windowing_init_azimuth for one detection
 *
 * @param cfgs AGEN configurations from windowing_init_azimuth
 * @param param FFT sizes
 */
void windowing_exec_azimuth(AgenCFG *cfgs, doa_angle_fft_params_t *param);

/**
 * @brief Run the azimuth FFT configured by fft_batched_init_azimuth over all packed bands
 *
 * @param cfgs AGEN configurations from fft_batched_init_azimuth
 * @param param FFT sizes and factorization
 * @param groups Detections packed into the grid (G)
 * @return 0
 */
int32_t fft_batched_exec_azimuth(AgenCFG *cfgs, doa_angle_fft_params_t *param, int32_t groups);

/**
 * @brief Execute elevation FFT
 *
 * Performs:
 * 1. 1D FFT in vertical direction
 * 2. Power spectrum computation (magnitude squared)
 *
 * @param cfgs AGEN configurations from fft_batched_init_elevation
 * @param param FFT sizes and factorization
 * @return Maximum power value from FFT
 */
float fft_batched_exec_elevation(AgenCFG *cfgs, doa_angle_fft_params_t *param);

#endif