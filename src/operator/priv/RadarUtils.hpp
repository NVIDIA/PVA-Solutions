/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @file RadarUtils.hpp
 * @brief Utility functions for radar FFT processing, including FFT size factorization
 *        and twiddle factor generation for mixed-radix FFT computation.
 */

#ifndef RADAR_UTILS_HPP
#define RADAR_UTILS_HPP

#include "RadarDefinitions.h"

#include <vector>

namespace pva_solutions::priv {

/**
 * @brief Factorizes an FFT size into powers of 5, 3, and 2 for mixed-radix FFT computation.
 *
 * This function decomposes the given FFT size @p n into a combination of radix-5, radix-3,
 * and radix-4/radix-2 stages. The factorization follows the order: radix-5 stages first,
 * then radix-3 stages, and finally radix-4/radix-2 stages for the remaining power-of-2 factor.
 *
 * @param[in] n                 The FFT size to factorize. Must be in range [2, 1024].
 *                              For n=1024, transposeOutput and hasPostProcessing must be false.
 * @param[in] transposeOutput   If true, configures strides for transposed output (pitch=2).
 *                              If false, uses standard output pitch (pitch=16).
 * @param[in] hasPostProcessing If true, allocates additional twiddle space for post-processing
 *                              coefficients.
 * @param[in] outGroupsPerTile  Range-bin groups per output tile; scales the non-transposed pitch to
 *                              16 * outGroupsPerTile.
 * @param[in] outPitchOverride  If > 0, the output pitch for both transposed and non-transposed output,
 *                              for a consumer that reads a padded row.
 *
 * @return FFTFactorization structure containing:
 *         - @c n: The original FFT size
 *         - @c groups: Array of RadixGroup structures for radix-3, radix-4, and radix-5
 *         - @c iters: Iteration counts for each stage
 *         - @c strides: Stride values for each stage
 *         - @c twiddleSize: Total twiddle buffer size required
 *         - @c twiddlePPSizePerCoef: Post-processing twiddle size per coefficient
 *         - @c twiddleOffsets: Offsets into twiddle buffer for each radix group
 *         - @c isValid: 1 if factorization is valid, 0 otherwise
 *
 * @note The function returns an invalid factorization (isValid=0) if:
 *       - n < 2 or n > 1024
 *       - n=1024 with transposeOutput=true or hasPostProcessing=true
 *       - n is not factorizable into powers of 5, 3, and 2
 *       - Total number of stages exceeds 6
 */
FFTFactorization FactorizeFFTSize(int32_t n, bool transposeOutput = false, bool hasPostProcessing = false,
                                  int32_t outGroupsPerTile = 1, int32_t outPitchOverride = 0);

/**
 * @brief Generates twiddle factors for intra-stage processing of a specified radix.
 *
 * This function extracts and arranges twiddle factors from an interleaved source array
 * for use in mixed-radix FFT computation. The twiddle factors are organized based on
 * the radix group configuration in the factorization structure.
 *
 * @param[out] twiddle              Pointer to the output buffer where twiddle factors will be stored.
 *                                  Must have sufficient space for (n/radix) * (radix-1) * 2 int32_t values.
 * @param[in]  twiddle_interleaved  Reference to the source vector containing interleaved twiddle factors
 *                                  in complex format (real, imag pairs).
 * @param[in]  factorization        Reference to the FFTFactorization structure containing FFT configuration.
 * @param[in]  radix                The radix value (3, 4, or 5) for which to generate twiddle factors.
 *
 * @note The twiddle factors are stored as interleaved complex pairs (real, imaginary).
 *
 * @throws std::runtime_error If an unsupported radix value is provided.
 */
void generateComplexFFTTwiddleIntraStage(int32_t *twiddle, std::vector<int32_t> &twiddle_interleaved,
                                         FFTFactorization &factorization, int32_t radix);

/**
 * @brief Generates twiddle factors specifically for radix-3 FFT stages.
 *
 * This function generates the twiddle factors needed for a radix-3 butterfly operation.
 * It extracts 2 complex twiddle factors (for indices 1 and 2) from the interleaved source.
 *
 * @param[out] twiddle              Pointer to the output buffer where twiddle factors will be stored.
 *                                  Must have space for at least 4 int32_t values (2 complex pairs).
 * @param[in]  twiddle_interleaved  Reference to the source vector containing interleaved twiddle factors
 *                                  in complex format (real, imag pairs).
 * @param[in]  factorization        Reference to the FFTFactorization structure containing FFT configuration.
 *
 * @note The stride used is n/3, where n is the FFT size from the factorization structure.
 */
void generateComplexFFTTwiddleRadix3(int32_t *twiddle, std::vector<int32_t> &twiddle_interleaved,
                                     FFTFactorization &factorization);

/**
 * @brief Generates twiddle factors specifically for radix-5 FFT stages.
 *
 * This function generates the twiddle factors needed for a radix-5 butterfly operation.
 * It extracts 4 complex twiddle factors (for indices 1, 2, 3, and 4) from the interleaved source.
 *
 * @param[out] twiddle              Pointer to the output buffer where twiddle factors will be stored.
 *                                  Must have space for at least 8 int32_t values (4 complex pairs).
 * @param[in]  twiddle_interleaved  Reference to the source vector containing interleaved twiddle factors
 *                                  in complex format (real, imag pairs).
 * @param[in]  factorization        Reference to the FFTFactorization structure containing FFT configuration.
 *
 * @note The stride used is n/5, where n is the FFT size from the factorization structure.
 */
void generateComplexFFTTwiddleRadix5(int32_t *twiddle, std::vector<int32_t> &twiddle_interleaved,
                                     FFTFactorization &factorization);

/**
 * @brief Generates twiddle factors for FFT post-processing with coefficients A and B.
 *
 * This function computes two sets of twiddle coefficients used for post-processing
 * in real-valued FFT computation. The coefficients are based on the formula:
 * - Coefficient A: 0.5 * (1 - sin(angle)), 0.5 * (-cos(angle))
 * - Coefficient B: 0.5 * cos(angle), 0.5 * (1 + sin(angle))
 *
 * where angle = 2π * i / (2n) for each index i.
 *
 * @param[out] twiddle        Pointer to the output buffer where twiddle factors will be stored.
 *                            Coefficient A is stored as int32_t pairs, coefficient B follows
 *                            at offset twiddlePPSizePerCoef as int16_t pairs.
 * @param[in]  qbitsTwiddle   Number of fractional bits for fixed-point representation of coefficient A.
 * @param[in]  qbitsTrunc     Number of bits to truncate for coefficient B scaling.
 *                            Coefficient B uses min(qbitsTwiddle - qbitsTrunc, qbitsTwiddle) fractional bits.
 * @param[in]  factorization  Reference to the FFTFactorization structure containing FFT configuration,
 *                            including the FFT size (n) and twiddlePPSizePerCoef offset.
 *
 * @note Coefficient A is stored as 32-bit fixed-point (real, imag) pairs.
 * @note Coefficient B is stored as 16-bit fixed-point (imag, real) pairs with reversed order.
 */
void generateComplexFFTTwiddlePostProcess(int32_t *twiddle, int32_t qbitsTwiddle, int32_t qbitsTrunc,
                                          FFTFactorization &factorization);

} // namespace pva_solutions::priv

#endif // RADAR_UTILS_HPP