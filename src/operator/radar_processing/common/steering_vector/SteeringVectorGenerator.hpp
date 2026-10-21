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

#ifndef PVA_SOLUTIONS_STEERING_VECTOR_GENERATOR_HPP
#define PVA_SOLUTIONS_STEERING_VECTOR_GENERATOR_HPP

#include <cstdint>

namespace pva_solutions::priv::steering_vector {

/**
 * @brief View of strided two-dimensional antenna positions.
 *
 * Positions are expressed in wavelengths. Each position contains adjacent x
 * and y float values. @c strideBytes advances from one channel's x value to
 * the next channel's x value.
 */
struct StridedPositionView
{
    float const *data;
    int64_t strideBytes;
};

/**
 * @brief View of strided scalar float values.
 */
struct StridedFloatView
{
    float const *data;
    int64_t strideBytes;
};

/**
 * @brief Selects the complex-value convention stored in the output.
 */
enum class OutputConvention
{
    /** Store the quantized value of exp(-j * 2 * pi * phase). */
    Direct,

    /**
     * Quantize exp(-j * 2 * pi * phase), then form its conjugate by saturated
     * negation of the imaginary component. The order matters: a quantized
     * -32768 maps to +32767.
     */
    ConjugatedAfterQuantization,
};

/**
 * @brief Generate packed Q15 steering vectors directly from positions and angle bins.
 *
 * For every channel and angle bin, this function converts the angles from
 * degrees to radians and computes:
 *
 * @code
 * phase = x * cos(elevation) * sin(azimuth) + y * sin(elevation)
 * directSteering = exp(-j * 2 * pi * phase)
 * @endcode
 *
 * Each real and imaginary component is multiplied by 2^15, rounded to nearest
 * with halfway cases away from zero, and clamped to the int16_t range.
 *
 * A channel is one entry in @p positions and the corresponding complex slot in
 * @p output. The output channel axis follows the position order exactly.
 *
 * The computation is evaluated in float32.
 *
 * Position and angle values must be finite. This low-level function does not
 * validate pointers, strides, dimensions, or input values; callers must
 * provide valid views, positive dimensions, and sufficient packed output
 * storage.
 *
 * @param[in] positions Antenna positions in wavelengths and in channel order.
 * @param[in] azimuthBins Azimuth angles in degrees.
 * @param[in] elevationBins Elevation angles in degrees.
 * @param[out] output Receives
 *                   `2 * numAzimuthBins * numElevationBins * numChannels`
 *                   int16_t values ordered
 *                   `[azimuth][elevation][channel][real, imaginary]`.
 * @param[in] numChannels Number of antenna or virtual-antenna channels.
 * @param[in] numAzimuthBins Number of azimuth bins.
 * @param[in] numElevationBins Number of elevation bins.
 * @param[in] convention Complex-value convention stored in @p output.
 */
void generateSteeringVectorsQ15(StridedPositionView positions, StridedFloatView azimuthBins,
                                StridedFloatView elevationBins, int16_t *output, int32_t numChannels,
                                int32_t numAzimuthBins, int32_t numElevationBins, OutputConvention convention);

} // namespace pva_solutions::priv::steering_vector

#endif // PVA_SOLUTIONS_STEERING_VECTOR_GENERATOR_HPP
