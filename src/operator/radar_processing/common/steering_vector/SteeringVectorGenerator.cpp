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

#include "SteeringVectorGenerator.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace pva_solutions::priv::steering_vector {

namespace {

constexpr float kPi            = 3.14159265358979323846f;
constexpr int32_t kQ15Scale    = 1 << 15;
constexpr int32_t kComplexSize = 2;

template<typename View>
float const *elementPointer(View view, int32_t index)
{
    auto const *base = reinterpret_cast<uint8_t const *>(view.data);
    return reinterpret_cast<float const *>(base + static_cast<int64_t>(index) * view.strideBytes);
}

int16_t quantizeQ15(float value)
{
    int32_t const rounded = static_cast<int32_t>(std::round(value * kQ15Scale));
    int32_t const clamped = std::clamp(rounded, static_cast<int32_t>(std::numeric_limits<int16_t>::min()),
                                       static_cast<int32_t>(std::numeric_limits<int16_t>::max()));
    return static_cast<int16_t>(clamped);
}

int16_t conjugateImaginaryQ15(int16_t value)
{
    return (value == std::numeric_limits<int16_t>::min()) ? std::numeric_limits<int16_t>::max()
                                                          : static_cast<int16_t>(-value);
}

} // namespace

void generateSteeringVectorsQ15(StridedPositionView positions, StridedFloatView azimuthBins,
                                StridedFloatView elevationBins, int16_t *output, int32_t numChannels,
                                int32_t numAzimuthBins, int32_t numElevationBins, OutputConvention convention)
{
    for (int32_t azimuthIndex = 0; azimuthIndex < numAzimuthBins; azimuthIndex++)
    {
        float const azimuthRadians = *elementPointer(azimuthBins, azimuthIndex) * kPi / 180.0f;
        for (int32_t elevationIndex = 0; elevationIndex < numElevationBins; elevationIndex++)
        {
            float const elevationRadians = *elementPointer(elevationBins, elevationIndex) * kPi / 180.0f;
            for (int32_t channelIndex = 0; channelIndex < numChannels; channelIndex++)
            {
                float const *position = elementPointer(positions, channelIndex);
                float const phase     = position[0] * std::cos(elevationRadians) * std::sin(azimuthRadians) +
                                    position[1] * std::sin(elevationRadians);
                float const angleRadians = -2.0f * kPi * phase;
                float const realFloat    = std::cos(angleRadians);
                float const imagFloat    = std::sin(angleRadians);
                int16_t const realFixed  = quantizeQ15(realFloat);
                int16_t imagFixed        = quantizeQ15(imagFloat);

                if (convention == OutputConvention::ConjugatedAfterQuantization)
                {
                    imagFixed = conjugateImaginaryQ15(imagFixed);
                }

                int32_t const linearIndex =
                    (azimuthIndex * numElevationBins + elevationIndex) * numChannels + channelIndex;
                output[kComplexSize * linearIndex]     = realFixed;
                output[kComplexSize * linearIndex + 1] = imagFixed;
            }
        }
    }
}

} // namespace pva_solutions::priv::steering_vector
