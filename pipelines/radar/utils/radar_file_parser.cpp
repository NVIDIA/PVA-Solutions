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

#include "radar_file_parser.hpp"

#include "radar_raw_adc_cube.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

std::vector<std::vector<std::vector<int32_t>>> ReadRawDataCube(const std::string &dataFile, int32_t numChirps,
                                                               int32_t numRx, int32_t numSamples, int32_t qbits)
{
    if (qbits < 0 || qbits > 30)
    {
        throw std::runtime_error("ReadRawDataCube: qbits must be in [0, 30], got " + std::to_string(qbits));
    }
    const double dataScale = static_cast<double>(1L << qbits);

    // Throws unless the file holds exactly [numChirps][numRx][numSamples] float32 values
    const std::vector<float> fileCube = ReadRawAdcCubeFloat(dataFile, numChirps, numRx, numSamples);

    // Convert to integer with the 2^qbits scale
    std::vector<std::vector<std::vector<int32_t>>> intDataCube(
        numChirps, std::vector<std::vector<int32_t>>(numRx, std::vector<int32_t>(numSamples)));

    for (int32_t chirp = 0; chirp < numChirps; chirp++)
    {
        for (int32_t rx = 0; rx < numRx; rx++)
        {
            for (int32_t sample = 0; sample < numSamples; sample++)
            {
                const size_t index  = (static_cast<size_t>(chirp) * numRx + rx) * numSamples + sample;
                const double scaled = static_cast<double>(fileCube[index]) * dataScale;
                // Converting a non-finite or out-of-range value to int32 is undefined behavior
                if (!std::isfinite(scaled) || scaled < std::numeric_limits<int32_t>::min() ||
                    scaled > std::numeric_limits<int32_t>::max())
                {
                    throw std::runtime_error("Raw ADC cube " + dataFile + ": sample [" + std::to_string(chirp) + "][" +
                                             std::to_string(rx) + "][" + std::to_string(sample) +
                                             "] = " + std::to_string(fileCube[index]) + " does not fit in int32 at Q" +
                                             std::to_string(qbits));
                }
                intDataCube[chirp][rx][sample] = static_cast<int32_t>(scaled);
            }
        }
    }

    return intDataCube;
}
