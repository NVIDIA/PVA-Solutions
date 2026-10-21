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

#ifndef RADAR_RAW_ADC_CUBE_HPP
#define RADAR_RAW_ADC_CUBE_HPP

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @brief Reads a raw ADC capture: a headerless little-endian float32 cube in C order
 *        [numChirps][numRx][numSamples] of real samples, as written by the MATLAB radar simulator.
 *
 * Header-only so the pipeline utilities and the reference tools share one parser without a link
 * dependency.
 *
 * @return The flat cube, element (chirp, rx, sample) at (chirp * numRx + rx) * numSamples + sample.
 * @throws std::runtime_error if the file cannot be opened or its size is not exactly
 *         numChirps * numRx * numSamples float32 values.
 */
inline std::vector<float> ReadRawAdcCubeFloat(const std::string &path, int32_t numChirps, int32_t numRx,
                                              int32_t numSamples)
{
    if (numChirps <= 0 || numRx <= 0 || numSamples <= 0)
    {
        throw std::runtime_error("Invalid raw ADC cube dimensions for " + path);
    }
    const size_t numValues = static_cast<size_t>(numChirps) * static_cast<size_t>(numRx) * numSamples;
    const size_t numBytes  = numValues * sizeof(float);

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open raw ADC cube: " + path);
    }
    const std::streamoff fileBytes = file.tellg();
    if (fileBytes < 0 || static_cast<size_t>(fileBytes) != numBytes)
    {
        throw std::runtime_error("Raw ADC cube " + path + " has " + std::to_string(fileBytes) + " bytes, expected " +
                                 std::to_string(numBytes) + " ([" + std::to_string(numChirps) + "][" +
                                 std::to_string(numRx) + "][" + std::to_string(numSamples) + "] float32)");
    }
    file.seekg(0, std::ios::beg);

    std::vector<float> cube(numValues);
    if (!file.read(reinterpret_cast<char *>(cube.data()), static_cast<std::streamsize>(numBytes)))
    {
        throw std::runtime_error("Could not read raw ADC cube: " + path);
    }
    return cube;
}

#endif // RADAR_RAW_ADC_CUBE_HPP
