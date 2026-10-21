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

#ifndef RADAR_FILE_PARSER_HPP
#define RADAR_FILE_PARSER_HPP

#include <cstdint>
#include <string>
#include <vector>

/**
 * @brief Reads a headerless raw ADC cube (float32 [numChirps][numRx][numSamples], see ReadRawAdcCubeFloat) and
 *        scales it by 2^qbits into int32 [numChirps][numRx][numSamples].
 *
 * @throws std::runtime_error if the file cannot be read, its size does not match the cube, or a sample is not
 *         finite or does not fit in int32 after scaling.
 */
std::vector<std::vector<std::vector<int32_t>>> ReadRawDataCube(const std::string &dataFile, int32_t numChirps,
                                                               int32_t numRx, int32_t numSamples, int32_t qbits);

#endif // RADAR_FILE_PARSER_HPP
