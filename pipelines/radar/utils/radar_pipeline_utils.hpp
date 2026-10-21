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

#pragma once

#include "radar_file_parser.hpp"

#include "nvcv/Tensor.h"

#include <cstdint>
#include <string>
#include <vector>

// Structure to hold radar configuration parameters
struct RadarConfig
{
    std::vector<float> ddmDopplerOffsets;
    double carrierFrequency;
    int32_t numAzimuthReceiveAntennas;
    int32_t numElevationReceiveAntennas;
    int32_t numTransmitAntennas;
    int32_t numSamples;
    int32_t numChirps;
    int32_t ddmRepeatFoldFactor;
    float rangeResolution;
    float sweepTime;
    std::vector<float> mountingAngle;

    // RV decoupling parameters
    float bandwidth;
    float chirpDwellTime;
    float deltaF;
    float fs;
    float PRI;
    float deltaV;
    float contR;
    float contVFast;
    float contVSlow;
};

// Function to get assets directory from command line arguments
std::string GetAssetsDirectory(int32_t argc, char **argv);

// Function to find files with a certain prefix in a directory
std::vector<std::string> FindFilesWithPrefix(const std::string &directoryPath, const std::string &prefix);

/**
 * Load steering vectors from a raw binary file (int16_t, complex Q15) into a 1D vector.
 * The number of elements is determined from the file size (file size must be a multiple of sizeof(int16_t)).
 * File layout: [numAzimuthBins][numElevationBins][numTx][numRx] with 2 int16 per complex (real, imag).
 *
 * @param rawFilePath Path to raw binary file
 * @return std::vector<int16_t> with size = file size / sizeof(int16_t)
 * @throws std::runtime_error if file cannot be opened or file size is not a multiple of sizeof(int16_t)
 */
std::vector<int16_t> LoadSteeringVectorsFromRawFile(const std::string &rawFilePath);

// Print target processing results
void PrintTargetProcessingResults(NVCVTensorHandle targetCountTensorHandle, NVCVTensorHandle targetListTensorHandle);

// Function to parse radar configuration from JSON file
RadarConfig ParseRadarConfig(const std::string &configFile);
