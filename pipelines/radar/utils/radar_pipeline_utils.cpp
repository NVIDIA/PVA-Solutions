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

#include "radar_pipeline_utils.hpp"

#include "nlohmann/json.hpp"

#include "cupva_host.h"
#include "nvcv/Tensor.h"
#include "nvcv/TensorShape.h"

#include <dirent.h>   // For DIR, opendir, readdir, closedir
#include <sys/stat.h> // For stat, struct stat, S_ISREG

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <type_traits>

using json = nlohmann::json;

// Function to get assets directory from command line arguments
std::string GetAssetsDirectory(int32_t argc, char **argv)
{
    for (int32_t i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "-a") || !strcmp(argv[i], "--assets"))
        {
            if (i + 1 < argc)
            {
                std::string assetsDirectory = argv[++i];
                std::cout << "Using asset directory: " << assetsDirectory << std::endl;
                return assetsDirectory;
            }
            else
            {
                std::cout << "--assets option requires one argument." << std::endl;
                return "";
            }
        }
    }

    // No assets directory specified
    return "";
}

// Function to find files with a certain prefix in a directory
std::vector<std::string> FindFilesWithPrefix(const std::string &directoryPath, const std::string &prefix)
{
    std::vector<std::string> matchingFiles;

    // Check if directory exists
    DIR *dir = opendir(directoryPath.c_str());
    if (dir == nullptr)
    {
        printf("Error: Cannot open directory: %s\n", directoryPath.c_str());
        return matchingFiles;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr)
    {
        std::string fileName = entry->d_name;

        // Skip . and .. directories
        if (fileName == "." || fileName == "..")
        {
            continue;
        }

        // Check if file starts with the specified prefix
        if (fileName.substr(0, prefix.length()) == prefix)
        {
            std::string fullPath = directoryPath + "/" + fileName;

            // Check if it's a regular file
            struct stat fileStat;
            if (stat(fullPath.c_str(), &fileStat) == 0 && S_ISREG(fileStat.st_mode))
            {
                matchingFiles.push_back(fullPath);
            }
        }
    }

    closedir(dir);

    // Sort files using natural (alphanumeric) ordering
    // This ensures "radar_raw_4t4r_1.bin", "radar_raw_4t4r_2.bin", ..., "radar_raw_4t4r_10.bin" order
    std::sort(matchingFiles.begin(), matchingFiles.end(), [](const std::string &a, const std::string &b) {
        // Find the last '/' to get just the filename
        size_t aLastSlash = a.find_last_of('/');
        size_t bLastSlash = b.find_last_of('/');
        std::string aName = (aLastSlash != std::string::npos) ? a.substr(aLastSlash + 1) : a;
        std::string bName = (bLastSlash != std::string::npos) ? b.substr(bLastSlash + 1) : b;

        size_t aIdx = 0, bIdx = 0;
        while (aIdx < aName.length() && bIdx < bName.length())
        {
            // Check if both current characters are digits
            if (std::isdigit(aName[aIdx]) && std::isdigit(bName[bIdx]))
            {
                // Extract the full numeric value
                size_t aNumStart = aIdx;
                size_t bNumStart = bIdx;
                while (aIdx < aName.length() && std::isdigit(aName[aIdx])) aIdx++;
                while (bIdx < bName.length() && std::isdigit(bName[bIdx])) bIdx++;

                // Convert to integers and compare numerically
                int aNum = std::stoi(aName.substr(aNumStart, aIdx - aNumStart));
                int bNum = std::stoi(bName.substr(bNumStart, bIdx - bNumStart));

                if (aNum != bNum)
                    return aNum < bNum;
            }
            else
            {
                // Non-numeric characters, compare lexicographically
                if (aName[aIdx] != bName[bIdx])
                    return aName[aIdx] < bName[bIdx];
                aIdx++;
                bIdx++;
            }
        }
        // If one string is a prefix of the other, shorter one comes first
        return aName.length() < bName.length();
    });

    printf("Found %zu files with prefix '%s' in directory '%s'\n", matchingFiles.size(), prefix.c_str(),
           directoryPath.c_str());

    return matchingFiles;
}

// Load steering vectors from raw binary file into a 1D vector (size from file size)
std::vector<int16_t> LoadSteeringVectorsFromRawFile(const std::string &rawFilePath)
{
    std::ifstream file(rawFilePath, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open steering vectors file: " + rawFilePath);
    }

    file.seekg(0, std::ios::end);
    const std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    if (fileSize <= 0 || (fileSize % static_cast<std::streamsize>(sizeof(int16_t)) != 0))
    {
        throw std::runtime_error("Steering vectors file size must be a positive multiple of sizeof(int16_t): " +
                                 rawFilePath);
    }

    const size_t numInt16 = static_cast<size_t>(fileSize) / sizeof(int16_t);
    std::vector<int16_t> data(numInt16);
    file.read(reinterpret_cast<char *>(data.data()), fileSize);
    if (!file || file.gcount() != fileSize)
    {
        throw std::runtime_error("Failed to read steering vectors file: " + rawFilePath);
    }
    file.close();
    return data;
}

// Print target processing results
void PrintTargetProcessingResults(NVCVTensorHandle targetCountTensorHandle, NVCVTensorHandle targetListTensorHandle)
{
    NVCVTensorData targetCountData;
    if (nvcvTensorExportData(targetCountTensorHandle, &targetCountData) != NVCV_SUCCESS)
    {
        fprintf(stderr, "PrintTargetProcessingResults: failed to export target count tensor\n");
        return;
    }
    int32_t *targetCountPtr = NULL;
    if (CupvaMemGetHostPointer((void **)&targetCountPtr, (void *)targetCountData.buffer.strided.basePtr) !=
            CUPVA_ERROR_NONE ||
        targetCountPtr == NULL)
    {
        fprintf(stderr, "PrintTargetProcessingResults: failed to get host pointer for target count\n");
        return;
    }
    const int32_t numTargets = targetCountPtr[0];

    NVCVTensorData targetListData;
    if (nvcvTensorExportData(targetListTensorHandle, &targetListData) != NVCV_SUCCESS)
    {
        fprintf(stderr, "PrintTargetProcessingResults: failed to export target list tensor\n");
        return;
    }
    float *targetListPtr = NULL;
    if (CupvaMemGetHostPointer((void **)&targetListPtr, (void *)targetListData.buffer.strided.basePtr) !=
            CUPVA_ERROR_NONE ||
        targetListPtr == NULL)
    {
        fprintf(stderr, "PrintTargetProcessingResults: failed to get host pointer for target list\n");
        return;
    }

    // Get actual tensor dimensions to determine stride
    int64_t targetListShapeHW[2];
    nvcvTensorShapePermute(targetListData.layout, targetListData.shape, NVCV_TENSOR_HW, targetListShapeHW);
    const int32_t numProperties = targetListShapeHW[0]; // Should be 8
    const int32_t maxTargets    = targetListShapeHW[1]; // Tensor width (stride)

    std::cout << "Number of targets: " << numTargets << std::endl;
    std::cout << std::setw(5) << "Idx" << std::setw(18) << "Velocity(m/s)" << std::setw(12) << "Range(m)"
              << std::setw(12) << "Az(deg)" << std::setw(12) << "El(deg)" << std::setw(12) << "X(m)" << std::setw(12)
              << "Y(m)" << std::setw(12) << "Z(m)" << std::setw(14) << "Power(dB)" << std::endl;
    std::cout << std::string(109, '-') << std::endl;
    for (int32_t i = 0; i < numTargets; i++)
    {
        // Print target processing results - use maxTargets as stride, not numTargets!
        std::cout << std::setw(5) << (i + 1) // 1-based indexing
                  << std::fixed << std::setprecision(3) << std::setw(18) << targetListPtr[0 * maxTargets + i]
                  << std::setw(12) << targetListPtr[1 * maxTargets + i] << std::setw(12)
                  << targetListPtr[2 * maxTargets + i] << std::setw(12) << targetListPtr[3 * maxTargets + i]
                  << std::setw(12) << targetListPtr[4 * maxTargets + i] << std::setw(12)
                  << targetListPtr[5 * maxTargets + i] << std::setw(12) << targetListPtr[6 * maxTargets + i]
                  << std::setw(14) << targetListPtr[7 * maxTargets + i] << std::endl;
    }
    std::cout << std::string(109, '-') << std::endl;
}

// Function to parse radar configuration from JSON file
RadarConfig ParseRadarConfig(const std::string &configFile)
{
    RadarConfig config;

    std::ifstream file(configFile);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open radar config file: " + configFile);
    }

    json j;
    file >> j;

    // Parse configuration parameters
    config.ddmDopplerOffsets           = j["ddmDopplerOffsets"].get<std::vector<float>>();
    config.carrierFrequency            = j["carrierFrequency"];
    config.numAzimuthReceiveAntennas   = j["numAzimuthReceiveAntennas"];
    config.numElevationReceiveAntennas = j["numElevationReceiveAntennas"];
    config.numTransmitAntennas         = j["numTransmitAntennas"];
    config.ddmRepeatFoldFactor         = j["ddmRepeatFoldFactor"];
    config.numSamples                  = j["numSamples"];
    config.numChirps                   = j["numChirps"];
    config.rangeResolution             = j["rangeResolution"];
    config.sweepTime                   = j["sweepTime"];
    config.mountingAngle               = j["mountingAngle"].get<std::vector<float>>();

    // RV decoupling parameters - read from JSON or use defaults
    config.bandwidth      = j.value("bandwidth", 150.0e6f);       // Effective bandwidth (Hz) - 150 MHz
    config.chirpDwellTime = j.value("chirpDwellTime", 2.048e-5f); // Chirp dwell time (s)
    config.deltaF         = j.value("deltaF", 1.0e6f);            // Frequency step (Hz) - 1 MHz

    // Compute RV decoupling parameters
    const float c = 299792458.0f; // Speed of light

    // fs = numSamples / chirpDwellTime
    config.fs = static_cast<float>(config.numSamples) / config.chirpDwellTime;

    // PRI = sweepTime (Pulse Repetition Interval)
    config.PRI = config.sweepTime;

    // Denominator used in RV decoupling formulas
    float denom = config.bandwidth * config.sweepTime - config.deltaF * config.chirpDwellTime;

    // Compute coupling constants
    if (std::abs(denom) < 1e-15f)
    {
        // Avoid division by zero - use defaults
        config.contR     = 0.0f;
        config.contVFast = 0.0f;
        config.contVSlow = 0.0f;
    }
    else
    {
        // contR = c/2 * (chirpDwellTime * sweepTime / denom)
        config.contR = (c / 2.0f) * (config.chirpDwellTime * config.sweepTime / denom);

        // contVFast = c/2 / carrierFrequency * chirpDwellTime / (-denom) * deltaF
        config.contVFast =
            (c / 2.0f) / static_cast<float>(config.carrierFrequency) * config.chirpDwellTime / (-denom) * config.deltaF;

        // contVSlow = c/2 / carrierFrequency * sweepTime / (-denom) * bandwidth
        config.contVSlow =
            (c / 2.0f) / static_cast<float>(config.carrierFrequency) * config.sweepTime / (-denom) * config.bandwidth;
    }

    // deltaV = c/2 / carrierFrequency / sweepTime / numChirps * 3.6 (km/h)
    config.deltaV = (c / 2.0f) / static_cast<float>(config.carrierFrequency) / config.sweepTime /
                    static_cast<float>(config.numChirps) * 3.6f;

    return config;
}
