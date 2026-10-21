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

#include "cfar_processing_opt.hpp"

#include "radar_cfar_test_ref.h"

#include <nvcv/Tensor.hpp>
#include <nvcv/TensorData.hpp>

#include <unistd.h>

#include <cstring>
#include <random>
#include <set>

// Static storage for magnitude data (needed for checkagainstRadarCFARRef to print detection values)
static uint32_t *g_magnitudeData = nullptr;
static int32_t g_width           = 0;
static int32_t g_height          = 0;

/**
 * @brief Structure to specify a target position for debugging
 *
 * Used internally by getDebugTargets() for fixed target placement.
 */
struct TargetPosition
{
    int row;           ///< Row position of the target
    int col;           ///< Column position of the target
    int targetValue;   ///< Value at peak (use 0 for default: 2 * CFAR_FIXED_POINT_SCALE)
    int neighborValue; ///< Value at neighbors (use 0 for default: targetValue - 1)
};

/**
 * @brief Insert a single target with 4-connected neighbors
 *
 * @param data Magnitude data buffer
 * @param width Width of data
 * @param height Height of data
 * @param row Target row position
 * @param col Target column position
 * @param targetValue Peak value for the target
 * @param neighborValue Value for 4-connected neighbors
 * @param useLargeValues True to generate large magnitude values, false for normal values
 * @return true if target was inserted, false if out of bounds
 */
static bool insertTargetWithNeighbors(uint32_t *data, int32_t width, int32_t height, int row, int col, int targetValue,
                                      int neighborValue, bool useLargeValues)
{
    // Validate position
    if (row < 0 || row >= height || col < 0 || col >= width)
    {
        return false;
    }

    // Set target peak value
    data[row * width + col] = useLargeValues ? INT_MAX : targetValue;

    // Set 4-connected neighbors (up, down, left, right)
    const int neighbors[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    for (int n = 0; n < 4; n++)
    {
        int r = row + neighbors[n][0];
        int c = col + neighbors[n][1];

        // Bounds check
        if (r >= 0 && r < height && c >= 0 && c < width)
        {
            data[r * width + c] = useLargeValues ? INT_MAX : neighborValue;
        }
    }

    return true;
}

/**
 * @brief Generate random noise background
 *
 * @param data Magnitude data buffer
 * @param width Width of data
 * @param height Height of data
 * @param gen Random number generator
 * @param useLargeValues True to generate large magnitude values, false for normal values
 */
static void generateNoiseBackground(uint32_t *data, int32_t width, int32_t height, std::mt19937 &gen,
                                    bool useLargeValues = false)
{
    std::uniform_real_distribution<float> dis(0.0f, 0.1f);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            float noise = dis(gen);
            data[i * width + j] =
                useLargeValues ? (INT_MAX / width) : static_cast<uint32_t>(noise * CFAR_FIXED_POINT_SCALE);
        }
    }
}

/**
 * @brief Generate random targets with neighbors
 *
 * @param data Magnitude data buffer
 * @param width Width of data
 * @param height Height of data
 * @param gen Random number generator
 * @param targetCount Number of targets to generate
 * @param targetValue Peak value for targets
 * @param neighborValue Value for 4-connected neighbors
 * @param useLargeValues True to generate large magnitude values, false for normal values
 */
static void generateRandomTargets(uint32_t *data, int32_t width, int32_t height, std::mt19937 &gen, int targetCount,
                                  int targetValue, int neighborValue, bool useLargeValues)
{
    const int edgeMargin = 0;
    std::uniform_int_distribution<int> dist_row(edgeMargin, height - edgeMargin - 1);
    std::uniform_int_distribution<int> dist_col(edgeMargin, width - edgeMargin - 1);

    if (CFAR_DEBUG_ENABLED)
    {
        printf("Using random target placement:\n");
    }
    for (int t = 0; t < targetCount; t++)
    {
        int target_row = dist_row(gen);
        int target_col = dist_col(gen);

        insertTargetWithNeighbors(data, width, height, target_row, target_col, targetValue, neighborValue,
                                  useLargeValues);

        if (CFAR_DEBUG_ENABLED)
        {
            printf("  Target %d: (row=%d, col=%d) value=%d, neighbors=%d\n", t, target_row, target_col, targetValue,
                   neighborValue);
        }
    }
}

/**
 * @brief Generate fixed targets for debugging
 *
 * @param data Magnitude data buffer
 * @param width Width of data
 * @param height Height of data
 * @param targets Vector of target positions
 * @param defaultTargetValue Default peak value (used if target.targetValue is 0)
 * @param defaultNeighborValue Default neighbor value (used if target.neighborValue is 0)
 * @param useLargeValues True to generate large magnitude values, false for normal values
 */
static void generateFixedTargets(uint32_t *data, int32_t width, int32_t height,
                                 const std::vector<TargetPosition> &targets, int defaultTargetValue,
                                 int defaultNeighborValue, bool useLargeValues)
{
    printf("Using %zu fixed target position(s) for debugging:\n", targets.size());

    for (size_t t = 0; t < targets.size(); t++)
    {
        const TargetPosition &target = targets[t];
        int targetValue              = (target.targetValue > 0) ? target.targetValue : defaultTargetValue;
        int neighborValue            = (target.neighborValue > 0) ? target.neighborValue : (targetValue - 1);

        bool inserted = insertTargetWithNeighbors(data, width, height, target.row, target.col, targetValue,
                                                  neighborValue, useLargeValues);

        if (inserted)
        {
            printf("  Target %zu: (row=%d, col=%d) value=%d, neighbors=%d\n", t, target.row, target.col, targetValue,
                   neighborValue);
        }
        else
        {
            printf("  WARNING: Target %zu at (row=%d, col=%d) is out of bounds [%d x %d], skipped\n", t, target.row,
                   target.col, height, width);
        }
    }
}

/**
 * @brief Get debug target configuration (modify this for debugging specific scenarios)
 *
 * @return Vector of fixed target positions, or empty vector for random placement
 */
static std::vector<TargetPosition> getDebugTargets()
{
    // ========== DEBUG CONFIGURATION ==========
    // Set to true to use fixed target positions for debugging
    bool useDebugTargets = false;

    if (useDebugTargets)
    {
        // Configure specific target positions for debugging
        return {
            {1, 1, 32768, 32767},
            // Add more targets as needed:
            // {100, 15, 0, 0},        // Another target with default values
            // {200, 20, 0, 0},        // Another target with default values
        };
    }
    // =========================================

    return {}; // Empty vector = use random placement
}

int32_t generateRadarCFARInput(NVCVTensorHandle tensorHandle, bool useLargeValues)
{
    // Extract tensor data
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);
    uint32_t *data = nullptr;
    CupvaMemGetHostPointer((void **)&data, (void *)tensorData.buffer.strided.basePtr);
    int32_t height = tensorData.shape[0];
    int32_t width  = tensorData.shape[1];

    // Store for later use in checkagainstRadarCFARRef
    g_magnitudeData = data;
    g_width         = width;
    g_height        = height;

    // Initialize random generator
    bool fixedSeed = true;
    std::random_device rd;
    std::vector<uint32_t> seeds = {12345};
    uint32_t seed               = fixedSeed ? seeds[0] : rd();
    std::mt19937 gen(seed);
    if (CFAR_DEBUG_ENABLED)
    {
        printf("Using seed: %u\n", seed);
    }

    // Clear and generate noise background
    memset(data, 0, height * width * sizeof(uint32_t));
    generateNoiseBackground(data, width, height, gen, useLargeValues);

    // Generate targets (fixed or random)
    const int defaultTargetValue             = 2 * CFAR_FIXED_POINT_SCALE;
    const int defaultNeighborValue           = defaultTargetValue - 1;
    std::vector<TargetPosition> debugTargets = getDebugTargets();

    if (!debugTargets.empty())
    {
        generateFixedTargets(data, width, height, debugTargets, defaultTargetValue, defaultNeighborValue,
                             useLargeValues);
    }
    else
    {
        const int targetCount = 50;
        generateRandomTargets(data, width, height, gen, targetCount, defaultTargetValue, defaultNeighborValue,
                              useLargeValues);
    }

    return 0;
}

int32_t checkagainstRadarCFARRef(NVCVTensorHandle refTensorHandle, NVCVTensorHandle outTensorHandle,
                                 NVCVTensorHandle detectionCountTensorHandle,
                                 NVCVTensorHandle detectionCountOutTensorHandle)
{
    NVCVTensorData refTensorData;
    nvcvTensorExportData(refTensorHandle, &refTensorData);
    int32_t *refData = nullptr;
    CupvaMemGetHostPointer((void **)&refData, (void *)refTensorData.buffer.strided.basePtr);

    NVCVTensorData outTensorData;
    nvcvTensorExportData(outTensorHandle, &outTensorData);
    int32_t *outData = nullptr;
    CupvaMemGetHostPointer((void **)&outData, (void *)outTensorData.buffer.strided.basePtr);

    /* Export tensor data for detection count tensor */
    NVCVTensorData detectionCountTensorData;
    nvcvTensorExportData(detectionCountTensorHandle, &detectionCountTensorData);
    int32_t *detectionCountData = nullptr;
    CupvaMemGetHostPointer((void **)&detectionCountData, (void *)detectionCountTensorData.buffer.strided.basePtr);

    /* Export tensor data for detection count out tensor */
    NVCVTensorData detectionCountOutTensorData;
    nvcvTensorExportData(detectionCountOutTensorHandle, &detectionCountOutTensorData);
    int32_t *detectionCountOutData = nullptr;
    CupvaMemGetHostPointer((void **)&detectionCountOutData, (void *)detectionCountOutTensorData.buffer.strided.basePtr);
    int32_t height       = refTensorData.shape[0];
    int32_t width        = refTensorData.shape[1];
    int32_t mismatch     = 0;
    int32_t refPeakCount = *detectionCountData;
    int32_t dstPeakCount = *detectionCountOutData;

    // Use pair structure for cleaner sorting
    struct Detection
    {
        int32_t y;
        int32_t x;
        bool operator<(const Detection &other) const
        {
            if (y != other.y)
                return y < other.y;
            return x < other.x;
        }
    };

    // Extract reference detections into vector
    std::vector<Detection> refDetections;
    for (int i = 0; i < refPeakCount; i++)
    {
        refDetections.push_back({refData[i * width], refData[i * width + 1]});
    }

    // Extract output detections into vector
    std::vector<Detection> dstDetections;
    for (int i = 0; i < dstPeakCount; i++)
    {
        dstDetections.push_back({outData[i * width], outData[i * width + 1]});
    }

    // Sort both lists
    std::sort(refDetections.begin(), refDetections.end());
    std::sort(dstDetections.begin(), dstDetections.end());

    if (CFAR_DEBUG_ENABLED)
    {
        if (refDetections.size() != dstDetections.size())
        {
            printf("Number of detections mismatch: ref = %zu, pva = %zu\n", refDetections.size(), dstDetections.size());
        }
        else
        {
            printf("Number of detections match: ref = %zu, pva = %zu\n", refDetections.size(), dstDetections.size());
        }
    }

    // Find matches between ref and dst by searching
    // This must run even when debug is false to calculate mismatch for pass/fail
    // Use actual vector sizes to avoid out-of-bounds access
    std::set<size_t> matchedDst; // Track which dst detections have been matched
    int matchCount = 0;

    if (CFAR_DEBUG_ENABLED)
    {
        printf("\n=== MATCHING DETECTIONS ===\n");
    }

    for (size_t i = 0; i < refDetections.size(); i++)
    {
        bool found = false;
        for (size_t j = 0; j < dstDetections.size(); j++)
        {
            if (matchedDst.count(j))
                continue; // Already matched

            if (refDetections[i].x == dstDetections[j].x && refDetections[i].y == dstDetections[j].y)
            {
                // Found match
                if (CFAR_DEBUG_ENABLED)
                {
                    uint32_t value = 0;
                    if (g_magnitudeData && refDetections[i].y >= 0 && refDetections[i].y < g_height &&
                        refDetections[i].x >= 0 && refDetections[i].x < g_width)
                    {
                        value = g_magnitudeData[refDetections[i].y * g_width + refDetections[i].x];
                    }

                    if (matchCount < 10)
                    {
                        printf("  ✓ Match: (x=%d, y=%d) val=%u\n", refDetections[i].x, refDetections[i].y, value);
                    }
                }
                matchedDst.insert(j);
                found = true;
                matchCount++;
                break;
            }
        }

        if (!found)
        {
            // Only in reference - this is a mismatch
            mismatch++;
            if (CFAR_DEBUG_ENABLED)
            {
                uint32_t value = 0;
                if (g_magnitudeData && refDetections[i].y >= 0 && refDetections[i].y < g_height &&
                    refDetections[i].x >= 0 && refDetections[i].x < g_width)
                {
                    value = g_magnitudeData[refDetections[i].y * g_width + refDetections[i].x];
                }
                printf("  ✗ ONLY IN REF: (x=%d, y=%d) val=%u\n", refDetections[i].x, refDetections[i].y, value);
            }
        }
    }

    // Find detections only in PVA output (extra detections)
    size_t extraPvaDetections = dstDetections.size() - matchedDst.size();
    if (CFAR_DEBUG_ENABLED && extraPvaDetections > 0)
    {
        printf("\n=== EXTRA PVA DETECTIONS (KPI) ===\n");
        for (size_t j = 0; j < dstDetections.size(); j++)
        {
            if (!matchedDst.count(j))
            {
                uint32_t value = 0;
                if (g_magnitudeData && dstDetections[j].y >= 0 && dstDetections[j].y < g_height &&
                    dstDetections[j].x >= 0 && dstDetections[j].x < g_width)
                {
                    value = g_magnitudeData[dstDetections[j].y * g_width + dstDetections[j].x];
                }
                printf("  (x=%d, y=%d) val=%u\n", dstDetections[j].x, dstDetections[j].y, value);
            }
        }
    }

    // Print summary
    size_t missedRefDetections = refDetections.size() - matchCount;
    if (CFAR_DEBUG_ENABLED)
    {
        printf("\n=== SUMMARY ===\n");
    }
    if (CFAR_DEBUG_ENABLED)
    {
        printf("Matched: %d\n", matchCount);
        printf("Only in REF (missed): %zu\n", missedRefDetections);
        printf("Only in PVA (extra): %zu\n", extraPvaDetections);
    }

    // Only fail if PVA missed detections that reference found
    if (mismatch > 0)
    {
        printf("CFAR FAIL: %d reference detections not found in PVA output\n", mismatch);
        return -1;
    }

    // NOTE: Extra PVA detections (false positives) are considered test failures.
    // They indicate the PVA CFAR is detecting targets that the reference does not,
    // which could lead to downstream processing errors and wasted computational resources.
    // WAR: Currently ignoring extra PVA detections and returning success (reported as KPI only).
    if (extraPvaDetections > 0)
    {
        printf("CFAR: PVA has %zu extra detections\n", extraPvaDetections);
    }

    return 0;
}