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

#include <PvaOperatorTypes.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

// Debug flag for verbose output
constexpr bool CFAR_DEBUG_ENABLED = false;

// Fixed-point scaling factor for SQ19.14 (2^14 = 16384) float to int32_t conversion
constexpr int32_t CFAR_QBITS             = 14;
constexpr int32_t CFAR_FIXED_POINT_SCALE = (1 << CFAR_QBITS);
// Structure to hold reference cell counts
struct CfarCellCounts
{
    std::vector<uint32_t> totalCounts;
    std::vector<uint32_t> trailingCounts;
    std::vector<uint32_t> leadingCounts;
};

// CFAR detection result structure
struct CfarResultFixedPoint
{
    std::vector<std::vector<bool>> detectionMap;
    std::vector<std::pair<int32_t, int32_t>> detectionList;
    std::vector<std::vector<uint32_t>> horizontalThresholdMap;
    std::vector<std::vector<uint32_t>> verticalThresholdMap;
};
// Fixed-point configuration
constexpr int32_t DATA_QBITS = 14;
constexpr int32_t DATA_SCALE = (1UL << DATA_QBITS);

// Fixed-point 1D Cell Averaging CFAR detector with int32_t arithmetic
CfarResultFixedPoint CfarCa1dOptFixedPoint(const std::vector<std::vector<uint32_t>> &inputData,
                                           std::vector<int32_t> guardCells     = {2, 2},
                                           std::vector<int32_t> referenceCells = {6, 6}, float probabilityFA = 1e-6f,
                                           std::vector<float> thresholdFactor      = {},
                                           std::vector<std::string> borderHandling = {"valid", "valid"},
                                           bool peakGrouping = true, PVARadarCFARType algorithm = PVA_CFAR_CA);