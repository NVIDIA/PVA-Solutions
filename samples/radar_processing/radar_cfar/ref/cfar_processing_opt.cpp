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

#include "cfar_processing_opt.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>

// Utility functions for conversion
static inline int32_t FloatToFixedPoint(float value)
{
    // Round to nearest integer for better accuracy
    return static_cast<int32_t>(std::round(value * DATA_SCALE));
}

// Helper function to create padded data for border handling - fixed-point version
std::vector<std::vector<uint32_t>> createPaddedDataFixedPoint(const std::vector<std::vector<uint32_t>> &inputData,
                                                              int32_t horizontalPadding, int32_t verticalPadding,
                                                              const std::string &horizontalPaddingType,
                                                              const std::string &verticalPaddingType)
{
    int32_t numVerticalBins   = static_cast<int32_t>(inputData.size());
    int32_t numHorizontalBins = static_cast<int32_t>(inputData[0].size());

    // Calculate padded dimensions
    int32_t paddedVerticalBins   = numVerticalBins + 2 * verticalPadding;
    int32_t paddedHorizontalBins = numHorizontalBins + 2 * horizontalPadding;

    // Initialize padded data with fixed-point values
    std::vector<std::vector<uint32_t>> paddedData(paddedVerticalBins, std::vector<uint32_t>(paddedHorizontalBins, 0));

    // Copy original data to center of padded array, converting to fixed-point
    for (int32_t v = 0; v < numVerticalBins; v++)
    {
        for (int32_t h = 0; h < numHorizontalBins; h++)
        {
            paddedData[v + verticalPadding][h + horizontalPadding] = inputData[v][h];
        }
    }

    // Handle horizontal dimension padding
    if (horizontalPaddingType == "cyclic")
    {
        // Left padding (wrap from right side)
        for (int32_t v = 0; v < paddedVerticalBins; v++)
        {
            for (int32_t h = 0; h < horizontalPadding; h++)
            {
                int32_t srcH = numHorizontalBins - horizontalPadding + h;
                if (v >= verticalPadding && v < verticalPadding + numVerticalBins)
                {
                    paddedData[v][h] = inputData[v - verticalPadding][srcH];
                }
            }
        }

        // Right padding (wrap from left side)
        for (int32_t v = 0; v < paddedVerticalBins; v++)
        {
            for (int32_t h = 0; h < horizontalPadding; h++)
            {
                int32_t destH = horizontalPadding + numHorizontalBins + h;
                if (v >= verticalPadding && v < verticalPadding + numVerticalBins)
                {
                    paddedData[v][destH] = inputData[v - verticalPadding][h];
                }
            }
        }
    }
    // For "full" method, padding is already initialized to 0 (zero padding)

    // Handle vertical dimension padding
    if (verticalPaddingType == "cyclic")
    {
        // Top padding (wrap from bottom side)
        for (int32_t v = 0; v < verticalPadding; v++)
        {
            for (int32_t h = 0; h < paddedHorizontalBins; h++)
            {
                int32_t srcV = numVerticalBins - verticalPadding + v;
                if (h >= horizontalPadding && h < horizontalPadding + numHorizontalBins)
                {
                    paddedData[v][h] = inputData[srcV][h - horizontalPadding];
                }
                else if (horizontalPaddingType == "cyclic")
                {
                    // Handle corner cases for cyclic-cyclic
                    int32_t srcH;
                    if (h < horizontalPadding)
                    {
                        srcH = numHorizontalBins - horizontalPadding + h;
                    }
                    else
                    {
                        srcH = h - horizontalPadding - numHorizontalBins;
                    }
                    paddedData[v][h] = inputData[srcV][srcH];
                }
            }
        }

        // Bottom padding (wrap from top side)
        for (int32_t v = 0; v < verticalPadding; v++)
        {
            int32_t destV = verticalPadding + numVerticalBins + v;
            for (int32_t h = 0; h < paddedHorizontalBins; h++)
            {
                if (h >= horizontalPadding && h < horizontalPadding + numHorizontalBins)
                {
                    paddedData[destV][h] = inputData[v][h - horizontalPadding];
                }
                else if (horizontalPaddingType == "cyclic")
                {
                    // Handle corner cases for cyclic-cyclic
                    int32_t srcH;
                    if (h < horizontalPadding)
                    {
                        srcH = numHorizontalBins - horizontalPadding + h;
                    }
                    else
                    {
                        srcH = h - horizontalPadding - numHorizontalBins;
                    }
                    paddedData[destV][h] = inputData[v][srcH];
                }
            }
        }
    }
    // For "full" method, padding is already initialized to 0 (zero padding)

    return paddedData;
}

// Helper function to compute horizontal reference cell counts
CfarCellCounts computeHorizontalCounts(int32_t numHorizontalBins, const std::string &horizontalPaddingType,
                                       int32_t horizontalWindowSize, int32_t horizontalGuardCells,
                                       int32_t horizontalReferenceCells, PVARadarCFARType algorithm)
{
    CfarCellCounts counts;
    counts.totalCounts.resize(numHorizontalBins);
    counts.trailingCounts.resize(numHorizontalBins);
    counts.leadingCounts.resize(numHorizontalBins);

    for (int32_t horizontalIndex = 0; horizontalIndex < numHorizontalBins; horizontalIndex++)
    {
        if (horizontalPaddingType == "cyclic")
        {
            if (algorithm == PVA_CFAR_CA)
            {
                counts.totalCounts[horizontalIndex]    = 2 * horizontalReferenceCells;
                counts.trailingCounts[horizontalIndex] = horizontalReferenceCells; //Can be removed
                counts.leadingCounts[horizontalIndex]  = horizontalReferenceCells; //Can be removed
            }
            else /* GO and SO algorithm */
            {
                counts.totalCounts[horizontalIndex]    = horizontalReferenceCells; //Can be removed
                counts.trailingCounts[horizontalIndex] = horizontalReferenceCells;
                counts.leadingCounts[horizontalIndex]  = horizontalReferenceCells;
            }
        }
        else // "full"
        {
            // Full: count only non-zero padded cells using position
            int32_t horizontalStart      = std::max(0, horizontalIndex - horizontalWindowSize);
            int32_t horizontalEnd        = std::min(numHorizontalBins - 1, horizontalIndex + horizontalWindowSize);
            int32_t horizontalGuardStart = std::max(horizontalStart, horizontalIndex - horizontalGuardCells);
            int32_t horizontalGuardEnd   = std::min(horizontalEnd, horizontalIndex + horizontalGuardCells);
            counts.totalCounts[horizontalIndex] =
                (horizontalEnd - horizontalStart + 1) - (horizontalGuardEnd - horizontalGuardStart + 1);
            counts.trailingCounts[horizontalIndex] = horizontalGuardStart - horizontalStart;
            counts.leadingCounts[horizontalIndex]  = horizontalEnd - horizontalGuardEnd;
            if (counts.trailingCounts[horizontalIndex] == 0)
            {
                counts.trailingCounts[horizontalIndex] = 1;
            }
            if (counts.leadingCounts[horizontalIndex] == 0)
            {
                counts.leadingCounts[horizontalIndex] = 1;
            }
        }
    }

    return counts;
}

// Helper function to compute vertical reference cell counts
CfarCellCounts computeVerticalCounts(int32_t numVerticalBins, const std::string &verticalPaddingType,
                                     int32_t verticalWindowSize, int32_t verticalGuardCells,
                                     int32_t verticalReferenceCells, PVARadarCFARType algorithm)
{
    CfarCellCounts counts;
    counts.totalCounts.resize(numVerticalBins);
    counts.trailingCounts.resize(numVerticalBins);
    counts.leadingCounts.resize(numVerticalBins);

    for (int32_t verticalIndex = 0; verticalIndex < numVerticalBins; verticalIndex++)
    {
        if (verticalPaddingType == "cyclic")
        {
            if (algorithm == PVA_CFAR_CA)
            {
                counts.totalCounts[verticalIndex]    = 2 * verticalReferenceCells;
                counts.trailingCounts[verticalIndex] = verticalReferenceCells;
                counts.leadingCounts[verticalIndex]  = verticalReferenceCells;
            }
            else /* GO and SO algorithm */
            {
                counts.totalCounts[verticalIndex]    = 2 * verticalReferenceCells;
                counts.trailingCounts[verticalIndex] = verticalReferenceCells;
                counts.leadingCounts[verticalIndex]  = verticalReferenceCells;
            }
        }
        else // "full"
        {
            // Full: count only non-zero padded cells using position
            int32_t verticalStart      = std::max(0, verticalIndex - verticalWindowSize);
            int32_t verticalEnd        = std::min(numVerticalBins - 1, verticalIndex + verticalWindowSize);
            int32_t verticalGuardStart = std::max(verticalStart, verticalIndex - verticalGuardCells);
            int32_t verticalGuardEnd   = std::min(verticalEnd, verticalIndex + verticalGuardCells);
            counts.totalCounts[verticalIndex] =
                (verticalEnd - verticalStart + 1) - (verticalGuardEnd - verticalGuardStart + 1);
            counts.trailingCounts[verticalIndex] = verticalGuardStart - verticalStart;
            counts.leadingCounts[verticalIndex]  = verticalEnd - verticalGuardEnd;
            if (counts.trailingCounts[verticalIndex] == 0)
            {
                counts.trailingCounts[verticalIndex] = 1;
            }
            if (counts.leadingCounts[verticalIndex] == 0)
            {
                counts.leadingCounts[verticalIndex] = 1;
            }
        }
    }
    return counts;
}

// Horizontal pass first sub-pass: Compute horizontal cumulative sums
void computeHorizontalCumulativeSums(std::vector<std::vector<float>> &horizontalCumSum,
                                     const std::vector<std::vector<uint32_t>> &paddedData,
                                     int32_t tileStartVerticalIndex, int32_t batchSize, int32_t verticalPaddingOffset,
                                     int32_t paddedHorizontalBins)
{
    for (int32_t batchIndex = 0; batchIndex < batchSize; batchIndex++)
    {
        int32_t paddedVerticalIndex = tileStartVerticalIndex + batchIndex + verticalPaddingOffset;
        for (int32_t horizontalIndex = 0; horizontalIndex < paddedHorizontalBins; horizontalIndex++)
        {
            horizontalCumSum[batchIndex][horizontalIndex + 1] =
                horizontalCumSum[batchIndex][horizontalIndex] +
                static_cast<float>(paddedData[paddedVerticalIndex][horizontalIndex]);
        }
    }
}

// Horizontal pass second sub-pass: Horizontal cell averaging and thresholding
void performHorizontalCfarThresholding(
    std::vector<std::vector<bool>> &detectionMap, std::vector<std::vector<uint32_t>> &horizontalThresholdMap,
    const std::vector<std::vector<float>> &horizontalCumSum, const std::vector<std::vector<uint32_t>> &inputData,
    const std::vector<uint32_t> &horizontalCounts, const std::vector<uint32_t> &horizontalTrailingCounts,
    const std::vector<uint32_t> &horizontalLeadingCounts, int32_t tileStartVerticalIndex, int32_t batchSize,
    int32_t numHorizontalBins, int32_t horizontalPaddingOffset, int32_t horizontalGuardCells,
    int32_t horizontalWindowSize, const std::vector<int32_t> &horizontalThresholdFactorFixed,
    PVARadarCFARType algorithm)
{
    for (int32_t horizontalIndex = 0; horizontalIndex < numHorizontalBins; horizontalIndex++)
    {
        int32_t paddedHorizontalIndex = horizontalIndex + horizontalPaddingOffset;

        for (int32_t batchIndex = 0; batchIndex < batchSize; batchIndex++)
        {
            int32_t verticalIndex = tileStartVerticalIndex + batchIndex;

            // Calculate horizontal threshold using lookup table and integral image
            int32_t horizontalCount         = horizontalCounts[horizontalIndex];
            int32_t horizontalTrailingCount = horizontalTrailingCounts[horizontalIndex];
            int32_t horizontalLeadingCount  = horizontalLeadingCounts[horizontalIndex];

            int64_t trailSumHorizontalfloat0 =
                static_cast<int64_t>(horizontalCumSum[batchIndex][paddedHorizontalIndex - horizontalGuardCells]);
            int64_t trailSumHorizontalfloat1 =
                static_cast<int64_t>(horizontalCumSum[batchIndex][paddedHorizontalIndex - horizontalWindowSize]);

            int64_t leadSumHorizontalfloat0 =
                static_cast<int64_t>(horizontalCumSum[batchIndex][paddedHorizontalIndex + horizontalWindowSize + 1]);
            int64_t leadSumHorizontalfloat1 =
                static_cast<int64_t>(horizontalCumSum[batchIndex][paddedHorizontalIndex + horizontalGuardCells + 1]);

            int64_t trailingSumHorizontal = trailSumHorizontalfloat0 - trailSumHorizontalfloat1;
            int64_t leadingSumHorizontal  = leadSumHorizontalfloat0 - leadSumHorizontalfloat1;

            // Use integral image for O(1) horizontal sum computation
            // Fixed-point division: horizontalSum / horizontalCount (result in fixed-point)
            int64_t horizontalCountReciprocal         = 65536 / horizontalCount;
            int64_t horizontalTrailingCountReciprocal = 65536 / horizontalTrailingCount;
            int64_t horizontalLeadingCountReciprocal  = 65536 / horizontalLeadingCount;
            int64_t horizontalNoiseLevelFixed;
            int64_t horizontalTrailingNoiseLevelFixed =
                (trailingSumHorizontal * horizontalTrailingCountReciprocal) / 65536;
            int64_t horizontalLeadingNoiseLevelFixed =
                (leadingSumHorizontal * horizontalLeadingCountReciprocal) / 65536;

            if (algorithm == PVA_CFAR_CA_GO)
            {
                // Greatest Of: take maximum of trailing and leading noise estimates
                if (horizontalTrailingNoiseLevelFixed > horizontalLeadingNoiseLevelFixed)
                {
                    horizontalNoiseLevelFixed = horizontalTrailingNoiseLevelFixed;
                }
                else
                {
                    horizontalNoiseLevelFixed = horizontalLeadingNoiseLevelFixed;
                }
            }
            else if (algorithm == PVA_CFAR_CA_SO)
            {
                if (horizontalTrailingNoiseLevelFixed < horizontalLeadingNoiseLevelFixed)
                {
                    horizontalNoiseLevelFixed = horizontalTrailingNoiseLevelFixed;
                }
                else
                {
                    horizontalNoiseLevelFixed = horizontalLeadingNoiseLevelFixed;
                }
            }
            else
            {
                // Cell Averaging (CA): average both trailing and leading cells
                int64_t horizontalSum     = trailingSumHorizontal + leadingSumHorizontal;
                horizontalNoiseLevelFixed = (horizontalSum * horizontalCountReciprocal) / 65536;
            }

            // Multiply by per-range-bin threshold factor (both in fixed-point, result has scale^2)
            int64_t horizontalThresholdFixed =
                (horizontalNoiseLevelFixed * horizontalThresholdFactorFixed[verticalIndex]) / CFAR_FIXED_POINT_SCALE;

            // Store threshold for output
            horizontalThresholdMap[verticalIndex][horizontalIndex] = horizontalThresholdFixed;

            // Horizontal CFAR detection - convert input value to fixed-point for comparison
            uint32_t currentValueFixed                   = inputData[verticalIndex][horizontalIndex];
            bool horizontalCfarDetected                  = (currentValueFixed >= horizontalThresholdFixed);
            detectionMap[verticalIndex][horizontalIndex] = horizontalCfarDetected;
        }
    }
}

// Horizontal pass third sub-pass: Horizontal NMS peak grouping
void performHorizontalNms(std::vector<std::vector<bool>> &detectionMap,
                          const std::vector<std::vector<uint32_t>> &paddedData,
                          const std::vector<std::vector<uint32_t>> &inputData, int32_t tileStartVerticalIndex,
                          int32_t batchSize, int32_t numHorizontalBins, int32_t horizontalPaddingOffset,
                          int32_t verticalPaddingOffset)
{
    for (int32_t horizontalIndex = 0; horizontalIndex < numHorizontalBins; horizontalIndex++)
    {
        int32_t paddedHorizontalIndex = horizontalIndex + horizontalPaddingOffset;

        for (int32_t batchIndex = 0; batchIndex < batchSize; batchIndex++)
        {
            int32_t verticalIndex       = tileStartVerticalIndex + batchIndex;
            int32_t paddedVerticalIndex = verticalIndex + verticalPaddingOffset;

            // Convert current value to fixed-point for comparison
            uint32_t currentValueFixed = inputData[verticalIndex][horizontalIndex];

            // Check horizontal neighbors only
            uint32_t horizontalLeftNeighbor  = paddedData[paddedVerticalIndex][paddedHorizontalIndex - 1];
            uint32_t horizontalRightNeighbor = paddedData[paddedVerticalIndex][paddedHorizontalIndex + 1];

            // NMS: current value must be greater than horizontal neighbors
            bool isHorizontalPeak =
                (currentValueFixed >= horizontalLeftNeighbor) && (currentValueFixed >= horizontalRightNeighbor);

            // Apply NMS result directly with logical AND
            detectionMap[verticalIndex][horizontalIndex] =
                detectionMap[verticalIndex][horizontalIndex] && isHorizontalPeak;
        }
    }
}

// Vertical pass first sub-pass: Compute vertical cumulative sums
void computeVerticalCumulativeSums(std::vector<std::vector<float>> &verticalCumSum,
                                   const std::vector<std::vector<uint32_t>> &paddedData,
                                   int32_t tileStartHorizontalIndex, int32_t batchSize, int32_t horizontalPaddingOffset,
                                   int32_t paddedVerticalBins)
{
    for (int32_t batchIndex = 0; batchIndex < batchSize; batchIndex++)
    {
        int32_t paddedHorizontalIndex = tileStartHorizontalIndex + batchIndex + horizontalPaddingOffset;
        for (int32_t verticalIndex = 0; verticalIndex < paddedVerticalBins; verticalIndex++)
        {
            verticalCumSum[verticalIndex + 1][batchIndex] =
                verticalCumSum[verticalIndex][batchIndex] +
                static_cast<float>(paddedData[verticalIndex][paddedHorizontalIndex]);
        }
    }
}

// Vertical pass second sub-pass: Vertical cell averaging and thresholding
void performVerticalCfarThresholding(
    std::vector<std::vector<bool>> &detectionMap, std::vector<std::vector<uint32_t>> &verticalThresholdMap,
    const std::vector<std::vector<float>> &verticalCumSum, const std::vector<std::vector<uint32_t>> &inputData,
    const std::vector<uint32_t> &verticalCounts, const std::vector<uint32_t> &verticalTrailingCounts,
    const std::vector<uint32_t> &verticalLeadingCounts, int32_t tileStartHorizontalIndex, int32_t batchSize,
    int32_t numVerticalBins, int32_t verticalPaddingOffset, int32_t verticalGuardCells, int32_t verticalWindowSize,
    int32_t verticalThresholdFactorFixed, PVARadarCFARType algorithm)
{
    for (int32_t verticalIndex = 0; verticalIndex < numVerticalBins; verticalIndex++)
    {
        int32_t paddedVerticalIndex = verticalIndex + verticalPaddingOffset;

        for (int32_t batchIndex = 0; batchIndex < batchSize; batchIndex++)
        {
            int32_t horizontalIndex = tileStartHorizontalIndex + batchIndex;

            // Calculate vertical threshold using lookup table and integral image
            int32_t verticalCount         = verticalCounts[verticalIndex];
            int32_t verticalTrailingCount = verticalTrailingCounts[verticalIndex];
            int32_t verticalLeadingCount  = verticalLeadingCounts[verticalIndex];

            int64_t trailSumVerticalfloat0 =
                static_cast<int64_t>(verticalCumSum[paddedVerticalIndex - verticalGuardCells][batchIndex]);
            int64_t trailSumVerticalfloat1 =
                static_cast<int64_t>(verticalCumSum[paddedVerticalIndex - verticalWindowSize][batchIndex]);

            int64_t leadSumVerticalfloat0 =
                static_cast<int64_t>(verticalCumSum[paddedVerticalIndex + verticalWindowSize + 1][batchIndex]);
            int64_t leadSumVerticalfloat1 =
                static_cast<int64_t>(verticalCumSum[paddedVerticalIndex + verticalGuardCells + 1][batchIndex]);

            // Use integral image for O(1) vertical sum computation
            int64_t trailingSumVertical = trailSumVerticalfloat0 - trailSumVerticalfloat1;
            int64_t leadingSumVertical  = leadSumVerticalfloat0 - leadSumVerticalfloat1;

            // Fixed-point division for noise level computation
            int64_t verticalCountReciprocal         = 65536 / verticalCount;
            int64_t verticalTrailingCountReciprocal = 65536 / verticalTrailingCount;
            int64_t verticalLeadingCountReciprocal  = 65536 / verticalLeadingCount;
            int64_t verticalNoiseLevelFixed;
            int64_t verticalNoiseLevelFixedTrailing = (trailingSumVertical * verticalTrailingCountReciprocal) / 65536;
            int64_t verticalNoiseLevelFixedLeading  = (leadingSumVertical * verticalLeadingCountReciprocal) / 65536;

            if (algorithm == PVA_CFAR_CA_GO)
            {
                // Greatest Of: take maximum of trailing and leading noise estimates
                if (verticalNoiseLevelFixedTrailing > verticalNoiseLevelFixedLeading)
                {
                    verticalNoiseLevelFixed = verticalNoiseLevelFixedTrailing;
                }
                else
                {
                    verticalNoiseLevelFixed = verticalNoiseLevelFixedLeading;
                }
            }
            else if (algorithm == PVA_CFAR_CA_SO)
            {
                // Smallest Of: take minimum of trailing and leading noise estimates
                if (verticalNoiseLevelFixedTrailing < verticalNoiseLevelFixedLeading)
                {
                    verticalNoiseLevelFixed = verticalNoiseLevelFixedTrailing;
                }
                else
                {
                    verticalNoiseLevelFixed = verticalNoiseLevelFixedLeading;
                }
            }
            else
            {
                // Cell Averaging (CA): average both trailing and leading cells
                int64_t verticalSum     = trailingSumVertical + leadingSumVertical;
                verticalNoiseLevelFixed = (verticalSum * verticalCountReciprocal) / 65536;
            }

            // Multiply by threshold factor (both in fixed-point, result has scale^2)
            int64_t verticalThresholdFixed =
                (verticalNoiseLevelFixed * verticalThresholdFactorFixed) / CFAR_FIXED_POINT_SCALE;

            // Store threshold for output
            verticalThresholdMap[verticalIndex][horizontalIndex] = verticalThresholdFixed;

            // Vertical CFAR detection - convert input value to fixed-point for comparison
            int32_t currentValueFixed = inputData[verticalIndex][horizontalIndex];
            bool verticalCfarDetected = (currentValueFixed >= verticalThresholdFixed);

            // Combine vertical CFAR result with existing horizontal detection using logical AND
            detectionMap[verticalIndex][horizontalIndex] =
                detectionMap[verticalIndex][horizontalIndex] && verticalCfarDetected;
        }
    }
}

// Vertical pass third sub-pass: Vertical NMS peak grouping
void performVerticalNms(std::vector<std::vector<bool>> &detectionMap,
                        const std::vector<std::vector<uint32_t>> &paddedData,
                        const std::vector<std::vector<uint32_t>> &inputData, int32_t tileStartHorizontalIndex,
                        int32_t batchSize, int32_t numVerticalBins, int32_t horizontalPaddingOffset,
                        int32_t verticalPaddingOffset)
{
    for (int32_t verticalIndex = 0; verticalIndex < numVerticalBins; verticalIndex++)
    {
        int32_t paddedVerticalIndex = verticalIndex + verticalPaddingOffset;

        for (int32_t batchIndex = 0; batchIndex < batchSize; batchIndex++)
        {
            int32_t horizontalIndex       = tileStartHorizontalIndex + batchIndex;
            int32_t paddedHorizontalIndex = horizontalIndex + horizontalPaddingOffset;

            // Convert current value to fixed-point for comparison
            uint32_t currentValueFixed = inputData[verticalIndex][horizontalIndex];

            // Check vertical neighbors only
            uint32_t verticalUpNeighbor   = paddedData[paddedVerticalIndex - 1][paddedHorizontalIndex];
            uint32_t verticalDownNeighbor = paddedData[paddedVerticalIndex + 1][paddedHorizontalIndex];

            // NMS: current value must be greater than vertical neighbors
            bool isVerticalPeak =
                (currentValueFixed >= verticalUpNeighbor) && (currentValueFixed >= verticalDownNeighbor);

            // Apply vertical NMS directly to detectionMap
            detectionMap[verticalIndex][horizontalIndex] =
                detectionMap[verticalIndex][horizontalIndex] && isVerticalPeak;
        }
    }
}

// Fixed-point 1D Cell Averaging CFAR detector with decoupled horizontal and vertical passes
CfarResultFixedPoint CfarCa1dOptFixedPoint(const std::vector<std::vector<uint32_t>> &inputData,
                                           std::vector<int32_t> guardCells, std::vector<int32_t> referenceCells,
                                           float probabilityFA, std::vector<float> thresholdFactor,
                                           std::vector<std::string> borderHandling, bool peakGrouping,
                                           PVARadarCFARType algorithm)
{
    // Get data dimensions - inputData format is [vertical][horizontal]
    int32_t numVerticalBins   = static_cast<int32_t>(inputData.size());
    int32_t numHorizontalBins = static_cast<int32_t>(inputData[0].size());

    // Parse parameters
    int32_t horizontalGuardCells     = guardCells[0];
    int32_t verticalGuardCells       = guardCells[1];
    int32_t horizontalReferenceCells = referenceCells[0];
    int32_t verticalReferenceCells   = referenceCells[1];

    // Parse border handling parameter
    if (borderHandling.size() != 2)
    {
        throw std::invalid_argument(
            "BorderHandling parameter must have exactly 2 elements: [horizontal_method, vertical_method]");
    }
    std::string horizontalPaddingType = borderHandling[0];
    std::string verticalPaddingType   = borderHandling[1];

    // Calculate window sizes
    int32_t horizontalWindowSize = horizontalGuardCells + horizontalReferenceCells;
    int32_t verticalWindowSize   = verticalGuardCells + verticalReferenceCells;

    // Calculate threshold factors
    // horizontalThresholdFactor is a per-range-bin vector (scaled by rangeThresholdScale)
    // verticalThresholdFactor remains a scalar (uniform across Doppler bins)
    int32_t numHorizontalReferenceCells = 2 * horizontalReferenceCells;
    int32_t numVerticalReferenceCells   = 2 * verticalReferenceCells;

    std::vector<float> horizontalThresholdFactor(numVerticalBins);
    float verticalThresholdFactor;

    if (thresholdFactor.empty())
    {
        // Compute per-range-bin threshold factors from effective PFA:
        //   effectivePFA(i) = basePFA * rangeThresholdScale(i)
        //   thresholdFactor(i) = N * (effectivePFA(i)^(-1/N) - 1)
        for (int32_t i = 0; i < numVerticalBins; i++)
        {
            horizontalThresholdFactor[i] =
                numHorizontalReferenceCells * (std::pow(probabilityFA, -1.0f / numHorizontalReferenceCells) - 1.0f);
        }
        verticalThresholdFactor =
            numVerticalReferenceCells * (std::pow(probabilityFA, -1.0f / numVerticalReferenceCells) - 1.0f);
    }
    else if (static_cast<int32_t>(thresholdFactor.size()) > 2 &&
             static_cast<int32_t>(thresholdFactor.size()) - 1 >= numVerticalBins)
    {
        // Per-range-bin horizontal factors already pre-computed (e.g., base * rangeThresholdScale).
        // Last element is the vertical threshold factor.
        for (int32_t i = 0; i < numVerticalBins; i++)
        {
            horizontalThresholdFactor[i] = thresholdFactor[i];
        }
        verticalThresholdFactor = thresholdFactor.back();
    }
    else
    {
        // Scalar base threshold: scale per range bin using rangeThresholdScale
        for (int32_t i = 0; i < numVerticalBins; i++)
        {
            horizontalThresholdFactor[i] = thresholdFactor[i];
        }
        verticalThresholdFactor = thresholdFactor.back();
    }

    // Convert threshold factors to fixed-point (per-range-bin vector for horizontal)
    std::vector<int32_t> horizontalThresholdFactorFixed(numVerticalBins);
    for (int32_t i = 0; i < numVerticalBins; i++)
    {
        horizontalThresholdFactorFixed[i] = FloatToFixedPoint(horizontalThresholdFactor[i]);
    }
    int32_t verticalThresholdFactorFixed = FloatToFixedPoint(verticalThresholdFactor);

    if (CFAR_DEBUG_ENABLED)
    {
        // Print CFAR parameters
        std::cout << "Fixed-Point CFAR Parameters:" << std::endl;
        std::cout << "  Guard cells: [" << horizontalGuardCells << ", " << verticalGuardCells
                  << "] (horizontal, vertical)" << std::endl;
        std::cout << "  Reference cells: [" << horizontalReferenceCells << ", " << verticalReferenceCells
                  << "] (horizontal, vertical)" << std::endl;
        std::cout << "  Probability of FA: " << probabilityFA << std::endl;
        std::cout << "  Horizontal threshold factor: [" << horizontalThresholdFactor[0] << " ... "
                  << horizontalThresholdFactor[numHorizontalBins - 1] << "] (per-range-bin)" << std::endl;
        std::cout << "  Vertical threshold factor: " << verticalThresholdFactor << std::endl;
        std::cout << "  Fixed-point scale: " << CFAR_FIXED_POINT_SCALE << std::endl;
        std::cout << "  Algorithm: ";
        switch (algorithm)
        {
        case PVA_CFAR_CA:
            std::cout << "Cell Averaging (CA)" << std::endl;
            break;
        case PVA_CFAR_CA_GO:
            std::cout << "Cell Averaging - Greatest Of (CA-GO)" << std::endl;
            break;
        case PVA_CFAR_CA_SO:
            std::cout << "Cell Averaging - Smallest Of (CA-SO)" << std::endl;
            break;
        case PVA_CFAR_CA_OS:
            std::cout << "Order Statistic (OS)" << std::endl;
            break;
        }
        if (horizontalPaddingType == verticalPaddingType)
        {
            std::cout << "  Border method: " << horizontalPaddingType << " (both dimensions)" << std::endl;
        }
        else
        {
            std::cout << "  Border method: horizontal=" << horizontalPaddingType << ", vertical=" << verticalPaddingType
                      << std::endl;
        }
        std::cout << "  Peak grouping: " << (peakGrouping ? "true" : "false") << std::endl;
    }

    // Create padded data for all border handling methods (full/cyclic only) - fixed-point version
    std::vector<std::vector<uint32_t>> paddedData;
    int32_t horizontalPaddingOffset = horizontalWindowSize;
    int32_t verticalPaddingOffset   = verticalWindowSize;

    paddedData = createPaddedDataFixedPoint(inputData, horizontalWindowSize, verticalWindowSize, horizontalPaddingType,
                                            verticalPaddingType);

    // Get padded data dimensions
    int32_t paddedVerticalBins   = static_cast<int32_t>(paddedData.size());
    int32_t paddedHorizontalBins = static_cast<int32_t>(paddedData[0].size());

    // Internal batching parameters
    const int32_t horizontalPassBatchSize = 16; // Process 16 vertical bins at a time
    const int32_t verticalPassBatchSize   = 16; // Process 16 horizontal bins at a time

    if (CFAR_DEBUG_ENABLED)
    {
        std::cout << "  Processing mode: Decoupled horizontal and vertical passes" << std::endl;
        std::cout << "  Horizontal batch size: " << horizontalPassBatchSize << " vertical bins" << std::endl;
        std::cout << "  Vertical batch size: " << verticalPassBatchSize << " horizontal bins" << std::endl;
    }
    // Initialize result structure - output format matches input: [vertical][horizontal]
    CfarResultFixedPoint result;
    result.detectionMap.resize(numVerticalBins, std::vector<bool>(numHorizontalBins, false));
    result.horizontalThresholdMap.resize(numVerticalBins, std::vector<uint32_t>(numHorizontalBins, 0));
    result.verticalThresholdMap.resize(numVerticalBins, std::vector<uint32_t>(numHorizontalBins, 0));

    // Single detection map for all processing
    std::vector<std::vector<bool>> detectionMap(numVerticalBins, std::vector<bool>(numHorizontalBins, false));

    // Precompute reference cell count lookup tables
    CfarCellCounts horizontalCounts =
        computeHorizontalCounts(numHorizontalBins, horizontalPaddingType, horizontalWindowSize, horizontalGuardCells,
                                horizontalReferenceCells, algorithm);
    CfarCellCounts verticalCounts = computeVerticalCounts(numVerticalBins, verticalPaddingType, verticalWindowSize,
                                                          verticalGuardCells, verticalReferenceCells, algorithm);

    // ========== Horizontal PASS ==========
    // Allocate horizontal cumulative sum buffer outside the loop
    std::vector<std::vector<float>> horizontalCumSum(horizontalPassBatchSize,
                                                     std::vector<float>(paddedHorizontalBins + 1, 0.0f));

    // Process vertical bins in batches for horizontal CFAR detection
    int32_t batchCount = (numVerticalBins + horizontalPassBatchSize - 1) / horizontalPassBatchSize;
    for (int32_t batchNumber = 0; batchNumber < batchCount; batchNumber++)
    {
        int32_t tileStartVerticalIndex = batchNumber * horizontalPassBatchSize;
        int32_t tileEndVerticalIndex   = std::min(tileStartVerticalIndex + horizontalPassBatchSize, numVerticalBins);
        int32_t batchSize              = tileEndVerticalIndex - tileStartVerticalIndex;

        // Clear cumulative sum arrays for this batch
        for (int32_t batchIndex = 0; batchIndex < batchSize; batchIndex++)
        {
            std::fill(horizontalCumSum[batchIndex].begin(), horizontalCumSum[batchIndex].end(), 0.0);
        }

        // First sub-pass: Compute horizontal cumulative sums
        computeHorizontalCumulativeSums(horizontalCumSum, paddedData, tileStartVerticalIndex, batchSize,
                                        verticalPaddingOffset, paddedHorizontalBins);

        // Second sub-pass: Horizontal cell averaging and thresholding
        performHorizontalCfarThresholding(detectionMap, result.horizontalThresholdMap, horizontalCumSum, inputData,
                                          horizontalCounts.totalCounts, horizontalCounts.trailingCounts,
                                          horizontalCounts.leadingCounts, tileStartVerticalIndex, batchSize,
                                          numHorizontalBins, horizontalPaddingOffset, horizontalGuardCells,
                                          horizontalWindowSize, horizontalThresholdFactorFixed, algorithm);
        // Third sub-pass: Horizontal NMS peak grouping (optional)
        if (peakGrouping)
        {
            performHorizontalNms(detectionMap, paddedData, inputData, tileStartVerticalIndex, batchSize,
                                 numHorizontalBins, horizontalPaddingOffset, verticalPaddingOffset);
        }
    }

    // ========== Vertical PASS ==========
    // Allocate vertical cumulative sum buffer outside the loop
    std::vector<std::vector<float>> verticalCumSum(paddedVerticalBins + 1,
                                                   std::vector<float>(verticalPassBatchSize, 0.0f));

    // Process horizontal bins in batches for vertical CFAR detection
    batchCount = (numHorizontalBins + verticalPassBatchSize - 1) / verticalPassBatchSize;
    for (int32_t batchNumber = 0; batchNumber < batchCount; batchNumber++)
    {
        int32_t tileStartHorizontalIndex = batchNumber * verticalPassBatchSize;
        int32_t tileEndHorizontalIndex = std::min(tileStartHorizontalIndex + verticalPassBatchSize, numHorizontalBins);
        int32_t batchSize              = tileEndHorizontalIndex - tileStartHorizontalIndex;

        // Clear cumulative sum arrays for this batch
        for (auto &row : verticalCumSum)
        {
            std::fill(row.begin(), row.begin() + batchSize, 0.0);
        }

        // First sub-pass: Compute vertical cumulative sums
        computeVerticalCumulativeSums(verticalCumSum, paddedData, tileStartHorizontalIndex, batchSize,
                                      horizontalPaddingOffset, paddedVerticalBins);

        // Second sub-pass: Vertical cell averaging and thresholding
        performVerticalCfarThresholding(detectionMap, result.verticalThresholdMap, verticalCumSum, inputData,
                                        verticalCounts.totalCounts, verticalCounts.trailingCounts,
                                        verticalCounts.leadingCounts, tileStartHorizontalIndex, batchSize,
                                        numVerticalBins, verticalPaddingOffset, verticalGuardCells, verticalWindowSize,
                                        verticalThresholdFactorFixed, algorithm);

        // Third sub-pass: Vertical NMS peak grouping (optional)
        if (peakGrouping)
        {
            performVerticalNms(detectionMap, paddedData, inputData, tileStartHorizontalIndex, batchSize,
                               numVerticalBins, horizontalPaddingOffset, verticalPaddingOffset);
        }
    }
    // ========== DETECTION LIST GENERATION ==========
    for (int32_t verticalTile = 0; verticalTile < numVerticalBins; verticalTile += 8)
    {
        for (int32_t horizontalTile = 0; horizontalTile < numHorizontalBins; horizontalTile += 8)
        {
            for (int32_t dy = 0; dy < 8 && (verticalTile + dy) < numVerticalBins; dy++)
            {
                for (int32_t dx = 0; dx < 8 && (horizontalTile + dx) < numHorizontalBins; dx++)
                {
                    int32_t verticalIndex                               = verticalTile + dy;
                    int32_t horizontalIndex                             = horizontalTile + dx;
                    result.detectionMap[verticalIndex][horizontalIndex] = detectionMap[verticalIndex][horizontalIndex];
                    if (detectionMap[verticalIndex][horizontalIndex])
                    {
                        result.detectionList.push_back({verticalIndex, horizontalIndex});
                    }
                }
            }
        }
    }

    if (CFAR_DEBUG_ENABLED)
    {
        std::cout << "  Ref total detections: " << result.detectionList.size() << std::endl;
    }
    return result;
}