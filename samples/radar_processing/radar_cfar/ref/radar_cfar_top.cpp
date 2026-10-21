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

#include "radar_cfar_top.hpp"

#include "cfar_processing_opt.hpp"

#include "radar_cfar_param.h"

#include <string.h>

#include <memory>

int32_t radar_cfar_func(int32_t *input, int32_t *peaks, PvaRadarCFARParams cfg, int32_t width, int32_t height,
                        PVARadarCFARType algorithm)
{
    int ret = 0;

    /* Validate input parameters */
    if (input == nullptr || peaks == nullptr || width <= 0 || height <= 0)
    {
        return -1;
    }

    if (cfg.horizontalThreshold.type != PVA_CFAR_HORIZONTAL_THRESHOLD_UNIFORM &&
        cfg.horizontalThreshold.type != PVA_CFAR_HORIZONTAL_THRESHOLD_PER_ROW)
    {
        return -1;
    }
    if (cfg.horizontalThreshold.type == PVA_CFAR_HORIZONTAL_THRESHOLD_PER_ROW &&
        cfg.horizontalThreshold.value.perRow == nullptr)
    {
        return -1;
    }

    bool isUniform        = cfg.horizontalThreshold.type == PVA_CFAR_HORIZONTAL_THRESHOLD_UNIFORM;
    int32_t validateCount = isUniform ? 1 : height;
    for (int32_t i = 0; i < validateCount; i++)
    {
        float factor = isUniform ? cfg.horizontalThreshold.value.uniform : cfg.horizontalThreshold.value.perRow[i];
        if (factor <= 0.0f)
        {
            return -1;
        }
    }

    std::vector<std::vector<uint32_t>> inputData(height, std::vector<uint32_t>(width));
    for (int32_t i = 0; i < height; i++)
    {
        for (int32_t j = 0; j < width; j++)
        {
            inputData[i][j] = ((uint32_t)input[i * width + j]);
        }
    }
    std::vector<int32_t> guardCells     = {cfg.numHorGuard, cfg.numVerGuard};
    std::vector<int32_t> referenceCells = {cfg.numHorTrain, cfg.numVerTrain};
    float probabilityFA                 = 0.001;

    std::vector<float> thresholdFactor;

    for (int32_t i = 0; i < height; i++)
    {
        float thresholdFactorValue =
            isUniform ? cfg.horizontalThreshold.value.uniform : cfg.horizontalThreshold.value.perRow[i];
        thresholdFactor.push_back(thresholdFactorValue);
    }
    thresholdFactor.push_back(cfg.verticalThresholdFactor);

    std::vector<std::string> borderHandling(2, "cyclic");
    borderHandling[0] = cfg.isHorizontalCyclicPadding ? "cyclic" : "full";
    borderHandling[1] = cfg.isVerticalCyclicPadding ? "cyclic" : "full";
    bool peakGrouping = cfg.enablePeakGrouping;

    CfarResultFixedPoint result = CfarCa1dOptFixedPoint(inputData, guardCells, referenceCells, probabilityFA,
                                                        thresholdFactor, borderHandling, peakGrouping, algorithm);
    //Copy result to peaks (output)
    for (int32_t i = 0; i < result.detectionList.size(); i++)
    {
        peaks[i * 2]     = result.detectionList[i].first;
        peaks[i * 2 + 1] = result.detectionList[i].second;
    }
    return result.detectionList.size();
}