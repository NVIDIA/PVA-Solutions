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

#include "yolov5_pipeline_iface.hpp"

#include <cstdio>

namespace yolov5_pipeline {

const StageMap GPU_ONLY_STAGE_MAP = {
    "GPU-only",
    {
        {Processor::Gpu, "CUDA letterbox kernel"},
        {Processor::Gpu, "CUDA preprocess kernel"},
        {Processor::Gpu, "TensorRT"},
        {Processor::Gpu, "TensorRT, in the engine graph"},
        {Processor::Gpu, "TensorRT EfficientNMS plugin"},
    },
};

const StageMap PVA_GPU_STAGE_MAP = {
    "PVA+GPU",
    {
        {Processor::Pva, "PVA ImageResize"},
        {Processor::Pva, "PVA NhwcToNchwLetterboxNormalize"},
        {Processor::Gpu, "TensorRT"},
        {Processor::Pva, "PVA YoloV5Decode, one per scale"},
        {Processor::Pva, "PVA BatchedNMS"},
    },
};

const char *stageName(Stage stage)
{
    switch (stage)
    {
    case Stage::Resize:
        return "Resize + letterbox";
    case Stage::Normalize:
        return "NHWC->NCHW + normalize";
    case Stage::Inference:
        return "Backbone + 3 heads";
    case Stage::Decode:
        return "YOLOv5 decode";
    case Stage::Nms:
        return "Class-aware NMS";
    case Stage::Count:
        break;
    }
    return "unknown";
}

void printStageMap(const StageMap &map)
{
    printf("Stage map: %s\n", map.pipelineName);
    for (int i = 0; i < STAGE_COUNT; i++)
    {
        const StageBinding &binding = map.stages[i];
        printf("  %d  %-24s %-4s %s\n", i + 1, stageName(static_cast<Stage>(i)),
               binding.processor == Processor::Pva ? "PVA" : "GPU", binding.implementation);
    }
    printf("\n");
}

} // namespace yolov5_pipeline
