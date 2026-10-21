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

#ifndef YOLOV5_PIPELINE_IFACE_HPP
#define YOLOV5_PIPELINE_IFACE_HPP

namespace yolov5_pipeline {

/// Frames in flight in the PVA+GPU pipeline: preprocess(N), infer(N-1) and
/// postprocess(N-2). Each in-flight frame owns one pipeline slot.
constexpr int NUM_PIPE = 3;

/// YOLOv5 detection heads: strides 8, 16 and 32.
constexpr int NUM_YOLO_SCALES = 3;

/// The YOLOv5 workload, as five stages. Both pipelines run all five stages and
/// differ only in the processor that executes each one.
enum class Stage
{
    Resize = 0, ///< Scale the source frame into the model's input, with letterbox padding.
    Normalize,  ///< NHWC uint8 to NCHW float32, scaled to [0,1].
    Inference,  ///< YOLOv5s backbone and its three detection heads.
    Decode,     ///< Anchor boxes and logits to candidate detections, per scale.
    Nms,        ///< Class-aware non-maximum suppression.
    Count
};

constexpr int STAGE_COUNT = static_cast<int>(Stage::Count);

/// Which processor executes a stage.
enum class Processor
{
    Gpu, ///< CUDA kernel or TensorRT, on the iGPU.
    Pva  ///< A PVA operator, on a VPU.
};

/// One stage's placement, plus what actually implements it.
struct StageBinding
{
    Processor processor;
    const char *implementation;
};

/// A pipeline's stage placement, in Stage order.
struct StageMap
{
    const char *pipelineName;
    StageBinding stages[STAGE_COUNT];
};

/// The GPU-only pipeline: every stage on the iGPU, with decode and NMS folded
/// into the TensorRT engine graph.
extern const StageMap GPU_ONLY_STAGE_MAP;

/// The PVA+GPU pipeline: pre- and postprocessing on the PVA, inference on the
/// iGPU. Inference is the same stage as in the GPU-only pipeline; the other
/// four stages run on the PVA.
extern const StageMap PVA_GPU_STAGE_MAP;

/// Print a stage map. Both pipelines print theirs at startup.
void printStageMap(const StageMap &map);

/// Human-readable name of a stage, e.g. "Resize + letterbox".
const char *stageName(Stage stage);

} // namespace yolov5_pipeline

#endif // YOLOV5_PIPELINE_IFACE_HPP
