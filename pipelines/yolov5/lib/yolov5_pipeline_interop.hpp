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

#ifndef YOLOV5_PIPELINE_INTEROP_HPP
#define YOLOV5_PIPELINE_INTEROP_HPP

#include "yolov5_pipeline_tensors.hpp"

#include <nvcv/Tensor.h>

namespace yolov5_pipeline {

/// The CUDA/PVA boundary between preprocessing, inference and postprocessing.
///
/// Both handoffs are aliases, not copies, which is what lets the three stages
/// overlap without a transfer between them:
///
///   PVA -> TensorRT   the letterbox output tensor is a PVA view of the pinned
///                     TensorRT input buffer, so PVA writes where TensorRT
///                     reads and no host-to-device copy is needed
///   TensorRT -> PVA   each prediction tensor is a PVA view of a TensorRT
///                     output buffer, so decode reads the inference result in
///                     place and no device-to-host copy is needed
///
/// A third handoff needs no object at all: decode and NMS are handed the same
/// PVA tensors, so that boundary is free by construction. See
/// PvaDetectionTensors.
///
/// Aliasing is only valid when the PVA layout is packed exactly as TensorRT
/// laid the buffer out. Both wrap calls check that and fail rather than
/// silently reinterpreting memory.
struct PvaTrtInterop
{
    /// TensorRT input, one per slot. Pinned host memory so PVA can write to it.
    void *trtInput[NUM_PIPE] = {};

    /// TensorRT outputs, one per slot per scale.
    void *trtOutput[NUM_PIPE][NUM_YOLO_SCALES] = {};

    /// PVA view of trtInput: the destination of NhwcToNchwLetterboxNormalize.
    NVCVTensorHandle lbOut[NUM_PIPE] = {};

    /// PVA view of trtOutput: the input to YoloV5Decode.
    NVCVTensorHandle predictions[NUM_PIPE][NUM_YOLO_SCALES] = {};

    /// True when trtInput was allocated pinned; it governs how it is freed.
    bool inputPinned = false;

    /// Alias the pinned TensorRT input for every slot as the letterbox output.
    /// Returns 0 on success, -1 if the layout is not packed or a wrap fails.
    int wrapLetterboxOutput(const NVCVTensorRequirements &reqs, int modelHeight, int modelWidth);

    /// Alias one TensorRT output buffer as a prediction tensor.
    /// Returns 0 on success, -1 if the layout is not packed or the wrap fails.
    int wrapPrediction(int slot, int scale, const NVCVTensorRequirements &reqs, int gridH, int gridW);

    /// Release the PVA views, then the buffers they alias. Order matters.
    void release();
};

} // namespace yolov5_pipeline

#endif // YOLOV5_PIPELINE_INTEROP_HPP
