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

#ifndef YOLOV5_PIPELINE_TENSORS_HPP
#define YOLOV5_PIPELINE_TENSORS_HPP

#include "yolov5_pipeline_iface.hpp" // NUM_PIPE, NUM_YOLO_SCALES

#include <nvcv/Image.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>

#include <cstdint>

namespace yolov5_pipeline {

/// Image and tensor handles for the PVA preprocessing stage: the source image
/// the host fills, the resize output, and the HWC view of that output which
/// NhwcToNchwLetterboxNormalize consumes.
///
/// One set per pipeline slot. recreateSlot() is re-callable because the source
/// resolution can change between frames in evaluation mode; it releases the
/// slot's previous handles first.
///
/// The letterbox *output* tensor is not here: it is wrapped over the pinned
/// TensorRT input buffer so PVA writes where TensorRT reads, which makes it
/// part of the inference handoff rather than of preprocessing.
struct PvaPreprocessTensors
{
    NVCVImageHandle inImage[NUM_PIPE]  = {};
    NVCVImageHandle outImage[NUM_PIPE] = {};

    /// Wraps outImage rather than owning storage: the resize output and the
    /// normalize input are the same bytes.
    NVCVTensorHandle lbIn[NUM_PIPE] = {};

    uint8_t *inHostPtr[NUM_PIPE] = {};
    int inRowStride[NUM_PIPE]    = {};
    int srcW[NUM_PIPE]           = {};
    int srcH[NUM_PIPE]           = {};

    /// Release and rebuild one slot for the given source resolution.
    /// Returns 0 on success.
    int recreateSlot(NVCVAllocatorHandle alloc, int slot, int srcWidth, int srcHeight, int lbW, int lbH, int channels);

    /// Row stride of a slot's resize output, which the preprocessing operator
    /// needs in order to describe its input.
    int outRowStride(int slot, int64_t *stride) const;

    void release();
};

/// Anchor boxes for the PVA decode stage, one tensor per scale, shared across
/// pipeline slots because they are constant for the model.
///
/// The predictions the decode stage reads are not here: they alias TensorRT
/// output buffers rather than being allocated, so they live with the other
/// CUDA/PVA boundaries in PvaTrtInterop.
struct PvaAnchorTensors
{
    NVCVTensorHandle anchors[NUM_YOLO_SCALES] = {};
    float *anchorHost[NUM_YOLO_SCALES]        = {};

    /// Allocate one scale's anchor tensor and resolve its host pointer.
    int allocate(NVCVAllocatorHandle alloc, int scale, int numAnchors, NVCVTensorRequirements *reqsOut);

    void release();
};

/// Detection tensors for the PVA postprocessing stage: the decode outputs that
/// the three YoloV5Decode submissions append into, and the BatchedNMS outputs.
///
/// One set per pipeline slot. Decode writes and NMS reads the same decode
/// tensors, so the handoff between them is zero-copy.
///
/// The NVCVTensorRequirements are kept after allocation because operator
/// creation consumes them: pvaBatchedNMSCreate() needs the decode and NMS
/// requirements to describe its inputs and outputs.
struct PvaDetectionTensors
{
    // Decode outputs, written by all three scales via a shared count cursor.
    NVCVTensorHandle decBoxes[NUM_PIPE]   = {};
    NVCVTensorHandle decScores[NUM_PIPE]  = {};
    NVCVTensorHandle decClasses[NUM_PIPE] = {};
    NVCVTensorHandle decCount[NUM_PIPE]   = {};

    float *decBoxesHost[NUM_PIPE]     = {};
    float *decScoresHost[NUM_PIPE]    = {};
    int16_t *decClassesHost[NUM_PIPE] = {};
    int32_t *decCountHost[NUM_PIPE]   = {};

    // NMS outputs, read back on the host to build the final detection list.
    NVCVTensorHandle nmsBoxes[NUM_PIPE]   = {};
    NVCVTensorHandle nmsScores[NUM_PIPE]  = {};
    NVCVTensorHandle nmsClasses[NUM_PIPE] = {};
    NVCVTensorHandle nmsCount[NUM_PIPE]   = {};

    float *nmsBoxesHost[NUM_PIPE]     = {};
    float *nmsScoresHost[NUM_PIPE]    = {};
    int32_t *nmsClassesHost[NUM_PIPE] = {};
    int32_t *nmsCountHost[NUM_PIPE]   = {};

    // Decode output requirements, also used to create the BatchedNMS operator.
    NVCVTensorRequirements decBoxesReqs   = {};
    NVCVTensorRequirements decScoresReqs  = {};
    NVCVTensorRequirements decClassesReqs = {};
    NVCVTensorRequirements decCountReqs   = {};
    NVCVTensorRequirements nmsBoxesReqs   = {};
    NVCVTensorRequirements nmsScoresReqs  = {};
    NVCVTensorRequirements nmsClassesReqs = {};
    NVCVTensorRequirements nmsCountReqs   = {};

    /// Allocate the decode outputs for every slot and resolve their host
    /// pointers. Returns 0 on success.
    int allocateDecode(NVCVAllocatorHandle alloc, int maxDecodeBoxes);

    /// Allocate the NMS outputs for every slot and resolve their host pointers.
    ///
    /// Kept separate from allocateDecode() so that callers preserve the order
    /// in which PVA memory is claimed: the per-scale prediction and anchor
    /// tensors are allocated between the two, and changing that order changes
    /// the resulting memory layout.
    int allocateNms(NVCVAllocatorHandle alloc, int maxNmsBoxes);

    /// Release every handle. Safe to call on a partially allocated object.
    void release();
};

} // namespace yolov5_pipeline

#endif // YOLOV5_PIPELINE_TENSORS_HPP
