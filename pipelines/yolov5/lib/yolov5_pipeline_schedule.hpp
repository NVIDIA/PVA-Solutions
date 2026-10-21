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

#ifndef YOLOV5_PIPELINE_SCHEDULE_HPP
#define YOLOV5_PIPELINE_SCHEDULE_HPP

#include "yolov5_pipeline_iface.hpp" // NUM_PIPE

namespace yolov5_pipeline {

/// The frames in flight on one iteration of the depth-three software pipeline.
///
/// On iteration N three frames are in flight at once, each on its own processor
/// and its own slot, so none of them touch the same buffers:
///
///     VPU0: preprocess(N)   ||   GPU: infer(N-1)   ||   VPU1: postprocess(N-2)
///
/// Preprocessing and postprocessing run on the PVA while the GPU infers a
/// different frame, so the PVA work is hidden behind inference when it is
/// shorter.
///
/// The first two iterations and the last two are partial. There is nothing to
/// infer on iteration 0 and nothing to postprocess on iterations 0 and 1 while
/// the pipeline fills, and nothing left to preprocess once the input is
/// exhausted while it drains. A full run therefore needs
/// frameCount + NUM_PIPE - 1 iterations, and benchmarks measure only those where
/// all three parts are active. In steady state one iteration completes one
/// frame, which is why the benchmark's iteration count means the same thing here
/// as in the GPU-only pipeline.
///
/// Slot indices for inactive parts are 0. They are never read, because every
/// use is guarded by the matching preActive / infActive / postActive flag, but
/// they stay in range so a missing guard cannot index out of bounds.
struct FramesInFlight
{
    int preFrame  = 0;
    int infFrame  = 0;
    int postFrame = 0;

    int preSlot  = 0;
    int infSlot  = 0;
    int postSlot = 0;

    bool preActive  = false;
    bool infActive  = false;
    bool postActive = false;

    /// Frames in flight on the given iteration of a run over frameCount frames.
    static FramesInFlight at(int iter, int frameCount);
};

/// Iterations needed to push frameCount frames through the pipeline, including
/// the ones that fill and drain it.
constexpr int iterationsFor(int frameCount)
{
    return frameCount + NUM_PIPE - 1;
}

} // namespace yolov5_pipeline

#endif // YOLOV5_PIPELINE_SCHEDULE_HPP
