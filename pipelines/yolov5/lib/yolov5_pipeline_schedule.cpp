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

#include "yolov5_pipeline_schedule.hpp"

namespace yolov5_pipeline {

FramesInFlight FramesInFlight::at(int iter, int frameCount)
{
    FramesInFlight turn;

    // Each part of the pipeline trails the one before it by one iteration.
    turn.preFrame  = iter;
    turn.infFrame  = iter - 1;
    turn.postFrame = iter - 2;

    turn.preActive  = (turn.preFrame >= 0 && turn.preFrame < frameCount);
    turn.infActive  = (turn.infFrame >= 0 && turn.infFrame < frameCount);
    turn.postActive = (turn.postFrame >= 0 && turn.postFrame < frameCount);

    // Inactive parts have no frame, so park them on slot 0 rather than letting
    // truncating division hand out a negative index.
    turn.preSlot  = turn.preActive ? turn.preFrame % NUM_PIPE : 0;
    turn.infSlot  = turn.infActive ? turn.infFrame % NUM_PIPE : 0;
    turn.postSlot = turn.postActive ? turn.postFrame % NUM_PIPE : 0;

    return turn;
}

} // namespace yolov5_pipeline
