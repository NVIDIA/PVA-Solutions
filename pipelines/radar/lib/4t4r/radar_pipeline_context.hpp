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

#ifndef RADAR_PIPELINE_CONTEXT_HPP
#define RADAR_PIPELINE_CONTEXT_HPP

#include "radar_pipeline_iface.hpp"

#include <memory>

class RadarPipelineTensorRequirements;
class RadarPipelineTensors;
class RadarPipelineOperators;
class RadarPipelinePVAWorkloadParams;

// Internal context structure
class RadarPipelineContext
{
public:
    // tensorReqs before tensors so it outlives RadarPipelineTensors (which holds a reference to it).
    std::unique_ptr<RadarPipelineTensorRequirements> tensorReqs;
    std::unique_ptr<RadarPipelineTensors> tensors;
    std::unique_ptr<RadarPipelineOperators> operators;
    std::unique_ptr<RadarPipelinePVAWorkloadParams> pvaParams;

    int32_t Ns;           // NofSamples (ADC samples per ramp)
    int32_t Nr;           // NofRamps   (number of chirps; Doppler FFT size)
    int32_t NofRangeBins; // NofRangeBins (range bins after real FFT; <= Ns/2+1)
    int32_t NofRx;
    int32_t NofTx;
    int32_t repeatFold;
    int32_t targetCount;

    ~RadarPipelineContext();
};

#endif
