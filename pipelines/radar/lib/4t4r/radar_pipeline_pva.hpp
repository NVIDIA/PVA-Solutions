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

#ifndef RADAR_PIPELINE_PVA_HPP
#define RADAR_PIPELINE_PVA_HPP

#include "radar_pipeline_tensors.hpp"

#include <PvaOperator.h>
#include <cupva_host.h>
#include <cupva_host_scheduling.h>
#include <nvcv/alloc/Allocator.h>

#include <cstdint>

// ============================================================================
// RadarPipelineOperators - Manages PVA operator handles
// ============================================================================

class RadarPipelineOperators
{
public:
    RadarPipelineOperators()
    {
        initRadarPipelineOperators();
    }

    ~RadarPipelineOperators()
    {
        cleanupRadarPipelineOperators();
    }

    void initRadarPipelineOperators();
    void cleanupRadarPipelineOperators();

    NVCVOperatorHandle rangeFFTOperatorHandle;
    NVCVOperatorHandle dopplerFFTOperatorHandle;
    NVCVOperatorHandle nciOperatorHandle;
    NVCVOperatorHandle peakDetectionOperatorHandle;
    NVCVOperatorHandle doaOperatorHandle;
};

// ============================================================================
// RadarPipelinePVAWorkloadParams - Manages PVA workload parameters
// ============================================================================

class RadarPipelinePVAWorkloadParams
{
public:
    RadarPipelinePVAWorkloadParams()
    {
        initRadarPipelinePVAWorkloadParams();
    }

    ~RadarPipelinePVAWorkloadParams()
    {
        cleanupRadarPipelinePVAWorkloadParams();
    }

    void initRadarPipelinePVAWorkloadParams();
    void cleanupRadarPipelinePVAWorkloadParams();

    // CUPVA objects
    cupvaSyncObj_t sync;
    cupvaFence_t fence;
    cupvaCmd_t rf;
    cupvaStream_t stream;
};

// ============================================================================
// PVA Workload Functions
// ============================================================================

// Function to create all PVA workloads at once
int radar_create_pva_workloads(RadarPipelineOperators &operators, RadarPipelineTensors &tensors, int32_t NofTx,
                               int32_t repeatFold);

// Function to submit all PVA workloads at once
int radar_submit_pva_workloads(RadarPipelineOperators &operators, RadarPipelineTensors &tensors,
                               RadarPipelinePVAWorkloadParams &pvaWorkloadParams);

#endif // RADAR_PIPELINE_PVA_HPP
