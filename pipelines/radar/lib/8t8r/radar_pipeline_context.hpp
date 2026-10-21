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
#include "radar_pipeline_pva.hpp"
#include "radar_pipeline_tensors.hpp"

#include <OpDopplerFFT.h>
#include <OpRangeFFT.h>
#include <OpSnapshotExtraction.h>
#include <PvaOperatorTypes.h>

#include <memory>
#include <vector>

// Internal context structure
class RadarPipelineContext
{
public:
    // tensorReqs declared before tensors so it is destroyed after tensors (~RadarPipelineTensors uses m_tensorReqs ref).
    std::unique_ptr<RadarPipelineTensorRequirements> tensorReqs;
    std::unique_ptr<RadarPipelineTensors> tensors;
    std::unique_ptr<RadarPipelineOperators> operators;
    std::unique_ptr<RadarPipelinePVAWorkloadParams> pvaParams;

    std::vector<float> ddmDopplerOffsets;
    RadarPipelineGP *gp;
    RadarPipelineBartlettConfig bartlettConfig;
    RadarPipelineCaponConfig caponConfig;       // valid when doaMethod == CAPON_ROI
    PVADoaCaponROIParams caponParams;           // operator params (CAPON_ROI)
    CaponROIPreprocConfig caponPreprocConfig{}; // ROI preprocessing config (CAPON_ROI), shared with the reference
    PvaRadarCFARParams cfarParams;
    /** Storage for @ref cfarParams.horizontalThresholdFactor (reference CFAR reads this during validate()). */
    std::vector<float> cfarHorizontalThresholdFactors;
    PVADoaAngleFFTParams doaAngleFFTParams;
    PVADoaBartlettBeamformingParams bartlettParams; // valid when doaMethod == BARTLETT
    PVATargetProcessingParams targetProcessingParams;
    RadarDoaMethod doaMethod;

    // Stored for reference validation (radar_pipeline_validate). Value-initialized so the layout fields are
    // defined even on a path that never assigns them; the factory assigns them explicitly on create.
    PVARangeFFTParams rangeParams{};
    PVADopplerFFTParams dopplerParams{};
    PVARadarNCIParams nciParams;
    PVASnapshotExtractionParams snapshotExtractionParams;

    int32_t numSamples;
    int32_t numRx;
    int32_t numChirps;
    int32_t numDopplerFolds;
    int32_t numTx;
    int32_t numRangeBins;
    int32_t targetCount;

    /** Calibration data (e.g. steering vectors); set from config at create. */
    RadarPipelineCalibData calib_data;

    ~RadarPipelineContext();
};

#endif