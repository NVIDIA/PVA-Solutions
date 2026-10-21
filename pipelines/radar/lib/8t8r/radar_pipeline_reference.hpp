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

#ifndef RADAR_PIPELINE_REFERENCE_HPP
#define RADAR_PIPELINE_REFERENCE_HPP

#include "radar_pipeline_tensors.hpp"

#include "capon_roi_beamspace.h"

#include <OpDoaBartlettBeamforming.h>
#include <OpDoaCaponROI.h>
#include <OpNci.h>
#include <OpTargetProcessing.h>

#include <cstdint>

// ============================================================================
// Reference Implementation Functions
// ============================================================================

// Individual processing stage functions
int rangeFFTProcessing(RadarPipelineTensors &tensors, PVARangeFFTParams rangeParams);
int dopplerFFTProcessing(RadarPipelineTensors &tensors, PVADopplerFFTParams dopplerParams);
int nciProcessing(RadarPipelineTensors &tensors, PVARadarNCIParams nciParams);
int cfarProcessing(RadarPipelineTensors &tensors, PvaRadarCFARParams cfarParams, PVARadarCFARType cfarType);
int snapshotExtractionProcessing(RadarPipelineTensors &tensors, PVASnapshotExtractionParams snapshotExtractionParams);
int doaAngleFFTProcessing(RadarPipelineTensors &tensors, PVADoaAngleFFTParams doaAngleFFTParams,
                          PVATargetProcessingParams const *targetProcessingParams);
int doaBartlettProcessing(RadarPipelineTensors &tensors, PVADoaBartlettBeamformingParams const *bartlettParams,
                          PVATargetProcessingParams const *targetProcessingParams);
int caponROIProcessing(RadarPipelineTensors &tensors, const CaponROIPreprocConfig *caponPreproc,
                       const PVADoaCaponROIParams *caponParams);

// DOA Method selection (must match radar_pipeline_pva.hpp)
enum class RadarDoaMethod;

// Function to execute all reference workloads
// @param doaMethod: Select ANGLE_FFT or BARTLETT
// @param doaAngleFFTParams: Used when doaMethod == ANGLE_FFT (can be nullptr otherwise)
// @param bartlettParams: Used when doaMethod == BARTLETT (can be nullptr otherwise)
// For BARTLETT method, tensors must have been created with RadarPipelineBartlettConfig
int radarExecuteRefWorkloads(RadarPipelineTensors &tensors, PVARangeFFTParams rangeParams,
                             PVADopplerFFTParams dopplerParams, PVARadarNCIParams nciParams,
                             PvaRadarCFARParams cfarParams, PVARadarCFARType cfarType,
                             PVASnapshotExtractionParams snapshotExtractionParams, RadarDoaMethod doaMethod,
                             PVADoaAngleFFTParams const *doaAngleFFTParams           = nullptr,
                             PVADoaBartlettBeamformingParams const *bartlettParams   = nullptr,
                             PVATargetProcessingParams const *targetProcessingParams = nullptr,
                             CaponROIPreprocConfig const *caponPreproc               = nullptr,
                             PVADoaCaponROIParams const *caponParams                 = nullptr);

#endif // RADAR_PIPELINE_REFERENCE_HPP
