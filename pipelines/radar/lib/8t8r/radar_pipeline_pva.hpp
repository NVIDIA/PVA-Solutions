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

#include "capon_roi_beamspace.h"

#include <OpDoaBartlettBeamforming.h>
#include <OpDoaCaponROI.h>
#include <OpNci.h>
#include <PvaOperator.h>
#include <cupva_host.h>
#include <cupva_host_scheduling.h>
#include <nvcv/alloc/Allocator.h>

#include <cstdint>

// ============================================================================
// DOA Method Selection
// ============================================================================

enum class RadarDoaMethod
{
    ANGLE_FFT, // DOA Angle FFT (requires exactly 8 TX antennas)
    BARTLETT,  // Bartlett Beamforming (supports any numTx*numRx <= 64, divisible by 8)
    CAPON_ROI  // Capon (MVDR) DoA on ROI beamspace (PVA ROI preprocessing + Capon operators)
};

// ============================================================================
// Capon ROI preprocessing geometry
// ============================================================================

// Derived beamspace geometry for the Capon ROI preprocessing operator (OpCaponROIPreprocess.h), computed from
// the preprocessing config shared with the CPU reference. Tensor sizing, operator creation and the per-frame
// marshalling all take their geometry from here.
struct CaponPreprocGeometry
{
    int32_t numTx;       ///< transmit channels
    int32_t numRx;       ///< receive channels
    int32_t numChirps;   ///< Doppler dimension of the range-Doppler cube
    int32_t numAzVirt;   ///< virtual azimuth aperture, numAzimuthElements*numTx unless set (= beamAz)
    int32_t numElVirt;   ///< virtual elevation aperture = numElevationElements
    int32_t beamAz;      ///< beamspace azimuth FFT size
    int32_t beamEl;      ///< beamspace elevation FFT size (zero-padded)
    int32_t roiAz;       ///< ROI azimuth size in beamspace bins
    int32_t roiEl;       ///< ROI elevation size in beamspace bins
    int32_t svDim;       ///< beamspace channels per ROI = roiAz*roiEl
    int32_t numAzBins;   ///< steering-grid azimuth directions per ROI
    int32_t numElBins;   ///< steering-grid elevation directions per ROI
    int32_t nvc;         ///< element channels = numAzVirt*numElVirt
    int32_t numRoiPos;   ///< steering LUT positions = (beamAz-roiAz+1)*(beamEl-roiEl+1)
    int32_t svBlockCplx; ///< complex steering entries per ROI block = svDim*numElBins*numAzBins
};

inline CaponPreprocGeometry caponPreprocGeometry(const CaponROIPreprocConfig &pp)
{
    CaponPreprocGeometry g;
    g.numTx       = pp.numTx;
    g.numRx       = pp.numRx;
    g.numChirps   = pp.numChirps;
    g.numElVirt   = pp.numElevationElements;
    g.numAzVirt   = (pp.numAzVirt != 0) ? pp.numAzVirt : pp.numAzimuthElements * pp.numTx;
    g.beamAz      = g.numAzVirt;
    g.beamEl      = (g.numElVirt > g.numAzVirt) ? g.numElVirt : g.numAzVirt;
    g.roiAz       = pp.roiAzSize;
    g.roiEl       = pp.roiElSize;
    g.svDim       = g.roiAz * g.roiEl;
    g.numAzBins   = pp.numAzBins;
    g.numElBins   = pp.numElBins;
    g.nvc         = g.numAzVirt * g.numElVirt;
    g.numRoiPos   = (g.beamAz - g.roiAz + 1) * (g.beamEl - g.roiEl + 1);
    g.svBlockCplx = g.svDim * g.numElBins * g.numAzBins;
    return g;
}

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
    NVCVOperatorHandle cfarOperatorHandle;
    NVCVOperatorHandle snapshotExtractionOperatorHandle;
    NVCVOperatorHandle doaAngleFFTOperatorHandle;
    NVCVOperatorHandle doaBartlettOperatorHandle;
    NVCVOperatorHandle caponPreprocOperatorHandle; // Capon ROI beamspace-projection preprocessing
    NVCVOperatorHandle doaCaponROIOperatorHandle;  // Capon ROI DoA (snapshots + steering -> peaks)
    NVCVOperatorHandle targetProcessingOperatorHandle;
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

    // Allocator
    NVCVAllocatorHandle allocatorHandle;

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
// @param doaMethod: Select ANGLE_FFT (requires 8 TX) or BARTLETT (flexible TX count)
// @param doaAngleFFTParams: Used when doaMethod == ANGLE_FFT (can be nullptr otherwise)
// @param bartlettParams: Used when doaMethod == BARTLETT (can be nullptr otherwise)
int radarCreatePvaWorkloads(RadarPipelineOperators &operators, RadarPipelineTensorRequirements &tensorReqs,
                            PVARangeFFTParams rangeParams, PVADopplerFFTParams dopplerParams,
                            PVARadarNCIParams nciParams, PVARadarCFARType cfarType,
                            PvaRadarCFARParams const *cfarParams, PVASnapshotExtractionParams snapshotExtractionParams,
                            RadarDoaMethod doaMethod, PVADoaAngleFFTParams const *doaAngleFFTParams = nullptr,
                            PVADoaBartlettBeamformingParams const *bartlettParams   = nullptr,
                            PVATargetProcessingParams const *targetProcessingParams = nullptr,
                            PVADoaCaponROIParams const *caponParams                 = nullptr,
                            CaponROIPreprocConfig const *caponPreproc               = nullptr);

// Function to submit all PVA workloads at once
// For BARTLETT method, tensors must have been created with BartlettConfig.
// For CAPON_ROI method, caponPreproc must be non-null: the pre-DOA stages are submitted and
// awaited, the per-detection inputs are marshaled on the host, then the ROI preprocessing and
// Capon operators are submitted. targetCount is set to the number of valid Capon peaks.
int radarSubmitPvaWorkloads(RadarPipelineOperators &operators, RadarPipelineTensors &tensors,
                            RadarPipelinePVAWorkloadParams &pvaWorkloadParams, RadarDoaMethod doaMethod,
                            CaponROIPreprocConfig const *caponPreproc = nullptr);
#endif // RADAR_PIPELINE_PVA_HPP
