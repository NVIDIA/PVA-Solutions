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

#ifndef RADAR_PIPELINE_TENSORS_HPP
#define RADAR_PIPELINE_TENSORS_HPP

#include "radar_pipeline_iface.hpp"

#include "capon_roi_beamspace.h" // CaponROIPreprocConfig

struct RadarPipelineContext; // forward declaration; full definition in radar_pipeline_context.hpp

#include <PvaAllocator.h>
#include <PvaOperatorTypes.h>
#include <nvcv/Fwd.h>
#include <nvcv/alloc/Allocator.h>

#include <vector>

/** @return true if Bartlett tensors are requested (numAzimuthBins and numElevationBins > 0). */
inline bool RadarPipelineBartlettConfigIsEnabled(const RadarPipelineBartlettConfig &c)
{
    return c.numAzimuthBins > 0 && c.numElevationBins > 0;
}

// ============================================================================
// RadarPipelineTensorRequirements - Holds tensor requirements without allocation
// ============================================================================

class RadarPipelineTensorRequirements
{
public:
    RadarPipelineTensorRequirements(const int32_t numSamples, const int32_t numRx, const int32_t numChirps,
                                    const int32_t numDopplerFolds, const int32_t numTx, const int32_t numRangeBins = 0,
                                    const RadarPipelineBartlettConfig &bartlettConfig = {},
                                    const CaponROIPreprocConfig *caponPreproc = nullptr, const int32_t caponMaxRoi = 0)
    {
        // If numRangeBins is 0 or not specified, default to numSamples/2
        const int32_t effectiveNumRangeBins = (numRangeBins > 0) ? numRangeBins : (numSamples / 2);
        createTensorRequirements(numSamples, numRx, numChirps, numDopplerFolds, numTx, effectiveNumRangeBins,
                                 bartlettConfig, caponPreproc, caponMaxRoi);
    }

    void createTensorRequirements(const int32_t numSamples, const int32_t numRx, const int32_t numChirps,
                                  const int32_t numDopplerFolds, const int32_t numTx, const int32_t numRangeBins,
                                  const RadarPipelineBartlettConfig &bartlettConfig,
                                  const CaponROIPreprocConfig *caponPreproc, int32_t caponMaxRoi);

    // Tensor requirements
    // Range FFT requirements
    NVCVTensorRequirements rawAdcDataTensorRequirements;
    NVCVTensorRequirements rangeProfileTensorRequirements;
    NVCVTensorRequirements rangeFFTWindowTensorRequirements;

    // Doppler FFT requirements
    NVCVTensorRequirements dopplerFFTWindowTensorRequirements;
    NVCVTensorRequirements rangeDopplerMapTensorRequirements;

    // NCI requirements
    NVCVTensorRequirements nciRxTensorRequirements;
    NVCVTensorRequirements nciFinalTensorRequirements;

    // CFAR requirements
    NVCVTensorRequirements detectionCountTensorRequirements;
    NVCVTensorRequirements foldedDetectionListTensorRequirements;

    // Snapshot Extraction requirements
    NVCVTensorRequirements ddmDopplerOffsetsTensorRequirements;
    NVCVTensorRequirements detectionListTensorRequirements;
    NVCVTensorRequirements snapshotsTensorRequirements;

    // DOA Angle FFT + Target Processing requirements
    NVCVTensorRequirements targetCountTensorRequirements;
    NVCVTensorRequirements targetListTensorRequirements;

    // DOA intermediate output requirements (used between DOA and standalone Target Processing)
    NVCVTensorRequirements targetAnglesTensorRequirements;
    NVCVTensorRequirements targetIndexMapTensorRequirements;

    // DOA Bartlett Beamforming requirements
    NVCVTensorRequirements virtualArrayLocationsTensorRequirements;
    NVCVTensorRequirements azimuthBinsTensorRequirements;
    NVCVTensorRequirements elevationBinsTensorRequirements;
    NVCVTensorRequirements steeringVectorsTensorRequirements;

    // DOA Capon ROI requirements
    NVCVTensorRequirements caponSnapshotsRoiTensorRequirements; // [maxRoi, H, svDim] 2S32 (NHW)
    NVCVTensorRequirements
        caponSteeringVectorsRoiTensorRequirements;           // [maxRoi, svDim, numElBins, numAzBins] 2S32 (NCHW)
    NVCVTensorRequirements caponROICountTensorRequirements;  // [1] S32
    NVCVTensorRequirements caponPeakPowerTensorRequirements; // [maxRoi, numPeaks] F32 (HW)
    NVCVTensorRequirements caponAzBinTensorRequirements;     // [maxRoi, numPeaks] F32 (HW)
    NVCVTensorRequirements caponElBinTensorRequirements;     // [maxRoi, numPeaks] F32 (HW)

    // DOA Capon ROI preprocessing operator inputs (pvaCaponROIPreprocess*). Constant/frame-varying
    // host-staged tensors feeding the VPU projection-core operator.
    NVCVTensorRequirements caponDetDopTensorRequirements;      // [1, maxDet] S32 (HW) detected Doppler
    NVCVTensorRequirements caponElemTensorRequirements;        // [maxDet, nvc] 2F32 (HW) element snapshot
    NVCVTensorRequirements caponPlaneTensorRequirements;       // [maxDet, numChirps*numRx] 2F32 (HW) RD plane
    NVCVTensorRequirements caponSteeringLutTensorRequirements; // [numRoiPos, svBlockCplx] 2S32 (HW) steering LUT

    std::vector<const NVCVTensorRequirements *> outNciTensorReqs;
};

// ============================================================================
// RadarPipelineTensors - Manages all tensor allocations
// ============================================================================

class RadarPipelineTensors
{
public:
    // Allocator handle
    NVCVAllocatorHandle m_allocatorHandle;

    /** Requirements owned by @ref RadarPipelineContext::tensorReqs; must outlive this object. */
    RadarPipelineTensorRequirements &m_tensorReqs;

    RadarPipelineContext *m_ctx;

    /** Constructor defined in .cpp so RadarPipelineContext is complete. @a ctx->tensorReqs must be non-null. */
    RadarPipelineTensors(RadarPipelineContext *ctx);

    ~RadarPipelineTensors()
    {
        cleanupRadarPipelineTensors();
    }

    void initRadarPipelineTensors();
    void cleanupRadarPipelineTensors();
    void createRadarPipelineTensors();

    // Range FFT tensors
    NVCVTensorHandle inRangeFFTTensorHandle; // set per-call in radar_pipeline_process/validate(); not owned
    NVCVTensorHandle rangeFFTWindowTensorHandle;
    NVCVTensorHandle rangeProfileTensorHandle;
    NVCVTensorHandle refRangeProfileTensorHandle;

    // Doppler FFT tensors
    NVCVTensorHandle dopplerFFTWindowTensorHandle;
    NVCVTensorHandle rangeDopplerMapTensorHandle;
    NVCVTensorHandle refRangeDopplerMapTensorHandle;

    // NCI tensors
    NVCVTensorHandle nciRxTensorHandle;
    NVCVTensorHandle nciFinalTensorHandle;
    NVCVTensorHandle refNciRxTensorHandle;
    NVCVTensorHandle refNciFinalTensorHandle;

    // CFAR tensors
    NVCVTensorHandle detectionCountTensorHandle;
    NVCVTensorHandle foldedDetectionListTensorHandle;
    NVCVTensorHandle refDetectionCountTensorHandle;
    NVCVTensorHandle refFoldedDetectionListTensorHandle;

    // Snapshot Extraction tensors
    NVCVTensorHandle ddmDopplerOffsetsTensorHandle;
    NVCVTensorHandle detectionListTensorHandle;
    NVCVTensorHandle snapshotsTensorHandle;
    NVCVTensorHandle refDetectionListTensorHandle;
    NVCVTensorHandle refSnapshotsTensorHandle;

    // DOA Angle FFT + Target Processing tensors
    NVCVTensorHandle targetCountTensorHandle;
    NVCVTensorHandle targetListTensorHandle;
    NVCVTensorHandle refTargetCountTensorHandle;
    NVCVTensorHandle refTargetListTensorHandle;

    // DOA intermediate output tensors
    NVCVTensorHandle targetAnglesTensorHandle;
    NVCVTensorHandle targetIndexMapTensorHandle;
    NVCVTensorHandle refTargetAnglesTensorHandle;
    NVCVTensorHandle refTargetIndexMapTensorHandle;

    // DOA Bartlett Beamforming tensors
    NVCVTensorHandle virtualArrayLocationsTensorHandle;
    NVCVTensorHandle azimuthBinsTensorHandle;
    NVCVTensorHandle elevationBinsTensorHandle;
    NVCVTensorHandle steeringVectorsTensorHandle;

    // DOA Capon ROI tensors (allocated only when doaMethod == CAPON_ROI)
    NVCVTensorHandle caponSnapshotsRoiTensorHandle;
    NVCVTensorHandle caponSteeringVectorsRoiTensorHandle;
    NVCVTensorHandle caponROICountTensorHandle;
    NVCVTensorHandle caponPeakPowerTensorHandle;
    NVCVTensorHandle caponAzBinTensorHandle;
    NVCVTensorHandle caponElBinTensorHandle;
    // DoaCaponROI CPU reference outputs on the PVA ROI tensors (validate() only).
    NVCVTensorHandle refCaponPeakPowerTensorHandle;
    NVCVTensorHandle refCaponAzBinTensorHandle;
    NVCVTensorHandle refCaponElBinTensorHandle;

    // DOA Capon ROI preprocessing operator inputs (allocated only when doaMethod == CAPON_ROI).
    // caponSteeringLut is precomputed once at allocation (the operator generates its twiddles); caponDetDop/
    // caponElem/caponPlane are marshaled from the pipeline data every frame in radarSubmitCaponROI().
    NVCVTensorHandle caponDetDopTensorHandle;
    NVCVTensorHandle caponElemTensorHandle;
    NVCVTensorHandle caponPlaneTensorHandle;
    NVCVTensorHandle caponSteeringLutTensorHandle;

    // Capon host staging, reused across frames: the dequantized element snapshots of the frame's ROIs.
    std::vector<float> caponSnapScratch;
    // Set once the "detections exceed maxRoi" notice has been printed, so a stream does not repeat it.
    bool caponMaxRoiNoticed = false;
};

#endif // RADAR_PIPELINE_TENSORS_HPP
