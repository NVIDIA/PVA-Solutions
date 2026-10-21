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

#include "radar_pipeline_tensors.hpp"

#include "radar_pipeline_context.hpp"
#include "radar_pipeline_host_utils.hpp"

#include "doa_bartlett_beamforming_ref.h"
#include "doppler_fft_ref.h"
#include "range_fft_ref.h"
#include "snapshot_extraction_ref.h"

#include <ErrorCheckMacros.h>
#include <OpDoaBartlettBeamforming.h>
#include <PvaAllocator.h>
#include <cupva_host.h>
#include <nvcv/Tensor.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>

// ============================================================================
// RadarPipelineTensorRequirements Implementation
// ============================================================================

// Function to create tensor requirements without allocating actual tensors
void RadarPipelineTensorRequirements::createTensorRequirements(const int32_t numSamples, const int32_t numRx,
                                                               const int32_t numChirps, const int32_t numDopplerFolds,
                                                               const int32_t numTx, const int32_t numRangeBins,
                                                               const RadarPipelineBartlettConfig &bartlettConfig,
                                                               const CaponROIPreprocConfig *caponPreproc,
                                                               int32_t caponMaxRoi)
{
    int32_t err = 0;

    // PVA OpRangeFFT allows at most numSamples/2 + 1 range bins (RANGE_BINS_REAL_FFT_CALCULATE).
    const int32_t maxRangeBinsPva    = numSamples / 2 + 1;
    const int32_t requestedBins      = (numRangeBins > 0) ? numRangeBins : (numSamples / 2);
    const int32_t actualNumRangeBins = std::min(requestedBins, maxRangeBinsPva);
    if (requestedBins > maxRangeBinsPva)
    {
        fprintf(stderr,
                "RadarPipeline: NofRangeBins %d exceeds PVA Range FFT maximum %d for numSamples=%d; using %d.\n",
                requestedBins, maxRangeBinsPva, numSamples, actualNumRangeBins);
    }
    const int32_t numDopplerBins     = numChirps;
    const int32_t numVirtualAntennas = numTx * numRx;
    const int32_t maxDetections      = 8192;
    const int32_t maxTargets         = 8192;

    // Range FFT tensor data types and shapes
    NVCVDataType rawAdcDataTensorDataType     = NVCV_DATA_TYPE_S32;
    int64_t rawAdcDataTensorShape[]           = {numChirps, numRx, numSamples};
    NVCVDataType rangeFFTWindowTensorDataType = NVCV_DATA_TYPE_S32;
    int64_t rangeFFTWindowTensorShape[]       = {numSamples};
    NVCVDataType rangeProfileTensorDataType   = NVCV_DATA_TYPE_2S32;
    int64_t rangeProfileTensorShape[]         = {numChirps, numRx, actualNumRangeBins};

    // Doppler FFT tensor data types and shapes
    NVCVDataType dopplerFFTWindowTensorDataType = NVCV_DATA_TYPE_S32;
    int64_t dopplerFFTWindowTensorShape[]       = {numChirps};
    NVCVDataType rangeDopplerMapTensorDataType  = NVCV_DATA_TYPE_2S32;
    int64_t rangeDopplerMapTensorShape[3];
    if (numTx >= 8)
    {
        rangeDopplerMapTensorShape[0] = actualNumRangeBins;
        rangeDopplerMapTensorShape[1] = numDopplerBins;
        rangeDopplerMapTensorShape[2] = numRx;
    }
    else
    {
        rangeDopplerMapTensorShape[0] = actualNumRangeBins;
        rangeDopplerMapTensorShape[1] = numRx;
        rangeDopplerMapTensorShape[2] = numDopplerBins;
    }

    // NCI tensor data types and shapes
    NVCVDataType nciRxTensorDataType    = NVCV_DATA_TYPE_U32;
    int64_t nciRxTensorShape[]          = {actualNumRangeBins, numDopplerBins};
    NVCVDataType nciFinalTensorDataType = NVCV_DATA_TYPE_U32;
    int64_t nciFinalTensorShape[]       = {actualNumRangeBins, numDopplerBins / numDopplerFolds};

    // CFAR shapes
    NVCVDataType detectionCountTensorDataType      = NVCV_DATA_TYPE_S32;
    int64_t detectionCountTensorShape[]            = {1};
    NVCVDataType foldedDetectionListTensorDataType = NVCV_DATA_TYPE_S32;
    int64_t foldedDetectionListTensorShape[]       = {maxDetections, 2};

    // Snapshot Extraction shapes
    NVCVDataType ddmDopplerOffsetsTensorDataType = NVCV_DATA_TYPE_F32;
    int64_t ddmDopplerOffsetsTensorShape[]       = {numDopplerFolds};
    NVCVDataType detectionListTensorDataType     = NVCV_DATA_TYPE_S32;
    int64_t detectionListTensorShape[]           = {maxDetections, 2};
    NVCVDataType snapshotsTensorDataType         = NVCV_DATA_TYPE_2S32;
    int64_t snapshotsTensorShape[]               = {maxDetections, numTx, numRx};

    // DOA Angle FFT shapes
    NVCVDataType targetCountTensorDataType = NVCV_DATA_TYPE_S32;
    int64_t targetCountTensorShape[]       = {1};

    // Target Processing shapes
    NVCVDataType targetListTensorDataType = NVCV_DATA_TYPE_F32;
    int64_t targetListTensorShape[]       = {8, maxTargets}; // 8 target properties

    // Declare tensor layout variables
    NVCVTensorLayout tensorLayoutHCW;
    NVCVTensorLayout tensorLayoutCHW;
    NVCVTensorLayout tensorLayoutHW;
    NVCVTensorLayout tensorLayoutW;

    // Define tensor ranks
    const int32_t tensorRank3D = 3;
    const int32_t tensorRank2D = 2;
    const int32_t tensorRank1D = 1;

    // Create tensor layouts
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HCW", &tensorLayoutHCW), err, EXIT);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("CHW", &tensorLayoutCHW), err, EXIT);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HW", &tensorLayoutHW), err, EXIT);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("W", &tensorLayoutW), err, EXIT);

    // Calculate tensor requirements (no actual allocation)
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank3D, rawAdcDataTensorShape, rawAdcDataTensorDataType,
                                                        tensorLayoutHCW, 0, 0, &rawAdcDataTensorRequirements),
                          err, EXIT);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(tensorRank3D, rangeProfileTensorShape, rangeProfileTensorDataType,
                                      tensorLayoutHCW, 0, 0, &rangeProfileTensorRequirements),
        err, EXIT);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(tensorRank1D, rangeFFTWindowTensorShape, rangeFFTWindowTensorDataType,
                                      tensorLayoutW, 0, 0, &rangeFFTWindowTensorRequirements),
        err, EXIT);

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(tensorRank1D, dopplerFFTWindowTensorShape, dopplerFFTWindowTensorDataType,
                                      tensorLayoutW, 0, 0, &dopplerFFTWindowTensorRequirements),
        err, EXIT);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(tensorRank3D, rangeDopplerMapTensorShape, rangeDopplerMapTensorDataType,
                                      tensorLayoutHCW, 0, 0, &rangeDopplerMapTensorRequirements),
        err, EXIT);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank2D, nciRxTensorShape, nciRxTensorDataType,
                                                        tensorLayoutHW, 0, 0, &nciRxTensorRequirements),
                          err, EXIT);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank2D, nciFinalTensorShape, nciFinalTensorDataType,
                                                        tensorLayoutHW, 0, 0, &nciFinalTensorRequirements),
                          err, EXIT);

    // CFAR tensor requirements
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(tensorRank1D, detectionCountTensorShape, detectionCountTensorDataType,
                                      tensorLayoutW, 0, 0, &detectionCountTensorRequirements),
        err, EXIT);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(tensorRank2D, foldedDetectionListTensorShape, foldedDetectionListTensorDataType,
                                      tensorLayoutHW, 0, 0, &foldedDetectionListTensorRequirements),
        err, EXIT);

    // Snapshot Extraction tensor requirements
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(tensorRank1D, ddmDopplerOffsetsTensorShape, ddmDopplerOffsetsTensorDataType,
                                      tensorLayoutW, 0, 0, &ddmDopplerOffsetsTensorRequirements),
        err, EXIT);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(tensorRank2D, detectionListTensorShape, detectionListTensorDataType,
                                      tensorLayoutHW, 0, 0, &detectionListTensorRequirements),
        err, EXIT);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank3D, snapshotsTensorShape, snapshotsTensorDataType,
                                                        tensorLayoutCHW, 0, 0, &snapshotsTensorRequirements),
                          err, EXIT);

    // DOA Angle FFT + Target Processing tensor requirements
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank1D, targetCountTensorShape, targetCountTensorDataType,
                                                        tensorLayoutW, 0, 0, &targetCountTensorRequirements),
                          err, EXIT);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank2D, targetListTensorShape, targetListTensorDataType,
                                                        tensorLayoutHW, 0, 0, &targetListTensorRequirements),
                          err, EXIT);

    // Target Processing intermediate tensors
    // targetAngles: [3, maxTargets] F32 - azimuth, elevation, power per target
    {
        int64_t targetAnglesTensorShape[] = {3, maxTargets};
        NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank2D, targetAnglesTensorShape, NVCV_DATA_TYPE_F32,
                                                            tensorLayoutHW, 0, 0, &targetAnglesTensorRequirements),
                              err, EXIT);
    }
    // targetIndexMap: [maxTargets] S32
    {
        int64_t targetIndexMapTensorShape[] = {maxTargets};
        NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank1D, targetIndexMapTensorShape, NVCV_DATA_TYPE_S32,
                                                            tensorLayoutW, 0, 0, &targetIndexMapTensorRequirements),
                              err, EXIT);
    }

    // DOA Bartlett Beamforming tensor requirements (only if Bartlett is enabled)
    if (RadarPipelineBartlettConfigIsEnabled(bartlettConfig))
    {
        NVCVTensorLayout tensorLayoutNCHW;
        const int32_t tensorRank4D = 4;
        NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("NCHW", &tensorLayoutNCHW), err, EXIT);

        // Virtual array locations [numTx, numRx] with 2 channels (x, y)
        int64_t valShape[] = {numTx, numRx};
        NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank2D, valShape, NVCV_DATA_TYPE_2F32, tensorLayoutHW,
                                                            0, 0, &virtualArrayLocationsTensorRequirements),
                              err, EXIT);

        // Azimuth bins [numAzimuthBins]
        int64_t azShape[] = {bartlettConfig.numAzimuthBins};
        NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank1D, azShape, NVCV_DATA_TYPE_F32, tensorLayoutW, 0,
                                                            0, &azimuthBinsTensorRequirements),
                              err, EXIT);

        // Elevation bins [numElevationBins]
        int64_t elShape[] = {bartlettConfig.numElevationBins};
        NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank1D, elShape, NVCV_DATA_TYPE_F32, tensorLayoutW, 0,
                                                            0, &elevationBinsTensorRequirements),
                              err, EXIT);

        // Steering vectors [numAzimuthBins, numElevationBins, numTx, numRx] complex Q15
        int64_t svShape[] = {bartlettConfig.numAzimuthBins, bartlettConfig.numElevationBins, numTx, numRx};
        NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank4D, svShape, NVCV_DATA_TYPE_2S16,
                                                            tensorLayoutNCHW, 0, 0, &steeringVectorsTensorRequirements),
                              err, EXIT);
    }

    // DOA Capon ROI tensor requirements, only for the Capon method: sized from its validated preprocessing
    // config through caponPreprocGeometry(), the geometry the operators are created with.
    if (caponPreproc != nullptr)
    {
        const CaponPreprocGeometry g = caponPreprocGeometry(*caponPreproc);
        const int64_t maxRoi         = caponMaxRoi;
        const int32_t H              = caponPreproc->numDopplerBinsCovMat;

        NVCVTensorLayout tensorLayoutNHW;
        NVCVTensorLayout tensorLayoutNCHW;
        const int32_t tensorRank4D = 4;
        NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("NHW", &tensorLayoutNHW), err, EXIT);
        NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("NCHW", &tensorLayoutNCHW), err, EXIT);

        // Capon operator (OpDoaCaponROI.h): ROI snapshots and steering in, peaks out.
        int64_t caponSnapshotsShape[] = {maxRoi, H, g.svDim};
        NVCV_CHECK_ERROR_GOTO(
            nvcvTensorCalcRequirementsPva(tensorRank3D, caponSnapshotsShape, NVCV_DATA_TYPE_2S32, tensorLayoutNHW, 0, 0,
                                          &caponSnapshotsRoiTensorRequirements),
            err, EXIT);
        int64_t caponSteeringShape[] = {maxRoi, g.svDim, g.numElBins, g.numAzBins};
        NVCV_CHECK_ERROR_GOTO(
            nvcvTensorCalcRequirementsPva(tensorRank4D, caponSteeringShape, NVCV_DATA_TYPE_2S32, tensorLayoutNCHW, 0, 0,
                                          &caponSteeringVectorsRoiTensorRequirements),
            err, EXIT);
        int64_t caponPeakShape[] = {maxRoi, RADAR_PIPELINE_CAPON_NUM_PEAKS};
        NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank2D, caponPeakShape, NVCV_DATA_TYPE_F32,
                                                            tensorLayoutHW, 0, 0, &caponPeakPowerTensorRequirements),
                              err, EXIT);
        NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank2D, caponPeakShape, NVCV_DATA_TYPE_F32,
                                                            tensorLayoutHW, 0, 0, &caponAzBinTensorRequirements),
                              err, EXIT);
        NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank2D, caponPeakShape, NVCV_DATA_TYPE_F32,
                                                            tensorLayoutHW, 0, 0, &caponElBinTensorRequirements),
                              err, EXIT);

        // ROI preprocessing operator (OpCaponROIPreprocess.h) inputs.
        int64_t caponROICountShape[] = {1};
        NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank1D, caponROICountShape, NVCV_DATA_TYPE_S32,
                                                            tensorLayoutW, 0, 0, &caponROICountTensorRequirements),
                              err, EXIT);
        int64_t caponDetDopShape[] = {1, maxRoi};
        NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank2D, caponDetDopShape, NVCV_DATA_TYPE_S32,
                                                            tensorLayoutHW, 0, 0, &caponDetDopTensorRequirements),
                              err, EXIT);
        int64_t caponElemShape[] = {maxRoi, g.nvc};
        NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank2D, caponElemShape, NVCV_DATA_TYPE_2F32,
                                                            tensorLayoutHW, 0, 0, &caponElemTensorRequirements),
                              err, EXIT);
        int64_t caponPlaneShape[] = {maxRoi, static_cast<int64_t>(g.numChirps) * g.numRx};
        NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank2D, caponPlaneShape, NVCV_DATA_TYPE_2F32,
                                                            tensorLayoutHW, 0, 0, &caponPlaneTensorRequirements),
                              err, EXIT);
        int64_t caponLutShape[] = {g.numRoiPos, g.svBlockCplx};
        NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank2D, caponLutShape, NVCV_DATA_TYPE_2S32,
                                                            tensorLayoutHW, 0, 0, &caponSteeringLutTensorRequirements),
                              err, EXIT);
    }

    // Set up NCI output tensor requirements pointers
    outNciTensorReqs = {&nciRxTensorRequirements, &nciFinalTensorRequirements, NULL};

EXIT:
    if (err != 0)
    {
        std::string errorMessage = "Failed to create tensor requirements: " + std::to_string(err);
        throw std::runtime_error(errorMessage);
    }
}

// ============================================================================
// RadarPipelineTensors Implementation
// ============================================================================

RadarPipelineTensors::RadarPipelineTensors(RadarPipelineContext *ctx)
    : m_tensorReqs(*ctx->tensorReqs)
    , m_ctx(ctx)
{
    nvcvAllocatorConstructPva(&m_allocatorHandle);
    initRadarPipelineTensors();
    createRadarPipelineTensors();
}

// Function to initialize all tensor handles to NULL
void RadarPipelineTensors::initRadarPipelineTensors()
{
    // Range FFT tensors
    inRangeFFTTensorHandle      = NULL; // not owned; set transiently by radar_pipeline_process/validate()
    rangeFFTWindowTensorHandle  = NULL;
    rangeProfileTensorHandle    = NULL;
    refRangeProfileTensorHandle = NULL;

    // Doppler FFT tensors
    dopplerFFTWindowTensorHandle   = NULL;
    rangeDopplerMapTensorHandle    = NULL;
    refRangeDopplerMapTensorHandle = NULL;

    // NCI tensors
    nciRxTensorHandle       = NULL;
    nciFinalTensorHandle    = NULL;
    refNciRxTensorHandle    = NULL;
    refNciFinalTensorHandle = NULL;

    // CFAR tensors
    detectionCountTensorHandle         = NULL;
    foldedDetectionListTensorHandle    = NULL;
    refDetectionCountTensorHandle      = NULL;
    refFoldedDetectionListTensorHandle = NULL;

    // Snapshot Extraction tensors
    ddmDopplerOffsetsTensorHandle = NULL;
    detectionListTensorHandle     = NULL;
    snapshotsTensorHandle         = NULL;
    refDetectionListTensorHandle  = NULL;
    refSnapshotsTensorHandle      = NULL;

    // DOA Angle FFT + Target Processing tensors
    targetCountTensorHandle    = NULL;
    refTargetCountTensorHandle = NULL;

    // Target list tensors
    targetListTensorHandle    = NULL;
    refTargetListTensorHandle = NULL;

    // DOA intermediate output tensors
    targetAnglesTensorHandle      = NULL;
    targetIndexMapTensorHandle    = NULL;
    refTargetAnglesTensorHandle   = NULL;
    refTargetIndexMapTensorHandle = NULL;

    // DOA Bartlett Beamforming tensors
    virtualArrayLocationsTensorHandle = NULL;
    azimuthBinsTensorHandle           = NULL;
    elevationBinsTensorHandle         = NULL;
    steeringVectorsTensorHandle       = NULL;

    // DOA Capon ROI tensors
    caponSnapshotsRoiTensorHandle       = NULL;
    caponSteeringVectorsRoiTensorHandle = NULL;
    caponROICountTensorHandle           = NULL;
    caponPeakPowerTensorHandle          = NULL;
    caponAzBinTensorHandle              = NULL;
    caponElBinTensorHandle              = NULL;
    refCaponPeakPowerTensorHandle       = NULL;
    refCaponAzBinTensorHandle           = NULL;
    refCaponElBinTensorHandle           = NULL;

    // DOA Capon ROI preprocessing input tensors
    caponDetDopTensorHandle      = NULL;
    caponElemTensorHandle        = NULL;
    caponPlaneTensorHandle       = NULL;
    caponSteeringLutTensorHandle = NULL;
}

// Function to cleanup all allocated resources
void RadarPipelineTensors::cleanupRadarPipelineTensors()
{
    // Range FFT tensors (inRangeFFTTensorHandle not owned; skip)
    if (rangeFFTWindowTensorHandle)
        nvcvTensorDecRef(rangeFFTWindowTensorHandle, NULL);
    if (rangeProfileTensorHandle)
        nvcvTensorDecRef(rangeProfileTensorHandle, NULL);
    if (refRangeProfileTensorHandle)
        nvcvTensorDecRef(refRangeProfileTensorHandle, NULL);

    // Doppler FFT tensors
    if (dopplerFFTWindowTensorHandle)
        nvcvTensorDecRef(dopplerFFTWindowTensorHandle, NULL);
    if (rangeDopplerMapTensorHandle)
        nvcvTensorDecRef(rangeDopplerMapTensorHandle, NULL);
    if (refRangeDopplerMapTensorHandle)
        nvcvTensorDecRef(refRangeDopplerMapTensorHandle, NULL);

    // NCI tensors
    if (nciRxTensorHandle)
        nvcvTensorDecRef(nciRxTensorHandle, NULL);
    if (nciFinalTensorHandle)
        nvcvTensorDecRef(nciFinalTensorHandle, NULL);
    if (refNciRxTensorHandle)
        nvcvTensorDecRef(refNciRxTensorHandle, NULL);
    if (refNciFinalTensorHandle)
        nvcvTensorDecRef(refNciFinalTensorHandle, NULL);

    // CFAR tensors
    if (detectionCountTensorHandle)
        nvcvTensorDecRef(detectionCountTensorHandle, NULL);
    if (foldedDetectionListTensorHandle)
        nvcvTensorDecRef(foldedDetectionListTensorHandle, NULL);
    if (refDetectionCountTensorHandle)
        nvcvTensorDecRef(refDetectionCountTensorHandle, NULL);
    if (refFoldedDetectionListTensorHandle)
        nvcvTensorDecRef(refFoldedDetectionListTensorHandle, NULL);

    // Snapshot Extraction tensors
    if (ddmDopplerOffsetsTensorHandle)
        nvcvTensorDecRef(ddmDopplerOffsetsTensorHandle, NULL);
    if (detectionListTensorHandle)
        nvcvTensorDecRef(detectionListTensorHandle, NULL);
    if (snapshotsTensorHandle)
        nvcvTensorDecRef(snapshotsTensorHandle, NULL);
    if (refDetectionListTensorHandle)
        nvcvTensorDecRef(refDetectionListTensorHandle, NULL);
    if (refSnapshotsTensorHandle)
        nvcvTensorDecRef(refSnapshotsTensorHandle, NULL);

    // DOA Angle FFT + Target Processing tensors
    if (targetCountTensorHandle)
        nvcvTensorDecRef(targetCountTensorHandle, NULL);
    if (refTargetCountTensorHandle)
        nvcvTensorDecRef(refTargetCountTensorHandle, NULL);

    // Target list tensors
    if (targetListTensorHandle)
        nvcvTensorDecRef(targetListTensorHandle, NULL);
    if (refTargetListTensorHandle)
        nvcvTensorDecRef(refTargetListTensorHandle, NULL);

    // DOA intermediate output tensors
    if (targetAnglesTensorHandle)
    {
        nvcvTensorDecRef(targetAnglesTensorHandle, NULL);
        targetAnglesTensorHandle = NULL;
    }
    if (targetIndexMapTensorHandle)
    {
        nvcvTensorDecRef(targetIndexMapTensorHandle, NULL);
        targetIndexMapTensorHandle = NULL;
    }
    if (refTargetAnglesTensorHandle)
    {
        nvcvTensorDecRef(refTargetAnglesTensorHandle, NULL);
        refTargetAnglesTensorHandle = NULL;
    }
    if (refTargetIndexMapTensorHandle)
    {
        nvcvTensorDecRef(refTargetIndexMapTensorHandle, NULL);
        refTargetIndexMapTensorHandle = NULL;
    }

    // DOA Bartlett Beamforming tensors (only if allocated)
    if (virtualArrayLocationsTensorHandle)
        nvcvTensorDecRef(virtualArrayLocationsTensorHandle, NULL);
    if (azimuthBinsTensorHandle)
        nvcvTensorDecRef(azimuthBinsTensorHandle, NULL);
    if (elevationBinsTensorHandle)
        nvcvTensorDecRef(elevationBinsTensorHandle, NULL);
    if (steeringVectorsTensorHandle)
        nvcvTensorDecRef(steeringVectorsTensorHandle, NULL);

    // DOA Capon ROI tensors
    if (caponSnapshotsRoiTensorHandle)
        nvcvTensorDecRef(caponSnapshotsRoiTensorHandle, NULL);
    if (caponSteeringVectorsRoiTensorHandle)
        nvcvTensorDecRef(caponSteeringVectorsRoiTensorHandle, NULL);
    if (caponROICountTensorHandle)
        nvcvTensorDecRef(caponROICountTensorHandle, NULL);
    if (caponPeakPowerTensorHandle)
        nvcvTensorDecRef(caponPeakPowerTensorHandle, NULL);
    if (caponAzBinTensorHandle)
        nvcvTensorDecRef(caponAzBinTensorHandle, NULL);
    if (caponElBinTensorHandle)
        nvcvTensorDecRef(caponElBinTensorHandle, NULL);
    for (NVCVTensorHandle h : {refCaponPeakPowerTensorHandle, refCaponAzBinTensorHandle, refCaponElBinTensorHandle})
        if (h)
            nvcvTensorDecRef(h, NULL);

    // DOA Capon ROI preprocessing input tensors
    if (caponDetDopTensorHandle)
        nvcvTensorDecRef(caponDetDopTensorHandle, NULL);
    if (caponElemTensorHandle)
        nvcvTensorDecRef(caponElemTensorHandle, NULL);
    if (caponPlaneTensorHandle)
        nvcvTensorDecRef(caponPlaneTensorHandle, NULL);
    if (caponSteeringLutTensorHandle)
        nvcvTensorDecRef(caponSteeringLutTensorHandle, NULL);

    // Cleanup allocator
    if (m_allocatorHandle)
    {
        nvcvAllocatorDecRef(m_allocatorHandle, NULL);
        m_allocatorHandle = NULL;
    }
}

// Function to create all tensors using pre-computed requirements
void RadarPipelineTensors::createRadarPipelineTensors()
{
    int32_t err                                       = 0;
    const RadarPipelineBartlettConfig &bartlettConfig = m_ctx->bartlettConfig;
    const RadarPipelineGP *gp                         = m_ctx->gp;
    const std::vector<float> &ddmDopplerOffsets       = m_ctx->ddmDopplerOffsets;

    // Range FFT tensors (inRangeFFTTensorHandle is caller-supplied; not constructed here)
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(&m_tensorReqs.rangeProfileTensorRequirements, m_allocatorHandle, &rangeProfileTensorHandle),
        err, EXIT);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&m_tensorReqs.rangeProfileTensorRequirements, m_allocatorHandle,
                                              &refRangeProfileTensorHandle),
                          err, EXIT);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&m_tensorReqs.rangeFFTWindowTensorRequirements, m_allocatorHandle,
                                              &rangeFFTWindowTensorHandle),
                          err, EXIT);
    generateRangeFFTWindow(rangeFFTWindowTensorHandle, PVA_BATCH_FFT_WINDOW_HANNING);

    // Doppler FFT tensors
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&m_tensorReqs.dopplerFFTWindowTensorRequirements, m_allocatorHandle,
                                              &dopplerFFTWindowTensorHandle),
                          err, EXIT);
    generateDopplerFFTWindow(dopplerFFTWindowTensorHandle, PVA_BATCH_FFT_WINDOW_HANNING);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&m_tensorReqs.rangeDopplerMapTensorRequirements, m_allocatorHandle,
                                              &rangeDopplerMapTensorHandle),
                          err, EXIT);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&m_tensorReqs.rangeDopplerMapTensorRequirements, m_allocatorHandle,
                                              &refRangeDopplerMapTensorHandle),
                          err, EXIT);

    // NCI tensors
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(&m_tensorReqs.nciRxTensorRequirements, m_allocatorHandle, &nciRxTensorHandle), err, EXIT);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(&m_tensorReqs.nciRxTensorRequirements, m_allocatorHandle, &refNciRxTensorHandle), err,
        EXIT);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(&m_tensorReqs.nciFinalTensorRequirements, m_allocatorHandle, &nciFinalTensorHandle), err,
        EXIT);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(&m_tensorReqs.nciFinalTensorRequirements, m_allocatorHandle, &refNciFinalTensorHandle), err,
        EXIT);

    // CFAR tensors
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&m_tensorReqs.detectionCountTensorRequirements, m_allocatorHandle,
                                              &detectionCountTensorHandle),
                          err, EXIT);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&m_tensorReqs.foldedDetectionListTensorRequirements, m_allocatorHandle,
                                              &foldedDetectionListTensorHandle),
                          err, EXIT);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&m_tensorReqs.detectionCountTensorRequirements, m_allocatorHandle,
                                              &refDetectionCountTensorHandle),
                          err, EXIT);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&m_tensorReqs.foldedDetectionListTensorRequirements, m_allocatorHandle,
                                              &refFoldedDetectionListTensorHandle),
                          err, EXIT);

    // Snapshot Extraction tensors
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&m_tensorReqs.ddmDopplerOffsetsTensorRequirements, m_allocatorHandle,
                                              &ddmDopplerOffsetsTensorHandle),
                          err, EXIT);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&m_tensorReqs.detectionListTensorRequirements, m_allocatorHandle,
                                              &detectionListTensorHandle),
                          err, EXIT);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(&m_tensorReqs.snapshotsTensorRequirements, m_allocatorHandle, &snapshotsTensorHandle), err,
        EXIT);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&m_tensorReqs.detectionListTensorRequirements, m_allocatorHandle,
                                              &refDetectionListTensorHandle),
                          err, EXIT);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(&m_tensorReqs.snapshotsTensorRequirements, m_allocatorHandle, &refSnapshotsTensorHandle),
        err, EXIT);

    // Load DDM Doppler Offsets
    loadDdmDopplerOffsets(ddmDopplerOffsetsTensorHandle, ddmDopplerOffsets);

    // DOA Angle FFT tensors
    // DOA Angle FFT + Target Processing tensors
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(&m_tensorReqs.targetCountTensorRequirements, m_allocatorHandle, &targetCountTensorHandle),
        err, EXIT);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&m_tensorReqs.targetCountTensorRequirements, m_allocatorHandle,
                                              &refTargetCountTensorHandle),
                          err, EXIT);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(&m_tensorReqs.targetListTensorRequirements, m_allocatorHandle, &targetListTensorHandle),
        err, EXIT);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(&m_tensorReqs.targetListTensorRequirements, m_allocatorHandle, &refTargetListTensorHandle),
        err, EXIT);

    // DOA intermediate output tensors
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(&m_tensorReqs.targetAnglesTensorRequirements, m_allocatorHandle, &targetAnglesTensorHandle),
        err, EXIT);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&m_tensorReqs.targetIndexMapTensorRequirements, m_allocatorHandle,
                                              &targetIndexMapTensorHandle),
                          err, EXIT);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&m_tensorReqs.targetAnglesTensorRequirements, m_allocatorHandle,
                                              &refTargetAnglesTensorHandle),
                          err, EXIT);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&m_tensorReqs.targetIndexMapTensorRequirements, m_allocatorHandle,
                                              &refTargetIndexMapTensorHandle),
                          err, EXIT);

    // DOA Bartlett Beamforming tensors (only if Bartlett is enabled)
    if (RadarPipelineBartlettConfigIsEnabled(bartlettConfig))
    {
        bool const hasCalibData =
            (m_ctx->calib_data.steeringVectorsPtr != nullptr) && (m_ctx->calib_data.steeringVectorsSize > 0);

        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&m_tensorReqs.virtualArrayLocationsTensorRequirements,
                                                  m_allocatorHandle, &virtualArrayLocationsTensorHandle),
                              err, EXIT);
        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&m_tensorReqs.azimuthBinsTensorRequirements, m_allocatorHandle,
                                                  &azimuthBinsTensorHandle),
                              err, EXIT);
        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&m_tensorReqs.elevationBinsTensorRequirements, m_allocatorHandle,
                                                  &elevationBinsTensorHandle),
                              err, EXIT);
        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&m_tensorReqs.steeringVectorsTensorRequirements, m_allocatorHandle,
                                                  &steeringVectorsTensorHandle),
                              err, EXIT);

        // Fill azimuth bins
        loadAngleBins(azimuthBinsTensorHandle, bartlettConfig.numAzimuthBins, bartlettConfig.azimuthStart,
                      bartlettConfig.azimuthStep);
        // Fill elevation bins
        loadAngleBins(elevationBinsTensorHandle, bartlettConfig.numElevationBins, bartlettConfig.elevationStart,
                      bartlettConfig.elevationStep);

        if (hasCalibData)
        {
            PopulateSteeringVectorsTensor(steeringVectorsTensorHandle, m_ctx->calib_data.steeringVectorsPtr,
                                          static_cast<size_t>(m_ctx->calib_data.steeringVectorsSize));
        }
        else
        {
            PopulateVirtualArrayLocationsTensor(virtualArrayLocationsTensorHandle, m_ctx->numTx, m_ctx->numRx, gp->Dis,
                                                gp->DisE);
            NVCV_CHECK_ERROR_GOTO(pvaDoaBartlettBeamformingComputeSteeringVectors(
                                      virtualArrayLocationsTensorHandle, azimuthBinsTensorHandle,
                                      elevationBinsTensorHandle, steeringVectorsTensorHandle),
                                  err, EXIT);
        }
    }

    // DOA Capon ROI tensors (only when Capon is the selected DOA method)
    if (m_ctx->doaMethod == RadarDoaMethod::CAPON_ROI)
    {
        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&m_tensorReqs.caponSnapshotsRoiTensorRequirements, m_allocatorHandle,
                                                  &caponSnapshotsRoiTensorHandle),
                              err, EXIT);
        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&m_tensorReqs.caponSteeringVectorsRoiTensorRequirements,
                                                  m_allocatorHandle, &caponSteeringVectorsRoiTensorHandle),
                              err, EXIT);
        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&m_tensorReqs.caponROICountTensorRequirements, m_allocatorHandle,
                                                  &caponROICountTensorHandle),
                              err, EXIT);
        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&m_tensorReqs.caponPeakPowerTensorRequirements, m_allocatorHandle,
                                                  &caponPeakPowerTensorHandle),
                              err, EXIT);
        NVCV_CHECK_ERROR_GOTO(
            nvcvTensorConstruct(&m_tensorReqs.caponAzBinTensorRequirements, m_allocatorHandle, &caponAzBinTensorHandle),
            err, EXIT);
        NVCV_CHECK_ERROR_GOTO(
            nvcvTensorConstruct(&m_tensorReqs.caponElBinTensorRequirements, m_allocatorHandle, &caponElBinTensorHandle),
            err, EXIT);
        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&m_tensorReqs.caponPeakPowerTensorRequirements, m_allocatorHandle,
                                                  &refCaponPeakPowerTensorHandle),
                              err, EXIT);
        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&m_tensorReqs.caponAzBinTensorRequirements, m_allocatorHandle,
                                                  &refCaponAzBinTensorHandle),
                              err, EXIT);
        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&m_tensorReqs.caponElBinTensorRequirements, m_allocatorHandle,
                                                  &refCaponElBinTensorHandle),
                              err, EXIT);

        // Capon ROI preprocessing operator input tensors.
        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&m_tensorReqs.caponDetDopTensorRequirements, m_allocatorHandle,
                                                  &caponDetDopTensorHandle),
                              err, EXIT);
        NVCV_CHECK_ERROR_GOTO(
            nvcvTensorConstruct(&m_tensorReqs.caponElemTensorRequirements, m_allocatorHandle, &caponElemTensorHandle),
            err, EXIT);
        NVCV_CHECK_ERROR_GOTO(
            nvcvTensorConstruct(&m_tensorReqs.caponPlaneTensorRequirements, m_allocatorHandle, &caponPlaneTensorHandle),
            err, EXIT);
        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&m_tensorReqs.caponSteeringLutTensorRequirements, m_allocatorHandle,
                                                  &caponSteeringLutTensorHandle),
                              err, EXIT);

        // Precompute the frame-invariant steering LUT once (pva_capon_roi_beamspace); the operator
        // generates the DFT twiddle tables itself. The per-detection detected-Doppler /
        // element-snapshot / range-Doppler-plane inputs are marshaled each frame.
        {
            const CaponROIPreprocConfig &pp = m_ctx->caponPreprocConfig;

            void *lutPtr = nullptr;
            if (!radarTensorHostPtr(caponSteeringLutTensorHandle, &lutPtr))
            {
                err = -1;
                goto EXIT;
            }

            int rc = caponROIPreprocessSteeringLut(&pp, static_cast<int32_t *>(lutPtr));
            if (rc != 0)
            {
                fprintf(stderr, "RadarPipeline: Capon ROI steering LUT build failed (%d)\n", rc);
                err = rc;
                goto EXIT;
            }
        }
    }

EXIT:
    if (err != 0)
    {
        cleanupRadarPipelineTensors();
        std::string errorMessage = "Failed to create radar pipeline tensors: " + std::to_string(err);
        throw std::runtime_error(errorMessage);
    }
}
