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

#include "radar_pipeline_pva.hpp"

#include "radar_pipeline_host_utils.hpp"

#include <ErrorCheckMacros.h>
#include <OpCaponROIPreprocess.h>
#include <OpDoaAngleFFT.h>
#include <OpDoaBartlettBeamforming.h>
#include <OpDoaCaponROI.h>
#include <OpDopplerFFT.h>
#include <OpNci.h>
#include <OpRadarCFAR.h>
#include <OpRangeFFT.h>
#include <OpSnapshotExtraction.h>
#include <OpTargetProcessing.h>
#include <PvaAllocator.h>
#include <cupva_host.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iterator> // For std::size
#include <string>

// ============================================================================
// RadarPipelineOperators Implementation
// ============================================================================

// Function to initialize all operator handles to NULL
void RadarPipelineOperators::initRadarPipelineOperators()
{
    rangeFFTOperatorHandle           = NULL;
    dopplerFFTOperatorHandle         = NULL;
    nciOperatorHandle                = NULL;
    cfarOperatorHandle               = NULL;
    snapshotExtractionOperatorHandle = NULL;
    doaAngleFFTOperatorHandle        = NULL;
    doaBartlettOperatorHandle        = NULL;
    caponPreprocOperatorHandle       = NULL;
    doaCaponROIOperatorHandle        = NULL;
    targetProcessingOperatorHandle   = NULL;
}

// Function to cleanup all operator handles
void RadarPipelineOperators::cleanupRadarPipelineOperators()
{
    // Destroy PVA operators
    if (rangeFFTOperatorHandle != NULL)
    {
        nvcvOperatorDestroy(rangeFFTOperatorHandle);
        rangeFFTOperatorHandle = NULL;
    }

    if (dopplerFFTOperatorHandle != NULL)
    {
        nvcvOperatorDestroy(dopplerFFTOperatorHandle);
        dopplerFFTOperatorHandle = NULL;
    }

    if (nciOperatorHandle != NULL)
    {
        nvcvOperatorDestroy(nciOperatorHandle);
        nciOperatorHandle = NULL;
    }

    if (cfarOperatorHandle != NULL)
    {
        nvcvOperatorDestroy(cfarOperatorHandle);
        cfarOperatorHandle = NULL;
    }

    if (snapshotExtractionOperatorHandle != NULL)
    {
        nvcvOperatorDestroy(snapshotExtractionOperatorHandle);
        snapshotExtractionOperatorHandle = NULL;
    }

    if (doaAngleFFTOperatorHandle != NULL)
    {
        nvcvOperatorDestroy(doaAngleFFTOperatorHandle);
        doaAngleFFTOperatorHandle = NULL;
    }

    if (doaBartlettOperatorHandle != NULL)
    {
        nvcvOperatorDestroy(doaBartlettOperatorHandle);
        doaBartlettOperatorHandle = NULL;
    }

    if (caponPreprocOperatorHandle != NULL)
    {
        nvcvOperatorDestroy(caponPreprocOperatorHandle);
        caponPreprocOperatorHandle = NULL;
    }

    if (doaCaponROIOperatorHandle != NULL)
    {
        nvcvOperatorDestroy(doaCaponROIOperatorHandle);
        doaCaponROIOperatorHandle = NULL;
    }

    if (targetProcessingOperatorHandle != NULL)
    {
        nvcvOperatorDestroy(targetProcessingOperatorHandle);
        targetProcessingOperatorHandle = NULL;
    }
}

// ============================================================================
// RadarPipelinePVAWorkloadParams Implementation
// ============================================================================

// Function to initialize RadarPipelinePVAWorkloadParams
void RadarPipelinePVAWorkloadParams::initRadarPipelinePVAWorkloadParams()
{
    int32_t err = 0;
    // Create allocator
    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(&allocatorHandle), err, EXIT);

    // Initialize CUPVA objects
    CUPVA_CHECK_ERROR_GOTO(CupvaSyncObjCreate(&sync, false, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD), err, EXIT);
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceInit(&fence, sync), err, EXIT);
    CUPVA_CHECK_ERROR_GOTO(CupvaCmdRequestFencesInit(&rf, &fence, 1), err, EXIT);
    CUPVA_CHECK_ERROR_GOTO(CupvaStreamCreate(&stream, CUPVA_PVA0, CUPVA_VPU_ANY), err, EXIT);

EXIT:
    if (err != 0)
    {
        std::string errorMessage = "Failed to initialize RadarPipelinePVAWorkloadParams: " + std::to_string(err);
        throw std::runtime_error(errorMessage);
    }
}

// Function to tear down RadarPipelinePVAWorkloadParams
void RadarPipelinePVAWorkloadParams::cleanupRadarPipelinePVAWorkloadParams()
{
    nvcvAllocatorDecRef(allocatorHandle, NULL);
    allocatorHandle = NULL;

    // Destroy CUPVA objects
    if (stream != NULL)
    {
        CupvaStreamDestroy(stream);
        stream = NULL;
    }

    if (sync != NULL)
    {
        CupvaSyncObjDestroy(sync);
        sync = NULL;
    }

    printf("Successfully torn down PVA workload parameters\n");
}

// ============================================================================
// PVA Workload Functions
// ============================================================================

// Function to create all PVA workloads at once
namespace {

// Prints why a Capon operator rejected the pipeline's configuration (init() then returns
// RADAR_PIPELINE_CREATE_ERR_CAPON_CONFIG): the operators' own limits are the single check.
void printCaponRejection(const char *op)
{
    char msg[NVCV_MAX_STATUS_MESSAGE_LENGTH] = {};
    nvcvPeekAtLastErrorMessage(msg, sizeof(msg));
    std::fprintf(stderr, "radar pipeline: %s does not support this Capon configuration: %s\n", op, msg);
}

// Submits the queued work with the fence request and waits for it. Fails if either call fails or the queued
// work did not complete, so the host never reads the outputs of a failed run.
int32_t flushAndWait(RadarPipelinePVAWorkloadParams &w)
{
    cupvaCmd_t const *cmds[1]  = {&w.rf};
    cupvaCmdStatus_t status[1] = {NULL};
    bool completed             = false;
    cupvaError_t e             = CupvaStreamSubmit(w.stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1);
    if (e == CUPVA_ERROR_NONE)
        e = CupvaFenceWait(&w.fence, -1, &completed);
    if (e != CUPVA_ERROR_NONE || !completed)
    {
        std::fprintf(stderr, "radar pipeline: PVA work did not complete (error %d)\n", static_cast<int>(e));
        return -1;
    }
    return 0;
}

// Fills the ROI preprocessing inputs for this frame from the pre-DOA outputs: the ROI count and, per ROI, the
// detected Doppler bin, the element snapshot (permuted to vc = rx*numTx + tx) and the range-Doppler plane at
// its range bin. Returns the ROI count (the detections, capped at maxRoi), or -1.
int32_t marshalCaponInputs(RadarPipelineTensors &t, const CaponROIPreprocConfig &cfg)
{
    void *detCountPtr = nullptr, *detListPtr = nullptr, *snapPtr = nullptr, *rdPtr = nullptr;
    void *roiCountPtr = nullptr, *detDopPtr = nullptr, *elemPtr = nullptr, *planePtr = nullptr;
    NVCVTensorData rdData, snapRoiData;
    if (!radarTensorHostPtr(t.detectionCountTensorHandle, &detCountPtr) ||
        !radarTensorHostPtr(t.detectionListTensorHandle, &detListPtr) ||
        !radarTensorHostPtr(t.snapshotsTensorHandle, &snapPtr) ||
        !radarTensorHostPtr(t.rangeDopplerMapTensorHandle, &rdPtr, &rdData) ||
        !radarTensorHostPtr(t.caponROICountTensorHandle, &roiCountPtr) ||
        !radarTensorHostPtr(t.caponDetDopTensorHandle, &detDopPtr) ||
        !radarTensorHostPtr(t.caponElemTensorHandle, &elemPtr) ||
        !radarTensorHostPtr(t.caponPlaneTensorHandle, &planePtr) ||
        nvcvTensorExportData(t.caponSnapshotsRoiTensorHandle, &snapRoiData) != NVCV_SUCCESS)
    {
        std::fprintf(stderr, "radar pipeline: Capon ROI tensors are not host-accessible\n");
        return -1;
    }

    const int32_t detectionCount = *static_cast<const int32_t *>(detCountPtr);
    const int32_t maxRoi         = static_cast<int32_t>(snapRoiData.shape[0]);
    int32_t roiNum               = detectionCount < 0 ? 0 : detectionCount;
    if (roiNum > maxRoi)
    {
        if (!t.caponMaxRoiNoticed)
        {
            std::fprintf(stderr,
                         "radar pipeline: %d detections exceed the Capon maxRoi %d; processing the first %d (not "
                         "reported again)\n",
                         detectionCount, maxRoi, maxRoi);
            t.caponMaxRoiNoticed = true;
        }
        roiNum = maxRoi;
    }
    *static_cast<int32_t *>(roiCountPtr) = roiNum;
    if (roiNum == 0)
        return 0;

    const CaponPreprocGeometry g = caponPreprocGeometry(cfg);
    const int32_t numTx = g.numTx, numRx = g.numRx, numChirps = g.numChirps, nvc = g.nvc;
    const int32_t numRangeBins = static_cast<int32_t>(rdData.shape[0]); // [range][chirp][rx] (8 RX layout)

    // Normalize exactly as the CPU reference does (radarDequantNormalized), so both see identical inputs: the
    // element snapshots of this frame's ROIs, and the planes with the scale of the whole cube. Only the
    // gathered planes are converted.
    radarDequantNormalized(static_cast<const int32_t *>(snapPtr), static_cast<size_t>(roiNum) * numTx * numRx * 2u,
                           t.caponSnapScratch);
    const int32_t *rd   = static_cast<const int32_t *>(rdPtr);
    const float rdScale = radarDequantScale(rd, static_cast<size_t>(numRangeBins) * numChirps * numRx * 2u);

    const int32_t *detList = static_cast<const int32_t *>(detListPtr);
    int32_t *detDop        = static_cast<int32_t *>(detDopPtr);
    float *elem            = static_cast<float *>(elemPtr);
    float *plane           = static_cast<float *>(planePtr);
    const float *snap      = t.caponSnapScratch.data();
    const size_t planeInts = static_cast<size_t>(numChirps) * numRx * 2u;
    for (int32_t d = 0; d < roiNum; ++d)
    {
        const int32_t rangeIdx = detList[d * 2 + 0];
        const int32_t doppler  = detList[d * 2 + 1];
        if (rangeIdx < 0 || rangeIdx >= numRangeBins || doppler < 0 || doppler >= numChirps)
        {
            std::fprintf(stderr, "radar pipeline: detection %d at (range %d, Doppler %d) is outside the cube\n", d,
                         rangeIdx, doppler);
            return -1;
        }
        detDop[d] = doppler;

        // Element snapshot: CHW [d][tx][rx] -> vc-index i = rx*numTx + tx.
        for (int32_t i = 0; i < nvc; ++i)
        {
            const size_t src = (static_cast<size_t>(d * numTx + i % numTx) * numRx + i / numTx) * 2u;
            const size_t dst = (static_cast<size_t>(d) * nvc + i) * 2u;
            elem[dst + 0]    = snap[src + 0];
            elem[dst + 1]    = snap[src + 1];
        }

        // Range-Doppler plane [numChirps, numRx] at the detection's range bin.
        const int32_t *src = rd + static_cast<size_t>(rangeIdx) * planeInts;
        float *dst         = plane + static_cast<size_t>(d) * planeInts;
        for (size_t k = 0; k < planeInts; ++k) dst[k] = static_cast<float>(src[k]) * rdScale;
    }
    return roiNum;
}

} // namespace

int radarCreatePvaWorkloads(RadarPipelineOperators &operators, RadarPipelineTensorRequirements &tensorReqs,
                            PVARangeFFTParams rangeParams, PVADopplerFFTParams dopplerParams,
                            PVARadarNCIParams nciParams, PVARadarCFARType cfarType,
                            PvaRadarCFARParams const *cfarParams, PVASnapshotExtractionParams snapshotExtractionParams,
                            RadarDoaMethod doaMethod, PVADoaAngleFFTParams const *doaAngleFFTParams,
                            PVADoaBartlettBeamformingParams const *bartlettParams,
                            PVATargetProcessingParams const *targetProcessingParams,
                            PVADoaCaponROIParams const *caponParams, CaponROIPreprocConfig const *caponPreproc)
{
    int32_t err = 0;

    // Create PVA RangeFFT operator
    NVCV_CHECK_ERROR_GOTO(pvaRangeFFTCreate(&operators.rangeFFTOperatorHandle, &tensorReqs.rawAdcDataTensorRequirements,
                                            &tensorReqs.rangeFFTWindowTensorRequirements,
                                            &tensorReqs.rangeProfileTensorRequirements, &rangeParams),
                          err, EXIT);

    // Create PVA DopplerFFT operator
    NVCV_CHECK_ERROR_GOTO(
        pvaDopplerFFTCreate(&operators.dopplerFFTOperatorHandle, &tensorReqs.rangeProfileTensorRequirements,
                            &tensorReqs.dopplerFFTWindowTensorRequirements,
                            &tensorReqs.rangeDopplerMapTensorRequirements, &dopplerParams),
        err, EXIT);

    // Create PVA NCI operator
    NVCV_CHECK_ERROR_GOTO(pvaNciCreate(&operators.nciOperatorHandle, &tensorReqs.rangeDopplerMapTensorRequirements,
                                       tensorReqs.outNciTensorReqs.data(), &nciParams),
                          err, EXIT);

    // Create PVA CFAR operator
    NVCV_CHECK_ERROR_GOTO(pvaRadarCFARCreate(&operators.cfarOperatorHandle, &tensorReqs.nciFinalTensorRequirements,
                                             &tensorReqs.foldedDetectionListTensorRequirements, cfarType, cfarParams),
                          err, EXIT);

    // Create PVA Snapshot Extraction operator. No calibration-weights tensor: calibration weights
    // replace (not multiply) the DDMA decode weights, so passing none selects the DDMA path.
    NVCV_CHECK_ERROR_GOTO(
        pvaSnapshotExtractionCreate(
            &operators.snapshotExtractionOperatorHandle, &tensorReqs.foldedDetectionListTensorRequirements,
            &tensorReqs.ddmDopplerOffsetsTensorRequirements, &tensorReqs.nciRxTensorRequirements,
            &tensorReqs.rangeDopplerMapTensorRequirements, &tensorReqs.detectionListTensorRequirements,
            &tensorReqs.snapshotsTensorRequirements, nullptr, &snapshotExtractionParams),
        err, EXIT);

    // Create DOA operator based on selected method
    if (doaMethod == RadarDoaMethod::ANGLE_FFT)
    {
        if (doaAngleFFTParams == nullptr)
        {
            printf("Error: doaAngleFFTParams is required for ANGLE_FFT method\n");
            err = -1;
            goto EXIT;
        }
        // Create PVA DOA Angle FFT operator
        NVCV_CHECK_ERROR_GOTO(
            pvaDoaAngleFFTCreate(&operators.doaAngleFFTOperatorHandle, &tensorReqs.snapshotsTensorRequirements,
                                 &tensorReqs.detectionListTensorRequirements,
                                 &tensorReqs.ddmDopplerOffsetsTensorRequirements,
                                 &tensorReqs.targetAnglesTensorRequirements,
                                 &tensorReqs.targetIndexMapTensorRequirements, doaAngleFFTParams),
            err, EXIT);
    }
    else if (doaMethod == RadarDoaMethod::BARTLETT)
    {
        if (bartlettParams == nullptr)
        {
            printf("Error: bartlettParams is required for BARTLETT method\n");
            err = -1;
            goto EXIT;
        }
        // Create PVA Bartlett Beamforming operator
        NVCV_CHECK_ERROR_GOTO(
            pvaDoaBartlettBeamformingCreate(
                &operators.doaBartlettOperatorHandle, &tensorReqs.snapshotsTensorRequirements,
                &tensorReqs.detectionListTensorRequirements, &tensorReqs.ddmDopplerOffsetsTensorRequirements,
                &tensorReqs.targetAnglesTensorRequirements, &tensorReqs.targetIndexMapTensorRequirements,
                bartlettParams),
            err, EXIT);
    }
    else if (doaMethod == RadarDoaMethod::CAPON_ROI)
    {
        if (caponParams == nullptr)
        {
            printf("Error: caponParams is required for CAPON_ROI method\n");
            err = -1;
            goto EXIT;
        }
        if (caponPreproc == nullptr)
        {
            printf("Error: caponPreproc is required for CAPON_ROI method\n");
            err = -1;
            goto EXIT;
        }

        // Create the Capon ROI beamspace-projection preprocessing operator. It projects the element
        // snapshot / range-Doppler plane into per-detection ROI beamspace on the VPU, writing the
        // snapshotsROI / steeringVectorsROI tensors that the Capon operator consumes. Build
        // its params from the shared preprocessing config (mirrors the standalone sample); maxDet
        // is the N dimension of the snapshotsROI output tensor.
        {
            const CaponPreprocGeometry g = caponPreprocGeometry(*caponPreproc);

            PVACaponROIPreprocessParams pp;
            std::memset(&pp, 0, sizeof(pp));
            pp.numTx                = g.numTx;
            pp.numRx                = g.numRx;
            pp.numChirps            = g.numChirps;
            pp.numAzVirt            = g.numAzVirt;
            pp.numElVirt            = g.numElVirt;
            pp.beamAz               = g.beamAz;
            pp.beamEl               = g.beamEl;
            pp.roiAz                = g.roiAz;
            pp.roiEl                = g.roiEl;
            pp.svDim                = g.svDim;
            pp.numAzBins            = g.numAzBins;
            pp.numElBins            = g.numElBins;
            pp.numDopplerBinsCovMat = caponPreproc->numDopplerBinsCovMat;
            pp.dopplerNeighborhood  = caponPreproc->dopplerNeighborhood;
            pp.enableFBA            = caponPreproc->enableFBA != 0;
            pp.inputQBits           = caponPreproc->inputQBits;
            pp.steeringQBits        = caponPreproc->steeringQBits;
            pp.maxDet               = static_cast<int32_t>(tensorReqs.caponSnapshotsRoiTensorRequirements.shape[0]);
            // Batched FFT-64 coarse DoA. Create() rejects it unless the geometry is azimuth-only with a
            // 64-tap, 64-bin azimuth beamspace, so enable it only then.
            if (g.numElVirt == 1 && g.beamAz == 64 && g.numAzVirt == 64)
            {
                pp.coarseDoAUseFft = true;
            }

            // DDM decode terms, built once in the factory and shared with the reference.
            for (int tx = 0; tx < g.numTx; ++tx)
            {
                pp.relDopplerOffset[tx] = caponPreproc->relDopplerOffset[tx];
                pp.ddmaWeightRe[tx]     = caponPreproc->ddmaWeightRe[tx];
                pp.ddmaWeightIm[tx]     = caponPreproc->ddmaWeightIm[tx];
            }

            // waz/wel are NULL: the operator generates the DFT twiddle tables (useCustomTwiddles == false).
            if (pvaCaponROIPreprocessCreate(
                    &operators.caponPreprocOperatorHandle, nullptr, nullptr,
                    &tensorReqs.caponROICountTensorRequirements, &tensorReqs.caponDetDopTensorRequirements,
                    &tensorReqs.caponElemTensorRequirements, &tensorReqs.caponPlaneTensorRequirements,
                    &tensorReqs.caponSteeringLutTensorRequirements, &tensorReqs.caponSnapshotsRoiTensorRequirements,
                    &tensorReqs.caponSteeringVectorsRoiTensorRequirements, &pp) != NVCV_SUCCESS)
            {
                printCaponRejection("Capon ROI preprocessing");
                err = RADAR_PIPELINE_CREATE_ERR_CAPON_CONFIG;
                goto EXIT;
            }
        }

        // Create the PVA Capon DoA-on-ROI operator. Inputs are the ROI beamspace snapshots and
        // steering vectors from the preprocessing operator; outputs are top-3 peak power / az / el
        // bin indices per ROI.
#if CUPVA_LLVM_SUPPORTED == 1
        if (pvaDoaCaponROICreate(&operators.doaCaponROIOperatorHandle, &tensorReqs.caponSnapshotsRoiTensorRequirements,
                                 &tensorReqs.caponSteeringVectorsRoiTensorRequirements,
                                 &tensorReqs.caponPeakPowerTensorRequirements, &tensorReqs.caponAzBinTensorRequirements,
                                 &tensorReqs.caponElBinTensorRequirements, caponParams) != NVCV_SUCCESS)
        {
            printCaponRejection("Capon DoA");
            err = RADAR_PIPELINE_CREATE_ERR_CAPON_CONFIG;
            goto EXIT;
        }
#else
        (void)caponParams; // unreachable: init() rejects Capon without LLVM support
        err = RADAR_PIPELINE_CREATE_ERR_DOA_METHOD;
        goto EXIT;
#endif
    }

    // Create standalone Target Processing operator. Capon ROI produces its own per-ROI peak
    // outputs and does not feed the target-processing stage, so skip it for CAPON_ROI.
    if (doaMethod != RadarDoaMethod::CAPON_ROI)
    {
        // Determine NCI tensor requirements for standalone Target Processing
        bool enableQuadraticInterpolation =
            targetProcessingParams != nullptr && targetProcessingParams->enableQuadraticInterpolation;
        const NVCVTensorRequirements *nciReqs =
            enableQuadraticInterpolation ? &tensorReqs.nciFinalTensorRequirements : nullptr;

        NVCV_CHECK_ERROR_GOTO(
            pvaTargetProcessingCreate(
                &operators.targetProcessingOperatorHandle, &tensorReqs.detectionListTensorRequirements,
                &tensorReqs.targetIndexMapTensorRequirements, &tensorReqs.targetAnglesTensorRequirements,
                &tensorReqs.ddmDopplerOffsetsTensorRequirements, &tensorReqs.targetListTensorRequirements, nciReqs,
                targetProcessingParams),
            err, EXIT);
    }

EXIT:
    return err;
}

// ============================================================================
// Capon ROI submit
// ============================================================================

// Submit the Capon DoA-on-ROI path: run the pre-DOA stages, marshal the per-detection inputs on the host, then
// run the ROI preprocessing and Capon operators. Capon produces no target list, so the target count is 0 (see
// IRadarPipeline::get_target_count()); the ROI count goes to RADAR_PIPELINE_TENSOR_CAPON_ROI_COUNT.
static int radarSubmitCaponROI(RadarPipelineOperators &operators, RadarPipelineTensors &tensors,
                               RadarPipelinePVAWorkloadParams &pvaWorkloadParams,
                               CaponROIPreprocConfig const &caponPreproc)
{
    void *targetCountPtr = nullptr;
    if (!radarTensorHostPtr(tensors.targetCountTensorHandle, &targetCountPtr))
        return -1;
    *static_cast<int32_t *>(targetCountPtr) = 0;

    // 1) Run the queued pre-DOA stages, so their outputs are readable on the host.
    if (flushAndWait(pvaWorkloadParams) != 0)
        return -1;

    // 2) Host marshalling.
    const int32_t roiNum = marshalCaponInputs(tensors, caponPreproc);
    if (roiNum <= 0)
        return roiNum; // -1 on error; with no ROIs both Capon operators are skipped

    // 3) ROI preprocessing, then Capon, in order on the same stream.
    if (pvaCaponROIPreprocessSubmit(operators.caponPreprocOperatorHandle, pvaWorkloadParams.stream, nullptr, nullptr,
                                    tensors.caponROICountTensorHandle, tensors.caponDetDopTensorHandle,
                                    tensors.caponElemTensorHandle, tensors.caponPlaneTensorHandle,
                                    tensors.caponSteeringLutTensorHandle, tensors.caponSnapshotsRoiTensorHandle,
                                    tensors.caponSteeringVectorsRoiTensorHandle) != NVCV_SUCCESS)
    {
        printCaponRejection("Capon ROI preprocessing submit");
        return -1;
    }
#if CUPVA_LLVM_SUPPORTED == 1
    if (pvaDoaCaponROISubmit(operators.doaCaponROIOperatorHandle, pvaWorkloadParams.stream,
                             tensors.caponSnapshotsRoiTensorHandle, tensors.caponSteeringVectorsRoiTensorHandle,
                             tensors.caponPeakPowerTensorHandle, tensors.caponAzBinTensorHandle,
                             tensors.caponElBinTensorHandle) != NVCV_SUCCESS)
    {
        printCaponRejection("Capon DoA submit");
        return -1;
    }
    return flushAndWait(pvaWorkloadParams);
#else
    return -1; // unreachable: init() rejects Capon without LLVM support
#endif
}

// Function to submit all PVA workloads at once
int radarSubmitPvaWorkloads(RadarPipelineOperators &operators, RadarPipelineTensors &tensors,
                            RadarPipelinePVAWorkloadParams &pvaWorkloadParams, RadarDoaMethod doaMethod,
                            CaponROIPreprocConfig const *caponPreproc)
{
    int32_t err = 0;
    // Set up command arrays for submission
    cupvaCmd_t *cmds[1];
    cupvaCmdStatus_t status[1];
    cmds[0]          = &pvaWorkloadParams.rf;
    status[0]        = NULL;
    bool waitSuccess = false;

    NVCVTensorHandle inNciTensorHandles[]  = {tensors.rangeDopplerMapTensorHandle};
    NVCVTensorHandle outNciTensorHandles[] = {tensors.nciRxTensorHandle, tensors.nciFinalTensorHandle, NULL};

    // Submit PVA RangeFFT operator
    NVCV_CHECK_ERROR_GOTO(
        pvaRangeFFTSubmit(operators.rangeFFTOperatorHandle, pvaWorkloadParams.stream, tensors.inRangeFFTTensorHandle,
                          tensors.rangeFFTWindowTensorHandle, tensors.rangeProfileTensorHandle),
        err, EXIT);

    // Submit PVA DopplerFFT operator
    NVCV_CHECK_ERROR_GOTO(pvaDopplerFFTSubmit(operators.dopplerFFTOperatorHandle, pvaWorkloadParams.stream,
                                              tensors.rangeProfileTensorHandle, tensors.dopplerFFTWindowTensorHandle,
                                              tensors.rangeDopplerMapTensorHandle),
                          err, EXIT);

    // Submit PVA NCI operator
    NVCV_CHECK_ERROR_GOTO(
        pvaNciSubmit(operators.nciOperatorHandle, pvaWorkloadParams.stream, inNciTensorHandles, outNciTensorHandles),
        err, EXIT);

    // Submit PVA CFAR operator
    NVCV_CHECK_ERROR_GOTO(
        pvaRadarCFARSubmit(operators.cfarOperatorHandle, pvaWorkloadParams.stream, tensors.nciFinalTensorHandle,
                           tensors.foldedDetectionListTensorHandle, tensors.detectionCountTensorHandle),
        err, EXIT);

    // Submit PVA Snapshot Extraction operator
    NVCV_CHECK_ERROR_GOTO(
        pvaSnapshotExtractionSubmit(operators.snapshotExtractionOperatorHandle, pvaWorkloadParams.stream,
                                    tensors.detectionCountTensorHandle, tensors.foldedDetectionListTensorHandle,
                                    tensors.ddmDopplerOffsetsTensorHandle, tensors.nciRxTensorHandle,
                                    tensors.rangeDopplerMapTensorHandle, NULL, tensors.detectionListTensorHandle,
                                    tensors.snapshotsTensorHandle),
        err, EXIT);

    // Capon ROI: flush the pre-DOA stages, submit the ROI preprocessing and Capon operators, and return
    // (it does not use the target-processing stage).
    if (doaMethod == RadarDoaMethod::CAPON_ROI)
    {
        if (caponPreproc == nullptr)
        {
            printf("Error: caponPreproc is required for CAPON_ROI method\n");
            err = -1;
            goto EXIT;
        }
        err = radarSubmitCaponROI(operators, tensors, pvaWorkloadParams, *caponPreproc);
        goto EXIT;
    }

    // Submit DOA operator based on selected method
    if (doaMethod == RadarDoaMethod::ANGLE_FFT)
    {
        // Submit PVA DOA Angle FFT operator
        NVCV_CHECK_ERROR_GOTO(
            pvaDoaAngleFFTSubmit(
                operators.doaAngleFFTOperatorHandle, pvaWorkloadParams.stream, tensors.detectionCountTensorHandle,
                tensors.snapshotsTensorHandle, tensors.detectionListTensorHandle, tensors.ddmDopplerOffsetsTensorHandle,
                tensors.targetCountTensorHandle, tensors.targetAnglesTensorHandle, tensors.targetIndexMapTensorHandle),
            err, EXIT);
    }
    else if (doaMethod == RadarDoaMethod::BARTLETT)
    {
        if (tensors.steeringVectorsTensorHandle == nullptr || tensors.azimuthBinsTensorHandle == nullptr ||
            tensors.elevationBinsTensorHandle == nullptr)
        {
            printf("Error: Bartlett tensors not initialized. Create tensors with RadarPipelineBartlettConfig.\n");
            err = -1;
            goto EXIT;
        }
        // Submit PVA Bartlett Beamforming operator
        NVCV_CHECK_ERROR_GOTO(
            pvaDoaBartlettBeamformingSubmit(operators.doaBartlettOperatorHandle, pvaWorkloadParams.stream,
                                            tensors.detectionCountTensorHandle, tensors.snapshotsTensorHandle,
                                            tensors.steeringVectorsTensorHandle, tensors.azimuthBinsTensorHandle,
                                            tensors.elevationBinsTensorHandle, tensors.detectionListTensorHandle,
                                            tensors.ddmDopplerOffsetsTensorHandle, tensors.targetCountTensorHandle,
                                            tensors.targetAnglesTensorHandle, tensors.targetIndexMapTensorHandle),
            err, EXIT);
    }

    // Submit Target Processing operator
    NVCV_CHECK_ERROR_GOTO(
        pvaTargetProcessingSubmit(operators.targetProcessingOperatorHandle, pvaWorkloadParams.stream,
                                  tensors.targetCountTensorHandle, tensors.detectionCountTensorHandle,
                                  tensors.detectionListTensorHandle, tensors.targetIndexMapTensorHandle,
                                  tensors.targetAnglesTensorHandle, tensors.ddmDopplerOffsetsTensorHandle,
                                  tensors.nciFinalTensorHandle, tensors.targetListTensorHandle),
        err, EXIT);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(pvaWorkloadParams.stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err,
                           EXIT);

    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&pvaWorkloadParams.fence, -1, &waitSuccess), err, EXIT);

EXIT:
    return err;
}
