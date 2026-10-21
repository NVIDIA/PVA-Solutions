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

#include <ErrorCheckMacros.h>
#include <OpDoaDAFFT.h>
#include <OpDopplerFFT.h>
#include <OpNci.h>
#include <OpPeakDetection.h>
#include <OpRangeFFT.h>
#include <PvaAllocator.h>

#include <cstdio>
#include <iterator> // For std::size
#include <string>

// ============================================================================
// RadarPipelineOperators Implementation
// ============================================================================

// Function to initialize all operator handles to NULL
void RadarPipelineOperators::initRadarPipelineOperators()
{
    rangeFFTOperatorHandle      = NULL;
    dopplerFFTOperatorHandle    = NULL;
    nciOperatorHandle           = NULL;
    peakDetectionOperatorHandle = NULL;
    doaOperatorHandle           = NULL;
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

    if (peakDetectionOperatorHandle != NULL)
    {
        nvcvOperatorDestroy(peakDetectionOperatorHandle);
        peakDetectionOperatorHandle = NULL;
    }

    if (doaOperatorHandle != NULL)
    {
        nvcvOperatorDestroy(doaOperatorHandle);
        doaOperatorHandle = NULL;
    }
}

// ============================================================================
// RadarPipelinePVAWorkloadParams Implementation
// ============================================================================

// Function to initialize RadarPipelinePVAWorkloadParams
void RadarPipelinePVAWorkloadParams::initRadarPipelinePVAWorkloadParams()
{
    int32_t err = 0;

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
int radar_create_pva_workloads(RadarPipelineOperators &operators, RadarPipelineTensors &tensors, int32_t NofTx,
                               int32_t repeatFold)
{
    PVARadarNCIParams nciParams = {
        .repeatFold             = repeatFold,
        .noiseEstimationEnabled = true,
        .inputLayout            = PVA_FFT_LAYOUT_RANGE_RX_DOPPLER,
    };
    PVARangeFFTParams rangeParams     = {.windowType   = PVA_BATCH_FFT_WINDOW_HANNING,
                                         .inputLayout  = PVA_FFT_LAYOUT_DOPPLER_RX_RANGE,
                                         .outputLayout = PVA_FFT_LAYOUT_DOPPLER_RX_RANGE};
    PVADopplerFFTParams dopplerParams = {.windowType   = PVA_BATCH_FFT_WINDOW_HANNING,
                                         .inputLayout  = PVA_FFT_LAYOUT_DOPPLER_RX_RANGE,
                                         .outputLayout = PVA_FFT_LAYOUT_RANGE_RX_DOPPLER};
    int32_t err                       = 0;

    RadarPipelineTensorRequirements &tensorReqs = tensors.m_tensorReqs;

    // Create PVA RangeFFT operator
    NVCV_CHECK_ERROR_GOTO(
        pvaRangeFFTCreate(&operators.rangeFFTOperatorHandle, &tensorReqs.inRangeFFTRequirements,
                          &tensorReqs.winRangeFFTRequirements, &tensorReqs.outRangeFFTRequirements, &rangeParams),
        err, EXIT);

    // Create PVA DopplerFFT operator
    NVCV_CHECK_ERROR_GOTO(pvaDopplerFFTCreate(&operators.dopplerFFTOperatorHandle, &tensorReqs.inDopplerFFTRequirements,
                                              &tensorReqs.winDopplerFFTRequirements,
                                              &tensorReqs.outDopplerFFTRequirements, &dopplerParams),
                          err, EXIT);

    // Create PVA NCI operator
    NVCV_CHECK_ERROR_GOTO(pvaNciCreate(&operators.nciOperatorHandle, &tensorReqs.inNciTensorRequirements,
                                       tensorReqs.outNciTensorReqs.data(), &nciParams),
                          err, EXIT);

    // Create PVA Peak Detection operator
    NVCV_CHECK_ERROR_GOTO(
        pvaPeakDetectionCreate(&operators.peakDetectionOperatorHandle, &tensorReqs.inNciTensorRequirements,
                               &tensorReqs.outPeakSnapTensorRequirements, NofTx, repeatFold),
        err, EXIT);

    // Create PVA DOA operator
    NVCV_CHECK_ERROR_GOTO(
        pvaDoaDAFFTCreate(&operators.doaOperatorHandle, tensorReqs.inDOATensorReqs.data(),
                          &tensorReqs.outDOATensorRequirements, tensorReqs.inDOATensorReqs.size(), &tensors.GP),
        err, EXIT);

EXIT:
    return err;
}

// Function to submit all PVA workloads at once
int radar_submit_pva_workloads(RadarPipelineOperators &operators, RadarPipelineTensors &tensors,
                               RadarPipelinePVAWorkloadParams &pvaWorkloadParams)
{
    int32_t err = 0;
    // Set up command arrays for submission
    cupvaCmd_t *cmds[1];
    cupvaCmdStatus_t status[1];
    cmds[0]          = &pvaWorkloadParams.rf;
    status[0]        = NULL;
    bool waitSuccess = false;

    NVCVTensorHandle inNciTensorHandles[]            = {tensors.inNciTensorHandle};
    NVCVTensorHandle outNciTensorHandles[]           = {tensors.outNciRxTensorHandle, tensors.outNciFinalTensorHandle,
                                                        tensors.outNoiseEstimateTensorHandle};
    NVCVTensorHandle inPeakDetectionTensorHandles[]  = {tensors.inNciTensorHandle, tensors.outNciRxTensorHandle,
                                                        tensors.outNciFinalTensorHandle,
                                                        tensors.outNoiseEstimateTensorHandle};
    NVCVTensorHandle outPeakDetectionTensorHandles[] = {
        tensors.outPeakCountTensorHandle, tensors.outPeakIndexTensorHandle, tensors.outPeakSnapTensorHandle};

    NVCVTensorHandle inDOATensorHandles[] = {tensors.outPeakCountTensorHandle, tensors.inDOACalibVectorTensorHandle,
                                             tensors.outPeakIndexTensorHandle, tensors.outPeakSnapTensorHandle,
                                             tensors.outNciFinalTensorHandle};

    // Submit PVA RangeFFT operator
    NVCV_CHECK_ERROR_GOTO(
        pvaRangeFFTSubmit(operators.rangeFFTOperatorHandle, pvaWorkloadParams.stream, tensors.inRangeFFTTensorHandle,
                          tensors.winRangeFFTTensorHandle, tensors.outRangeFFTTensorHandle),
        err, EXIT);

    // Submit PVA DopplerFFT operator
    NVCV_CHECK_ERROR_GOTO(pvaDopplerFFTSubmit(operators.dopplerFFTOperatorHandle, pvaWorkloadParams.stream,
                                              tensors.inDopplerFFTTensorHandle, tensors.winDopplerFFTTensorHandle,
                                              tensors.outDopplerFFTTensorHandle),
                          err, EXIT);

    // Submit PVA NCI operator
    NVCV_CHECK_ERROR_GOTO(
        pvaNciSubmit(operators.nciOperatorHandle, pvaWorkloadParams.stream, inNciTensorHandles, outNciTensorHandles),
        err, EXIT);

    // Submit PVA Peak Detection operator
    NVCV_CHECK_ERROR_GOTO(pvaPeakDetectionSubmit(operators.peakDetectionOperatorHandle, pvaWorkloadParams.stream,
                                                 inPeakDetectionTensorHandles, outPeakDetectionTensorHandles, 4, 3),
                          err, EXIT);

    // Submit PVA DOA operator
    NVCV_CHECK_ERROR_GOTO(pvaDoaDAFFTSubmit(operators.doaOperatorHandle, pvaWorkloadParams.stream, inDOATensorHandles,
                                            tensors.outDOATensorHandle, std::size(inDOATensorHandles)),
                          err, EXIT);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(pvaWorkloadParams.stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err,
                           EXIT);
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&pvaWorkloadParams.fence, -1, &waitSuccess), err, EXIT);

EXIT:
    return err;
}
