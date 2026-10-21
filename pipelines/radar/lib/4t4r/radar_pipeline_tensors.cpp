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

#include "doa_target_processing_ref.h"
#include "doppler_fft_ref.h"
#include "range_fft_ref.h"

#include <ErrorCheckMacros.h>
#include <PvaAllocator.h>
#include <nvcv/Tensor.h>

#include <string>

// ============================================================================
// RadarPipelineTensorRequirements Implementation
// ============================================================================

// Function to create tensor requirements without allocating actual tensors
void RadarPipelineTensorRequirements::createTensorRequirements(const int32_t Ns, const int32_t NofRx, const int32_t Nr,
                                                               const int32_t NofRangeBins, const int32_t repeatFold,
                                                               const int32_t NofTx)
{
    int32_t err = 0;

    // Define data types
    NVCVDataType inRangeFFTDataType             = NVCV_DATA_TYPE_S32;
    NVCVDataType winRangeFFTDataType            = NVCV_DATA_TYPE_S32;
    NVCVDataType outRangeFFTDataType            = NVCV_DATA_TYPE_2S32;
    NVCVDataType inDopplerFFTDataType           = NVCV_DATA_TYPE_2S32;
    NVCVDataType winDopplerFFTDataType          = NVCV_DATA_TYPE_S32;
    NVCVDataType outDopplerFFTDataType          = NVCV_DATA_TYPE_2S32;
    NVCVDataType nciInputDataType               = NVCV_DATA_TYPE_2S32;
    NVCVDataType nciOutputDataType              = NVCV_DATA_TYPE_U32;
    NVCVDataType peakIndexDataType              = NVCV_DATA_TYPE_U32;
    NVCVDataType peakSnapDataType               = NVCV_DATA_TYPE_2S32;
    NVCVDataType inDOACalibVectorTensorDataType = NVCV_DATA_TYPE_2S32;
    NVCVDataType outDOATensorDataType           = NVCV_DATA_TYPE_F32;

    // Define tensor ranks
    const int32_t tensorRank              = 3;
    const int32_t winRangeFFTTensorRank   = 1;
    const int32_t winDopplerFFTTensorRank = 1;
    const int32_t doaTensorRank           = 2;

    // Define shapes
    int64_t inRangeFFTShape[]             = {Nr, NofRx, Ns};
    int64_t winRangeFFTShape[]            = {Ns};
    int64_t outRangeFFTShape[]            = {Nr, NofRx, NofRangeBins};
    int64_t inDopplerFFTShape[]           = {Nr, NofRx, NofRangeBins};
    int64_t winDopplerFFTShape[]          = {Nr};
    int64_t outDopplerFFTShape[]          = {NofRangeBins, NofRx, Nr};
    int64_t nciInputShape[]               = {NofRangeBins, NofRx, Nr};
    int64_t outNciRxTensorShape[]         = {NofRangeBins, Nr};
    int64_t outNciFinalTensorShape[]      = {NofRangeBins, Nr / repeatFold};
    int64_t outNoiseEstimateTensorShape[] = {NofRangeBins};
    int64_t outPeakCountTensorShape[]     = {1};
    int64_t outPeakIndexTensorShape[]     = {PVA_RADAR_PEAKDET_NUM_PEAK_INDICES, PVA_RADAR_MAX_TARGET_COUNT};
    int64_t outPeakSnapTensorShape[]      = {PVA_RADAR_MAX_TARGET_COUNT, 16};
    int64_t inDOACalibVectorTensorShape[] = {1, 16};
    int64_t outDOATensorShape[]           = {PVA_RADAR_NUM_TARGET_DETECTION_PROPERTIES, PVA_RADAR_MAX_TARGET_COUNT};

    // Declare tensor layout variables
    NVCVTensorLayout tensorLayout;
    NVCVTensorLayout winRangeFFTLayout;
    NVCVTensorLayout winDopplerFFTLayout;
    NVCVTensorLayout outNciRxTensorLayout;
    NVCVTensorLayout outNciFinalTensorLayout;
    NVCVTensorLayout outNoiseEstimateTensorLayout;
    NVCVTensorLayout nciInputTensorLayout;
    NVCVTensorLayout outPeakCountTensorLayout;
    NVCVTensorLayout outPeakIndexTensorLayout;
    NVCVTensorLayout outPeakSnapTensorLayout;
    NVCVTensorLayout inDOACalibVectorTensorLayout;
    NVCVTensorLayout outDOATensorLayout;

    // Create tensor layouts
    nvcvTensorLayoutMake("HCW", &tensorLayout);
    nvcvTensorLayoutMake("W", &winRangeFFTLayout);
    nvcvTensorLayoutMake("W", &winDopplerFFTLayout);
    nvcvTensorLayoutMake("HW", &outNciRxTensorLayout);
    nvcvTensorLayoutMake("HW", &outNciFinalTensorLayout);
    nvcvTensorLayoutMake("W", &outNoiseEstimateTensorLayout);
    nvcvTensorLayoutMake("HCW", &nciInputTensorLayout);
    nvcvTensorLayoutMake("W", &outPeakCountTensorLayout);
    nvcvTensorLayoutMake("HW", &outPeakIndexTensorLayout);
    nvcvTensorLayoutMake("HW", &outPeakSnapTensorLayout);
    nvcvTensorLayoutMake("HW", &inDOACalibVectorTensorLayout);
    nvcvTensorLayoutMake("HW", &outDOATensorLayout);

    // Calculate tensor requirements (no actual allocation)
    nvcvTensorCalcRequirementsPva(tensorRank, inRangeFFTShape, inRangeFFTDataType, tensorLayout, 0, 0,
                                  &inRangeFFTRequirements);
    nvcvTensorCalcRequirementsPva(tensorRank, outRangeFFTShape, outRangeFFTDataType, tensorLayout, 0, 0,
                                  &outRangeFFTRequirements);
    nvcvTensorCalcRequirementsPva(winRangeFFTTensorRank, winRangeFFTShape, winRangeFFTDataType, winRangeFFTLayout, 0, 0,
                                  &winRangeFFTRequirements);
    nvcvTensorCalcRequirementsPva(tensorRank, inDopplerFFTShape, inDopplerFFTDataType, tensorLayout, 0, 0,
                                  &inDopplerFFTRequirements);
    nvcvTensorCalcRequirementsPva(winDopplerFFTTensorRank, winDopplerFFTShape, winDopplerFFTDataType,
                                  winDopplerFFTLayout, 0, 0, &winDopplerFFTRequirements);
    nvcvTensorCalcRequirementsPva(tensorRank, outDopplerFFTShape, outDopplerFFTDataType, tensorLayout, 0, 0,
                                  &outDopplerFFTRequirements);
    nvcvTensorCalcRequirementsPva(sizeof(outNciRxTensorShape) / sizeof(outNciRxTensorShape[0]), outNciRxTensorShape,
                                  nciOutputDataType, outNciRxTensorLayout, 0, 0, &outNciRxTensorRequirements);
    nvcvTensorCalcRequirementsPva(sizeof(outNciFinalTensorShape) / sizeof(outNciFinalTensorShape[0]),
                                  outNciFinalTensorShape, nciOutputDataType, outNciFinalTensorLayout, 0, 0,
                                  &outNciFinalTensorRequirements);
    nvcvTensorCalcRequirementsPva(sizeof(outNoiseEstimateTensorShape) / sizeof(outNoiseEstimateTensorShape[0]),
                                  outNoiseEstimateTensorShape, nciOutputDataType, outNoiseEstimateTensorLayout, 0, 0,
                                  &outNoiseEstimateTensorRequirements);
    nvcvTensorCalcRequirementsPva(tensorRank, nciInputShape, nciInputDataType, nciInputTensorLayout, 0, 0,
                                  &inNciTensorRequirements);
    nvcvTensorCalcRequirementsPva(sizeof(outPeakCountTensorShape) / sizeof(outPeakCountTensorShape[0]),
                                  outPeakCountTensorShape, NVCV_DATA_TYPE_U32, outPeakCountTensorLayout, 0, 0,
                                  &outPeakCountTensorRequirements);
    nvcvTensorCalcRequirementsPva(sizeof(outPeakIndexTensorShape) / sizeof(outPeakIndexTensorShape[0]),
                                  outPeakIndexTensorShape, peakIndexDataType, outPeakIndexTensorLayout, 0, 0,
                                  &outPeakIndexTensorRequirements);
    nvcvTensorCalcRequirementsPva(sizeof(outPeakSnapTensorShape) / sizeof(outPeakSnapTensorShape[0]),
                                  outPeakSnapTensorShape, peakSnapDataType, outPeakSnapTensorLayout, 0, 0,
                                  &outPeakSnapTensorRequirements);
    nvcvTensorCalcRequirementsPva(doaTensorRank, inDOACalibVectorTensorShape, inDOACalibVectorTensorDataType,
                                  inDOACalibVectorTensorLayout, 0, 0, &inDOACalibVectorTensorRequirements);
    nvcvTensorCalcRequirementsPva(doaTensorRank, outDOATensorShape, outDOATensorDataType, outDOATensorLayout, 0, 0,
                                  &outDOATensorRequirements);

    // Set up NCI output tensor requirements pointers
    outNciTensorReqs = {&outNciRxTensorRequirements, &outNciFinalTensorRequirements,
                        &outNoiseEstimateTensorRequirements};

    // Set up DOA input tensor requirements pointers
    inDOATensorReqs = {&outPeakCountTensorRequirements, &inDOACalibVectorTensorRequirements,
                       &outPeakIndexTensorRequirements, &outPeakSnapTensorRequirements, &outNciFinalTensorRequirements};
}

// ============================================================================
// RadarPipelineTensors Implementation
// ============================================================================

// Function to initialize all tensor handles to NULL
void RadarPipelineTensors::initRadarPipelineTensors()
{
    // Initialize tensor handles to NULL
    inRangeFFTTensorHandle       = NULL; // not owned; set transiently by radar_pipeline_process()
    winRangeFFTTensorHandle      = NULL;
    outRangeFFTTensorHandle      = NULL;
    refRangeFFTTensorHandle      = NULL;
    inDopplerFFTTensorHandle     = NULL;
    winDopplerFFTTensorHandle    = NULL;
    outDopplerFFTTensorHandle    = NULL;
    refDopplerFFTTensorHandle    = NULL;
    inNciTensorHandle            = NULL;
    outNciRxTensorHandle         = NULL;
    outNciFinalTensorHandle      = NULL;
    outNoiseEstimateTensorHandle = NULL;
    refNciRxTensorHandle         = NULL;
    refNciFinalTensorHandle      = NULL;
    refNoiseEstimateTensorHandle = NULL;
    outPeakCountTensorHandle     = NULL;
    outPeakIndexTensorHandle     = NULL;
    outPeakSnapTensorHandle      = NULL;
    refPeakCountTensorHandle     = NULL;
    refPeakIndexTensorHandle     = NULL;
    refPeakSnapTensorHandle      = NULL;
    inDOACalibVectorTensorHandle = NULL;
    outDOATensorHandle           = NULL;
    refDOATensorHandle           = NULL;
}

// Function to cleanup all allocated resources
void RadarPipelineTensors::cleanupRadarPipelineTensors()
{
    if (winRangeFFTTensorHandle)
        nvcvTensorDecRef(winRangeFFTTensorHandle, NULL);
    if (outRangeFFTTensorHandle)
        nvcvTensorDecRef(outRangeFFTTensorHandle, NULL);
    if (refRangeFFTTensorHandle)
        nvcvTensorDecRef(refRangeFFTTensorHandle, NULL);
    if (winDopplerFFTTensorHandle)
        nvcvTensorDecRef(winDopplerFFTTensorHandle, NULL);
    if (outDopplerFFTTensorHandle)
        nvcvTensorDecRef(outDopplerFFTTensorHandle, NULL);
    if (refDopplerFFTTensorHandle)
        nvcvTensorDecRef(refDopplerFFTTensorHandle, NULL);
    if (outNciRxTensorHandle)
        nvcvTensorDecRef(outNciRxTensorHandle, NULL);
    if (outNciFinalTensorHandle)
        nvcvTensorDecRef(outNciFinalTensorHandle, NULL);
    if (outNoiseEstimateTensorHandle)
        nvcvTensorDecRef(outNoiseEstimateTensorHandle, NULL);
    if (refNciRxTensorHandle)
        nvcvTensorDecRef(refNciRxTensorHandle, NULL);
    if (refNciFinalTensorHandle)
        nvcvTensorDecRef(refNciFinalTensorHandle, NULL);
    if (refNoiseEstimateTensorHandle)
        nvcvTensorDecRef(refNoiseEstimateTensorHandle, NULL);
    if (refPeakCountTensorHandle)
        nvcvTensorDecRef(refPeakCountTensorHandle, NULL);
    if (refPeakIndexTensorHandle)
        nvcvTensorDecRef(refPeakIndexTensorHandle, NULL);
    if (refPeakSnapTensorHandle)
        nvcvTensorDecRef(refPeakSnapTensorHandle, NULL);
    if (outPeakCountTensorHandle)
        nvcvTensorDecRef(outPeakCountTensorHandle, NULL);
    if (outPeakIndexTensorHandle)
        nvcvTensorDecRef(outPeakIndexTensorHandle, NULL);
    if (outPeakSnapTensorHandle)
        nvcvTensorDecRef(outPeakSnapTensorHandle, NULL);
    if (inDOACalibVectorTensorHandle)
        nvcvTensorDecRef(inDOACalibVectorTensorHandle, NULL);
    if (outDOATensorHandle)
        nvcvTensorDecRef(outDOATensorHandle, NULL);
    if (refDOATensorHandle)
        nvcvTensorDecRef(refDOATensorHandle, NULL);

    // Cleanup allocator
    if (m_allocatorHandle)
    {
        nvcvAllocatorDecRef(m_allocatorHandle, NULL);
        m_allocatorHandle = NULL;
    }
}

// Function to create all tensors and calculate requirements
void RadarPipelineTensors::createRadarPipelineTensors()
{
    int32_t err = 0;

    // Range FFT tensors (inRangeFFTTensorHandle is caller-supplied; not constructed here)
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(&m_tensorReqs.outRangeFFTRequirements, m_allocatorHandle, &outRangeFFTTensorHandle), err,
        EXIT);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(&m_tensorReqs.outRangeFFTRequirements, m_allocatorHandle, &refRangeFFTTensorHandle), err,
        EXIT);

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(&m_tensorReqs.winRangeFFTRequirements, m_allocatorHandle, &winRangeFFTTensorHandle), err,
        EXIT);

    // Doppler FFT tensors
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(&m_tensorReqs.winDopplerFFTRequirements, m_allocatorHandle, &winDopplerFFTTensorHandle),
        err, EXIT);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(&m_tensorReqs.outDopplerFFTRequirements, m_allocatorHandle, &outDopplerFFTTensorHandle),
        err, EXIT);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(&m_tensorReqs.outDopplerFFTRequirements, m_allocatorHandle, &refDopplerFFTTensorHandle),
        err, EXIT);

    // NCI tensors
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(&m_tensorReqs.outNciRxTensorRequirements, m_allocatorHandle, &outNciRxTensorHandle), err,
        EXIT);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(&m_tensorReqs.outNciRxTensorRequirements, m_allocatorHandle, &refNciRxTensorHandle), err,
        EXIT);

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(&m_tensorReqs.outNciFinalTensorRequirements, m_allocatorHandle, &outNciFinalTensorHandle),
        err, EXIT);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(&m_tensorReqs.outNciFinalTensorRequirements, m_allocatorHandle, &refNciFinalTensorHandle),
        err, EXIT);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&m_tensorReqs.outNoiseEstimateTensorRequirements, m_allocatorHandle,
                                              &outNoiseEstimateTensorHandle),
                          err, EXIT);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&m_tensorReqs.outNoiseEstimateTensorRequirements, m_allocatorHandle,
                                              &refNoiseEstimateTensorHandle),
                          err, EXIT);

    // Peak Detection tensors
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(&m_tensorReqs.outPeakCountTensorRequirements, m_allocatorHandle, &refPeakCountTensorHandle),
        err, EXIT);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(&m_tensorReqs.outPeakCountTensorRequirements, m_allocatorHandle, &outPeakCountTensorHandle),
        err, EXIT);

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(&m_tensorReqs.outPeakIndexTensorRequirements, m_allocatorHandle, &outPeakIndexTensorHandle),
        err, EXIT);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(&m_tensorReqs.outPeakIndexTensorRequirements, m_allocatorHandle, &refPeakIndexTensorHandle),
        err, EXIT);

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(&m_tensorReqs.outPeakSnapTensorRequirements, m_allocatorHandle, &outPeakSnapTensorHandle),
        err, EXIT);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(&m_tensorReqs.outPeakSnapTensorRequirements, m_allocatorHandle, &refPeakSnapTensorHandle),
        err, EXIT);

    // DOA tensors
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&m_tensorReqs.inDOACalibVectorTensorRequirements, m_allocatorHandle,
                                              &inDOACalibVectorTensorHandle),
                          err, EXIT);

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(&m_tensorReqs.outDOATensorRequirements, m_allocatorHandle, &outDOATensorHandle), err, EXIT);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(&m_tensorReqs.outDOATensorRequirements, m_allocatorHandle, &refDOATensorHandle), err, EXIT);

    // Set up tensor handle aliases
    inDopplerFFTTensorHandle = outRangeFFTTensorHandle;
    inNciTensorHandle        = outDopplerFFTTensorHandle;

    // Initialize GP parameters
    populateGP(&GP, 30, 20);

    generateRangeFFTWindow(winRangeFFTTensorHandle, PVA_BATCH_FFT_WINDOW_HANNING);
    generateDopplerFFTWindow(winDopplerFFTTensorHandle, PVA_BATCH_FFT_WINDOW_HANNING);

    // Load an identity calibration vector to tensor. The bundled 4T4R capture is synthetic
    // (ideal channels), so no per-channel gain/phase correction is applied.
    int32_t calibVector[PVA_RADAR_NUM_TOTAL_ANTENNA_ELEMENTS * 2];
    populateIdentityCalibVector(calibVector, 30);
    loadCalibVectorToTensorFixedPoint(inDOACalibVectorTensorHandle, calibVector,
                                      sizeof(calibVector) / sizeof(calibVector[0]));

EXIT:
    if (err != 0)
    {
        cleanupRadarPipelineTensors();
        std::string errorMessage = "Failed to create radar pipeline tensors: " + std::to_string(err);
        throw std::runtime_error(errorMessage);
    }
}

RadarPipelineTensors::RadarPipelineTensors(RadarPipelineContext *ctx)
    : m_tensorReqs(*ctx->tensorReqs)
    , m_ctx(ctx)
{
    nvcvAllocatorConstructPva(&m_allocatorHandle);
    initRadarPipelineTensors();
    createRadarPipelineTensors();
}
