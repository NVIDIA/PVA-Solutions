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

#include "ref/radar_cfar_test_ref.h"

#include <ErrorCheckMacros.h>
#include <OpRadarCFAR.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <math.h>
#include <nvcv/Fwd.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>
#include <stdio.h>

#define kDefaultTensorWidth 32
#define kDefaultTensorHeight 512

static float calc_threshold_factor(int32_t numTrain, float probabilityFA)
{
    return (2 * numTrain) * (powf(probabilityFA, -1.0f / (2 * numTrain)) - 1.0f);
}

static void calc_horizontal_threshold_factors(float *factors, int32_t count, int32_t numHorTrain, float probabilityFA)
{
    float baseThreshold = calc_threshold_factor(numHorTrain, probabilityFA);
    for (int32_t i = 0; i < count; i++)
    {
        factors[i] = baseThreshold;
    }
}

/**
 * @brief Allocate and initialize tensors
 */
static int AllocateAndInitializeTensors(
    int32_t width, int32_t height, NVCVAllocatorHandle *allocatorHandle, NVCVTensorHandle *inTensorHandle,
    NVCVTensorHandle *outTensorHandle, NVCVTensorHandle *refTensorHandle,
    NVCVTensorHandle *detectionCountOutTensorHandle, NVCVTensorHandle *detectionCountRefTensorHandle,
    NVCVTensorRequirements *inTensorRequirements, NVCVTensorRequirements *outTensorRequirements,
    NVCVTensorRequirements *refTensorRequirements, NVCVTensorRequirements *detectionCountOutTensorRequirements,
    NVCVTensorRequirements *detectionCountRefTensorRequirements)
{
    int32_t err                = 0;
    const int32_t channelCount = 1;
    const int32_t tensorRank   = 3;
    NVCVTensorLayout tensorLayout;

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HWC", &tensorLayout), err, Error);

    int64_t inTensorShape[] = {height, width, channelCount};
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank, inTensorShape, NVCV_DATA_TYPE_S32, tensorLayout, 0,
                                                        0, inTensorRequirements),
                          err, Error);

    int64_t outTensorShape[] = {512, 2, channelCount};
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank, outTensorShape, NVCV_DATA_TYPE_S32, tensorLayout, 0,
                                                        0, outTensorRequirements),
                          err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank, outTensorShape, NVCV_DATA_TYPE_S32, tensorLayout, 0,
                                                        0, refTensorRequirements),
                          err, Error);

    int64_t detectionCountShape[] = {1};
    NVCVTensorLayout detectionCountLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("W", &detectionCountLayout), err, Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(1, detectionCountShape, NVCV_DATA_TYPE_S32, detectionCountLayout, 0, 0,
                                      detectionCountRefTensorRequirements),
        err, Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(1, detectionCountShape, NVCV_DATA_TYPE_S32, detectionCountLayout, 0, 0,
                                      detectionCountOutTensorRequirements),
        err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(inTensorRequirements, *allocatorHandle, inTensorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(outTensorRequirements, *allocatorHandle, outTensorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(refTensorRequirements, *allocatorHandle, refTensorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(detectionCountRefTensorRequirements, *allocatorHandle, detectionCountRefTensorHandle), err,
        Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(detectionCountOutTensorRequirements, *allocatorHandle, detectionCountOutTensorHandle), err,
        Error);

    NVCV_CHECK_ERROR_GOTO(generateRadarCFARInput(*inTensorHandle, false), err, Error);

    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(NVCVTensorHandle inTensorHandle, NVCVTensorHandle refTensorHandle,
                          NVCVTensorHandle outTensorHandle, NVCVTensorHandle detectionCountRefTensorHandle,
                          NVCVTensorHandle detectionCountOutTensorHandle, PvaRadarCFARParams *radarCFARParams,
                          PVARadarCFARType algo, const char *testName)
{
    int32_t err =
        radarCFARRefSubmit(inTensorHandle, refTensorHandle, *radarCFARParams, detectionCountRefTensorHandle, algo);
    if (err)
    {
        printf("%s test failed during reference computation\n", testName);
        return err;
    }

    err = checkagainstRadarCFARRef(refTensorHandle, outTensorHandle, detectionCountRefTensorHandle,
                                   detectionCountOutTensorHandle);
    if (err)
    {
        printf("%s test failed\n", testName);
    }
    return err;
}

/**
 * @brief Test radar CFAR operation using CUDA stream
 */
int SubmitRadarCFARToCudaStream(int32_t width, int32_t height, PVARadarCFARType algo,
                                PvaRadarCFARParams *radarCFARParams)
{
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cudaStream_t stream                 = NULL;
    int32_t err                         = 0;

    NVCVTensorHandle inTensorHandle                = NULL;
    NVCVTensorHandle outTensorHandle               = NULL;
    NVCVTensorHandle refTensorHandle               = NULL;
    NVCVTensorHandle detectionCountOutTensorHandle = NULL;
    NVCVTensorHandle detectionCountRefTensorHandle = NULL;

    NVCVTensorRequirements inTensorRequirements;
    NVCVTensorRequirements outTensorRequirements;
    NVCVTensorRequirements refTensorRequirements;
    NVCVTensorRequirements detectionCountOutTensorRequirements;
    NVCVTensorRequirements detectionCountRefTensorRequirements;

    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeTensors(width, height, &allocatorHandle, &inTensorHandle, &outTensorHandle,
                                       &refTensorHandle, &detectionCountOutTensorHandle, &detectionCountRefTensorHandle,
                                       &inTensorRequirements, &outTensorRequirements, &refTensorRequirements,
                                       &detectionCountOutTensorRequirements, &detectionCountRefTensorRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(
        pvaRadarCFARCreate(&operatorHandle, &inTensorRequirements, &outTensorRequirements, algo, radarCFARParams), err,
        FreeResources);

    /* Submit to CUDA stream */
    NVCV_CHECK_ERROR_GOTO(
        pvaRadarCFARSubmit(operatorHandle, stream, inTensorHandle, outTensorHandle, detectionCountOutTensorHandle), err,
        FreeResources);

    /* Synchronize stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(inTensorHandle, refTensorHandle, outTensorHandle, detectionCountRefTensorHandle,
                         detectionCountOutTensorHandle, radarCFARParams, algo, "Submit to CUDA stream");

FreeResources:
    cudaStreamDestroy(stream);
    nvcvTensorDecRef(inTensorHandle, NULL);
    nvcvTensorDecRef(outTensorHandle, NULL);
    nvcvTensorDecRef(refTensorHandle, NULL);
    nvcvTensorDecRef(detectionCountRefTensorHandle, NULL);
    nvcvTensorDecRef(detectionCountOutTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);
    return err;
}

/**
 * @brief Test radar CFAR operation using cuPVA stream
 */
int SubmitRadarCFARToCupvaStream(int32_t width, int32_t height, PVARadarCFARType algo,
                                 PvaRadarCFARParams *radarCFARParams)
{
    const int32_t channelCount = 1;
    const int32_t tensorRank   = 3;

    NVCVTensorLayout tensorLayout;
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cupvaSyncObj_t sync                 = NULL;
    cupvaStream_t stream                = NULL;
    int32_t err                         = 0;
    cupvaFence_t fence;
    cupvaCmd_t rf;
    cupvaCmd_t const *cmds[1]  = {&rf};
    cupvaCmdStatus_t status[1] = {NULL};
    bool waitSuccess;

    NVCVDataType inDataType         = NVCV_DATA_TYPE_S32;
    NVCVTensorHandle inTensorHandle = NULL;
    int64_t inTensorShape[]         = {height, width, channelCount};
    NVCVTensorRequirements inTensorRequirements;

    NVCVDataType outDataType         = NVCV_DATA_TYPE_S32;
    NVCVTensorHandle outTensorHandle = NULL;
    int64_t outTensorShape[]         = {512, 2, channelCount};
    NVCVTensorRequirements outTensorRequirements;

    NVCVDataType refDataType         = NVCV_DATA_TYPE_S32;
    const int32_t refTensorRank      = 3;
    NVCVTensorHandle refTensorHandle = NULL;
    int64_t refTensorShape[]         = {512, 2, channelCount};
    NVCVTensorRequirements refTensorRequirements;
    NVCVTensorLayout refTensorLayout;

    NVCVDataType detectionCountOutDataType         = NVCV_DATA_TYPE_S32;
    NVCVTensorHandle detectionCountOutTensorHandle = NULL;
    const int32_t detectionCountOutTensorRank      = 1;
    int64_t detectionCountOutTensorShape[]         = {1};
    NVCVTensorRequirements detectionCountOutTensorRequirements;
    NVCVTensorLayout detectionCountOutTensorLayout;

    NVCVDataType detectionCountRefDataType         = NVCV_DATA_TYPE_S32;
    NVCVTensorHandle detectionCountRefTensorHandle = NULL;
    const int32_t detectionCountRefTensorRank      = 1;
    int64_t detectionCountRefTensorShape[]         = {1};
    NVCVTensorRequirements detectionCountRefTensorRequirements;
    NVCVTensorLayout detectionCountRefTensorLayout;

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(&allocatorHandle), err, FreeResources);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HWC", &tensorLayout), err, FreeResources);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HWC", &refTensorLayout), err, FreeResources);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("W", &detectionCountRefTensorLayout), err, FreeResources);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("W", &detectionCountOutTensorLayout), err, FreeResources);

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(tensorRank, inTensorShape, inDataType, tensorLayout, 0, 0, &inTensorRequirements),
        err, FreeResources);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank, outTensorShape, outDataType, tensorLayout, 0, 0,
                                                        &outTensorRequirements),
                          err, FreeResources);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank, refTensorShape, refDataType, refTensorLayout, 0, 0,
                                                        &refTensorRequirements),
                          err, FreeResources);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(detectionCountRefTensorRank, detectionCountRefTensorShape,
                                                        detectionCountRefDataType, detectionCountRefTensorLayout, 0, 0,
                                                        &detectionCountRefTensorRequirements),
                          err, FreeResources);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(detectionCountOutTensorRank, detectionCountOutTensorShape,
                                                        detectionCountOutDataType, detectionCountOutTensorLayout, 0, 0,
                                                        &detectionCountOutTensorRequirements),
                          err, FreeResources);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&inTensorRequirements, allocatorHandle, &inTensorHandle), err,
                          FreeResources);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&outTensorRequirements, allocatorHandle, &outTensorHandle), err,
                          FreeResources);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&refTensorRequirements, allocatorHandle, &refTensorHandle), err,
                          FreeResources);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(&detectionCountRefTensorRequirements, allocatorHandle, &detectionCountRefTensorHandle), err,
        FreeResources);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(&detectionCountOutTensorRequirements, allocatorHandle, &detectionCountOutTensorHandle), err,
        FreeResources);

    NVCV_CHECK_ERROR_GOTO(generateRadarCFARInput(inTensorHandle, false), err, FreeResources);
    NVCV_CHECK_ERROR_GOTO(
        pvaRadarCFARCreate(&operatorHandle, &inTensorRequirements, &outTensorRequirements, algo, radarCFARParams), err,
        FreeResources);
    CUPVA_CHECK_ERROR_GOTO(CupvaSyncObjCreate(&sync, false, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD), err,
                           FreeResources);
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceInit(&fence, sync), err, FreeResources);
    CUPVA_CHECK_ERROR_GOTO(CupvaCmdRequestFencesInit(&rf, &fence, 1), err, FreeResources);
    CUPVA_CHECK_ERROR_GOTO(CupvaStreamCreate(&stream, CUPVA_PVA0, CUPVA_VPU_ANY), err, FreeResources);

    NVCV_CHECK_ERROR_GOTO(
        pvaRadarCFARSubmit(operatorHandle, stream, inTensorHandle, outTensorHandle, detectionCountOutTensorHandle), err,
        FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);
    if (!waitSuccess)
    {
        printf("PVA execution did not complete successfully\n");
        err = -1;
        goto FreeResources;
    }

    NVCV_CHECK_ERROR_GOTO(
        radarCFARRefSubmit(inTensorHandle, refTensorHandle, *radarCFARParams, detectionCountRefTensorHandle, algo), err,
        FreeResources);
    err = checkagainstRadarCFARRef(refTensorHandle, outTensorHandle, detectionCountRefTensorHandle,
                                   detectionCountOutTensorHandle);

    if (err)
    {
        printf("Submit to cuPVA stream test failed\n");
    }

FreeResources:
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvTensorDecRef(inTensorHandle, NULL);
    nvcvTensorDecRef(outTensorHandle, NULL);
    nvcvTensorDecRef(refTensorHandle, NULL);
    nvcvTensorDecRef(detectionCountRefTensorHandle, NULL);
    nvcvTensorDecRef(detectionCountOutTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);
    return err;
}

int main(int argc, char **argv)
{
    const PVARadarCFARType algo = PVA_CFAR_CA_GO;
    float probabilityFA         = 1e-3f;
    const int32_t width         = kDefaultTensorWidth;
    const int32_t height        = kDefaultTensorHeight;

    float horizontalThresholdFactors[kDefaultTensorHeight];
    calc_horizontal_threshold_factors(horizontalThresholdFactors, height, 6, probabilityFA);

    PvaRadarCFARParams radarCFARParams;
    radarCFARParams.numHorTrain                      = 6;
    radarCFARParams.numHorGuard                      = 3;
    radarCFARParams.numVerTrain                      = 8;
    radarCFARParams.numVerGuard                      = 4;
    radarCFARParams.horizontalThreshold.type         = PVA_CFAR_HORIZONTAL_THRESHOLD_PER_ROW;
    radarCFARParams.horizontalThreshold.value.perRow = horizontalThresholdFactors;
    radarCFARParams.verticalThresholdFactor   = calc_threshold_factor(radarCFARParams.numVerTrain, probabilityFA);
    radarCFARParams.isHorizontalCyclicPadding = false;
    radarCFARParams.isVerticalCyclicPadding   = false;
    radarCFARParams.enablePeakGrouping        = false;

    int32_t cudaStreamSubmitTestResult  = 0;
    int32_t cupvaStreamSubmitTestResult = 0;

#if CUPVA_HAS_CUDA_STREAM_SUPPORT == 1

    CUDA_CHECK_ERROR_GOTO(cudaSetDevice(0), cudaStreamSubmitTestResult, CheckResults);

    uint32_t driverVersion;
    CUPVA_CHECK_ERROR_GOTO(CupvaGetDriverVersion(&driverVersion), cudaStreamSubmitTestResult, CheckResults);
    if (driverVersion >= 2007)
    {
        cudaStreamSubmitTestResult = SubmitRadarCFARToCudaStream(width, height, algo, &radarCFARParams);
    }
#endif

    cupvaStreamSubmitTestResult = SubmitRadarCFARToCupvaStream(width, height, algo, &radarCFARParams);

CheckResults:
    if (cudaStreamSubmitTestResult || cupvaStreamSubmitTestResult)
    {
        printf("Test Fail\n");
    }
    else
    {
        printf("Test Pass\n");
    }

    return (cudaStreamSubmitTestResult || cupvaStreamSubmitTestResult);
}
