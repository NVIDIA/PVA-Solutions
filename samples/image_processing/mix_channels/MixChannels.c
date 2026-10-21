/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ErrorCheckMacros.h"
#include "ref/MixChannelsRef.h"

#include <OpMixChannels.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Fwd.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>
#include <stdlib.h>

/**
 * The maximum input/output tensor number in mix channels operator.
 */
#define MAX_TENSOR_NUM (4)

/**
 * @brief Allocate and initialize tensors
 */
static int AllocateAndInitializeTensors(int32_t imgWidth, int32_t imgHeight, int32_t tensorRank, int32_t inTensorCount,
                                        int32_t outTensorCount, int32_t inChannels, int32_t outChannels,
                                        NVCVAllocatorHandle *allocatorHandle, NVCVTensorHandle *inHandles,
                                        NVCVTensorHandle *outHandles, NVCVTensorHandle *refHandles,
                                        NVCVTensorRequirements *inTensorReqPtrs[],
                                        NVCVTensorRequirements *outTensorReqPtrs[])
{
    int32_t err = 0;
    NVCVTensorLayout tensorLayout;

    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HWC", &tensorLayout), err, Error);

    int64_t inputTensorShape[]  = {imgHeight, imgWidth, inChannels};
    int64_t outputTensorShape[] = {imgHeight, imgWidth, outChannels};

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);

    for (int i = 0; i < inTensorCount; ++i)
    {
        NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank, inputTensorShape, NVCV_DATA_TYPE_U8,
                                                            tensorLayout, 0, 0, inTensorReqPtrs[i]),
                              err, Error);

        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(inTensorReqPtrs[i], *allocatorHandle, &inHandles[i]), err, Error);
    }

    for (int i = 0; i < outTensorCount; ++i)
    {
        NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank, outputTensorShape, NVCV_DATA_TYPE_U8,
                                                            tensorLayout, 0, 0, outTensorReqPtrs[i]),
                              err, Error);

        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(outTensorReqPtrs[i], *allocatorHandle, &outHandles[i]), err, Error);
        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(outTensorReqPtrs[i], *allocatorHandle, &refHandles[i]), err, Error);
    }

    generateRandomMixChannelsInput(inHandles, inTensorCount);

    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(NVCVTensorHandle *inHandles, NVCVTensorHandle *refHandles, NVCVTensorHandle *outHandles,
                          int32_t inTensorCount, int32_t outTensorCount, PVAMixChannelsCode code, const char *testName)
{
    mixChannelsRef(inHandles, refHandles, inTensorCount, outTensorCount, code);
    int32_t err = checkMixChannelsOutput(refHandles, outHandles, outTensorCount);
    if (err)
    {
        printf("%s test failed\n", testName);
    }
    return err;
}

/**
 * @brief Test mix channels operation using CUDA stream
 */
int SubmitMixChannelsToCudaStream(int32_t imgWidth, int32_t imgHeight, int32_t tensorRank, PVAMixChannelsCode code)
{
    int32_t inTensorCount;
    int32_t outTensorCount;
    int32_t inChannels;
    int32_t outChannels;
    NVCVDataType inDataType;
    NVCVDataType outDataType;

    switch (code)
    {
    case PVA_SPLIT_RGBA8_TO_U8:
        inTensorCount  = 1;
        outTensorCount = 4;
        inChannels     = 4;
        outChannels    = 1;
        inDataType     = NVCV_DATA_TYPE_U8;
        outDataType    = NVCV_DATA_TYPE_U8;
        break;

    case PVA_MERGE_U8_TO_RGBA8:
        inTensorCount  = 4;
        outTensorCount = 1;
        inChannels     = 1;
        outChannels    = 4;
        inDataType     = NVCV_DATA_TYPE_U8;
        outDataType    = NVCV_DATA_TYPE_U8;
        break;
    }

    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cudaStream_t stream                 = NULL;
    int32_t err                         = 0;

    NVCVTensorHandle inHandles[MAX_TENSOR_NUM];
    NVCVTensorHandle outHandles[MAX_TENSOR_NUM];
    NVCVTensorHandle refHandles[MAX_TENSOR_NUM];

    NVCVTensorRequirements inTensorReqs[MAX_TENSOR_NUM];
    NVCVTensorRequirements outTensorReqs[MAX_TENSOR_NUM];
    NVCVTensorRequirements *inTensorReqPtrs[MAX_TENSOR_NUM];
    NVCVTensorRequirements *outTensorReqPtrs[MAX_TENSOR_NUM];

    for (int i = 0; i < MAX_TENSOR_NUM; ++i)
    {
        inTensorReqPtrs[i]  = &inTensorReqs[i];
        outTensorReqPtrs[i] = &outTensorReqs[i];
    }

    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeTensors(imgWidth, imgHeight, tensorRank, inTensorCount, outTensorCount, inChannels,
                                       outChannels, &allocatorHandle, inHandles, outHandles, refHandles,
                                       inTensorReqPtrs, outTensorReqPtrs);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(
        pvaMixChannelsCreate(&operatorHandle, inTensorReqPtrs, outTensorReqPtrs, inTensorCount, outTensorCount, code),
        err, FreeResources);

    /* Submit to CUDA stream */
    NVCV_CHECK_ERROR_GOTO(
        pvaMixChannelsSubmit(operatorHandle, stream, inHandles, outHandles, inTensorCount, outTensorCount), err,
        FreeResources);

    /* Synchronize stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    /* Validate output */
    err =
        ValidateOutput(inHandles, refHandles, outHandles, inTensorCount, outTensorCount, code, "Submit to CUDA stream");

FreeResources: /* clean up all allocated resources */
    cudaStreamDestroy(stream);

    for (int i = 0; i < inTensorCount; ++i)
    {
        nvcvTensorDecRef(inHandles[i], NULL);
    }

    for (int i = 0; i < outTensorCount; ++i)
    {
        nvcvTensorDecRef(outHandles[i], NULL);
        nvcvTensorDecRef(refHandles[i], NULL);
    }

    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

/**
 * @brief Test mix channels operation using cuPVA stream
 */
int SubmitMixChannelsToCupvaStream(int32_t imgWidth, int32_t imgHeight, int32_t tensorRank, PVAMixChannelsCode code)
{
    int32_t inTensorCount;
    int32_t outTensorCount;
    int32_t inChannels;
    int32_t outChannels;
    NVCVDataType inDataType;
    NVCVDataType outDataType;

    switch (code)
    {
    case PVA_SPLIT_RGBA8_TO_U8:
        inTensorCount  = 1;
        outTensorCount = 4;
        inChannels     = 4;
        outChannels    = 1;
        inDataType     = NVCV_DATA_TYPE_U8;
        outDataType    = NVCV_DATA_TYPE_U8;
        break;

    case PVA_MERGE_U8_TO_RGBA8:
        inTensorCount  = 4;
        outTensorCount = 1;
        inChannels     = 1;
        outChannels    = 4;
        inDataType     = NVCV_DATA_TYPE_U8;
        outDataType    = NVCV_DATA_TYPE_U8;
        break;
    }

    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cupvaSyncObj_t sync                 = NULL;
    cupvaStream_t stream                = NULL;
    int32_t err                         = 0;

    NVCVTensorHandle inHandles[MAX_TENSOR_NUM];
    NVCVTensorHandle outHandles[MAX_TENSOR_NUM];
    NVCVTensorHandle refHandles[MAX_TENSOR_NUM];

    NVCVTensorRequirements inTensorReqs[MAX_TENSOR_NUM];
    NVCVTensorRequirements outTensorReqs[MAX_TENSOR_NUM];
    NVCVTensorRequirements *inTensorReqPtrs[MAX_TENSOR_NUM];
    NVCVTensorRequirements *outTensorReqPtrs[MAX_TENSOR_NUM];

    for (int i = 0; i < MAX_TENSOR_NUM; ++i)
    {
        inTensorReqPtrs[i]  = &inTensorReqs[i];
        outTensorReqPtrs[i] = &outTensorReqs[i];
    }

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeTensors(imgWidth, imgHeight, tensorRank, inTensorCount, outTensorCount, inChannels,
                                       outChannels, &allocatorHandle, inHandles, outHandles, refHandles,
                                       inTensorReqPtrs, outTensorReqPtrs);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(
        pvaMixChannelsCreate(&operatorHandle, inTensorReqPtrs, outTensorReqPtrs, inTensorCount, outTensorCount, code),
        err, FreeResources);

    /* Create cuPVA stream and synchronization objects */
    CUPVA_CHECK_ERROR_GOTO(CupvaSyncObjCreate(&sync, false, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD), err,
                           FreeResources);

    cupvaFence_t fence;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceInit(&fence, sync), err, FreeResources);

    cupvaCmd_t rf;
    CUPVA_CHECK_ERROR_GOTO(CupvaCmdRequestFencesInit(&rf, &fence, 1), err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamCreate(&stream, CUPVA_PVA0, CUPVA_VPU_ANY), err, FreeResources);

    /* Submit to cuPVA stream */
    cupvaCmd_t const *cmds[1]  = {&rf};
    cupvaCmdStatus_t status[1] = {NULL};

    NVCV_CHECK_ERROR_GOTO(
        pvaMixChannelsSubmit(operatorHandle, stream, inHandles, outHandles, inTensorCount, outTensorCount), err,
        FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    /* Wait for completion */
    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(inHandles, refHandles, outHandles, inTensorCount, outTensorCount, code,
                         "Submit to cuPVA stream");

FreeResources: /* clean up all allocated resources */
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);

    for (int i = 0; i < inTensorCount; ++i)
    {
        nvcvTensorDecRef(inHandles[i], NULL);
    }

    for (int i = 0; i < outTensorCount; ++i)
    {
        nvcvTensorDecRef(outHandles[i], NULL);
        nvcvTensorDecRef(refHandles[i], NULL);
    }

    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char **argv)
{
    /* Parameter settings */
    const int32_t imgWidth   = 1920;
    const int32_t imgHeight  = 1080;
    const int32_t tensorRank = 3;
    PVAMixChannelsCode code  = PVA_SPLIT_RGBA8_TO_U8;

    int32_t cudaStreamSubmitTestResult  = 0;
    int32_t cupvaStreamSubmitTestResult = 0;

#if CUPVA_HAS_CUDA_STREAM_SUPPORT == 1

    /* Initialize CUDA context*/
    CUDA_CHECK_ERROR_GOTO(cudaSetDevice(0), cudaStreamSubmitTestResult, CheckResults);

    /* Submit to CUDA stream if driver version is greater than or equal to 2007 */
    uint32_t driverVersion;
    CUPVA_CHECK_ERROR_GOTO(CupvaGetDriverVersion(&driverVersion), cudaStreamSubmitTestResult, CheckResults);
    if (driverVersion >= 2007)
    {
        /* Submit to CUDA stream */
        cudaStreamSubmitTestResult = SubmitMixChannelsToCudaStream(imgWidth, imgHeight, tensorRank, code);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult = SubmitMixChannelsToCupvaStream(imgWidth, imgHeight, tensorRank, code);

CheckResults:
    /* Overall result */
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
