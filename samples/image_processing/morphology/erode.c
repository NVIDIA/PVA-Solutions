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
#include "ref/morphology_ref.h"

#include <OpMorphology.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Fwd.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>

/**
 * @brief Allocate and initialize tensors
 */
static int AllocateAndInitializeTensors(int32_t imgWidth, int32_t imgHeight, int32_t channelCount, int32_t tensorRank,
                                        NVCVAllocatorHandle *allocatorHandle, NVCVTensorHandle *inTensorHandle,
                                        NVCVTensorHandle *outTensorHandle, NVCVTensorHandle *refTensorHandle,
                                        NVCVTensorRequirements *tensorRequirements)
{
    int32_t err = 0;

    NVCVTensorLayout tensorLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HWC", &tensorLayout), err, Error);

    int64_t tensorShape[] = {imgHeight, imgWidth, channelCount};
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank, tensorShape, NVCV_DATA_TYPE_U8, tensorLayout, 0, 0,
                                                        tensorRequirements),
                          err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensorRequirements, *allocatorHandle, inTensorHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensorRequirements, *allocatorHandle, outTensorHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensorRequirements, *allocatorHandle, refTensorHandle), err, Error);

    generateRandomMorphologyInput(*inTensorHandle);

    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(NVCVTensorHandle inTensorHandle, NVCVTensorHandle refTensorHandle,
                          NVCVTensorHandle outTensorHandle, int32_t knlWidth, int32_t knlHeight, int32_t borderValue,
                          const char *testName)
{
    PvaMorphologyMaskParams maskParams;
    maskParams.maskWidth  = knlWidth;
    maskParams.maskHeight = knlHeight;
    maskParams.maskShape  = RECTANGLE_MASK;

    morphologyRef(inTensorHandle, refTensorHandle, PVA_ERODE, &maskParams, NVCV_BORDER_CONSTANT, borderValue);

    int32_t err = checkMorphologyOutput(refTensorHandle, outTensorHandle);
    if (err)
    {
        printf("%s test failed\n", testName);
    }
    return err;
}

/**
 * @brief Test morphology erode operation using CUDA stream
 */
int SubmitErodeToCudaStream(int32_t imgWidth, int32_t imgHeight, int32_t channelCount, int32_t tensorRank,
                            int32_t knlWidth, int32_t knlHeight, int32_t borderValue)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVTensorHandle inTensorHandle     = NULL;
    NVCVTensorHandle outTensorHandle    = NULL;
    NVCVTensorHandle refTensorHandle    = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cudaStream_t stream                 = NULL;
    int32_t err                         = 0;

    NVCVTensorRequirements tensorRequirements;

    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeTensors(imgWidth, imgHeight, channelCount, tensorRank, &allocatorHandle, &inTensorHandle,
                                       &outTensorHandle, &refTensorHandle, &tensorRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    PvaMorphologyMaskParams maskParams;
    maskParams.maskWidth  = knlWidth;
    maskParams.maskHeight = knlHeight;
    maskParams.maskShape  = RECTANGLE_MASK;

    NVCV_CHECK_ERROR_GOTO(pvaMorphologyCreate(&operatorHandle, &tensorRequirements, PVA_ERODE, &maskParams,
                                              NVCV_BORDER_CONSTANT, borderValue),
                          err, FreeResources);

    /* Submit to CUDA stream */
    NVCV_CHECK_ERROR_GOTO(pvaMorphologySubmit(operatorHandle, stream, inTensorHandle, outTensorHandle), err,
                          FreeResources);

    /* Synchronize stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(inTensorHandle, refTensorHandle, outTensorHandle, knlWidth, knlHeight, borderValue,
                         "Submit to CUDA stream");

FreeResources: /* clean up all allocated resources */
    cudaStreamDestroy(stream);
    nvcvTensorDecRef(inTensorHandle, NULL);
    nvcvTensorDecRef(outTensorHandle, NULL);
    nvcvTensorDecRef(refTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

/**
 * @brief Test morphology erode operation using cuPVA stream
 */
int SubmitErodeToCupvaStream(int32_t imgWidth, int32_t imgHeight, int32_t channelCount, int32_t tensorRank,
                             int32_t knlWidth, int32_t knlHeight, int32_t borderValue)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVTensorHandle inTensorHandle     = NULL;
    NVCVTensorHandle outTensorHandle    = NULL;
    NVCVTensorHandle refTensorHandle    = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cupvaSyncObj_t sync                 = NULL;
    cupvaStream_t stream                = NULL;
    int32_t err                         = 0;

    NVCVTensorRequirements tensorRequirements;

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeTensors(imgWidth, imgHeight, channelCount, tensorRank, &allocatorHandle, &inTensorHandle,
                                       &outTensorHandle, &refTensorHandle, &tensorRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    PvaMorphologyMaskParams maskParams;
    maskParams.maskWidth  = knlWidth;
    maskParams.maskHeight = knlHeight;
    maskParams.maskShape  = RECTANGLE_MASK;

    NVCV_CHECK_ERROR_GOTO(pvaMorphologyCreate(&operatorHandle, &tensorRequirements, PVA_ERODE, &maskParams,
                                              NVCV_BORDER_CONSTANT, borderValue),
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

    NVCV_CHECK_ERROR_GOTO(pvaMorphologySubmit(operatorHandle, stream, inTensorHandle, outTensorHandle), err,
                          FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    /* Wait for completion */
    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(inTensorHandle, refTensorHandle, outTensorHandle, knlWidth, knlHeight, borderValue,
                         "Submit to cuPVA stream");

FreeResources: /* clean up all allocated resources */
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvTensorDecRef(inTensorHandle, NULL);
    nvcvTensorDecRef(outTensorHandle, NULL);
    nvcvTensorDecRef(refTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char **argv)
{
    /* Parameter settings */
    const int32_t imgWidth     = 1920;
    const int32_t imgHeight    = 1080;
    const int32_t channelCount = 1;
    const int32_t tensorRank   = 3;

    const int32_t knlWidth  = 5;
    const int32_t knlHeight = 5;

    const int32_t borderValue = 0;

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
        cudaStreamSubmitTestResult =
            SubmitErodeToCudaStream(imgWidth, imgHeight, channelCount, tensorRank, knlWidth, knlHeight, borderValue);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult =
        SubmitErodeToCupvaStream(imgWidth, imgHeight, channelCount, tensorRank, knlWidth, knlHeight, borderValue);

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
