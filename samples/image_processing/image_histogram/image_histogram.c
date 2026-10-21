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
#include "ref/image_histogram_ref.h"

#include <OpImageHistogram.h>
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
                                        int32_t numBins, NVCVAllocatorHandle *allocatorHandle,
                                        NVCVTensorHandle *inTensorHandle, NVCVTensorHandle *outTensorHandle,
                                        NVCVTensorHandle *refTensorHandle, NVCVTensorRequirements *inTensorRequirements,
                                        NVCVTensorRequirements *outTensorRequirements)
{
    int32_t err = 0;
    NVCVTensorLayout tensorLayout;

    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HWC", &tensorLayout), err, Error);

    int64_t inTensorShape[] = {imgHeight, imgWidth, channelCount};
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank, inTensorShape, NVCV_DATA_TYPE_U8, tensorLayout, 0,
                                                        0, inTensorRequirements),
                          err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(inTensorRequirements, *allocatorHandle, inTensorHandle), err, Error);

    int64_t outTensorShape[] = {1, numBins, channelCount};
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank, outTensorShape, NVCV_DATA_TYPE_U32, tensorLayout, 0,
                                                        0, outTensorRequirements),
                          err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(outTensorRequirements, *allocatorHandle, outTensorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(outTensorRequirements, *allocatorHandle, refTensorHandle), err, Error);

    generateImageHistogramInput(*inTensorHandle);

    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(NVCVTensorHandle inTensorHandle, NVCVTensorHandle refTensorHandle,
                          NVCVTensorHandle outTensorHandle, PvaImageHistogramParams *imageHistogramParams,
                          const char *testName)
{
    imageHistogramRef(inTensorHandle, refTensorHandle, imageHistogramParams);
    int32_t err = checkImageHistogramOutput(refTensorHandle, outTensorHandle);
    if (err)
    {
        printf("%s test failed\n", testName);
    }
    return err;
}

/**
 * @brief Test image histogram operation using CUDA stream
 */
int SubmitImageHistogramToCudaStream(int32_t imgWidth, int32_t imgHeight, int32_t channelCount, int32_t tensorRank,
                                     float start, float end, int32_t numBins)
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

    NVCVTensorRequirements inTensorRequirements;
    NVCVTensorRequirements outTensorRequirements;

    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeTensors(imgWidth, imgHeight, channelCount, tensorRank, numBins, &allocatorHandle,
                                       &inTensorHandle, &outTensorHandle, &refTensorHandle, &inTensorRequirements,
                                       &outTensorRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    PvaImageHistogramParams imageHistogramParams;
    imageHistogramParams.start   = start;
    imageHistogramParams.end     = end;
    imageHistogramParams.numBins = numBins;

    NVCV_CHECK_ERROR_GOTO(
        pvaImageHistogramCreate(&operatorHandle, &inTensorRequirements, &outTensorRequirements, &imageHistogramParams),
        err, FreeResources);

    /* Submit to CUDA stream */
    NVCV_CHECK_ERROR_GOTO(pvaImageHistogramSubmit(operatorHandle, stream, inTensorHandle, outTensorHandle), err,
                          FreeResources);

    /* Synchronize stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(inTensorHandle, refTensorHandle, outTensorHandle, &imageHistogramParams,
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
 * @brief Test image histogram operation using cuPVA stream
 */
int SubmitImageHistogramToCupvaStream(int32_t imgWidth, int32_t imgHeight, int32_t channelCount, int32_t tensorRank,
                                      float start, float end, int32_t numBins)
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

    NVCVTensorRequirements inTensorRequirements;
    NVCVTensorRequirements outTensorRequirements;

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeTensors(imgWidth, imgHeight, channelCount, tensorRank, numBins, &allocatorHandle,
                                       &inTensorHandle, &outTensorHandle, &refTensorHandle, &inTensorRequirements,
                                       &outTensorRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    PvaImageHistogramParams imageHistogramParams;
    imageHistogramParams.start   = start;
    imageHistogramParams.end     = end;
    imageHistogramParams.numBins = numBins;

    NVCV_CHECK_ERROR_GOTO(
        pvaImageHistogramCreate(&operatorHandle, &inTensorRequirements, &outTensorRequirements, &imageHistogramParams),
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

    NVCV_CHECK_ERROR_GOTO(pvaImageHistogramSubmit(operatorHandle, stream, inTensorHandle, outTensorHandle), err,
                          FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    /* Wait for completion */
    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(inTensorHandle, refTensorHandle, outTensorHandle, &imageHistogramParams,
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

    const float start     = 0;
    const float end       = 255;
    const int32_t numBins = 255;

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
            SubmitImageHistogramToCudaStream(imgWidth, imgHeight, channelCount, tensorRank, start, end, numBins);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult =
        SubmitImageHistogramToCupvaStream(imgWidth, imgHeight, channelCount, tensorRank, start, end, numBins);

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
