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

#include "ref/canny_edge_detector_ref.h"

#include <ErrorCheckMacros.h>
#include <ImageIO.h>
#include <OpCannyEdgeDetector.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Fwd.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>
#include <stdio.h>

/**
 * @brief Allocate and initialize tensors
 */
static int AllocateAndInitializeTensors(int argc, char **argv, int32_t imgWidth, int32_t imgHeight, int32_t numChannels,
                                        int32_t imgLinePitch, int32_t rowAlign, NVCVAllocatorHandle *allocatorHandle,
                                        NVCVTensorHandle *inTensorHandle, NVCVTensorHandle *outTensorHandle,
                                        NVCVTensorHandle *refTensorHandle, NVCVTensorRequirements *imageRequirements,
                                        char *assetsDirectory, int32_t maxPathLength)
{
    int32_t err                = 0;
    int64_t imageTensorShape[] = {numChannels, imgHeight, imgWidth};
    NVCVTensorLayout imageTensorLayout;

    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("CHW", &imageTensorLayout), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(3, imageTensorShape, NVCV_DATA_TYPE_U8, imageTensorLayout, 0,
                                                        rowAlign, imageRequirements),
                          err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(imageRequirements, *allocatorHandle, inTensorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(imageRequirements, *allocatorHandle, outTensorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(imageRequirements, *allocatorHandle, refTensorHandle), err, Error);

    err = GetAssetsDirectory(argc, argv, assetsDirectory, maxPathLength);
    err |= loadCannyImageInput(*inTensorHandle, assetsDirectory, "/canny_edge_detector/kodim08.bin",
                               imgLinePitch * imgHeight);
    if (err)
    {
        printf("Canny edge detector input image could not be loaded\n");
        return err;
    }

    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(NVCVTensorHandle inTensorHandle, NVCVTensorHandle refTensorHandle,
                          NVCVTensorHandle outTensorHandle, int32_t gradientSize, int32_t thresholdStrong,
                          int32_t thresholdWeak, const char *testName)
{
    CannyEdgeDetectorFixedPointRef(inTensorHandle, refTensorHandle, gradientSize, thresholdStrong, thresholdWeak);
    int32_t err = checkCannyEdgeDetectorOutput(refTensorHandle, outTensorHandle);
    if (err)
    {
        printf("%s test failed\n", testName);
    }
    return err;
}

/**
 * @brief Test canny edge detector operation using CUDA stream
 */
int SubmitCannyEdgeDetectorToCudaStream(int argc, char **argv, int32_t imgWidth, int32_t imgHeight, int32_t numChannels,
                                        int32_t imgLinePitch, int32_t rowAlign, int32_t gradientSize,
                                        int32_t thresholdStrong, int32_t thresholdWeak)
{
    const int32_t maxPathLength = 200;
    char assetsDirectory[maxPathLength];

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

    cupvaCmdStatus_t status[2] = {NULL, NULL};
    NVCVTensorRequirements imageRequirements;

    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeTensors(argc, argv, imgWidth, imgHeight, numChannels, imgLinePitch, rowAlign,
                                       &allocatorHandle, &inTensorHandle, &outTensorHandle, &refTensorHandle,
                                       &imageRequirements, assetsDirectory, maxPathLength);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaCannyEdgeDetectorCreate(&operatorHandle, &imageRequirements, gradientSize), err,
                          FreeResources);

    /* Submit to CUDA stream */
    NVCV_CHECK_ERROR_GOTO(pvaCannyEdgeDetectorSubmit(operatorHandle, stream, status, inTensorHandle, outTensorHandle,
                                                     thresholdStrong, thresholdWeak),
                          err, FreeResources);

    /* Synchronize stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(inTensorHandle, refTensorHandle, outTensorHandle, gradientSize, thresholdStrong, thresholdWeak,
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
 * @brief Test canny edge detector operation using cuPVA stream
 */
int SubmitCannyEdgeDetectorToCupvaStream(int argc, char **argv, int32_t imgWidth, int32_t imgHeight,
                                         int32_t numChannels, int32_t imgLinePitch, int32_t rowAlign,
                                         int32_t gradientSize, int32_t thresholdStrong, int32_t thresholdWeak)
{
    const int32_t maxPathLength = 200;
    char assetsDirectory[maxPathLength];

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

    cupvaCmdStatus_t status[2] = {NULL, NULL};
    cupvaError_t statusCode    = CUPVA_ERROR_NONE;

    int64_t imageTensorShape[] = {numChannels, imgHeight, imgWidth};

    NVCVTensorLayout imageTensorLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("CHW", &imageTensorLayout), err, FreeResources);

    NVCVTensorRequirements imageRequirements;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(3, imageTensorShape, NVCV_DATA_TYPE_U8, imageTensorLayout, 0,
                                                        rowAlign, &imageRequirements),
                          err, FreeResources);

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(&allocatorHandle), err, FreeResources);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&imageRequirements, allocatorHandle, &inTensorHandle), err,
                          FreeResources);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&imageRequirements, allocatorHandle, &outTensorHandle), err,
                          FreeResources);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&imageRequirements, allocatorHandle, &refTensorHandle), err,
                          FreeResources);

    err = GetAssetsDirectory(argc, argv, assetsDirectory, maxPathLength);

    err |= loadCannyImageInput(inTensorHandle, assetsDirectory, "/canny_edge_detector/kodim08.bin",
                               imgLinePitch * imgHeight);
    if (err)
    {
        printf("Submit to cuPVA stream test failed: Canny edge detector input image could not be loaded\n");
        goto FreeResources;
    }

    NVCV_CHECK_ERROR_GOTO(pvaCannyEdgeDetectorCreate(&operatorHandle, &imageRequirements, gradientSize), err,
                          FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaSyncObjCreate(&sync, false, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD), err,
                           FreeResources);

    cupvaFence_t fence;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceInit(&fence, sync), err, FreeResources);

    cupvaCmd_t rf;
    CUPVA_CHECK_ERROR_GOTO(CupvaCmdRequestFencesInit(&rf, &fence, 1), err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamCreate(&stream, CUPVA_PVA0, CUPVA_VPU_ANY), err, FreeResources);

    NVCV_CHECK_ERROR_GOTO(pvaCannyEdgeDetectorSubmit(operatorHandle, stream, status, inTensorHandle, outTensorHandle,
                                                     thresholdStrong, thresholdWeak),
                          err, FreeResources);

    cupvaCmd_t const *cmds[1] = {&rf};
    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, &status[1], 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    CupvaCheckCommandStatus(status[0], &statusCode);
    if (statusCode != CUPVA_ERROR_NONE)
    {
        printf("Submit to cuPVA stream test failed: Canny edge detector operator returned error code %d\n", statusCode);
        err = -1;
        goto FreeResources;
    }

    CannyEdgeDetectorFixedPointRef(inTensorHandle, refTensorHandle, gradientSize, thresholdStrong, thresholdWeak);
    err = checkCannyEdgeDetectorOutput(refTensorHandle, outTensorHandle);

    if (err)
    {
        printf("Submit to cuPVA stream test failed\n");
    }

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
    const int32_t imgWidth     = 768;
    const int32_t imgHeight    = 512;
    const int32_t numChannels  = 1;
    const int32_t imgLinePitch = 768;
    const int32_t rowAlign     = 0;

    const int32_t gradientSize    = 3;
    const int32_t thresholdStrong = 300;
    const int32_t thresholdWeak   = 100;

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
            SubmitCannyEdgeDetectorToCudaStream(argc, argv, imgWidth, imgHeight, numChannels, imgLinePitch, rowAlign,
                                                gradientSize, thresholdStrong, thresholdWeak);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult =
        SubmitCannyEdgeDetectorToCupvaStream(argc, argv, imgWidth, imgHeight, numChannels, imgLinePitch, rowAlign,
                                             gradientSize, thresholdStrong, thresholdWeak);

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
