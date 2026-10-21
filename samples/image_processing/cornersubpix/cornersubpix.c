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

#include "ref/cornersubpix_ref.h"

#include <ErrorCheckMacros.h>
#include <ImageIO.h>
#include <OpCornerSubPix.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Fwd.h>
#include <nvcv/Size.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>
#include <stdio.h>

/**
 * @brief Allocate and initialize tensors
 */
static int AllocateAndInitializeTensors(int argc, char **argv, int32_t imgWidth, int32_t imgHeight, int32_t numChannels,
                                        int32_t numCorners, NVCVAllocatorHandle *allocatorHandle,
                                        NVCVTensorHandle *imageTensorHandle, NVCVTensorHandle *inCornerTensorHandle,
                                        NVCVTensorHandle *outCornerTensorHandle, NVCVTensorHandle *refTensorHandle,
                                        NVCVTensorRequirements *imageRequirements,
                                        NVCVTensorRequirements *cornerRequirements, char *assetsDirectory,
                                        int32_t maxPathLength)
{
    int32_t err                 = 0;
    int64_t imageTensorShape[]  = {numChannels, imgHeight, imgWidth};
    int64_t cornerTensorShape[] = {numCorners, 2};

    NVCVTensorLayout imageTensorLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("CHW", &imageTensorLayout), err, Error);
    NVCVTensorLayout cornerTensorLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HW", &cornerTensorLayout), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(3, imageTensorShape, NVCV_DATA_TYPE_U8, imageTensorLayout, 0, 0,
                                                        imageRequirements),
                          err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(2, cornerTensorShape, NVCV_DATA_TYPE_F32, cornerTensorLayout, 0,
                                                        0, cornerRequirements),
                          err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(imageRequirements, *allocatorHandle, imageTensorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(cornerRequirements, *allocatorHandle, inCornerTensorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(cornerRequirements, *allocatorHandle, outCornerTensorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(cornerRequirements, *allocatorHandle, refTensorHandle), err, Error);

    err = GetAssetsDirectory(argc, argv, assetsDirectory, maxPathLength);
    err |= loadCornerSubPixImageInput(*imageTensorHandle, assetsDirectory, "/cornersubpix/kodim08.bin",
                                      imgWidth * imgHeight);
    err |= loadCornerSubPixCornerInput(*inCornerTensorHandle, assetsDirectory,
                                       "/cornersubpix/input_corners_kodim08.bin", numCorners);
    err |= loadCornerSubPixCornerInput(*refTensorHandle, assetsDirectory, "/cornersubpix/input_corners_kodim08.bin",
                                       numCorners);
    if (err)
    {
        printf("Input files could not be loaded\n");
        return err;
    }

    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(NVCVTensorHandle refTensorHandle, NVCVTensorHandle outCornerTensorHandle,
                          NVCVTensorHandle imageTensorHandle, NVCVSize2D src, NVCVSize2D winSize, NVCVSize2D zeroZone,
                          int32_t maxIters, float eps, int32_t numCorners, const char *testName)
{
    cornerSubPixFixedPointRef(refTensorHandle, imageTensorHandle, src, winSize, zeroZone, maxIters, eps, numCorners);
    int32_t err = checkCornerSubPixOutput(refTensorHandle, outCornerTensorHandle, numCorners);
    if (err)
    {
        printf("%s test failed\n", testName);
    }
    return err;
}

/**
 * @brief Test corner subpix operation using CUDA stream
 */
int SubmitCornerSubPixToCudaStream(int argc, char **argv, int32_t imgWidth, int32_t imgHeight, int32_t numChannels,
                                   int32_t numCorners, NVCVSize2D winSize, NVCVSize2D zeroZone, NVCVSize2D src,
                                   int32_t maxIters, float eps)
{
    const int32_t maxPathLength = 200;
    char assetsDirectory[maxPathLength];

    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle    = NULL;
    NVCVTensorHandle inCornerTensorHandle  = NULL;
    NVCVTensorHandle outCornerTensorHandle = NULL;
    NVCVTensorHandle imageTensorHandle     = NULL;
    NVCVTensorHandle refTensorHandle       = NULL;
    NVCVOperatorHandle operatorHandle      = NULL;
    cudaStream_t stream                    = NULL;
    int32_t err                            = 0;

    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    NVCVTensorRequirements imageRequirements;
    NVCVTensorRequirements cornerRequirements;

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeTensors(argc, argv, imgWidth, imgHeight, numChannels, numCorners, &allocatorHandle,
                                       &imageTensorHandle, &inCornerTensorHandle, &outCornerTensorHandle,
                                       &refTensorHandle, &imageRequirements, &cornerRequirements, assetsDirectory,
                                       maxPathLength);
    if (err)
        goto FreeResources;

    NVCV_CHECK_ERROR_GOTO(
        pvaCornerSubPixCreate(&operatorHandle, &cornerRequirements, &imageRequirements, winSize, zeroZone), err,
        FreeResources);

    NVCV_CHECK_ERROR_GOTO(pvaCornerSubPixSubmit(operatorHandle, stream, inCornerTensorHandle, outCornerTensorHandle,
                                                imageTensorHandle, maxIters, eps, numCorners),
                          err, FreeResources);

    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(refTensorHandle, outCornerTensorHandle, imageTensorHandle, src, winSize, zeroZone, maxIters,
                         eps, numCorners, "Submit to CUDA stream");

FreeResources: /* clean up all allocated resources */
    cudaStreamDestroy(stream);
    nvcvTensorDecRef(imageTensorHandle, NULL);
    nvcvTensorDecRef(inCornerTensorHandle, NULL);
    nvcvTensorDecRef(outCornerTensorHandle, NULL);
    nvcvTensorDecRef(refTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

/**
 * @brief Test corner subpix operation using cuPVA stream
 */
int SubmitCornerSubPixToCupvaStream(int argc, char **argv, int32_t imgWidth, int32_t imgHeight, int32_t numChannels,
                                    int32_t numCorners, NVCVSize2D winSize, NVCVSize2D zeroZone, NVCVSize2D src,
                                    int32_t maxIters, float eps)
{
    const int32_t maxPathLength = 200;
    char assetsDirectory[maxPathLength];

    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle    = NULL;
    NVCVTensorHandle inCornerTensorHandle  = NULL;
    NVCVTensorHandle outCornerTensorHandle = NULL;
    NVCVTensorHandle imageTensorHandle     = NULL;
    NVCVTensorHandle refTensorHandle       = NULL;
    NVCVOperatorHandle operatorHandle      = NULL;
    cupvaSyncObj_t sync                    = NULL;
    cupvaStream_t stream                   = NULL;
    int32_t err                            = 0;

    cupvaCmdStatus_t status[1] = {NULL};

    NVCVTensorRequirements imageRequirements;
    NVCVTensorRequirements cornerRequirements;

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeTensors(argc, argv, imgWidth, imgHeight, numChannels, numCorners, &allocatorHandle,
                                       &imageTensorHandle, &inCornerTensorHandle, &outCornerTensorHandle,
                                       &refTensorHandle, &imageRequirements, &cornerRequirements, assetsDirectory,
                                       maxPathLength);
    if (err)
        goto FreeResources;

    NVCV_CHECK_ERROR_GOTO(
        pvaCornerSubPixCreate(&operatorHandle, &cornerRequirements, &imageRequirements, winSize, zeroZone), err,
        FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaSyncObjCreate(&sync, false, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD), err,
                           FreeResources);

    cupvaFence_t fence;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceInit(&fence, sync), err, FreeResources);

    cupvaCmd_t rf;
    CUPVA_CHECK_ERROR_GOTO(CupvaCmdRequestFencesInit(&rf, &fence, 1), err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamCreate(&stream, CUPVA_PVA0, CUPVA_VPU_ANY), err, FreeResources);

    NVCV_CHECK_ERROR_GOTO(pvaCornerSubPixSubmit(operatorHandle, stream, inCornerTensorHandle, outCornerTensorHandle,
                                                imageTensorHandle, maxIters, eps, numCorners),
                          err, FreeResources);

    cupvaCmd_t const *cmds[1] = {&rf};
    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(refTensorHandle, outCornerTensorHandle, imageTensorHandle, src, winSize, zeroZone, maxIters,
                         eps, numCorners, "Submit to cuPVA stream");

FreeResources: /* clean up all allocated resources */
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvTensorDecRef(imageTensorHandle, NULL);
    nvcvTensorDecRef(inCornerTensorHandle, NULL);
    nvcvTensorDecRef(outCornerTensorHandle, NULL);
    nvcvTensorDecRef(refTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char **argv)
{
    /* Parameter settings */
    const int32_t imgWidth    = 768;
    const int32_t imgHeight   = 512;
    const int32_t numChannels = 1;
    const int32_t numCorners  = 1480;
    const NVCVSize2D winSize  = {3, 3};
    const NVCVSize2D zeroZone = {-1, -1};
    const NVCVSize2D src      = {imgWidth, imgHeight};
    const int32_t maxIters    = 30;
    const float eps           = 0.01f;

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
        cudaStreamSubmitTestResult = SubmitCornerSubPixToCudaStream(argc, argv, imgWidth, imgHeight, numChannels,
                                                                    numCorners, winSize, zeroZone, src, maxIters, eps);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult = SubmitCornerSubPixToCupvaStream(argc, argv, imgWidth, imgHeight, numChannels,
                                                                  numCorners, winSize, zeroZone, src, maxIters, eps);

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
