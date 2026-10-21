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

#include "ErrorCheckMacros.h"
#include "ref/warp_perspective_ref.h"

#include <OpWarpPerspective.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Fwd.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>

/**
 * @brief Allocate and initialize images
 */
static int AllocateAndInitializeImages(int32_t inImgWidth, int32_t inImgHeight, int32_t outImgWidth,
                                       int32_t outImgHeight, NVCVAllocatorHandle *allocatorHandle,
                                       NVCVImageHandle *inImageHandle, NVCVImageHandle *outImageHandle,
                                       NVCVImageHandle *refOutImageHandle,
                                       NVCVImageRequirements *imageInputRequirements,
                                       NVCVImageRequirements *imageOutputRequirements)
{
    int32_t err = 0;

    NVCV_CHECK_ERROR_GOTO(
        nvcvImageCalcRequirementsPva(inImgWidth, inImgHeight, NVCV_IMAGE_FORMAT_NV12_ER, 0, 0, imageInputRequirements),
        err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvImageCalcRequirementsPva(outImgWidth, outImgHeight, NVCV_IMAGE_FORMAT_NV12_ER, 0, 0,
                                                       imageOutputRequirements),
                          err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(imageInputRequirements, *allocatorHandle, inImageHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(imageOutputRequirements, *allocatorHandle, outImageHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(imageOutputRequirements, *allocatorHandle, refOutImageHandle), err, Error);

    generateRandomWarpPerspectiveInput(*inImageHandle, false);

    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(NVCVImageHandle inImageHandle, NVCVImageHandle refOutImageHandle,
                          NVCVImageHandle outImageHandle, PVAInterpolationType interpType,
                          PVAWarpTransformationType transformType, const char *testName)
{
    warpPerspectiveRef(inImageHandle, refOutImageHandle, interpType, transformType);
    int32_t err = checkWarpPerspectiveOutput(refOutImageHandle, outImageHandle);
    if (err)
    {
        printf("%s test failed\n", testName);
    }
    return err;
}

/**
 * @brief Test warp perspective operation using CUDA stream
 */
int SubmitWarpPerspectiveToCudaStream(int32_t outImgWidth, int32_t outImgHeight, int32_t inImgWidth,
                                      int32_t inImgHeight, PVAInterpolationType interpType,
                                      PVAWarpTransformationType transformType)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVImageHandle inImageHandle       = NULL;
    NVCVImageHandle outImageHandle      = NULL;
    NVCVImageHandle refOutImageHandle   = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cudaStream_t stream                 = NULL;
    int32_t err                         = 0;

    NVCVImageRequirements imageInputRequirements, imageOutputRequirements;

    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    /* Allocate and initialize images */
    err = AllocateAndInitializeImages(inImgWidth, inImgHeight, outImgWidth, outImgHeight, &allocatorHandle,
                                      &inImageHandle, &outImageHandle, &refOutImageHandle, &imageInputRequirements,
                                      &imageOutputRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaWarpPerspectiveCreate(&operatorHandle, &imageInputRequirements, &imageOutputRequirements,
                                                   interpType, transformType),
                          err, FreeResources);

    /* Setup and validate warp matrix */
    WarpMatrixType invWarpMatrix;
    setupWarpMatrix(invWarpMatrix, inImgWidth, inImgHeight, outImgWidth, outImgHeight);
    /* PVA can only handle tile size smaller than 32KB */
    NVCV_CHECK_ERROR_GOTO(pvaWarpPerspectiveValidate(invWarpMatrix, inImgWidth, inImgHeight, inImgWidth, outImgWidth,
                                                     outImgHeight, transformType),
                          err, FreeResources);

    /* Submit to CUDA stream */
    NVCV_CHECK_ERROR_GOTO(
        pvaWarpPerspectiveSubmit(operatorHandle, stream, inImageHandle, invWarpMatrix, outImageHandle), err,
        FreeResources);

    /* Synchronize stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(inImageHandle, refOutImageHandle, outImageHandle, interpType, transformType,
                         "Submit to CUDA stream");

FreeResources: /* clean up all allocated resources */
    cudaStreamDestroy(stream);
    nvcvImageDecRef(inImageHandle, NULL);
    nvcvImageDecRef(outImageHandle, NULL);
    nvcvImageDecRef(refOutImageHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

/**
 * @brief Test warp perspective operation using cuPVA stream
 */
int SubmitWarpPerspectiveToCupvaStream(int32_t outImgWidth, int32_t outImgHeight, int32_t inImgWidth,
                                       int32_t inImgHeight, PVAInterpolationType interpType,
                                       PVAWarpTransformationType transformType)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVImageHandle inImageHandle       = NULL;
    NVCVImageHandle outImageHandle      = NULL;
    NVCVImageHandle refOutImageHandle   = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cupvaSyncObj_t sync                 = NULL;
    cupvaStream_t stream                = NULL;
    int32_t err                         = 0;

    NVCVImageRequirements imageInputRequirements, imageOutputRequirements;

    /* Allocate and initialize images */
    err = AllocateAndInitializeImages(inImgWidth, inImgHeight, outImgWidth, outImgHeight, &allocatorHandle,
                                      &inImageHandle, &outImageHandle, &refOutImageHandle, &imageInputRequirements,
                                      &imageOutputRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaWarpPerspectiveCreate(&operatorHandle, &imageInputRequirements, &imageOutputRequirements,
                                                   interpType, transformType),
                          err, FreeResources);

    /* Setup and validate warp matrix */
    WarpMatrixType invWarpMatrix;
    setupWarpMatrix(invWarpMatrix, inImgWidth, inImgHeight, outImgWidth, outImgHeight);
    /* PVA can only handle tile size smaller than 32KB */
    NVCV_CHECK_ERROR_GOTO(pvaWarpPerspectiveValidate(invWarpMatrix, inImgWidth, inImgHeight, inImgWidth, outImgWidth,
                                                     outImgHeight, transformType),
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
        pvaWarpPerspectiveSubmit(operatorHandle, stream, inImageHandle, invWarpMatrix, outImageHandle), err,
        FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    /* Wait for completion */
    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(inImageHandle, refOutImageHandle, outImageHandle, interpType, transformType,
                         "Submit to cuPVA stream");

FreeResources: /* clean up all allocated resources */
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvImageDecRef(inImageHandle, NULL);
    nvcvImageDecRef(outImageHandle, NULL);
    nvcvImageDecRef(refOutImageHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char **argv)
{
    /* Parameter settings */
    const int32_t outImgWidth  = 1920;
    const int32_t outImgHeight = 1080;
    const int32_t inImgWidth   = 1920;
    const int32_t inImgHeight  = 1080;

    PVAInterpolationType interpType = PVA_INTERPOLATION_NN;
    // affine matrix forced the third row to be 0 0 1
    PVAWarpTransformationType transformType = PVA_WARP_TRANSFORMATION_AFFINE;

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
        cudaStreamSubmitTestResult = SubmitWarpPerspectiveToCudaStream(outImgWidth, outImgHeight, inImgWidth,
                                                                       inImgHeight, interpType, transformType);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult = SubmitWarpPerspectiveToCupvaStream(outImgWidth, outImgHeight, inImgWidth, inImgHeight,
                                                                     interpType, transformType);

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
