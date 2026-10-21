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
#include "ref/image_blend_ref.h"

#include <OpImageBlend.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Fwd.h>
#include <nvcv/Image.h>
#include <nvcv/alloc/Allocator.h>

/**
 * @brief Allocate and initialize images
 */
static int AllocateAndInitializeImages(int32_t imageWidth, int32_t imageHeight, NVCVImageFormat imageFormat,
                                       NVCVAllocatorHandle *allocatorHandle, NVCVImageHandle *src0ImageHandle,
                                       NVCVImageHandle *src1ImageHandle, NVCVImageHandle *dstImageHandle,
                                       NVCVImageHandle *refImageHandle, NVCVImageRequirements *imageRequirements)
{
    int32_t err = 0;

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvImageCalcRequirementsPva(imageWidth, imageHeight, imageFormat, 0, 0, imageRequirements),
                          err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(imageRequirements, *allocatorHandle, src0ImageHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(imageRequirements, *allocatorHandle, src1ImageHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(imageRequirements, *allocatorHandle, dstImageHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(imageRequirements, *allocatorHandle, refImageHandle), err, Error);

    generateRandomImageBlendInput(*src0ImageHandle, *src1ImageHandle);

    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(NVCVImageHandle refImageHandle, NVCVImageHandle dstImageHandle, const char *testName)
{
    int32_t err = checkImageBlendOutput(refImageHandle, dstImageHandle);
    if (err)
    {
        printf("%s test failed\n", testName);
    }
    return err;
}

/**
 * @brief Test image blend operation using CUDA stream
 */
int SubmitImageBlendToCudaStream(int32_t imageWidth, int32_t imageHeight, NVCVImageFormat imageFormat, float alpha)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVImageHandle src0ImageHandle     = NULL;
    NVCVImageHandle src1ImageHandle     = NULL;
    NVCVImageHandle dstImageHandle      = NULL;
    NVCVImageHandle refImageHandle      = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cudaStream_t stream                 = NULL;
    int32_t err                         = 0;

    NVCVImageRequirements imageRequirements;

    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    /* Allocate and initialize images */
    err = AllocateAndInitializeImages(imageWidth, imageHeight, imageFormat, &allocatorHandle, &src0ImageHandle,
                                      &src1ImageHandle, &dstImageHandle, &refImageHandle, &imageRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaImageBlendCreate(&operatorHandle, &imageRequirements), err, FreeResources);

    /* Submit to CUDA stream */
    NVCV_CHECK_ERROR_GOTO(
        pvaImageBlendSubmit(operatorHandle, stream, src0ImageHandle, src1ImageHandle, alpha, dstImageHandle), err,
        FreeResources);

    /* Synchronize stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    /* Compute reference */
    imageBlendRef(src0ImageHandle, src1ImageHandle, refImageHandle, alpha);

    /* Validate output */
    err = ValidateOutput(refImageHandle, dstImageHandle, "Submit to CUDA stream");

FreeResources: /* clean up all allocated resources */
    cudaStreamDestroy(stream);
    nvcvImageDecRef(src0ImageHandle, NULL);
    nvcvImageDecRef(src1ImageHandle, NULL);
    nvcvImageDecRef(dstImageHandle, NULL);
    nvcvImageDecRef(refImageHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

/**
 * @brief Test image blend operation using cuPVA stream
 */
int SubmitImageBlendToCupvaStream(int32_t imageWidth, int32_t imageHeight, NVCVImageFormat imageFormat, float alpha)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVImageHandle src0ImageHandle     = NULL;
    NVCVImageHandle src1ImageHandle     = NULL;
    NVCVImageHandle dstImageHandle      = NULL;
    NVCVImageHandle refImageHandle      = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cupvaSyncObj_t sync                 = NULL;
    cupvaStream_t stream                = NULL;
    int32_t err                         = 0;

    NVCVImageRequirements imageRequirements;

    /* Allocate and initialize images */
    err = AllocateAndInitializeImages(imageWidth, imageHeight, imageFormat, &allocatorHandle, &src0ImageHandle,
                                      &src1ImageHandle, &dstImageHandle, &refImageHandle, &imageRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaImageBlendCreate(&operatorHandle, &imageRequirements), err, FreeResources);

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
        pvaImageBlendSubmit(operatorHandle, stream, src0ImageHandle, src1ImageHandle, alpha, dstImageHandle), err,
        FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    /* Wait for completion */
    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    /* Compute reference */
    imageBlendRef(src0ImageHandle, src1ImageHandle, refImageHandle, alpha);

    /* Validate output */
    err = ValidateOutput(refImageHandle, dstImageHandle, "Submit to cuPVA stream");

FreeResources: /* clean up all allocated resources */
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvImageDecRef(src0ImageHandle, NULL);
    nvcvImageDecRef(src1ImageHandle, NULL);
    nvcvImageDecRef(dstImageHandle, NULL);
    nvcvImageDecRef(refImageHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char **argv)
{
    /* Parameter settings */
    const int32_t imageWidth  = 1920;
    const int32_t imageHeight = 1080;
    const float alpha         = 0.5f;

    NVCVImageFormat imageFormat = NVCV_IMAGE_FORMAT_BGRA8;

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
        cudaStreamSubmitTestResult = SubmitImageBlendToCudaStream(imageWidth, imageHeight, imageFormat, alpha);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult = SubmitImageBlendToCupvaStream(imageWidth, imageHeight, imageFormat, alpha);

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
