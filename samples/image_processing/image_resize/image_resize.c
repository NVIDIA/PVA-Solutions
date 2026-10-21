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
#include "ref/image_resize_ref.h"

#include <OpImageResize.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Fwd.h>
#include <nvcv/Image.h>
#include <nvcv/alloc/Allocator.h>

/**
 * @brief Allocate and initialize images
 */
static int AllocateAndInitializeImages(int32_t srcImageWidth, int32_t srcImageHeight, int32_t dstImageWidth,
                                       int32_t dstImageHeight, NVCVImageFormat imageFormat,
                                       NVCVAllocatorHandle *allocatorHandle, NVCVImageHandle *inImageHandle,
                                       NVCVImageHandle *outImageHandle, NVCVImageHandle *refFixedImageHandle,
                                       NVCVImageRequirements *inImageRequirements,
                                       NVCVImageRequirements *outImageRequirements)
{
    int32_t err = 0;

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(
        nvcvImageCalcRequirementsPva(srcImageWidth, srcImageHeight, imageFormat, 0, 0, inImageRequirements), err,
        Error);

    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(inImageRequirements, *allocatorHandle, inImageHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(
        nvcvImageCalcRequirementsPva(dstImageWidth, dstImageHeight, imageFormat, 0, 0, outImageRequirements), err,
        Error);

    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(outImageRequirements, *allocatorHandle, outImageHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(outImageRequirements, *allocatorHandle, refFixedImageHandle), err, Error);

    generateRandomImageInput(*inImageHandle);

    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(NVCVImageHandle refFixedImageHandle, NVCVImageHandle outImageHandle, const char *testName)
{
    int32_t err = checkImageOutput(refFixedImageHandle, outImageHandle, 0);
    if (err)
    {
        printf("%s test failed\n", testName);
    }
    return err;
}

/**
 * @brief Test image resize operation using CUDA stream
 */
int SubmitImageResizeToCudaStream(int32_t srcImageWidth, int32_t srcImageHeight, int32_t dstImageWidth,
                                  int32_t dstImageHeight, NVCVImageFormat imageFormat, PVAInterpolationType resizeType)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVImageHandle inImageHandle       = NULL;
    NVCVImageHandle outImageHandle      = NULL;
    NVCVImageHandle refFixedImageHandle = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cudaStream_t stream                 = NULL;
    int32_t err                         = 0;

    NVCVImageRequirements inImageRequirements;
    NVCVImageRequirements outImageRequirements;

    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    /* Allocate and initialize images */
    err = AllocateAndInitializeImages(srcImageWidth, srcImageHeight, dstImageWidth, dstImageHeight, imageFormat,
                                      &allocatorHandle, &inImageHandle, &outImageHandle, &refFixedImageHandle,
                                      &inImageRequirements, &outImageRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(
        pvaImageResizeCreate(&operatorHandle, &inImageRequirements, &outImageRequirements, resizeType), err,
        FreeResources);

    /* Submit to CUDA stream */
    NVCV_CHECK_ERROR_GOTO(pvaImageResizeSubmit(operatorHandle, stream, inImageHandle, outImageHandle), err,
                          FreeResources);

    /* Synchronize stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    /* Generate reference output */
    imageResizeFixedPointRef(inImageHandle, refFixedImageHandle, resizeType);

    /* Validate output */
    err = ValidateOutput(refFixedImageHandle, outImageHandle, "Submit to CUDA stream");

FreeResources: /* clean up all allocated resources */
    cudaStreamDestroy(stream);
    nvcvImageDecRef(inImageHandle, NULL);
    nvcvImageDecRef(outImageHandle, NULL);
    nvcvImageDecRef(refFixedImageHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

/**
 * @brief Test image resize operation using cuPVA stream
 */
int SubmitImageResizeToCupvaStream(int32_t srcImageWidth, int32_t srcImageHeight, int32_t dstImageWidth,
                                   int32_t dstImageHeight, NVCVImageFormat imageFormat, PVAInterpolationType resizeType)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVImageHandle inImageHandle       = NULL;
    NVCVImageHandle outImageHandle      = NULL;
    NVCVImageHandle refFixedImageHandle = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cupvaSyncObj_t sync                 = NULL;
    cupvaStream_t stream                = NULL;
    int32_t err                         = 0;

    NVCVImageRequirements inImageRequirements;
    NVCVImageRequirements outImageRequirements;

    /* Allocate and initialize images */
    err = AllocateAndInitializeImages(srcImageWidth, srcImageHeight, dstImageWidth, dstImageHeight, imageFormat,
                                      &allocatorHandle, &inImageHandle, &outImageHandle, &refFixedImageHandle,
                                      &inImageRequirements, &outImageRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(
        pvaImageResizeCreate(&operatorHandle, &inImageRequirements, &outImageRequirements, resizeType), err,
        FreeResources);

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

    NVCV_CHECK_ERROR_GOTO(pvaImageResizeSubmit(operatorHandle, stream, inImageHandle, outImageHandle), err,
                          FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    /* Wait for completion */
    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    /* Generate reference output */
    imageResizeFixedPointRef(inImageHandle, refFixedImageHandle, resizeType);

    /* Validate output */
    err = ValidateOutput(refFixedImageHandle, outImageHandle, "Submit to cuPVA stream");

FreeResources: /* clean up all allocated resources */
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvImageDecRef(inImageHandle, NULL);
    nvcvImageDecRef(outImageHandle, NULL);
    nvcvImageDecRef(refFixedImageHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char **argv)
{
    /* Parameter settings */
    NVCVImageFormat imageFormat     = NVCV_IMAGE_FORMAT_U8;
    PVAInterpolationType resizeType = PVA_INTERPOLATION_LINEAR;

    // Parse command-line arguments
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "-f") || !strcmp(argv[i], "--format"))
        {
            if (!strcmp(argv[i + 1], "u8"))
            {
                printf("ImageResize sample test: format=U8\n");
                imageFormat = NVCV_IMAGE_FORMAT_U8;
            }
            else if (!strcmp(argv[i + 1], "u16"))
            {
                printf("ImageResize sample test: format=U16\n");
                imageFormat = NVCV_IMAGE_FORMAT_U16;
            }
            else if (!strcmp(argv[i + 1], "rgb"))
            {
                printf("ImageResize sample test: format=RGB8\n");
                imageFormat = NVCV_IMAGE_FORMAT_RGB8;
            }
            else if (!strcmp(argv[i + 1], "rgba"))
            {
                printf("ImageResize sample test: format=RGBA\n");
                imageFormat = NVCV_IMAGE_FORMAT_RGBA8;
            }
            else if (!strcmp(argv[i + 1], "nv12"))
            {
                printf("ImageResize sample test: format=NV12\n");
                imageFormat = NVCV_IMAGE_FORMAT_NV12;
            }
            else
            {
                printf("Image resize sample test: invalid image format!\n");
                return -1;
            }
            i++;
        }
        else if (!strcmp(argv[i], "-r") || !strcmp(argv[i], "--resize_type"))
        {
            if (!strcmp(argv[i + 1], "nn"))
            {
                printf("ImageResize sample test: type=NN\n");
                resizeType = PVA_INTERPOLATION_NN;
            }
            else
            {
                printf("ImageResize sample test: type=LINEAR\n");
                resizeType = PVA_INTERPOLATION_LINEAR;
            }
            i++;
        }
    }

    const int32_t srcImageWidth  = 3840;
    const int32_t srcImageHeight = 2160;
    const int32_t dstImageWidth  = 1920;
    const int32_t dstImageHeight = 960;

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
        cudaStreamSubmitTestResult = SubmitImageResizeToCudaStream(srcImageWidth, srcImageHeight, dstImageWidth,
                                                                   dstImageHeight, imageFormat, resizeType);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult = SubmitImageResizeToCupvaStream(srcImageWidth, srcImageHeight, dstImageWidth,
                                                                 dstImageHeight, imageFormat, resizeType);

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
