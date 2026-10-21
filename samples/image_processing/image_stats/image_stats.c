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
#include "ref/image_stats_ref.h"

#include <ImageIO.h>
#include <OpImageStats.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Fwd.h>
#include <nvcv/Image.h>
#include <nvcv/alloc/Allocator.h>
#include <stdlib.h>

/**
 * @brief Allocate and initialize images and tensors
 */
static int AllocateAndInitializeResources(int argc, char **argv, uint32_t srcImageWidth, uint32_t srcImageHeight,
                                          NVCVImageFormat imageFormat, bool useFixedInput, bool hasMask,
                                          NVCVAllocatorHandle *allocatorHandle, NVCVImageHandle *imageHandle,
                                          NVCVImageHandle *maskHandle, NVCVTensorHandle *outTensorHandle,
                                          NVCVTensorHandle *refTensorHandle, NVCVImageRequirements *inImageRequirements,
                                          NVCVImageRequirements *maskRequirements,
                                          NVCVImageRequirements **maskRequirementsPtr,
                                          NVCVTensorRequirements *outTensorRequirements)
{
    int32_t err = 0;

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);

    // Input image
    NVCV_CHECK_ERROR_GOTO(
        nvcvImageCalcRequirementsPva(srcImageWidth, srcImageHeight, imageFormat, 0, 0, inImageRequirements), err,
        Error);
    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(inImageRequirements, *allocatorHandle, imageHandle), err, Error);

    // Mask image (optional)
    *maskRequirementsPtr = NULL;
    if (hasMask)
    {
        *maskRequirementsPtr = maskRequirements;
        NVCV_CHECK_ERROR_GOTO(nvcvImageCalcRequirementsPva(srcImageWidth, srcImageHeight, NVCV_IMAGE_FORMAT_U8, 0, 0,
                                                           *maskRequirementsPtr),
                              err, Error);
        NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(*maskRequirementsPtr, *allocatorHandle, maskHandle), err, Error);
    }

    // Output tensor
    int32_t tensorRank;
    int64_t tensorShape[3];
    NVCVTensorLayout tensorLayout;

    tensorRank     = 1;
    tensorShape[0] = sizeof(PVAImageStatOutput);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("W", &tensorLayout), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank, tensorShape, NVCV_DATA_TYPE_U8, tensorLayout, 0, 0,
                                                        outTensorRequirements),
                          err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(outTensorRequirements, *allocatorHandle, outTensorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(outTensorRequirements, *allocatorHandle, refTensorHandle), err, Error);

    // Generate input image data: fixed or random
    if (useFixedInput)
    {
        const int32_t maxPathLength = 200;
        char assetDir[maxPathLength];

        NVCV_CHECK_ERROR_GOTO(GetAssetsDirectory(argc, argv, assetDir, maxPathLength), err, Error);
        generateImageStatsFixedImageInput(*imageHandle, assetDir);
    }
    else
    {
        generateImageStatsRandomImageInput(*imageHandle);
    }

    // Generate random binary mask
    if (hasMask)
    {
        generateImageStatsRandomBinaryMask(*maskHandle, 0.7f);
    }

    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(NVCVImageHandle imageHandle, NVCVImageHandle maskHandle, NVCVTensorHandle outTensorHandle,
                          NVCVTensorHandle refTensorHandle, uint32_t flags, const char *testName)
{
    imageStatsRef(imageHandle, maskHandle, refTensorHandle, flags);
    int32_t err = checkImageStatsOutput(imageHandle, outTensorHandle, refTensorHandle);
    if (err)
    {
        printf("%s test failed\n", testName);
    }
    return err;
}

/**
 * @brief Test image stats operation using CUDA stream
 */
int SubmitImageStatsToCudaStream(int argc, char **argv, uint32_t srcImageWidth, uint32_t srcImageHeight,
                                 NVCVImageFormat imageFormat, bool useFixedInput, bool hasMask, uint32_t flags)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVImageHandle imageHandle         = NULL;
    NVCVImageHandle maskHandle          = NULL;
    NVCVTensorHandle outTensorHandle    = NULL;
    NVCVTensorHandle refTensorHandle    = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cudaStream_t stream                 = NULL;
    int32_t err                         = 0;

    NVCVImageRequirements inImageRequirements;
    NVCVImageRequirements maskRequirements     = {0};
    NVCVImageRequirements *maskRequirementsPtr = NULL;
    NVCVTensorRequirements outTensorRequirements;

    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    /* Allocate and initialize resources */
    err = AllocateAndInitializeResources(argc, argv, srcImageWidth, srcImageHeight, imageFormat, useFixedInput, hasMask,
                                         &allocatorHandle, &imageHandle, &maskHandle, &outTensorHandle,
                                         &refTensorHandle, &inImageRequirements, &maskRequirements,
                                         &maskRequirementsPtr, &outTensorRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaImageStatsCreate(&operatorHandle, &inImageRequirements, maskRequirementsPtr, flags), err,
                          FreeResources);

    /* Submit to CUDA stream */
    NVCV_CHECK_ERROR_GOTO(pvaImageStatsSubmit(operatorHandle, stream, imageHandle, maskHandle, outTensorHandle), err,
                          FreeResources);

    /* Synchronize stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(imageHandle, maskHandle, outTensorHandle, refTensorHandle, flags, "Submit to CUDA stream");

FreeResources: /* clean up all allocated resources */
    cudaStreamDestroy(stream);
    nvcvImageDecRef(imageHandle, NULL);
    nvcvImageDecRef(maskHandle, NULL);
    nvcvTensorDecRef(outTensorHandle, NULL);
    nvcvTensorDecRef(refTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

/**
 * @brief Test image stats operation using cuPVA stream
 */
int SubmitImageStatsToCupvaStream(int argc, char **argv, uint32_t srcImageWidth, uint32_t srcImageHeight,
                                  NVCVImageFormat imageFormat, bool useFixedInput, bool hasMask, uint32_t flags)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVImageHandle imageHandle         = NULL;
    NVCVImageHandle maskHandle          = NULL;
    NVCVTensorHandle outTensorHandle    = NULL;
    NVCVTensorHandle refTensorHandle    = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cupvaSyncObj_t sync                 = NULL;
    cupvaStream_t stream                = NULL;
    int32_t err                         = 0;

    NVCVImageRequirements inImageRequirements;
    NVCVImageRequirements maskRequirements     = {0};
    NVCVImageRequirements *maskRequirementsPtr = NULL;
    NVCVTensorRequirements outTensorRequirements;

    /* Allocate and initialize resources */
    err = AllocateAndInitializeResources(argc, argv, srcImageWidth, srcImageHeight, imageFormat, useFixedInput, hasMask,
                                         &allocatorHandle, &imageHandle, &maskHandle, &outTensorHandle,
                                         &refTensorHandle, &inImageRequirements, &maskRequirements,
                                         &maskRequirementsPtr, &outTensorRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaImageStatsCreate(&operatorHandle, &inImageRequirements, maskRequirementsPtr, flags), err,
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

    NVCV_CHECK_ERROR_GOTO(pvaImageStatsSubmit(operatorHandle, stream, imageHandle, maskHandle, outTensorHandle), err,
                          FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    /* Wait for completion */
    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(imageHandle, maskHandle, outTensorHandle, refTensorHandle, flags, "Submit to cuPVA stream");

FreeResources: /* clean up all allocated resources */
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvImageDecRef(imageHandle, NULL);
    nvcvImageDecRef(maskHandle, NULL);
    nvcvTensorDecRef(outTensorHandle, NULL);
    nvcvTensorDecRef(refTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char **argv)
{
    /* Parameter settings */
    NVCVImageFormat imageFormat = NVCV_IMAGE_FORMAT_U8;
    bool useFixedInput          = false;
    bool hasMask                = false;

    uint32_t srcImageWidth  = 960;
    uint32_t srcImageHeight = 540;

    uint32_t flags = PVA_IMAGE_STAT_FLAG_COVARIANCE;

    // Parse command-line arguments
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "-f") || !strcmp(argv[i], "--format"))
        {
            if (!strcmp(argv[i + 1], "u8"))
            {
                printf("ImageStats sample test: format=U8\n");
                imageFormat = NVCV_IMAGE_FORMAT_U8;
            }
            else if (!strcmp(argv[i + 1], "rgb"))
            {
                printf("ImageStats sample test: format=RGB8\n");
                imageFormat = NVCV_IMAGE_FORMAT_RGB8;
            }
            else if (!strcmp(argv[i + 1], "bgr"))
            {
                printf("ImageStats sample test: format=BGR8\n");
                imageFormat = NVCV_IMAGE_FORMAT_BGR8;
            }
            else if (!strcmp(argv[i + 1], "nv12"))
            {
                printf("ImageStats sample test: format=NV12\n");
                imageFormat = NVCV_IMAGE_FORMAT_NV12;
            }
            else if (!strcmp(argv[i + 1], "nv12_er"))
            {
                printf("ImageStats sample test: format=NV12_ER\n");
                imageFormat = NVCV_IMAGE_FORMAT_NV12_ER;
            }
            else
            {
                printf("Image resize sample test: invalid image format!\n");
                return -1;
            }
            i++;
        }
        // use fixed input image
        else if (!strcmp(argv[i], "--fixed"))
        {
            useFixedInput  = true;
            srcImageWidth  = 768;
            srcImageHeight = 512;
        }
        // has mask
        else if (!strcmp(argv[i], "--mask") || !strcmp(argv[i], "-m"))
        {
            hasMask = true;
        }
        // custom flags
        else if (!strcmp(argv[i], "--flags"))
        {
            flags = atoi(argv[i + 1]);
            i++;
        }
    }

    printf("IMAGE_STATS SAMPLE TEST, w=%d, h=%d, hasMask=%d, flags=%d\n", srcImageWidth, srcImageHeight, hasMask,
           flags);

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
        cudaStreamSubmitTestResult = SubmitImageStatsToCudaStream(argc, argv, srcImageWidth, srcImageHeight,
                                                                  imageFormat, useFixedInput, hasMask, flags);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult = SubmitImageStatsToCupvaStream(argc, argv, srcImageWidth, srcImageHeight, imageFormat,
                                                                useFixedInput, hasMask, flags);

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
