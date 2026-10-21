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
#include "ref/median_filter_ref.h"
#include "stdlib.h"

#include <OpMedianFilter.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Fwd.h>
#include <nvcv/Image.h>
#include <nvcv/alloc/Allocator.h>

/**
 * @brief Allocate and initialize images and kernel
 */
static int AllocateAndInitializeResources(int32_t imageWidth, int32_t imageHeight, int32_t kernelWidth,
                                          int32_t kernelHeight, NVCVImageFormat imageFormat,
                                          NVCVAllocatorHandle *allocatorHandle, NVCVImageHandle *inImageHandle,
                                          NVCVImageHandle *outImageHandle, NVCVImageHandle *refImageHandle,
                                          NVCVTensorHandle *kernelHandle, NVCVImageRequirements *imageRequirements,
                                          NVCVTensorRequirements *kernelTensorRequirements)
{
    int32_t err = 0;

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvImageCalcRequirementsPva(imageWidth, imageHeight, imageFormat, 0, 0, imageRequirements),
                          err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(imageRequirements, *allocatorHandle, inImageHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(imageRequirements, *allocatorHandle, outImageHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(imageRequirements, *allocatorHandle, refImageHandle), err, Error);

    generateMedianFilterRandomImageInput(*inImageHandle);

    // Generate box filter kernel data
    int32_t tensorRank     = 2;
    int64_t tensorShape[3] = {kernelHeight, kernelWidth, 0};
    NVCVTensorLayout tensorLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HW", &tensorLayout), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank, tensorShape, NVCV_DATA_TYPE_S8, tensorLayout, 0, 0,
                                                        kernelTensorRequirements),
                          err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(kernelTensorRequirements, *allocatorHandle, kernelHandle), err, Error);
    generateMedianFilterBoxFilterKernel(*kernelHandle, kernelHeight, kernelWidth);

    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(NVCVImageHandle inImageHandle, NVCVImageHandle refImageHandle, NVCVImageHandle outImageHandle,
                          int32_t kernelHeight, int32_t kernelWidth, NVCVTensorHandle kernelHandle,
                          NVCVBorderType borderType, int32_t borderValue, const char *testName)
{
    medianFilterRef(inImageHandle, refImageHandle, kernelHeight, kernelWidth, kernelHandle, borderType, borderValue);
    int32_t err = checkMedianFilterImageOutput(refImageHandle, outImageHandle);
    if (err)
    {
        printf("%s test failed\n", testName);
    }
    return err;
}

/**
 * @brief Test median filter operation using CUDA stream
 */
int SubmitMedianFilterToCudaStream(int32_t imageWidth, int32_t imageHeight, int32_t kernelWidth, int32_t kernelHeight,
                                   NVCVImageFormat imageFormat, NVCVBorderType borderType, int32_t borderValue)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVImageHandle inImageHandle       = NULL;
    NVCVImageHandle outImageHandle      = NULL;
    NVCVImageHandle refImageHandle      = NULL;
    NVCVTensorHandle kernelHandle       = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cudaStream_t stream                 = NULL;
    int32_t err                         = 0;

    NVCVImageRequirements imageRequirements;
    NVCVTensorRequirements kernelTensorRequirements;

    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    /* Allocate and initialize resources */
    err = AllocateAndInitializeResources(imageWidth, imageHeight, kernelWidth, kernelHeight, imageFormat,
                                         &allocatorHandle, &inImageHandle, &outImageHandle, &refImageHandle,
                                         &kernelHandle, &imageRequirements, &kernelTensorRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaMedianFilterCreate(&operatorHandle, &imageRequirements, &kernelTensorRequirements,
                                                &imageRequirements, borderType, borderValue),
                          err, FreeResources);

    /* Submit to CUDA stream */
    NVCV_CHECK_ERROR_GOTO(pvaMedianFilterSubmit(operatorHandle, stream, inImageHandle, kernelHandle, outImageHandle),
                          err, FreeResources);

    /* Synchronize stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(inImageHandle, refImageHandle, outImageHandle, kernelHeight, kernelWidth, kernelHandle,
                         borderType, borderValue, "Submit to CUDA stream");

FreeResources: /* clean up all allocated resources */
    cudaStreamDestroy(stream);
    nvcvImageDecRef(inImageHandle, NULL);
    nvcvImageDecRef(outImageHandle, NULL);
    nvcvImageDecRef(refImageHandle, NULL);
    nvcvTensorDecRef(kernelHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

/**
 * @brief Test median filter operation using cuPVA stream
 */
int SubmitMedianFilterToCupvaStream(int32_t imageWidth, int32_t imageHeight, int32_t kernelWidth, int32_t kernelHeight,
                                    NVCVImageFormat imageFormat, NVCVBorderType borderType, int32_t borderValue)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVImageHandle inImageHandle       = NULL;
    NVCVImageHandle outImageHandle      = NULL;
    NVCVImageHandle refImageHandle      = NULL;
    NVCVTensorHandle kernelHandle       = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cupvaSyncObj_t sync                 = NULL;
    cupvaStream_t stream                = NULL;
    int32_t err                         = 0;

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(&allocatorHandle), err, FreeResources);

    NVCVImageRequirements imageRequirements;
    NVCV_CHECK_ERROR_GOTO(nvcvImageCalcRequirementsPva(imageWidth, imageHeight, imageFormat, 0, 0, &imageRequirements),
                          err, FreeResources);

    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(&imageRequirements, allocatorHandle, &inImageHandle), err, FreeResources);

    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(&imageRequirements, allocatorHandle, &outImageHandle), err, FreeResources);
    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(&imageRequirements, allocatorHandle, &refImageHandle), err, FreeResources);

    generateMedianFilterRandomImageInput(inImageHandle);

    // generate box filter kernel data
    NVCVTensorRequirements kernelTensorRequirements;
    int32_t tensorRank     = 2;
    int64_t tensorShape[3] = {kernelHeight, kernelWidth, 0};
    NVCVTensorLayout tensorLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HW", &tensorLayout), err, FreeResources);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank, tensorShape, NVCV_DATA_TYPE_S8, tensorLayout, 0, 0,
                                                        &kernelTensorRequirements),
                          err, FreeResources);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&kernelTensorRequirements, allocatorHandle, &kernelHandle), err,
                          FreeResources);
    generateMedianFilterBoxFilterKernel(kernelHandle, kernelHeight, kernelWidth);

    NVCV_CHECK_ERROR_GOTO(pvaMedianFilterCreate(&operatorHandle, &imageRequirements, &kernelTensorRequirements,
                                                &imageRequirements, borderType, borderValue),
                          err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaSyncObjCreate(&sync, false, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD), err,
                           FreeResources);

    cupvaFence_t fence;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceInit(&fence, sync), err, FreeResources);

    cupvaCmd_t rf;
    CUPVA_CHECK_ERROR_GOTO(CupvaCmdRequestFencesInit(&rf, &fence, 1), err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamCreate(&stream, CUPVA_PVA0, CUPVA_VPU_ANY), err, FreeResources);

    cupvaCmd_t const *cmds[1]  = {&rf};
    cupvaCmdStatus_t status[1] = {NULL};

    NVCV_CHECK_ERROR_GOTO(pvaMedianFilterSubmit(operatorHandle, stream, inImageHandle, kernelHandle, outImageHandle),
                          err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    // generate ref output
    medianFilterRef(inImageHandle, refImageHandle, kernelHeight, kernelWidth, kernelHandle, borderType, borderValue);

    // check results
    err = checkMedianFilterImageOutput(refImageHandle, outImageHandle);

    if (err)
    {
        printf("Submit to cuPVA stream test failed\n");
    }

FreeResources: /* clean up all allocated resources */
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvImageDecRef(inImageHandle, NULL);
    nvcvImageDecRef(outImageHandle, NULL);
    nvcvImageDecRef(refImageHandle, NULL);
    nvcvTensorDecRef(kernelHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char **argv)
{
    /* Parameter settings */
    NVCVImageFormat imageFormat = NVCV_IMAGE_FORMAT_U8;
    NVCVBorderType borderType   = NVCV_BORDER_REPLICATE;
    int32_t borderValue         = 0;

    const int32_t imageWidth  = 960;
    const int32_t imageHeight = 540;
    int32_t kernelWidth       = 3;
    int32_t kernelHeight      = 3;

    // Parse command-line arguments
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "-f") || !strcmp(argv[i], "--format"))
        {
            if (!strcmp(argv[i + 1], "u8"))
            {
                printf("MedianFilter sample test: format=U8\n");
                imageFormat = NVCV_IMAGE_FORMAT_U8;
            }
            else if (!strcmp(argv[i + 1], "y8"))
            {
                printf("MedianFilter sample test: format=Y8\n");
                imageFormat = NVCV_IMAGE_FORMAT_Y8;
            }
            else if (!strcmp(argv[i + 1], "y8_er"))
            {
                printf("MedianFilter sample test: format=Y8_ER\n");
                imageFormat = NVCV_IMAGE_FORMAT_Y8_ER;
            }
            else if (!strcmp(argv[i + 1], "s8"))
            {
                printf("MedianFilter sample test: format=S8\n");
                imageFormat = NVCV_IMAGE_FORMAT_S8;
            }
            else if (!strcmp(argv[i + 1], "u16"))
            {
                printf("MedianFilter sample test: format=U16\n");
                imageFormat = NVCV_IMAGE_FORMAT_U16;
            }
            else if (!strcmp(argv[i + 1], "y16"))
            {
                printf("MedianFilter sample test: format=Y16\n");
                imageFormat = NVCV_IMAGE_FORMAT_Y16;
            }
            else if (!strcmp(argv[i + 1], "y16_er"))
            {
                printf("MedianFilter sample test: format=Y16_ER\n");
                imageFormat = NVCV_IMAGE_FORMAT_Y16_ER;
            }
            else if (!strcmp(argv[i + 1], "s16"))
            {
                printf("MedianFilter sample test: format=S16\n");
                imageFormat = NVCV_IMAGE_FORMAT_S16;
            }
            else if (!strcmp(argv[i + 1], "2s16"))
            {
                printf("MedianFilter sample test: format=2S16\n");
                imageFormat = NVCV_IMAGE_FORMAT_2S16;
            }
            else if (!strcmp(argv[i + 1], "u32"))
            {
                printf("MedianFilter sample test: format=U32\n");
                imageFormat = NVCV_IMAGE_FORMAT_U32;
            }
            else if (!strcmp(argv[i + 1], "s32"))
            {
                printf("MedianFilter sample test: format=S32\n");
                imageFormat = NVCV_IMAGE_FORMAT_S32;
            }
            else
            {
                printf("MedianFilter sample test: invalid image format!\n");
                return -1;
            }
            i++;
        }
        else if (!strcmp(argv[i], "-b") || !strcmp(argv[i], "--border_type"))
        {
            if (!strcmp(argv[i + 1], "zero"))
            {
                borderType = NVCV_BORDER_CONSTANT;
            }
            else if (!strcmp(argv[i + 1], "clamp"))
            {
                borderType = NVCV_BORDER_REPLICATE;
            }
            else
            {
                printf("MedianFilter sample test: invalid border type!\n");
                return -1;
            }
            i++;
        }
        else if (!strcmp(argv[i], "-kw") || !strcmp(argv[i], "--kernel_width"))
        {
            kernelWidth = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-kh") || !strcmp(argv[i], "--kernel_height"))
        {
            kernelHeight = atoi(argv[i + 1]);
            i++;
        }
    }

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
        cudaStreamSubmitTestResult = SubmitMedianFilterToCudaStream(imageWidth, imageHeight, kernelWidth, kernelHeight,
                                                                    imageFormat, borderType, borderValue);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult = SubmitMedianFilterToCupvaStream(imageWidth, imageHeight, kernelWidth, kernelHeight,
                                                                  imageFormat, borderType, borderValue);

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
