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

#include <ImageIO.h>
#include <OpBackgroundSubtractor.h>
#include <PvaAllocator.h>
#include <back_sub_ref_wrap.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Allocate and initialize images
 */
static int AllocateAndInitializeImages(int32_t imgWidth, int32_t imgHeight, NVCVImageFormat imageFormat,
                                       NVCVAllocatorHandle *allocatorHandle, NVCVImageHandle *inImageHandle,
                                       NVCVImageHandle *outImageHandle, NVCVImageHandle *outBGImageHandle,
                                       NVCVImageRequirements *imageRequirements)
{
    int32_t err = 0;

    NVCV_CHECK_ERROR_GOTO(nvcvImageCalcRequirementsPva(imgWidth, imgHeight, imageFormat, 0, 0, imageRequirements), err,
                          Error);

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(imageRequirements, *allocatorHandle, inImageHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(imageRequirements, *allocatorHandle, outImageHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(imageRequirements, *allocatorHandle, outBGImageHandle), err, Error);

    return 0;

Error:
    return err;
}

/**
 * @brief Validate background subtractor output
 */
static int ValidateOutput(uint8_t *refMask, uint8_t *outImagePtr_h, uint8_t *refBGImg, uint8_t *outBGImagePtr_h,
                          int32_t imgWidth, int32_t imgHeight, int32_t errCountMask, int32_t errCountBG,
                          const char *testName)
{
    float maxErrFrac = 0.00005;
    int maxNumErrs   = maxErrFrac * imgWidth * imgHeight;
    int32_t err      = 0;

    if (errCountMask > maxNumErrs)
    {
        printf("%s test failed: Num pixel mismatches for mask exceeded threshold\n", testName);
        err = 1;
    }
    if (errCountBG > maxNumErrs)
    {
        printf("%s test failed: Num pixel mismatches for background exceeded threshold\n", testName);
        err = 1;
    }

    if (err)
    {
        printf("%s test failed\n", testName);
    }

    return err;
}

/**
 * @brief Test background subtractor operation using CUDA stream
 */
int SubmitBackgroundSubtractorToCudaStream(int argc, char **argv)
{
    const int32_t maxPathLength = 200;
    char assetsDirectory[maxPathLength];

    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVImageHandle inImageHandle       = NULL;
    NVCVImageHandle outImageHandle      = NULL;
    NVCVImageHandle outBGImageHandle    = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cudaStream_t stream                 = NULL;
    int32_t err                         = 0;
    BGSubRefHandle *refHandle           = NULL;

    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    err = GetAssetsDirectory(argc, argv, assetsDirectory, maxPathLength);
    if (err)
        goto FreeResources;

    NVCVImageFormat inImageFormat = NVCV_IMAGE_FORMAT_U8;

    int imgHeight = 540;
    int imgWidth  = 960;
    NVCVImageRequirements imageRequirements;

    uint8_t *refMask  = (uint8_t *)malloc(imgWidth * imgHeight * sizeof(uint8_t));
    uint8_t *refBGImg = (uint8_t *)malloc(imgWidth * imgHeight * sizeof(uint8_t));

    /* Allocate and initialize images */
    err = AllocateAndInitializeImages(imgWidth, imgHeight, inImageFormat, &allocatorHandle, &inImageHandle,
                                      &outImageHandle, &outBGImageHandle, &imageRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(
        pvaBackgroundSubtractorCreate(&operatorHandle, &imageRequirements, &imageRequirements, PVA_GMM_DATA_TYPE_FP16),
        err, FreeResources);

    BGSubRefCreate(&refHandle, imgWidth, imgHeight);

    NVCVImageData inImageData, outImageData, outBGImageData;
    nvcvImageExportData(inImageHandle, &inImageData);
    nvcvImageExportData(outImageHandle, &outImageData);
    nvcvImageExportData(outBGImageHandle, &outBGImageData);
    uint8_t *inImagePtr_h = NULL, *outImagePtr_h = NULL, *outBGImagePtr_h = NULL;
    CupvaMemGetHostPointer((void **)&inImagePtr_h, (void *)inImageData.buffer.strided.planes[0].basePtr);
    CupvaMemGetHostPointer((void **)&outImagePtr_h, (void *)outImageData.buffer.strided.planes[0].basePtr);
    CupvaMemGetHostPointer((void **)&outBGImagePtr_h, (void *)outBGImageData.buffer.strided.planes[0].basePtr);

    int errCountMask = 0, errCountBG = 0;

    /* Process frames */
    char bgAssetsDir[MAX_PATH_LENGTH];
    for (int i = 0; i < 5; i++)
    {
        printf("Frame %d\n", i);
        snprintf(bgAssetsDir, sizeof(bgAssetsDir), "%s/background_subtractor/", assetsDirectory);
        generateBackgroundSubtractorInput(inImagePtr_h, bgAssetsDir, i, imgWidth, imgHeight);

        NVCV_CHECK_ERROR_GOTO(pvaBackgroundSubtractorSubmit(operatorHandle, stream, inImageHandle, outImageHandle,
                                                            outBGImageHandle, 0.3, 16, 1, 128),
                              err, FreeResources);

        CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

        BGSubRefSubmit(refHandle, inImagePtr_h, refMask, refBGImg, 16, 0.3, 1, 128);

        int32_t maskDifferenceTolerance            = 0;
        int32_t backgroundImageDifferenceTolerance = 1;
        errCountMask += compareArrays(refMask, outImagePtr_h, imgWidth, imgHeight, maskDifferenceTolerance);
        errCountBG += compareArrays(refBGImg, outBGImagePtr_h, imgWidth, imgHeight, backgroundImageDifferenceTolerance);
    }

    /* Validate output */
    err = ValidateOutput(refMask, outImagePtr_h, refBGImg, outBGImagePtr_h, imgWidth, imgHeight, errCountMask,
                         errCountBG, "Submit to CUDA stream");

FreeResources:
    cudaStreamDestroy(stream);
    nvcvImageDecRef(inImageHandle, NULL);
    nvcvImageDecRef(outImageHandle, NULL);
    nvcvImageDecRef(outBGImageHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    BGSubRefDestroy(refHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);
    free(refMask);
    free(refBGImg);

    return err;
}

/**
 * @brief Test background subtractor operation using cuPVA stream
 */
int SubmitBackgroundSubtractorToCupvaStream(int argc, char **argv)
{
    const int32_t maxPathLength = 200;
    char assetsDirectory[maxPathLength];

    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVImageHandle inImageHandle       = NULL;
    NVCVImageHandle outImageHandle      = NULL;
    NVCVImageHandle outBGImageHandle    = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cupvaSyncObj_t sync                 = NULL;
    cupvaStream_t stream                = NULL;
    int32_t err                         = 0;
    BGSubRefHandle *refHandle           = NULL;

    err = GetAssetsDirectory(argc, argv, assetsDirectory, maxPathLength);
    if (err)
        goto FreeResources;

    NVCVImageFormat inImageFormat = NVCV_IMAGE_FORMAT_U8;

    int imgHeight = 540;
    int imgWidth  = 960;
    NVCVImageRequirements imageRequirements;

    uint8_t *refMask  = (uint8_t *)malloc(imgWidth * imgHeight * sizeof(uint8_t));
    uint8_t *refBGImg = (uint8_t *)malloc(imgWidth * imgHeight * sizeof(uint8_t));

    /* Allocate and initialize images */
    err = AllocateAndInitializeImages(imgWidth, imgHeight, inImageFormat, &allocatorHandle, &inImageHandle,
                                      &outImageHandle, &outBGImageHandle, &imageRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(
        pvaBackgroundSubtractorCreate(&operatorHandle, &imageRequirements, &imageRequirements, PVA_GMM_DATA_TYPE_FP16),
        err, FreeResources);

    BGSubRefCreate(&refHandle, imgWidth, imgHeight);

    /* Create cuPVA stream and synchronization objects */
    CUPVA_CHECK_ERROR_GOTO(CupvaSyncObjCreate(&sync, false, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD), err,
                           FreeResources);

    NVCVImageData inImageData, outImageData, outBGImageData;
    nvcvImageExportData(inImageHandle, &inImageData);
    nvcvImageExportData(outImageHandle, &outImageData);
    nvcvImageExportData(outBGImageHandle, &outBGImageData);
    uint8_t *inImagePtr_h = NULL, *outImagePtr_h = NULL, *outBGImagePtr_h = NULL;
    CupvaMemGetHostPointer((void **)&inImagePtr_h, (void *)inImageData.buffer.strided.planes[0].basePtr);
    CupvaMemGetHostPointer((void **)&outImagePtr_h, (void *)outImageData.buffer.strided.planes[0].basePtr);
    CupvaMemGetHostPointer((void **)&outBGImagePtr_h, (void *)outBGImageData.buffer.strided.planes[0].basePtr);

    int errCountMask = 0, errCountBG = 0;
    cupvaFence_t fence;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceInit(&fence, sync), err, FreeResources);

    cupvaCmd_t rf;
    CUPVA_CHECK_ERROR_GOTO(CupvaCmdRequestFencesInit(&rf, &fence, 1), err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamCreate(&stream, CUPVA_PVA0, CUPVA_VPU_ANY), err, FreeResources);

    /* Submit to cuPVA stream */
    cupvaCmd_t const *cmds[1]  = {&rf};
    cupvaCmdStatus_t status[1] = {NULL};

    /* Process frames */
    char bgAssetsDir[MAX_PATH_LENGTH];
    for (int i = 0; i < 5; i++)
    {
        printf("Frame %d\n", i);
        snprintf(bgAssetsDir, sizeof(bgAssetsDir), "%s/background_subtractor/", assetsDirectory);
        generateBackgroundSubtractorInput(inImagePtr_h, bgAssetsDir, i, imgWidth, imgHeight);

        NVCV_CHECK_ERROR_GOTO(pvaBackgroundSubtractorSubmit(operatorHandle, stream, inImageHandle, outImageHandle,
                                                            outBGImageHandle, 0.3, 16, 1, 128),
                              err, FreeResources);

        CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

        /* Wait for completion */
        bool waitSuccess;
        CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

        BGSubRefSubmit(refHandle, inImagePtr_h, refMask, refBGImg, 16, 0.3, 1, 128);

        int32_t maskDifferenceTolerance            = 0;
        int32_t backgroundImageDifferenceTolerance = 1;
        errCountMask += compareArrays(refMask, outImagePtr_h, imgWidth, imgHeight, maskDifferenceTolerance);
        errCountBG += compareArrays(refBGImg, outBGImagePtr_h, imgWidth, imgHeight, backgroundImageDifferenceTolerance);
    }

    /* Validate output */
    err = ValidateOutput(refMask, outImagePtr_h, refBGImg, outBGImagePtr_h, imgWidth, imgHeight, errCountMask,
                         errCountBG, "Submit to cuPVA stream");

FreeResources:
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvImageDecRef(inImageHandle, NULL);
    nvcvImageDecRef(outImageHandle, NULL);
    nvcvImageDecRef(outBGImageHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    BGSubRefDestroy(refHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);
    free(refMask);
    free(refBGImg);

    return err;
}

int main(int argc, char **argv)
{
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
        cudaStreamSubmitTestResult = SubmitBackgroundSubtractorToCudaStream(argc, argv);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult = SubmitBackgroundSubtractorToCupvaStream(argc, argv);

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
