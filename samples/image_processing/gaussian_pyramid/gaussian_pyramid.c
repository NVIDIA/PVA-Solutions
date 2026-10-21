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
#include "ref/gaussian_pyramid_ref.h"

#include <ImageIO.h>
#include <OpGaussianPyramid.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Fwd.h>
#include <nvcv/Image.h>
#include <nvcv/alloc/Allocator.h>
#include <stdio.h>

#define DEFAULT_INPUT_FILE "cornersubpix/kodim08.bin"

/**
 * @brief Allocate and initialize images
 */
static int AllocateAndInitializeImages(int argc, char **argv, int32_t imageWidth, int32_t imageHeight,
                                       int32_t numLevels, NVCVImageFormat imageFormat, bool useInputFile,
                                       NVCVAllocatorHandle *allocatorHandle, NVCVImageHandle *srcImageHandle,
                                       NVCVImageHandle *dstImageHandles, NVCVImageHandle *refImageHandles,
                                       NVCVImageRequirements *srcImageRequirements)
{
    int32_t err                           = 0;
    char assetsDirectory[MAX_PATH_LENGTH] = {0};

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(
        nvcvImageCalcRequirementsPva(imageWidth, imageHeight, imageFormat, 0, 0, srcImageRequirements), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(srcImageRequirements, *allocatorHandle, srcImageHandle), err, Error);

    int32_t dstImageWidth  = imageWidth;
    int32_t dstImageHeight = imageHeight;
    for (int32_t i = 0; i < numLevels; ++i)
    {
        NVCVImageRequirements dstImageRequirements;
        NVCV_CHECK_ERROR_GOTO(
            nvcvImageCalcRequirementsPva(dstImageWidth, dstImageHeight, imageFormat, 0, 0, &dstImageRequirements), err,
            Error);

        NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(&dstImageRequirements, *allocatorHandle, &dstImageHandles[i]), err,
                              Error);
        NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(&dstImageRequirements, *allocatorHandle, &refImageHandles[i]), err,
                              Error);

        dstImageWidth  = (dstImageWidth + 1) / 2;
        dstImageHeight = (dstImageHeight + 1) / 2;
    }

    // Load input from file if assets directory specified, otherwise generate random input
    if (useInputFile)
    {
        int assetsErr = GetAssetsDirectory(argc, argv, assetsDirectory, MAX_PATH_LENGTH);
        if (assetsErr)
        {
            printf("Assets directory could not be found\n");
            return assetsErr;
        }

        err = loadGaussianPyramidInput(*srcImageHandle, assetsDirectory, DEFAULT_INPUT_FILE, imageWidth * imageHeight);
        if (err)
        {
            printf("Input files could not be loaded\n");
            return err;
        }
    }
    else
    {
        generateRandomGaussianPyramidInput(*srcImageHandle);
    }

    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(NVCVImageHandle srcImageHandle, NVCVImageHandle *refImageHandles,
                          NVCVImageHandle *dstImageHandles, int32_t numLevels, NVCVBorderType borderType,
                          int32_t borderValue, bool useInputFile, int32_t imageWidth, int32_t imageHeight,
                          const char *testName)
{
    int32_t err = 0;

    gaussianPyramidRef(srcImageHandle, refImageHandles, numLevels, borderType, borderValue);

    // Write output to file if assets directory was specified
    if (useInputFile)
    {
        int32_t dstImageWidth  = imageWidth;
        int32_t dstImageHeight = imageHeight;
        for (int32_t i = 0; i < numLevels; ++i)
        {
            char filename[256];
            snprintf(filename, sizeof(filename), "kodim08_gaussian_pyramid_level%d.bin", i);
            err = saveGaussianPyramidOutput(dstImageHandles[i], ".", filename, dstImageWidth * dstImageHeight);
            if (err)
            {
                printf("%s test failed: Output files could not be saved\n", testName);
                return err;
            }
            dstImageWidth  = (dstImageWidth + 1) / 2;
            dstImageHeight = (dstImageHeight + 1) / 2;
        }
    }

    err = checkGaussianPyramidOutput(refImageHandles, dstImageHandles, numLevels);
    if (err)
    {
        printf("%s test failed\n", testName);
    }
    return err;
}

/**
 * @brief Test gaussian pyramid operation using CUDA stream
 */
int SubmitGaussianPyramidToCudaStream(int argc, char **argv, int32_t imageWidth, int32_t imageHeight, int32_t numLevels,
                                      NVCVBorderType borderType, int32_t borderValue, NVCVImageFormat imageFormat,
                                      bool useInputFile)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVImageHandle srcImageHandle      = NULL;
    NVCVImageHandle dstImageHandles[numLevels];
    NVCVImageHandle refImageHandles[numLevels];
    for (int32_t i = 0; i < numLevels; i++)
    {
        dstImageHandles[i] = NULL;
        refImageHandles[i] = NULL;
    }

    NVCVOperatorHandle operatorHandle = NULL;
    cudaStream_t stream               = NULL;
    int32_t err                       = 0;

    NVCVImageRequirements srcImageRequirements;

    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    /* Allocate and initialize images */
    err = AllocateAndInitializeImages(argc, argv, imageWidth, imageHeight, numLevels, imageFormat, useInputFile,
                                      &allocatorHandle, &srcImageHandle, dstImageHandles, refImageHandles,
                                      &srcImageRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(
        pvaGaussianPyramidCreate(&operatorHandle, &srcImageRequirements, numLevels, borderType, borderValue), err,
        FreeResources);

    /* Submit to CUDA stream */
    NVCV_CHECK_ERROR_GOTO(pvaGaussianPyramidSubmit(operatorHandle, stream, srcImageHandle, dstImageHandles), err,
                          FreeResources);

    /* Synchronize stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(srcImageHandle, refImageHandles, dstImageHandles, numLevels, borderType, borderValue,
                         useInputFile, imageWidth, imageHeight, "Submit to CUDA stream");

FreeResources: /* clean up all allocated resources */
    cudaStreamDestroy(stream);
    nvcvImageDecRef(srcImageHandle, NULL);
    for (int32_t i = 0; i < numLevels; ++i)
    {
        nvcvImageDecRef(dstImageHandles[i], NULL);
        nvcvImageDecRef(refImageHandles[i], NULL);
    }
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

/**
 * @brief Test gaussian pyramid operation using cuPVA stream
 */
int SubmitGaussianPyramidToCupvaStream(int argc, char **argv, int32_t imageWidth, int32_t imageHeight,
                                       int32_t numLevels, NVCVBorderType borderType, int32_t borderValue,
                                       NVCVImageFormat imageFormat, bool useInputFile)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVImageHandle srcImageHandle      = NULL;
    NVCVImageHandle dstImageHandles[numLevels];
    NVCVImageHandle refImageHandles[numLevels];
    for (int32_t i = 0; i < numLevels; i++)
    {
        dstImageHandles[i] = NULL;
        refImageHandles[i] = NULL;
    }

    NVCVOperatorHandle operatorHandle = NULL;
    cupvaSyncObj_t sync               = NULL;
    cupvaStream_t stream              = NULL;
    int32_t err                       = 0;

    NVCVImageRequirements srcImageRequirements;

    /* Allocate and initialize images */
    err = AllocateAndInitializeImages(argc, argv, imageWidth, imageHeight, numLevels, imageFormat, useInputFile,
                                      &allocatorHandle, &srcImageHandle, dstImageHandles, refImageHandles,
                                      &srcImageRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(
        pvaGaussianPyramidCreate(&operatorHandle, &srcImageRequirements, numLevels, borderType, borderValue), err,
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

    NVCV_CHECK_ERROR_GOTO(pvaGaussianPyramidSubmit(operatorHandle, stream, srcImageHandle, dstImageHandles), err,
                          FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    /* Wait for completion */
    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(srcImageHandle, refImageHandles, dstImageHandles, numLevels, borderType, borderValue,
                         useInputFile, imageWidth, imageHeight, "Submit to cuPVA stream");

FreeResources: /* clean up all allocated resources */
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvImageDecRef(srcImageHandle, NULL);
    for (int32_t i = 0; i < numLevels; ++i)
    {
        nvcvImageDecRef(dstImageHandles[i], NULL);
        nvcvImageDecRef(refImageHandles[i], NULL);
    }
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char **argv)
{
    /* Parameter settings */
    const int32_t imageWidth        = 1920;
    const int32_t imageHeight       = 1080;
    const int32_t numLevels         = 3;
    const NVCVBorderType borderType = NVCV_BORDER_CONSTANT;
    const int32_t borderValue       = 0;
    NVCVImageFormat imageFormat     = NVCV_IMAGE_FORMAT_U16;

    // Flag to indicate if we're using an input file
    bool useInputFile = (argc > 1); // Use input file if any command line argument was provided

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
        cudaStreamSubmitTestResult = SubmitGaussianPyramidToCudaStream(
            argc, argv, imageWidth, imageHeight, numLevels, borderType, borderValue, imageFormat, useInputFile);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult = SubmitGaussianPyramidToCupvaStream(
        argc, argv, imageWidth, imageHeight, numLevels, borderType, borderValue, imageFormat, useInputFile);

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
