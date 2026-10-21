/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "ref/hog_ref.h"

#include <ImageIO.h>
#include <OpHOG.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Fwd.h>
#include <nvcv/Image.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>

// Hardcoded filenames for input
#define DEFAULT_INPUT_FILE "hog/hog_input_rgb8_planar_out8.raw"

/**
  * @brief Allocate and initialize tensors
  */
static int AllocateAndInitialize(int argc, char **argv, int32_t imageWidth, int32_t imageHeight,
                                 NVCVImageFormat imageFormat, int32_t cellSize, int32_t orientationsNum,
                                 int32_t tensorRank, int32_t rowAlignment, bool useInputFile,
                                 NVCVAllocatorHandle *allocatorHandle, NVCVImageHandle *inImageHandle,
                                 NVCVTensorHandle *outTensorHandle, NVCVTensorHandle *refTensorHandle,
                                 NVCVImageRequirements *imageRequirements,
                                 NVCVTensorRequirements *outTensorRequirements,
                                 NVCVTensorRequirements *refTensorRequirements)
{
    int32_t err                           = 0;
    char assetsDirectory[MAX_PATH_LENGTH] = {0};

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvImageCalcRequirementsPva(imageWidth, imageHeight, imageFormat, 0, 0, imageRequirements),
                          err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(imageRequirements, *allocatorHandle, inImageHandle), err, Error);

    NVCVTensorLayout outTensorLayout;
    NVCVTensorLayout refTensorLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("CHW", &outTensorLayout), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HWC", &refTensorLayout), err, Error);

    int32_t tensorWidth      = (imageWidth + cellSize - 1) / cellSize;
    int32_t tensorHeight     = (imageHeight + cellSize - 1) / cellSize;
    int64_t outTensorShape[] = {orientationsNum, tensorHeight, tensorWidth};
    int64_t refTensorShape[] = {tensorHeight, tensorWidth, orientationsNum};

    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank, outTensorShape, NVCV_DATA_TYPE_U8, outTensorLayout,
                                                        0, rowAlignment, outTensorRequirements),
                          err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank, refTensorShape, NVCV_DATA_TYPE_U8, refTensorLayout,
                                                        0, rowAlignment, refTensorRequirements),
                          err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(outTensorRequirements, *allocatorHandle, outTensorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(refTensorRequirements, *allocatorHandle, refTensorHandle), err, Error);

    // Load input from file if assets directory specified, otherwise generate random input
    if (useInputFile)
    {
        int assetsErr = GetAssetsDirectory(argc, argv, assetsDirectory, MAX_PATH_LENGTH);
        if (assetsErr)
        {
            printf("Assets directory could not be found\n");
            return assetsErr;
        }

        loadHOGDataFromFile(*inImageHandle, assetsDirectory, DEFAULT_INPUT_FILE);
    }
    else
    {
        generateRandomHOGInput(*inImageHandle);
    }

    return 0;

Error:
    return err;
}

/**
  * @brief Validate output against reference
  */
static int ValidateOutput(NVCVTensorHandle refTensorHandle, NVCVTensorHandle outTensorHandle, const char *testName)
{
    int32_t err = 0;

    err = checkHOGOutput(refTensorHandle, outTensorHandle);
    if (err)
    {
        printf("%s test failed\n", testName);
    }
    return err;
}

/**
  * @brief Test HOG operation using CUDA stream
  */
int SubmitHOGToCudaStream(int argc, char **argv, int32_t imageWidth, int32_t imageHeight, NVCVImageFormat imageFormat,
                          int32_t cellSize, int32_t orientationsNum, int32_t tensorRank, int32_t rowAlignment,
                          bool useInputFile)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
      * This is done because calling the corresponding free functions with NULL is a no-op.
      */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVImageHandle inImageHandle       = NULL;
    NVCVTensorHandle outTensorHandle    = NULL;
    NVCVTensorHandle refTensorHandle    = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cudaStream_t stream                 = NULL;
    int32_t err                         = 0;
    NVCVImageRequirements imageRequirements;
    NVCVTensorRequirements outTensorRequirements;
    NVCVTensorRequirements refTensorRequirements;

    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    /* Allocate and initialize images */
    err = AllocateAndInitialize(argc, argv, imageWidth, imageHeight, imageFormat, cellSize, orientationsNum, tensorRank,
                                rowAlignment, useInputFile, &allocatorHandle, &inImageHandle, &outTensorHandle,
                                &refTensorHandle, &imageRequirements, &outTensorRequirements, &refTensorRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaHOGCreate(&operatorHandle, &imageRequirements, &outTensorRequirements), err,
                          FreeResources);

    /* Submit to CUDA stream */
    NVCV_CHECK_ERROR_GOTO(pvaHOGSubmit(operatorHandle, stream, inImageHandle, outTensorHandle), err, FreeResources);

    /* Synchronize stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    /* Compute reference */
    hogRef(inImageHandle, refTensorHandle);

    /* Validate output */
    err = ValidateOutput(refTensorHandle, outTensorHandle, "Submit to CUDA stream");

FreeResources: /* clean up all allocated resources */
    cudaStreamDestroy(stream);
    nvcvImageDecRef(inImageHandle, NULL);
    nvcvTensorDecRef(outTensorHandle, NULL);
    nvcvTensorDecRef(refTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

/**
  * @brief Test HOG operation using cuPVA stream
  */
int SubmitHOGToCupvaStream(int argc, char **argv, int32_t imageWidth, int32_t imageHeight, NVCVImageFormat imageFormat,
                           int32_t cellSize, int32_t orientationsNum, int32_t tensorRank, int32_t rowAlignment,
                           bool useInputFile)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
      * This is done because calling the corresponding free functions with NULL is a no-op.
      */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVImageHandle inImageHandle       = NULL;
    NVCVTensorHandle outTensorHandle    = NULL;
    NVCVTensorHandle refTensorHandle    = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cupvaSyncObj_t sync                 = NULL;
    cupvaStream_t stream                = NULL;
    int32_t err                         = 0;

    NVCVImageRequirements imageRequirements;
    NVCVTensorRequirements outTensorRequirements;
    NVCVTensorRequirements refTensorRequirements;

    /* Allocate and initialize tensors */
    err = AllocateAndInitialize(argc, argv, imageWidth, imageHeight, imageFormat, cellSize, orientationsNum, tensorRank,
                                rowAlignment, useInputFile, &allocatorHandle, &inImageHandle, &outTensorHandle,
                                &refTensorHandle, &imageRequirements, &outTensorRequirements, &refTensorRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaHOGCreate(&operatorHandle, &imageRequirements, &outTensorRequirements), err,
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

    NVCV_CHECK_ERROR_GOTO(pvaHOGSubmit(operatorHandle, stream, inImageHandle, outTensorHandle), err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    /* Wait for completion */
    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    /* Compute reference */
    hogRef(inImageHandle, refTensorHandle);

    /* Validate output */
    err = ValidateOutput(refTensorHandle, outTensorHandle, "Submit to cuPVA stream");

FreeResources: /* clean up all allocated resources */
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvImageDecRef(inImageHandle, NULL);
    nvcvTensorDecRef(outTensorHandle, NULL);
    nvcvTensorDecRef(refTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char **argv)
{
    /* Parameter settings */
    const int32_t imageWidth      = 576;
    const int32_t imageHeight     = 576;
    const int32_t cellSize        = 4;
    const int32_t orientationsNum = 18;
    const int32_t tensorRank      = 3;
    const int32_t rowAlignment    = 0;

    NVCVImageFormat imageFormat = NVCV_IMAGE_FORMAT_RGB8p;

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
        cudaStreamSubmitTestResult = SubmitHOGToCudaStream(argc, argv, imageWidth, imageHeight, imageFormat, cellSize,
                                                           orientationsNum, tensorRank, rowAlignment, useInputFile);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult = SubmitHOGToCupvaStream(argc, argv, imageWidth, imageHeight, imageFormat, cellSize,
                                                         orientationsNum, tensorRank, rowAlignment, useInputFile);

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
