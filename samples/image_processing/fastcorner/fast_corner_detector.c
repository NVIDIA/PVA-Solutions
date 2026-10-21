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
#include "ref/fast_corner_detector_ref.h"

#include <ImageIO.h>
#include <OpFastCorner.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Fwd.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NUM_LOC (PVA_FAST_CORNER_MAX_TOPK_FEATURES)

/**
 * @brief Allocate and initialize tensors
 */
static int AllocateAndInitializeTensors(int32_t imgWidth, int32_t imgHeight, NVCVDataType dataType,
                                        char *fileAbsPathName, NVCVAllocatorHandle *allocatorHandle,
                                        NVCVTensorHandle *inTensorHandle, NVCVTensorHandle *outLocHandle,
                                        NVCVTensorHandle *refLocHandle, NVCVTensorHandle *numLocHandle,
                                        NVCVTensorHandle *refNumLocHandle, NVCVTensorRequirements *tensorRequirementsIn)
{
    int32_t err = 0;

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);

    // Input tensor
    {
        NVCVTensorLayout layout;
        NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HW", &layout), err, Error);

        int64_t tensorShape[] = {imgHeight, imgWidth};
        const int32_t rank    = sizeof(tensorShape) / sizeof(tensorShape[0]);

        NVCV_CHECK_ERROR_GOTO(
            nvcvTensorCalcRequirementsPva(rank, tensorShape, dataType, layout, 0, 0, tensorRequirementsIn), err, Error);

        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensorRequirementsIn, *allocatorHandle, inTensorHandle), err, Error);

        int32_t ret = generateFastCornerDetectorInput(*inTensorHandle, fileAbsPathName);
        if (ret)
        {
            return ret;
        }
    }

    // Output location tensors
    {
        NVCVTensorLayout layout;
        NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("W", &layout), err, Error);

        NVCVTensorRequirements tensorRequirements;
        int64_t tensorShape[] = {MAX_NUM_LOC};
        const int32_t rank    = sizeof(tensorShape) / sizeof(tensorShape[0]);

        NVCV_CHECK_ERROR_GOTO(
            nvcvTensorCalcRequirementsPva(rank, tensorShape, NVCV_DATA_TYPE_2F32, layout, 0, 0, &tensorRequirements),
            err, Error);

        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&tensorRequirements, *allocatorHandle, outLocHandle), err, Error);
        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&tensorRequirements, *allocatorHandle, refLocHandle), err, Error);
    }

    // Number of locations tensors
    {
        NVCVTensorLayout layout;
        NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("W", &layout), err, Error);

        NVCVTensorRequirements tensorRequirements;
        int64_t tensorShape[] = {1};
        const int32_t rank    = sizeof(tensorShape) / sizeof(tensorShape[0]);

        NVCV_CHECK_ERROR_GOTO(
            nvcvTensorCalcRequirementsPva(rank, tensorShape, NVCV_DATA_TYPE_S32, layout, 0, 0, &tensorRequirements),
            err, Error);

        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&tensorRequirements, *allocatorHandle, refNumLocHandle), err, Error);
        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&tensorRequirements, *allocatorHandle, numLocHandle), err, Error);
    }

    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(NVCVTensorHandle inTensorHandle, NVCVTensorHandle refLocHandle,
                          NVCVTensorHandle refNumLocHandle, NVCVTensorHandle outLocHandle,
                          NVCVTensorHandle numLocHandle, PvaFastCornerDetectorParams *param, int32_t intensityThreshold,
                          NVCVBorderType borderMode, int32_t borderValue, const char *testName)
{
    fastCornerDetectorRef(inTensorHandle, param, intensityThreshold, borderMode, borderValue, refLocHandle,
                          refNumLocHandle);

    NVCVTensorData numLocData;
    nvcvTensorExportData(numLocHandle, &numLocData);
    int32_t *numLoc_h = NULL;
    CupvaMemGetHostPointer((void **)&numLoc_h, (void *)numLocData.buffer.strided.basePtr);

    int32_t err = checkFastCornerDetectorOutput(refLocHandle, refNumLocHandle, outLocHandle, numLocHandle);
    printf("[%-13s] %s - detected %d corners\n", testName, err ? "FAIL" : "PASS", *numLoc_h);
    return err;
}

/**
 * @brief Test fast corner detector operation using CUDA stream
 */
int SubmitFastCornerDetectorToCudaStream(int32_t imgWidth, int32_t imgHeight, NVCVDataType dataType, bool isNmsOn,
                                         int32_t intensityThreshold, char *fileAbsPathName,
                                         PvaFastCornerDetectorParams *param, NVCVBorderType borderMode,
                                         int32_t borderValue)
{
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVTensorHandle inTensorHandle     = NULL;
    NVCVTensorHandle outLocHandle       = NULL;
    NVCVTensorHandle refLocHandle       = NULL;
    NVCVTensorHandle numLocHandle       = NULL;
    NVCVTensorHandle refNumLocHandle    = NULL;

    NVCVOperatorHandle operatorHandle = NULL;
    cudaStream_t stream               = NULL;
    int32_t err                       = 0;

    NVCVTensorRequirements tensorRequirementsIn;

    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeTensors(imgWidth, imgHeight, dataType, fileAbsPathName, &allocatorHandle,
                                       &inTensorHandle, &outLocHandle, &refLocHandle, &numLocHandle, &refNumLocHandle,
                                       &tensorRequirementsIn);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(
        pvaFastCornerDetectorCreate(&operatorHandle, &tensorRequirementsIn, param, borderMode, borderValue), err,
        FreeResources);

    /* Submit to CUDA stream */
    NVCV_CHECK_ERROR_GOTO(pvaFastCornerDetectorSubmit(operatorHandle, stream, inTensorHandle, intensityThreshold,
                                                      outLocHandle, numLocHandle),
                          err, FreeResources);

    /* Synchronize stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(inTensorHandle, refLocHandle, refNumLocHandle, outLocHandle, numLocHandle, param,
                         intensityThreshold, borderMode, borderValue, "CUDA stream");

FreeResources: // Clean up all allocated resources.
    cudaStreamDestroy(stream);
    nvcvTensorDecRef(inTensorHandle, NULL);
    nvcvTensorDecRef(outLocHandle, NULL);
    nvcvTensorDecRef(refLocHandle, NULL);
    nvcvTensorDecRef(refNumLocHandle, NULL);
    nvcvTensorDecRef(numLocHandle, NULL);

    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

/**
 * @brief Test fast corner detector operation using cuPVA stream
 */
int SubmitFastCornerDetectorToCupvaStream(int32_t imgWidth, int32_t imgHeight, NVCVDataType dataType, bool isNmsOn,
                                          int32_t intensityThreshold, char *fileAbsPathName,
                                          PvaFastCornerDetectorParams *param, NVCVBorderType borderMode,
                                          int32_t borderValue)
{
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVTensorHandle inTensorHandle     = NULL;
    NVCVTensorHandle outLocHandle       = NULL;
    NVCVTensorHandle refLocHandle       = NULL;
    NVCVTensorHandle numLocHandle       = NULL;
    NVCVTensorHandle refNumLocHandle    = NULL;

    NVCVOperatorHandle operatorHandle = NULL;
    int32_t err                       = 0;
    cupvaStream_t stream              = NULL;
    cupvaSyncObj_t sync               = NULL;

    NVCVTensorRequirements tensorRequirementsIn;

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeTensors(imgWidth, imgHeight, dataType, fileAbsPathName, &allocatorHandle,
                                       &inTensorHandle, &outLocHandle, &refLocHandle, &numLocHandle, &refNumLocHandle,
                                       &tensorRequirementsIn);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(
        pvaFastCornerDetectorCreate(&operatorHandle, &tensorRequirementsIn, param, borderMode, borderValue), err,
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

    NVCV_CHECK_ERROR_GOTO(pvaFastCornerDetectorSubmit(operatorHandle, stream, inTensorHandle, intensityThreshold,
                                                      outLocHandle, numLocHandle),
                          err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    /* Wait for completion */
    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(inTensorHandle, refLocHandle, refNumLocHandle, outLocHandle, numLocHandle, param,
                         intensityThreshold, borderMode, borderValue, "cuPVA stream");

FreeResources: // Clean up all allocated resources.
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvTensorDecRef(inTensorHandle, NULL);
    nvcvTensorDecRef(outLocHandle, NULL);
    nvcvTensorDecRef(refLocHandle, NULL);
    nvcvTensorDecRef(refNumLocHandle, NULL);
    nvcvTensorDecRef(numLocHandle, NULL);

    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char **argv)
{
    /* Default parameter settings */
    const int32_t imgWidth          = 1920;
    const int32_t imgHeight         = 1080;
    const NVCVDataType dataType     = NVCV_DATA_TYPE_U8;
    int32_t nms                     = 1;
    int32_t numTopKFeatures         = 4000;
    int32_t intensityThreshold      = 5;
    const NVCVBorderType borderMode = NVCV_BORDER_REPLICATE;
    const int32_t borderValue       = 0;

    const int32_t maxPathLength = 200;
    char assetsDirectory[maxPathLength];
    char *fileAbsPathName = NULL;

    /* Parse command-line arguments */
    for (int i = 1; i < argc; i++)
    {
        if ((!strcmp(argv[i], "-a") || !strcmp(argv[i], "--assets")) && i + 1 < argc)
        {
            GetAssetsDirectory(argc, argv, assetsDirectory, maxPathLength);
            char *filePath = "/fastcorner/kodim08_grayscale_1920x1080_u16.bin";
            size_t curLen  = strlen(assetsDirectory);
            size_t pathLen = strlen(filePath);
            if (curLen + pathLen + 1 > (size_t)maxPathLength)
            {
                printf("Combined asset path exceeds buffer size of %d\n", maxPathLength);
                return -1;
            }
            memcpy(assetsDirectory + curLen, filePath, pathLen + 1);
            fileAbsPathName = assetsDirectory;

            switch (dataType)
            {
            case NVCV_DATA_TYPE_U16:
                intensityThreshold = 15428;
                break;
            case NVCV_DATA_TYPE_U8:
                intensityThreshold = 60;
                break;
            default:
                break;
            }
            i++;
        }
        else if ((!strcmp(argv[i], "--nms")) && i + 1 < argc)
        {
            nms = atoi(argv[++i]);
        }
        else if ((!strcmp(argv[i], "--topk")) && i + 1 < argc)
        {
            numTopKFeatures = atoi(argv[++i]);
        }
        else if ((!strcmp(argv[i], "--threshold") || !strcmp(argv[i], "-t")) && i + 1 < argc)
        {
            intensityThreshold = atoi(argv[++i]);
        }
    }

    PvaFastCornerDetectorParams param;
    param.circleRadius      = 3;
    param.arcLength         = 9;
    param.nonMaxSuppression = nms;
    param.numTopKFeatures   = numTopKFeatures;

    const char *offOnStr[] = {"OFF", "ON"};
    printf("\n=== FAST Corner Detector Sample ===\n");
    printf("Config: %dx%d U8 | threshold=%d | NMS=%s | TopK=%d | input=%s\n\n", imgWidth, imgHeight, intensityThreshold,
           offOnStr[param.nonMaxSuppression], param.numTopKFeatures, fileAbsPathName ? "image" : "random");

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
        cudaStreamSubmitTestResult = SubmitFastCornerDetectorToCudaStream(
            imgWidth, imgHeight, dataType, nms, intensityThreshold, fileAbsPathName, &param, borderMode, borderValue);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult = SubmitFastCornerDetectorToCupvaStream(
        imgWidth, imgHeight, dataType, nms, intensityThreshold, fileAbsPathName, &param, borderMode, borderValue);

CheckResults:
    /* Overall result */
    printf("\nResult: %s\n", (cudaStreamSubmitTestResult || cupvaStreamSubmitTestResult) ? "FAIL" : "PASS");

    return (cudaStreamSubmitTestResult || cupvaStreamSubmitTestResult);
}
