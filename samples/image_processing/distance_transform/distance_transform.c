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
#include "PvaOperatorTypes.h"
#include "ref/distance_transform_ref.h"

#include <ImageIO.h>
#include <OpDistanceTransform.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Fwd.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>
#include <stdlib.h>

/**
 * @brief Allocate and initialize tensors
 */
static int AllocateAndInitializeTensors(int argc, char **argv, int32_t imgWidth, int32_t imgHeight,
                                        int32_t channelCount, int32_t tensorRank, NVCVAllocatorHandle *allocatorHandle,
                                        NVCVTensorHandle *imgTensorHandle, NVCVTensorHandle *distTensorHandle,
                                        NVCVTensorHandle *labelTensorHandle, NVCVTensorRequirements *tensorRequirements,
                                        char *assetsDirectory, int32_t maxPathLength)
{
    int32_t err = 0;
    NVCVTensorLayout tensorLayout;

    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HWC", &tensorLayout), err, Error);

    int64_t tensorShape[] = {imgHeight, imgWidth, channelCount};
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank, tensorShape, NVCV_DATA_TYPE_U16, tensorLayout, 0, 0,
                                                        tensorRequirements),
                          err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensorRequirements, *allocatorHandle, imgTensorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensorRequirements, *allocatorHandle, distTensorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensorRequirements, *allocatorHandle, labelTensorHandle), err, Error);

    err = GetAssetsDirectory(argc, argv, assetsDirectory, maxPathLength);
    if (err)
    {
        printf("Distance transform input image could not be loaded\n");
        return err;
    }
    err = loadDistanceTransformData(*imgTensorHandle, assetsDirectory, "/distance_transform/sample_image_640x640.bin",
                                    imgWidth * imgHeight * sizeof(uint16_t));
    if (err)
    {
        printf("Distance transform input image could not be loaded\n");
        return err;
    }

    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(NVCVTensorHandle imgTensorHandle, NVCVTensorHandle distTensorHandle,
                          NVCVTensorHandle labelTensorHandle, NVCVTensorHandle distRefTensorHandle,
                          NVCVTensorHandle labelRefTensorHandle, int32_t imgWidth, int32_t imgHeight, uint16_t offPixel,
                          uint16_t maxDistance, const char *testName)
{
    int32_t err = getDistanceTransformRefData(imgTensorHandle, distRefTensorHandle, labelRefTensorHandle, imgWidth,
                                              imgHeight, offPixel, maxDistance);
    if (err)
    {
        printf("%s test failed: Distance transform reference data could not be generated\n", testName);
        return err;
    }

    err = checkDistanceTransformResult(distTensorHandle, distRefTensorHandle, imgWidth * imgHeight * sizeof(uint16_t));
    if (err)
    {
        printf("%s test failed: Distance transform result could not be checked\n", testName);
        return err;
    }
    err =
        checkDistanceTransformResult(labelTensorHandle, labelRefTensorHandle, imgWidth * imgHeight * sizeof(uint16_t));

    if (err)
    {
        printf("%s test failed\n", testName);
    }
    return err;
}

/**
 * @brief Test distance transform operation using CUDA stream
 */
int SubmitDistanceTransformToCudaStream(int argc, char **argv, int32_t imgWidth, int32_t imgHeight,
                                        int32_t channelCount, int32_t tensorRank, int32_t qbits, uint16_t maxDistance,
                                        uint16_t offPixel)
{
    const int32_t maxPathLength = 200;
    char assetsDirectory[maxPathLength];

    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle   = NULL;
    NVCVTensorHandle imgTensorHandle      = NULL;
    NVCVTensorHandle distTensorHandle     = NULL;
    NVCVTensorHandle labelTensorHandle    = NULL;
    NVCVTensorHandle distRefTensorHandle  = NULL;
    NVCVTensorHandle labelRefTensorHandle = NULL;
    NVCVOperatorHandle operatorHandle     = NULL;
    cudaStream_t stream                   = NULL;
    int32_t err                           = 0;

    NVCVTensorRequirements tensorRequirements;

    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeTensors(argc, argv, imgWidth, imgHeight, channelCount, tensorRank, &allocatorHandle,
                                       &imgTensorHandle, &distTensorHandle, &labelTensorHandle, &tensorRequirements,
                                       assetsDirectory, maxPathLength);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaDistanceTransformCreate(&operatorHandle, &tensorRequirements, PVA_DIST_L1), err,
                          FreeResources);

    /* Submit to CUDA stream */
    NVCV_CHECK_ERROR_GOTO(pvaDistanceTransformSubmit(operatorHandle, stream, imgTensorHandle, maxDistance,
                                                     distTensorHandle, labelTensorHandle),
                          err, FreeResources);

    /* Synchronize stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    /* Allocate reference tensors after operator execution */
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&tensorRequirements, allocatorHandle, &distRefTensorHandle), err,
                          FreeResources);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&tensorRequirements, allocatorHandle, &labelRefTensorHandle), err,
                          FreeResources);

    /* Validate output */
    err = ValidateOutput(imgTensorHandle, distTensorHandle, labelTensorHandle, distRefTensorHandle,
                         labelRefTensorHandle, imgWidth, imgHeight, offPixel, maxDistance, "Submit to CUDA stream");

FreeResources: /* clean up all allocated resources */
    cudaStreamDestroy(stream);
    nvcvTensorDecRef(imgTensorHandle, NULL);
    nvcvTensorDecRef(distTensorHandle, NULL);
    nvcvTensorDecRef(labelTensorHandle, NULL);
    nvcvTensorDecRef(distRefTensorHandle, NULL);
    nvcvTensorDecRef(labelRefTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

/**
 * @brief Test distance transform operation using cuPVA stream
 */
int SubmitDistanceTransformToCupvaStream(int argc, char **argv, int32_t imgWidth, int32_t imgHeight,
                                         int32_t channelCount, int32_t tensorRank, int32_t qbits, uint16_t maxDistance,
                                         uint16_t offPixel)
{
    const int32_t maxPathLength = 200;
    char assetsDirectory[maxPathLength];

    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle   = NULL;
    NVCVTensorHandle imgTensorHandle      = NULL;
    NVCVTensorHandle distTensorHandle     = NULL;
    NVCVTensorHandle labelTensorHandle    = NULL;
    NVCVTensorHandle distRefTensorHandle  = NULL;
    NVCVTensorHandle labelRefTensorHandle = NULL;
    NVCVOperatorHandle operatorHandle     = NULL;
    cupvaSyncObj_t sync                   = NULL;
    cupvaStream_t stream                  = NULL;
    int32_t err                           = 0;

    NVCVTensorLayout tensorLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HWC", &tensorLayout), err, FreeResources);

    int64_t tensorShape[] = {imgHeight, imgWidth, channelCount};
    NVCVTensorRequirements tensorRequirements;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank, tensorShape, NVCV_DATA_TYPE_U16, tensorLayout, 0, 0,
                                                        &tensorRequirements),
                          err, FreeResources);

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(&allocatorHandle), err, FreeResources);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&tensorRequirements, allocatorHandle, &imgTensorHandle), err,
                          FreeResources);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&tensorRequirements, allocatorHandle, &distTensorHandle), err,
                          FreeResources);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&tensorRequirements, allocatorHandle, &labelTensorHandle), err,
                          FreeResources);

    NVCV_CHECK_ERROR_GOTO(pvaDistanceTransformCreate(&operatorHandle, &tensorRequirements, PVA_DIST_L1), err,
                          FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaSyncObjCreate(&sync, false, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD), err,
                           FreeResources);

    cupvaFence_t fence;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceInit(&fence, sync), err, FreeResources);

    cupvaCmd_t rf;
    CUPVA_CHECK_ERROR_GOTO(CupvaCmdRequestFencesInit(&rf, &fence, 1), err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamCreate(&stream, CUPVA_PVA0, CUPVA_VPU_ANY), err, FreeResources);

    cupvaCmd_t const *cmds[1]  = {&rf};
    cupvaCmdStatus_t status[1] = {NULL};

    err = GetAssetsDirectory(argc, argv, assetsDirectory, maxPathLength);
    if (err)
    {
        printf("Submit to cuPVA stream test failed: Distance transform input image could not be loaded\n");
        goto FreeResources;
    }

    err = loadDistanceTransformData(imgTensorHandle, assetsDirectory, "/distance_transform/sample_image_640x640.bin",
                                    imgWidth * imgHeight * sizeof(uint16_t));
    if (err)
    {
        printf("Submit to cuPVA stream test failed: Distance transform input image could not be loaded\n");
        goto FreeResources;
    }

    NVCV_CHECK_ERROR_GOTO(pvaDistanceTransformSubmit(operatorHandle, stream, imgTensorHandle, maxDistance,
                                                     distTensorHandle, labelTensorHandle),
                          err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    /* Allocate reference tensors after operator execution */
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&tensorRequirements, allocatorHandle, &distRefTensorHandle), err,
                          FreeResources);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&tensorRequirements, allocatorHandle, &labelRefTensorHandle), err,
                          FreeResources);

    /* Validate output */
    err = ValidateOutput(imgTensorHandle, distTensorHandle, labelTensorHandle, distRefTensorHandle,
                         labelRefTensorHandle, imgWidth, imgHeight, offPixel, maxDistance, "Submit to cuPVA stream");

FreeResources: /* clean up all allocated resources */
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvTensorDecRef(imgTensorHandle, NULL);
    nvcvTensorDecRef(distTensorHandle, NULL);
    nvcvTensorDecRef(labelTensorHandle, NULL);
    nvcvTensorDecRef(distRefTensorHandle, NULL);
    nvcvTensorDecRef(labelRefTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char **argv)
{
    /* Parameter settings */
    const int32_t imgWidth     = 640;
    const int32_t imgHeight    = 640;
    const int32_t channelCount = 1;
    const int32_t tensorRank   = 3;
    const int32_t qbits        = 3;
    const uint16_t maxDistance = (imgHeight + imgWidth) << qbits;
    const uint16_t offPixel    = 0xffff;

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
        cudaStreamSubmitTestResult = SubmitDistanceTransformToCudaStream(argc, argv, imgWidth, imgHeight, channelCount,
                                                                         tensorRank, qbits, maxDistance, offPixel);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult = SubmitDistanceTransformToCupvaStream(argc, argv, imgWidth, imgHeight, channelCount,
                                                                       tensorRank, qbits, maxDistance, offPixel);

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
