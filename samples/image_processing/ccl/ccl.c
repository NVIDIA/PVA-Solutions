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

#include "ref/ccl_ref.h"

#include <ErrorCheckMacros.h>
#include <ImageIO.h>
#include <OpCCL.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Fwd.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>
#include <stdio.h>
#include <stdlib.h>
/**
  * @brief Allocate and initialize tensors
  */
static int AllocateAndInitializeTensors(int argc, char **argv, int32_t imgWidth, int32_t imgHeight, int32_t numChannels,
                                        int32_t imgLinePitch, int32_t rowAlign, NVCVAllocatorHandle *allocatorHandle,
                                        NVCVTensorHandle *imageTensorHandle, NVCVTensorHandle *labelTensorHandle,
                                        NVCVTensorHandle *numLabelsTensorHandle, NVCVTensorHandle *refLabelTensorHandle,
                                        NVCVTensorHandle *refNumLabelsTensorHandle,
                                        NVCVTensorRequirements *imageRequirements,
                                        NVCVTensorRequirements *labelRequirements)
{
    int32_t err                = 0;
    int64_t imageTensorShape[] = {numChannels, imgHeight, imgWidth};
    NVCVTensorLayout imageTensorLayout;

    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("CHW", &imageTensorLayout), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(3, imageTensorShape, NVCV_DATA_TYPE_U8, imageTensorLayout, 0,
                                                        rowAlign, imageRequirements),
                          err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(3, imageTensorShape, NVCV_DATA_TYPE_U16, imageTensorLayout, 0,
                                                        rowAlign, labelRequirements),
                          err, Error);

    NVCVTensorRequirements numLabelsRequirements;
    int64_t numLabelsTensorShape[] = {1};
    NVCVTensorLayout numLabelsTensorLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("W", &numLabelsTensorLayout), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(1, numLabelsTensorShape, NVCV_DATA_TYPE_S32,
                                                        numLabelsTensorLayout, 0, 0, &numLabelsRequirements),
                          err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(imageRequirements, *allocatorHandle, imageTensorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(labelRequirements, *allocatorHandle, labelTensorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&numLabelsRequirements, *allocatorHandle, numLabelsTensorHandle), err,
                          Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(labelRequirements, *allocatorHandle, refLabelTensorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&numLabelsRequirements, *allocatorHandle, refNumLabelsTensorHandle), err,
                          Error);

    generateRandomCCLInput(*imageTensorHandle, imgWidth, imgHeight, imgLinePitch);
    return 0;

Error:
    return err;
}

/**
  * @brief Validate output against reference
  */
static int ValidateOutput(NVCVTensorHandle imageHdl, NVCVTensorHandle refLabelHdl, NVCVTensorHandle pvaLabelHdl,
                          NVCVTensorHandle refNumLabelsHdl, NVCVTensorHandle pvaNumLabelsHdl,
                          PVAConnectivityType connectivity, const uint8_t threshold, const char *testName)
{
    CCLRef(imageHdl, refLabelHdl, refNumLabelsHdl, connectivity, threshold);
    int32_t err = checkCCLOutput(refLabelHdl, pvaLabelHdl, refNumLabelsHdl, pvaNumLabelsHdl);
    if (err)
    {
        printf("%s test failed\n", testName);
    }
    return err;
}

/**
  * @brief Test CCL operation using CUDA stream
  */
int SubmitCCLToCudaStream(int argc, char **argv, int32_t imgWidth, int32_t imgHeight, int32_t numChannels,
                          int32_t imgLinePitch, int32_t rowAlign, PVAConnectivityType connectivity,
                          const uint8_t threshold)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
      * This is done because calling the corresponding free functions with NULL is a no-op.
      */
    NVCVAllocatorHandle allocatorHandle       = NULL;
    NVCVTensorHandle imageTensorHandle        = NULL;
    NVCVTensorHandle labelTensorHandle        = NULL;
    NVCVTensorHandle numLabelsTensorHandle    = NULL;
    NVCVTensorHandle refLabelTensorHandle     = NULL;
    NVCVTensorHandle refNumLabelsTensorHandle = NULL;
    NVCVOperatorHandle operatorHandle         = NULL;
    cudaStream_t stream                       = NULL;
    int32_t err                               = 0;

    NVCVTensorRequirements imageRequirements;
    NVCVTensorRequirements labelRequirements;

    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeTensors(argc, argv, imgWidth, imgHeight, numChannels, imgLinePitch, rowAlign,
                                       &allocatorHandle, &imageTensorHandle, &labelTensorHandle, &numLabelsTensorHandle,
                                       &refLabelTensorHandle, &refNumLabelsTensorHandle, &imageRequirements,
                                       &labelRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(
        pvaCCLCreate(&operatorHandle, &imageRequirements, &labelRequirements, connectivity, threshold), err,
        FreeResources);

    /* Submit to CUDA stream */
    NVCV_CHECK_ERROR_GOTO(
        pvaCCLSubmit(operatorHandle, stream, imageTensorHandle, labelTensorHandle, numLabelsTensorHandle), err,
        FreeResources);
    /* Synchronize stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(imageTensorHandle, refLabelTensorHandle, labelTensorHandle, refNumLabelsTensorHandle,
                         numLabelsTensorHandle, connectivity, threshold, "Submit to CUDA stream");

FreeResources: /* clean up all allocated resources */
    cudaStreamDestroy(stream);
    nvcvTensorDecRef(imageTensorHandle, NULL);
    nvcvTensorDecRef(labelTensorHandle, NULL);
    nvcvTensorDecRef(numLabelsTensorHandle, NULL);
    nvcvTensorDecRef(refLabelTensorHandle, NULL);
    nvcvTensorDecRef(refNumLabelsTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

/**
  * @brief Test CCL operation using cuPVA stream
  */
int SubmitCCLToCupvaStream(int argc, char **argv, int32_t imgWidth, int32_t imgHeight, int32_t numChannels,
                           int32_t imgLinePitch, int32_t rowAlign, PVAConnectivityType connectivity,
                           const uint8_t threshold)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
      * This is done because calling the corresponding free functions with NULL is a no-op.
      */
    NVCVAllocatorHandle allocatorHandle       = NULL;
    NVCVTensorHandle imageTensorHandle        = NULL;
    NVCVTensorHandle labelTensorHandle        = NULL;
    NVCVTensorHandle numLabelsTensorHandle    = NULL;
    NVCVTensorHandle refLabelTensorHandle     = NULL;
    NVCVTensorHandle refNumLabelsTensorHandle = NULL;
    NVCVOperatorHandle operatorHandle         = NULL;
    cupvaSyncObj_t sync                       = NULL;
    cupvaStream_t stream                      = NULL;
    int32_t err                               = 0;

    cupvaError_t statusCode = CUPVA_ERROR_NONE;

    NVCVTensorRequirements imageRequirements;
    NVCVTensorRequirements labelRequirements;

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeTensors(argc, argv, imgWidth, imgHeight, numChannels, imgLinePitch, rowAlign,
                                       &allocatorHandle, &imageTensorHandle, &labelTensorHandle, &numLabelsTensorHandle,
                                       &refLabelTensorHandle, &refNumLabelsTensorHandle, &imageRequirements,
                                       &labelRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(
        pvaCCLCreate(&operatorHandle, &imageRequirements, &labelRequirements, connectivity, threshold), err,
        FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaSyncObjCreate(&sync, false, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD), err,
                           FreeResources);

    cupvaFence_t fence;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceInit(&fence, sync), err, FreeResources);

    cupvaCmd_t rf;
    CUPVA_CHECK_ERROR_GOTO(CupvaCmdRequestFencesInit(&rf, &fence, 1), err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamCreate(&stream, CUPVA_PVA0, CUPVA_VPU_ANY), err, FreeResources);

    /* Submit to cuPVA stream */
    NVCV_CHECK_ERROR_GOTO(
        pvaCCLSubmit(operatorHandle, stream, imageTensorHandle, labelTensorHandle, numLabelsTensorHandle), err,
        FreeResources);

    cupvaCmdStatus_t status[1] = {NULL};
    cupvaCmd_t const *cmds[1]  = {&rf};
    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(imageTensorHandle, refLabelTensorHandle, labelTensorHandle, refNumLabelsTensorHandle,
                         numLabelsTensorHandle, connectivity, threshold, "Submit to cuPVA stream");

FreeResources: /* clean up all allocated resources */
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvTensorDecRef(imageTensorHandle, NULL);
    nvcvTensorDecRef(labelTensorHandle, NULL);
    nvcvTensorDecRef(numLabelsTensorHandle, NULL);
    nvcvTensorDecRef(refLabelTensorHandle, NULL);
    nvcvTensorDecRef(refNumLabelsTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char **argv)
{
    /* Parameter settings */
    const int32_t imgWidth     = 699;
    const int32_t imgHeight    = 500;
    const int32_t numChannels  = 1;
    const int32_t imgLinePitch = 699;
    const int32_t rowAlign     = 0;

    const uint8_t threshold                = 100;
    const PVAConnectivityType connectivity = PVA_8_CONNECTIVITY;

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
        cudaStreamSubmitTestResult = SubmitCCLToCudaStream(argc, argv, imgWidth, imgHeight, numChannels, imgLinePitch,
                                                           rowAlign, connectivity, threshold);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult = SubmitCCLToCupvaStream(argc, argv, imgWidth, imgHeight, numChannels, imgLinePitch,
                                                         rowAlign, connectivity, threshold);

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
