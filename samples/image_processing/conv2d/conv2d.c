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
#include "ref/conv2d_ref.h"

#include <ImageIO.h>
#include <OpConv2d.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>

// Hardcoded filenames for input and output
#define DEFAULT_INPUT_FILE "cornersubpix/kodim08.bin"
#define DEFAULT_OUTPUT_FILE "filtered_kodim08.bin"

/**
 * @brief Allocate and initialize tensors
 */
static int AllocateAndInitializeTensors(int argc, char **argv, int32_t imgWidth, int32_t imgHeight, int32_t kernelSize,
                                        bool useInputFile, NVCVAllocatorHandle *allocatorHandle,
                                        NVCVTensorHandle *inTensorHandle, NVCVTensorHandle *outTensorHandle,
                                        NVCVTensorHandle *refTensorHandle, NVCVTensorHandle *kernelTensorHandle,
                                        NVCVTensorRequirements *tensorRequirements)
{
    int32_t err                           = 0;
    const int32_t imgChannel              = 1;
    const int32_t tensorRank              = 3;
    char assetsDirectory[MAX_PATH_LENGTH] = {0};
    NVCVTensorLayout tensorLayout;

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HWC", &tensorLayout), err, Error);

    int64_t tensorShape[] = {imgHeight, imgWidth, imgChannel};
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank, tensorShape, NVCV_DATA_TYPE_U8, tensorLayout, 0, 0,
                                                        tensorRequirements),
                          err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensorRequirements, *allocatorHandle, inTensorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensorRequirements, *allocatorHandle, outTensorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensorRequirements, *allocatorHandle, refTensorHandle), err, Error);

    // Load input from file if assets directory specified, otherwise generate random input
    if (useInputFile)
    {
        int assetsErr = GetAssetsDirectory(argc, argv, assetsDirectory, MAX_PATH_LENGTH);
        if (assetsErr)
        {
            printf("Assets directory could not be found\n");
            return assetsErr;
        }

        err = loadConv2dInput(*inTensorHandle, assetsDirectory, DEFAULT_INPUT_FILE, imgWidth * imgHeight);
        if (err)
        {
            printf("Input files could not be loaded\n");
            return err;
        }
    }
    else
    {
        generateRandomConv2dInput(*inTensorHandle);
    }

    // Create kernel tensor with coefficients
    *kernelTensorHandle = createConv2dKernel(*allocatorHandle, kernelSize, NVCV_DATA_TYPE_U8);
    if (*kernelTensorHandle == NULL)
    {
        printf("Failed to create kernel tensor\n");
        return -1;
    }

    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(NVCVTensorHandle inTensorHandle, NVCVTensorHandle refTensorHandle,
                          NVCVTensorHandle outTensorHandle, NVCVTensorHandle kernelTensorHandle,
                          NVCVBorderType borderMode, int32_t borderValue, bool useInputFile, int32_t imgWidth,
                          int32_t imgHeight, const char *testName)
{
    int32_t err = 0;

    conv2dRefWithKernel(inTensorHandle, refTensorHandle, kernelTensorHandle, borderMode, borderValue);

    // Write output to file if assets directory was specified
    if (useInputFile)
    {
        err = saveConv2dOutput(outTensorHandle, ".", DEFAULT_OUTPUT_FILE, imgWidth * imgHeight);
        if (err)
        {
            printf("%s test failed: Output files could not be saved\n", testName);
            return err;
        }
    }

    err = checkConv2dOutput(refTensorHandle, outTensorHandle);
    if (err)
    {
        printf("%s test failed\n", testName);
    }
    return err;
}

/**
 * @brief Test conv2d operation using CUDA stream
 */
int SubmitConv2dToCudaStream(int argc, char **argv, int32_t imgWidth, int32_t imgHeight, int32_t kernelSize,
                             NVCVBorderType borderMode, int32_t borderValue, bool useInputFile)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVTensorHandle inTensorHandle     = NULL;
    NVCVTensorHandle outTensorHandle    = NULL;
    NVCVTensorHandle refTensorHandle    = NULL;
    NVCVTensorHandle kernelTensorHandle = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cudaStream_t stream                 = NULL;
    int32_t err                         = 0;

    NVCVTensorRequirements tensorRequirements;

    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeTensors(argc, argv, imgWidth, imgHeight, kernelSize, useInputFile, &allocatorHandle,
                                       &inTensorHandle, &outTensorHandle, &refTensorHandle, &kernelTensorHandle,
                                       &tensorRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(
        pvaConv2dCreate(&operatorHandle, &tensorRequirements, borderMode, borderValue, kernelTensorHandle), err,
        FreeResources);

    /* Submit to CUDA stream */
    NVCV_CHECK_ERROR_GOTO(pvaConv2dSubmit(operatorHandle, stream, inTensorHandle, outTensorHandle), err, FreeResources);

    /* Synchronize stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(inTensorHandle, refTensorHandle, outTensorHandle, kernelTensorHandle, borderMode, borderValue,
                         useInputFile, imgWidth, imgHeight, "Submit to CUDA stream");

FreeResources: /* clean up all allocated resources */
    cudaStreamDestroy(stream);
    nvcvTensorDecRef(inTensorHandle, NULL);
    nvcvTensorDecRef(outTensorHandle, NULL);
    nvcvTensorDecRef(refTensorHandle, NULL);
    nvcvTensorDecRef(kernelTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

/**
 * @brief Test conv2d operation using cuPVA stream
 */
int SubmitConv2dToCupvaStream(int argc, char **argv, int32_t imgWidth, int32_t imgHeight, int32_t kernelSize,
                              NVCVBorderType borderMode, int32_t borderValue, bool useInputFile)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVTensorHandle inTensorHandle     = NULL;
    NVCVTensorHandle outTensorHandle    = NULL;
    NVCVTensorHandle refTensorHandle    = NULL;
    NVCVTensorHandle kernelTensorHandle = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cupvaSyncObj_t sync                 = NULL;
    cupvaStream_t stream                = NULL;
    int32_t err                         = 0;

    NVCVTensorRequirements tensorRequirements;

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeTensors(argc, argv, imgWidth, imgHeight, kernelSize, useInputFile, &allocatorHandle,
                                       &inTensorHandle, &outTensorHandle, &refTensorHandle, &kernelTensorHandle,
                                       &tensorRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(
        pvaConv2dCreate(&operatorHandle, &tensorRequirements, borderMode, borderValue, kernelTensorHandle), err,
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

    NVCV_CHECK_ERROR_GOTO(pvaConv2dSubmit(operatorHandle, stream, inTensorHandle, outTensorHandle), err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    /* Wait for completion */
    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(inTensorHandle, refTensorHandle, outTensorHandle, kernelTensorHandle, borderMode, borderValue,
                         useInputFile, imgWidth, imgHeight, "Submit to cuPVA stream");

FreeResources: /* clean up all allocated resources */
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvTensorDecRef(inTensorHandle, NULL);
    nvcvTensorDecRef(outTensorHandle, NULL);
    nvcvTensorDecRef(refTensorHandle, NULL);
    nvcvTensorDecRef(kernelTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char **argv)
{
    /* Parameter settings */
    const int32_t imgWidth   = 768;
    const int32_t imgHeight  = 512;
    const int32_t kernelSize = 5;

    const NVCVBorderType borderMode = NVCV_BORDER_CONSTANT;
    const int32_t borderValue       = 99;

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
        cudaStreamSubmitTestResult = SubmitConv2dToCudaStream(argc, argv, imgWidth, imgHeight, kernelSize, borderMode,
                                                              borderValue, useInputFile);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult =
        SubmitConv2dToCupvaStream(argc, argv, imgWidth, imgHeight, kernelSize, borderMode, borderValue, useInputFile);

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
