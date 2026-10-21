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

#include <ImageIO.h>
#include <OpDLInference.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Fwd.h>
#include <nvcv/Size.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Structure for index-value pair for sorting
typedef struct
{
    int index;
    float value;
} IndexValuePair;

// Comparison function for qsort (descending order)
static int compareIndexValuePair(const void *a, const void *b)
{
    const IndexValuePair *pair_a = (const IndexValuePair *)a;
    const IndexValuePair *pair_b = (const IndexValuePair *)b;
    if (pair_a->value > pair_b->value)
        return -1;
    if (pair_a->value < pair_b->value)
        return 1;
    return 0;
}

// C implementation of topK - returns indices of top k values
static void topK(const float *values, int size, int k, int *outputIndices)
{
    IndexValuePair *pairs = (IndexValuePair *)malloc(size * sizeof(IndexValuePair));
    if (pairs == NULL)
    {
        printf("Error: Memory allocation failed in topK function\n");
        return;
    }

    for (int i = 0; i < size; i++)
    {
        pairs[i].index = i;
        pairs[i].value = values[i];
    }

    qsort(pairs, size, sizeof(IndexValuePair), compareIndexValuePair);

    for (int i = 0; i < k; i++)
    {
        outputIndices[i] = pairs[i].index;
    }

    free(pairs);
}

// Function to get model name from command line arguments
int GetModelName(int argc, char **argv, char *modelName, int maxNameLength)
{
    int err = -1;

    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "-m") || !strcmp(argv[i], "--model"))
        {
            if (i + 1 < argc)
            {
                strncpy(modelName, argv[++i], maxNameLength);
                modelName[maxNameLength - 1] = '\0';
                printf("Using model: %s\n", modelName);
                err = 0;
            }
            else
            {
                printf("--model option requires one argument.\n");
            }
            break;
        }
    }

    if (err != 0)
    {
        printf("Please specify model name using -m or --model option\n");
    }

    return err;
}

/**
 * @brief Allocate and initialize tensors
 */
static int AllocateAndInitializeTensors(int argc, char **argv, int32_t imgHeight, int32_t imgWidth, int32_t imgChannels,
                                        int64_t outputSize, NVCVAllocatorHandle *allocatorHandle,
                                        NVCVTensorHandle *inTensorHandle, NVCVTensorHandle *outTensorHandle,
                                        NVCVTensorRequirements *inputRequirements,
                                        NVCVTensorRequirements *outputRequirements, char *assetsDirectory,
                                        int32_t maxPathLength)
{
    int32_t err = 0;

    err = GetAssetsDirectory(argc, argv, assetsDirectory, maxPathLength);
    if (err)
    {
        printf("Failed to get assets directory\n");
        return err;
    }

    nvcvAllocatorConstructPva(allocatorHandle);

    NVCVTensorLayout inputLayout;
    nvcvTensorLayoutMake("HWC", &inputLayout);
    int64_t imageTensorShape[] = {imgHeight, imgWidth, imgChannels};
    nvcvTensorCalcRequirementsPva(3, imageTensorShape, NVCV_DATA_TYPE_S8, inputLayout, 0, 0, inputRequirements);

    nvcvTensorConstruct(inputRequirements, *allocatorHandle, inTensorHandle);

    NVCVTensorData inputTensorData;
    nvcvTensorExportData(*inTensorHandle, &inputTensorData);

    uint8_t *image_h;
    CupvaMemGetHostPointer((void **)&image_h, (void *)inputTensorData.buffer.strided.basePtr);

    if (ReadImageBuffer("/onnx/daisy.bin", assetsDirectory, image_h,
                        sizeof(uint8_t) * imgHeight * imgWidth * imgChannels) != 0)
    {
        printf("Failed to read image\n");
        return -1;
    }

    NVCVTensorLayout outputLayout;
    nvcvTensorLayoutMake("", &outputLayout);
    int64_t outputTensorShape[] = {outputSize};
    nvcvTensorCalcRequirementsPva(1, outputTensorShape, NVCV_DATA_TYPE_F32, outputLayout, 0, 0, outputRequirements);
    nvcvTensorConstruct(outputRequirements, *allocatorHandle, outTensorHandle);

    return 0;
}

/**
 * @brief Validate output
 */
static int ValidateOutput(NVCVTensorHandle outTensorHandle, int64_t outputSize, const int *referenceIndices,
                          int referenceCount, const char *testName)
{
    NVCVTensorData outTensorData;
    nvcvTensorExportData(outTensorHandle, &outTensorData);

    float *output_h = NULL;
    CupvaMemGetHostPointer((void **)&output_h, (void *)outTensorData.buffer.strided.basePtr);

    int outputIndices[5];
    topK(output_h, outputSize, 5, outputIndices);

    printf("%s - Top 5 predicted classes:\n", testName);
    for (int i = 0; i < 5; i++)
    {
        printf("  Class %d: confidence %.6f\n", outputIndices[i], output_h[outputIndices[i]]);
    }

    // Validate against reference indices
    int err = 0;
    for (int i = 0; i < referenceCount; i++)
    {
        int found = 0;
        for (int j = 0; j < 5; j++)
        {
            if (outputIndices[j] == referenceIndices[i])
            {
                found = 1;
                break;
            }
        }
        if (!found)
        {
            printf("Expected index: %d not found in output\n", referenceIndices[i]);
            err = 1;
        }
    }

    if (err)
    {
        printf("%s test failed\n", testName);
    }

    return err;
}

/**
 * @brief Test classification (DL inference) operation using CUDA stream
 */
int SubmitClassificationToCudaStream(int argc, char **argv, const char *modelName, int32_t imgHeight, int32_t imgWidth,
                                     int32_t imgChannels, int64_t outputSize)
{
    const int32_t maxPathLength = 200;
    char assetsDirectory[maxPathLength];

    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVTensorHandle inTensorHandle     = NULL;
    NVCVTensorHandle outTensorHandle    = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cudaStream_t stream                 = NULL;
    int32_t err                         = 0;

    NVCVTensorRequirements inputRequirements;
    NVCVTensorRequirements outputRequirements;

    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeTensors(argc, argv, imgHeight, imgWidth, imgChannels, outputSize, &allocatorHandle,
                                       &inTensorHandle, &outTensorHandle, &inputRequirements, &outputRequirements,
                                       assetsDirectory, maxPathLength);
    if (err)
        goto FreeResources;

    /* Create operator */
    PvaDLInferenceCreateParams params;
    params.inReqs[0]   = &inputRequirements;
    params.outReqs[0]  = &outputRequirements;
    params.numInReqs   = 1;
    params.numOutReqs  = 1;
    params.networkName = (char *)modelName;
    pvaDLInferenceCreate(&operatorHandle, &params);

    /* Submit to CUDA stream */
    PvaDLInferenceSubmitParams submitParams;
    submitParams.inTensors[0]  = inTensorHandle;
    submitParams.outTensors[0] = outTensorHandle;
    submitParams.numInTensors  = 1;
    submitParams.numOutTensors = 1;
    pvaDLInferenceSubmit(operatorHandle, stream, &submitParams);

    /* Synchronize stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    /* Validate output and check against reference indices */
    int referenceIndices[2] = {309, 985}; // bee, daisy
    err                     = ValidateOutput(outTensorHandle, outputSize, referenceIndices, 2, "Submit to CUDA stream");

FreeResources:
    cudaStreamDestroy(stream);
    nvcvTensorDecRef(inTensorHandle, NULL);
    nvcvTensorDecRef(outTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

/**
 * @brief Test classification (DL inference) operation using cuPVA stream
 */
int SubmitClassificationToCupvaStream(int argc, char **argv, const char *modelName, int32_t imgHeight, int32_t imgWidth,
                                      int32_t imgChannels, int64_t outputSize)
{
    const int32_t maxPathLength = 200;
    char assetsDirectory[maxPathLength];

    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVTensorHandle inTensorHandle     = NULL;
    NVCVTensorHandle outTensorHandle    = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cupvaSyncObj_t sync                 = NULL;
    cupvaStream_t stream                = NULL;
    int32_t err                         = 0;

    NVCVTensorRequirements inputRequirements;
    NVCVTensorRequirements outputRequirements;

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeTensors(argc, argv, imgHeight, imgWidth, imgChannels, outputSize, &allocatorHandle,
                                       &inTensorHandle, &outTensorHandle, &inputRequirements, &outputRequirements,
                                       assetsDirectory, maxPathLength);
    if (err)
        goto FreeResources;

    /* Create operator */
    PvaDLInferenceCreateParams params;
    params.inReqs[0]   = &inputRequirements;
    params.outReqs[0]  = &outputRequirements;
    params.numInReqs   = 1;
    params.numOutReqs  = 1;
    params.networkName = (char *)modelName;
    pvaDLInferenceCreate(&operatorHandle, &params);

    /* Create cuPVA stream and synchronization objects */
    CupvaSyncObjCreate(&sync, true, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD);

    cupvaFence_t fence;
    CupvaFenceInit(&fence, sync);

    cupvaCmd_t rf;
    CupvaCmdRequestFencesInit(&rf, &fence, 1);

    CupvaStreamCreate(&stream, CUPVA_PVA0, CUPVA_VPU_ANY);

    /* Submit to cuPVA stream */
    cupvaCmd_t const *cmds[1]  = {&rf};
    cupvaCmdStatus_t status[1] = {NULL};

    PvaDLInferenceSubmitParams submitParams;
    submitParams.inTensors[0]  = inTensorHandle;
    submitParams.outTensors[0] = outTensorHandle;
    submitParams.numInTensors  = 1;
    submitParams.numOutTensors = 1;
    pvaDLInferenceSubmit(operatorHandle, stream, &submitParams);

    CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1);

    /* Wait for completion */
    bool waitSuccess;
    CupvaFenceWait(&fence, -1, &waitSuccess);

    /* Validate output and check against reference indices */
    int referenceIndices[2] = {309, 985}; // bee, daisy
    err = ValidateOutput(outTensorHandle, outputSize, referenceIndices, 2, "Submit to cuPVA stream");

FreeResources:
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvTensorDecRef(inTensorHandle, NULL);
    nvcvTensorDecRef(outTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char **argv)
{
    /* Parameter settings */
    int32_t imgHeight   = 224;
    int32_t imgWidth    = 224;
    int32_t imgChannels = 3;
    int64_t outputSize  = 1000;

    const int32_t maxNameLength = 100;
    char modelName[maxNameLength];
    if (GetModelName(argc, argv, modelName, maxNameLength) != 0)
    {
        printf("Failed to get model name\n");
        return -1;
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
        cudaStreamSubmitTestResult =
            SubmitClassificationToCudaStream(argc, argv, modelName, imgHeight, imgWidth, imgChannels, outputSize);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult =
        SubmitClassificationToCupvaStream(argc, argv, modelName, imgHeight, imgWidth, imgChannels, outputSize);

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
