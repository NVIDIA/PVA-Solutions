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
#include "ref/floyd_warshall_ref.h"

#include <OpFloydWarshall.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Fwd.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>

/**
 * @brief Allocate and initialize tensors
 */
static int AllocateAndInitializeResources(int32_t height, int32_t width, bool directed, bool connected,
                                          NVCVAllocatorHandle *allocatorHandle, NVCVTensorHandle *refLenTensor,
                                          NVCVTensorHandle *refNextTensor, NVCVTensorHandle *pvaLenTensor,
                                          NVCVTensorHandle *pvaNextTensor, NVCVTensorRequirements *tensorRequirements)
{
    int32_t err        = 0;
    int32_t tensorRank = 3;
    NVCVTensorLayout tensorLayout;
    int64_t tensorShape[] = {height, width, 1};

    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HWC", &tensorLayout), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank, tensorShape, NVCV_DATA_TYPE_U16, tensorLayout, 0, 0,
                                                        tensorRequirements),
                          err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensorRequirements, *allocatorHandle, refLenTensor), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensorRequirements, *allocatorHandle, refNextTensor), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensorRequirements, *allocatorHandle, pvaLenTensor), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensorRequirements, *allocatorHandle, pvaNextTensor), err, Error);

    generateRandomFloydWarshallInput(*refLenTensor, *refNextTensor, directed, connected);
    copyInputTensor(*refLenTensor, *pvaLenTensor);
    copyInputTensor(*refNextTensor, *pvaNextTensor);
    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(NVCVTensorHandle refLenTensor, NVCVTensorHandle refNextTensor, NVCVTensorHandle pvaLenTensor,
                          NVCVTensorHandle pvaNextTensor, const char *testName)
{
    floydWarshallRef(refLenTensor, refNextTensor);
    int32_t err = checkFloydWarshallOutput(refLenTensor, refNextTensor, pvaLenTensor, pvaNextTensor);
    if (err)
    {
        printf("%s test failed\n", testName);
    }
    return err;
}

/**
 * @brief Test Floyd-Warshall operation using CUDA stream
 */
int SubmitFloydWarshallToCudaStream(int32_t height, int32_t width, bool directed, bool connected)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVTensorHandle refLenTensor       = NULL;
    NVCVTensorHandle refNextTensor      = NULL;
    NVCVTensorHandle pvaLenTensor       = NULL;
    NVCVTensorHandle pvaNextTensor      = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cudaStream_t stream                 = NULL;
    int32_t err                         = 0;

    NVCVTensorRequirements tensorRequirements;

    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeResources(height, width, directed, connected, &allocatorHandle, &refLenTensor,
                                         &refNextTensor, &pvaLenTensor, &pvaNextTensor, &tensorRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaFloydWarshallCreate(&operatorHandle, &tensorRequirements), err, FreeResources);

    /* Submit to CUDA stream */
    NVCV_CHECK_ERROR_GOTO(pvaFloydWarshallSubmit(operatorHandle, stream, pvaLenTensor, pvaNextTensor), err,
                          FreeResources);

    /* Synchronize stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(refLenTensor, refNextTensor, pvaLenTensor, pvaNextTensor, "Submit to CUDA stream");

FreeResources:
    cudaStreamDestroy(stream);
    nvcvTensorDecRef(refLenTensor, NULL);
    nvcvTensorDecRef(refNextTensor, NULL);
    nvcvTensorDecRef(pvaLenTensor, NULL);
    nvcvTensorDecRef(pvaNextTensor, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

/**
 * @brief Test Floyd-Warshall operation using cuPVA stream
 */
int SubmitFloydWarshallToCupvaStream(int32_t height, int32_t width, bool directed, bool connected)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVTensorHandle refLenTensor       = NULL;
    NVCVTensorHandle refNextTensor      = NULL;
    NVCVTensorHandle pvaLenTensor       = NULL;
    NVCVTensorHandle pvaNextTensor      = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cupvaSyncObj_t sync                 = NULL;
    cupvaStream_t stream                = NULL;
    int32_t err                         = 0;

    NVCVTensorRequirements tensorRequirements;

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeResources(height, width, directed, connected, &allocatorHandle, &refLenTensor,
                                         &refNextTensor, &pvaLenTensor, &pvaNextTensor, &tensorRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaFloydWarshallCreate(&operatorHandle, &tensorRequirements), err, FreeResources);

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

    NVCV_CHECK_ERROR_GOTO(pvaFloydWarshallSubmit(operatorHandle, stream, pvaLenTensor, pvaNextTensor), err,
                          FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    /* Wait for completion */
    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(refLenTensor, refNextTensor, pvaLenTensor, pvaNextTensor, "Submit to cuPVA stream");

FreeResources:
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvTensorDecRef(refLenTensor, NULL);
    nvcvTensorDecRef(refNextTensor, NULL);
    nvcvTensorDecRef(pvaLenTensor, NULL);
    nvcvTensorDecRef(pvaNextTensor, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char **argv)
{
    /* Parameter settings */
    int32_t height = 240;
    int32_t width  = 240;
    bool directed  = false;
    bool connected = true;

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
        cudaStreamSubmitTestResult = SubmitFloydWarshallToCudaStream(height, width, directed, connected);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult = SubmitFloydWarshallToCupvaStream(height, width, directed, connected);

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