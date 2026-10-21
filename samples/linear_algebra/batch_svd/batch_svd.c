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
#include "ref/batch_svd_ref.h"

#include <OpBatchSVD.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Fwd.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>
#include <stdio.h>

/**
 * @brief Allocate and initialize tensors
 */
static int AllocateAndInitializeTensors(int32_t imgWidth, int32_t imgHeight, int32_t numChannels, int32_t tensorRank,
                                        NVCVAllocatorHandle *allocatorHandle, NVCVTensorHandle *srcTensorHandle,
                                        NVCVTensorHandle *uTensorHandle, NVCVTensorHandle *sTensorHandle,
                                        NVCVTensorHandle *vTensorHandle, NVCVTensorHandle *uRefTensorHandle,
                                        NVCVTensorHandle *sRefTensorHandle, NVCVTensorHandle *vRefTensorHandle,
                                        NVCVTensorRequirements *srcRequirements)
{
    int32_t err = 0;

    int64_t srcTensorShape[] = {imgHeight, imgWidth, numChannels};
    int64_t uTensorShape[]   = {imgHeight, imgWidth, numChannels};
    int64_t sTensorShape[]   = {1, imgWidth, numChannels};
    int64_t vTensorShape[]   = {imgWidth, imgWidth, numChannels};

    NVCVTensorLayout tensorLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HWC", &tensorLayout), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank, srcTensorShape, NVCV_DATA_TYPE_F32, tensorLayout, 0,
                                                        0, srcRequirements),
                          err, Error);

    NVCVTensorRequirements uRequirements, sRequirements, vRequirements;
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(tensorRank, uTensorShape, NVCV_DATA_TYPE_F32, tensorLayout, 0, 0, &uRequirements),
        err, Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(tensorRank, sTensorShape, NVCV_DATA_TYPE_F32, tensorLayout, 0, 0, &sRequirements),
        err, Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(tensorRank, vTensorShape, NVCV_DATA_TYPE_F32, tensorLayout, 0, 0, &vRequirements),
        err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(srcRequirements, *allocatorHandle, srcTensorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&uRequirements, *allocatorHandle, uTensorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&sRequirements, *allocatorHandle, sTensorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&vRequirements, *allocatorHandle, vTensorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&uRequirements, *allocatorHandle, uRefTensorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&sRequirements, *allocatorHandle, sRefTensorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&vRequirements, *allocatorHandle, vRefTensorHandle), err, Error);

    generateBatchSVDRandomInput(*srcTensorHandle, -1.f, 1.f, 0);

    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(NVCVTensorHandle srcTensorHandle, NVCVTensorHandle uTensorHandle,
                          NVCVTensorHandle sTensorHandle, NVCVTensorHandle vTensorHandle,
                          NVCVTensorHandle uRefTensorHandle, NVCVTensorHandle sRefTensorHandle,
                          NVCVTensorHandle vRefTensorHandle, int32_t maxIters, float tol, const char *testName)
{
    batchSVDRef(srcTensorHandle, uRefTensorHandle, sRefTensorHandle, vRefTensorHandle, maxIters, tol);
    int32_t err = checkBatchSVDOutput(uTensorHandle, sTensorHandle, vTensorHandle, uRefTensorHandle, sRefTensorHandle,
                                      vRefTensorHandle, tol);
    if (err)
    {
        printf("%s test failed\n", testName);
    }
    return err;
}

/**
 * @brief Test batch SVD operation using CUDA stream
 */
int SubmitBatchSVDToCudaStream(int32_t imgWidth, int32_t imgHeight, int32_t numChannels, int32_t tensorRank,
                               int32_t maxIters, float tol)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVTensorHandle srcTensorHandle    = NULL;
    NVCVTensorHandle uTensorHandle      = NULL;
    NVCVTensorHandle sTensorHandle      = NULL;
    NVCVTensorHandle vTensorHandle      = NULL;
    NVCVTensorHandle uRefTensorHandle   = NULL;
    NVCVTensorHandle sRefTensorHandle   = NULL;
    NVCVTensorHandle vRefTensorHandle   = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cudaStream_t stream                 = NULL;
    int32_t err                         = 0;

    NVCVTensorRequirements srcRequirements;

    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeTensors(imgWidth, imgHeight, numChannels, tensorRank, &allocatorHandle, &srcTensorHandle,
                                       &uTensorHandle, &sTensorHandle, &vTensorHandle, &uRefTensorHandle,
                                       &sRefTensorHandle, &vRefTensorHandle, &srcRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaBatchSVDCreate(&operatorHandle, &srcRequirements), err, FreeResources);

    /* Submit to CUDA stream */
    NVCV_CHECK_ERROR_GOTO(pvaBatchSVDSubmit(operatorHandle, stream, srcTensorHandle, uTensorHandle, sTensorHandle,
                                            vTensorHandle, maxIters, tol),
                          err, FreeResources);

    /* Synchronize stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(srcTensorHandle, uTensorHandle, sTensorHandle, vTensorHandle, uRefTensorHandle,
                         sRefTensorHandle, vRefTensorHandle, maxIters, tol, "Submit to CUDA stream");

FreeResources: /* clean up all allocated resources */
    cudaStreamDestroy(stream);
    nvcvTensorDecRef(srcTensorHandle, NULL);
    nvcvTensorDecRef(uTensorHandle, NULL);
    nvcvTensorDecRef(sTensorHandle, NULL);
    nvcvTensorDecRef(vTensorHandle, NULL);
    nvcvTensorDecRef(uRefTensorHandle, NULL);
    nvcvTensorDecRef(sRefTensorHandle, NULL);
    nvcvTensorDecRef(vRefTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

/**
 * @brief Test batch SVD operation using cuPVA stream
 */
int SubmitBatchSVDToCupvaStream(int32_t imgWidth, int32_t imgHeight, int32_t numChannels, int32_t tensorRank,
                                int32_t maxIters, float tol)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVTensorHandle srcTensorHandle    = NULL;
    NVCVTensorHandle uTensorHandle      = NULL;
    NVCVTensorHandle sTensorHandle      = NULL;
    NVCVTensorHandle vTensorHandle      = NULL;
    NVCVTensorHandle uRefTensorHandle   = NULL;
    NVCVTensorHandle sRefTensorHandle   = NULL;
    NVCVTensorHandle vRefTensorHandle   = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cupvaSyncObj_t sync                 = NULL;
    cupvaStream_t stream                = NULL;
    int32_t err                         = 0;

    NVCVTensorRequirements srcRequirements;

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeTensors(imgWidth, imgHeight, numChannels, tensorRank, &allocatorHandle, &srcTensorHandle,
                                       &uTensorHandle, &sTensorHandle, &vTensorHandle, &uRefTensorHandle,
                                       &sRefTensorHandle, &vRefTensorHandle, &srcRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaBatchSVDCreate(&operatorHandle, &srcRequirements), err, FreeResources);

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

    NVCV_CHECK_ERROR_GOTO(pvaBatchSVDSubmit(operatorHandle, stream, srcTensorHandle, uTensorHandle, sTensorHandle,
                                            vTensorHandle, maxIters, tol),
                          err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    /* Wait for completion */
    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(srcTensorHandle, uTensorHandle, sTensorHandle, vTensorHandle, uRefTensorHandle,
                         sRefTensorHandle, vRefTensorHandle, maxIters, tol, "Submit to cuPVA stream");

FreeResources: /* clean up all allocated resources */
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvTensorDecRef(srcTensorHandle, NULL);
    nvcvTensorDecRef(uTensorHandle, NULL);
    nvcvTensorDecRef(sTensorHandle, NULL);
    nvcvTensorDecRef(vTensorHandle, NULL);
    nvcvTensorDecRef(uRefTensorHandle, NULL);
    nvcvTensorDecRef(sRefTensorHandle, NULL);
    nvcvTensorDecRef(vRefTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char *argv[])
{
    /* Parameter settings */
    const int32_t imgWidth    = 24;
    const int32_t imgHeight   = 25;
    const int32_t numChannels = 64;
    const int32_t tensorRank  = 3;
    const int32_t maxIters    = 30;
    const float tol           = 1e-7f;

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
            SubmitBatchSVDToCudaStream(imgWidth, imgHeight, numChannels, tensorRank, maxIters, tol);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult =
        SubmitBatchSVDToCupvaStream(imgWidth, imgHeight, numChannels, tensorRank, maxIters, tol);

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
