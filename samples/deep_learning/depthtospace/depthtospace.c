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
#include "ref/depthtospace_ref.h"

#include <OpDepthToSpace.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>
#include <stdio.h>

/**
 * @brief Allocate and initialize the input, output and reference tensors.
 *
 * The reference model consumes the input through a 5-D view
 * [blkSz, blkSz, C/(blkSz*blkSz), H, W] which it permutes into the output, so an
 * extra reshaped handle onto the same input storage is created for it.
 */
static int AllocateAndInitializeTensors(int32_t channel, int32_t height, int32_t width, int32_t blkSz,
                                        NVCVDataType dataType, NVCVAllocatorHandle *allocatorHandle,
                                        NVCVTensorHandle *inTensorHandle, NVCVTensorHandle *outTensorHandle,
                                        NVCVTensorHandle *refInTensorHandle, NVCVTensorHandle *refOutTensorHandle,
                                        NVCVTensorRequirements *tensorRequirements)
{
    int32_t err              = 0;
    const int32_t tensorRank = 3;
    NVCVTensorLayout tensorLayout;
    NVCVTensorRequirements outTensorRequirements;

    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("CHW", &tensorLayout), err, Error);

    int64_t tensorShape[] = {channel, height, width};
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(tensorRank, tensorShape, dataType, tensorLayout, 0, 0, tensorRequirements), err,
        Error);

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensorRequirements, *allocatorHandle, inTensorHandle), err, Error);

    /* 5-D view onto the input storage, as the reference model expects. */
    {
        NVCVTensorLayout refInLayout;
        NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("01234", &refInLayout), err, Error);

        int64_t refInShape[] = {blkSz, blkSz, channel / (blkSz * blkSz), height, width};
        NVCV_CHECK_ERROR_GOTO(nvcvTensorReshape(*inTensorHandle, 5, refInShape, refInLayout, refInTensorHandle), err,
                              Error);
    }

    /* Output shape [C/(B*B), H*B, W*B]. */
    {
        int64_t outShape[] = {channel / (blkSz * blkSz), height * blkSz, width * blkSz};
        NVCV_CHECK_ERROR_GOTO(
            nvcvTensorCalcRequirementsPva(tensorRank, outShape, dataType, tensorLayout, 0, 0, &outTensorRequirements),
            err, Error);
    }

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&outTensorRequirements, *allocatorHandle, outTensorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&outTensorRequirements, *allocatorHandle, refOutTensorHandle), err,
                          Error);

    generateDepthToSpaceInput(*inTensorHandle, true);

    return 0;

Error:
    return err;
}

/**
 * @brief Validate the operator output against the CPU reference.
 */
static int ValidateOutput(NVCVTensorHandle refInTensorHandle, NVCVTensorHandle refOutTensorHandle,
                          NVCVTensorHandle outTensorHandle, int32_t blkSz, const char *testName)
{
    int32_t err = 0;

    depthToSpaceRef(refInTensorHandle, refOutTensorHandle, blkSz);

    err = checkDepthToSpaceOutput(refOutTensorHandle, outTensorHandle);
    if (err)
    {
        printf("%s test failed\n", testName);
    }
    return err;
}

/**
 * @brief Test the depth to space operation using a CUDA stream.
 */
int SubmitDepthToSpaceToCudaStream(int32_t channel, int32_t height, int32_t width, int32_t blkSz, NVCVDataType dataType)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVTensorHandle inTensorHandle     = NULL;
    NVCVTensorHandle outTensorHandle    = NULL;
    NVCVTensorHandle refInTensorHandle  = NULL;
    NVCVTensorHandle refOutTensorHandle = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cudaStream_t stream                 = NULL;
    int32_t err                         = 0;

    NVCVTensorRequirements tensorRequirements;

    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    err = AllocateAndInitializeTensors(channel, height, width, blkSz, dataType, &allocatorHandle, &inTensorHandle,
                                       &outTensorHandle, &refInTensorHandle, &refOutTensorHandle, &tensorRequirements);
    if (err)
        goto FreeResources;

    NVCV_CHECK_ERROR_GOTO(pvaDepthToSpaceCreate(&operatorHandle, &tensorRequirements, blkSz), err, FreeResources);

    NVCV_CHECK_ERROR_GOTO(pvaDepthToSpaceSubmit(operatorHandle, stream, inTensorHandle, outTensorHandle), err,
                          FreeResources);

    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    err = ValidateOutput(refInTensorHandle, refOutTensorHandle, outTensorHandle, blkSz, "Submit to CUDA stream");

FreeResources: /* clean up all allocated resources */
    cudaStreamDestroy(stream);
    nvcvTensorDecRef(inTensorHandle, NULL);
    nvcvTensorDecRef(outTensorHandle, NULL);
    nvcvTensorDecRef(refInTensorHandle, NULL);
    nvcvTensorDecRef(refOutTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

/**
 * @brief Test the depth to space operation using a cuPVA stream.
 */
int SubmitDepthToSpaceToCupvaStream(int32_t channel, int32_t height, int32_t width, int32_t blkSz,
                                    NVCVDataType dataType)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVTensorHandle inTensorHandle     = NULL;
    NVCVTensorHandle outTensorHandle    = NULL;
    NVCVTensorHandle refInTensorHandle  = NULL;
    NVCVTensorHandle refOutTensorHandle = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cupvaSyncObj_t sync                 = NULL;
    cupvaStream_t stream                = NULL;
    int32_t err                         = 0;

    NVCVTensorRequirements tensorRequirements;

    err = AllocateAndInitializeTensors(channel, height, width, blkSz, dataType, &allocatorHandle, &inTensorHandle,
                                       &outTensorHandle, &refInTensorHandle, &refOutTensorHandle, &tensorRequirements);
    if (err)
        goto FreeResources;

    NVCV_CHECK_ERROR_GOTO(pvaDepthToSpaceCreate(&operatorHandle, &tensorRequirements, blkSz), err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaSyncObjCreate(&sync, false, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD), err,
                           FreeResources);

    cupvaFence_t fence;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceInit(&fence, sync), err, FreeResources);

    cupvaCmd_t rf;
    CUPVA_CHECK_ERROR_GOTO(CupvaCmdRequestFencesInit(&rf, &fence, 1), err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamCreate(&stream, CUPVA_PVA0, CUPVA_VPU_ANY), err, FreeResources);

    cupvaCmd_t const *cmds[1]  = {&rf};
    cupvaCmdStatus_t status[1] = {NULL};

    NVCV_CHECK_ERROR_GOTO(pvaDepthToSpaceSubmit(operatorHandle, stream, inTensorHandle, outTensorHandle), err,
                          FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    err = ValidateOutput(refInTensorHandle, refOutTensorHandle, outTensorHandle, blkSz, "Submit to cuPVA stream");

FreeResources: /* clean up all allocated resources */
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvTensorDecRef(inTensorHandle, NULL);
    nvcvTensorDecRef(outTensorHandle, NULL);
    nvcvTensorDecRef(refInTensorHandle, NULL);
    nvcvTensorDecRef(refOutTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    /* Parameter settings */
    const int32_t blkSz         = 2;
    const int32_t channel       = 4;
    const int32_t height        = 128;
    const int32_t width         = 128;
    const NVCVDataType dataType = NVCV_DATA_TYPE_U8;

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
        cudaStreamSubmitTestResult = SubmitDepthToSpaceToCudaStream(channel, height, width, blkSz, dataType);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult = SubmitDepthToSpaceToCupvaStream(channel, height, width, blkSz, dataType);

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
