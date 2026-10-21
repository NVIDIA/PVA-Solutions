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
#include "ref/remap_ref.h"

#include <OpRemap.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Fwd.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>

/**
 * @brief Allocate and initialize images and tensors
 */
static int AllocateAndInitializeResources(int32_t inImgWidth, int32_t inImgHeight, int32_t outImgWidth,
                                          int32_t outImgHeight, bool isRandom, NVCVAllocatorHandle *allocatorHandle,
                                          NVCVImageHandle *inImageHandle, NVCVImageHandle *outImageHandle,
                                          NVCVImageHandle *refOutImageHandle, NVCVTensorHandle *warpMapHandle,
                                          NVCVImageRequirements *inImageRequirements,
                                          NVCVImageRequirements *outImageRequirements,
                                          NVCVTensorRequirements *warpMapRequirements)
{
    int32_t err        = 0;
    int32_t tensorRank = 3;
    NVCVTensorLayout tensorLayout;
    int64_t tensorShape[] = {outImgHeight, outImgWidth, 2};

    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HWC", &tensorLayout), err, Error);

    NVCV_CHECK_ERROR_GOTO(
        nvcvImageCalcRequirementsPva(inImgWidth, inImgHeight, NVCV_IMAGE_FORMAT_NV12_ER, 0, 0, inImageRequirements),
        err, Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvImageCalcRequirementsPva(outImgWidth, outImgHeight, NVCV_IMAGE_FORMAT_NV12_ER, 0, 0, outImageRequirements),
        err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank, tensorShape, NVCV_DATA_TYPE_F32, tensorLayout, 0, 0,
                                                        warpMapRequirements),
                          err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(inImageRequirements, *allocatorHandle, inImageHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(outImageRequirements, *allocatorHandle, outImageHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(outImageRequirements, *allocatorHandle, refOutImageHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(warpMapRequirements, *allocatorHandle, warpMapHandle), err, Error);

    generateRandomRemapInput(*inImageHandle, isRandom);
    generateRandomRemapWarpMap(*warpMapHandle, inImgWidth, inImgHeight, isRandom);

    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(NVCVImageHandle inImageHandle, NVCVImageHandle refOutImageHandle,
                          NVCVImageHandle outImageHandle, NVCVTensorHandle warpMapHandle,
                          PVAInterpolationType interpType, NVCVBorderType borderMode, const char *testName)
{
    remapRef(inImageHandle, refOutImageHandle, warpMapHandle, interpType, borderMode);
    int32_t err = checkRemapOutput(refOutImageHandle, outImageHandle);
    if (err)
    {
        printf("%s test failed\n", testName);
    }
    return err;
}

/**
 * @brief Test remap operation using CUDA stream
 */
int SubmitRemapToCudaStream(int32_t width, int32_t height, bool isRandom, PVAInterpolationType interpType,
                            NVCVBorderType borderMode)
{
    const int32_t outImgWidth  = width;
    const int32_t outImgHeight = height;
    const int32_t inImgWidth   = width;
    const int32_t inImgHeight  = height;

    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVImageHandle inImageHandle       = NULL;
    NVCVImageHandle outImageHandle      = NULL;
    NVCVImageHandle refOutImageHandle   = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cudaStream_t stream                 = NULL;
    int32_t err                         = 0;

    NVCVTensorHandle warpMapHandle = NULL;
    NVCVImageRequirements inImageRequirements, outImageRequirements;
    NVCVTensorRequirements warpMapRequirements;

    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    /* Allocate and initialize resources */
    err = AllocateAndInitializeResources(inImgWidth, inImgHeight, outImgWidth, outImgHeight, isRandom, &allocatorHandle,
                                         &inImageHandle, &outImageHandle, &refOutImageHandle, &warpMapHandle,
                                         &inImageRequirements, &outImageRequirements, &warpMapRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaRemapCreate(&operatorHandle, &inImageRequirements, &outImageRequirements,
                                         &warpMapRequirements, interpType, borderMode),
                          err, FreeResources);

    /* PVA can only handle tile size smaller than 32KB */
    NVCV_CHECK_ERROR_GOTO(pvaRemapValidate(warpMapHandle, inImgWidth, inImgHeight, outImgWidth, outImgHeight), err,
                          FreeResources);

    /* Submit to CUDA stream */
    NVCV_CHECK_ERROR_GOTO(pvaRemapSubmit(operatorHandle, stream, inImageHandle, outImageHandle, warpMapHandle), err,
                          FreeResources);

    /* Synchronize stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(inImageHandle, refOutImageHandle, outImageHandle, warpMapHandle, interpType, borderMode,
                         "Submit to CUDA stream");

FreeResources: /* clean up all allocated resources */
    cudaStreamDestroy(stream);
    nvcvImageDecRef(inImageHandle, NULL);
    nvcvImageDecRef(outImageHandle, NULL);
    nvcvImageDecRef(refOutImageHandle, NULL);
    nvcvTensorDecRef(warpMapHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

/**
 * @brief Test remap operation using cuPVA stream
 */
int SubmitRemapToCupvaStream(int32_t width, int32_t height, bool isRandom, PVAInterpolationType interpType,
                             NVCVBorderType borderMode)
{
    const int32_t outImgWidth  = width;
    const int32_t outImgHeight = height;
    const int32_t inImgWidth   = width;
    const int32_t inImgHeight  = height;

    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVImageHandle inImageHandle       = NULL;
    NVCVImageHandle outImageHandle      = NULL;
    NVCVImageHandle refOutImageHandle   = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cupvaSyncObj_t sync                 = NULL;
    cupvaStream_t stream                = NULL;
    int32_t err                         = 0;

    NVCVTensorHandle warpMapHandle = NULL;
    NVCVImageRequirements inImageRequirements, outImageRequirements;
    NVCVTensorRequirements warpMapRequirements;

    /* Allocate and initialize resources */
    err = AllocateAndInitializeResources(inImgWidth, inImgHeight, outImgWidth, outImgHeight, isRandom, &allocatorHandle,
                                         &inImageHandle, &outImageHandle, &refOutImageHandle, &warpMapHandle,
                                         &inImageRequirements, &outImageRequirements, &warpMapRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaRemapCreate(&operatorHandle, &inImageRequirements, &outImageRequirements,
                                         &warpMapRequirements, interpType, borderMode),
                          err, FreeResources);

    /* PVA can only handle tile size smaller than 32KB */
    NVCV_CHECK_ERROR_GOTO(pvaRemapValidate(warpMapHandle, inImgWidth, inImgHeight, outImgWidth, outImgHeight), err,
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

    NVCV_CHECK_ERROR_GOTO(pvaRemapSubmit(operatorHandle, stream, inImageHandle, outImageHandle, warpMapHandle), err,
                          FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    /* Wait for completion */
    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(inImageHandle, refOutImageHandle, outImageHandle, warpMapHandle, interpType, borderMode,
                         "Submit to cuPVA stream");

FreeResources: /* clean up all allocated resources */
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvImageDecRef(inImageHandle, NULL);
    nvcvImageDecRef(outImageHandle, NULL);
    nvcvImageDecRef(refOutImageHandle, NULL);
    nvcvTensorDecRef(warpMapHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char **argv)
{
    /* Parameter settings */
    // For random warp map, input image size <= 32KB (32 tiles for 64x16 tile size)
    int32_t width  = 768;
    int32_t height = 512;
    bool isRandom  = false;

    PVAInterpolationType interpType = PVA_INTERPOLATION_LINEAR;
    NVCVBorderType borderMode       = NVCV_BORDER_CONSTANT;

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
        cudaStreamSubmitTestResult = SubmitRemapToCudaStream(width, height, isRandom, interpType, borderMode);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult = SubmitRemapToCupvaStream(width, height, isRandom, interpType, borderMode);

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
