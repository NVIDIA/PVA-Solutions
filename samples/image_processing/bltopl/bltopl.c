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
#include "ref/bltopl_ref.h"

#include <BlockLinear.h>
#include <OpBlToPl.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Fwd.h>
#include <nvcv/Image.h>
#include <nvcv/alloc/Allocator.h>

/**
 * @brief Allocate and initialize images
 */
static int AllocateAndInitializeImages(int32_t imgWidth, int32_t imgHeight, NVCVAllocatorHandle *allocatorHandle,
                                       NVCVImageHandle *inPlImageHandle, NVCVImageHandle *inImageHandle,
                                       NVCVImageHandle *outImageHandle, NVCVImageHandle *refOutImageHandle,
                                       NVCVImageRequirements *imageRequirements)
{
    int32_t err = 0;

    // Use NVCV_IMAGE_FORMAT_NV12 instead of NVCV_IMAGE_FORMAT_NV12_BL since nvcvImageConstruct doesn't support BL format
    NVCV_CHECK_ERROR_GOTO(
        nvcvImageCalcRequirementsPva(imgWidth, imgHeight, NVCV_IMAGE_FORMAT_NV12, 0, 0, imageRequirements), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(imageRequirements, *allocatorHandle, inPlImageHandle), err, Error);

    // Fill PL image content
    generateRandomPlInput(*inPlImageHandle, 1);

    // Generate a NVCV image with the BL layout
    NVCVImageRequirements imageRequirementsBL = {imgWidth, imgHeight, NVCV_IMAGE_FORMAT_NV12_BL};
    NVCV_CHECK_ERROR_GOTO(nvcvImageConstructBL(&imageRequirementsBL, inImageHandle, NULL), err, Error);

    // Convert from PL layout into BL layout
    plToBl(*inPlImageHandle, *inImageHandle);

    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(imageRequirements, *allocatorHandle, outImageHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(imageRequirements, *allocatorHandle, refOutImageHandle), err, Error);

    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(NVCVImageHandle inImageHandle, NVCVImageHandle refOutImageHandle,
                          NVCVImageHandle outImageHandle, const char *testName)
{
    blToPl(inImageHandle, refOutImageHandle);
    int32_t err = checkBltoplOutput(refOutImageHandle, outImageHandle);
    if (err)
    {
        printf("%s test failed\n", testName);
    }
    return err;
}

/**
 * @brief Test BL to PL conversion operation using CUDA stream
 */
int SubmitBlToPlToCudaStream(int32_t imgWidth, int32_t imgHeight)
{
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVImageHandle inPlImageHandle     = NULL;
    NVCVImageHandle inImageHandle       = NULL;
    NVCVImageHandle outImageHandle      = NULL;
    NVCVImageHandle refOutImageHandle   = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cudaStream_t stream                 = NULL;
    int32_t err                         = 0;

    NVCVImageRequirements imageRequirements;

    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    /* Allocate and initialize images */
    err = AllocateAndInitializeImages(imgWidth, imgHeight, &allocatorHandle, &inPlImageHandle, &inImageHandle,
                                      &outImageHandle, &refOutImageHandle, &imageRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaBlToPlCreate(&operatorHandle, &imageRequirements), err, FreeResources);

    /* Submit to CUDA stream */
    NVCV_CHECK_ERROR_GOTO(pvaBlToPlSubmit(operatorHandle, stream, inImageHandle, outImageHandle), err, FreeResources);

    /* Synchronize stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(inImageHandle, refOutImageHandle, outImageHandle, "Submit to CUDA stream");

FreeResources: /* clean up all allocated resources */
    cudaStreamDestroy(stream);
    nvcvImageDecRef(inImageHandle, NULL);
    nvcvImageDecRef(inPlImageHandle, NULL);
    nvcvImageDecRef(outImageHandle, NULL);
    nvcvImageDecRef(refOutImageHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

/**
 * @brief Test BL to PL conversion operation using cuPVA stream
 */
int SubmitBlToPlToCupvaStream(int32_t imgWidth, int32_t imgHeight)
{
    NVCVAllocatorHandle allocatorHandle = NULL;
    // a PL layout image is needed to be constructed first, then use cupva_utils::CopyToPlane to convert into a BL layout,
    // since there is no direct way to construct a BL layout
    NVCVImageHandle inPlImageHandle   = NULL;
    NVCVImageHandle inImageHandle     = NULL;
    NVCVImageHandle outImageHandle    = NULL;
    NVCVImageHandle refOutImageHandle = NULL;
    NVCVOperatorHandle operatorHandle = NULL;
    cupvaStream_t stream              = NULL;
    cupvaSyncObj_t sync               = NULL;
    int32_t err                       = 0;

    NVCVImageRequirements imageRequirements;
    // use either NVCV_IMAGE_FORMAT_NV12 or NVCV_IMAGE_FORMAT_NV12_ER, instead of NVCV_IMAGE_FORMAT_NV12_BL or
    // NVCV_IMAGE_FORMAT_NV12_ER_BL, since nvcvImageConstruct does not support the latter two formats
    NVCV_CHECK_ERROR_GOTO(
        nvcvImageCalcRequirementsPva(imgWidth, imgHeight, NVCV_IMAGE_FORMAT_NV12, 0, 0, &imageRequirements), err,
        FreeResources);

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(&allocatorHandle), err, FreeResources);

    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(&imageRequirements, allocatorHandle, &inPlImageHandle), err,
                          FreeResources);

    // fill its content
    generateRandomPlInput(inPlImageHandle, 1);

    // generate a NVCV image with the BL layout
    NVCVImageRequirements imageRequirementsBL = {imgWidth, imgHeight, NVCV_IMAGE_FORMAT_NV12_BL};
    NVCV_CHECK_ERROR_GOTO(nvcvImageConstructBL(&imageRequirementsBL, &inImageHandle, NULL), err, FreeResources);

    // convert from the PL layout into the BL layout
    plToBl(inPlImageHandle, inImageHandle);

    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(&imageRequirements, allocatorHandle, &outImageHandle), err, FreeResources);

    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(&imageRequirements, allocatorHandle, &refOutImageHandle), err,
                          FreeResources);

    NVCV_CHECK_ERROR_GOTO(pvaBlToPlCreate(&operatorHandle, &imageRequirements), err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaSyncObjCreate(&sync, false, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD), err,
                           FreeResources);

    cupvaFence_t fence;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceInit(&fence, sync), err, FreeResources);

    cupvaCmd_t rf;
    CUPVA_CHECK_ERROR_GOTO(CupvaCmdRequestFencesInit(&rf, &fence, 1), err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamCreate(&stream, CUPVA_PVA0, CUPVA_VPU_ANY), err, FreeResources);

    cupvaCmd_t const *cmds[1]  = {&rf};
    cupvaCmdStatus_t status[1] = {NULL};

    NVCV_CHECK_ERROR_GOTO(pvaBlToPlSubmit(operatorHandle, stream, inImageHandle, outImageHandle), err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    blToPl(inImageHandle, refOutImageHandle);

    err = checkBltoplOutput(refOutImageHandle, outImageHandle);
    if (err)
    {
        printf("Submit to cuPVA stream test failed\n");
    }

FreeResources: /* clean up all allocated resources */
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvImageDecRef(inImageHandle, NULL);
    nvcvImageDecRef(inPlImageHandle, NULL);
    nvcvImageDecRef(outImageHandle, NULL);
    nvcvImageDecRef(refOutImageHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char **argv)
{
    /* Parameter settings */
    const int32_t imgWidth  = 256;
    const int32_t imgHeight = 128;

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
        cudaStreamSubmitTestResult = SubmitBlToPlToCudaStream(imgWidth, imgHeight);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult = SubmitBlToPlToCupvaStream(imgWidth, imgHeight);

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
