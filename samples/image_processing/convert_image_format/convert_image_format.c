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
#include "ref/convert_image_format_ref.h"

#include <OpConvertImageFormat.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Fwd.h>
#include <nvcv/Image.h>
#include <nvcv/alloc/Allocator.h>

/**
 * @brief Allocate and initialize images
 */
static int AllocateAndInitializeImages(int32_t imgWidth, int32_t imgHeight, NVCVImageFormat inImageFormat,
                                       NVCVImageFormat outImageFormat, NVCVAllocatorHandle *allocatorHandle,
                                       NVCVImageHandle *inImageHandle, NVCVImageHandle *outImageHandle,
                                       NVCVImageHandle *refImageHandle, NVCVImageRequirements *inImageRequirements,
                                       NVCVImageRequirements *outImageRequirements)
{
    int32_t err = 0;

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvImageCalcRequirementsPva(imgWidth, imgHeight, inImageFormat, 0, 0, inImageRequirements),
                          err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(inImageRequirements, *allocatorHandle, inImageHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvImageCalcRequirementsPva(imgWidth, imgHeight, outImageFormat, 0, 0, outImageRequirements),
                          err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(outImageRequirements, *allocatorHandle, outImageHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(outImageRequirements, *allocatorHandle, refImageHandle), err, Error);

    generateConvertImageFormatInput(*inImageHandle);

    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(NVCVImageHandle inImageHandle, NVCVImageHandle refImageHandle, NVCVImageHandle outImageHandle,
                          const char *testName)
{
    convertImageFormatRef(inImageHandle, refImageHandle);
    int32_t err = checkConvertImageFormatOutput(refImageHandle, outImageHandle);
    if (err)
    {
        printf("%s test failed\n", testName);
    }
    return err;
}

/**
 * @brief Test convert image format operation using CUDA stream
 */
int SubmitConvertImageFormatToCudaStream(int32_t imgWidth, int32_t imgHeight, NVCVImageFormat inImageFormat,
                                         NVCVImageFormat outImageFormat)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVImageHandle inImageHandle       = NULL;
    NVCVImageHandle outImageHandle      = NULL;
    NVCVImageHandle refImageHandle      = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cudaStream_t stream                 = NULL;
    int32_t err                         = 0;

    NVCVImageRequirements inImageRequirements;
    NVCVImageRequirements outImageRequirements;

    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    /* Allocate and initialize images */
    err = AllocateAndInitializeImages(imgWidth, imgHeight, inImageFormat, outImageFormat, &allocatorHandle,
                                      &inImageHandle, &outImageHandle, &refImageHandle, &inImageRequirements,
                                      &outImageRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaConvertImageFormatCreate(&operatorHandle, &inImageRequirements, &outImageRequirements),
                          err, FreeResources);

    /* Submit to CUDA stream */
    NVCV_CHECK_ERROR_GOTO(pvaConvertImageFormatSubmit(operatorHandle, stream, inImageHandle, outImageHandle), err,
                          FreeResources);

    /* Synchronize stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(inImageHandle, refImageHandle, outImageHandle, "Submit to CUDA stream");

FreeResources: /* clean up all allocated resources */
    cudaStreamDestroy(stream);
    nvcvImageDecRef(inImageHandle, NULL);
    nvcvImageDecRef(outImageHandle, NULL);
    nvcvImageDecRef(refImageHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

/**
 * @brief Test convert image format operation using cuPVA stream
 */
int SubmitConvertImageFormatToCupvaStream(int32_t imgWidth, int32_t imgHeight, NVCVImageFormat inImageFormat,
                                          NVCVImageFormat outImageFormat)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVImageHandle inImageHandle       = NULL;
    NVCVImageHandle outImageHandle      = NULL;
    NVCVImageHandle refImageHandle      = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cupvaSyncObj_t sync                 = NULL;
    cupvaStream_t stream                = NULL;
    int32_t err                         = 0;

    NVCVImageRequirements inImageRequirements;
    NVCVImageRequirements outImageRequirements;

    /* Allocate and initialize images */
    err = AllocateAndInitializeImages(imgWidth, imgHeight, inImageFormat, outImageFormat, &allocatorHandle,
                                      &inImageHandle, &outImageHandle, &refImageHandle, &inImageRequirements,
                                      &outImageRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaConvertImageFormatCreate(&operatorHandle, &inImageRequirements, &outImageRequirements),
                          err, FreeResources);

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

    NVCV_CHECK_ERROR_GOTO(pvaConvertImageFormatSubmit(operatorHandle, stream, inImageHandle, outImageHandle), err,
                          FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    /* Wait for completion */
    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(inImageHandle, refImageHandle, outImageHandle, "Submit to cuPVA stream");

FreeResources: /* clean up all allocated resources */
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvImageDecRef(inImageHandle, NULL);
    nvcvImageDecRef(outImageHandle, NULL);
    nvcvImageDecRef(refImageHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char **argv)
{
    /* Parameter settings */
    const int32_t imgWidth         = 1920;
    const int32_t imgHeight        = 1080;
    NVCVImageFormat inImageFormat  = NVCV_IMAGE_FORMAT_NV12;
    NVCVImageFormat outImageFormat = NVCV_IMAGE_FORMAT_RGB8;

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
            SubmitConvertImageFormatToCudaStream(imgWidth, imgHeight, inImageFormat, outImageFormat);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult =
        SubmitConvertImageFormatToCupvaStream(imgWidth, imgHeight, inImageFormat, outImageFormat);

    /* Also demonstrate two directly-supported conversions added for full-range
     * NV12 and 4-channel packed formats:
     *   - NV12_ER (full range) -> RGBA8 (single pass, alpha = 255)
     *   - RGBA8 -> RGB8        (single pass, alpha lane dropped) */
    cupvaStreamSubmitTestResult |=
        SubmitConvertImageFormatToCupvaStream(imgWidth, imgHeight, NVCV_IMAGE_FORMAT_NV12_ER, NVCV_IMAGE_FORMAT_RGBA8);
    cupvaStreamSubmitTestResult |=
        SubmitConvertImageFormatToCupvaStream(imgWidth, imgHeight, NVCV_IMAGE_FORMAT_RGBA8, NVCV_IMAGE_FORMAT_RGB8);

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
