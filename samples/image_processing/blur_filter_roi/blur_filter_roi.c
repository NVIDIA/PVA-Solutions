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
#include "ref/blur_filter_roi_ref.h"

#include <BlockLinear.h>
#include <OpBlurFilterROI.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Fwd.h>
#include <nvcv/Image.h>
#include <nvcv/alloc/Allocator.h>
#include <string.h>

/**
 * @brief Allocate and initialize images and tensors
 */
static int AllocateAndInitializeResources(
    int32_t imgWidth, int32_t imgHeight, NVCVImageFormat inImageFormat, NVCVImageFormat outImageFormat,
    NVCVRectI *rectArray, int32_t numRects, NVCVAllocatorHandle *allocatorHandle, NVCVImageHandle *inImagePLHandle,
    NVCVImageHandle *inImageBLHandle, NVCVImageHandle *outImagePLHandle, NVCVImageHandle *outImageBLHandle,
    NVCVImageHandle *refImageHandle, NVCVTensorHandle *rectTensorHandle, NVCVImageRequirements *inImageRequirements,
    NVCVImageRequirements *outImageRequirements, NVCVTensorRequirements *rectRequirements)
{
    int32_t err = 0;
    NVCVMemLayout inLayout, outLayout;

    nvcvImageFormatGetMemLayout(inImageFormat, &inLayout);
    nvcvImageFormatGetMemLayout(outImageFormat, &outLayout);

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvImageCalcRequirementsPva(imgWidth, imgHeight, inImageFormat, 0, 0, inImageRequirements),
                          err, Error);

    if (inLayout == NVCV_MEM_LAYOUT_BL)
    {
        NVCV_CHECK_ERROR_GOTO(nvcvImageConstructBL(inImageRequirements, inImageBLHandle, NULL), err, Error);

        NVCVImageRequirements inImagePLRequirements;
        memcpy(&inImagePLRequirements, inImageRequirements, sizeof(NVCVImageRequirements));
        nvcvImageFormatSetMemLayout(&inImagePLRequirements.format, NVCV_MEM_LAYOUT_PL);
        NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(&inImagePLRequirements, *allocatorHandle, inImagePLHandle), err,
                              Error);
    }
    else
    {
        NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(inImageRequirements, *allocatorHandle, inImagePLHandle), err, Error);
    }

    NVCV_CHECK_ERROR_GOTO(nvcvImageCalcRequirementsPva(imgWidth, imgHeight, outImageFormat, 0, 0, outImageRequirements),
                          err, Error);

    if (outLayout == NVCV_MEM_LAYOUT_BL)
    {
        NVCV_CHECK_ERROR_GOTO(nvcvImageConstructBL(outImageRequirements, outImageBLHandle, NULL), err, Error);

        NVCVImageRequirements outImagePLRequirements;
        memcpy(&outImagePLRequirements, outImageRequirements, sizeof(NVCVImageRequirements));
        nvcvImageFormatSetMemLayout(&outImagePLRequirements.format, NVCV_MEM_LAYOUT_PL);
        NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(&outImagePLRequirements, *allocatorHandle, outImagePLHandle), err,
                              Error);
        NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(&outImagePLRequirements, *allocatorHandle, refImageHandle), err,
                              Error);
    }
    else
    {
        NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(outImageRequirements, *allocatorHandle, outImagePLHandle), err, Error);
        NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(outImageRequirements, *allocatorHandle, refImageHandle), err, Error);
    }

    generateBlurFilterROIInput(*inImagePLHandle);

    if (inLayout == NVCV_MEM_LAYOUT_BL)
    {
        plToBl(*inImagePLHandle, *inImageBLHandle);
    }

    NVCVTensorLayout rectTensorLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("W", &rectTensorLayout), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(1, (int64_t[]){numRects}, NVCV_DATA_TYPE_4S32, rectTensorLayout,
                                                        0, 0, rectRequirements),
                          err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(rectRequirements, *allocatorHandle, rectTensorHandle), err, Error);

    memcpyToTensor(*rectTensorHandle, rectArray, numRects * sizeof(NVCVRectI));

    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(NVCVImageHandle inImagePLHandle, NVCVImageHandle refImageHandle,
                          NVCVImageHandle outImagePLHandle, NVCVImageHandle outImageBLHandle, NVCVRectI *rectArray,
                          int32_t numRects, NVCVSize2D blockSize, NVCVMemLayout outLayout, const char *testName)
{
    blurFilterROIRefC(inImagePLHandle, refImageHandle, rectArray, numRects, blockSize);

    if (outLayout == NVCV_MEM_LAYOUT_BL)
    {
        blToPl(outImageBLHandle, outImagePLHandle);
    }

    int32_t err = checkBlurFilterROIOutput(refImageHandle, outImagePLHandle);
    if (err)
    {
        printf("%s test failed\n", testName);
    }
    return err;
}

/**
 * @brief Test blur filter ROI operation using CUDA stream
 */
int SubmitBlurFilterROIToCudaStream(int32_t imgWidth, int32_t imgHeight, NVCVImageFormat inImageFormat,
                                    NVCVImageFormat outImageFormat, NVCVRectI *rectArray, int32_t numRects,
                                    NVCVSize2D blockSize)
{
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVImageHandle inImagePLHandle     = NULL;
    NVCVImageHandle inImageBLHandle     = NULL;
    NVCVImageHandle outImagePLHandle    = NULL;
    NVCVImageHandle outImageBLHandle    = NULL;
    NVCVImageHandle refImageHandle      = NULL;
    NVCVTensorHandle rectTensorHandle   = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cudaStream_t stream                 = NULL;
    int32_t err                         = 0;

    NVCVImageRequirements inImageRequirements;
    NVCVImageRequirements outImageRequirements;
    NVCVTensorRequirements rectRequirements;

    NVCVMemLayout inLayout, outLayout;
    nvcvImageFormatGetMemLayout(inImageFormat, &inLayout);
    nvcvImageFormatGetMemLayout(outImageFormat, &outLayout);

    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    /* Allocate and initialize resources */
    err = AllocateAndInitializeResources(imgWidth, imgHeight, inImageFormat, outImageFormat, rectArray, numRects,
                                         &allocatorHandle, &inImagePLHandle, &inImageBLHandle, &outImagePLHandle,
                                         &outImageBLHandle, &refImageHandle, &rectTensorHandle, &inImageRequirements,
                                         &outImageRequirements, &rectRequirements);
    if (err)
        goto FreeResources;

    /* Select the correct image handles based on layout */
    NVCVImageHandle inImageHandle  = (inLayout == NVCV_MEM_LAYOUT_BL) ? inImageBLHandle : inImagePLHandle;
    NVCVImageHandle outImageHandle = (outLayout == NVCV_MEM_LAYOUT_BL) ? outImageBLHandle : outImagePLHandle;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaBlurFilterROICreate(&operatorHandle, &inImageRequirements, &outImageRequirements,
                                                 &rectRequirements, blockSize),
                          err, FreeResources);

    /* Submit to CUDA stream */
    NVCV_CHECK_ERROR_GOTO(
        pvaBlurFilterROISubmit(operatorHandle, stream, inImageHandle, outImageHandle, rectTensorHandle, numRects), err,
        FreeResources);

    /* Synchronize stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(inImagePLHandle, refImageHandle, outImagePLHandle, outImageBLHandle, rectArray, numRects,
                         blockSize, outLayout, "Submit to CUDA stream");

FreeResources:
    cudaStreamDestroy(stream);
    nvcvImageDecRef(inImagePLHandle, NULL);
    nvcvImageDecRef(inImageBLHandle, NULL);
    nvcvImageDecRef(outImagePLHandle, NULL);
    nvcvImageDecRef(outImageBLHandle, NULL);
    nvcvImageDecRef(refImageHandle, NULL);
    nvcvTensorDecRef(rectTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

/**
 * @brief Test blur filter ROI operation using cuPVA stream
 */
int SubmitBlurFilterROIToCupvaStream(int32_t imgWidth, int32_t imgHeight, NVCVImageFormat inImageFormat,
                                     NVCVImageFormat outImageFormat, NVCVRectI *rectArray, int32_t numRects,
                                     NVCVSize2D blockSize)
{
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVImageHandle inImagePLHandle     = NULL;
    NVCVImageHandle inImageBLHandle     = NULL;
    NVCVImageHandle outImagePLHandle    = NULL;
    NVCVImageHandle outImageBLHandle    = NULL;
    NVCVImageHandle refImageHandle      = NULL;
    NVCVTensorHandle rectTensorHandle   = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cupvaSyncObj_t sync                 = NULL;
    cupvaStream_t stream                = NULL;
    int32_t err                         = 0;

    NVCVImageRequirements inImageRequirements;
    NVCVImageRequirements outImageRequirements;
    NVCVTensorRequirements rectRequirements;

    NVCVMemLayout inLayout, outLayout;
    nvcvImageFormatGetMemLayout(inImageFormat, &inLayout);
    nvcvImageFormatGetMemLayout(outImageFormat, &outLayout);

    /* Allocate and initialize resources */
    err = AllocateAndInitializeResources(imgWidth, imgHeight, inImageFormat, outImageFormat, rectArray, numRects,
                                         &allocatorHandle, &inImagePLHandle, &inImageBLHandle, &outImagePLHandle,
                                         &outImageBLHandle, &refImageHandle, &rectTensorHandle, &inImageRequirements,
                                         &outImageRequirements, &rectRequirements);
    if (err)
        goto FreeResources;

    /* Select the correct image handles based on layout */
    NVCVImageHandle inImageHandle  = (inLayout == NVCV_MEM_LAYOUT_BL) ? inImageBLHandle : inImagePLHandle;
    NVCVImageHandle outImageHandle = (outLayout == NVCV_MEM_LAYOUT_BL) ? outImageBLHandle : outImagePLHandle;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaBlurFilterROICreate(&operatorHandle, &inImageRequirements, &outImageRequirements,
                                                 &rectRequirements, blockSize),
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
    cupvaCmd_t const *cmds[1];
    cmds[0] = &rf;
    cupvaCmdStatus_t status[1];
    status[0] = NULL;

    NVCV_CHECK_ERROR_GOTO(
        pvaBlurFilterROISubmit(operatorHandle, stream, inImageHandle, outImageHandle, rectTensorHandle, numRects), err,
        FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    /* Wait for completion */
    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(inImagePLHandle, refImageHandle, outImagePLHandle, outImageBLHandle, rectArray, numRects,
                         blockSize, outLayout, "Submit to cuPVA stream");

FreeResources:
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvImageDecRef(inImagePLHandle, NULL);
    nvcvImageDecRef(inImageBLHandle, NULL);
    nvcvImageDecRef(outImagePLHandle, NULL);
    nvcvImageDecRef(outImageBLHandle, NULL);
    nvcvImageDecRef(refImageHandle, NULL);
    nvcvTensorDecRef(rectTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char **argv)
{
    const int32_t imgWidth         = 640;
    const int32_t imgHeight        = 480;
    NVCVImageFormat inImageFormat  = NVCV_IMAGE_FORMAT_NV12_BL;
    NVCVImageFormat outImageFormat = NVCV_IMAGE_FORMAT_NV12;

    NVCVRectI rectArray[] = {{32, 32, 32, 32}, {48, 48, 64, 64}, {64, 64, 128, 128}};
    int32_t numRects      = sizeof(rectArray) / sizeof(rectArray[0]);

    NVCVSize2D blockSize = {16, 16};

    int32_t cudaStreamSubmitTestResult  = 0;
    int32_t cupvaStreamSubmitTestResult = 0;

#if CUPVA_HAS_CUDA_STREAM_SUPPORT == 1

    CUDA_CHECK_ERROR_GOTO(cudaSetDevice(0), cudaStreamSubmitTestResult, CheckResults);

    uint32_t driverVersion;
    CUPVA_CHECK_ERROR_GOTO(CupvaGetDriverVersion(&driverVersion), cudaStreamSubmitTestResult, CheckResults);
    if (driverVersion >= 2007)
    {
        cudaStreamSubmitTestResult = SubmitBlurFilterROIToCudaStream(imgWidth, imgHeight, inImageFormat, outImageFormat,
                                                                     rectArray, numRects, blockSize);
    }
#endif

    cupvaStreamSubmitTestResult = SubmitBlurFilterROIToCupvaStream(imgWidth, imgHeight, inImageFormat, outImageFormat,
                                                                   rectArray, numRects, blockSize);

CheckResults:
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
