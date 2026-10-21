/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "ref/contour_tracing_ref.h"

#include <OpContourTracing.h>
#include <PvaAllocator.h>

#define MAX_RECT_NUM 32
#define MAX_RECT_WIDTH 128
#define MAX_RECT_HEIGHT 128
#define MAX_RECT_AREA (MAX_RECT_WIDTH * MAX_RECT_HEIGHT)

/**
 * @brief Allocate and initialize tensors
 */
static int AllocateAndInitializeResources(
    int32_t imgWidth, int32_t imgHeight, NVCVRectI *rectArray, int32_t numRects, NVCVAllocatorHandle *allocatorHandle,
    NVCVTensorHandle *inTensorHandle, NVCVTensorHandle *rectTensorHandle, NVCVTensorHandle *refOutTensorHandle,
    NVCVTensorHandle *pvaOutTensorHandle, NVCVTensorHandle *refLenTensorHandle, NVCVTensorHandle *pvaLenTensorHandle,
    NVCVTensorRequirements *inTensorRequirements, NVCVTensorRequirements *rectTensorRequirements,
    NVCVTensorRequirements *outTensorRequirements, NVCVTensorRequirements *lenTensorRequirements)
{
    int32_t err        = 0;
    int32_t tensorRank = 3;
    NVCVTensorLayout tensorLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HWC", &tensorLayout), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);

    /* Input tensor: HWC {imgHeight, imgWidth, 1}, U8 */
    int64_t inTensorShape[] = {imgHeight, imgWidth, 1};
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank, inTensorShape, NVCV_DATA_TYPE_U8, tensorLayout, 0,
                                                        0, inTensorRequirements),
                          err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(inTensorRequirements, *allocatorHandle, inTensorHandle), err, Error);

    /* Rect tensor: W {MAX_RECT_NUM}, 4S32 */
    NVCVTensorLayout rectTensorLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("W", &rectTensorLayout), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(1, (int64_t[]){MAX_RECT_NUM}, NVCV_DATA_TYPE_4S32,
                                                        rectTensorLayout, 0, 0, rectTensorRequirements),
                          err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(rectTensorRequirements, *allocatorHandle, rectTensorHandle), err, Error);
    contourTracingMemcpyToTensor(*rectTensorHandle, rectArray, numRects * sizeof(NVCVRectI));

    /* Output tensor: HWC {MAX_RECT_NUM, MAX_RECT_AREA, 2}, U16 */
    int64_t outTensorShape[] = {MAX_RECT_NUM, MAX_RECT_AREA, 2};
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank, outTensorShape, NVCV_DATA_TYPE_U16, tensorLayout, 0,
                                                        0, outTensorRequirements),
                          err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(outTensorRequirements, *allocatorHandle, refOutTensorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(outTensorRequirements, *allocatorHandle, pvaOutTensorHandle), err, Error);

    /* Len tensor: W {MAX_RECT_NUM}, U16 */
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(1, (int64_t[]){MAX_RECT_NUM}, NVCV_DATA_TYPE_U16,
                                                        rectTensorLayout, 0, 0, lenTensorRequirements),
                          err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(lenTensorRequirements, *allocatorHandle, refLenTensorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(lenTensorRequirements, *allocatorHandle, pvaLenTensorHandle), err, Error);

    /* Generate input binary image */
    generateContourTracingInput(*inTensorHandle, rectArray, numRects);

    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(NVCVTensorHandle inTensorHandle, const NVCVRectI *rectArray, int32_t numRects,
                          NVCVTensorHandle refOutTensorHandle, NVCVTensorHandle pvaOutTensorHandle,
                          NVCVTensorHandle refLenTensorHandle, NVCVTensorHandle pvaLenTensorHandle,
                          const char *testName)
{
    contourTracingRef(inTensorHandle, rectArray, numRects, refOutTensorHandle, refLenTensorHandle);
    int32_t err = checkContourTracingOutput(refOutTensorHandle, pvaOutTensorHandle, refLenTensorHandle,
                                            pvaLenTensorHandle, numRects);
    if (err)
    {
        printf("%s test failed\n", testName);
    }
    return err;
}

/**
 * @brief Test contour tracing operation using CUDA stream
 */
int SubmitContourTracingToCudaStream(int32_t imgWidth, int32_t imgHeight, NVCVRectI *rectArray, int32_t numRects)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVTensorHandle inTensorHandle     = NULL;
    NVCVTensorHandle rectTensorHandle   = NULL;
    NVCVTensorHandle refOutTensorHandle = NULL;
    NVCVTensorHandle pvaOutTensorHandle = NULL;
    NVCVTensorHandle refLenTensorHandle = NULL;
    NVCVTensorHandle pvaLenTensorHandle = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cudaStream_t stream                 = NULL;
    int32_t err                         = 0;

    NVCVTensorRequirements inTensorRequirements;
    NVCVTensorRequirements rectTensorRequirements;
    NVCVTensorRequirements outTensorRequirements;
    NVCVTensorRequirements lenTensorRequirements;

    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeResources(imgWidth, imgHeight, rectArray, numRects, &allocatorHandle, &inTensorHandle,
                                         &rectTensorHandle, &refOutTensorHandle, &pvaOutTensorHandle,
                                         &refLenTensorHandle, &pvaLenTensorHandle, &inTensorRequirements,
                                         &rectTensorRequirements, &outTensorRequirements, &lenTensorRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaContourTracingCreate(&operatorHandle, &inTensorRequirements, &rectTensorRequirements,
                                                  &outTensorRequirements, &lenTensorRequirements),
                          err, FreeResources);

    /* Submit to CUDA stream */
    NVCV_CHECK_ERROR_GOTO(pvaContourTracingSubmit(operatorHandle, stream, inTensorHandle, rectTensorHandle,
                                                  pvaOutTensorHandle, pvaLenTensorHandle, numRects),
                          err, FreeResources);

    /* Synchronize stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(inTensorHandle, rectArray, numRects, refOutTensorHandle, pvaOutTensorHandle,
                         refLenTensorHandle, pvaLenTensorHandle, "Submit to CUDA stream");

FreeResources:
    cudaStreamDestroy(stream);
    nvcvTensorDecRef(inTensorHandle, NULL);
    nvcvTensorDecRef(rectTensorHandle, NULL);
    nvcvTensorDecRef(refOutTensorHandle, NULL);
    nvcvTensorDecRef(pvaOutTensorHandle, NULL);
    nvcvTensorDecRef(refLenTensorHandle, NULL);
    nvcvTensorDecRef(pvaLenTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

/**
* @brief Test contour tracing operation using cuPVA stream
*/
int SubmitContourTracingToCupvaStream(int32_t imgWidth, int32_t imgHeight, NVCVRectI *rectArray, int32_t numRects)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVTensorHandle inTensorHandle     = NULL;
    NVCVTensorHandle rectTensorHandle   = NULL;
    NVCVTensorHandle refOutTensorHandle = NULL;
    NVCVTensorHandle pvaOutTensorHandle = NULL;
    NVCVTensorHandle refLenTensorHandle = NULL;
    NVCVTensorHandle pvaLenTensorHandle = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cupvaSyncObj_t sync                 = NULL;
    cupvaStream_t stream                = NULL;
    int32_t err                         = 0;

    NVCVTensorRequirements inTensorRequirements;
    NVCVTensorRequirements rectTensorRequirements;
    NVCVTensorRequirements outTensorRequirements;
    NVCVTensorRequirements lenTensorRequirements;

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeResources(imgWidth, imgHeight, rectArray, numRects, &allocatorHandle, &inTensorHandle,
                                         &rectTensorHandle, &refOutTensorHandle, &pvaOutTensorHandle,
                                         &refLenTensorHandle, &pvaLenTensorHandle, &inTensorRequirements,
                                         &rectTensorRequirements, &outTensorRequirements, &lenTensorRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaContourTracingCreate(&operatorHandle, &inTensorRequirements, &rectTensorRequirements,
                                                  &outTensorRequirements, &lenTensorRequirements),
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

    NVCV_CHECK_ERROR_GOTO(pvaContourTracingSubmit(operatorHandle, stream, inTensorHandle, rectTensorHandle,
                                                  pvaOutTensorHandle, pvaLenTensorHandle, numRects),
                          err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    /* Wait for completion */
    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(inTensorHandle, rectArray, numRects, refOutTensorHandle, pvaOutTensorHandle,
                         refLenTensorHandle, pvaLenTensorHandle, "Submit to cuPVA stream");

FreeResources:
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvTensorDecRef(inTensorHandle, NULL);
    nvcvTensorDecRef(rectTensorHandle, NULL);
    nvcvTensorDecRef(refOutTensorHandle, NULL);
    nvcvTensorDecRef(pvaOutTensorHandle, NULL);
    nvcvTensorDecRef(refLenTensorHandle, NULL);
    nvcvTensorDecRef(pvaLenTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char **argv)
{
    /* Parameter settings */
    const int32_t imgWidth  = 256;
    const int32_t imgHeight = 256;

    NVCVRectI rectArray[] = {{0, 0, 70, 70}, {70, 70, 128, 128}, {200, 200, 20, 20}};
    int32_t numRects      = sizeof(rectArray) / sizeof(rectArray[0]);

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
        cudaStreamSubmitTestResult = SubmitContourTracingToCudaStream(imgWidth, imgHeight, rectArray, numRects);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult = SubmitContourTracingToCupvaStream(imgWidth, imgHeight, rectArray, numRects);

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
