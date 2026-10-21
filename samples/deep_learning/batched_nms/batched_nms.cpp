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

/**
 * @file batched_nms.cpp
 * @brief Sample demonstrating the BatchedNMS operator.
 *
 * Boxes are passed in structure-of-arrays form: rows 0 to 3 of the boxes tensors hold the left,
 * top, right and bottom coordinates. Input scores are FP16 and input class identifiers are int16;
 * the operator writes FP32 scores and int32 class identifiers. The output is compared against a
 * CPU reference model.
 */

#include "ErrorCheckMacros.h"
#include "ref/batched_nms_ref.h"

#include <OpBatchedNMS.h>
#include <PvaAllocator.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int32_t kMaxInputBoxes  = 256;
constexpr int32_t kMaxOutputBoxes = 100;
constexpr float kIouThreshold     = 0.45f;
constexpr float kTolerance        = 1e-4f;

/* Boxes 0 and 1 belong to class 0 and overlap, so box 1 is suppressed. Boxes 5 and 6 have class
 * identifiers outside [0, 299] and are ignored. */
constexpr int32_t kNumInputBoxes           = 7;
constexpr float kLeft[kNumInputBoxes]      = {10.f, 12.f, 100.f, 50.f, 200.f, 300.f, 350.f};
constexpr float kTop[kNumInputBoxes]       = {10.f, 12.f, 100.f, 50.f, 200.f, 300.f, 350.f};
constexpr float kRight[kNumInputBoxes]     = {50.f, 52.f, 140.f, 90.f, 240.f, 340.f, 390.f};
constexpr float kBottom[kNumInputBoxes]    = {50.f, 52.f, 140.f, 90.f, 240.f, 340.f, 390.f};
constexpr float kScores[kNumInputBoxes]    = {0.9f, 0.7f, 0.8f, 0.6f, 0.95f, 0.99f, 0.98f};
constexpr int16_t kClasses[kNumInputBoxes] = {0, 0, 1, 2, 0, -1, 300};

struct BatchedNMSRequirements
{
    NVCVTensorRequirements inBoxes, inScores, inClasses, count, outBoxes, outScores, outClasses;
};

struct BatchedNMSTensors
{
    NVCVTensorHandle inBoxes, inScores, inClasses, inCount, outBoxes, outScores, outClasses, outCount;
};

/**
 * @brief Release the tensors. Releasing a NULL handle is a no-op.
 */
void DestroyTensors(BatchedNMSTensors *tensors)
{
    nvcvTensorDecRef(tensors->inBoxes, NULL);
    nvcvTensorDecRef(tensors->inScores, NULL);
    nvcvTensorDecRef(tensors->inClasses, NULL);
    nvcvTensorDecRef(tensors->inCount, NULL);
    nvcvTensorDecRef(tensors->outBoxes, NULL);
    nvcvTensorDecRef(tensors->outScores, NULL);
    nvcvTensorDecRef(tensors->outClasses, NULL);
    nvcvTensorDecRef(tensors->outCount, NULL);
}

/**
 * @brief Get the host address of a tensor's data and, optionally, its outermost stride in bytes.
 */
int32_t MapTensor(NVCVTensorHandle tensor, void **hostPtr, int64_t *outerStride)
{
    int32_t err = 0;
    NVCVTensorData data;

    NVCV_CHECK_ERROR_GOTO(nvcvTensorExportData(tensor, &data), err, Error);
    CUPVA_CHECK_ERROR_GOTO(CupvaMemGetHostPointer(hostPtr, reinterpret_cast<void *>(data.buffer.strided.basePtr)), err,
                           Error);
    if (outerStride != NULL)
    {
        *outerStride = data.buffer.strided.strides[0];
    }

Error:
    return err;
}

/**
 * @brief Write the sample boxes, scores, class identifiers and box count into the input tensors.
 */
int32_t InitializeInputs(const BatchedNMSTensors *tensors)
{
    int32_t err          = 0;
    void *boxesPtr       = NULL;
    void *scoresPtr      = NULL;
    void *classesPtr     = NULL;
    void *countPtr       = NULL;
    int64_t boxRowStride = 0;
    float *boxes         = NULL;
    __half *scores       = NULL;
    int16_t *classes     = NULL;
    int64_t rowPitch     = 0;

    err = MapTensor(tensors->inBoxes, &boxesPtr, &boxRowStride);
    if (err)
        goto Error;
    err = MapTensor(tensors->inScores, &scoresPtr, NULL);
    if (err)
        goto Error;
    err = MapTensor(tensors->inClasses, &classesPtr, NULL);
    if (err)
        goto Error;
    err = MapTensor(tensors->inCount, &countPtr, NULL);
    if (err)
        goto Error;

    boxes    = static_cast<float *>(boxesPtr);
    scores   = static_cast<__half *>(scoresPtr);
    classes  = static_cast<int16_t *>(classesPtr);
    rowPitch = boxRowStride / static_cast<int64_t>(sizeof(float));

    for (int32_t i = 0; i < kNumInputBoxes; ++i)
    {
        boxes[i]                = kLeft[i];
        boxes[rowPitch + i]     = kTop[i];
        boxes[2 * rowPitch + i] = kRight[i];
        boxes[3 * rowPitch + i] = kBottom[i];
        scores[i]               = __float2half(kScores[i]);
        classes[i]              = kClasses[i];
    }
    *static_cast<int32_t *>(countPtr) = kNumInputBoxes;

Error:
    return err;
}

/**
 * @brief Create the allocator, the tensor requirements and the tensors, and initialize the inputs.
 */
int32_t AllocateAndInitializeTensors(NVCVAllocatorHandle *allocatorHandle, BatchedNMSTensors *tensors,
                                     BatchedNMSRequirements *reqs)
{
    int32_t err = 0;
    NVCVTensorLayout layoutHW;
    NVCVTensorLayout layoutW;
    int64_t inBoxesShape[]  = {4, kMaxInputBoxes};
    int64_t inShape[]       = {kMaxInputBoxes};
    int64_t outBoxesShape[] = {4, kMaxOutputBoxes};
    int64_t outShape[]      = {kMaxOutputBoxes};
    int64_t countShape[]    = {1};

    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HW", &layoutHW), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("W", &layoutW), err, Error);

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(2, inBoxesShape, NVCV_DATA_TYPE_F32, layoutHW, 0, 0, &reqs->inBoxes), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(1, inShape, NVCV_DATA_TYPE_F16, layoutW, 0, 0, &reqs->inScores),
                          err, Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(1, inShape, NVCV_DATA_TYPE_S16, layoutW, 0, 0, &reqs->inClasses), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(1, countShape, NVCV_DATA_TYPE_S32, layoutW, 0, 0, &reqs->count),
                          err, Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(2, outBoxesShape, NVCV_DATA_TYPE_F32, layoutHW, 0, 0, &reqs->outBoxes), err,
        Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(1, outShape, NVCV_DATA_TYPE_F32, layoutW, 0, 0, &reqs->outScores), err, Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(1, outShape, NVCV_DATA_TYPE_S32, layoutW, 0, 0, &reqs->outClasses), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&reqs->inBoxes, *allocatorHandle, &tensors->inBoxes), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&reqs->inScores, *allocatorHandle, &tensors->inScores), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&reqs->inClasses, *allocatorHandle, &tensors->inClasses), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&reqs->count, *allocatorHandle, &tensors->inCount), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&reqs->outBoxes, *allocatorHandle, &tensors->outBoxes), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&reqs->outScores, *allocatorHandle, &tensors->outScores), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&reqs->outClasses, *allocatorHandle, &tensors->outClasses), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&reqs->count, *allocatorHandle, &tensors->outCount), err, Error);

    err = InitializeInputs(tensors);

Error:
    return err;
}

/**
 * @brief Create the BatchedNMS operator for the given tensor requirements.
 */
NVCVStatus CreateOperator(NVCVOperatorHandle *operatorHandle, const BatchedNMSRequirements *reqs)
{
    return pvaBatchedNMSCreate(operatorHandle, &reqs->inBoxes, &reqs->inScores, &reqs->inClasses, &reqs->count,
                               &reqs->outBoxes, &reqs->outScores, &reqs->outClasses, &reqs->count, kIouThreshold);
}

/**
 * @brief Validate the operator output against the CPU reference.
 */
int32_t ValidateOutput(const BatchedNMSTensors *tensors, const char *testName)
{
    int32_t err          = 0;
    void *boxesPtr       = NULL;
    void *scoresPtr      = NULL;
    void *classesPtr     = NULL;
    void *countPtr       = NULL;
    int64_t boxRowStride = 0;
    const float *boxes   = NULL;
    const float *scores  = NULL;
    const int32_t *cls   = NULL;
    int64_t rowPitch     = 0;
    int32_t numOut       = 0;
    int32_t mismatches   = 0;
    float refScores[kNumInputBoxes];
    std::vector<BatchedNMSRefBox> ref;

    /* The operator orders FP16 scores, so the reference model receives the FP16-rounded values. */
    for (int32_t i = 0; i < kNumInputBoxes; ++i)
    {
        refScores[i] = __half2float(__float2half(kScores[i]));
    }
    ref = batchedNMSRef(kLeft, kTop, kRight, kBottom, refScores, kClasses, kNumInputBoxes, kIouThreshold,
                        kMaxOutputBoxes);

    err = MapTensor(tensors->outBoxes, &boxesPtr, &boxRowStride);
    if (err)
        goto Error;
    err = MapTensor(tensors->outScores, &scoresPtr, NULL);
    if (err)
        goto Error;
    err = MapTensor(tensors->outClasses, &classesPtr, NULL);
    if (err)
        goto Error;
    err = MapTensor(tensors->outCount, &countPtr, NULL);
    if (err)
        goto Error;

    boxes    = static_cast<const float *>(boxesPtr);
    scores   = static_cast<const float *>(scoresPtr);
    cls      = static_cast<const int32_t *>(classesPtr);
    rowPitch = boxRowStride / static_cast<int64_t>(sizeof(float));
    numOut   = *static_cast<const int32_t *>(countPtr);

    printf("%s: %d of %d boxes kept\n", testName, numOut, kNumInputBoxes);
    if (numOut != static_cast<int32_t>(ref.size()))
    {
        printf("%s: count mismatch, PVA=%d CPU=%d\n", testName, numOut, static_cast<int32_t>(ref.size()));
        err = -1;
        goto Error;
    }

    for (int32_t i = 0; i < numOut; ++i)
    {
        const float left   = boxes[i];
        const float top    = boxes[rowPitch + i];
        const float right  = boxes[2 * rowPitch + i];
        const float bottom = boxes[3 * rowPitch + i];

        printf("  [%d] box=(%.1f, %.1f, %.1f, %.1f) score=%.3f class=%d\n", i, left, top, right, bottom, scores[i],
               cls[i]);

        if (std::fabs(left - ref[i].left) > kTolerance || std::fabs(top - ref[i].top) > kTolerance ||
            std::fabs(right - ref[i].right) > kTolerance || std::fabs(bottom - ref[i].bottom) > kTolerance ||
            std::fabs(scores[i] - ref[i].score) > kTolerance || cls[i] != ref[i].classId)
        {
            printf("  [%d] mismatch, CPU box=(%.1f, %.1f, %.1f, %.1f) score=%.3f class=%d\n", i, ref[i].left,
                   ref[i].top, ref[i].right, ref[i].bottom, ref[i].score, ref[i].classId);
            ++mismatches;
        }
    }

    if (mismatches > 0)
    {
        printf("%s test failed\n", testName);
        err = -1;
    }

Error:
    return err;
}

} // namespace

/**
 * @brief Run the BatchedNMS operator on a CUDA stream.
 */
int32_t SubmitBatchedNMSToCudaStream()
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    BatchedNMSTensors tensors           = {};
    NVCVOperatorHandle operatorHandle   = NULL;
    cudaStream_t stream                 = NULL;
    int32_t err                         = 0;
    cudaError_t cudaErr                 = cudaSuccess;
    NVCVStatus submitStatus             = NVCV_SUCCESS;
    BatchedNMSRequirements reqs;

    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaErr, FreeResources);

    err = AllocateAndInitializeTensors(&allocatorHandle, &tensors, &reqs);
    if (err)
        goto FreeResources;

    NVCV_CHECK_ERROR_GOTO(CreateOperator(&operatorHandle, &reqs), err, FreeResources);

    submitStatus =
        pvaBatchedNMSSubmit(operatorHandle, stream, tensors.inBoxes, tensors.inScores, tensors.inClasses,
                            tensors.inCount, tensors.outBoxes, tensors.outScores, tensors.outClasses, tensors.outCount);
    if (submitStatus == NVCV_ERROR_NOT_IMPLEMENTED)
    {
        printf("Submit to CUDA stream: not supported by the cuPVA runtime, skipped\n");
        goto FreeResources;
    }
    NVCV_CHECK_ERROR_GOTO(submitStatus, err, FreeResources);

    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), cudaErr, FreeResources);

    err = ValidateOutput(&tensors, "Submit to CUDA stream");

FreeResources: /* clean up all allocated resources */
    cudaStreamDestroy(stream);
    DestroyTensors(&tensors);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return (err != 0) ? err : static_cast<int32_t>(cudaErr);
}

/**
 * @brief Run the BatchedNMS operator on a cuPVA stream.
 */
int32_t SubmitBatchedNMSToCupvaStream()
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    BatchedNMSTensors tensors           = {};
    NVCVOperatorHandle operatorHandle   = NULL;
    cupvaSyncObj_t sync                 = NULL;
    cupvaStream_t stream                = NULL;
    int32_t err                         = 0;
    bool waitSuccess                    = false;
    BatchedNMSRequirements reqs;
    cupvaFence_t fence;
    cupvaCmd_t rf;
    cupvaCmd_t const *cmds[1];
    cupvaCmdStatus_t status[1];

    err = AllocateAndInitializeTensors(&allocatorHandle, &tensors, &reqs);
    if (err)
        goto FreeResources;

    NVCV_CHECK_ERROR_GOTO(CreateOperator(&operatorHandle, &reqs), err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaSyncObjCreate(&sync, false, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD), err,
                           FreeResources);
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceInit(&fence, sync), err, FreeResources);
    CUPVA_CHECK_ERROR_GOTO(CupvaCmdRequestFencesInit(&rf, &fence, 1), err, FreeResources);
    CUPVA_CHECK_ERROR_GOTO(CupvaStreamCreate(&stream, CUPVA_PVA0, CUPVA_VPU_ANY), err, FreeResources);

    NVCV_CHECK_ERROR_GOTO(
        pvaBatchedNMSSubmit(operatorHandle, stream, tensors.inBoxes, tensors.inScores, tensors.inClasses,
                            tensors.inCount, tensors.outBoxes, tensors.outScores, tensors.outClasses, tensors.outCount),
        err, FreeResources);

    cmds[0]   = &rf;
    status[0] = NULL;
    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);
    if (!waitSuccess)
    {
        printf("Submit to cuPVA stream: fence wait did not complete\n");
        err = -1;
        goto FreeResources;
    }

    err = ValidateOutput(&tensors, "Submit to cuPVA stream");

FreeResources: /* clean up all allocated resources */
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    DestroyTensors(&tensors);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    int32_t cudaStreamSubmitTestResult  = 0;
    int32_t cupvaStreamSubmitTestResult = 0;

#if CUPVA_HAS_CUDA_STREAM_SUPPORT == 1
    /* Initialize the CUDA context before any cuPVA object is created. */
    uint32_t driverVersion    = 0;
    const cudaError_t cudaErr = cudaSetDevice(0);
    if (cudaErr != cudaSuccess)
    {
        printf("CUDA error: %d in function cudaSetDevice(0)\n", cudaErr);
        printf("Error message: %s\n", cudaGetErrorString(cudaErr));
        cudaStreamSubmitTestResult = -1;
    }
    else if (CupvaGetDriverVersion(&driverVersion) != CUPVA_ERROR_NONE)
    {
        printf("CUPVA error: failed to query the PVA driver version\n");
        cudaStreamSubmitTestResult = -1;
    }
    else if (driverVersion >= 2007)
    {
        /* Submit to CUDA stream if driver version is greater than or equal to 2007 */
        cudaStreamSubmitTestResult = SubmitBatchedNMSToCudaStream();
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult = SubmitBatchedNMSToCupvaStream();

    /* Overall result */
    if (cudaStreamSubmitTestResult || cupvaStreamSubmitTestResult)
    {
        printf("Test Fail\n");
        return 1;
    }

    printf("Test Pass\n");
    return 0;
}
