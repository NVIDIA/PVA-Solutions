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

#include "yolov5_pipeline_tensors.hpp"

#include <ErrorCheckMacros.h> // CUPVA_CHECK_ERROR
#include <PvaAllocator.h>     // nvcvTensorCalcRequirementsPva
#include <cupva_host.h>       // CupvaMemGetHostPointer

#include <cstdio>

#define TENSORS_NVCV_CHECK(call)                                                           \
    do                                                                                     \
    {                                                                                      \
        NVCVStatus status_ = (call);                                                       \
        if (status_ != NVCV_SUCCESS)                                                       \
        {                                                                                  \
            fprintf(stderr, "NVCV error %d at %s:%d\n", (int)status_, __FILE__, __LINE__); \
            return -1;                                                                     \
        }                                                                                  \
    } while (0)

namespace yolov5_pipeline {

namespace {

/// Resolve the PVA host-visible pointer backing a tensor.
template<typename T>
int hostPointerOf(NVCVTensorHandle tensor, T **out)
{
    NVCVTensorData data;
    TENSORS_NVCV_CHECK(nvcvTensorExportData(tensor, &data));
    CUPVA_CHECK_ERROR(
        CupvaMemGetHostPointer(reinterpret_cast<void **>(out), reinterpret_cast<void *>(data.buffer.strided.basePtr)));
    return 0;
}

} // namespace

int PvaPreprocessTensors::recreateSlot(NVCVAllocatorHandle alloc, int slot, int srcWidth, int srcHeight, int lbW,
                                       int lbH, int channels)
{
    if (lbIn[slot] != nullptr)
    {
        nvcvTensorDecRef(lbIn[slot], nullptr);
        lbIn[slot] = nullptr;
    }
    if (outImage[slot] != nullptr)
    {
        nvcvImageDecRef(outImage[slot], nullptr);
        outImage[slot] = nullptr;
    }
    if (inImage[slot] != nullptr)
    {
        nvcvImageDecRef(inImage[slot], nullptr);
        inImage[slot] = nullptr;
    }

    NVCVImageRequirements inReqs, outReqs;
    TENSORS_NVCV_CHECK(nvcvImageCalcRequirementsPva(srcWidth, srcHeight, NVCV_IMAGE_FORMAT_RGB8, 0, 0, &inReqs));
    TENSORS_NVCV_CHECK(nvcvImageCalcRequirementsPva(lbW, lbH, NVCV_IMAGE_FORMAT_RGB8, 0, 0, &outReqs));
    TENSORS_NVCV_CHECK(nvcvImageConstruct(&inReqs, alloc, &inImage[slot]));
    TENSORS_NVCV_CHECK(nvcvImageConstruct(&outReqs, alloc, &outImage[slot]));

    NVCVImageData inData;
    TENSORS_NVCV_CHECK(nvcvImageExportData(inImage[slot], &inData));
    CUPVA_CHECK_ERROR(CupvaMemGetHostPointer(reinterpret_cast<void **>(&inHostPtr[slot]),
                                             reinterpret_cast<void *>(inData.buffer.strided.planes[0].basePtr)));
    inRowStride[slot] = inData.buffer.strided.planes[0].rowStride;

    NVCVImageData outData;
    TENSORS_NVCV_CHECK(nvcvImageExportData(outImage[slot], &outData));

    // The normalize input is a HWC view of the resize output, so the tensor
    // wraps those bytes and adopts their row stride instead of allocating.
    NVCVTensorLayout hwcLayout;
    nvcvTensorLayoutMake("HWC", &hwcLayout);

    NVCVTensorRequirements lbInReqs;
    const int64_t lbInShape[] = {lbH, lbW, channels};
    TENSORS_NVCV_CHECK(nvcvTensorCalcRequirementsPva(3, lbInShape, NVCV_DATA_TYPE_U8, hwcLayout, 0, 0, &lbInReqs));
    lbInReqs.strides[0] = outData.buffer.strided.planes[0].rowStride;

    NVCVTensorData lbInData = {};
    lbInData.dtype          = lbInReqs.dtype;
    lbInData.layout         = lbInReqs.layout;
    lbInData.rank           = lbInReqs.rank;
    lbInData.bufferType     = NVCV_TENSOR_BUFFER_STRIDED_CUDA;
    for (int i = 0; i < lbInReqs.rank; ++i)
    {
        lbInData.shape[i]                  = lbInReqs.shape[i];
        lbInData.buffer.strided.strides[i] = lbInReqs.strides[i];
    }
    lbInData.buffer.strided.basePtr = outData.buffer.strided.planes[0].basePtr;
    TENSORS_NVCV_CHECK(nvcvTensorWrapDataConstruct(&lbInData, nullptr, nullptr, &lbIn[slot]));

    srcW[slot] = srcWidth;
    srcH[slot] = srcHeight;
    return 0;
}

int PvaPreprocessTensors::outRowStride(int slot, int64_t *stride) const
{
    NVCVImageData data;
    TENSORS_NVCV_CHECK(nvcvImageExportData(outImage[slot], &data));
    *stride = data.buffer.strided.planes[0].rowStride;
    return 0;
}

void PvaPreprocessTensors::release()
{
    for (int s = 0; s < NUM_PIPE; s++)
    {
        if (lbIn[s] != nullptr)
        {
            nvcvTensorDecRef(lbIn[s], nullptr);
            lbIn[s] = nullptr;
        }
        if (outImage[s] != nullptr)
        {
            nvcvImageDecRef(outImage[s], nullptr);
            outImage[s] = nullptr;
        }
        if (inImage[s] != nullptr)
        {
            nvcvImageDecRef(inImage[s], nullptr);
            inImage[s] = nullptr;
        }
    }
}

int PvaAnchorTensors::allocate(NVCVAllocatorHandle alloc, int scale, int numAnchors, NVCVTensorRequirements *reqsOut)
{
    NVCVTensorLayout layout2d;
    nvcvTensorLayoutMake("HW", &layout2d);

    NVCVTensorRequirements reqs = {};
    const int64_t shape[]       = {numAnchors, 5};
    TENSORS_NVCV_CHECK(nvcvTensorCalcRequirementsPva(2, shape, NVCV_DATA_TYPE_F32, layout2d, 0, 0, &reqs));
    TENSORS_NVCV_CHECK(nvcvTensorConstruct(&reqs, alloc, &anchors[scale]));

    if (hostPointerOf(anchors[scale], &anchorHost[scale]) != 0)
        return -1;

    if (reqsOut != nullptr)
        *reqsOut = reqs;
    return 0;
}

void PvaAnchorTensors::release()
{
    for (int sc = NUM_YOLO_SCALES - 1; sc >= 0; --sc)
    {
        if (anchors[sc] != nullptr)
        {
            nvcvTensorDecRef(anchors[sc], nullptr);
            anchors[sc] = nullptr;
        }
    }
}

int PvaDetectionTensors::allocateDecode(NVCVAllocatorHandle alloc, int maxDecodeBoxes)
{
    NVCVTensorLayout layout2d, layoutW;
    nvcvTensorLayoutMake("HW", &layout2d);
    nvcvTensorLayoutMake("W", &layoutW);

    const int64_t boxesShape[]   = {4, maxDecodeBoxes};
    const int64_t scoresShape[]  = {maxDecodeBoxes};
    const int64_t classesShape[] = {maxDecodeBoxes};
    const int64_t countShape[]   = {1};

    TENSORS_NVCV_CHECK(nvcvTensorCalcRequirementsPva(2, boxesShape, NVCV_DATA_TYPE_F32, layout2d, 0, 0, &decBoxesReqs));
    TENSORS_NVCV_CHECK(
        nvcvTensorCalcRequirementsPva(1, scoresShape, NVCV_DATA_TYPE_F16, layoutW, 0, 0, &decScoresReqs));
    TENSORS_NVCV_CHECK(
        nvcvTensorCalcRequirementsPva(1, classesShape, NVCV_DATA_TYPE_S16, layoutW, 0, 0, &decClassesReqs));
    TENSORS_NVCV_CHECK(nvcvTensorCalcRequirementsPva(1, countShape, NVCV_DATA_TYPE_S32, layoutW, 0, 0, &decCountReqs));

    for (int s = 0; s < NUM_PIPE; s++)
    {
        TENSORS_NVCV_CHECK(nvcvTensorConstruct(&decBoxesReqs, alloc, &decBoxes[s]));
        TENSORS_NVCV_CHECK(nvcvTensorConstruct(&decScoresReqs, alloc, &decScores[s]));
        TENSORS_NVCV_CHECK(nvcvTensorConstruct(&decClassesReqs, alloc, &decClasses[s]));
        TENSORS_NVCV_CHECK(nvcvTensorConstruct(&decCountReqs, alloc, &decCount[s]));

        if (hostPointerOf(decBoxes[s], &decBoxesHost[s]) != 0 || hostPointerOf(decScores[s], &decScoresHost[s]) != 0 ||
            hostPointerOf(decClasses[s], &decClassesHost[s]) != 0 || hostPointerOf(decCount[s], &decCountHost[s]) != 0)
            return -1;
    }

    return 0;
}

int PvaDetectionTensors::allocateNms(NVCVAllocatorHandle alloc, int maxNmsBoxes)
{
    NVCVTensorLayout layout2d, layoutW;
    nvcvTensorLayoutMake("HW", &layout2d);
    nvcvTensorLayoutMake("W", &layoutW);

    const int64_t boxesShape[]   = {4, maxNmsBoxes};
    const int64_t scoresShape[]  = {maxNmsBoxes};
    const int64_t classesShape[] = {maxNmsBoxes};
    const int64_t countShape[]   = {1};

    TENSORS_NVCV_CHECK(nvcvTensorCalcRequirementsPva(2, boxesShape, NVCV_DATA_TYPE_F32, layout2d, 0, 0, &nmsBoxesReqs));
    TENSORS_NVCV_CHECK(
        nvcvTensorCalcRequirementsPva(1, scoresShape, NVCV_DATA_TYPE_F32, layoutW, 0, 0, &nmsScoresReqs));
    TENSORS_NVCV_CHECK(
        nvcvTensorCalcRequirementsPva(1, classesShape, NVCV_DATA_TYPE_S32, layoutW, 0, 0, &nmsClassesReqs));
    TENSORS_NVCV_CHECK(nvcvTensorCalcRequirementsPva(1, countShape, NVCV_DATA_TYPE_S32, layoutW, 0, 0, &nmsCountReqs));

    for (int s = 0; s < NUM_PIPE; s++)
    {
        TENSORS_NVCV_CHECK(nvcvTensorConstruct(&nmsBoxesReqs, alloc, &nmsBoxes[s]));
        TENSORS_NVCV_CHECK(nvcvTensorConstruct(&nmsScoresReqs, alloc, &nmsScores[s]));
        TENSORS_NVCV_CHECK(nvcvTensorConstruct(&nmsClassesReqs, alloc, &nmsClasses[s]));
        TENSORS_NVCV_CHECK(nvcvTensorConstruct(&nmsCountReqs, alloc, &nmsCount[s]));

        if (hostPointerOf(nmsBoxes[s], &nmsBoxesHost[s]) != 0 || hostPointerOf(nmsScores[s], &nmsScoresHost[s]) != 0 ||
            hostPointerOf(nmsClasses[s], &nmsClassesHost[s]) != 0 || hostPointerOf(nmsCount[s], &nmsCountHost[s]) != 0)
            return -1;
    }

    return 0;
}

void PvaDetectionTensors::release()
{
    for (int s = 0; s < NUM_PIPE; s++)
    {
        for (NVCVTensorHandle *t : {&decBoxes[s], &decScores[s], &decClasses[s], &decCount[s], &nmsBoxes[s],
                                    &nmsScores[s], &nmsClasses[s], &nmsCount[s]})
        {
            if (*t != nullptr)
            {
                nvcvTensorDecRef(*t, nullptr);
                *t = nullptr;
            }
        }
    }
}

} // namespace yolov5_pipeline
