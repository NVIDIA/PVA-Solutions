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

#include "template_matching_ref.h"

#include <nvcv/Tensor.hpp>

#include <OpTemplateMatching.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <type_traits>
#include <vector>

#define DIFF_TOL 1e-4 // threshold of CPU and VPU floating point outputs' absdiff

namespace {
// get image width, height and line pitch from NVCVTensorData
template<typename T>
std::tuple<int, int, int> getImageShapeFromTensor(const NVCVTensorData &tensorData)
{
    nvcv::TensorShape shape(tensorData.shape, tensorData.rank, tensorData.layout);

    int imageWidth = 0, imageHeight = 0, linePitch = 0;
    // currently only supporting [HWC/CHW/NCHW/NHWC] layouts
    if (shape.layout() == nvcv::TENSOR_HWC)
    {
        imageHeight = shape[0];
        imageWidth  = shape[1];
        linePitch   = tensorData.buffer.strided.strides[0] / sizeof(T);
    }
    else if (shape.layout() == nvcv::TENSOR_CHW)
    {
        imageHeight = shape[1];
        imageWidth  = shape[2];
        linePitch   = tensorData.buffer.strided.strides[1] / sizeof(T);
    }
    else if (shape.layout() == nvcv::TENSOR_NCHW)
    {
        imageHeight = shape[2];
        imageWidth  = shape[3];
        linePitch   = tensorData.buffer.strided.strides[2] / sizeof(T);
    }
    else if (shape.layout() == nvcv::TENSOR_NHWC)
    {
        imageHeight = shape[1];
        imageWidth  = shape[2];
        linePitch   = tensorData.buffer.strided.strides[1] / sizeof(T);
    }
    return std::make_tuple(imageWidth, imageHeight, linePitch);
}

template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value>>
void generateRandomInputImpl(NVCVTensorHandle inImageTensorHandle, NVCVTensorHandle inTemplateTensorHandle)
{
    NVCVTensorData inImageTensorData;
    nvcvTensorExportData(inImageTensorHandle, &inImageTensorData);
    T *inImageTensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&inImageTensorPtr, (void *)inImageTensorData.buffer.strided.basePtr);

    NVCVTensorData inTemplateTensorData;
    nvcvTensorExportData(inTemplateTensorHandle, &inTemplateTensorData);
    T *inTemplateTensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&inTemplateTensorPtr, (void *)inTemplateTensorData.buffer.strided.basePtr);

    int32_t imageHeight, imageWidth, imageLinePitch;
    std::tie(imageWidth, imageHeight, imageLinePitch) = getImageShapeFromTensor<T>(inImageTensorData);

    int32_t templateHeight, templateWidth, templateLinePitch;
    std::tie(templateWidth, templateHeight, templateLinePitch) = getImageShapeFromTensor<T>(inTemplateTensorData);

    for (int32_t i = 0; i < imageHeight; i++)
    {
        for (int32_t j = 0; j < imageWidth; j++)
        {
            if constexpr (std::is_signed<T>::value)
            {
                inImageTensorPtr[i * imageLinePitch + j] =
                    static_cast<T>(rand() % (std::numeric_limits<T>::max() - std::numeric_limits<T>::min() + 1) +
                                   std::numeric_limits<T>::min());
            }
            else
            {
                inImageTensorPtr[i * imageLinePitch + j] = static_cast<T>(rand() % (std::numeric_limits<T>::max() + 1));
            }
        }
    }

    for (int32_t i = 0; i < templateHeight; i++)
    {
        for (int32_t j = 0; j < templateWidth; j++)
        {
            if constexpr (std::is_signed<T>::value)
            {
                inTemplateTensorPtr[i * templateLinePitch + j] =
                    static_cast<T>(rand() % (std::numeric_limits<T>::max() - std::numeric_limits<T>::min() + 1) +
                                   std::numeric_limits<T>::min());
            }
            else
            {
                inTemplateTensorPtr[i * templateLinePitch + j] =
                    static_cast<T>(rand() % (std::numeric_limits<T>::max() + 1));
            }
        }
    }
}

template<typename T, typename = std::enable_if_t<std::is_same<T, float>::value>>
int32_t checkOutputImpl(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle)
{
    NVCVTensorData refTensorData;
    nvcvTensorExportData(refTensorHandle, &refTensorData);

    T *ref = NULL;
    CupvaMemGetHostPointer((void **)&ref, (void *)refTensorData.buffer.strided.basePtr);

    NVCVTensorData pvaTensorData;
    nvcvTensorExportData(pvaTensorHandle, &pvaTensorData);

    T *pva = NULL;
    CupvaMemGetHostPointer((void **)&pva, (void *)pvaTensorData.buffer.strided.basePtr);

    int32_t height, width, linePitch;
    std::tie(width, height, linePitch) = getImageShapeFromTensor<float>(refTensorData);

    int32_t err = 0;
    for (int32_t y = 0; y < height; y++)
    {
        for (int32_t x = 0; x < width; x++)
        {
            T refData = ref[y * linePitch + x];
            T cmpData = pva[y * linePitch + x];
            if (std::abs(refData - cmpData) >= DIFF_TOL)
            {
                if (err < 10)
                    printf("Mismatch[%d, %d]: ref = %f, pva = %f\n", y, x, refData, cmpData);
                err++;
            }
        }
    }
    if (err)
        printf("Error rate: %.2f%%\n", (float)err / (float)(height * width) * 100.0);
    return err;
}

template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value>>
void templateMatchingRefImpl(NVCVTensorHandle inImageTensorHandle, NVCVTensorHandle inTemplateTensorHandle,
                             NVCVTensorHandle outTensorHandle)
{
    NVCVTensorData inImageTensorData;
    nvcvTensorExportData(inImageTensorHandle, &inImageTensorData);
    T *inImageTensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&inImageTensorPtr, (void *)inImageTensorData.buffer.strided.basePtr);

    NVCVTensorData inTemplateTensorData;
    nvcvTensorExportData(inTemplateTensorHandle, &inTemplateTensorData);
    T *inTemplateTensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&inTemplateTensorPtr, (void *)inTemplateTensorData.buffer.strided.basePtr);

    NVCVTensorData outTensorData;
    nvcvTensorExportData(outTensorHandle, &outTensorData);
    float *outTensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&outTensorPtr, (void *)outTensorData.buffer.strided.basePtr);

    int32_t imageHeight, imageWidth, imageLinePitch;
    std::tie(imageWidth, imageHeight, imageLinePitch) = getImageShapeFromTensor<T>(inImageTensorData);

    int32_t templateHeight, templateWidth, templateLinePitch;
    std::tie(templateWidth, templateHeight, templateLinePitch) = getImageShapeFromTensor<T>(inTemplateTensorData);

    int32_t outHeight, outWidth, outLinePitch;
    std::tie(outWidth, outHeight, outLinePitch) = getImageShapeFromTensor<float>(outTensorData);

    float templateSum = 0;
    for (int32_t i = 0; i < templateHeight; i++)
    {
        for (int32_t j = 0; j < templateWidth; j++)
        {
            templateSum +=
                (inTemplateTensorPtr[i * templateLinePitch + j] * inTemplateTensorPtr[i * templateLinePitch + j]);
        }
    }

    for (int32_t y = 0; y < outHeight; y++)
    {
        for (int32_t x = 0; x < outWidth; x++)
        {
            int64_t diffSum  = 0;
            int64_t imageSum = 0;

            for (int32_t i = 0; i < templateHeight; i++)
            {
                for (int32_t j = 0; j < templateWidth; j++)
                {
                    imageSum += (inImageTensorPtr[(y + i) * imageLinePitch + x + j] *
                                 inImageTensorPtr[(y + i) * imageLinePitch + x + j]);

                    int32_t diff = inTemplateTensorPtr[i * templateLinePitch + j] -
                                   inImageTensorPtr[(y + i) * imageLinePitch + x + j];
                    diffSum += (diff * diff);
                }
            }
            outTensorPtr[y * outLinePitch + x] = (float)diffSum / sqrt(templateSum * (float)imageSum);
        }
    }
}
} // namespace

void generateRandomTemplateMatchingInput(NVCVTensorHandle inImageTensorHandle, NVCVTensorHandle inTemplateTensorHandle)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(inImageTensorHandle, &tensorData);

    switch (tensorData.dtype)
    {
    case NVCV_DATA_TYPE_U8:
        generateRandomInputImpl<uint8_t>(inImageTensorHandle, inTemplateTensorHandle);
        break;
    default:
        break;
    }
}

int32_t checkTemplateMatchingOutput(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle)
{
    NVCVTensorData refTensorData;
    nvcvTensorExportData(refTensorHandle, &refTensorData);

    switch (refTensorData.dtype)
    {
    case NVCV_DATA_TYPE_F32:
        return checkOutputImpl<float>(refTensorHandle, pvaTensorHandle);
    default:
        return -1;
    }
}

void templateMatchingRef(NVCVTensorHandle inImageTensorHandle, NVCVTensorHandle inTemplateTensorHandle,
                         NVCVTensorHandle outTensorHandle)
{
    NVCVTensorData inImageTensorData;
    nvcvTensorExportData(inImageTensorHandle, &inImageTensorData);

    switch (inImageTensorData.dtype)
    {
    case NVCV_DATA_TYPE_U8:
        templateMatchingRefImpl<uint8_t>(inImageTensorHandle, inTemplateTensorHandle, outTensorHandle);
        break;
    default:
        break;
    }
}