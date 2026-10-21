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

#include "bilateral_filter_ref.h"

#include "ImageIO.h"

#include <nvcv/Tensor.hpp>
#include <nvcv/TensorData.hpp>

#include <OpBilateralFilter.h>
#include <PvaAllocator.h>
#include <cupva_host.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <tuple>
#include <type_traits>
#include <vector>

#define QBITS 7
#define COLOR_8BIT 256
#define DIFF_TOL 1

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

template<typename T>
void generateRandomInputImpl(NVCVTensorHandle tensorHandle)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    T *tensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&tensorPtr, (void *)tensorData.buffer.strided.basePtr);

    int32_t imageHeight, imageWidth, imageLinePitch;
    std::tie(imageWidth, imageHeight, imageLinePitch) = getImageShapeFromTensor<T>(tensorData);

    for (int32_t i = 0; i < imageHeight; i++)
    {
        for (int32_t j = 0; j < imageWidth; j++)
        {
            if constexpr (std::is_signed<T>::value)
            {
                tensorPtr[i * imageLinePitch + j] =
                    static_cast<T>(rand() % (std::numeric_limits<T>::max() - std::numeric_limits<T>::min() + 1) +
                                   std::numeric_limits<T>::min());
            }
            else
            {
                tensorPtr[i * imageLinePitch + j] = static_cast<T>(rand() % (std::numeric_limits<T>::max() + 1));
            }
        }
    }
}

template<typename T, typename = std::enable_if_t<std::is_same_v<T, uint8_t>>>
int32_t checkOutputImpl(NVCVTensorHandle fixedPointRefTensorHandle, NVCVTensorHandle floatRefTensorHandle,
                        NVCVTensorHandle pvaTensorHandle)
{
    NVCVTensorData fixedPointRefTensorData;
    nvcvTensorExportData(fixedPointRefTensorHandle, &fixedPointRefTensorData);

    NVCVTensorData floatRefTensorData;
    nvcvTensorExportData(floatRefTensorHandle, &floatRefTensorData);

    T *fixedPointRef = NULL;
    CupvaMemGetHostPointer((void **)&fixedPointRef, (void *)fixedPointRefTensorData.buffer.strided.basePtr);

    T *floatRef = NULL;
    CupvaMemGetHostPointer((void **)&floatRef, (void *)floatRefTensorData.buffer.strided.basePtr);

    NVCVTensorData pvaTensorData;
    nvcvTensorExportData(pvaTensorHandle, &pvaTensorData);

    T *pva = NULL;
    CupvaMemGetHostPointer((void **)&pva, (void *)pvaTensorData.buffer.strided.basePtr);

    int32_t imageHeight, imageWidth, imageLinePitch;
    std::tie(imageWidth, imageHeight, imageLinePitch) = getImageShapeFromTensor<T>(fixedPointRefTensorData);

    int32_t err = 0;

    for (int32_t y = 0; y < imageHeight; y++)
    {
        for (int32_t x = 0; x < imageWidth; x++)
        {
            T fixedPointRefData = fixedPointRef[y * imageLinePitch + x];
            T pvaData           = pva[y * imageLinePitch + x];
            if (abs(fixedPointRefData - pvaData) > DIFF_TOL)
            {
                if (err < 10)
                    printf("Mismatch[%d, %d]: fixedPointRef = %u, pva = %u\n", y, x, fixedPointRefData, pvaData);
                err++;
            }
        }
    }
    if (err)
        printf("Error rate: %.2f%%\n", (float)err / (float)(imageHeight * imageWidth) * 100.0);

    double sumSquaredDiff = 0.0;
    int32_t maxAbsDiff    = 0;
    for (int32_t y = 0; y < imageHeight; y++)
    {
        for (int32_t x = 0; x < imageWidth; x++)
        {
            T floatRefData = floatRef[y * imageLinePitch + x];
            T pvaData      = pva[y * imageLinePitch + x];

            // Calculate squared difference for MSE and RMS
            int32_t diff = abs(floatRefData - pvaData);
            sumSquaredDiff += diff * diff;

            // Update max absolute difference
            if (diff > maxAbsDiff)
            {
                maxAbsDiff = diff;
            }
        }
    }

    // Calculate final metrics
    double mse = sumSquaredDiff / (imageHeight * imageWidth);
    double rms = std::sqrt(mse);

    printf("\nError Metrics between floatRef and PVA:\n");
    printf("MSE: %.4f\n", mse);
    printf("RMS: %.4f\n", rms);
    printf("Max Absolute Difference: %d\n", maxAbsDiff);

    return err;
}

template<typename T>
T float2Fixed(const float src, const int32_t qbits)
{
    // normalize values to range [-scale,scale] to preserve dynamic range before clamping to avoid overflow.
    float multiplier   = static_cast<float>(1U << qbits);
    int64_t upperBound = static_cast<int64_t>(std::numeric_limits<T>::max());
    int64_t lowerBound = static_cast<int64_t>(std::numeric_limits<T>::min());

    int64_t val = static_cast<int64_t>(std::rintf(src * multiplier));
    if (val < lowerBound)
        val = lowerBound;
    else if (val > upperBound)
        val = upperBound;
    return static_cast<T>(val);
}

template<typename T, typename = std::enable_if_t<std::is_same_v<T, uint8_t>>>
void bilateralFilterRefImpl(NVCVTensorHandle inTensorHandle, NVCVTensorHandle fixedPointRefTensorHandle,
                            NVCVTensorHandle floatRefTensorHandle, float sigmaRange, float sigmaSpace,
                            int32_t kernelSize, const NVCVBorderType borderMode, int32_t borderValue)
{
    int32_t kernelWidth  = kernelSize;
    int32_t kernelHeight = kernelSize;
    int32_t radius       = kernelSize >> 1;

    NVCVTensorData inTensorData;
    nvcvTensorExportData(inTensorHandle, &inTensorData);
    T *inTensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&inTensorPtr, (void *)inTensorData.buffer.strided.basePtr);

    NVCVTensorData fixedPointRefTensorData;
    nvcvTensorExportData(fixedPointRefTensorHandle, &fixedPointRefTensorData);
    T *fixedPointRefTensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&fixedPointRefTensorPtr, (void *)fixedPointRefTensorData.buffer.strided.basePtr);

    NVCVTensorData floatRefTensorData;
    nvcvTensorExportData(floatRefTensorHandle, &floatRefTensorData);
    T *floatRefTensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&floatRefTensorPtr, (void *)floatRefTensorData.buffer.strided.basePtr);

    int32_t imageHeight, imageWidth, imageLinePitch;
    std::tie(imageWidth, imageHeight, imageLinePitch) = getImageShapeFromTensor<T>(inTensorData);
    int32_t imageWidthPadding                         = imageWidth + radius * 2;
    int32_t imageHeightPadding                        = imageHeight + radius * 2;

    // Initialize the inPaddingTensor with border value assuming the border mode == NVCV_BORDER_CONSTANT.
    std::vector<T> inTensorWithPadding(imageHeightPadding * imageWidthPadding, borderValue);

    // Fill in with the original input image.
    for (int32_t y = 0; y < imageHeight; y++)
    {
        for (int32_t x = 0; x < imageWidth; x++)
        {
            inTensorWithPadding[(y + radius) * imageWidthPadding + x + radius] = inTensorPtr[y * imageLinePitch + x];
        }
    }

    // Revise the border if the border mode == NVCV_BORDER_REPLICATE.
    if (borderMode == NVCV_BORDER_REPLICATE)
    {
        for (int32_t y = 0; y < imageHeight; y++)
        {
            for (int32_t x = 0; x < radius; x++)
            {
                // Replicate left border
                inTensorWithPadding[(y + radius) * imageWidthPadding + x] = inTensorPtr[y * imageLinePitch];
                // Replicate right border
                inTensorWithPadding[(y + radius) * imageWidthPadding + imageWidthPadding - 1 - x] =
                    inTensorPtr[y * imageLinePitch + imageWidth - 1];
            }
        }

        for (int32_t y = 0; y < radius; y++)
        {
            // Replicate top border
            std::copy(inTensorWithPadding.begin() + radius * imageWidthPadding,
                      inTensorWithPadding.begin() + (radius + 1) * imageWidthPadding,
                      inTensorWithPadding.begin() + y * imageWidthPadding);
            // Replicate bottom border
            std::copy(inTensorWithPadding.end() - (radius + 1) * imageWidthPadding,
                      inTensorWithPadding.end() - radius * imageWidthPadding,
                      inTensorWithPadding.end() - (y + 1) * imageWidthPadding);
        }
    }

    float spaceCoeff = -0.5 / (sigmaSpace * sigmaSpace);
    float rangeCoeff = -0.5 / (sigmaRange * sigmaRange);

    // Initialize weight matrix.
    float spaceWeight[kernelSize * kernelSize];
    for (int32_t i = -radius; i <= radius; i++)
    {
        for (int32_t j = -radius; j <= radius; j++)
        {
            int32_t y                       = i + radius;
            int32_t x                       = j + radius;
            spaceWeight[y * kernelSize + x] = expf((i * i + j * j) * spaceCoeff);
        }
    }

    float rangeWeight[COLOR_8BIT];
    for (int32_t i = 0; i < COLOR_8BIT; i++)
    {
        rangeWeight[i] = expf(i * i * rangeCoeff);
    }

    for (int32_t i = 0; i < imageHeight; i++)
    {
        for (int32_t j = 0; j < imageWidth; j++)
        {
            int32_t sumFixedPointWeight = 0, sumFixedPointValue = 0;
            float sumFloatWeight = 0, sumFloatValue = 0;
            for (int32_t dy = -radius; dy <= radius; dy++)
            {
                for (int32_t dx = -radius; dx <= radius; dx++)
                {
                    if (dy * dy + dx * dx > radius * radius)
                    {
                        continue;
                    }

                    int32_t x = j + dx;
                    int32_t y = i + dy;

                    int32_t diff  = (int)abs(inTensorWithPadding[(y + radius) * imageWidthPadding + x + radius] -
                                             inTensorWithPadding[(i + radius) * imageWidthPadding + j + radius]);
                    float weight  = spaceWeight[(dy + radius) * kernelSize + dx + radius] * rangeWeight[diff];
                    T fixedWeight = float2Fixed<T>(weight, QBITS);

                    sumFixedPointValue +=
                        inTensorWithPadding[(y + radius) * imageWidthPadding + x + radius] * fixedWeight;
                    sumFixedPointWeight += fixedWeight;

                    sumFloatValue += inTensorWithPadding[(y + radius) * imageWidthPadding + x + radius] * weight;
                    sumFloatWeight += weight;
                }
            }
            fixedPointRefTensorPtr[i * imageLinePitch + j] =
                static_cast<T>(static_cast<float>(sumFixedPointValue) / static_cast<float>(sumFixedPointWeight) + 0.5f);
            floatRefTensorPtr[i * imageLinePitch + j] = static_cast<T>(sumFloatValue / sumFloatWeight + 0.5f);
        }
    }
}
} // namespace

void generateRandomBilateralFilterInput(NVCVTensorHandle tensorHandle)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    switch (tensorData.dtype)
    {
    case NVCV_DATA_TYPE_U8:
        generateRandomInputImpl<uint8_t>(tensorHandle);
        break;
    default:
        break;
    }
}

int32_t checkBilateralFilterOutput(NVCVTensorHandle fixedPointRefTensorHandle, NVCVTensorHandle floatRefTensorHandle,
                                   NVCVTensorHandle pvaTensorHandle)
{
    NVCVTensorData fixedPointRefTensorData;
    nvcvTensorExportData(fixedPointRefTensorHandle, &fixedPointRefTensorData);

    switch (fixedPointRefTensorData.dtype)
    {
    case NVCV_DATA_TYPE_U8:
        return checkOutputImpl<uint8_t>(fixedPointRefTensorHandle, floatRefTensorHandle, pvaTensorHandle);
    default:
        return -1;
    }
}

void bilateralFilterRef(NVCVTensorHandle inTensorHandle, NVCVTensorHandle fixedPointRefTensorHandle,
                        NVCVTensorHandle floatRefTensorHandle, float sigmaRange, float sigmaSpace, int32_t kernelSize,
                        const NVCVBorderType borderMode, int32_t borderValue)
{
    NVCVTensorData inTensorData;
    nvcvTensorExportData(inTensorHandle, &inTensorData);

    switch (inTensorData.dtype)
    {
    case NVCV_DATA_TYPE_U8:
        bilateralFilterRefImpl<uint8_t>(inTensorHandle, fixedPointRefTensorHandle, floatRefTensorHandle, sigmaRange,
                                        sigmaSpace, kernelSize, borderMode, borderValue);
        break;
    default:
        break;
    }
}

int32_t loadBilateralFilterInput(NVCVTensorHandle tensorHandle, const char *directory, const char *filename,
                                 const int32_t size)
{
    int32_t err = 0;
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    uint8_t *data = nullptr;
    CupvaMemGetHostPointer((void **)&data, (void *)tensorData.buffer.strided.basePtr);

    err = ReadImageBuffer(filename, directory, data, size);
    return err;
}

int32_t saveBilateralFilterOutput(NVCVTensorHandle tensorHandle, const char *directory, const char *filename,
                                  const int32_t size)
{
    int32_t err = 0;
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    uint8_t *data = nullptr;
    CupvaMemGetHostPointer((void **)&data, (void *)tensorData.buffer.strided.basePtr);

    err = WriteImageBuffer(filename, directory, data, size);
    return err;
}