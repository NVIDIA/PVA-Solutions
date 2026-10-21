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

#include "image_blend_ref.h"

#include <nvcv/Tensor.hpp>

#include <OpImageBlend.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <type_traits>
#include <vector>

#define QBITS_U8 (7)
#define QBITS_VALUE_U8 (1 << QBITS_U8)

namespace {

template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value>>
void generateRandomInputImpl(NVCVImageHandle src0ImageHandle, NVCVImageHandle src1ImageHandle)
{
    NVCVImageData src0ImageData;
    nvcvImageExportData(src0ImageHandle, &src0ImageData);

    NVCVImageFormat format = src0ImageData.format;
    int32_t numPlanes      = src0ImageData.buffer.strided.numPlanes;

    T *src0ImagePtr = NULL;
    CupvaMemGetHostPointer((void **)&src0ImagePtr, (void *)src0ImageData.buffer.strided.planes[0].basePtr);

    NVCVImageData src1ImageData;
    nvcvImageExportData(src1ImageHandle, &src1ImageData);

    T *src1ImagePtr = NULL;
    CupvaMemGetHostPointer((void **)&src1ImagePtr, (void *)src1ImageData.buffer.strided.planes[0].basePtr);

    for (int32_t p = 0; p < numPlanes; p++)
    {
        int32_t height    = src0ImageData.buffer.strided.planes[p].height;
        int32_t width     = src0ImageData.buffer.strided.planes[p].width;
        int32_t linePitch = src0ImageData.buffer.strided.planes[p].rowStride / sizeof(T);

        int32_t bitsPerPixel;
        nvcvImageFormatGetPlaneBitsPerPixel(format, p, &bitsPerPixel);

        int32_t channelsPerPixel = (bitsPerPixel / CHAR_BIT) / sizeof(T);

        // generate random pixel values in input image
        for (int32_t j = 0; j < height; j++)
        {
            for (int32_t i = 0; i < width * channelsPerPixel; i++)
            {
                if constexpr (std::is_signed<T>::value)
                {
                    src0ImagePtr[j * linePitch + i] =
                        static_cast<T>(rand() % (std::numeric_limits<T>::max() - std::numeric_limits<T>::min() + 1) +
                                       std::numeric_limits<T>::min());
                    src1ImagePtr[j * linePitch + i] =
                        static_cast<T>(rand() % (std::numeric_limits<T>::max() - std::numeric_limits<T>::min() + 1) +
                                       std::numeric_limits<T>::min());
                }
                else
                {
                    src0ImagePtr[j * linePitch + i] = static_cast<T>(rand() % (std::numeric_limits<T>::max() + 1));
                    src1ImagePtr[j * linePitch + i] = static_cast<T>(rand() % (std::numeric_limits<T>::max() + 1));
                }
            }
        }

        src0ImagePtr += height * linePitch;
        src1ImagePtr += height * linePitch;
    }
}

template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value>>
int32_t checkOutputImpl(NVCVImageHandle refImageHandle, NVCVImageHandle pvaImageHandle)
{
    NVCVImageData refImageData;
    nvcvImageExportData(refImageHandle, &refImageData);

    T *ref = NULL;
    CupvaMemGetHostPointer((void **)&ref, (void *)refImageData.buffer.strided.planes[0].basePtr);

    NVCVImageData pvaImageData;
    nvcvImageExportData(pvaImageHandle, &pvaImageData);

    T *pva = NULL;
    CupvaMemGetHostPointer((void **)&pva, (void *)pvaImageData.buffer.strided.planes[0].basePtr);

    NVCVImageFormat format = refImageData.format;
    int32_t numPlanes      = refImageData.buffer.strided.numPlanes;
    int32_t errorCount     = 0;
    int32_t pixelCount     = 0;

    for (int32_t p = 0; p < numPlanes; p++)
    {
        int32_t height    = refImageData.buffer.strided.planes[p].height;
        int32_t width     = refImageData.buffer.strided.planes[p].width;
        int32_t linePitch = refImageData.buffer.strided.planes[p].rowStride / sizeof(T);

        int32_t bitsPerPixel;
        nvcvImageFormatGetPlaneBitsPerPixel(format, p, &bitsPerPixel);

        int32_t channelsPerPixel = (bitsPerPixel / CHAR_BIT) / sizeof(T);

        for (int32_t y = 0; y < height; y++)
        {
            for (int32_t x = 0; x < width; x++)
            {
                for (int32_t c = 0; c < channelsPerPixel; c++)
                {
                    T refData = ref[y * linePitch + x * channelsPerPixel + c];
                    T cmpData = pva[y * linePitch + x * channelsPerPixel + c];
                    if (refData != cmpData)
                    {
                        if (errorCount < 10)
                            printf("Mismatch[%d, %d, %d, %d]: ref = %u, pva = %u\n", p, y, x, c, refData, cmpData);
                        errorCount++;
                    }
                    pixelCount++;
                }
            }
        }

        ref += height * linePitch;
        pva += height * linePitch;
    }

    if (!errorCount)
        return 0;
    else
    {
        printf("Error rate: %.2f%%\n", static_cast<float>(errorCount) / static_cast<float>(pixelCount) * 100.0);
        return -1;
    }
}

template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value>>
void imageBlendRefImpl(NVCVImageHandle src0ImageHandle, NVCVImageHandle src1ImageHandle, NVCVImageHandle dstImageHandle,
                       float alpha)
{
    NVCVImageData src0ImageData;
    nvcvImageExportData(src0ImageHandle, &src0ImageData);
    T *src0ImagePtr = NULL;
    CupvaMemGetHostPointer((void **)&src0ImagePtr, (void *)src0ImageData.buffer.strided.planes[0].basePtr);

    NVCVImageData src1ImageData;
    nvcvImageExportData(src1ImageHandle, &src1ImageData);
    T *src1ImagePtr = NULL;
    CupvaMemGetHostPointer((void **)&src1ImagePtr, (void *)src1ImageData.buffer.strided.planes[0].basePtr);

    NVCVImageData dstImageData;
    nvcvImageExportData(dstImageHandle, &dstImageData);
    T *dstImagePtr = NULL;
    CupvaMemGetHostPointer((void **)&dstImagePtr, (void *)dstImageData.buffer.strided.planes[0].basePtr);

    NVCVImageFormat format = src0ImageData.format;
    int32_t numPlanes      = src0ImageData.buffer.strided.numPlanes;

    int32_t fixedPointAlpha = std::round(alpha * QBITS_VALUE_U8);

    for (int32_t p = 0; p < numPlanes; p++)
    {
        int32_t height    = src0ImageData.buffer.strided.planes[p].height;
        int32_t width     = src0ImageData.buffer.strided.planes[p].width;
        int32_t linePitch = src0ImageData.buffer.strided.planes[p].rowStride / sizeof(T);

        int32_t bitsPerPixel;
        nvcvImageFormatGetPlaneBitsPerPixel(format, p, &bitsPerPixel);

        int32_t channelsPerPixel = (bitsPerPixel / CHAR_BIT) / sizeof(T);

        for (int32_t y = 0; y < height; y++)
        {
            for (int32_t x = 0; x < width; x++)
            {
                for (int32_t c = 0; c < channelsPerPixel; c++)
                {
                    int32_t src0 = src0ImagePtr[y * linePitch + x * channelsPerPixel + c];
                    int32_t src1 = src1ImagePtr[y * linePitch + x * channelsPerPixel + c];
                    // bit-exact with vblend
                    int32_t dst = std::floor(
                        (float)src0 + (float)(src1 * fixedPointAlpha - src0 * fixedPointAlpha) / (float)QBITS_VALUE_U8 +
                        0.5f);
                    dstImagePtr[y * linePitch + x * channelsPerPixel + c] = dst;
                }
            }
        }

        src0ImagePtr += height * linePitch;
        src1ImagePtr += height * linePitch;
        dstImagePtr += height * linePitch;
    }
}
} // namespace

void generateRandomImageBlendInput(NVCVImageHandle src0ImageHandle, NVCVImageHandle src1ImageHandle)
{
    NVCVImageFormat format;
    nvcvImageGetFormat(src0ImageHandle, &format);

    NVCVDataType dataType;
    nvcvImageFormatGetPlaneDataType(format, 0, &dataType);

    switch (dataType)
    {
    case NVCV_DATA_TYPE_U8:
    case NVCV_DATA_TYPE_2U8:
    case NVCV_DATA_TYPE_3U8:
    case NVCV_DATA_TYPE_4U8:
        generateRandomInputImpl<uint8_t>(src0ImageHandle, src1ImageHandle);
        break;
    default:
        break;
    }
}

int32_t checkImageBlendOutput(NVCVImageHandle refImageHandle, NVCVImageHandle pvaImageHandle)
{
    NVCVImageFormat refFormat;
    nvcvImageGetFormat(refImageHandle, &refFormat);

    NVCVDataType refDataType;
    nvcvImageFormatGetPlaneDataType(refFormat, 0, &refDataType);

    switch (refDataType)
    {
    case NVCV_DATA_TYPE_U8:
    case NVCV_DATA_TYPE_2U8:
    case NVCV_DATA_TYPE_3U8:
    case NVCV_DATA_TYPE_4U8:
        return checkOutputImpl<uint8_t>(refImageHandle, pvaImageHandle);
    default:
        return -1;
    }
}

void imageBlendRef(NVCVImageHandle src0ImageHandle, NVCVImageHandle src1ImageHandle, NVCVImageHandle dstImageHandle,
                   float alpha)
{
    NVCVImageFormat format;
    nvcvImageGetFormat(src0ImageHandle, &format);

    NVCVDataType dataType;
    nvcvImageFormatGetPlaneDataType(format, 0, &dataType);
    switch (dataType)
    {
    case NVCV_DATA_TYPE_U8:
    case NVCV_DATA_TYPE_2U8:
    case NVCV_DATA_TYPE_3U8:
    case NVCV_DATA_TYPE_4U8:
        imageBlendRefImpl<uint8_t>(src0ImageHandle, src1ImageHandle, dstImageHandle, alpha);
        break;
    default:
        break;
    }
}