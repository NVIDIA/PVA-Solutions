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

#include "blur_filter_roi_ref.h"

#include <climits>
#include <cstring>
#include <random>

template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value>>
void generateInputImpl(NVCVImageHandle imageHandle)
{
    NVCVImageData imageData;
    nvcvImageExportData(imageHandle, &imageData);
    T *imagePtr = NULL;
    CupvaMemGetHostPointer((void **)&imagePtr, (void *)imageData.buffer.strided.planes[0].basePtr);

    NVCVImageFormat format = imageData.format;
    int32_t numPlanes      = imageData.buffer.strided.numPlanes;

    for (int32_t p = 0; p < numPlanes; p++)
    {
        int32_t height    = imageData.buffer.strided.planes[p].height;
        int32_t width     = imageData.buffer.strided.planes[p].width;
        int32_t linePitch = imageData.buffer.strided.planes[p].rowStride / sizeof(T);

        int32_t bitsPerPixel;
        nvcvImageFormatGetPlaneBitsPerPixel(format, p, &bitsPerPixel);
        int32_t channelsPerPixel = (bitsPerPixel / CHAR_BIT) / sizeof(T);

        // use random generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<T> dist(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        // generate all possible value for input data
        for (int32_t j = 0; j < height; j++)
        {
            for (int32_t i = 0; i < width * channelsPerPixel; i++)
            {
                imagePtr[j * linePitch + i] = dist(gen);
            }
        }

        imagePtr += height * linePitch;
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
                        printf("Mismatch[%d, %d, %d, %d]: ref = %u, pva = %u\n", p, y, x, c, refData, cmpData);

                        return 1;
                    }
                }
            }
        }

        ref += height * linePitch;
        pva += height * linePitch;
    }

    return 0;
}

template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value>>
void blurFilterROIRefImpl(NVCVImageHandle inImageHandle, NVCVImageHandle outImageHandle,
                          const std::vector<NVCVRectI> &rectVec, const NVCVSize2D blockSize)
{
    NVCVImageData inImageData;
    nvcvImageExportData(inImageHandle, &inImageData);
    T *inImagePtr = NULL;
    CupvaMemGetHostPointer((void **)&inImagePtr, (void *)inImageData.buffer.strided.planes[0].basePtr);

    NVCVImageData outImageData;
    nvcvImageExportData(outImageHandle, &outImageData);
    T *outImagePtr = NULL;
    CupvaMemGetHostPointer((void **)&outImagePtr, (void *)outImageData.buffer.strided.planes[0].basePtr);

    NVCVImageFormat format = inImageData.format;
    int32_t numPlanes      = inImageData.buffer.strided.numPlanes;

    for (int32_t p = 0; p < numPlanes; p++)
    {
        int32_t height    = inImageData.buffer.strided.planes[p].height;
        int32_t width     = inImageData.buffer.strided.planes[p].width;
        int32_t linePitch = inImageData.buffer.strided.planes[p].rowStride / sizeof(T);

        int32_t bitsPerPixel;
        nvcvImageFormatGetPlaneBitsPerPixel(format, p, &bitsPerPixel);
        int32_t channelsPerPixel = (bitsPerPixel / CHAR_BIT) / sizeof(T);

        memcpy(outImagePtr, inImagePtr, height * linePitch * sizeof(T));

        if (p == 0)
        {
            for (const NVCVRectI &rect : rectVec)
            {
                int32_t rect_y1 = rect.y + rect.height;
                int32_t rect_x1 = rect.x + rect.width;
                for (int32_t by = rect.y; by < rect_y1; by += blockSize.h)
                {
                    for (int32_t bx = rect.x; bx < rect_x1; bx += blockSize.w)
                    {
                        T data = inImagePtr[by * linePitch + bx * channelsPerPixel];

                        int32_t block_y1 = std::min(by + blockSize.h, rect_y1);
                        int32_t block_x1 = std::min(bx + blockSize.w, rect_x1);
                        for (int32_t y = by; y < block_y1; y++)
                        {
                            for (int32_t x = bx; x < block_x1; x++)
                            {
                                outImagePtr[y * linePitch + x * channelsPerPixel] = data;
                            }
                        }
                    }
                }
            }
        }

        inImagePtr += height * linePitch;
        outImagePtr += height * linePitch;
    }
}

void generateBlurFilterROIInput(NVCVImageHandle imageHandle)
{
    NVCVImageFormat format;
    nvcvImageGetFormat(imageHandle, &format);
    NVCVDataType dataType;
    nvcvImageFormatGetPlaneDataType(format, 0, &dataType);

    switch (dataType)
    {
    case NVCV_DATA_TYPE_U8:
    case NVCV_DATA_TYPE_2U8:
    case NVCV_DATA_TYPE_3U8:
    case NVCV_DATA_TYPE_4U8:
        generateInputImpl<uint8_t>(imageHandle);
        break;
    default:
        break;
    }
}

int32_t checkBlurFilterROIOutput(NVCVImageHandle refImageHandle, NVCVImageHandle pvaImageHandle)
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

void blurFilterROIRef(NVCVImageHandle inImageHandle, NVCVImageHandle outImageHandle,
                      const std::vector<NVCVRectI> &rectVec, const NVCVSize2D blockSize)
{
    NVCVImageFormat inFormat;
    nvcvImageGetFormat(inImageHandle, &inFormat);
    NVCVDataType inDataType;
    nvcvImageFormatGetPlaneDataType(inFormat, 0, &inDataType);

    switch (inDataType)
    {
    case NVCV_DATA_TYPE_U8:
    case NVCV_DATA_TYPE_2U8:
    case NVCV_DATA_TYPE_3U8:
    case NVCV_DATA_TYPE_4U8:
        blurFilterROIRefImpl<uint8_t>(inImageHandle, outImageHandle, rectVec, blockSize);
        break;
    default:
        break;
    }
}

void blurFilterROIRefC(NVCVImageHandle inImageHandle, NVCVImageHandle outImageHandle, const NVCVRectI *rectArray,
                       int32_t numRects, const NVCVSize2D blockSize)
{
    std::vector<NVCVRectI> rectVec(rectArray, rectArray + numRects);
    blurFilterROIRef(inImageHandle, outImageHandle, rectVec, blockSize);
}

void memcpyToTensor(NVCVTensorHandle tensorHandle, const void *src, size_t count)
{
    NVCVDataType dataType;
    nvcvTensorGetDataType(tensorHandle, &dataType);
    int32_t bitsPerPixel;
    nvcvDataTypeGetBitsPerPixel(dataType, &bitsPerPixel);

    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    size_t sizeInBytes = tensorData.shape[0] * (bitsPerPixel / CHAR_BIT);
    if (count > sizeInBytes)
    {
        printf("memcpyToTensor exceeds maximum tensor size in bytes.\n");
        return;
    }

    void *tensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&tensorPtr, (void *)tensorData.buffer.strided.basePtr);

    memcpy(tensorPtr, src, count);
}