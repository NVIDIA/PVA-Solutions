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

#include "gaussian_pyramid_ref.h"

#include "ImageIO.h"

#include <OpGaussianPyramid.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#define KERNEL_SIZE (5)
#define QBITS (8)

namespace {

template<typename T>
void generateRandomInputImpl(NVCVImageHandle inputImageHandle)
{
    NVCVImageData inputImageData;
    nvcvImageExportData(inputImageHandle, &inputImageData);
    T *inputImagePtr = NULL;
    CupvaMemGetHostPointer((void **)&inputImagePtr, (void *)inputImageData.buffer.strided.planes[0].basePtr);

    int32_t height    = inputImageData.buffer.strided.planes[0].height;
    int32_t width     = inputImageData.buffer.strided.planes[0].width;
    int32_t linePitch = inputImageData.buffer.strided.planes[0].rowStride / sizeof(T);

    // generate random pixel values in input image
    for (int32_t j = 0; j < height; j++)
    {
        for (int32_t i = 0; i < width; i++)
        {
            if constexpr (std::is_signed<T>::value)
            {
                inputImagePtr[j * linePitch + i] =
                    static_cast<T>(rand() % (std::numeric_limits<T>::max() - std::numeric_limits<T>::min() + 1) +
                                   std::numeric_limits<T>::min());
            }
            else
            {
                inputImagePtr[j * linePitch + i] = static_cast<T>(rand() % (std::numeric_limits<T>::max() + 1));
            }
        }
    }
}

template<typename T>
int32_t checkGaussianPyramidOutputImpl(NVCVImageHandle refImageHandle, NVCVImageHandle pvaImageHandle)
{
    NVCVImageData refImageData;
    nvcvImageExportData(refImageHandle, &refImageData);

    T *ref = NULL;
    CupvaMemGetHostPointer((void **)&ref, (void *)refImageData.buffer.strided.planes[0].basePtr);

    NVCVImageData pvaImageData;
    nvcvImageExportData(pvaImageHandle, &pvaImageData);

    T *pva = NULL;
    CupvaMemGetHostPointer((void **)&pva, (void *)pvaImageData.buffer.strided.planes[0].basePtr);

    int32_t height       = refImageData.buffer.strided.planes[0].height;
    int32_t width        = refImageData.buffer.strided.planes[0].width;
    int32_t refLinePitch = refImageData.buffer.strided.planes[0].rowStride / sizeof(T);
    int32_t pvaLinePitch = pvaImageData.buffer.strided.planes[0].rowStride / sizeof(T);

    int32_t errorCount = 0;

    for (int32_t h = 0; h < height; ++h)
    {
        for (int32_t w = 0; w < width; ++w)
        {
            T refData = ref[h * refLinePitch + w];
            T cmpData = pva[h * pvaLinePitch + w];
            if (refData != cmpData)
            {
                if (errorCount < 10)
                    printf("Mismatch[%d, %d]: ref = %d, pva = %d\n", h, w, refData, cmpData);
                errorCount++;
            }
        }
    }

    return errorCount;
}

template<typename T>
void gaussianPyramidRefImpl(NVCVImageHandle inImageHandle, NVCVImageHandle *outImagesHandle, int32_t numLevels,
                            const NVCVBorderType borderMode, const int32_t borderValue)
{
    const int32_t totalLevels = numLevels;

    NVCVImageData inImageData;
    nvcvImageExportData(inImageHandle, &inImageData);
    T *inImagePtr = NULL;
    CupvaMemGetHostPointer((void **)&inImagePtr, (void *)inImageData.buffer.strided.planes[0].basePtr);

    T *outImagePtrs[totalLevels];
    for (int32_t i = 0; i < totalLevels; ++i)
    {
        outImagePtrs[i] = NULL;
        NVCVImageData outImageData;
        nvcvImageExportData(outImagesHandle[i], &outImageData);
        CupvaMemGetHostPointer((void **)&(outImagePtrs[i]), (void *)outImageData.buffer.strided.planes[0].basePtr);
    }

    int32_t height    = inImageData.buffer.strided.planes[0].height;
    int32_t width     = inImageData.buffer.strided.planes[0].width;
    int32_t linePitch = inImageData.buffer.strided.planes[0].rowStride / sizeof(T);

    // Get the linePitch for the first output image
    NVCVImageData firstOutImageData;
    nvcvImageExportData(outImagesHandle[0], &firstOutImageData);
    int32_t firstOutLinePitch = firstOutImageData.buffer.strided.planes[0].rowStride / sizeof(T);

    // Copy input image to first output using correct linePitch
    for (int32_t y = 0; y < height; ++y)
    {
        for (int32_t x = 0; x < width; ++x)
        {
            outImagePtrs[0][y * firstOutLinePitch + x] = inImagePtr[y * linePitch + x];
        }
    }

    // calculate the ref, allocate pyramids in DRAM
    uint32_t outputImageHeight                         = height;
    uint32_t outputImageWidth                          = width;
    uint16_t gaussianKernel[KERNEL_SIZE * KERNEL_SIZE] = {1,  4, 6, 4,  1,  4,  16, 24, 16, 4, 6, 24, 36,
                                                          24, 6, 4, 16, 24, 16, 4,  1,  4,  6, 4, 1};

    for (int32_t level = 1; level < totalLevels; level++)
    {
        // Use the pre-allocated output image pointer
        T *outImagePtr       = outImagePtrs[level];
        T *prevLevelImagePtr = outImagePtrs[level - 1]; // Use pre-allocated pointer

        // Get the linePitch for current output image
        NVCVImageData outImageData;
        nvcvImageExportData(outImagesHandle[level], &outImageData);
        int32_t outImageLinePitch = outImageData.buffer.strided.planes[0].rowStride / sizeof(T);

        // Get previous level image linePitch
        NVCVImageData prevLevelImageData;
        nvcvImageExportData(outImagesHandle[level - 1], &prevLevelImageData);
        int32_t prevLevelImageLinePitch = prevLevelImageData.buffer.strided.planes[0].rowStride / sizeof(T);

        uint32_t prevLevelImageHeight = outputImageHeight;
        uint32_t prevLevelImageWidth  = outputImageWidth;
        outputImageHeight             = (prevLevelImageHeight + 1) >> 1;
        outputImageWidth              = (prevLevelImageWidth + 1) >> 1;

        uint32_t radius                     = KERNEL_SIZE >> 1;
        uint32_t prevLevelImageHeightPadded = prevLevelImageHeight + radius * 2;
        uint32_t prevLevelImageWidthPadded  = prevLevelImageWidth + radius * 2;
        uint32_t rnd                        = (QBITS == 0) ? 0 : (1 << (QBITS - 1));

        // pad input image - default NVCV_BORDER_CONSTANT with borderValue
        std::vector<T> prevLevelImagePad;
        prevLevelImagePad.resize(prevLevelImageHeightPadded * prevLevelImageWidthPadded, borderValue);

        // Copy the image data to the center of the padded image
        for (size_t y = 0; y < prevLevelImageHeight; y++)
        {
            for (size_t x = 0; x < prevLevelImageWidth; x++)
            {
                int32_t index            = (y + radius) * prevLevelImageWidthPadded + (x + radius);
                prevLevelImagePad[index] = prevLevelImagePtr[y * prevLevelImageLinePitch + x];
            }
        }

        // compute the ref output
        for (int32_t y = 0; y < prevLevelImageHeight; y += 2)
        {
            for (int32_t x = 0; x < prevLevelImageWidth; x += 2)
            {
                int32_t acc = 0;
                for (int32_t k = 0; k < KERNEL_SIZE; k++)
                {
                    for (int32_t t = 0; t < KERNEL_SIZE; t++)
                    {
                        acc += static_cast<int32_t>(prevLevelImagePad[(y + k) * prevLevelImageWidthPadded + (x + t)]) *
                               static_cast<int32_t>(gaussianKernel[k * KERNEL_SIZE + t]);
                    }
                }
                outImagePtr[(y / 2) * outImageLinePitch + (x / 2)] = static_cast<T>((acc + rnd) >> QBITS);
            }
        }
    }
}

} // namespace

void generateRandomGaussianPyramidInput(NVCVImageHandle inputImageHandle)
{
    NVCVImageFormat format;
    nvcvImageGetFormat(inputImageHandle, &format);

    NVCVDataType dataType;
    nvcvImageFormatGetPlaneDataType(format, 0, &dataType);

    switch (dataType)
    {
    case NVCV_DATA_TYPE_U8:
        generateRandomInputImpl<uint8_t>(inputImageHandle);
        break;
    case NVCV_DATA_TYPE_S8:
        generateRandomInputImpl<int8_t>(inputImageHandle);
        break;
    case NVCV_DATA_TYPE_U16:
        generateRandomInputImpl<uint16_t>(inputImageHandle);
        break;
    case NVCV_DATA_TYPE_S16:
        generateRandomInputImpl<int16_t>(inputImageHandle);
        break;
    default:
        break;
    }
}

int32_t checkGaussianPyramidOutput(NVCVImageHandle *refImagesHandle, NVCVImageHandle *pvaImagesHandle,
                                   int32_t numLevels)
{
    NVCVImageFormat format;
    nvcvImageGetFormat(refImagesHandle[0], &format);

    NVCVDataType dataType;
    nvcvImageFormatGetPlaneDataType(format, 0, &dataType);

    for (int32_t i = 0; i < numLevels; ++i)
    {
        int32_t err = 0;
        switch (dataType)
        {
        case NVCV_DATA_TYPE_U8:
            err = checkGaussianPyramidOutputImpl<uint8_t>(refImagesHandle[i], pvaImagesHandle[i]);
            break;
        case NVCV_DATA_TYPE_S8:
            err = checkGaussianPyramidOutputImpl<int8_t>(refImagesHandle[i], pvaImagesHandle[i]);
            break;
        case NVCV_DATA_TYPE_U16:
            err = checkGaussianPyramidOutputImpl<uint16_t>(refImagesHandle[i], pvaImagesHandle[i]);
            break;
        case NVCV_DATA_TYPE_S16:
            err = checkGaussianPyramidOutputImpl<int16_t>(refImagesHandle[i], pvaImagesHandle[i]);
            break;
        default:
            return -1;
        }

        if (err != 0)
        {
            printf("Checking level %d failed\n", i);
            return err;
        }
    }

    return 0;
}

void gaussianPyramidRef(NVCVImageHandle inImageHandle, NVCVImageHandle *outImagesHandle, int32_t numLevels,
                        const NVCVBorderType borderMode, const int32_t borderValue)
{
    NVCVImageFormat format;
    nvcvImageGetFormat(inImageHandle, &format);

    NVCVDataType dataType;
    nvcvImageFormatGetPlaneDataType(format, 0, &dataType);
    switch (dataType)
    {
    case NVCV_DATA_TYPE_U8:
        gaussianPyramidRefImpl<uint8_t>(inImageHandle, outImagesHandle, numLevels, borderMode, borderValue);
        break;
    case NVCV_DATA_TYPE_S8:
        gaussianPyramidRefImpl<int8_t>(inImageHandle, outImagesHandle, numLevels, borderMode, borderValue);
        break;
    case NVCV_DATA_TYPE_U16:
        gaussianPyramidRefImpl<uint16_t>(inImageHandle, outImagesHandle, numLevels, borderMode, borderValue);
        break;
    case NVCV_DATA_TYPE_S16:
        gaussianPyramidRefImpl<int16_t>(inImageHandle, outImagesHandle, numLevels, borderMode, borderValue);
        break;
    default:
        break;
    }
}

int32_t loadGaussianPyramidInput(NVCVImageHandle imageHandle, const char *directory, const char *filename,
                                 const int32_t size)
{
    int32_t err = 0;
    NVCVImageData imageData;
    nvcvImageExportData(imageHandle, &imageData);

    uint8_t *data = nullptr;
    CupvaMemGetHostPointer((void **)&data, (void *)imageData.buffer.strided.planes[0].basePtr);

    err = ReadImageBuffer(filename, directory, data, size);
    return err;
}

int32_t saveGaussianPyramidOutput(NVCVImageHandle imageHandle, const char *directory, const char *filename,
                                  const int32_t size)
{
    int32_t err = 0;
    NVCVImageData imageData;
    nvcvImageExportData(imageHandle, &imageData);

    uint8_t *data = nullptr;
    CupvaMemGetHostPointer((void **)&data, (void *)imageData.buffer.strided.planes[0].basePtr);

    err = WriteImageBuffer(filename, directory, data, size);
    return err;
}
