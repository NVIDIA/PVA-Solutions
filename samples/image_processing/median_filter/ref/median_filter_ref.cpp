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

#include "median_filter_ref.h"

#include <nvcv/Tensor.hpp>

#include <OpMedianFilter.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <type_traits>

namespace {

template<typename T>
void generateRandomInputImpl(NVCVImageHandle imageHandle)
{
    unsigned int seed = 365;
    srand(seed);

    NVCVImageData imageData;
    nvcvImageExportData(imageHandle, &imageData);

    T *imagePtr_h = NULL;
    CupvaMemGetHostPointer((void **)&imagePtr_h, (void *)imageData.buffer.strided.planes[0].basePtr);

    NVCVImageFormat format = imageData.format;
    int numPlanes          = imageData.buffer.strided.numPlanes;

    int64_t minVal = std::numeric_limits<T>::min();
    int64_t maxVal = std::numeric_limits<T>::max();
    // fix limits for image formats with special range
    if (format == NVCV_IMAGE_FORMAT_Y8)
    {
        minVal = 16;
        maxVal = 235;
    }
    else if (format == NVCV_IMAGE_FORMAT_Y16)
    {
        minVal = 4096;
        maxVal = 60160;
    }

    for (int p = 0; p < numPlanes; p++)
    {
        int height    = imageData.buffer.strided.planes[p].height;
        int width     = imageData.buffer.strided.planes[p].width;
        int linePitch = imageData.buffer.strided.planes[p].rowStride / sizeof(T);

        int32_t numChannels;
        nvcvImageFormatGetPlaneNumChannels(format, p, &numChannels);

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                for (int c = 0; c < numChannels; c++)
                {
                    int64_t val                                     = (int64_t)rand() % (maxVal - minVal + 1) + minVal;
                    imagePtr_h[i * linePitch + j * numChannels + c] = static_cast<T>(val);
                }
            }
        }
        imagePtr_h += height * linePitch;
    }
}

void generateBoxFilterKernelImpl(NVCVTensorHandle kernelHandle, int32_t kernelWidth, int32_t kernelHeight)
{
    NVCVTensorData kernelData;
    nvcvTensorExportData(kernelHandle, &kernelData);

    int8_t *kernelPtr_h = NULL;
    CupvaMemGetHostPointer((void **)&kernelPtr_h, (void *)kernelData.buffer.strided.basePtr);

    for (int i = 0; i < kernelHeight; i++)
    {
        for (int j = 0; j < kernelWidth; j++)
        {
            kernelPtr_h[i * kernelWidth + j] = 1;
        }
    }
}

template<typename T>
int32_t checkImageOutputImpl(NVCVImageHandle refImageHandle, NVCVImageHandle pvaImageHandle)
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

        int32_t numChannels;
        nvcvImageFormatGetPlaneNumChannels(format, p, &numChannels);

        for (int32_t y = 0; y < height; y++)
        {
            for (int32_t x = 0; x < width; x++)
            {
                for (int32_t c = 0; c < numChannels; c++)
                {
                    T refData = ref[y * linePitch + x * numChannels + c];
                    T cmpData = pva[y * linePitch + x * numChannels + c];

                    if (refData != cmpData)
                    {
                        printf("Mismatch, plane %d, [%d, %d, %d]: ref = %s, pva = %s\n", p, y, x, c,
                               std::to_string(refData).c_str(), std::to_string(cmpData).c_str());
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

template<typename T>
void medianFilterRefImpl(NVCVImageHandle inImageHandle, NVCVImageHandle outImageHandle, int32_t kernelHeight,
                         int32_t kernelWidth, NVCVTensorHandle kernel, NVCVBorderType borderType, int32_t borderValue)
{
    NVCVImageData inImageData;
    nvcvImageExportData(inImageHandle, &inImageData);
    T *inImagePtr_h = NULL;
    CupvaMemGetHostPointer((void **)&inImagePtr_h, (void *)inImageData.buffer.strided.planes[0].basePtr);

    NVCVImageData outImageData;
    nvcvImageExportData(outImageHandle, &outImageData);
    T *outImagePtr_h = NULL;
    CupvaMemGetHostPointer((void **)&outImagePtr_h, (void *)outImageData.buffer.strided.planes[0].basePtr);

    NVCVTensorData kernelData;
    nvcvTensorExportData(kernel, &kernelData);
    int8_t *kernelPtr_h = NULL;
    CupvaMemGetHostPointer((void **)&kernelPtr_h, (void *)kernelData.buffer.strided.basePtr);

    NVCVImageFormat format = inImageData.format;
    int numPlanes          = inImageData.buffer.strided.numPlanes;

    // calculate number of valid elements in the kernel window
    int kernelMaskSize = 0;
    for (int i = 0; i < kernelHeight; i++)
    {
        for (int j = 0; j < kernelWidth; j++)
        {
            if (kernelPtr_h[i * kernelWidth + j] != 0)
            {
                kernelMaskSize++;
            }
        }
    }
    // if the kernel is empty, we don't need to compute anything
    if (kernelMaskSize == 0)
    {
        return;
    }
    int medianIdx = kernelMaskSize / 2;

    // init vector to store elements in the kernel window
    std::vector<T> vElements;
    vElements.reserve(kernelHeight * kernelWidth);

    for (int p = 0; p < numPlanes; p++)
    {
        int height    = inImageData.buffer.strided.planes[p].height;
        int width     = inImageData.buffer.strided.planes[p].width;
        int linePitch = inImageData.buffer.strided.planes[p].rowStride / sizeof(T);

        int32_t numChannels;
        nvcvImageFormatGetPlaneNumChannels(format, p, &numChannels);

        // find median for each pixel
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int c = 0; c < numChannels; c++)
                {
                    vElements.clear();
                    for (int ky = -kernelHeight / 2; ky <= kernelHeight / 2; ky++)
                    {
                        for (int kx = -kernelWidth / 2; kx <= kernelWidth / 2; kx++)
                        {
                            // skip kernel elements that are 0
                            if (kernelPtr_h[ky + kernelHeight / 2, kx + kernelWidth / 2] == 0)
                            {
                                continue;
                            }

                            int idx_y = y + ky;
                            int idx_x = x + kx;

                            T pixelValue;
                            // extend border according to the border type
                            if (idx_y < 0 || idx_y >= height || idx_x < 0 || idx_x >= width)
                            {
                                if (borderType == NVCV_BORDER_CONSTANT)
                                {
                                    pixelValue = borderValue;
                                }
                                else if (borderType == NVCV_BORDER_REPLICATE)
                                {
                                    idx_y      = std::clamp(idx_y, 0, height - 1);
                                    idx_x      = std::clamp(idx_x, 0, width - 1);
                                    pixelValue = inImagePtr_h[idx_y * linePitch + idx_x * numChannels + c];
                                }
                                else
                                {
                                    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid border type!");
                                }
                            }
                            else
                            {
                                pixelValue = inImagePtr_h[idx_y * linePitch + idx_x * numChannels + c];
                            }
                            vElements.push_back(pixelValue);
                        }
                    }

                    // Calculate median on individual channels
                    std::nth_element(vElements.begin(), vElements.begin() + medianIdx, vElements.end());

                    outImagePtr_h[y * linePitch + x * numChannels + c] = vElements[medianIdx];
                }
            }
        }

        inImagePtr_h += height * linePitch;
        outImagePtr_h += height * linePitch;
    }
}

} // namespace

void generateMedianFilterRandomImageInput(NVCVImageHandle imageHandle)
{
    NVCVImageFormat format;
    nvcvImageGetFormat(imageHandle, &format);

    switch (format)
    {
    case NVCV_IMAGE_FORMAT_U8:
    case NVCV_IMAGE_FORMAT_Y8:
    case NVCV_IMAGE_FORMAT_Y8_ER:
        generateRandomInputImpl<uint8_t>(imageHandle);
        break;
    case NVCV_IMAGE_FORMAT_S8:
        generateRandomInputImpl<int8_t>(imageHandle);
        break;
    case NVCV_IMAGE_FORMAT_U16:
    case NVCV_IMAGE_FORMAT_Y16:
    case NVCV_IMAGE_FORMAT_Y16_ER:
        generateRandomInputImpl<uint16_t>(imageHandle);
        break;
    case NVCV_IMAGE_FORMAT_S16:
    case NVCV_IMAGE_FORMAT_2S16:
        generateRandomInputImpl<int16_t>(imageHandle);
        break;
    case NVCV_IMAGE_FORMAT_U32:
        generateRandomInputImpl<uint32_t>(imageHandle);
        break;
    case NVCV_IMAGE_FORMAT_S32:
        generateRandomInputImpl<int32_t>(imageHandle);
        break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input image format!");
    }
}

void generateMedianFilterBoxFilterKernel(NVCVTensorHandle kernelHandle, int32_t kernelHeight, int32_t kernelWidth)
{
    generateBoxFilterKernelImpl(kernelHandle, kernelHeight, kernelWidth);
}

int32_t checkMedianFilterImageOutput(NVCVImageHandle refImageHandle, NVCVImageHandle pvaImageHandle)
{
    NVCVImageFormat format;
    nvcvImageGetFormat(refImageHandle, &format);

    switch (format)
    {
    case NVCV_IMAGE_FORMAT_U8:
    case NVCV_IMAGE_FORMAT_Y8:
    case NVCV_IMAGE_FORMAT_Y8_ER:
        return checkImageOutputImpl<uint8_t>(refImageHandle, pvaImageHandle);
    case NVCV_IMAGE_FORMAT_S8:
        return checkImageOutputImpl<int8_t>(refImageHandle, pvaImageHandle);
    case NVCV_IMAGE_FORMAT_U16:
    case NVCV_IMAGE_FORMAT_Y16:
    case NVCV_IMAGE_FORMAT_Y16_ER:
        return checkImageOutputImpl<uint16_t>(refImageHandle, pvaImageHandle);
    case NVCV_IMAGE_FORMAT_S16:
    case NVCV_IMAGE_FORMAT_2S16:
        return checkImageOutputImpl<int16_t>(refImageHandle, pvaImageHandle);
    case NVCV_IMAGE_FORMAT_U32:
        return checkImageOutputImpl<uint32_t>(refImageHandle, pvaImageHandle);
    case NVCV_IMAGE_FORMAT_S32:
        return checkImageOutputImpl<int32_t>(refImageHandle, pvaImageHandle);
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input image format!");
    }
}

void medianFilterRef(NVCVImageHandle inImageHandle, NVCVImageHandle outImageHandle, int32_t kernelHeight,
                     int32_t kernelWidth, NVCVTensorHandle kernel, NVCVBorderType borderType, int32_t borderValue)
{
    NVCVImageFormat format;
    nvcvImageGetFormat(inImageHandle, &format);

    switch (format)
    {
    case NVCV_IMAGE_FORMAT_U8:
    case NVCV_IMAGE_FORMAT_Y8:
    case NVCV_IMAGE_FORMAT_Y8_ER:
        medianFilterRefImpl<uint8_t>(inImageHandle, outImageHandle, kernelHeight, kernelWidth, kernel, borderType,
                                     borderValue);
        break;
    case NVCV_IMAGE_FORMAT_S8:
        medianFilterRefImpl<int8_t>(inImageHandle, outImageHandle, kernelHeight, kernelWidth, kernel, borderType,
                                    borderValue);
        break;
    case NVCV_IMAGE_FORMAT_U16:
    case NVCV_IMAGE_FORMAT_Y16:
    case NVCV_IMAGE_FORMAT_Y16_ER:
        medianFilterRefImpl<uint16_t>(inImageHandle, outImageHandle, kernelHeight, kernelWidth, kernel, borderType,
                                      borderValue);
        break;
    case NVCV_IMAGE_FORMAT_S16:
    case NVCV_IMAGE_FORMAT_2S16:
        medianFilterRefImpl<int16_t>(inImageHandle, outImageHandle, kernelHeight, kernelWidth, kernel, borderType,
                                     borderValue);
        break;
    case NVCV_IMAGE_FORMAT_U32:
        medianFilterRefImpl<uint32_t>(inImageHandle, outImageHandle, kernelHeight, kernelWidth, kernel, borderType,
                                      borderValue);
        break;
    case NVCV_IMAGE_FORMAT_S32:
        medianFilterRefImpl<int32_t>(inImageHandle, outImageHandle, kernelHeight, kernelWidth, kernel, borderType,
                                     borderValue);
        break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input image format!");
    }
}
