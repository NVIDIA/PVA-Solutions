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

#include "hog_ref.h"

#include "ImageIO.h"

#include <nvcv/Tensor.hpp>
#include <nvcv/TensorData.hpp>

#include <PvaAllocator.h>
#include <cupva_host.h>
#include <nvcv/ImageFormat.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

#define DIFF_TOL (1)

namespace {
template<typename IN_TYPE, typename OUT_TYPE>
int32_t clamp(const IN_TYPE x, const OUT_TYPE min, const OUT_TYPE max)
{
    return static_cast<OUT_TYPE>(x < min ? min : (x > max) ? max : x);
}

template<typename T, typename = std::enable_if_t<std::is_same_v<T, uint8_t>>>
T deNormalize(double v)
{
    T maxNum = std::numeric_limits<T>::max();
    return static_cast<T>(std::max(0.0, std::min((double)maxNum, v * maxNum + 0.5)));
}

template<typename T, typename = std::enable_if_t<std::is_same_v<T, uint8_t>>>
RGB fetchRGBPixel(const int32_t ix, const int32_t iy, int32_t rx, int32_t ry, Size2D imgSize, Size2D inputSize,
                  NVCVImageFormat imgFormat, T *input)
{
    rx                    = clamp(rx, 0, imgSize.w - 1);
    ry                    = clamp(ry, 0, imgSize.h - 1);
    const int32_t offsetX = ix * imgSize.w + rx;
    const int32_t offsetY = iy * imgSize.h + ry;

    uint8_t cr, cg, cb;
    const int32_t planeSize = inputSize.w * inputSize.h;
    const int32_t offset    = offsetY * inputSize.w + offsetX;
    cr                      = input[offset];
    cg                      = input[offset + planeSize];
    cb                      = input[offset + 2 * planeSize];

    return {cr, cg, cb};
}

template<typename T>
void generateRandomInputImpl(NVCVImageHandle imageHandle)
{
    NVCVImageData imageData;
    nvcvImageExportData(imageHandle, &imageData);

    NVCVImageFormat format = imageData.format;
    int32_t numPlanes      = imageData.buffer.strided.numPlanes;

    T *imagePtr = NULL;
    CupvaMemGetHostPointer((void **)&imagePtr, (void *)imageData.buffer.strided.planes[0].basePtr);

    for (int32_t p = 0; p < numPlanes; p++)
    {
        int32_t height    = imageData.buffer.strided.planes[p].height;
        int32_t width     = imageData.buffer.strided.planes[p].width;
        int32_t linePitch = imageData.buffer.strided.planes[p].rowStride / sizeof(T);

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
                    imagePtr[j * linePitch + i] =
                        static_cast<T>(rand() % (std::numeric_limits<T>::max() - std::numeric_limits<T>::min() + 1) +
                                       std::numeric_limits<T>::min());
                }
                else
                {
                    imagePtr[j * linePitch + i] = static_cast<T>(rand() % (std::numeric_limits<T>::max() + 1));
                }
            }
        }

        imagePtr += height * linePitch;
    }
}

template<typename T, typename = std::enable_if_t<std::is_same_v<T, uint8_t>>>
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

    int32_t height, width, channels;
    nvcv::TensorShape shape(refTensorData.shape, refTensorData.rank, refTensorData.layout);
    height   = shape[0];
    width    = shape[1];
    channels = shape[2];

    int32_t err = 0;
    for (int32_t y = 0; y < height; y++)
    {
        for (int32_t x = 0; x < width; x++)
        {
            for (int32_t c = 0; c < channels; c++)
            {
                // PVA output is CHW format, ref is always HWC format
                T refData = ref[y * width * channels + x * channels + c];
                T pvaData = pva[c * width * height + y * width + x];
                if (abs(refData - pvaData) > DIFF_TOL)
                {
                    if (err < 10)
                        printf("Mismatch[%d, %d, %d]: ref = %u, pva = %u\n", y, x, c, refData, pvaData);
                    err++;
                }
            }
        }
    }
    if (err)
        printf("Error rate: %.2f%%\n", (float)err / (float)(height * width * channels) * 100.0);
    return err;
}

template<typename T, typename = std::enable_if_t<std::is_same_v<T, uint8_t>>>
void hogRefImpl(NVCVImageHandle inImageHandle, NVCVTensorHandle outTensorHandle)
{
    const int32_t numOrientations = 18;
    const int32_t cellSize        = 4;
    const Size2D featureSize      = {24, 24};
    const Size2D objectsNum       = {6, 6};
    const Size2D padding          = {1, 1};

    NVCVImageData inImageData;
    nvcvImageExportData(inImageHandle, &inImageData);
    T *inImagePtr = NULL;
    CupvaMemGetHostPointer((void **)&inImagePtr, (void *)inImageData.buffer.strided.planes[0].basePtr);
    NVCVImageFormat imageFormat = inImageData.format;

    NVCVTensorData outTensorData;
    nvcvTensorExportData(outTensorHandle, &outTensorData);
    T *outTensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&outTensorPtr, (void *)outTensorData.buffer.strided.basePtr);

    // initialize HOG parameters
    Size2D imgSize   = {cellSize * featureSize.w, cellSize * featureSize.h};
    Size2D inputSize = {imgSize.w * objectsNum.w, imgSize.h * objectsNum.h};
    int32_t channels = numOrientations;

    Size2D outputSize        = {featureSize.w * channels * objectsNum.w, featureSize.h * objectsNum.h};
    Size2D featureSizePadded = {featureSize.w + 2 * padding.w, featureSize.h + 2 * padding.h};
    Size2D histSize = {numOrientations * featureSizePadded.w * objectsNum.w, featureSizePadded.h * objectsNum.h};
    Size2D normSize = {featureSizePadded.w * objectsNum.w, featureSizePadded.h * objectsNum.h};

    std::vector<double> hist(histSize.w * histSize.h);
    std::vector<double> norm(normSize.w * normSize.h);

    int32_t inputPixels = inputSize.w * inputSize.h;
    if (imageFormat == NVCV_IMAGE_FORMAT_RGB8p)
    {
        inputPixels = inputPixels * 3;
    }
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported image format");
    }
    std::vector<T> input(inputPixels);

    // epsilon to avoid division by zero
    const double eps = 0.0001;
    // number of orientations
    const int32_t numOrient = numOrientations;
    // keep the code stype in VPI
    const int32_t sbin = cellSize;

    // size of HOG features
    const Size2D outSize = {inputSize.w / sbin, inputSize.h / sbin};

    // block size
    const Size2D blockSize = {outSize.w + (imgSize.w == 0 ? 0 : 2 * padding.w * inputSize.w / imgSize.w),
                              outSize.h + (imgSize.h == 0 ? 0 : 2 * padding.h * inputSize.h / imgSize.h)};

    // size of visible
    const Size2D visible{blockSize.w * sbin, blockSize.h * sbin};

    // initialize histogram, norm, output feature matrices
    std::fill(hist.begin(), hist.end(), 0);
    std::fill(norm.begin(), norm.end(), 0);

    Size2D rblockSize = {imgSize.w / sbin + 2 * padding.w, imgSize.h / sbin + 2 * padding.h};

    for (int32_t y = 0; y < visible.h; y++)
    {
        // index of subimage row
        int32_t iy = y / (rblockSize.h * sbin);

        // y, relative to the current image
        int32_t ry = y - iy * (rblockSize.h * sbin);

        if (ry == 0 || ry >= rblockSize.h * sbin - 1)
        {
            continue;
        }

        for (int32_t x = 0; x < visible.w; x++)
        {
            // index of subimage col
            int32_t ix = x / (rblockSize.w * sbin);

            // x, relative to the current image
            int32_t rx = x - ix * (rblockSize.w * sbin);

            if (rx == 0 || rx >= rblockSize.w * sbin - 1)
            {
                continue;
            }

            auto rgbL = fetchRGBPixel(ix, iy, rx - 1 - sbin, ry - sbin, imgSize, inputSize, imageFormat, inImagePtr);
            auto rgbR = fetchRGBPixel(ix, iy, rx + 1 - sbin, ry - sbin, imgSize, inputSize, imageFormat, inImagePtr);
            auto rgbT = fetchRGBPixel(ix, iy, rx - sbin, ry - 1 - sbin, imgSize, inputSize, imageFormat, inImagePtr);
            auto rgbB = fetchRGBPixel(ix, iy, rx - sbin, ry + 1 - sbin, imgSize, inputSize, imageFormat, inImagePtr);

            // blue image channel
            int32_t dyb = (int32_t)rgbB.b - (int32_t)rgbT.b;
            int32_t dxb = (int32_t)rgbR.b - (int32_t)rgbL.b;
            int32_t vb  = dxb * dxb + dyb * dyb;

            // green image channel
            int32_t dyg = (int32_t)rgbB.g - (int32_t)rgbT.g;
            int32_t dxg = (int32_t)rgbR.g - (int32_t)rgbL.g;
            int32_t vg  = dxg * dxg + dyg * dyg;

            // red image channel
            int32_t dy = (int32_t)rgbB.r - (int32_t)rgbT.r;
            int32_t dx = (int32_t)rgbR.r - (int32_t)rgbL.r;
            int32_t v  = dx * dx + dy * dy;

            // pick the channel with the strongest gradient
            if (vg > v)
            {
                v  = vg;
                dx = dxg;
                dy = dyg;
            }
            if (vb > v)
            {
                v  = vb;
                dx = dxb;
                dy = dyb;
            }

            double phase   = std::atan2(dy, dx);
            int32_t best_o = (phase + M_PI) / (2 * M_PI) * numOrient + 0.5f;
            if (best_o == numOrient)
            {
                best_o = 0;
            }

            // add to 4 historgrams around pixel using bilinear interpolation
            // block index for x/y from a subimage
            double yp   = ((double)ry + 0.5) / (double)sbin - 0.5;
            double xp   = ((double)rx + 0.5) / (double)sbin - 0.5;
            int32_t iyp = (int32_t)std::floor(yp);
            int32_t ixp = (int32_t)std::floor(xp);
            double vy0  = yp - iyp;
            double vx0  = xp - ixp;
            double vy1  = 1.0 - vy0;
            double vx1  = 1.0 - vx0;
            double fv   = std::sqrt((double)v);

            // fill the value into the 4 neighborhood cells
            if (iyp >= 0 && ixp >= 0)
            {
                const int32_t offsetX = (rblockSize.w * ix + ixp) * numOrient + best_o;
                const int32_t offsetY = rblockSize.h * iy + iyp;
                hist[offsetY * histSize.w + offsetX] += vy1 * vx1 * fv;
            }

            if (iyp >= 0 && ixp + 1 < rblockSize.w)
            {
                const int32_t offsetX = (rblockSize.w * ix + ixp + 1) * numOrient + best_o;
                const int32_t offsetY = rblockSize.h * iy + iyp;
                hist[offsetY * histSize.w + offsetX] += vx0 * vy1 * fv;
            }

            if (iyp + 1 < rblockSize.h && ixp >= 0)
            {
                const int32_t offsetX = (rblockSize.w * ix + ixp) * numOrient + best_o;
                const int32_t offsetY = rblockSize.h * iy + iyp + 1;
                hist[offsetY * histSize.w + offsetX] += vy0 * vx1 * fv;
            }

            if (iyp + 1 < rblockSize.h && ixp + 1 < rblockSize.w)
            {

                const int32_t offsetX = (rblockSize.w * ix + ixp + 1) * numOrient + best_o;
                const int32_t offsetY = rblockSize.h * iy + iyp + 1;
                hist[offsetY * histSize.w + offsetX] += vy0 * vx0 * fv;
            }
        } // for y
    }     // for x

    // compute the energy in each cell by summing over orientation
    for (int32_t y = 0; y < blockSize.h; y++)
    {
        for (int32_t x = 0; x < blockSize.w; x++)
        {
            double energy      = 0.0;
            const int32_t base = y * histSize.w + x * numOrient;

            for (int32_t o = 0; o < numOrient / 2; o++)
            {
                double v0  = hist[base + o];
                double v1  = hist[base + o + numOrient / 2];
                double sum = v0 + v1;
                energy += sum * sum;
            }

            norm[y * normSize.w + x] = energy;
        }
    }

    // compute the features
    for (int32_t y = 0; y < outSize.h; y++)
    {
        for (int32_t x = 0; x < outSize.w; x++)
        {
            double n1, n2, n3, n4;
            const double *src;

            // subimage index
            int32_t ix = x / (imgSize.w / cellSize), iy = y / (imgSize.h / cellSize);

            // feature index relative to the subimage
            int32_t rx = x - ix * (imgSize.w / cellSize), ry = y - iy * (imgSize.h / cellSize);

            {
                int32_t cx = 0, cy = 0;

                auto p = [&](int32_t i, int32_t j) {
                    int32_t ny = std::clamp(iy * rblockSize.h + cy + i, 0, normSize.h - 1);
                    int32_t nx = std::clamp(ix * rblockSize.w + cx + j, 0, normSize.w - 1);
                    return norm[ny * normSize.w + nx];
                };

                cx = rx + 1, cy = ry + 1;
                n1 = 1.0f / sqrt(p(0, 0) + p(0, 1) + p(1, 0) + p(1, 1) + eps);

                cx = rx + 1, cy = ry;
                n2 = 1.0f / sqrt(p(0, 0) + p(0, 1) + p(1, 0) + p(1, 1) + eps);

                cx = rx, cy = ry + 1;
                n3 = 1.0f / sqrt(p(0, 0) + p(0, 1) + p(1, 0) + p(1, 1) + eps);

                cx = rx, cy = ry;
                n4 = 1.0f / sqrt(p(0, 0) + p(0, 1) + p(1, 0) + p(1, 1) + eps);
            }

            double t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0;

            int32_t fcur = 0;

            src = &hist[(iy * rblockSize.h + ry + padding.h) * histSize.w +
                        (ix * rblockSize.w + rx + padding.w) * numOrient];
            for (int32_t o = 0; o < numOrient; o++)
            {
                double val = *src++;
                double h1  = std::min(val * n1, 0.2);
                double h2  = std::min(val * n2, 0.2);
                double h3  = std::min(val * n3, 0.2);
                double h4  = std::min(val * n4, 0.2);

                outTensorPtr[y * outputSize.w + x * channels + fcur] = deNormalize<T>(0.5 * (h1 + h2 + h3 + h4));
                fcur++;
            }

            assert(fcur == channels);
        } // for x
    }     // for y
}

} // namespace

void generateRandomHOGInput(NVCVImageHandle imageHandle)
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
        generateRandomInputImpl<uint8_t>(imageHandle);
        break;
    default:
        break;
    }
}

int32_t checkHOGOutput(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle)
{
    NVCVTensorData refTensorData;
    nvcvTensorExportData(refTensorHandle, &refTensorData);

    switch (refTensorData.dtype)
    {
    case NVCV_DATA_TYPE_U8:
        return checkOutputImpl<uint8_t>(refTensorHandle, pvaTensorHandle);
    default:
        return -1;
    }
}

void hogRef(NVCVImageHandle inImageHandle, NVCVTensorHandle refTensorHandle)
{
    NVCVImageFormat format;
    nvcvImageGetFormat(inImageHandle, &format);

    NVCVDataType dataType;
    nvcvImageFormatGetPlaneDataType(format, 0, &dataType);
    switch (dataType)
    {
    case NVCV_DATA_TYPE_U8:
    case NVCV_DATA_TYPE_2U8:
    case NVCV_DATA_TYPE_3U8:
    case NVCV_DATA_TYPE_4U8:
        hogRefImpl<uint8_t>(inImageHandle, refTensorHandle);
        break;
    default:
        break;
    }
}

void loadHOGDataFromFile(NVCVImageHandle imageHandle, const char *directory, const char *filename)
{
    NVCVImageData imageData;
    nvcvImageExportData(imageHandle, &imageData);

    uint8_t *imagePtr_h = NULL;
    CupvaMemGetHostPointer((void **)&imagePtr_h, (void *)imageData.buffer.strided.planes[0].basePtr);

    NVCVImageFormat format = imageData.format;
    int32_t numPlanes      = imageData.buffer.strided.numPlanes;

    // calculate total image size
    int32_t totalImageSize = 0;
    for (int32_t p = 0; p < numPlanes; p++)
    {
        int32_t height = imageData.buffer.strided.planes[p].height;
        int32_t width  = imageData.buffer.strided.planes[p].width;

        int32_t numChannels;
        nvcvImageFormatGetPlaneNumChannels(format, p, &numChannels);

        totalImageSize += height * width * numChannels;
    }

    // load uint8_t data from raw file
    std::vector<uint8_t> rawImageData(totalImageSize);
    int32_t err = ReadImageBuffer(filename, directory, (void *)rawImageData.data(), totalImageSize);
    if (err)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_OPERATION, "Failed to load read image buffer!\n");
    }

    // fill data to image handle
    for (int32_t p = 0; p < numPlanes; p++)
    {
        int32_t height    = imageData.buffer.strided.planes[p].height;
        int32_t width     = imageData.buffer.strided.planes[p].width;
        int32_t linePitch = imageData.buffer.strided.planes[p].rowStride / sizeof(uint8_t);

        int32_t numChannels;
        nvcvImageFormatGetPlaneNumChannels(format, p, &numChannels);

        for (int32_t i = 0; i < height; i++)
        {
            for (int32_t j = 0; j < width; j++)
            {
                for (int32_t c = 0; c < numChannels; c++)
                {
                    imagePtr_h[i * linePitch + j * numChannels + c] =
                        (uint8_t)rawImageData[i * width * numChannels + j * numChannels + c];
                }
            }
        }
        imagePtr_h += height * linePitch;
    }
}