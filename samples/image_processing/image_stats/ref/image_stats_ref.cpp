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

#include "image_stats_ref.h"

#include <nvcv/Tensor.hpp>

#include <OpImageResize.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <type_traits>

#define MAX_PLANE_COUNT 2
#define REL_ERROR_THRESHOLD_SUM 1e-5
#define REL_ERROR_THRESHOLD_MEAN 1e-5
#define REL_ERROR_THRESHOLD_VAR 1e-3

namespace {

std::tuple<bool, bool, bool, bool, bool> unpackImageStatsFlags(uint32_t flags)
{
    bool pixelCountFlag = (flags & PVA_IMAGE_STAT_FLAG_PIXEL_COUNT) == PVA_IMAGE_STAT_FLAG_PIXEL_COUNT;
    bool sumFlag        = (flags & PVA_IMAGE_STAT_FLAG_SUM) == PVA_IMAGE_STAT_FLAG_SUM;
    bool meanFlag       = (flags & PVA_IMAGE_STAT_FLAG_MEAN) == PVA_IMAGE_STAT_FLAG_MEAN;
    bool varianceFlag   = (flags & PVA_IMAGE_STAT_FLAG_VARIANCE) == PVA_IMAGE_STAT_FLAG_VARIANCE;
    bool covarianceFlag = (flags & PVA_IMAGE_STAT_FLAG_COVARIANCE) == PVA_IMAGE_STAT_FLAG_COVARIANCE;
    return std::make_tuple(pixelCountFlag, sumFlag, meanFlag, varianceFlag, covarianceFlag);
}

template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value>>
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
                    imagePtr_h[i * linePitch + j * numChannels + c] =
                        static_cast<T>(rand() % (std::numeric_limits<T>::max() + 1));
                }
            }
        }
        imagePtr_h += height * linePitch;
    }
}

void generateRandomBinaryMaskImpl(NVCVImageHandle maskHandle, float maskRatio)
{
    NVCVImageData maskData;
    nvcvImageExportData(maskHandle, &maskData);
    NVCVImageFormat format = maskData.format;
    if (format != NVCV_IMAGE_FORMAT_U8)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid mask format!");
    }

    int seed = 365;
    std::mt19937 gen;
    gen.seed(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    uint8_t *maskPtr_h = NULL;
    CupvaMemGetHostPointer((void **)&maskPtr_h, (void *)maskData.buffer.strided.planes[0].basePtr);

    int height    = maskData.buffer.strided.planes[0].height;
    int width     = maskData.buffer.strided.planes[0].width;
    int linePitch = maskData.buffer.strided.planes[0].rowStride / sizeof(uint8_t);

    int maskCnt = 0;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            maskPtr_h[i * linePitch + j] = dist(gen) < maskRatio ? 1 : 0;
            maskCnt += maskPtr_h[i * linePitch + j];
        }
    }
}

template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value>>
void generateFixedImageInputImpl(NVCVImageHandle imageHandle, const char *assetDir, const char *fileName)
{
    NVCVImageData imageData;
    nvcvImageExportData(imageHandle, &imageData);

    T *imagePtr_h = NULL;
    CupvaMemGetHostPointer((void **)&imagePtr_h, (void *)imageData.buffer.strided.planes[0].basePtr);

    NVCVImageFormat format = imageData.format;
    int numPlanes          = imageData.buffer.strided.numPlanes;

    // calculate total image size
    int totalImageSize = 0;
    for (int p = 0; p < numPlanes; p++)
    {
        int height = imageData.buffer.strided.planes[p].height;
        int width  = imageData.buffer.strided.planes[p].width;

        int32_t numChannels;
        nvcvImageFormatGetPlaneNumChannels(format, p, &numChannels);

        totalImageSize += height * width * numChannels;
    }

    // load uint8_t data from raw file
    std::vector<uint8_t> rawImageData(totalImageSize);
    int err = ReadImageBuffer(fileName, assetDir, (void *)rawImageData.data(), totalImageSize);
    if (err)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_OPERATION, "Failed to load read image buffer!\n");
    }

    // fill data to image handle
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
                    imagePtr_h[i * linePitch + j * numChannels + c] =
                        (T)rawImageData[i * width * numChannels + j * numChannels + c];
                }
            }
        }
        imagePtr_h += height * linePitch;
    }
}

template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value>>
int32_t checkOutputImpl(NVCVImageHandle imageHandle, NVCVTensorHandle outTensorHandle, NVCVTensorHandle refTensorHandle)
{
    // get output results
    NVCVTensorData outTensorData;
    nvcvTensorExportData(outTensorHandle, &outTensorData);
    PVAImageStatOutput *outStats = NULL;
    CupvaMemGetHostPointer((void **)&outStats, (void *)outTensorData.buffer.strided.basePtr);

    // get reference results
    NVCVTensorData refTensorData;
    nvcvTensorExportData(refTensorHandle, &refTensorData);
    PVAImageStatOutput *refStats = NULL;
    CupvaMemGetHostPointer((void **)&refStats, (void *)refTensorData.buffer.strided.basePtr);

    // check pixel count
    if (outStats->pixelCount != refStats->pixelCount)
    {
        printf("Mismatch, pixel count: ref = %d, pva = %d\n", refStats->pixelCount, outStats->pixelCount);
        return 1;
    }

    double relErr;

    // check per channel sum
    for (int c = 0; c < 4; c++)
    {
        relErr = 0;
        if (refStats->sum[c] != 0)
        {
            relErr = std::abs((outStats->sum[c] - refStats->sum[c]) / refStats->sum[c]);
        }

        if (relErr > REL_ERROR_THRESHOLD_SUM)
        {
            printf("Mismatch, per channel sum: channel c=%d, ref = %f, pva = %f, relative error=%f > threshold %f\n", c,
                   refStats->sum[c], outStats->sum[c], relErr, REL_ERROR_THRESHOLD_SUM);
            return 1;
        }
    }

    // check per channel mean
    for (int c = 0; c < 4; c++)
    {
        relErr = 0;
        if (refStats->mean[c] != 0)
        {
            relErr = std::abs((outStats->mean[c] - refStats->mean[c]) / refStats->mean[c]);
        }

        if (relErr > REL_ERROR_THRESHOLD_MEAN)
        {
            printf("Mismatch, per channel mean: channel c=%d, ref = %f, pva = %f, relative error=%f > threshold %f\n",
                   c, refStats->mean[c], outStats->mean[c], relErr, REL_ERROR_THRESHOLD_MEAN);
            return 1;
        }
    }

    // check covariance matrix
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {

            relErr = 0;
            if (refStats->covariance[i][j] != 0)
            {
                relErr =
                    std::abs((outStats->covariance[i][j] - refStats->covariance[i][j]) / refStats->covariance[i][j]);
            }

            if (relErr > REL_ERROR_THRESHOLD_VAR)
            {
                printf(
                    "Mismatch, covariance matrix: idx=(%d, %d), ref = %f, pva = %f, relative error=%f > threshold %f\n",
                    i, j, refStats->covariance[i][j], outStats->covariance[i][j], relErr, REL_ERROR_THRESHOLD_VAR);
                return 1;
            }
        }
    }
    return 0;
}

template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value>>
void imageStatsRefImpl(NVCVImageHandle imageHandle, NVCVImageHandle maskHandle, NVCVTensorHandle refTensorHandle,
                       uint32_t flags)
{
    bool pixelCountFlag, sumFlag, meanFlag, varianceFlag, covarianceFlag;
    std::tie(pixelCountFlag, sumFlag, meanFlag, varianceFlag, covarianceFlag) = unpackImageStatsFlags(flags);

    // init data buffers
    NVCVImageData imageData;
    nvcvImageExportData(imageHandle, &imageData);
    T *imagePtr_h = NULL;
    CupvaMemGetHostPointer((void **)&imagePtr_h, (void *)imageData.buffer.strided.planes[0].basePtr);

    bool hasMask = (maskHandle != nullptr);

    T *maskPtr_h      = NULL;
    int maskLinePitch = 0;
    if (hasMask)
    {
        NVCVImageData maskData;
        nvcvImageExportData(maskHandle, &maskData);
        CupvaMemGetHostPointer((void **)&maskPtr_h, (void *)maskData.buffer.strided.planes[0].basePtr);
        maskLinePitch = maskData.buffer.strided.planes[0].rowStride / sizeof(T);
    }

    NVCVTensorData refTensorData;
    nvcvTensorExportData(refTensorHandle, &refTensorData);
    PVAImageStatOutput *refStats = NULL;
    CupvaMemGetHostPointer((void **)&refStats, (void *)refTensorData.buffer.strided.basePtr);

    NVCVImageFormat format = imageData.format;
    int numPlanes          = imageData.buffer.strided.numPlanes;

    int planeChannelNum[MAX_PLANE_COUNT] = {0};
    int planeWidth[MAX_PLANE_COUNT]      = {0};
    int planeHeight[MAX_PLANE_COUNT]     = {0};
    int planeLinePitch[MAX_PLANE_COUNT]  = {0};
    int planePixelCount[MAX_PLANE_COUNT] = {0};
    int planeImgOfst[MAX_PLANE_COUNT]    = {0};
    int scaleX[MAX_PLANE_COUNT]          = {0};
    int scaleY[MAX_PLANE_COUNT]          = {0};

    int totalChannelNum = 0;
    for (int p = 0; p < numPlanes; p++)
    {
        int height    = imageData.buffer.strided.planes[p].height;
        int width     = imageData.buffer.strided.planes[p].width;
        int linePitch = imageData.buffer.strided.planes[p].rowStride / sizeof(T);

        int32_t numChannels;
        nvcvImageFormatGetPlaneNumChannels(format, p, &numChannels);

        planeImgOfst[p] = p > 0 ? planeHeight[p - 1] * planeLinePitch[p - 1] : 0;

        planeChannelNum[p] = numChannels;
        planeWidth[p]      = width;
        planeHeight[p]     = height;
        planeLinePitch[p]  = linePitch;
        totalChannelNum += numChannels;
        // subsampling ratio for each plane
        scaleX[p] = std::ceil((float)planeWidth[0] / planeWidth[p]);
        scaleY[p] = std::ceil((float)planeHeight[0] / planeHeight[p]);
    }

    int imageWidth  = imageData.buffer.strided.planes[0].width;
    int imageHeight = imageData.buffer.strided.planes[0].height;

    // get pixel count
    refStats->pixelCount = 0;
    int imagePixelCount  = 0;
    if (pixelCountFlag)
    {
        for (int i = 0; i < imageHeight; i++)
        {
            for (int j = 0; j < imageWidth; j++)
            {
                int mask = hasMask ? (maskPtr_h[i * maskLinePitch + j]) : 1;
                imagePixelCount += mask;
            }
        }
        refStats->pixelCount = imagePixelCount;
    }

    // use double precision for intermediate results
    double tmpChannelSum[4];
    double tmpChannelMean[4];

    // init output values to 0
    memset(refStats->mean, 0, sizeof(refStats->mean));
    memset(refStats->sum, 0, sizeof(refStats->sum));
    memset(refStats->covariance, 0, sizeof(refStats->covariance));

    // calculate per channel sum
    if (sumFlag)
    {
        for (int c = 0; c < totalChannelNum; c++)
        {
            int p          = c >= planeChannelNum[0] ? 1 : 0;
            int channelIdx = p > 0 ? c - planeChannelNum[p - 1] : c;
            double sum     = 0;
            for (int i = 0; i < imageHeight; i++)
            {
                for (int j = 0; j < imageWidth; j++)
                {
                    int mask = hasMask ? (maskPtr_h[i * maskLinePitch + j]) : 1;
                    // for color space with subsampling, upsample by Nearest Neighbor
                    int y_idx = i / scaleY[p];
                    int x_idx = j / scaleX[p];
                    sum += imagePtr_h[planeImgOfst[p] + y_idx * planeLinePitch[p] + x_idx * planeChannelNum[p] +
                                      channelIdx] *
                           mask;
                }
            }
            tmpChannelSum[c] = sum;
            refStats->sum[c] = (float)sum;
        }
    }

    // calculate per channel mean
    if (meanFlag)
    {
        for (int c = 0; c < totalChannelNum; c++)
        {
            tmpChannelMean[c] = tmpChannelSum[c] / imagePixelCount;
            refStats->mean[c] = (float)tmpChannelMean[c];
        }
    }

    if (varianceFlag)
    {
        int divCnt = imagePixelCount > 1 ? imagePixelCount - 1 : 1;

        // calculate covariance matrix
        for (int c0 = 0; c0 < totalChannelNum; c0++)
        {
            int p0          = c0 >= planeChannelNum[0] ? 1 : 0;
            int w0          = planeWidth[p0];
            int h0          = planeHeight[p0];
            int lp0         = planeLinePitch[p0];
            int channelIdx0 = p0 > 0 ? c0 - planeChannelNum[p0 - 1] : c0;

            for (int c1 = c0; c1 < totalChannelNum; c1++)
            {
                // for variance only, skip c1>c0 indices
                if (!covarianceFlag && c1 > c0)
                {
                    break;
                }

                int p1          = c1 >= planeChannelNum[0] ? 1 : 0;
                int w1          = planeWidth[p1];
                int h1          = planeHeight[p1];
                int lp1         = planeLinePitch[p1];
                int channelIdx1 = p1 > 0 ? c1 - planeChannelNum[p1 - 1] : c1;

                double covSum = 0;
                for (int i = 0; i < imageHeight; i++)
                {
                    for (int j = 0; j < imageWidth; j++)
                    {
                        int mask = hasMask ? (maskPtr_h[i * maskLinePitch + j]) : 1;
                        // for color space with subsampling, upsample by Nearest Neighbor
                        int y0 = i / scaleY[p0];
                        int x0 = j / scaleX[p0];
                        int y1 = i / scaleY[p1];
                        int x1 = j / scaleX[p1];

                        covSum += (imagePtr_h[planeImgOfst[p0] + y0 * lp0 + x0 * planeChannelNum[p0] + channelIdx0] -
                                   tmpChannelMean[c0]) *
                                  (imagePtr_h[planeImgOfst[p1] + y1 * lp1 + x1 * planeChannelNum[p1] + channelIdx1] -
                                   tmpChannelMean[c1]) *
                                  mask;
                    }
                }

                double cov                   = covSum / divCnt;
                refStats->covariance[c0][c1] = (float)cov;

                if (c1 != c0)
                {
                    refStats->covariance[c1][c0] = refStats->covariance[c0][c1];
                }
            }
        }
    }
}

} // namespace

void generateImageStatsRandomImageInput(NVCVImageHandle imageHandle)
{
    NVCVImageFormat format;
    nvcvImageGetFormat(imageHandle, &format);

    NVCVDataType dataType;
    nvcvImageFormatGetPlaneDataType(format, 0, &dataType);

    switch (dataType)
    {
    case NVCV_DATA_TYPE_U8:
    case NVCV_DATA_TYPE_3U8:
        generateRandomInputImpl<uint8_t>(imageHandle);
        break;
    default:
        break;
    }
}

void generateImageStatsFixedImageInput(NVCVImageHandle imageHandle, const char *assetDir)
{
    NVCVImageFormat format;
    nvcvImageGetFormat(imageHandle, &format);

    switch (format)
    {
    case NVCV_IMAGE_FORMAT_U8:
        generateFixedImageInputImpl<uint8_t>(imageHandle, assetDir, "image_stats/kodim08_768x512_u8.raw");
        break;
    case NVCV_IMAGE_FORMAT_RGB8:
        generateFixedImageInputImpl<uint8_t>(imageHandle, assetDir, "image_stats/kodim08_768x512_rgb.raw");
        break;
    case NVCV_IMAGE_FORMAT_NV12:
        generateFixedImageInputImpl<uint8_t>(imageHandle, assetDir, "image_stats/kodim08_768x512_nv12.raw");
        break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input image format for fixed input!");
        break;
    }
}

void generateImageStatsRandomBinaryMask(NVCVImageHandle maskHandle, float maskRatio)
{
    NVCVImageFormat format;
    nvcvImageGetFormat(maskHandle, &format);

    switch (format)
    {
    case NVCV_IMAGE_FORMAT_U8:
        generateRandomBinaryMaskImpl(maskHandle, maskRatio);
        break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid mask format!");
    }
}

int32_t checkImageStatsOutput(NVCVImageHandle imageHandle, NVCVTensorHandle outTensorHandle,
                              NVCVTensorHandle refTensorHandle)
{
    NVCVImageFormat format;
    nvcvImageGetFormat(imageHandle, &format);

    NVCVDataType dataType;
    nvcvImageFormatGetPlaneDataType(format, 0, &dataType);

    switch (dataType)
    {
    case NVCV_DATA_TYPE_U8:
    case NVCV_DATA_TYPE_3U8:
        return checkOutputImpl<uint8_t>(imageHandle, outTensorHandle, refTensorHandle);
    default:
        return -1;
    }
}

void imageStatsRef(NVCVImageHandle imageHandle, NVCVImageHandle maskHandle, NVCVTensorHandle refTensorHandle,
                   uint32_t flags)
{
    NVCVImageFormat format;
    nvcvImageGetFormat(imageHandle, &format);

    NVCVDataType dataType;
    nvcvImageFormatGetPlaneDataType(format, 0, &dataType);

    switch (dataType)
    {
    case NVCV_DATA_TYPE_U8:
    case NVCV_DATA_TYPE_3U8:
        imageStatsRefImpl<uint8_t>(imageHandle, maskHandle, refTensorHandle, flags);
        break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input data type!");
        break;
    }
}
