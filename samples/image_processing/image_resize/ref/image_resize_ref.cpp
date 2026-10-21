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

#include "image_resize_ref.h"

#include <nvcv/Tensor.hpp>

#include <OpImageResize.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <type_traits>

#define QBIT (16)
#define QBIT_HIGH (32)

// nearest neighbor
#define MAX_NN_U8_SRC_BUF_SIZE (98304)
#define MAX_NN_U8_DST_BUF_SIZE (16384)
#define MAX_NN_NV12_SRC_BUF_SIZE (63488)
#define MAX_NN_NV12_DST_BUF_SIZE (16384)
#define MAX_NN_RGB8_SRC_BUF_SIZE (14336)
#define MAX_NN_RGB8_DST_BUF_SIZE (16384)
#define MAX_NN_RGBA8_SRC_BUF_SIZE (14080)
#define MAX_NN_RGBA8_DST_BUF_SIZE (14528)
#define MAX_NN_U16_SRC_BUF_SIZE (32768)
#define MAX_NN_U16_DST_BUF_SIZE (16384)
// bilinear interpolation
#define MAX_BL_U8_SRC_BUF_SIZE (43680)
#define MAX_BL_U8_DST_BUF_SIZE (43680)
#define MAX_BL_RGB8_SRC_BUF_SIZE (21845)
#define MAX_BL_RGB8_DST_BUF_SIZE (21845)
#define MAX_BL_RGBA8_SRC_BUF_SIZE (16384)
#define MAX_BL_RGBA8_DST_BUF_SIZE (16384)
#define MAX_BL_U16_SRC_BUF_SIZE (54000)
#define MAX_BL_U16_DST_BUF_SIZE (32768)

namespace {
template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value || std::is_same<T, uint16_t>::value>>
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

        int32_t bitsPerPixel;
        nvcvImageFormatGetPlaneBitsPerPixel(format, p, &bitsPerPixel);

        int channelsPerPixel = (bitsPerPixel / CHAR_BIT) / sizeof(T);

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                for (int c = 0; c < channelsPerPixel; c++)
                {
                    imagePtr_h[i * linePitch + j * channelsPerPixel + c] =
                        static_cast<T>(rand() % (std::numeric_limits<T>::max() + 1));
                }
            }
        }
        imagePtr_h += height * linePitch;
    }
}

template<typename T>
int32_t checkImageOutputImpl(NVCVImageHandle refImageHandle, NVCVImageHandle pvaImageHandle, uint8_t threshold)
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

                    int err = abs(refData - cmpData);
                    if (err > threshold)
                    {
                        printf("Mismatch, plane %d, [%d, %d, %d]: ref = %u, pva = %u, abs err=%d, err threshold=%d\n",
                               p, y, x, c, refData, cmpData, err, threshold);
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

template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value || std::is_same<T, uint16_t>::value>>
std::tuple<int, int, int, int> findBestTileSize(int srcWidth, int srcHeight, int dstWidth, int dstHeight,
                                                int channelNum, NVCVImageFormat imageFormat,
                                                PVAInterpolationType resizeType)
{
    int maxSrcBufSize, maxDstBufSize;
    int maxPreBufSize, maxPostBufSize;
    if (resizeType == PVA_INTERPOLATION_NN)
    {
        switch (imageFormat)
        {
        case NVCV_IMAGE_FORMAT_U8:
            maxSrcBufSize = MAX_NN_U8_SRC_BUF_SIZE;
            maxDstBufSize = MAX_NN_U8_DST_BUF_SIZE;
            break;
        case NVCV_IMAGE_FORMAT_NV12:
        case NVCV_IMAGE_FORMAT_NV12_ER:
            maxSrcBufSize  = MAX_NN_NV12_SRC_BUF_SIZE;
            maxDstBufSize  = MAX_NN_NV12_DST_BUF_SIZE;
            maxPreBufSize  = MAX_NN_NV12_SRC_BUF_SIZE / 4;
            maxPostBufSize = MAX_NN_NV12_DST_BUF_SIZE / 4;
            break;
        case NVCV_IMAGE_FORMAT_RGB8:
            maxSrcBufSize  = MAX_NN_RGB8_SRC_BUF_SIZE * 3;
            maxDstBufSize  = MAX_NN_RGB8_DST_BUF_SIZE * 3;
            maxPreBufSize  = MAX_NN_RGB8_SRC_BUF_SIZE;
            maxPostBufSize = MAX_NN_RGB8_DST_BUF_SIZE;
            break;
        case NVCV_IMAGE_FORMAT_RGBA8:
            maxSrcBufSize  = MAX_NN_RGBA8_SRC_BUF_SIZE * 4;
            maxDstBufSize  = MAX_NN_RGBA8_DST_BUF_SIZE * 4;
            maxPreBufSize  = MAX_NN_RGBA8_SRC_BUF_SIZE;
            maxPostBufSize = MAX_NN_RGBA8_DST_BUF_SIZE;
            break;
        case NVCV_DATA_TYPE_U16:
            maxSrcBufSize = MAX_NN_U16_SRC_BUF_SIZE;
            maxDstBufSize = MAX_NN_U16_DST_BUF_SIZE;
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }
    }
    else
    {
        switch (imageFormat)
        {
        case NVCV_IMAGE_FORMAT_U8:
        case NVCV_IMAGE_FORMAT_NV12:
        case NVCV_IMAGE_FORMAT_NV12_ER:
            maxSrcBufSize  = MAX_BL_U8_SRC_BUF_SIZE;
            maxDstBufSize  = MAX_BL_U8_DST_BUF_SIZE;
            maxPreBufSize  = MAX_BL_U8_SRC_BUF_SIZE;
            maxPostBufSize = MAX_BL_U8_DST_BUF_SIZE;
            break;
        case NVCV_IMAGE_FORMAT_RGB8:
            maxSrcBufSize  = MAX_BL_RGB8_SRC_BUF_SIZE * 3;
            maxDstBufSize  = MAX_BL_RGB8_DST_BUF_SIZE * 3;
            maxPreBufSize  = MAX_BL_RGB8_SRC_BUF_SIZE;
            maxPostBufSize = MAX_BL_RGB8_DST_BUF_SIZE;
            break;
        case NVCV_IMAGE_FORMAT_RGBA8:
            maxSrcBufSize  = MAX_BL_RGBA8_SRC_BUF_SIZE * 4;
            maxDstBufSize  = MAX_BL_RGBA8_DST_BUF_SIZE * 4;
            maxPreBufSize  = MAX_BL_RGBA8_SRC_BUF_SIZE;
            maxPostBufSize = MAX_BL_RGBA8_DST_BUF_SIZE;
            break;
        case NVCV_DATA_TYPE_U16:
            maxSrcBufSize = MAX_BL_U16_SRC_BUF_SIZE;
            maxDstBufSize = MAX_BL_U16_DST_BUF_SIZE;
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }
    }

    // for dlut auto-idx, patch width and height must be U8, thus w1/h1 must be less than 256
    // tile width is better 64 aligned
    std::vector<int> dstTileWList = {192, 128, 64, 32};

    // tile height is better 32 aligned
    std::vector<int> dstTileHList = {224, 192, 160, 128, 96, 64, 32};

    float scaleX        = srcWidth * 1.0f / dstWidth;
    float scaleY        = srcHeight * 1.0f / dstHeight;
    bool transpose_flag = channelNum > 1;

    int minTileCnt = -1;
    int optW0      = -1;
    int optH0      = -1;
    int optW1      = -1;
    int optH1      = -1;

    for (auto w1 : dstTileWList)
    {
        for (auto h1 : dstTileHList)
        {
            if (w1 >= dstWidth || h1 >= dstHeight)
            {
                continue;
            }

            int w0 = std::ceil(scaleX * w1) + 2;
            int h0 = std::ceil(scaleY * h1) + 2;

            // src with transpose load, align to 64k + 2
            int lp0 = transpose_flag ? (w0 * channelNum - 2 + 63) / 64 * 64 + 2 : w0;
            // check src buf size: align buffer size to 64B for GSDF
            int srcTileBufSize = (lp0 * h0 + 63) / 64 * 64;
            if (srcTileBufSize > maxSrcBufSize)
            {
                continue;
            }

            // dst with transpose store, align to 64k + 2
            int lp1 = transpose_flag ? (w1 * channelNum - 2 + 63) / 64 * 64 + 2 : w1;
            // check dst buf size
            if (lp1 * h1 > maxDstBufSize)
            {
                continue;
            }

            // if has pre/post processing
            if constexpr (std::is_same<T, uint8_t>::value)
            {
                if (resizeType == PVA_INTERPOLATION_LINEAR || transpose_flag)
                {
                    // check pre buf size
                    int inputW = transpose_flag ? h0 : w0;
                    int inputH = transpose_flag ? w0 : h0;
                    // align table line pitch to 32k+2 for bilinear interpolation
                    int tableLp = (resizeType == PVA_INTERPOLATION_LINEAR) ? (inputW - 2 + 31) / 32 * 32 + 2 : inputW;

                    // pre-processing kernel writes to buffer by multiples of 32
                    int preBufSize = (tableLp * inputH + 31) / 32 * 32;
                    if (preBufSize > maxPreBufSize)
                    {
                        continue;
                    }

                    // check post buf size
                    int outputW  = transpose_flag ? h1 : w1;
                    int outputH  = transpose_flag ? w1 : h1;
                    int outputLp = outputW;

                    if (outputLp * outputH > maxPostBufSize)
                    {
                        continue;
                    }
                }
            }

            // check RDF padding size: < 256
            int ntile_w = (dstWidth + w1 - 1) / w1;
            int pad_w   = (ntile_w * w1 - dstWidth) * channelNum;
            if (pad_w >= 256)
            {
                continue;
            }

            int ntile_h = (dstHeight + h1 - 1) / h1;
            int ntile   = ntile_w * ntile_h;

            // update the optimal solution: minimal tile counts
            if (minTileCnt < 0 || ntile < minTileCnt)
            {
                // for U8 data types, prefer 64 aligned tile width
                if constexpr (std::is_same<T, uint8_t>::value)
                {
                    if (w1 == 32 && optW1 > 32)
                    {
                        continue;
                    }
                }

                minTileCnt = ntile;
                optW0      = w0;
                optH0      = h0;
                optW1      = w1;
                optH1      = h1;
            }
        }
    }
    if (optW0 > 0)
    {
        return std::make_tuple(optW0, optH0, optW1, optH1);
    }
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Could not find valid tile size");
    }
}

template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value || std::is_same<T, uint16_t>::value>>
void imageResizeNearestNeighborFixedPointDLUTRefImpl(NVCVImageHandle inImageHandle, NVCVImageHandle outImageHandle)
{
    NVCVImageData inImageData;
    nvcvImageExportData(inImageHandle, &inImageData);
    T *inImagePtr_h = NULL;
    CupvaMemGetHostPointer((void **)&inImagePtr_h, (void *)inImageData.buffer.strided.planes[0].basePtr);

    NVCVImageData outImageData;
    nvcvImageExportData(outImageHandle, &outImageData);
    T *outImagePtr_h = NULL;
    CupvaMemGetHostPointer((void **)&outImagePtr_h, (void *)outImageData.buffer.strided.planes[0].basePtr);

    NVCVImageFormat format = inImageData.format;
    int numPlanes          = inImageData.buffer.strided.numPlanes;

    int srcHeight = inImageData.buffer.strided.planes[0].height;
    int srcWidth  = inImageData.buffer.strided.planes[0].width;

    int dstHeight = outImageData.buffer.strided.planes[0].height;
    int dstWidth  = outImageData.buffer.strided.planes[0].width;

    double scaleX = (double)srcWidth / dstWidth;
    double scaleY = (double)srcHeight / dstHeight;

    int32_t scaleX_q = std::floor(scaleX * (1lu << QBIT));
    int32_t scaleY_q = std::floor(scaleY * (1lu << QBIT));

    int64_t scaleX_q_high = std::floor(scaleX * (1llu << QBIT_HIGH));
    int64_t scaleY_q_high = std::floor(scaleY * (1llu << QBIT_HIGH));

    for (int p = 0; p < numPlanes; p++)
    {
        int srcH  = inImageData.buffer.strided.planes[p].height;
        int srcW  = inImageData.buffer.strided.planes[p].width;
        int srcLp = inImageData.buffer.strided.planes[p].rowStride / sizeof(T);

        int dstH  = outImageData.buffer.strided.planes[p].height;
        int dstW  = outImageData.buffer.strided.planes[p].width;
        int dstLp = outImageData.buffer.strided.planes[p].rowStride / sizeof(T);

        int32_t bitsPerPixel;
        nvcvImageFormatGetPlaneBitsPerPixel(format, p, &bitsPerPixel);
        int channelsPerPixel = (bitsPerPixel / CHAR_BIT) / sizeof(T);

        // calculate dynamic tile size
        int w0, h0, dstTileWidth, dstTileHeight;
        std::tie(w0, h0, dstTileWidth, dstTileHeight) =
            findBestTileSize<T>(srcW, srcH, dstW, dstH, channelsPerPixel, format, PVA_INTERPOLATION_NN);

        int nTileX = (dstW + dstTileWidth - 1) / dstTileWidth;
        int nTileY = (dstH + dstTileHeight - 1) / dstTileHeight;

        for (int32_t i = 0; i < nTileY; i++)
        {
            // use pixel center as coordinates: src_coordinate = (dst_idx + 0.5f) * scale - 0.5f;
            int64_t y_start_q_high = i * dstTileHeight * scaleY_q_high + scaleY_q_high / 2 - (1llu << QBIT_HIGH) / 2;
            int32_t y_start_idx    = static_cast<int32_t>(y_start_q_high >> QBIT_HIGH);
            int32_t y_start_ofst_i = static_cast<int32_t>(y_start_q_high >> (QBIT_HIGH - QBIT)) - (y_start_idx << QBIT);

            for (int32_t j = 0; j < nTileX; j++)
            {
                // use pixel center as coordinates: src_coordinate = (dst_idx + 0.5f) * scale - 0.5f;
                int64_t x_start_q_high = j * dstTileWidth * scaleX_q_high + scaleX_q_high / 2 - (1llu << QBIT_HIGH) / 2;
                int32_t x_start_idx    = static_cast<int32_t>(x_start_q_high >> QBIT_HIGH);
                int32_t x_start_ofst_i =
                    static_cast<int32_t>(x_start_q_high >> (QBIT_HIGH - QBIT)) - (x_start_idx << QBIT);

                // pixels in the tile
                for (int32_t k = 0; (k < dstTileHeight && i * dstTileHeight + k < dstH); k++)
                {
                    int32_t y_q = k * scaleY_q + y_start_ofst_i;
                    // find nearest int: floor(y + 0.5f)
                    int32_t iy = (y_q + (1u << QBIT) / 2) >> QBIT;

                    iy += y_start_idx;

                    // out-of-range: boundary-pixel-extention
                    iy = std::min(std::max(iy, 0), srcH - 1);

                    for (int32_t l = 0; (l < dstTileWidth && j * dstTileWidth + l < dstW); l++)
                    {

                        int32_t x_q = l * scaleX_q + x_start_ofst_i;
                        // find nearest int: floor(x + 0.5f)
                        int32_t ix = (x_q + (1u << QBIT) / 2) >> QBIT;

                        ix += x_start_idx;

                        // out-of-range: boundary-pixel-extention
                        ix = std::min(std::max(ix, 0), srcW - 1);

                        for (int32_t c = 0; c < channelsPerPixel; c++)
                        {
                            int dst_idx =
                                (i * dstTileHeight + k) * dstLp + (j * dstTileWidth + l) * channelsPerPixel + c;
                            outImagePtr_h[dst_idx] = inImagePtr_h[iy * srcLp + ix * channelsPerPixel + c];
                        }
                    }
                }
            }
        }
        inImagePtr_h += srcH * srcLp;
        outImagePtr_h += dstH * dstLp;
    }
}

template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value || std::is_same<T, uint16_t>::value>>
void imageResizeBilinearInterpFixedPointDLUTRefImpl(NVCVImageHandle inImageHandle, NVCVImageHandle outImageHandle)
{
    NVCVImageData inImageData;
    nvcvImageExportData(inImageHandle, &inImageData);
    T *inImagePtr_h = NULL;
    CupvaMemGetHostPointer((void **)&inImagePtr_h, (void *)inImageData.buffer.strided.planes[0].basePtr);

    NVCVImageData outImageData;
    nvcvImageExportData(outImageHandle, &outImageData);
    T *outImagePtr_h = NULL;
    CupvaMemGetHostPointer((void **)&outImagePtr_h, (void *)outImageData.buffer.strided.planes[0].basePtr);

    NVCVImageFormat format = inImageData.format;
    int numPlanes          = inImageData.buffer.strided.numPlanes;

    int srcHeight = inImageData.buffer.strided.planes[0].height;
    int srcWidth  = inImageData.buffer.strided.planes[0].width;

    int dstHeight = outImageData.buffer.strided.planes[0].height;
    int dstWidth  = outImageData.buffer.strided.planes[0].width;

    double scaleX = (double)srcWidth / dstWidth;
    double scaleY = (double)srcHeight / dstHeight;

    int32_t scaleX_q = std::floor(scaleX * (1lu << QBIT));
    int32_t scaleY_q = std::floor(scaleY * (1lu << QBIT));

    int64_t scaleX_q_high = std::floor(scaleX * (1llu << QBIT_HIGH));
    int64_t scaleY_q_high = std::floor(scaleY * (1llu << QBIT_HIGH));

    // rounding function used in CupvaSampler on VPU
    auto roundDiff = [&](int64_t left, int64_t right, int32_t frac) {
        return ((right - left) * frac + (1u << QBIT) / 2u) >> QBIT;
    };

    for (int p = 0; p < numPlanes; p++)
    {
        int srcH  = inImageData.buffer.strided.planes[p].height;
        int srcW  = inImageData.buffer.strided.planes[p].width;
        int srcLp = inImageData.buffer.strided.planes[p].rowStride / sizeof(T);

        int dstH  = outImageData.buffer.strided.planes[p].height;
        int dstW  = outImageData.buffer.strided.planes[p].width;
        int dstLp = outImageData.buffer.strided.planes[p].rowStride / sizeof(T);

        int32_t bitsPerPixel;
        nvcvImageFormatGetPlaneBitsPerPixel(format, p, &bitsPerPixel);
        int channelsPerPixel = (bitsPerPixel / CHAR_BIT) / sizeof(T);

        // for multi-channel inputs, transposed single channel image is used in the interpolation stage
        bool transpose_flag = (channelsPerPixel > 1);

        // calculate dynamic tile size
        int w0, h0, dstTileWidth, dstTileHeight;
        std::tie(w0, h0, dstTileWidth, dstTileHeight) =
            findBestTileSize<T>(srcW, srcH, dstW, dstH, channelsPerPixel, format, PVA_INTERPOLATION_LINEAR);

        int nTileX = (dstW + dstTileWidth - 1) / dstTileWidth;
        int nTileY = (dstH + dstTileHeight - 1) / dstTileHeight;

        for (int32_t i = 0; i < nTileY; i++)
        {
            // use pixel center as coordinates: src_coordinate = (dst_idx + 0.5f) * scale - 0.5f;
            int64_t y_start_q_high = i * dstTileHeight * scaleY_q_high + scaleY_q_high / 2 - (1llu << QBIT_HIGH) / 2;
            int32_t y_start_idx    = static_cast<int32_t>(y_start_q_high >> QBIT_HIGH);
            int32_t y_start_ofst_i = static_cast<int32_t>(y_start_q_high >> (QBIT_HIGH - QBIT)) - (y_start_idx << QBIT);

            for (int32_t j = 0; j < nTileX; j++)
            {
                // use pixel center as coordinates: src_coordinate = (dst_idx + 0.5f) * scale - 0.5f;
                int64_t x_start_q_high = j * dstTileWidth * scaleX_q_high + scaleX_q_high / 2 - (1llu << QBIT_HIGH) / 2;
                int32_t x_start_idx    = static_cast<int32_t>(x_start_q_high >> QBIT_HIGH);
                int32_t x_start_ofst_i =
                    static_cast<int32_t>(x_start_q_high >> (QBIT_HIGH - QBIT)) - (x_start_idx << QBIT);

                // pixels in the tile
                for (int32_t k = 0; (k < dstTileHeight && i * dstTileHeight + k < dstH); k++)
                {
                    int32_t y_q  = k * scaleY_q + y_start_ofst_i;
                    int32_t iy   = y_q >> QBIT;
                    int32_t fy_q = y_q - (iy << QBIT);
                    iy += y_start_idx;

                    int y1 = iy;
                    int y2 = iy + 1;
                    // out-of-range: boundary-pixel-extention
                    y1 = std::min(std::max(y1, 0), srcH - 1);
                    y2 = std::min(std::max(y2, 0), srcH - 1);

                    for (int32_t l = 0; (l < dstTileWidth && j * dstTileWidth + l < dstW); l++)
                    {
                        int32_t x_q  = l * scaleX_q + x_start_ofst_i;
                        int32_t ix   = x_q >> QBIT;
                        int32_t fx_q = x_q - (ix << QBIT);
                        ix += x_start_idx;

                        int x1 = ix;
                        int x2 = ix + 1;
                        // out-of-range: boundary-pixel-extention
                        x1 = std::min(std::max(x1, 0), srcW - 1);
                        x2 = std::min(std::max(x2, 0), srcW - 1);

                        for (int32_t c = 0; c < channelsPerPixel; c++)
                        {
                            int i11 = y1 * srcLp + x1 * channelsPerPixel + c;
                            int i12 = y1 * srcLp + x2 * channelsPerPixel + c;
                            int i21 = y2 * srcLp + x1 * channelsPerPixel + c;
                            int i22 = y2 * srcLp + x2 * channelsPerPixel + c;

                            int64_t topL = inImagePtr_h[i11];
                            int64_t topR = inImagePtr_h[i12];
                            int64_t botL = inImagePtr_h[i21];
                            int64_t botR = inImagePtr_h[i22];

                            int64_t res;

                            if (transpose_flag)
                            {
                                int64_t H_left  = topL + roundDiff(topL, botL, fy_q);
                                int64_t H_right = topR + roundDiff(topR, botR, fy_q);
                                res             = H_left + roundDiff(H_left, H_right, fx_q);
                            }
                            else
                            {
                                int64_t H_top = topL + roundDiff(topL, topR, fx_q);
                                int64_t H_bot = botL + roundDiff(botL, botR, fx_q);
                                res           = H_top + roundDiff(H_top, H_bot, fy_q);
                            }

                            int dst_idx =
                                (i * dstTileHeight + k) * dstLp + (j * dstTileWidth + l) * channelsPerPixel + c;
                            outImagePtr_h[dst_idx] = static_cast<T>(res);
                        }
                    }
                }
            }
        }
        inImagePtr_h += srcH * srcLp;
        outImagePtr_h += dstH * dstLp;
    }
}

} // namespace

void generateRandomImageInput(NVCVImageHandle imageHandle)
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
    case NVCV_DATA_TYPE_U16:
        generateRandomInputImpl<uint16_t>(imageHandle);
        break;
    default:
        break;
    }
}

int32_t checkImageOutput(NVCVImageHandle refImageHandle, NVCVImageHandle pvaImageHandle, uint8_t threshold)
{
    NVCVImageFormat format;
    nvcvImageGetFormat(refImageHandle, &format);

    NVCVDataType dataType;
    nvcvImageFormatGetPlaneDataType(format, 0, &dataType);

    switch (dataType)
    {
    case NVCV_DATA_TYPE_U8:
    case NVCV_DATA_TYPE_2U8:
    case NVCV_DATA_TYPE_3U8:
    case NVCV_DATA_TYPE_4U8:
        return checkImageOutputImpl<uint8_t>(refImageHandle, pvaImageHandle, threshold);
    case NVCV_DATA_TYPE_U16:
        return checkImageOutputImpl<uint16_t>(refImageHandle, pvaImageHandle, threshold);
    default:
        return -1;
    }
}

void imageResizeFixedPointRef(NVCVImageHandle inImageHandle, NVCVImageHandle outImageHandle,
                              PVAInterpolationType resizeType)
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
        if (resizeType == PVA_INTERPOLATION_NN)
        {
            imageResizeNearestNeighborFixedPointDLUTRefImpl<uint8_t>(inImageHandle, outImageHandle);
        }
        else
        {
            imageResizeBilinearInterpFixedPointDLUTRefImpl<uint8_t>(inImageHandle, outImageHandle);
        }
        break;
    case NVCV_DATA_TYPE_U16:
        if (resizeType == PVA_INTERPOLATION_NN)
        {
            imageResizeNearestNeighborFixedPointDLUTRefImpl<uint16_t>(inImageHandle, outImageHandle);
        }
        else
        {
            imageResizeBilinearInterpFixedPointDLUTRefImpl<uint16_t>(inImageHandle, outImageHandle);
        }
        break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input data type!");
        break;
    }
}
