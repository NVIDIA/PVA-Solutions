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

#include "convert_image_format_ref.h"

#include "../../../../src/operator/image_processing/convert_image_format/device/convert_image_format_param.h"

#include <HostPointer.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

static ImageFormatConversionCode getImageFormatConversionCode(NVCVImageFormat inFormat, NVCVImageFormat outFormat)
{
    std::map<std::tuple<NVCVImageFormat, NVCVImageFormat>, ImageFormatConversionCode> codeMap = {
        {{NVCV_IMAGE_FORMAT_YUYV, NVCV_IMAGE_FORMAT_NV12}, CONVERT_IMAGE_FORMAT_YUYV2NV12},
        {{NVCV_IMAGE_FORMAT_UYVY, NVCV_IMAGE_FORMAT_NV12}, CONVERT_IMAGE_FORMAT_UYVY2NV12},
        {{NVCV_IMAGE_FORMAT_VYUY, NVCV_IMAGE_FORMAT_NV12}, CONVERT_IMAGE_FORMAT_VYUY2NV12},
        {{NVCV_IMAGE_FORMAT_YUV8p, NVCV_IMAGE_FORMAT_NV12}, CONVERT_IMAGE_FORMAT_YUV8p2NV12},
        {{NVCV_IMAGE_FORMAT_YUV8p, NVCV_IMAGE_FORMAT_YUYV}, CONVERT_IMAGE_FORMAT_YUV8p2YUYV},
        {{NVCV_IMAGE_FORMAT_YUV8p, NVCV_IMAGE_FORMAT_UYVY}, CONVERT_IMAGE_FORMAT_YUV8p2UYVY},
        {{NVCV_IMAGE_FORMAT_YUV8p, NVCV_IMAGE_FORMAT_VYUY}, CONVERT_IMAGE_FORMAT_YUV8p2VYUY},
        {{NVCV_IMAGE_FORMAT_NV12, NVCV_IMAGE_FORMAT_BGR8}, CONVERT_IMAGE_FORMAT_NV122BGR8},
        {{NVCV_IMAGE_FORMAT_NV12, NVCV_IMAGE_FORMAT_RGB8}, CONVERT_IMAGE_FORMAT_NV122RGB8},
        {{NVCV_IMAGE_FORMAT_NV12, NVCV_IMAGE_FORMAT_BGR8p}, CONVERT_IMAGE_FORMAT_NV122BGR8p},
        {{NVCV_IMAGE_FORMAT_NV12, NVCV_IMAGE_FORMAT_RGB8p}, CONVERT_IMAGE_FORMAT_NV122RGB8p},
        {{NVCV_IMAGE_FORMAT_YUYV, NVCV_IMAGE_FORMAT_BGR8}, CONVERT_IMAGE_FORMAT_YUYV2BGR8},
        {{NVCV_IMAGE_FORMAT_YUYV, NVCV_IMAGE_FORMAT_RGB8}, CONVERT_IMAGE_FORMAT_YUYV2RGB8},
        {{NVCV_IMAGE_FORMAT_UYVY, NVCV_IMAGE_FORMAT_BGR8}, CONVERT_IMAGE_FORMAT_UYVY2BGR8},
        {{NVCV_IMAGE_FORMAT_UYVY, NVCV_IMAGE_FORMAT_RGB8}, CONVERT_IMAGE_FORMAT_UYVY2RGB8},
        {{NVCV_IMAGE_FORMAT_VYUY, NVCV_IMAGE_FORMAT_BGR8}, CONVERT_IMAGE_FORMAT_VYUY2BGR8},
        {{NVCV_IMAGE_FORMAT_VYUY, NVCV_IMAGE_FORMAT_RGB8}, CONVERT_IMAGE_FORMAT_VYUY2RGB8},
        {{NVCV_IMAGE_FORMAT_YUYV, NVCV_IMAGE_FORMAT_BGR8p}, CONVERT_IMAGE_FORMAT_YUYV2BGR8p},
        {{NVCV_IMAGE_FORMAT_YUYV, NVCV_IMAGE_FORMAT_RGB8p}, CONVERT_IMAGE_FORMAT_YUYV2RGB8p},
        {{NVCV_IMAGE_FORMAT_UYVY, NVCV_IMAGE_FORMAT_BGR8p}, CONVERT_IMAGE_FORMAT_UYVY2BGR8p},
        {{NVCV_IMAGE_FORMAT_UYVY, NVCV_IMAGE_FORMAT_RGB8p}, CONVERT_IMAGE_FORMAT_UYVY2RGB8p},
        {{NVCV_IMAGE_FORMAT_VYUY, NVCV_IMAGE_FORMAT_BGR8p}, CONVERT_IMAGE_FORMAT_VYUY2BGR8p},
        {{NVCV_IMAGE_FORMAT_VYUY, NVCV_IMAGE_FORMAT_RGB8p}, CONVERT_IMAGE_FORMAT_VYUY2RGB8p},
        {{NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRA8}, CONVERT_IMAGE_FORMAT_RGB82BGRA8},
        {{NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_NV12}, CONVERT_IMAGE_FORMAT_BGR82NV12},
        {{NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_NV12}, CONVERT_IMAGE_FORMAT_RGB82NV12},
        {{NVCV_IMAGE_FORMAT_BGR8p, NVCV_IMAGE_FORMAT_NV12}, CONVERT_IMAGE_FORMAT_BGR8p2NV12},
        {{NVCV_IMAGE_FORMAT_RGB8p, NVCV_IMAGE_FORMAT_NV12}, CONVERT_IMAGE_FORMAT_RGB8p2NV12},
        {{NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_YUYV}, CONVERT_IMAGE_FORMAT_BGR82YUYV},
        {{NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_YUYV}, CONVERT_IMAGE_FORMAT_RGB82YUYV},
        {{NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_UYVY}, CONVERT_IMAGE_FORMAT_BGR82UYVY},
        {{NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_UYVY}, CONVERT_IMAGE_FORMAT_RGB82UYVY},
        {{NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_VYUY}, CONVERT_IMAGE_FORMAT_BGR82VYUY},
        {{NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_VYUY}, CONVERT_IMAGE_FORMAT_RGB82VYUY},
        {{NVCV_IMAGE_FORMAT_BGR8p, NVCV_IMAGE_FORMAT_YUYV}, CONVERT_IMAGE_FORMAT_BGR8p2YUYV},
        {{NVCV_IMAGE_FORMAT_RGB8p, NVCV_IMAGE_FORMAT_YUYV}, CONVERT_IMAGE_FORMAT_RGB8p2YUYV},
        {{NVCV_IMAGE_FORMAT_BGR8p, NVCV_IMAGE_FORMAT_UYVY}, CONVERT_IMAGE_FORMAT_BGR8p2UYVY},
        {{NVCV_IMAGE_FORMAT_RGB8p, NVCV_IMAGE_FORMAT_UYVY}, CONVERT_IMAGE_FORMAT_RGB8p2UYVY},
        {{NVCV_IMAGE_FORMAT_BGR8p, NVCV_IMAGE_FORMAT_VYUY}, CONVERT_IMAGE_FORMAT_BGR8p2VYUY},
        {{NVCV_IMAGE_FORMAT_RGB8p, NVCV_IMAGE_FORMAT_VYUY}, CONVERT_IMAGE_FORMAT_RGB8p2VYUY},
        /* NV12_ER inputs from packed YUV: chroma resampling only — no color
         * matrix involved, so the studio/ER distinction is irrelevant. */
        {{NVCV_IMAGE_FORMAT_YUYV, NVCV_IMAGE_FORMAT_NV12_ER}, CONVERT_IMAGE_FORMAT_YUYV2NV12_ER},
        {{NVCV_IMAGE_FORMAT_UYVY, NVCV_IMAGE_FORMAT_NV12_ER}, CONVERT_IMAGE_FORMAT_UYVY2NV12_ER},
        {{NVCV_IMAGE_FORMAT_VYUY, NVCV_IMAGE_FORMAT_NV12_ER}, CONVERT_IMAGE_FORMAT_VYUY2NV12_ER},
        {{NVCV_IMAGE_FORMAT_YUV8p, NVCV_IMAGE_FORMAT_NV12_ER}, CONVERT_IMAGE_FORMAT_YUV8p2NV12_ER},
        /* RGB8 / BGR8 / planar → NV12_ER (full-range RGB→YUV). */
        {{NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_NV12_ER}, CONVERT_IMAGE_FORMAT_BGR82NV12_ER},
        {{NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_NV12_ER}, CONVERT_IMAGE_FORMAT_RGB82NV12_ER},
        {{NVCV_IMAGE_FORMAT_BGR8p, NVCV_IMAGE_FORMAT_NV12_ER}, CONVERT_IMAGE_FORMAT_BGR8p2NV12_ER},
        {{NVCV_IMAGE_FORMAT_RGB8p, NVCV_IMAGE_FORMAT_NV12_ER}, CONVERT_IMAGE_FORMAT_RGB8p2NV12_ER},
        /* NV12_ER (full range) → packed/planar RGB/BGR/RGBA8. The kernel now skips the
         * Y-16 luma offset (cfg.lumaOffset = 0) and uses the ER color matrix;
         * the c-ref mirrors this via isErCode() → yuv2rgb(is_er = true). */
        {{NVCV_IMAGE_FORMAT_NV12_ER, NVCV_IMAGE_FORMAT_RGB8}, CONVERT_IMAGE_FORMAT_NV12_ER2RGB8},
        {{NVCV_IMAGE_FORMAT_NV12_ER, NVCV_IMAGE_FORMAT_BGR8}, CONVERT_IMAGE_FORMAT_NV12_ER2BGR8},
        {{NVCV_IMAGE_FORMAT_NV12_ER, NVCV_IMAGE_FORMAT_RGB8p}, CONVERT_IMAGE_FORMAT_NV12_ER2RGB8p},
        {{NVCV_IMAGE_FORMAT_NV12_ER, NVCV_IMAGE_FORMAT_BGR8p}, CONVERT_IMAGE_FORMAT_NV12_ER2BGR8p},
        {{NVCV_IMAGE_FORMAT_NV12_ER, NVCV_IMAGE_FORMAT_RGBA8}, CONVERT_IMAGE_FORMAT_NV12_ER2RGBA8},
        /* RGB → RGBA / BGR → BGRA (no R/B swap; appends alpha = 255). */
        {{NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGBA8}, CONVERT_IMAGE_FORMAT_RGB82RGBA8},
        {{NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_BGRA8}, CONVERT_IMAGE_FORMAT_BGR82BGRA8},
        /* RGBA → RGB / BGRA → BGR (4→3; alpha lane dropped). */
        {{NVCV_IMAGE_FORMAT_RGBA8, NVCV_IMAGE_FORMAT_RGB8}, CONVERT_IMAGE_FORMAT_RGBA82RGB8},
        {{NVCV_IMAGE_FORMAT_BGRA8, NVCV_IMAGE_FORMAT_BGR8}, CONVERT_IMAGE_FORMAT_BGRA82BGR8},
    };

    auto codeIt = codeMap.find(std::tuple<NVCVImageFormat, NVCVImageFormat>(inFormat, outFormat));
    if (codeIt != codeMap.end())
    {
        return codeIt->second;
    }
    else
    {
        throw std::invalid_argument("Unsupported input/output image format conversion");
    }
}

uint8_t coeff_r_fixed[] = {COEFF_R_FIXED}; // 74, 0, 102
int8_t coeff_g_fixed[]  = {COEFF_G_FIXED}; // 74, -25, -52
uint8_t coeff_b_fixed[] = {COEFF_B_FIXED}; // 74, 129, 0

uint8_t coeff_y_fixed[] = {COEFF_Y_FIXED}; // 16, 32, 6
int8_t coeff_u_fixed[]  = {COEFF_U_FIXED}; // -9, -18, 28
int8_t coeff_v_fixed[]  = {COEFF_V_FIXED}; // 28, -23, -4

/* ITU-R BT.601 Extended-Range (NV12_ER) coefficients — mirrors the kernel-side
 * COEFF_*_ER_FIXED macros in device/convert_image_format_param.h. The c-ref
 * selects between studio and ER coefficient tables at runtime based on whether
 * the conversion code is an _ER variant; this keeps the existing
 * yuv2rgb/rgb2y/rgb2uv helpers unchanged in shape. */
uint8_t coeff_r_er_fixed[] = {COEFF_R_ER_FIXED}; // 64, 0, 90
int8_t coeff_g_er_fixed[]  = {COEFF_G_ER_FIXED}; // 64, -22, -46
uint8_t coeff_b_er_fixed[] = {COEFF_B_ER_FIXED}; // 64, 113, 0

uint8_t coeff_y_er_fixed[] = {COEFF_Y_ER_FIXED}; // 19, 38, 7
int8_t coeff_u_er_fixed[]  = {COEFF_U_ER_FIXED}; // -11, -21, 32
int8_t coeff_v_er_fixed[]  = {COEFF_V_ER_FIXED}; // 32, -27, -5

/* Returns true if the conversion code is an NV12_ER variant (full-range YUV).
 * For ER codes the c-ref uses the *_er coefficient tables AND skips the
 * (Y-16) luma offset (and the +16 on the reverse path). */
static bool isErCode(ImageFormatConversionCode code)
{
    switch (code)
    {
    case CONVERT_IMAGE_FORMAT_NV12_ER2BGR8:
    case CONVERT_IMAGE_FORMAT_NV12_ER2RGB8:
    case CONVERT_IMAGE_FORMAT_NV12_ER2BGR8p:
    case CONVERT_IMAGE_FORMAT_NV12_ER2RGB8p:
    case CONVERT_IMAGE_FORMAT_NV12_ER2RGBA8:
    case CONVERT_IMAGE_FORMAT_BGR82NV12_ER:
    case CONVERT_IMAGE_FORMAT_RGB82NV12_ER:
    case CONVERT_IMAGE_FORMAT_BGR8p2NV12_ER:
    case CONVERT_IMAGE_FORMAT_RGB8p2NV12_ER:
    case CONVERT_IMAGE_FORMAT_YUYV2NV12_ER:
    case CONVERT_IMAGE_FORMAT_UYVY2NV12_ER:
    case CONVERT_IMAGE_FORMAT_VYUY2NV12_ER:
    case CONVERT_IMAGE_FORMAT_YUV8p2NV12_ER:
        return true;
    default:
        return false;
    }
}

template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value>>
static void yuv2rgb(T y, T u, T v, T &r, T &g, T &b, bool is_er = false)
{
    int32_t dataY = y;
    int32_t dataU = u;
    int32_t dataV = v;

    if (!is_er)
    {
        dataY -= COLOR_OFFSET_L;
    }
    dataU -= COLOR_OFFSET_C;
    dataV -= COLOR_OFFSET_C;

    const uint8_t *cr = is_er ? coeff_r_er_fixed : coeff_r_fixed;
    const int8_t *cg  = is_er ? coeff_g_er_fixed : coeff_g_fixed;
    const uint8_t *cb = is_er ? coeff_b_er_fixed : coeff_b_fixed;

    int32_t dataR = (dataY * cr[0] + dataU * cr[1] + dataV * cr[2] + QBITS_SCALE_HALF) >> QBITS;
    int32_t dataG = (dataY * cg[0] + dataU * cg[1] + dataV * cg[2] + QBITS_SCALE_HALF) >> QBITS;
    int32_t dataB = (dataY * cb[0] + dataU * cb[1] + dataV * cb[2] + QBITS_SCALE_HALF) >> QBITS;

    // clamp to [0, 255]
    dataR = std::clamp(dataR, 0, static_cast<int32_t>(std::numeric_limits<T>::max()));
    dataG = std::clamp(dataG, 0, static_cast<int32_t>(std::numeric_limits<T>::max()));
    dataB = std::clamp(dataB, 0, static_cast<int32_t>(std::numeric_limits<T>::max()));

    r = static_cast<T>(dataR);
    g = static_cast<T>(dataG);
    b = static_cast<T>(dataB);
}

template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value>>
static void rgb2y(T r, T g, T b, T &y, bool is_er = false)
{
    int32_t dataR = r;
    int32_t dataG = g;
    int32_t dataB = b;

    const uint8_t *cy = is_er ? coeff_y_er_fixed : coeff_y_fixed;

    int32_t dataY = (dataR * cy[0] + dataG * cy[1] + dataB * cy[2] + QBITS_SCALE_HALF) >> QBITS;

    if (!is_er)
    {
        dataY += COLOR_OFFSET_L;
    }

    // clamp to [0, 255]
    dataY = std::clamp(dataY, 0, static_cast<int32_t>(std::numeric_limits<T>::max()));

    y = static_cast<T>(dataY);
}

template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value>>
static void rgb2uv(T r, T g, T b, T &u, T &v, bool is_er = false)
{
    int32_t dataR = r;
    int32_t dataG = g;
    int32_t dataB = b;

    const int8_t *cu = is_er ? coeff_u_er_fixed : coeff_u_fixed;
    const int8_t *cv = is_er ? coeff_v_er_fixed : coeff_v_fixed;

    int32_t dataU = (dataR * cu[0] + dataG * cu[1] + dataB * cu[2] + QBITS_SCALE_HALF) >> QBITS;
    int32_t dataV = (dataR * cv[0] + dataG * cv[1] + dataB * cv[2] + QBITS_SCALE_HALF) >> QBITS;

    dataU += COLOR_OFFSET_C;
    dataV += COLOR_OFFSET_C;

    // clamp to [0, 255]
    dataU = std::clamp(dataU, 0, static_cast<int32_t>(std::numeric_limits<T>::max()));
    dataV = std::clamp(dataV, 0, static_cast<int32_t>(std::numeric_limits<T>::max()));

    u = static_cast<T>(dataU);
    v = static_cast<T>(dataV);
}

template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value>>
static void nv122rgbImage(T *srcY, T *srcU, T *srcV, T *dstR, T *dstG, T *dstB, int32_t width, int32_t height,
                          int32_t srcLinePitch, int32_t dstLinePitch, int32_t srcChannel, int32_t dstChannel,
                          bool is_er = false)
{
    for (int32_t y = 0; y < height; y++)
    {
        for (int32_t x = 0; x < width; x++)
        {
            int32_t srcBaseOffsetY  = y * srcLinePitch + x * srcChannel;
            int32_t srcBaseOffsetUV = (y / 2) * srcLinePitch + (x / 2) * (srcChannel * 2);
            int32_t dstBaseOffset   = y * dstLinePitch + x * dstChannel;

            yuv2rgb(srcY[srcBaseOffsetY], srcU[srcBaseOffsetUV], srcV[srcBaseOffsetUV], dstR[dstBaseOffset],
                    dstG[dstBaseOffset], dstB[dstBaseOffset], is_er);
        }
    }
}

template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value>>
static void yuv2rgbImage(T *srcY, T *srcU, T *srcV, T *dstR, T *dstG, T *dstB, int32_t width, int32_t height,
                         int32_t srcLinePitch, int32_t dstLinePitch, int32_t srcChannel, int32_t dstChannel,
                         bool is_er = false)
{
    for (int32_t y = 0; y < height; y++)
    {
        for (int32_t x = 0; x < width; x++)
        {
            int32_t srcBaseOffsetY  = y * srcLinePitch + x * srcChannel;
            int32_t srcBaseOffsetUV = y * srcLinePitch + (x / 2) * (srcChannel * 2);
            int32_t dstBaseOffset   = y * dstLinePitch + x * dstChannel;

            yuv2rgb(srcY[srcBaseOffsetY], srcU[srcBaseOffsetUV], srcV[srcBaseOffsetUV], dstR[dstBaseOffset],
                    dstG[dstBaseOffset], dstB[dstBaseOffset], is_er);
        }
    }
}

template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value>>
static void rgb2nv12Image(T *srcR, T *srcG, T *srcB, T *dstY, T *dstU, T *dstV, int32_t width, int32_t height,
                          int32_t srcLinePitch, int32_t dstLinePitch, int32_t srcChannel, int32_t dstChannel,
                          bool is_er = false)
{
    for (int32_t y = 0; y < height; y++)
    {
        for (int32_t x = 0; x < width; x++)
        {
            int32_t srcBaseOffset   = y * srcLinePitch + x * srcChannel;
            int32_t dstBaseOffsetY  = y * dstLinePitch + x * dstChannel;
            int32_t dstBaseOffsetUV = (y / 2) * dstLinePitch + (x / 2) * (dstChannel * 2);

            rgb2y(srcR[srcBaseOffset], srcG[srcBaseOffset], srcB[srcBaseOffset], dstY[dstBaseOffsetY], is_er);
            if (((y & 1) == 0) && ((x & 1) == 0))
            {
                rgb2uv(srcR[srcBaseOffset], srcG[srcBaseOffset], srcB[srcBaseOffset], dstU[dstBaseOffsetUV],
                       dstV[dstBaseOffsetUV], is_er);
            }
        }
    }
}

template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value>>
static void rgb2yuvImage(T *srcR, T *srcG, T *srcB, T *dstY, T *dstU, T *dstV, int32_t width, int32_t height,
                         int32_t srcLinePitch, int32_t dstLinePitch, int32_t srcChannel, int32_t dstChannel,
                         bool is_er = false)
{
    for (int32_t y = 0; y < height; y++)
    {
        for (int32_t x = 0; x < width; x++)
        {
            int32_t srcBaseOffset = y * srcLinePitch + x * srcChannel;
            int32_t dstBaseOffset = y * dstLinePitch + x * dstChannel;

            rgb2y(srcR[srcBaseOffset], srcG[srcBaseOffset], srcB[srcBaseOffset], dstY[dstBaseOffset], is_er);
            if ((x & 1) == 0)
            {
                rgb2uv(srcR[srcBaseOffset], srcG[srcBaseOffset], srcB[srcBaseOffset], dstU[dstBaseOffset],
                       dstV[dstBaseOffset], is_er);
            }
        }
    }
}

template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value>>
static void yuv2nv12Image(T *srcY, T *srcU, T *srcV, T *dstY, T *dstU, T *dstV, int32_t width, int32_t height,
                          int32_t srcLinePitch, int32_t dstLinePitch, int32_t srcChannel, int32_t dstChannel)
{
    for (int32_t y = 0; y < height; y++)
    {
        for (int32_t x = 0; x < width; x++)
        {
            int32_t srcBaseOffset   = y * srcLinePitch + x * srcChannel;
            int32_t dstBaseOffsetY  = y * dstLinePitch + x * dstChannel;
            int32_t dstBaseOffsetUV = (y / 2) * dstLinePitch + (x / 2) * (dstChannel * 2);

            dstY[dstBaseOffsetY] = srcY[srcBaseOffset];
            if (((y & 1) == 0) && ((x & 1) == 0))
            {
                dstU[dstBaseOffsetUV] = srcU[srcBaseOffset];
                dstV[dstBaseOffsetUV] = srcV[srcBaseOffset];
            }
        }
    }
}

template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value>>
static void yuv2yuvImage(T *srcY, T *srcU, T *srcV, T *dstY, T *dstU, T *dstV, int32_t width, int32_t height,
                         int32_t srcLinePitch, int32_t dstLinePitch, int32_t srcChannel, int32_t dstChannel)
{
    for (int32_t y = 0; y < height; y++)
    {
        for (int32_t x = 0; x < width; x++)
        {
            int32_t srcBaseOffset = y * srcLinePitch + x * srcChannel;
            int32_t dstBaseOffset = y * dstLinePitch + x * dstChannel;

            dstY[dstBaseOffset] = srcY[srcBaseOffset];
            if ((x & 1) == 0)
            {
                dstU[dstBaseOffset] = srcU[srcBaseOffset];
                dstV[dstBaseOffset] = srcV[srcBaseOffset];
            }
        }
    }
}

template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value>>
static void rgba2rgbaImage(T *srcR, T *srcG, T *srcB, T *srcA, T *dstR, T *dstG, T *dstB, T *dstA, int32_t width,
                           int32_t height, int32_t srcLinePitch, int32_t dstLinePitch, int32_t srcChannel,
                           int32_t dstChannel)
{
    for (int32_t y = 0; y < height; y++)
    {
        for (int32_t x = 0; x < width; x++)
        {
            int32_t srcBaseOffset = y * srcLinePitch + x * srcChannel;
            int32_t dstBaseOffset = y * dstLinePitch + x * dstChannel;

            dstR[dstBaseOffset] = srcR[srcBaseOffset];
            dstG[dstBaseOffset] = srcG[srcBaseOffset];
            dstB[dstBaseOffset] = srcB[srcBaseOffset];
            if (dstA)
            {
                if (srcA)
                {
                    dstA[dstBaseOffset] = srcA[srcBaseOffset];
                }
                else
                {
                    dstA[dstBaseOffset] = ALPHA_VALUE;
                }
            }
        }
    }
}

template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value>>
void generateInputImpl(NVCVImageHandle imageHandle)
{
    NVCVImageData imageData;
    nvcvImageExportData(imageHandle, &imageData);

    T *imagePtr_h = static_cast<T *>(PvaGetImageHostPointer(imageHandle, 0));

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

        // generate all kinds of pixel value in input data
        for (int32_t j = 0; j < height; j++)
        {
            for (int32_t i = 0; i < width * channelsPerPixel; i++)
            {
                imagePtr_h[j * linePitch + i] = static_cast<T>(j * width * channelsPerPixel + i);
            }
        }

        imagePtr_h += height * linePitch;
    }
}

template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value>>
int32_t checkOutputImpl(NVCVImageHandle refImageHandle, NVCVImageHandle pvaImageHandle)
{
    NVCVImageData refImageData;
    nvcvImageExportData(refImageHandle, &refImageData);

    T *ref = static_cast<T *>(PvaGetImageHostPointer(refImageHandle, 0));

    NVCVImageData pvaImageData;
    nvcvImageExportData(pvaImageHandle, &pvaImageData);

    T *pva = static_cast<T *>(PvaGetImageHostPointer(pvaImageHandle, 0));

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
void convertImageFormatRefImpl(NVCVImageHandle inImageHandle, NVCVImageHandle outImageHandle)
{
    NVCVImageData inImageData;
    nvcvImageExportData(inImageHandle, &inImageData);
    T *inImagePtr = static_cast<T *>(PvaGetImageHostPointer(inImageHandle, 0));

    NVCVImageData outImageData;
    nvcvImageExportData(outImageHandle, &outImageData);
    T *outImagePtr = static_cast<T *>(PvaGetImageHostPointer(outImageHandle, 0));

    NVCVImageFormat inFormat       = inImageData.format;
    NVCVImageFormat outFormat      = outImageData.format;
    ImageFormatConversionCode code = getImageFormatConversionCode(inFormat, outFormat);

    int32_t height       = inImageData.buffer.strided.planes[0].height;
    int32_t width        = inImageData.buffer.strided.planes[0].width;
    int32_t srcLinePitch = inImageData.buffer.strided.planes[0].rowStride / sizeof(T);
    int32_t dstLinePitch = outImageData.buffer.strided.planes[0].rowStride / sizeof(T);

    int32_t srcBitsPerPixel, dstBitsPerPixel;
    nvcvImageFormatGetPlaneBitsPerPixel(inFormat, 0, &srcBitsPerPixel);
    nvcvImageFormatGetPlaneBitsPerPixel(outFormat, 0, &dstBitsPerPixel);

    int32_t srcChannel = (srcBitsPerPixel / CHAR_BIT) / sizeof(T);
    int32_t dstChannel = (dstBitsPerPixel / CHAR_BIT) / sizeof(T);

    switch (code)
    {
    case CONVERT_IMAGE_FORMAT_YUYV2NV12:
    case CONVERT_IMAGE_FORMAT_UYVY2NV12:
    case CONVERT_IMAGE_FORMAT_VYUY2NV12:
    case CONVERT_IMAGE_FORMAT_YUV8p2NV12:
    case CONVERT_IMAGE_FORMAT_YUYV2NV12_ER:
    case CONVERT_IMAGE_FORMAT_UYVY2NV12_ER:
    case CONVERT_IMAGE_FORMAT_VYUY2NV12_ER:
    case CONVERT_IMAGE_FORMAT_YUV8p2NV12_ER:
    {
        T *srcY = inImagePtr;
        T *srcU = srcY + 1;
        T *srcV = srcU + 2;
        if (code == CONVERT_IMAGE_FORMAT_UYVY2NV12 || code == CONVERT_IMAGE_FORMAT_UYVY2NV12_ER)
        {
            srcU = inImagePtr;
            srcY = srcU + 1;
            srcV = srcU + 2;
        }
        else if (code == CONVERT_IMAGE_FORMAT_VYUY2NV12 || code == CONVERT_IMAGE_FORMAT_VYUY2NV12_ER)
        {
            srcV = inImagePtr;
            srcY = srcV + 1;
            srcU = srcV + 2;
        }
        else if (code == CONVERT_IMAGE_FORMAT_YUV8p2NV12 || code == CONVERT_IMAGE_FORMAT_YUV8p2NV12_ER)
        {
            srcU = srcY + height * srcLinePitch;
            srcV = srcU + height * srcLinePitch;
        }

        T *dstY = outImagePtr;
        T *dstU = dstY + height * dstLinePitch;
        T *dstV = dstU + 1;

        /* YUV → NV12 chroma resampling only — no color-matrix conversion
         * happens, so the studio/ER distinction is irrelevant at the byte
         * level. The output is range-tagged in the image-format descriptor. */
        yuv2nv12Image(srcY, srcU, srcV, dstY, dstU, dstV, width, height, srcLinePitch, dstLinePitch, srcChannel,
                      dstChannel);
    }
    break;
    case CONVERT_IMAGE_FORMAT_YUV8p2YUYV:
    case CONVERT_IMAGE_FORMAT_YUV8p2UYVY:
    case CONVERT_IMAGE_FORMAT_YUV8p2VYUY:
    {
        T *srcY = inImagePtr;
        T *srcU = srcY + height * srcLinePitch;
        T *srcV = srcU + height * srcLinePitch;

        T *dstY = outImagePtr;
        T *dstU = dstY + 1;
        T *dstV = dstU + 2;
        if (code == CONVERT_IMAGE_FORMAT_YUV8p2UYVY)
        {
            dstU = outImagePtr;
            dstY = dstU + 1;
            dstV = dstU + 2;
        }
        else if (code == CONVERT_IMAGE_FORMAT_YUV8p2VYUY)
        {
            dstV = outImagePtr;
            dstY = dstV + 1;
            dstU = dstV + 2;
        }

        yuv2yuvImage(srcY, srcU, srcV, dstY, dstU, dstV, width, height, srcLinePitch, dstLinePitch, srcChannel,
                     dstChannel);
    }
    break;
    case CONVERT_IMAGE_FORMAT_NV122BGR8:
    case CONVERT_IMAGE_FORMAT_NV122RGB8:
    case CONVERT_IMAGE_FORMAT_NV122BGR8p:
    case CONVERT_IMAGE_FORMAT_NV122RGB8p:
    case CONVERT_IMAGE_FORMAT_NV12_ER2BGR8:
    case CONVERT_IMAGE_FORMAT_NV12_ER2RGB8:
    case CONVERT_IMAGE_FORMAT_NV12_ER2BGR8p:
    case CONVERT_IMAGE_FORMAT_NV12_ER2RGB8p:
    {
        T *srcY = inImagePtr;
        T *srcU = srcY + height * srcLinePitch;
        T *srcV = srcU + 1;

        T *dstB = outImagePtr;
        T *dstG = dstB + 1;
        T *dstR = dstG + 1;
        if (code == CONVERT_IMAGE_FORMAT_NV122RGB8 || code == CONVERT_IMAGE_FORMAT_NV12_ER2RGB8)
        {
            dstR = outImagePtr;
            dstG = dstR + 1;
            dstB = dstG + 1;
        }
        else if (code == CONVERT_IMAGE_FORMAT_NV122BGR8p || code == CONVERT_IMAGE_FORMAT_NV12_ER2BGR8p)
        {
            dstG = dstB + height * dstLinePitch;
            dstR = dstG + height * dstLinePitch;
        }
        else if (code == CONVERT_IMAGE_FORMAT_NV122RGB8p || code == CONVERT_IMAGE_FORMAT_NV12_ER2RGB8p)
        {
            dstR = outImagePtr;
            dstG = dstR + height * dstLinePitch;
            dstB = dstG + height * dstLinePitch;
        }

        nv122rgbImage(srcY, srcU, srcV, dstR, dstG, dstB, width, height, srcLinePitch, dstLinePitch, srcChannel,
                      dstChannel, isErCode(code));
    }
    break;
    case CONVERT_IMAGE_FORMAT_NV12_ER2RGBA8:
    {
        /* NV12_ER → packed RGBA8: full-range YUV→RGB, alpha (byte 3) = 255. */
        T *srcY = inImagePtr;
        T *srcU = srcY + height * srcLinePitch;
        T *srcV = srcU + 1;

        T *dstR = outImagePtr;
        T *dstG = dstR + 1;
        T *dstB = dstG + 1;
        T *dstA = dstB + 1;

        nv122rgbImage(srcY, srcU, srcV, dstR, dstG, dstB, width, height, srcLinePitch, dstLinePitch, srcChannel,
                      dstChannel, isErCode(code));

        for (int32_t y = 0; y < height; y++)
        {
            for (int32_t x = 0; x < width; x++)
            {
                dstA[y * dstLinePitch + x * dstChannel] = ALPHA_VALUE;
            }
        }
    }
    break;
    case CONVERT_IMAGE_FORMAT_RGBA82RGB8:
    case CONVERT_IMAGE_FORMAT_BGRA82BGR8:
    {
        /* Packed 4→3: copy the first 3 channels of every pixel, drop alpha.
         * Byte handling is identical for RGBA→RGB and BGRA→BGR. */
        T *src0 = inImagePtr;
        T *src1 = src0 + 1;
        T *src2 = src1 + 1;
        T *srcA = nullptr;

        T *dst0 = outImagePtr;
        T *dst1 = dst0 + 1;
        T *dst2 = dst1 + 1;
        T *dstA = nullptr; // RGB output has no alpha lane

        rgba2rgbaImage(src0, src1, src2, srcA, dst0, dst1, dst2, dstA, width, height, srcLinePitch, dstLinePitch,
                       srcChannel, dstChannel);
    }
    break;
    case CONVERT_IMAGE_FORMAT_YUYV2BGR8:
    case CONVERT_IMAGE_FORMAT_YUYV2RGB8:
    case CONVERT_IMAGE_FORMAT_UYVY2BGR8:
    case CONVERT_IMAGE_FORMAT_UYVY2RGB8:
    case CONVERT_IMAGE_FORMAT_VYUY2BGR8:
    case CONVERT_IMAGE_FORMAT_VYUY2RGB8:
    case CONVERT_IMAGE_FORMAT_YUYV2BGR8p:
    case CONVERT_IMAGE_FORMAT_YUYV2RGB8p:
    case CONVERT_IMAGE_FORMAT_UYVY2BGR8p:
    case CONVERT_IMAGE_FORMAT_UYVY2RGB8p:
    case CONVERT_IMAGE_FORMAT_VYUY2BGR8p:
    case CONVERT_IMAGE_FORMAT_VYUY2RGB8p:
    {
        T *srcY = inImagePtr;
        T *srcU = srcY + 1;
        T *srcV = srcU + 2;
        if ((code == CONVERT_IMAGE_FORMAT_UYVY2BGR8) || (code == CONVERT_IMAGE_FORMAT_UYVY2RGB8) ||
            (code == CONVERT_IMAGE_FORMAT_UYVY2BGR8p) || (code == CONVERT_IMAGE_FORMAT_UYVY2RGB8p))
        {
            srcU = inImagePtr;
            srcY = srcU + 1;
            srcV = srcU + 2;
        }
        else if ((code == CONVERT_IMAGE_FORMAT_VYUY2BGR8) || (code == CONVERT_IMAGE_FORMAT_VYUY2RGB8) ||
                 (code == CONVERT_IMAGE_FORMAT_VYUY2BGR8p) || (code == CONVERT_IMAGE_FORMAT_VYUY2RGB8p))
        {
            srcV = inImagePtr;
            srcY = srcV + 1;
            srcU = srcV + 2;
        }

        T *dstB = outImagePtr;
        T *dstG = dstB + 1;
        T *dstR = dstG + 1;
        if ((code == CONVERT_IMAGE_FORMAT_YUYV2RGB8) || (code == CONVERT_IMAGE_FORMAT_UYVY2RGB8) ||
            (code == CONVERT_IMAGE_FORMAT_VYUY2RGB8))
        {
            dstR = outImagePtr;
            dstG = dstR + 1;
            dstB = dstG + 1;
        }
        else if ((code == CONVERT_IMAGE_FORMAT_YUYV2BGR8p) || (code == CONVERT_IMAGE_FORMAT_UYVY2BGR8p) ||
                 (code == CONVERT_IMAGE_FORMAT_VYUY2BGR8p))
        {
            dstG = dstB + height * dstLinePitch;
            dstR = dstG + height * dstLinePitch;
        }
        else if ((code == CONVERT_IMAGE_FORMAT_YUYV2RGB8p) || (code == CONVERT_IMAGE_FORMAT_UYVY2RGB8p) ||
                 (code == CONVERT_IMAGE_FORMAT_VYUY2RGB8p))
        {
            dstR = outImagePtr;
            dstG = dstR + height * dstLinePitch;
            dstB = dstG + height * dstLinePitch;
        }

        yuv2rgbImage(srcY, srcU, srcV, dstR, dstG, dstB, width, height, srcLinePitch, dstLinePitch, srcChannel,
                     dstChannel);
    }
    break;
    case CONVERT_IMAGE_FORMAT_RGB82BGRA8:
    {
        T *srcR = inImagePtr;
        T *srcG = srcR + 1;
        T *srcB = srcG + 1;
        T *srcA = nullptr;

        T *dstB = outImagePtr;
        T *dstG = dstB + 1;
        T *dstR = dstG + 1;
        T *dstA = dstR + 1;

        rgba2rgbaImage(srcR, srcG, srcB, srcA, dstR, dstG, dstB, dstA, width, height, srcLinePitch, dstLinePitch,
                       srcChannel, dstChannel);
    }
    break;
    case CONVERT_IMAGE_FORMAT_RGB82RGBA8:
    {
        /* RGB → RGBA: keep R/G/B order; alpha at byte 3 = 255 (srcA = nullptr). */
        T *srcR = inImagePtr;
        T *srcG = srcR + 1;
        T *srcB = srcG + 1;
        T *srcA = nullptr;

        T *dstR = outImagePtr;
        T *dstG = dstR + 1;
        T *dstB = dstG + 1;
        T *dstA = dstB + 1;

        rgba2rgbaImage(srcR, srcG, srcB, srcA, dstR, dstG, dstB, dstA, width, height, srcLinePitch, dstLinePitch,
                       srcChannel, dstChannel);
    }
    break;
    case CONVERT_IMAGE_FORMAT_BGR82BGRA8:
    {
        /* BGR → BGRA: keep B/G/R order; alpha at byte 3 = 255. */
        T *srcB = inImagePtr;
        T *srcG = srcB + 1;
        T *srcR = srcG + 1;
        T *srcA = nullptr;

        T *dstB = outImagePtr;
        T *dstG = dstB + 1;
        T *dstR = dstG + 1;
        T *dstA = dstR + 1;

        rgba2rgbaImage(srcR, srcG, srcB, srcA, dstR, dstG, dstB, dstA, width, height, srcLinePitch, dstLinePitch,
                       srcChannel, dstChannel);
    }
    break;
    case CONVERT_IMAGE_FORMAT_BGR82NV12:
    case CONVERT_IMAGE_FORMAT_RGB82NV12:
    case CONVERT_IMAGE_FORMAT_BGR8p2NV12:
    case CONVERT_IMAGE_FORMAT_RGB8p2NV12:
    case CONVERT_IMAGE_FORMAT_BGR82NV12_ER:
    case CONVERT_IMAGE_FORMAT_RGB82NV12_ER:
    case CONVERT_IMAGE_FORMAT_BGR8p2NV12_ER:
    case CONVERT_IMAGE_FORMAT_RGB8p2NV12_ER:
    {
        T *srcB = inImagePtr;
        T *srcG = srcB + 1;
        T *srcR = srcG + 1;
        if (code == CONVERT_IMAGE_FORMAT_RGB82NV12 || code == CONVERT_IMAGE_FORMAT_RGB82NV12_ER)
        {
            srcR = inImagePtr;
            srcG = srcR + 1;
            srcB = srcG + 1;
        }
        else if (code == CONVERT_IMAGE_FORMAT_BGR8p2NV12 || code == CONVERT_IMAGE_FORMAT_BGR8p2NV12_ER)
        {
            srcG = srcB + height * srcLinePitch;
            srcR = srcG + height * srcLinePitch;
        }
        else if (code == CONVERT_IMAGE_FORMAT_RGB8p2NV12 || code == CONVERT_IMAGE_FORMAT_RGB8p2NV12_ER)
        {
            srcR = inImagePtr;
            srcG = srcR + height * srcLinePitch;
            srcB = srcG + height * srcLinePitch;
        }

        T *dstY = outImagePtr;
        T *dstU = dstY + height * dstLinePitch;
        T *dstV = dstU + 1;

        rgb2nv12Image(srcR, srcG, srcB, dstY, dstU, dstV, width, height, srcLinePitch, dstLinePitch, srcChannel,
                      dstChannel, isErCode(code));
    }
    break;
    case CONVERT_IMAGE_FORMAT_BGR82YUYV:
    case CONVERT_IMAGE_FORMAT_RGB82YUYV:
    case CONVERT_IMAGE_FORMAT_BGR82UYVY:
    case CONVERT_IMAGE_FORMAT_RGB82UYVY:
    case CONVERT_IMAGE_FORMAT_BGR82VYUY:
    case CONVERT_IMAGE_FORMAT_RGB82VYUY:
    case CONVERT_IMAGE_FORMAT_BGR8p2YUYV:
    case CONVERT_IMAGE_FORMAT_RGB8p2YUYV:
    case CONVERT_IMAGE_FORMAT_BGR8p2UYVY:
    case CONVERT_IMAGE_FORMAT_RGB8p2UYVY:
    case CONVERT_IMAGE_FORMAT_BGR8p2VYUY:
    case CONVERT_IMAGE_FORMAT_RGB8p2VYUY:
    {
        T *srcB = inImagePtr;
        T *srcG = srcB + 1;
        T *srcR = srcG + 1;
        if ((code == CONVERT_IMAGE_FORMAT_RGB82YUYV) || (code == CONVERT_IMAGE_FORMAT_RGB82UYVY) ||
            (code == CONVERT_IMAGE_FORMAT_RGB82VYUY))
        {
            srcR = inImagePtr;
            srcG = srcR + 1;
            srcB = srcG + 1;
        }
        else if ((code == CONVERT_IMAGE_FORMAT_BGR8p2YUYV) || (code == CONVERT_IMAGE_FORMAT_BGR8p2UYVY) ||
                 (code == CONVERT_IMAGE_FORMAT_BGR8p2VYUY))
        {
            srcG = srcB + height * srcLinePitch;
            srcR = srcG + height * srcLinePitch;
        }
        else if ((code == CONVERT_IMAGE_FORMAT_RGB8p2YUYV) || (code == CONVERT_IMAGE_FORMAT_RGB8p2UYVY) ||
                 (code == CONVERT_IMAGE_FORMAT_RGB8p2VYUY))
        {
            srcR = inImagePtr;
            srcG = srcR + height * srcLinePitch;
            srcB = srcG + height * srcLinePitch;
        }

        T *dstY = outImagePtr;
        T *dstU = dstY + 1;
        T *dstV = dstU + 2;
        if ((code == CONVERT_IMAGE_FORMAT_BGR82UYVY) || (code == CONVERT_IMAGE_FORMAT_RGB82UYVY) ||
            (code == CONVERT_IMAGE_FORMAT_BGR8p2UYVY) || (code == CONVERT_IMAGE_FORMAT_RGB8p2UYVY))
        {
            dstU = outImagePtr;
            dstY = dstU + 1;
            dstV = dstU + 2;
        }
        else if ((code == CONVERT_IMAGE_FORMAT_BGR82VYUY) || (code == CONVERT_IMAGE_FORMAT_RGB82VYUY) ||
                 (code == CONVERT_IMAGE_FORMAT_BGR8p2VYUY) || (code == CONVERT_IMAGE_FORMAT_RGB8p2VYUY))
        {
            dstV = outImagePtr;
            dstY = dstV + 1;
            dstU = dstV + 2;
        }

        rgb2yuvImage(srcR, srcG, srcB, dstY, dstU, dstV, width, height, srcLinePitch, dstLinePitch, srcChannel,
                     dstChannel);
    }
    break;
    default:
        break;
    }
}

void generateConvertImageFormatInput(NVCVImageHandle imageHandle)
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

int32_t checkConvertImageFormatOutput(NVCVImageHandle refImageHandle, NVCVImageHandle pvaImageHandle)
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

void convertImageFormatRef(NVCVImageHandle inImageHandle, NVCVImageHandle outImageHandle)
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
        convertImageFormatRefImpl<uint8_t>(inImageHandle, outImageHandle);
        break;
    default:
        break;
    }
}