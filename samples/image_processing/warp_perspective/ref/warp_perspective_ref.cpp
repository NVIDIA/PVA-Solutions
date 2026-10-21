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

#include "warp_perspective_ref.h"

#include <nvcv/Tensor.hpp>

#include <OpWarpPerspective.h>
#include <nvcv/Status.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <type_traits>

#define TILE_W 64
#define TILE_H 54
#define QBITS 16
// input is variable, we assign the half superbank(64KB) for double buffer
#define HALF_SUPERBANK_BYTE_SIZE (64 * 1024)
#define MAX_TABLE_SIZE (HALF_SUPERBANK_BYTE_SIZE / 2)

namespace {
void generateRandomInputImpl(NVCVImageHandle imageHandle, bool isRandom = true)
{
    NVCVImageData imageData;
    nvcvImageExportData(imageHandle, &imageData);

    std::mt19937 gen;
    gen.seed(1);
    std::uniform_int_distribution<> distrib(0, std::numeric_limits<uint8_t>::max());
    int cnt{0};
    auto generatorU8 = [&]() {
        return static_cast<uint8_t>(isRandom ? (distrib(gen)) : cnt++);
    };

    int32_t numPlanes;
    nvcvImageFormatGetNumPlanes(imageData.format, &numPlanes);

    uint8_t *imageBasePtr   = (uint8_t *)imageData.buffer.strided.planes[0].basePtr;
    uint8_t *imageBasePtr_h = NULL;
    CupvaMemGetHostPointer((void **)&imageBasePtr_h, (void *)imageBasePtr);

    for (int p = 0; p < numPlanes; p++)
    {
        int32_t height    = imageData.buffer.strided.planes[p].height;
        int32_t width     = imageData.buffer.strided.planes[p].width;
        int32_t linePitch = imageData.buffer.strided.planes[p].rowStride;

        int32_t bpp;
        nvcvImageFormatGetPlaneBitsPerPixel(imageData.format, p, &bpp);
        int32_t bytesPerPixel = bpp / 8;

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                for (int k = 0; k < bytesPerPixel; k++)
                {
                    imageBasePtr_h[i * linePitch + j * bytesPerPixel + k] = generatorU8();
                }
            }
        }
        imageBasePtr_h += linePitch * height;
    }
}

int32_t checkOutputImpl(NVCVImageHandle refImageHandle, NVCVImageHandle pvaImageHandle)
{
    NVCVImageData refImageData;
    nvcvImageExportData(refImageHandle, &refImageData);

    NVCVImageData pvaImageData;
    nvcvImageExportData(pvaImageHandle, &pvaImageData);

    int32_t numPlanes;
    nvcvImageFormatGetNumPlanes(refImageData.format, &numPlanes);

    int32_t errCnt                  = 0;
    const float errorRatioThreshold = 1e-4;
    uint8_t *ref                    = NULL;
    CupvaMemGetHostPointer((void **)&ref, (void *)refImageData.buffer.strided.planes[0].basePtr);
    uint8_t *pva = NULL;
    CupvaMemGetHostPointer((void **)&pva, (void *)pvaImageData.buffer.strided.planes[0].basePtr);
    for (int32_t p = 0; p < numPlanes; p++)
    {
        int32_t bpp;
        nvcvImageFormatGetPlaneBitsPerPixel(refImageData.format, p, &bpp);
        int32_t bytesPerPixel = bpp / 8;

        int32_t height = refImageData.buffer.strided.planes[p].height;
        int32_t width  = refImageData.buffer.strided.planes[p].width;
        width          = width * bytesPerPixel;
        // linePitch is the number of bytes per line
        int32_t linePitch = refImageData.buffer.strided.planes[p].rowStride;

        for (int32_t y = 0; y < height; y++)
        {
            for (int32_t x = 0; x < width; x++)
            {
                uint8_t refData = ref[y * linePitch + x];
                uint8_t cmpData = pva[y * linePitch + x];

                if (std::abs(refData - cmpData) > 1)
                {
                    errCnt++;
                }
            }
        }
        const int32_t totalPixelsPerPlane = height * width;
        const float errorRatio            = static_cast<float>(errCnt) / totalPixelsPerPlane;

        // There are some reasons why VPU is hard to achieve bit exactness with CPU reference:
        // 1. VPU relies on MUFU functions like vrcp to generate coordinates in flight, while the CPU reference directly uses floating point division.
        // 2. It's hard to ensure CPU floating reference having same fp32 operation orders with VPU. Essentially, VPU and CPU have different floating instructions.
        if (errorRatio > errorRatioThreshold)
        {
            printf("Too many mismatches for plane %d, get %f%% mismatch pixels\n", p, errorRatio * 100);
            return 1;
        }

        errCnt = 0;
        ref += linePitch * height;
        pva += linePitch * height;
    }
    return 0;
}

void samplingFilter(uint8_t *inImagePtr, uint8_t *outImagePtr, PVAInterpolationType interpType, float inX, float inY,
                    int32_t outX, int32_t outY, int32_t inWidth, int32_t inHeight, int32_t inLinePitch,
                    int32_t outLinePitch)
{
    if (interpType == PVA_INTERPOLATION_NN)
    {
        int32_t inX_i                           = static_cast<int32_t>(inX * (1 << QBITS) + 0.5f);
        int32_t inY_i                           = static_cast<int32_t>(inY * (1 << QBITS) + 0.5f);
        int32_t x_i                             = static_cast<int32_t>((inX_i + (1 << (QBITS - 1))) >> QBITS);
        int32_t y_i                             = static_cast<int32_t>((inY_i + (1 << (QBITS - 1))) >> QBITS);
        bool outOfRange                         = (x_i >= inWidth) || (y_i >= inHeight);
        outImagePtr[outY * outLinePitch + outX] = outOfRange ? 0 : inImagePtr[y_i * inLinePitch + x_i];
    }
    else if (interpType == PVA_INTERPOLATION_LINEAR)
    {
        int32_t x_l = static_cast<int32_t>(inX);
        int32_t x_r = x_l + 1;
        int32_t y_t = static_cast<int32_t>(inY);
        int32_t y_b = y_t + 1;

        // matched with DLUT behavior, when index is out of boundary within 1 pixel at right/bottom direction, interpolate with 0
        int32_t p_tl = inImagePtr[y_t * inLinePitch + x_l];
        int32_t p_tr = x_r == inWidth ? 0 : inImagePtr[y_t * inLinePitch + x_r];
        int32_t p_bl = y_b == inHeight ? 0 : inImagePtr[y_b * inLinePitch + x_l];
        int32_t p_br = x_r == inWidth ? 0 : y_b == inHeight ? 0 : inImagePtr[y_b * inLinePitch + x_r];

        float frac_x = inX - x_l;
        float frac_y = inY - y_t;

        auto qbitsInterpolation = [=](float frac, int32_t in1, int32_t in2) {
            int32_t frac_i = static_cast<int32_t>(frac * (float)(1 << QBITS) + 0.5f);
            int32_t one_q  = 1 << QBITS;
            return (in1 * one_q + (in2 - in1) * frac_i + (one_q / 2u)) >> QBITS;
        };
        int32_t res_top                         = qbitsInterpolation(frac_x, p_tl, p_tr);
        int32_t res_bot                         = qbitsInterpolation(frac_x, p_bl, p_br);
        int32_t res                             = qbitsInterpolation(frac_y, res_top, res_bot);
        outImagePtr[outY * outLinePitch + outX] = res;
    }
}

void samplingFilterSP(uint8_t *inImagePtr, uint8_t *outImagePtr, PVAInterpolationType interpType, int32_t inX_q,
                      int32_t inY_q, int32_t outX, int32_t outY, int32_t inWidth, int32_t inHeight, int32_t inLinePitch,
                      int32_t outLinePitch)
{
    if (interpType == PVA_INTERPOLATION_NN)
    {
        int32_t half_q                              = 1 << (QBITS - 1);
        int32_t x_i                                 = (inX_q + half_q) >> QBITS;
        int32_t y_i                                 = (inY_q + half_q) >> QBITS;
        bool outOfRange                             = (x_i >= inWidth / 2) || (y_i >= inHeight / 2);
        outImagePtr[outY * outLinePitch + outX * 2] = outOfRange ? 128 : inImagePtr[y_i * inLinePitch + x_i * 2];
        outImagePtr[outY * outLinePitch + outX * 2 + 1] =
            outOfRange ? 128 : inImagePtr[y_i * inLinePitch + x_i * 2 + 1];
    }
    else if (interpType == PVA_INTERPOLATION_LINEAR)
    {
        int32_t x_l = inX_q >> QBITS;
        int32_t x_r = x_l + 1;
        int32_t y_t = inY_q >> QBITS;
        int32_t y_b = y_t + 1;

        const int32_t UVWidth  = inWidth / 2;
        const int32_t UVHeight = inHeight / 2;
        int32_t u_tl           = inImagePtr[y_t * inLinePitch + x_l * 2];
        int32_t u_tr           = x_r == UVWidth ? 0 : inImagePtr[y_t * inLinePitch + x_r * 2];
        int32_t u_bl           = y_b == UVHeight ? 0 : inImagePtr[y_b * inLinePitch + x_l * 2];
        int32_t u_br           = x_r == UVWidth ? 0 : y_b == UVHeight ? 0 : inImagePtr[y_b * inLinePitch + x_r * 2];
        int32_t v_tl           = inImagePtr[y_t * inLinePitch + x_l * 2 + 1];
        int32_t v_tr           = x_r == UVWidth ? 0 : inImagePtr[y_t * inLinePitch + x_r * 2 + 1];
        int32_t v_bl           = y_b == UVHeight ? 0 : inImagePtr[y_b * inLinePitch + x_l * 2 + 1];
        int32_t v_br           = x_r == UVWidth ? 0 : y_b == UVHeight ? 0 : inImagePtr[y_b * inLinePitch + x_r * 2 + 1];

        int32_t frac_x = inX_q - (x_l * (1 << QBITS));
        int32_t frac_y = inY_q - (y_t * (1 << QBITS));

        auto qbitsInterpolation = [=](int32_t frac, int32_t in1, int32_t in2) {
            int32_t one_q = 1 << QBITS;
            return (in1 * one_q + (in2 - in1) * frac + (one_q / 2u)) >> QBITS;
        };

        int32_t resTop0                             = qbitsInterpolation(frac_x, u_tl, u_tr);
        int32_t resBot0                             = qbitsInterpolation(frac_x, u_bl, u_br);
        int32_t res0                                = qbitsInterpolation(frac_y, resTop0, resBot0);
        outImagePtr[outY * outLinePitch + outX * 2] = res0;

        int32_t resTop1                                 = qbitsInterpolation(frac_x, v_tl, v_tr);
        int32_t resBot1                                 = qbitsInterpolation(frac_x, v_bl, v_br);
        int32_t res1                                    = qbitsInterpolation(frac_y, resTop1, resBot1);
        outImagePtr[outY * outLinePitch + outX * 2 + 1] = res1;
    }
}

void processQuadPlanar(uint8_t *inImagePtr, uint8_t *outImagePtr, PVAInterpolationType interpType,
                       std::array<float, 4> inQuadX, std::array<float, 4> inQuadY, int32_t outXStart, int32_t outYStart,
                       int32_t inWidth, int32_t inHeight, int32_t inLinePitch, int32_t outLinePitch)
{
    for (int32_t yy = 0; yy < 2; yy++)
    {
        for (int32_t xx = 0; xx < 2; xx++)
        {
            int32_t outX = outXStart + xx;
            int32_t outY = outYStart + yy;
            float inX    = inQuadX[yy * 2 + xx];
            float inY    = inQuadY[yy * 2 + xx];
            // border conditions copied from VPI
            if (inX < 0.0f || inX >= static_cast<float>(inWidth) || inY < 0.0f || inY >= static_cast<float>(inHeight))
            {
                outImagePtr[outY * outLinePitch + outX] = 0;
            }
            else
            {
                samplingFilter(inImagePtr, outImagePtr, interpType, inX, inY, outX, outY, inWidth, inHeight,
                               inLinePitch, outLinePitch);
            }
        }
    }
}

// process the semi-planar image in NV12 format
void processQuadSemiPlanar(uint8_t *inImagePtr, uint8_t *outImagePtr, PVAInterpolationType interpType,
                           std::array<float, 4> inQuadX, std::array<float, 4> inQuadY, int32_t outXStart,
                           int32_t outYStart, int32_t inWidth, int32_t inHeight, int32_t inLinePitch,
                           int32_t outLinePitch)
{
    bool isClipped = false;
    for (int32_t yy = 0; yy < 2; yy++)
    {
        for (int32_t xx = 0; xx < 2; xx++)
        {
            if (inQuadX[yy * 2 + xx] < 0 || inQuadX[yy * 2 + xx] >= inWidth || inQuadY[yy * 2 + xx] < 0 ||
                inQuadY[yy * 2 + xx] >= inHeight)
            {
                isClipped = true;
            }
        }
    }

    if (isClipped)
    {
        constexpr uint8_t chromaBlack{1 << ((sizeof(uint8_t) * 8) - 1)};
        const int32_t idx    = (outYStart / 2) * outLinePitch + (outXStart / 2) * 2;
        outImagePtr[idx]     = chromaBlack;
        outImagePtr[idx + 1] = chromaBlack;
    }
    else
    {
        auto getSubsampleCoords = [](float a, float b, float c, float d) {
            const int32_t half_q = 1 << (QBITS - 1);
            auto fp2Fixed        = [=](float a) {
                return static_cast<int32_t>(a * (1 << QBITS) + 0.5f);
            };

            int32_t fixedA = fp2Fixed(a);
            int32_t fixedB = fp2Fixed(b);
            int32_t fixedC = fp2Fixed(c);
            int32_t fixedD = fp2Fixed(d);
            int32_t chroma = ((fixedA + fixedB + fixedC + fixedD) + 2) >> 2;
            chroma         = ((chroma - half_q) + 1) >> 1;
            return chroma;
        };

        // chroma = ((x1 + x2 + x3 + x4) * 0.25f - 0.5f) * 0.5f
        int32_t chromaX = getSubsampleCoords(inQuadX[0], inQuadX[1], inQuadX[2], inQuadX[3]);
        int32_t chromaY = getSubsampleCoords(inQuadY[0], inQuadY[1], inQuadY[2], inQuadY[3]);
        samplingFilterSP(inImagePtr, outImagePtr, interpType, chromaX, chromaY, outXStart / 2, outYStart / 2, inWidth,
                         inHeight, inLinePitch, outLinePitch);
    }
}

void matrixMultiply3x3(WarpMatrixType a, WarpMatrixType b, WarpMatrixType c)
{
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
}

void remappingCords(std::array<int32_t, 2> &inCords, const std::array<float, 3> &outCords, WarpMatrixType invWarpMatrix,
                    PVAWarpTransformationType transformType)
{
    float inX = invWarpMatrix[0][0] * outCords[0] + invWarpMatrix[0][1] * outCords[1] + invWarpMatrix[0][2];
    float inY = invWarpMatrix[1][0] * outCords[0] + invWarpMatrix[1][1] * outCords[1] + invWarpMatrix[1][2];
    if (transformType == PVAWarpTransformationType::PVA_WARP_TRANSFORMATION_PERSPECTIVE)
    {
        float inZ = invWarpMatrix[2][0] * outCords[0] + invWarpMatrix[2][1] * outCords[1] + invWarpMatrix[2][2];
        if (inZ != 0.0f)
        {
            // VPU will use vrcp for multiplication with reciprocal of inZ
            inZ = 1.0f / inZ;
        }
        inCords[0] = static_cast<int32_t>(inX * inZ);
        inCords[1] = static_cast<int32_t>(inY * inZ);
    }
    else
    {
        inCords[0] = static_cast<int32_t>(inX);
        inCords[1] = static_cast<int32_t>(inY);
    }
}

float det(const WarpMatrixType m)
{
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) + m[0][1] * (m[1][2] * m[2][0] - m[1][0] * m[2][2]) +
           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

void inv(WarpMatrixType im, const WarpMatrixType m, const float d)
{
    im[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / d;
    im[0][1] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) / d;
    im[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / d;
    im[1][0] = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) / d;
    im[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / d;
    im[1][2] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) / d;
    im[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / d;
    im[2][1] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) / d;
    im[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / d;
}

void warpPerspectiveRefImpl(NVCVImageHandle inImageHandle, NVCVImageHandle outImageHandle,
                            PVAInterpolationType interpType, PVAWarpTransformationType transformType)
{
    NVCVImageData inImageData;
    nvcvImageExportData(inImageHandle, &inImageData);
    uint8_t *inImagePtr = NULL;
    CupvaMemGetHostPointer((void **)&inImagePtr, (void *)inImageData.buffer.strided.planes[0].basePtr);

    NVCVImageData outImageData;
    nvcvImageExportData(outImageHandle, &outImageData);
    uint8_t *outImagePtr = NULL;
    CupvaMemGetHostPointer((void **)&outImagePtr, (void *)outImageData.buffer.strided.planes[0].basePtr);

    int32_t inHeight    = inImageData.buffer.strided.planes[0].height;
    int32_t inWidth     = inImageData.buffer.strided.planes[0].width;
    int32_t inLinePitch = inImageData.buffer.strided.planes[0].rowStride;

    int32_t outHeight    = outImageData.buffer.strided.planes[0].height;
    int32_t outWidth     = outImageData.buffer.strided.planes[0].width;
    int32_t outLinePitch = outImageData.buffer.strided.planes[0].rowStride;

    WarpMatrixType invWarpMatrix;
    setupWarpMatrix(invWarpMatrix, inWidth, inHeight, outWidth, outHeight);

    // emulate the VPU computations for coordinates tile by tile
    const int32_t tileSize = TILE_W * TILE_H;
    std::vector<float> cordsBaseNumX(tileSize);
    std::vector<float> cordsBaseNumY(tileSize);
    std::vector<float> cordsBaseDen(tileSize);
    for (int32_t i = 0; i < TILE_H; i++)
    {
        for (int32_t j = 0; j < TILE_W; j++)
        {
            cordsBaseNumX[i * TILE_W + j] =
                (invWarpMatrix[0][0] * (j)) + (invWarpMatrix[0][1] * (i)) + invWarpMatrix[0][2];
            cordsBaseNumY[i * TILE_W + j] =
                (invWarpMatrix[1][0] * (j)) + (invWarpMatrix[1][1] * (i)) + invWarpMatrix[1][2];
            cordsBaseDen[i * TILE_W + j] =
                (invWarpMatrix[2][0] * (j)) + (invWarpMatrix[2][1] * (i)) + invWarpMatrix[2][2];
        }
    }
    std::array<float, 3> dMatX, dMatY;
    for (int32_t i = 0; i < 3; i++)
    {
        dMatX[i] = invWarpMatrix[i][0] * (TILE_W);
        dMatY[i] = invWarpMatrix[i][1] * (TILE_H);
    }

    std::array<float, 4> cordsX = {};
    std::array<float, 4> cordsY = {};
    for (int32_t y = 0; y < outHeight; y += 2)
    {
        const int32_t tileIdY = y / TILE_H;
        for (int32_t x = 0; x < outWidth; x += 2)
        {
            const int32_t tileIdX = x / TILE_W;
            for (int32_t yy = 0; yy < 2; yy++)
            {
                const int32_t dy = y + yy - tileIdY * TILE_H;
                for (int32_t xx = 0; xx < 2; xx++)
                {
                    const int32_t dx    = x + xx - tileIdX * TILE_W;
                    float resX          = cordsBaseNumX[dy * TILE_W + dx];
                    float dResX         = dMatX[0] * tileIdX + dMatY[0] * tileIdY;
                    cordsX[yy * 2 + xx] = resX + dResX;
                    float resY          = cordsBaseNumY[dy * TILE_W + dx];
                    float dResY         = dMatX[1] * tileIdX + dMatY[1] * tileIdY;
                    cordsY[yy * 2 + xx] = resY + dResY;

                    if (transformType == PVAWarpTransformationType::PVA_WARP_TRANSFORMATION_PERSPECTIVE)
                    {
                        float resZ   = cordsBaseDen[dy * TILE_W + dx];
                        float dResZ  = dMatX[2] * tileIdX + dMatY[2] * tileIdY;
                        float cordsZ = resZ + dResZ;

                        if (cordsZ != 0.0f)
                        {
                            // VPU will use vrcp for multiplication with reciprocal of cordsZ
                            cordsZ = 1.0f / cordsZ;
                            cordsX[yy * 2 + xx] *= cordsZ;
                            cordsY[yy * 2 + xx] *= cordsZ;
                        }
                    }
                }
            }

            switch (inImageData.format)
            {
            case NVCV_IMAGE_FORMAT_U8:
            {
                processQuadPlanar(inImagePtr, outImagePtr, interpType, cordsX, cordsY, x, y, inWidth, inHeight,
                                  inLinePitch, outLinePitch);
                break;
            }
            case NVCV_IMAGE_FORMAT_NV12_ER:
            case NVCV_IMAGE_FORMAT_NV12:
            {
                processQuadPlanar(inImagePtr, outImagePtr, interpType, cordsX, cordsY, x, y, inWidth, inHeight,
                                  inLinePitch, outLinePitch);

                // process semi-planar image
                uint8_t *inImageSPPtr  = inImagePtr + inLinePitch * inHeight;
                uint8_t *outImageSPPtr = outImagePtr + outLinePitch * outHeight;
                processQuadSemiPlanar(inImageSPPtr, outImageSPPtr, interpType, cordsX, cordsY, x, y, inWidth, inHeight,
                                      inLinePitch, outLinePitch);
                break;
            }
            default:
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input data format!");
                break;
            }
            }
        }
    }
}

} // namespace

NVCVStatus pvaWarpPerspectiveValidate(WarpMatrixType invWarpMatrix, const int32_t inWidth, const int32_t inHeight,
                                      const int32_t inLinePitch, const int32_t outWidth, const int32_t outHeight,
                                      const PVAWarpTransformationType transformType)
{
    return nvcv::ProtectCall([&] {
        const int32_t nTileX    = (outWidth + TILE_W - 1) / TILE_W;
        const int32_t nTileY    = (outHeight + TILE_H - 1) / TILE_H;
        const int32_t nTileSize = nTileX * nTileY;

        for (int32_t i = 0; i < nTileY; i++)
        {
            for (int32_t j = 0; j < nTileX; j++)
            {
                float left = static_cast<float>(j * TILE_W);
                float right =
                    static_cast<float>((left + TILE_W - 1) > (inWidth - 1) ? (inWidth - 1) : (left + TILE_W - 1));
                float top = static_cast<float>(i * TILE_H);
                float bottom =
                    static_cast<float>((top + TILE_H - 1) > (inHeight - 1) ? (inHeight - 1) : (top + TILE_H - 1));

                std::array<int32_t, 2> inTl, inTr, inBl, inBr;
                remappingCords(inTl, {left, top, 1.0f}, invWarpMatrix, transformType);
                remappingCords(inTr, {right, top, 1.0f}, invWarpMatrix, transformType);
                remappingCords(inBl, {left, bottom, 1.0f}, invWarpMatrix, transformType);
                remappingCords(inBr, {right, bottom, 1.0f}, invWarpMatrix, transformType);

                int32_t minX = std::min({inTl[0], inTr[0], inBl[0], inBr[0]});
                int32_t maxX = std::max({inTl[0], inTr[0], inBl[0], inBr[0]});
                int32_t minY = std::min({inTl[1], inTr[1], inBl[1], inBr[1]});
                int32_t maxY = std::max({inTl[1], inTr[1], inBl[1], inBr[1]});

                if (maxX < 0 || maxY < 0 || minX >= inWidth || minY >= inHeight)
                {
                    // all black tiles, treating tile size as 0
                    continue;
                }

                minX = std::max(0, minX);
                minY = std::max(0, minY);
                // VPI treat [0, width/height) as valid range
                maxX = std::min(inWidth, maxX);
                maxY = std::min(inHeight, maxY);

                // VPU implementation details for calculating the tile size
                int32_t tx = maxX - minX + 3;
                int32_t ty = maxY - minY + 3;

                if (tx * ty > MAX_TABLE_SIZE)
                {
                    throw nvcv::Exception(
                        nvcv::Status::ERROR_INVALID_ARGUMENT,
                        "Invalid inverse warp matrix, PVA can only handle tile size smaller than 32KB!");
                }
            }
        }
    });
}

void setupWarpMatrix(WarpMatrixType invWarpMatrix, const int32_t inWidth, const int32_t inHeight,
                     const int32_t outWidth, const int32_t outHeight)
{
    // align the warp matrix with VPI
    WarpMatrixType t1 = {{1.0f, 0.0f, inWidth / -2.0f}, {0.0f, 1.0f, inHeight / -2.0f}, {0.0f, 0.0f, 1.0f}};
    // construct a perspective transform
    float v1 = sin(M_PI / 10) * 0.0005f;
    float v2 = cos(M_PI / 8) * 0.0005f;
    // VPI sample
    WarpMatrixType P  = {{1.1f, 0.0f, 0.0f}, {0.0f, 1.1f, 0.0f}, {v1, v2, 1.0f}};
    WarpMatrixType t2 = {{1.0f, 0.0f, outWidth / 2.0f}, {0.0f, 1.0f, outHeight / 2.0f}, {0.0f, 0.0f, 1.0f}};

    // warp_matrix = t2 * P * t1
    WarpMatrixType tmp;
    matrixMultiply3x3(t2, P, tmp);
    WarpMatrixType warpMatrix;
    matrixMultiply3x3(tmp, t1, warpMatrix);

    // reverse mapping need to inverse the warp matrix
    float d = det(warpMatrix);
    inv(invWarpMatrix, warpMatrix, d);
}

void generateRandomWarpPerspectiveInput(NVCVImageHandle imageHandle, bool isRandom = true)
{
    generateRandomInputImpl(imageHandle, isRandom);
}

int32_t checkWarpPerspectiveOutput(NVCVImageHandle refImageHandle, NVCVImageHandle pvaImageHandle)
{
    return checkOutputImpl(refImageHandle, pvaImageHandle);
}

void warpPerspectiveRef(NVCVImageHandle inImageHandle, NVCVImageHandle outImageHandle, PVAInterpolationType interpType,
                        PVAWarpTransformationType transformType)
{
    NVCVImageData inImageData;
    nvcvImageExportData(inImageHandle, &inImageData);

    warpPerspectiveRefImpl(inImageHandle, outImageHandle, interpType, transformType);
}