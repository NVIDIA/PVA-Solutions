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

#include "remap_ref.h"

#include <nvcv/Tensor.hpp>

#include <OpRemap.h>
#include <nvcv/Status.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <type_traits>

#define QBITS 16

namespace {
void generateRandomInputImpl(NVCVImageHandle imageHandle, bool isRandom)
{
    NVCVImageData imageData;
    nvcvImageExportData(imageHandle, &imageData);

    int32_t numPlanes;
    nvcvImageFormatGetNumPlanes(imageData.format, &numPlanes);

    uint8_t *imageBasePtr   = (uint8_t *)imageData.buffer.strided.planes[0].basePtr;
    uint8_t *imageBasePtr_h = NULL;
    CupvaMemGetHostPointer((void **)&imageBasePtr_h, (void *)imageBasePtr);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dist(0, std::numeric_limits<uint8_t>::max());

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
            for (int j = 0; j < width * bytesPerPixel; j++)
            {
                imageBasePtr_h[i * linePitch + j] = isRandom ? dist(gen) : (i * linePitch + j) % 256;
            }
        }

        imageBasePtr_h += linePitch * height;
    }
}

void generateRandomWarpMapImpl(NVCVTensorHandle tensorHandle, int32_t inWidth, int32_t inHeight, bool isRandom)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    float *tensorBasePtr = (float *)tensorData.buffer.strided.basePtr;
    float *tensorPtr_h   = NULL;
    CupvaMemGetHostPointer((void **)&tensorPtr_h, (void *)tensorBasePtr);

    int32_t height    = tensorData.shape[0];
    int32_t width     = tensorData.shape[1];
    int32_t linePitch = tensorData.buffer.strided.strides[0] / sizeof(float);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distX(-5.0f, static_cast<float>(inWidth + 5));
    std::uniform_real_distribution<float> distY(-5.0f, static_cast<float>(inHeight + 5));

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            tensorPtr_h[i * linePitch + 2 * j]     = isRandom ? distX(gen) : static_cast<float>(inWidth - 1 - j);
            tensorPtr_h[i * linePitch + 2 * j + 1] = isRandom ? distY(gen) : static_cast<float>(inHeight - 1 - i);
        }
    }
}

int32_t checkOutputImpl(NVCVImageHandle refImageHandle, NVCVImageHandle pvaImageHandle)
{
    NVCVImageData refImageData;
    nvcvImageExportData(refImageHandle, &refImageData);
    uint8_t *ref = NULL;
    CupvaMemGetHostPointer((void **)&ref, (void *)refImageData.buffer.strided.planes[0].basePtr);

    NVCVImageData pvaImageData;
    nvcvImageExportData(pvaImageHandle, &pvaImageData);
    uint8_t *pva = NULL;
    CupvaMemGetHostPointer((void **)&pva, (void *)pvaImageData.buffer.strided.planes[0].basePtr);

    int32_t numPlanes;
    nvcvImageFormatGetNumPlanes(refImageData.format, &numPlanes);

    for (int32_t p = 0; p < numPlanes; p++)
    {
        int32_t bpp;
        nvcvImageFormatGetPlaneBitsPerPixel(refImageData.format, p, &bpp);
        int32_t bytesPerPixel = bpp / 8;

        int32_t height    = refImageData.buffer.strided.planes[p].height;
        int32_t width     = refImageData.buffer.strided.planes[p].width * bytesPerPixel;
        int32_t linePitch = refImageData.buffer.strided.planes[p].rowStride;

        int32_t errCnt = 0;
        for (int32_t y = 0; y < height; y++)
        {
            for (int32_t x = 0; x < width; x++)
            {
                uint8_t refData = ref[y * linePitch + x];
                uint8_t cmpData = pva[y * linePitch + x];

                if (refData != cmpData)
                {
                    printf("Found mismatch in plane %d, where y = %d, x = %d, refData = %d, cmpData = %d\n", p, y, x,
                           refData, cmpData);
                    errCnt++;
                }
            }
        }

        if (errCnt > 0)
        {
            printf("Found %d errors in plane %d. \n", errCnt, p);
            return 1;
        }

        ref += linePitch * height;
        pva += linePitch * height;
    }
    return 0;
}

void samplingFilter(uint8_t *inImagePtr, uint8_t *outImagePtr, PVAInterpolationType interpType, float inX, float inY,
                    int32_t outX, int32_t outY, int32_t inWidth, int32_t inHeight, int32_t inLinePitch,
                    int32_t outLinePitch)
{
    // For CONSTANT mode, clamp in bound coords; for BPE mode, clamp all coords
    inX = std::max(std::min(inX, static_cast<float>(inWidth - 1)), 0.0f);
    inY = std::max(std::min(inY, static_cast<float>(inHeight - 1)), 0.0f);

    if (interpType == PVA_INTERPOLATION_NN)
    {
        int32_t inXq = static_cast<int32_t>(inX * (1 << QBITS) + 0.5f);
        int32_t inYq = static_cast<int32_t>(inY * (1 << QBITS) + 0.5f);
        int32_t xi   = static_cast<int32_t>((inXq + (1 << (QBITS - 1))) >> QBITS);
        int32_t yi   = static_cast<int32_t>((inYq + (1 << (QBITS - 1))) >> QBITS);

        outImagePtr[outY * outLinePitch + outX] = inImagePtr[yi * inLinePitch + xi];
    }
    else if (interpType == PVA_INTERPOLATION_LINEAR)
    {
        int32_t xl = static_cast<int32_t>(inX);
        int32_t xr = std::min(xl + 1, inWidth - 1);
        int32_t yt = static_cast<int32_t>(inY);
        int32_t yb = std::min(yt + 1, inHeight - 1);

        int32_t tl = inImagePtr[yt * inLinePitch + xl];
        int32_t tr = inImagePtr[yt * inLinePitch + xr];
        int32_t bl = inImagePtr[yb * inLinePitch + xl];
        int32_t br = inImagePtr[yb * inLinePitch + xr];

        int32_t fracXQ = static_cast<int32_t>(inX * (1 << QBITS) + 0.5f) - static_cast<int32_t>(xl * (1 << QBITS));
        int32_t fracYQ = static_cast<int32_t>(inY * (1 << QBITS) + 0.5f) - static_cast<int32_t>(yt * (1 << QBITS));

        auto qbitsInterpolation = [=](int32_t fracQ, int32_t in1, int32_t in2) {
            int32_t oneQ = 1 << QBITS;
            return (in1 * oneQ + (in2 - in1) * fracQ + (oneQ / 2)) >> QBITS;
        };

        int32_t top                             = qbitsInterpolation(fracXQ, tl, tr);   // tl + (tr - tl) * fracX
        int32_t bot                             = qbitsInterpolation(fracXQ, bl, br);   // bl + (br - bl) * fracX
        outImagePtr[outY * outLinePitch + outX] = qbitsInterpolation(fracYQ, top, bot); // top + (bot - top) * fracY
    }
}

void samplingFilterUV(uint8_t *inImagePtr, uint8_t *outImagePtr, PVAInterpolationType interpType, int32_t inXq,
                      int32_t inYq, int32_t outX, int32_t outY, int32_t inWidth, int32_t inHeight, int32_t inLinePitch,
                      int32_t outLinePitch)
{
    // For CONSTANT mode, clamp in bound coords; for BPE mode, clamp all coords
    inXq = std::max(std::min(inXq, (inWidth / 2 - 1) << QBITS), 0);
    inYq = std::max(std::min(inYq, (inHeight / 2 - 1) << QBITS), 0);

    if (interpType == PVA_INTERPOLATION_NN)
    {
        int32_t halfQ = 1 << (QBITS - 1);
        int32_t xi    = (inXq + halfQ) >> QBITS;
        int32_t yi    = (inYq + halfQ) >> QBITS;

        outImagePtr[outY * outLinePitch + outX * 2]     = inImagePtr[yi * inLinePitch + xi * 2];
        outImagePtr[outY * outLinePitch + outX * 2 + 1] = inImagePtr[yi * inLinePitch + xi * 2 + 1];
    }
    else if (interpType == PVA_INTERPOLATION_LINEAR)
    {
        int32_t xl = inXq >> QBITS;
        int32_t xr = std::min(xl + 1, inWidth / 2 - 1);
        int32_t yt = inYq >> QBITS;
        int32_t yb = std::min(yt + 1, inHeight / 2 - 1);

        int32_t utl = inImagePtr[yt * inLinePitch + xl * 2];
        int32_t utr = inImagePtr[yt * inLinePitch + xr * 2];
        int32_t ubl = inImagePtr[yb * inLinePitch + xl * 2];
        int32_t ubr = inImagePtr[yb * inLinePitch + xr * 2];
        int32_t vtl = inImagePtr[yt * inLinePitch + xl * 2 + 1];
        int32_t vtr = inImagePtr[yt * inLinePitch + xr * 2 + 1];
        int32_t vbl = inImagePtr[yb * inLinePitch + xl * 2 + 1];
        int32_t vbr = inImagePtr[yb * inLinePitch + xr * 2 + 1];

        int32_t fracXQ = inXq - (xl << QBITS);
        int32_t fracYQ = inYq - (yt << QBITS);

        auto qbitsInterpolation = [=](int32_t fracQ, int32_t in1, int32_t in2) {
            int32_t oneQ = 1 << QBITS;
            return (in1 * oneQ + (in2 - in1) * fracQ + (oneQ / 2)) >> QBITS;
        };

        int32_t topU = qbitsInterpolation(fracXQ, utl, utr);
        int32_t botU = qbitsInterpolation(fracXQ, ubl, ubr);
        int32_t topV = qbitsInterpolation(fracXQ, vtl, vtr);
        int32_t botV = qbitsInterpolation(fracXQ, vbl, vbr);

        outImagePtr[outY * outLinePitch + outX * 2]     = qbitsInterpolation(fracYQ, topU, botU);
        outImagePtr[outY * outLinePitch + outX * 2 + 1] = qbitsInterpolation(fracYQ, topV, botV);
    }
}

void processQuadPlanar(uint8_t *inImagePtr, uint8_t *outImagePtr, PVAInterpolationType interpType,
                       std::array<float, 4> inQuadX, std::array<float, 4> inQuadY, int32_t outXStart, int32_t outYStart,
                       int32_t inWidth, int32_t inHeight, int32_t inLinePitch, int32_t outLinePitch,
                       NVCVBorderType borderMode)
{
    for (int32_t j = 0; j < 2; j++)
    {
        for (int32_t i = 0; i < 2; i++)
        {
            int32_t outX = outXStart + i;
            int32_t outY = outYStart + j;
            float inX    = inQuadX[j * 2 + i];
            float inY    = inQuadY[j * 2 + i];
            int32_t inXq = static_cast<int32_t>(inX * (1 << QBITS) + 0.5f);
            int32_t inYq = static_cast<int32_t>(inY * (1 << QBITS) + 0.5f);

            // border conditions copied from VPI
            if ((inXq < 0 || inXq >= (inWidth << QBITS) || inYq < 0 || inYq >= (inHeight << QBITS)) &&
                borderMode == NVCV_BORDER_CONSTANT)
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

// process the UV plane in NV12 format
void processQuadUVPlanar(uint8_t *inImagePtr, uint8_t *outImagePtr, PVAInterpolationType interpType,
                         std::array<float, 4> inQuadX, std::array<float, 4> inQuadY, int32_t outXStart,
                         int32_t outYStart, int32_t inWidth, int32_t inHeight, int32_t inLinePitch,
                         int32_t outLinePitch, NVCVBorderType borderMode)
{
    bool isClipped = false;
    for (int32_t j = 0; j < 2; j++)
    {
        for (int32_t i = 0; i < 2; i++)
        {
            float inX    = inQuadX[j * 2 + i];
            float inY    = inQuadY[j * 2 + i];
            int32_t inXq = static_cast<int32_t>(inX * (1 << QBITS) + 0.5f);
            int32_t inYq = static_cast<int32_t>(inY * (1 << QBITS) + 0.5f);

            if (inXq < 0 || inXq >= (inWidth << QBITS) || inYq < 0 || inYq >= (inHeight << QBITS))
            {
                isClipped = true;
            }
        }
    }

    if (isClipped && borderMode == NVCV_BORDER_CONSTANT)
    {
        // border condition copied from VPI
        uint8_t chromaBlack  = 128;
        int32_t idx          = (outYStart / 2) * outLinePitch + (outXStart / 2) * 2;
        outImagePtr[idx]     = chromaBlack;
        outImagePtr[idx + 1] = chromaBlack;
    }
    else
    {
        auto getSubsampleCoords = [](float a, float b, float c, float d) {
            const int32_t halfQ = 1 << (QBITS - 1);

            int32_t fixedA = static_cast<int32_t>(a * (1 << QBITS) + 0.5f);
            int32_t fixedB = static_cast<int32_t>(b * (1 << QBITS) + 0.5f);
            int32_t fixedC = static_cast<int32_t>(c * (1 << QBITS) + 0.5f);
            int32_t fixedD = static_cast<int32_t>(d * (1 << QBITS) + 0.5f);
            int32_t chroma = ((fixedA + fixedB + fixedC + fixedD) + 2) >> 2;
            chroma         = ((chroma - halfQ) + 1) >> 1;
            return chroma;
        };

        // chroma = ((x1 + x2 + x3 + x4) * 0.25f - 0.5f) * 0.5f
        int32_t chromaX = getSubsampleCoords(inQuadX[0], inQuadX[1], inQuadX[2], inQuadX[3]);
        int32_t chromaY = getSubsampleCoords(inQuadY[0], inQuadY[1], inQuadY[2], inQuadY[3]);
        samplingFilterUV(inImagePtr, outImagePtr, interpType, chromaX, chromaY, outXStart / 2, outYStart / 2, inWidth,
                         inHeight, inLinePitch, outLinePitch);
    }
}

void remapRefImpl(NVCVImageHandle inImageHandle, NVCVImageHandle outImageHandle, NVCVTensorHandle warpMapHandle,
                  PVAInterpolationType interpType, NVCVBorderType borderMode)
{
    NVCVImageData inImageData;
    nvcvImageExportData(inImageHandle, &inImageData);
    uint8_t *inImagePtr = NULL;
    CupvaMemGetHostPointer((void **)&inImagePtr, (void *)inImageData.buffer.strided.planes[0].basePtr);

    NVCVImageData outImageData;
    nvcvImageExportData(outImageHandle, &outImageData);
    uint8_t *outImagePtr = NULL;
    CupvaMemGetHostPointer((void **)&outImagePtr, (void *)outImageData.buffer.strided.planes[0].basePtr);

    NVCVTensorData warpMapData;
    nvcvTensorExportData(warpMapHandle, &warpMapData);
    float *warpMapPtr = NULL;
    CupvaMemGetHostPointer((void **)&warpMapPtr, (void *)warpMapData.buffer.strided.basePtr);

    int32_t inHeight    = inImageData.buffer.strided.planes[0].height;
    int32_t inWidth     = inImageData.buffer.strided.planes[0].width;
    int32_t inLinePitch = inImageData.buffer.strided.planes[0].rowStride;

    int32_t outHeight     = outImageData.buffer.strided.planes[0].height;
    int32_t outWidth      = outImageData.buffer.strided.planes[0].width;
    int32_t outLinePitch  = outImageData.buffer.strided.planes[0].rowStride;
    int32_t warpLinePitch = warpMapData.buffer.strided.strides[0] / sizeof(float);

    std::array<float, 4> coordsX = {};
    std::array<float, 4> coordsY = {};
    for (int32_t y = 0; y < outHeight; y += 2)
    {
        for (int32_t x = 0; x < outWidth; x += 2)
        {
            for (int32_t j = 0; j < 2; j++)
            {
                for (int32_t i = 0; i < 2; i++)
                {
                    int32_t index      = (y + j) * warpLinePitch + (x + i) * 2;
                    coordsX[j * 2 + i] = warpMapPtr[index];
                    coordsY[j * 2 + i] = warpMapPtr[index + 1];
                }
            }

            switch (inImageData.format)
            {
            case NVCV_IMAGE_FORMAT_U8:
            {
                processQuadPlanar(inImagePtr, outImagePtr, interpType, coordsX, coordsY, x, y, inWidth, inHeight,
                                  inLinePitch, outLinePitch, borderMode);
                break;
            }
            case NVCV_IMAGE_FORMAT_NV12_ER:
            case NVCV_IMAGE_FORMAT_NV12:
            {
                // process Y plane
                processQuadPlanar(inImagePtr, outImagePtr, interpType, coordsX, coordsY, x, y, inWidth, inHeight,
                                  inLinePitch, outLinePitch, borderMode);

                // process UV plane
                uint8_t *inImageUVPtr  = inImagePtr + inLinePitch * inHeight;
                uint8_t *outImageUVPtr = outImagePtr + outLinePitch * outHeight;
                processQuadUVPlanar(inImageUVPtr, outImageUVPtr, interpType, coordsX, coordsY, x, y, inWidth, inHeight,
                                    inLinePitch, outLinePitch, borderMode);
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

void generateRandomRemapInput(NVCVImageHandle imageHandle, bool isRandom = false)
{
    generateRandomInputImpl(imageHandle, isRandom);
}

void generateRandomRemapWarpMap(NVCVTensorHandle tensorHandle, int32_t inWidth, int32_t inHeight, bool isRandom = false)
{
    generateRandomWarpMapImpl(tensorHandle, inWidth, inHeight, isRandom);
}

int32_t checkRemapOutput(NVCVImageHandle refImageHandle, NVCVImageHandle pvaImageHandle)
{
    return checkOutputImpl(refImageHandle, pvaImageHandle);
}

void remapRef(NVCVImageHandle inImageHandle, NVCVImageHandle outImageHandle, NVCVTensorHandle warpMapHandle,
              PVAInterpolationType interpType, NVCVBorderType borderMode)
{
    remapRefImpl(inImageHandle, outImageHandle, warpMapHandle, interpType, borderMode);
}