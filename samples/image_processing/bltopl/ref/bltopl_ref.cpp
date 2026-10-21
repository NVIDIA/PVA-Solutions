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

#include "bltopl_ref.h"

#include <cupva_host_utils.hpp>

#include <OpBlToPl.h>

#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

namespace {
void generateRandomPlInputImpl(NVCVImageHandle imageHandle, bool isRandom)
{
    NVCVImageData imageData;
    nvcvImageExportData(imageHandle, &imageData);

    uint8_t *basePtr_h{nullptr};
    CupvaMemGetHostPointer((void **)&basePtr_h, (void *)imageData.buffer.strided.planes[0].basePtr);

    std::array<uint8_t *, 2> output = {basePtr_h, basePtr_h + imageData.buffer.strided.planes[0].height *
                                                                  imageData.buffer.strided.planes[0].rowStride};

    std::mt19937 gen;
    gen.seed(1);
    std::uniform_int_distribution<> distrib(0, std::numeric_limits<uint8_t>::max());
    int cnt{0};
    auto generatorU8 = [&]() {
        return static_cast<uint8_t>(isRandom ? (distrib(gen)) : cnt++);
    };

    int32_t numPlanes;
    nvcvImageFormatGetNumPlanes(imageData.format, &numPlanes);
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
                    output[p][i * linePitch + j * bytesPerPixel + k] = generatorU8();
                }
            }
        }
    }
}
} // namespace

void generateRandomPlInput(NVCVImageHandle imageHandle, bool isRandom = true)
{
    generateRandomPlInputImpl(imageHandle, isRandom);
}

int32_t checkBltoplOutput(NVCVImageHandle refImageHandle, NVCVImageHandle pvaImageHandle)
{
    NVCVImageData refImageData;
    nvcvImageExportData(refImageHandle, &refImageData);

    uint8_t *ref{nullptr};
    CupvaMemGetHostPointer((void **)&ref, (void *)refImageData.buffer.strided.planes[0].basePtr);

    NVCVImageData pvaImageData;
    nvcvImageExportData(pvaImageHandle, &pvaImageData);

    uint8_t *pva{nullptr};
    CupvaMemGetHostPointer((void **)&pva, (void *)pvaImageData.buffer.strided.planes[0].basePtr);

    std::array<uint8_t *, 2> refs = {
        ref, ref + refImageData.buffer.strided.planes[0].height * refImageData.buffer.strided.planes[0].rowStride};
    std::array<uint8_t *, 2> pvas = {
        pva, pva + pvaImageData.buffer.strided.planes[0].height * pvaImageData.buffer.strided.planes[0].rowStride};

    int32_t numPlanes;
    nvcvImageFormatGetNumPlanes(refImageData.format, &numPlanes);
    for (int p = 0; p < numPlanes; p++)
    {
        int32_t height    = refImageData.buffer.strided.planes[p].height;
        int32_t width     = refImageData.buffer.strided.planes[p].width;
        int32_t linePitch = refImageData.buffer.strided.planes[p].rowStride;

        int32_t bpp;
        nvcvImageFormatGetPlaneBitsPerPixel(refImageData.format, p, &bpp);
        int32_t bytesPerPixel = bpp / 8;

        // Note that height = Y plane's height + UV plane's height, need to be separately compared if later the layout has been changed
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int z = 0; z < bytesPerPixel; z++)
                {
                    auto refData = refs[p][y * linePitch + x * bytesPerPixel + z];
                    auto cmpData = pvas[p][y * linePitch + x * bytesPerPixel + z];

                    if (refData != cmpData)
                    {
                        printf("Mismatch[%d, %d, %d]: ref = %u, pva = %u\n", y, x, z, refData, cmpData);
                        return 1;
                    }
                }
            }
        }
    }
    return 0;
}
