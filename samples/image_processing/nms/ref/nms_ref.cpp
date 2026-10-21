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

#include "nms_ref.h"

#include <PvaAllocator.h>

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

// Narrow enough that equal and nearly equal neighbors show up frequently.
constexpr int32_t kValueRange = 256;

// Fetch a source pixel, applying the operator's border mode outside the image.
int32_t fetchSource(const int32_t *src, int32_t linePitch, int32_t width, int32_t height, int32_t y, int32_t x,
                    NVCVBorderType borderMode, int32_t borderValue)
{
    if (borderMode == NVCV_BORDER_REPLICATE)
    {
        y = (y < 0) ? 0 : ((y >= height) ? (height - 1) : y);
        x = (x < 0) ? 0 : ((x >= width) ? (width - 1) : x);
    }
    else if (y < 0 || y >= height || x < 0 || x >= width)
    {
        return borderValue;
    }

    return src[(y * linePitch) + x];
}

} // namespace

void generateRandomNmsInput(NVCVTensorHandle tensorHandle)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    int32_t *tensorPtr_h = nullptr;
    CupvaMemGetHostPointer(reinterpret_cast<void **>(&tensorPtr_h),
                           reinterpret_cast<void *>(tensorData.buffer.strided.basePtr));

    const int32_t height    = static_cast<int32_t>(tensorData.shape[0]);
    const int32_t width     = static_cast<int32_t>(tensorData.shape[1]);
    const int32_t linePitch = static_cast<int32_t>(tensorData.buffer.strided.strides[0] / sizeof(int32_t));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int32_t> dist(0, kValueRange - 1);

    for (int32_t i = 0; i < height; i++)
    {
        for (int32_t j = 0; j < width; j++)
        {
            tensorPtr_h[(i * linePitch) + j] = dist(gen);
        }
    }
}

void nmsRef(NVCVTensorHandle in, NVCVTensorHandle out, int32_t nmsSize, NVCVBorderType borderMode, int32_t borderValue)
{
    NVCVTensorData inData, outData;
    nvcvTensorExportData(in, &inData);
    nvcvTensorExportData(out, &outData);

    int32_t *src = nullptr;
    int32_t *dst = nullptr;
    CupvaMemGetHostPointer(reinterpret_cast<void **>(&src), reinterpret_cast<void *>(inData.buffer.strided.basePtr));
    CupvaMemGetHostPointer(reinterpret_cast<void **>(&dst), reinterpret_cast<void *>(outData.buffer.strided.basePtr));

    const int32_t height     = static_cast<int32_t>(inData.shape[0]);
    const int32_t width      = static_cast<int32_t>(inData.shape[1]);
    const int32_t srcPitch   = static_cast<int32_t>(inData.buffer.strided.strides[0] / sizeof(int32_t));
    const int32_t dstPitch   = static_cast<int32_t>(outData.buffer.strided.strides[0] / sizeof(int32_t));
    const int32_t radius     = nmsSize >> 1;
    const int32_t centerBias = (radius * nmsSize) + radius;

    // Horizontal pass over the padded image: one row per output row plus the vertical halo rows.
    const int32_t scratchHeight = height + (2 * radius);
    std::vector<int32_t> scratch(static_cast<size_t>(width) * static_cast<size_t>(scratchHeight));

    for (int32_t sy = 0; sy < scratchHeight; sy++)
    {
        const int32_t y = sy - radius;
        for (int32_t x = 0; x < width; x++)
        {
            int32_t best = fetchSource(src, srcPitch, width, height, y, x - radius, borderMode, borderValue);
            for (int32_t dx = 1; dx < nmsSize; dx++)
            {
                const int32_t candidate =
                    fetchSource(src, srcPitch, width, height, y, x - radius + dx, borderMode, borderValue) + dx;
                best = (candidate > best) ? candidate : best;
            }
            scratch[(static_cast<size_t>(sy) * static_cast<size_t>(width)) + static_cast<size_t>(x)] = best;
        }
    }

    for (int32_t y = 0; y < height; y++)
    {
        for (int32_t x = 0; x < width; x++)
        {
            int32_t best = scratch[(static_cast<size_t>(y) * static_cast<size_t>(width)) + static_cast<size_t>(x)];
            for (int32_t dy = 1; dy < nmsSize; dy++)
            {
                const int32_t candidate =
                    scratch[(static_cast<size_t>(y + dy) * static_cast<size_t>(width)) + static_cast<size_t>(x)] +
                    (nmsSize * dy);
                best = (candidate > best) ? candidate : best;
            }

            const int32_t center    = src[(y * srcPitch) + x];
            dst[(y * dstPitch) + x] = ((center + centerBias) == best) ? center : 0;
        }
    }
}

int32_t checkNmsOutput(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle)
{
    NVCVTensorData refData, pvaData;
    nvcvTensorExportData(refTensorHandle, &refData);
    nvcvTensorExportData(pvaTensorHandle, &pvaData);

    int32_t *ref = nullptr;
    int32_t *pva = nullptr;
    CupvaMemGetHostPointer(reinterpret_cast<void **>(&ref), reinterpret_cast<void *>(refData.buffer.strided.basePtr));
    CupvaMemGetHostPointer(reinterpret_cast<void **>(&pva), reinterpret_cast<void *>(pvaData.buffer.strided.basePtr));

    const int32_t height   = static_cast<int32_t>(refData.shape[0]);
    const int32_t width    = static_cast<int32_t>(refData.shape[1]);
    const int32_t refPitch = static_cast<int32_t>(refData.buffer.strided.strides[0] / sizeof(int32_t));
    const int32_t pvaPitch = static_cast<int32_t>(pvaData.buffer.strided.strides[0] / sizeof(int32_t));

    int32_t mismatches = 0;
    for (int32_t i = 0; i < height; i++)
    {
        for (int32_t j = 0; j < width; j++)
        {
            const int32_t refValue = ref[(i * refPitch) + j];
            const int32_t pvaValue = pva[(i * pvaPitch) + j];
            if (refValue != pvaValue)
            {
                if (mismatches < 10)
                {
                    printf("Mismatch[%d, %d]: ref = %d, pva = %d\n", i, j, refValue, pvaValue);
                }
                mismatches++;
            }
        }
    }

    if (mismatches != 0)
    {
        printf("Total mismatches: %d out of %d elements\n", mismatches, width * height);
    }

    return mismatches;
}
