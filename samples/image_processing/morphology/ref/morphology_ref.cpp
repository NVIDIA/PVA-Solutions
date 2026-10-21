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

#include "morphology_ref.h"

#include <OpMorphology.h>

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

namespace {
template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value || std::is_same<T, uint16_t>::value ||
                                                 std::is_same<T, int8_t>::value || std::is_same<T, int16_t>::value>>
void generateRandomInputImpl(NVCVTensorHandle tensorHandle)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    T *tensorPtr_h = NULL;
    CupvaMemGetHostPointer((void **)&tensorPtr_h, (void *)tensorData.buffer.strided.basePtr);

    int height    = tensorData.shape[0];
    int width     = tensorData.shape[1];
    int linePitch = tensorData.buffer.strided.strides[0] / sizeof(T);

    // use random generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<T> dist(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            tensorPtr_h[i * linePitch + j] = dist(gen);
        }
    }
}

template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value || std::is_same<T, uint16_t>::value ||
                                                 std::is_same<T, int8_t>::value || std::is_same<T, int16_t>::value>>
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

    int32_t height = refTensorData.shape[0];
    int32_t width  = refTensorData.shape[1];
    int linePitch  = refTensorData.buffer.strided.strides[0] / sizeof(T);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            T refData = ref[y * linePitch + x];
            T cmpData = pva[y * linePitch + x];
            if (refData != cmpData)
            {
                printf("Mismatch[%d, %d]: ref = %u, pva = %u\n", y, x, refData, cmpData);
                return 1;
            }
        }
    }
    return 0;
}

template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value || std::is_same<T, uint16_t>::value ||
                                                 std::is_same<T, int8_t>::value || std::is_same<T, int16_t>::value>>
void morphologyRefImpl(NVCVTensorHandle inTensorHandle, NVCVTensorHandle outTensorHandle, PVAMorphologyType morphType,
                       PvaMorphologyMaskParams *maskParams, const NVCVBorderType borderMode, int32_t borderValue)
{
    NVCVTensorData inTensorData;
    nvcvTensorExportData(inTensorHandle, &inTensorData);
    T *inTensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&inTensorPtr, (void *)inTensorData.buffer.strided.basePtr);

    NVCVTensorData outTensorData;
    nvcvTensorExportData(outTensorHandle, &outTensorData);
    T *outTensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&outTensorPtr, (void *)outTensorData.buffer.strided.basePtr);

    int32_t height    = inTensorData.shape[0];
    int32_t width     = inTensorData.shape[1];
    int32_t linePitch = inTensorData.buffer.strided.strides[0] / sizeof(T);

    int32_t knlWidth  = maskParams->maskWidth;
    int32_t knlHeight = maskParams->maskHeight;

    int32_t knlWidthRad  = knlWidth / 2;
    int32_t knlHeightRad = knlHeight / 2;

    int32_t inPadLinePitch = width + knlWidthRad * 2;
    std::vector<T> inPadTensor(inPadLinePitch * (height + knlHeightRad * 2), borderValue);
    T *inPadTensorOfstPtr = inPadTensor.data() + knlHeightRad * inPadLinePitch + knlWidthRad;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            inPadTensorOfstPtr[y * inPadLinePitch + x] = inTensorPtr[y * linePitch + x];
        }
    }
    if (borderMode == NVCV_BORDER_REPLICATE)
    {
        // Replicate left and right borders
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < knlWidthRad; x++)
            {
                // Replicate left border
                inPadTensor[(y + knlHeightRad) * inPadLinePitch + x] =
                    inPadTensor[(y + knlHeightRad) * inPadLinePitch + knlWidthRad];
                // Replicate right border
                inPadTensor[(y + knlHeightRad) * inPadLinePitch + inPadLinePitch - 1 - x] =
                    inPadTensor[(y + knlHeightRad) * inPadLinePitch + inPadLinePitch - 1 - knlWidthRad];
            }
        }

        // Replicate top and bottom borders
        for (int y = 0; y < knlHeightRad; y++)
        {
            // Replicate top border
            memcpy(&inPadTensor[y * inPadLinePitch], &inPadTensor[knlHeightRad * inPadLinePitch],
                   inPadLinePitch * sizeof(T));
            // Replicate bottom border
            memcpy(&inPadTensor[(height + 2 * knlHeightRad - 1 - y) * inPadLinePitch],
                   &inPadTensor[(height + 2 * knlHeightRad - 1 - knlHeightRad) * inPadLinePitch],
                   inPadLinePitch * sizeof(T));
        }
    }

    uint8_t knl[knlWidth * knlHeight];
    if (maskParams->maskShape == CROSS_MASK && knlWidth == 3)
    {
        uint8_t crossShape[3][3] = {{0, 1, 0}, {1, 1, 1}, {0, 1, 0}};
        for (int32_t j = 0; j < knlHeight; j++)
        {
            for (int32_t i = 0; i < knlWidth; i++)
            {
                knl[j * knlWidth + i] = crossShape[j][i];
            }
        }
    }
    else
    {
        for (int32_t i = 0; i < knlWidth * knlHeight; i++)
        {
            knl[i] = 1;
        }
    }
    if (morphType == PVA_ERODE)
    {
        for (int32_t y = 0; y < height; y++)
        {
            for (int32_t x = 0; x < width; x++)
            {
                T dstData = std::numeric_limits<T>::max();
                for (int32_t j = 0; j < knlHeight; j++)
                {
                    for (int32_t i = 0; i < knlWidth; i++)
                    {
                        T srcData = inPadTensor[(y + j) * inPadLinePitch + (x + i)];
                        T knlData = knl[j * knlWidth + i];
                        if (knlData)
                        {
                            dstData = (srcData < dstData) ? srcData : dstData;
                        }
                    }
                }
                outTensorPtr[y * linePitch + x] = dstData;
            }
        }
    }
    else
    {
        for (int32_t y = 0; y < height; y++)
        {
            for (int32_t x = 0; x < width; x++)
            {
                T dstData = std::numeric_limits<T>::min();
                for (int32_t j = 0; j < knlHeight; j++)
                {
                    for (int32_t i = 0; i < knlWidth; i++)
                    {
                        T srcData = inPadTensor[(y + j) * inPadLinePitch + (x + i)];
                        T knlData = knl[j * knlWidth + i];
                        if (knlData)
                        {
                            dstData = (srcData > dstData) ? srcData : dstData;
                        }
                    }
                }
                outTensorPtr[y * linePitch + x] = dstData;
            }
        }
    }
}
} // namespace

void generateRandomMorphologyInput(NVCVTensorHandle tensorHandle)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    switch (tensorData.dtype)
    {
    case NVCV_DATA_TYPE_U8:
        generateRandomInputImpl<uint8_t>(tensorHandle);
        break;
    case NVCV_DATA_TYPE_S8:
        generateRandomInputImpl<int8_t>(tensorHandle);
        break;
    case NVCV_DATA_TYPE_U16:
        generateRandomInputImpl<uint16_t>(tensorHandle);
        break;
    case NVCV_DATA_TYPE_S16:
        generateRandomInputImpl<int16_t>(tensorHandle);
        break;
    default:
        break;
    }
}

int32_t checkMorphologyOutput(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle)
{
    NVCVTensorData refTensorData;
    nvcvTensorExportData(refTensorHandle, &refTensorData);

    switch (refTensorData.dtype)
    {
    case NVCV_DATA_TYPE_U8:
        return checkOutputImpl<uint8_t>(refTensorHandle, pvaTensorHandle);
    case NVCV_DATA_TYPE_S8:
        return checkOutputImpl<int8_t>(refTensorHandle, pvaTensorHandle);
    case NVCV_DATA_TYPE_U16:
        return checkOutputImpl<uint16_t>(refTensorHandle, pvaTensorHandle);
    case NVCV_DATA_TYPE_S16:
        return checkOutputImpl<int16_t>(refTensorHandle, pvaTensorHandle);
    default:
        return -1;
    }
}

void morphologyRef(NVCVTensorHandle inTensorHandle, NVCVTensorHandle outTensorHandle, PVAMorphologyType morphType,
                   PvaMorphologyMaskParams *maskParams, const NVCVBorderType borderMode, int32_t borderValue)
{
    NVCVTensorData inTensorData;
    nvcvTensorExportData(inTensorHandle, &inTensorData);

    switch (inTensorData.dtype)
    {
    case NVCV_DATA_TYPE_U8:
        morphologyRefImpl<uint8_t>(inTensorHandle, outTensorHandle, morphType, maskParams, borderMode, borderValue);
        break;
    case NVCV_DATA_TYPE_S8:
        morphologyRefImpl<int8_t>(inTensorHandle, outTensorHandle, morphType, maskParams, borderMode, borderValue);
        break;
    case NVCV_DATA_TYPE_U16:
        morphologyRefImpl<uint16_t>(inTensorHandle, outTensorHandle, morphType, maskParams, borderMode, borderValue);
        break;
    case NVCV_DATA_TYPE_S16:
        morphologyRefImpl<int16_t>(inTensorHandle, outTensorHandle, morphType, maskParams, borderMode, borderValue);
        break;
    default:
        break;
    }
}