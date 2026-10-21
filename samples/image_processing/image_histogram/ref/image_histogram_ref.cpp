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

#include "image_histogram_ref.h"

#include <cstdint>
#include <cstdlib>
#include <vector>

template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value || std::is_same<T, uint16_t>::value>>
void generateInputImpl(NVCVTensorHandle tensorHandle)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    T *tensorPtr_h = NULL;
    CupvaMemGetHostPointer((void **)&tensorPtr_h, (void *)tensorData.buffer.strided.basePtr);

    int32_t height    = tensorData.shape[0];
    int32_t width     = tensorData.shape[1];
    int32_t linePitch = tensorData.buffer.strided.strides[0] / sizeof(T);

    // generate all kinds of pixel value in input data
    for (int32_t j = 0; j < height; j++)
    {
        for (int32_t i = 0; i < width; i++)
        {
            tensorPtr_h[j * linePitch + i] = static_cast<T>(j * width + i);
        }
    }
}

template<typename T, typename = std::enable_if_t<std::is_same<T, uint32_t>::value>>
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

    int32_t height    = refTensorData.shape[0];
    int32_t width     = refTensorData.shape[1];
    int32_t linePitch = refTensorData.buffer.strided.strides[0] / sizeof(T);

    for (int32_t y = 0; y < height; y++)
    {
        for (int32_t x = 0; x < width; x++)
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

template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value || std::is_same<T, uint16_t>::value>>
void imageHistogramRefImpl(NVCVTensorHandle inTensorHandle, NVCVTensorHandle outTensorHandle,
                           PvaImageHistogramParams *imageHistogramParams)
{
    NVCVTensorData inTensorData;
    nvcvTensorExportData(inTensorHandle, &inTensorData);
    T *inTensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&inTensorPtr, (void *)inTensorData.buffer.strided.basePtr);

    NVCVTensorData outTensorData;
    nvcvTensorExportData(outTensorHandle, &outTensorData);
    uint32_t *outTensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&outTensorPtr, (void *)outTensorData.buffer.strided.basePtr);

    int32_t height    = inTensorData.shape[0];
    int32_t width     = inTensorData.shape[1];
    int32_t linePitch = inTensorData.buffer.strided.strides[0] / sizeof(T);

    float start     = imageHistogramParams->start;
    float end       = imageHistogramParams->end;
    int32_t numBins = imageHistogramParams->numBins;

    float range = end - start;

    float a = static_cast<float>(numBins) / range;
    float b = -start * a;

    memset(outTensorPtr, 0, numBins * sizeof(uint32_t));

    for (int32_t j = 0; j < height; ++j)
    {
        for (int32_t i = 0; i < width; ++i)
        {
            T val = inTensorPtr[j * linePitch + i];
            if (val >= start && val < end)
            {
                int32_t idx = static_cast<int32_t>(static_cast<float>(val) * a + b);
                outTensorPtr[idx]++;
            }
        }
    }
}

void generateImageHistogramInput(NVCVTensorHandle tensorHandle)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    switch (tensorData.dtype)
    {
    case NVCV_DATA_TYPE_U8:
        generateInputImpl<uint8_t>(tensorHandle);
        break;
    case NVCV_DATA_TYPE_U16:
        generateInputImpl<uint16_t>(tensorHandle);
        break;
    default:
        break;
    }
}

int32_t checkImageHistogramOutput(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle)
{
    NVCVTensorData refTensorData;
    nvcvTensorExportData(refTensorHandle, &refTensorData);

    switch (refTensorData.dtype)
    {
    case NVCV_DATA_TYPE_U32:
        return checkOutputImpl<uint32_t>(refTensorHandle, pvaTensorHandle);
    default:
        return -1;
    }
}

void imageHistogramRef(NVCVTensorHandle inTensorHandle, NVCVTensorHandle outTensorHandle,
                       PvaImageHistogramParams *imageHistogramParams)
{
    NVCVTensorData inTensorData;
    nvcvTensorExportData(inTensorHandle, &inTensorData);

    switch (inTensorData.dtype)
    {
    case NVCV_DATA_TYPE_U8:
        imageHistogramRefImpl<uint8_t>(inTensorHandle, outTensorHandle, imageHistogramParams);
        break;
    case NVCV_DATA_TYPE_U16:
        imageHistogramRefImpl<uint16_t>(inTensorHandle, outTensorHandle, imageHistogramParams);
        break;
    default:
        break;
    }
}