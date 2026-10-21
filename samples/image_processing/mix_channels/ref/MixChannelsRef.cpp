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

#include "MixChannelsRef.h"

#include <OpMixChannels.h>

#include <cstdint>
#include <cstdlib>
#include <vector>

namespace {
template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value || std::is_same<T, float>::value>>
void generateTensorInput(NVCVTensorHandle inHandle)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(inHandle, &tensorData);

    T *tensorPtr_h = NULL;
    CupvaMemGetHostPointer((void **)&tensorPtr_h, (void *)tensorData.buffer.strided.basePtr);

    int height    = tensorData.shape[0];
    int width     = tensorData.shape[1];
    int channel   = tensorData.shape[2];
    int linePitch = tensorData.buffer.strided.strides[0] / sizeof(T);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            for (int k = 0; k < channel; k++)
            {
                tensorPtr_h[i * linePitch + j * channel + k] = rand() % (1UL << (sizeof(T) * 8));
            }
        }
    }
}

template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value || std::is_same<T, float>::value>>
int32_t checkTensorOutput(NVCVTensorHandle refHandle, NVCVTensorHandle pvaHandle)
{
    NVCVTensorData refTensorData;
    nvcvTensorExportData(refHandle, &refTensorData);

    T *ref = NULL;
    CupvaMemGetHostPointer((void **)&ref, (void *)refTensorData.buffer.strided.basePtr);

    NVCVTensorData pvaTensorData;
    nvcvTensorExportData(pvaHandle, &pvaTensorData);

    T *pva = NULL;
    CupvaMemGetHostPointer((void **)&pva, (void *)pvaTensorData.buffer.strided.basePtr);

    int32_t height    = refTensorData.shape[0];
    int32_t width     = refTensorData.shape[1];
    int32_t channel   = refTensorData.shape[2];
    int32_t linePitch = refTensorData.buffer.strided.strides[0] / sizeof(T);

    for (int h = 0; h < height; ++h)
    {
        for (int w = 0; w < width; ++w)
        {
            for (int c = 0; c < channel; ++c)
            {
                int index = h * linePitch + w * channel + c;
                T refData = ref[index];
                T cmpData = pva[index];
                if (refData != cmpData)
                {
                    if constexpr (std::is_same<T, uint8_t>::value)
                    {
                        printf("Mismatch[%d, %d, %d]: ref = %u, pva = %u\n", h, w, c, refData, cmpData);
                    }
                    return 1;
                }
            }
        }
    }

    return 0;
}
} // namespace

void generateRandomMixChannelsInput(NVCVTensorHandle *inHandles, int inTensorCount)
{
    for (int i = 0; i < inTensorCount; ++i)
    {
        NVCVTensorData tensorData;
        nvcvTensorExportData(inHandles[i], &tensorData);

        switch (tensorData.dtype)
        {
        case NVCV_DATA_TYPE_U8:
            generateTensorInput<uint8_t>(inHandles[i]);
            break;
        case NVCV_DATA_TYPE_F32:
            generateTensorInput<float>(inHandles[i]);
            break;
        default:
            break;
        }
    }
}

int32_t checkMixChannelsOutput(NVCVTensorHandle *refHandles, NVCVTensorHandle *pvaHandles, int outTensorCount)
{
    for (int i = 0; i < outTensorCount; ++i)
    {
        NVCVTensorData refTensorData;
        nvcvTensorExportData(refHandles[i], &refTensorData);

        int err = 0;
        switch (refTensorData.dtype)
        {
        case NVCV_DATA_TYPE_U8:
            err = checkTensorOutput<uint8_t>(refHandles[i], pvaHandles[i]);
            break;
        case NVCV_DATA_TYPE_F32:
            err = checkTensorOutput<float>(refHandles[i], pvaHandles[i]);
            break;
        default:
            return -1;
        }

        if (err != 0)
        {
            return err;
        }
    }

    return 0;
}

void mixChannelsRef(NVCVTensorHandle *inHandles, NVCVTensorHandle *outHandles, int inTensorCount, int outTensorCount,
                    PVAMixChannelsCode code)
{
    uint8_t *inTensorPtrs[inTensorCount];
    uint8_t *outTensorPtrs[outTensorCount];

    for (int i = 0; i < inTensorCount; ++i)
    {
        inTensorPtrs[i] = NULL;
        NVCVTensorData inTensorData;
        nvcvTensorExportData(inHandles[i], &inTensorData);
        CupvaMemGetHostPointer((void **)&(inTensorPtrs[i]), (void *)inTensorData.buffer.strided.basePtr);
    }

    for (int i = 0; i < outTensorCount; ++i)
    {
        outTensorPtrs[i] = NULL;
        NVCVTensorData outTensorData;
        nvcvTensorExportData(outHandles[i], &outTensorData);
        CupvaMemGetHostPointer((void **)&(outTensorPtrs[i]), (void *)outTensorData.buffer.strided.basePtr);
    }

    switch (code)
    {
    case PVA_SPLIT_RGBA8_TO_U8:
    {
        NVCVTensorData inTensorData;
        nvcvTensorExportData(inHandles[0], &inTensorData);
        NVCVTensorData outTensorData;
        nvcvTensorExportData(outHandles[0], &outTensorData);

        int32_t height          = inTensorData.shape[0];
        int32_t width           = inTensorData.shape[1];
        int32_t channel         = inTensorData.shape[2];
        int32_t inputLinePitch  = inTensorData.buffer.strided.strides[0] / sizeof(inTensorPtrs[0][0]);
        int32_t outputLinePitch = outTensorData.buffer.strided.strides[0] / sizeof(outTensorPtrs[0][0]);

        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                for (int c = 0; c < channel; ++c)
                {
                    int out_index               = h * outputLinePitch + w;
                    int in_index                = h * inputLinePitch + w * channel + c;
                    outTensorPtrs[c][out_index] = inTensorPtrs[0][in_index];
                }
            }
        }

        break;
    }
    case PVA_MERGE_U8_TO_RGBA8:
    {
        NVCVTensorData inTensorData;
        nvcvTensorExportData(inHandles[0], &inTensorData);
        NVCVTensorData outTensorData;
        nvcvTensorExportData(outHandles[0], &outTensorData);

        int32_t height          = outTensorData.shape[0];
        int32_t width           = outTensorData.shape[1];
        int32_t channel         = outTensorData.shape[2];
        int32_t inputLinePitch  = inTensorData.buffer.strided.strides[0] / sizeof(inTensorPtrs[0][0]);
        int32_t outputLinePitch = outTensorData.buffer.strided.strides[0] / sizeof(outTensorPtrs[0][0]);

        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                for (int c = 0; c < channel; ++c)
                {
                    int in_index                = h * inputLinePitch + w;
                    int out_index               = h * outputLinePitch + w * channel + c;
                    outTensorPtrs[0][out_index] = inTensorPtrs[c][in_index];
                }
            }
        }

        break;
    }
    }
}
