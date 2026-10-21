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

#include "box_filter_ref.h"

#include <ImageIO.h>
#include <OpBoxFilter.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#define QBITS_8 8
#define QBITS_16 16

// Function to load box filter input data from a file
int32_t loadBoxFilterInput(NVCVTensorHandle tensorHandle, const char *directory, const char *filename,
                           const int64_t size)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    uint8_t *image_h = NULL;
    CupvaMemGetHostPointer((void **)&image_h, (void *)tensorData.buffer.strided.basePtr);
    // Use ReadImageBuffer with the explicitly provided size
    int32_t err = ReadImageBuffer(filename, directory, image_h, sizeof(uint8_t) * size);
    return err;
}

// Function to save box filter output data to a file
int32_t saveBoxFilterOutput(NVCVTensorHandle tensorHandle, const char *directory, const char *filename,
                            const int64_t size)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    uint8_t *image_h = NULL;
    CupvaMemGetHostPointer((void **)&image_h, (void *)tensorData.buffer.strided.basePtr);

    // Use WriteImageBuffer with the explicitly provided size
    int32_t err = WriteImageBuffer(filename, directory, image_h, sizeof(uint8_t) * size);
    return err;
}

namespace {
template<typename T, typename = std::enable_if_t<std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> ||
                                                 std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t>>>
void generateRandomInputImpl(NVCVTensorHandle tensorHandle)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    T *tensorPtr_h = NULL;
    CupvaMemGetHostPointer((void **)&tensorPtr_h, (void *)tensorData.buffer.strided.basePtr);

    int height    = tensorData.shape[0];
    int width     = tensorData.shape[1];
    int linePitch = tensorData.buffer.strided.strides[0] / sizeof(T);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if (std::is_signed<T>::value)
            {
                int32_t range = static_cast<int32_t>(std::numeric_limits<T>::max()) -
                                static_cast<int32_t>(std::numeric_limits<T>::min());
                tensorPtr_h[i * linePitch + j] = static_cast<T>(rand() % (range + 1) + std::numeric_limits<T>::min());
            }
            else
            {
                tensorPtr_h[i * linePitch + j] = static_cast<T>(rand() % (std::numeric_limits<T>::max() + 1));
            }
        }
    }
}

template<typename T, typename = std::enable_if_t<std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> ||
                                                 std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t>>>
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

template<typename T>
void Float2Fixed(T *dst, const float *src, const int32_t size, const int qbits)
{
    // normalize values to range [-scale,scale] to preserve dynamic range before clamping to avoid overflow.
    int64_t sum        = 0;
    float multiplier   = static_cast<float>(1U << qbits) / 1.0;
    int64_t upperBound = static_cast<int64_t>(std::numeric_limits<T>::max());
    int64_t lowerBound = static_cast<int64_t>(std::numeric_limits<T>::min());
    for (int32_t i = 0; i < size; ++i)
    {
        int64_t val = static_cast<int64_t>(src[i] * multiplier);
        val         = std::min(val, upperBound);
        val         = std::max(val, lowerBound);
        dst[i]      = static_cast<T>(val);
        sum += dst[i];
    }

    // For box filter, ensure sum equals (1 << qbits)
    int64_t expected_sum = static_cast<int64_t>(1U << qbits);
    int64_t diff         = expected_sum - sum;

    // Distribute the difference among the kernel elements
    if (diff != 0)
    {
        int32_t elements_per_adjust = size / std::abs(diff);
        if (elements_per_adjust == 0)
            elements_per_adjust = 1;

        for (int32_t i = 0; i < size && diff != 0; i += elements_per_adjust)
        {
            if (diff > 0)
            {
                dst[i] = std::min(static_cast<int64_t>(dst[i]) + 1, upperBound);
                diff--;
            }
            else
            {
                dst[i] = std::max(static_cast<int64_t>(dst[i]) - 1, lowerBound);
                diff++;
            }
        }
    }
}

template<typename T>
void get_2D_box_kernel(T *outKernel, uint32_t kernelSize, int qbits)
{
    uint32_t totalElements = kernelSize * kernelSize;
    std::vector<float> kernelFloat(totalElements);

    // Box filter: all elements are equal (1/total_elements)
    float kernelValue = 1.0f / static_cast<float>(totalElements);

    for (uint32_t i = 0; i < totalElements; ++i)
    {
        kernelFloat[i] = kernelValue;
    }

    Float2Fixed<T>(outKernel, kernelFloat.data(), totalElements, qbits);
}

template<typename DT, typename KT>
void convolution(DT *src_pad, KT *knl, DT *dst, int iw, int ih, int lp, int kw, int kh, int qbits)
{
    int kwr = kw >> 1;
    int khr = kh >> 1;
    int iwp = iw + kwr * 2;
    int ihp = ih + khr * 2;

    int rnd_add = (qbits == 0) ? 0 : (1 << (qbits - 1));
    for (int i = 0; i < ih; i++)
    {
        for (int j = 0; j < iw; j++)
        {
            long long t1 = 0;
            for (int k = 0; k < kh; k++)
            {
                for (int l = 0; l < kw; l++)
                {
                    DT src_pix = src_pad[(i + k) * iwp + (j + l)];
                    KT knl_pix = knl[k * kw + l];
                    t1 += (long long)src_pix * knl_pix;
                }
            }
            int dst_pix     = (t1 + rnd_add) >> qbits;
            dst_pix         = std::min(dst_pix, static_cast<int>(std::numeric_limits<DT>::max()));
            dst_pix         = std::max(dst_pix, static_cast<int>(std::numeric_limits<DT>::min()));
            dst[i * lp + j] = dst_pix;
        }
    }
}

template<typename T, typename = std::enable_if_t<std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> ||
                                                 std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t>>>
void boxFilterRefImpl(NVCVTensorHandle inTensorHandle, NVCVTensorHandle outTensorHandle, int32_t kernelSize,
                      const NVCVBorderType borderMode, int32_t borderValue)
{
    int32_t kw  = kernelSize;
    int32_t kh  = kernelSize;
    int32_t kwr = kw >> 1;
    int32_t khr = kh >> 1;

    NVCVTensorData inTensorData;
    nvcvTensorExportData(inTensorHandle, &inTensorData);
    T *inTensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&inTensorPtr, (void *)inTensorData.buffer.strided.basePtr);

    NVCVTensorData outTensorData;
    nvcvTensorExportData(outTensorHandle, &outTensorData);
    T *outTensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&outTensorPtr, (void *)outTensorData.buffer.strided.basePtr);

    int32_t height = inTensorData.shape[0];
    int32_t width  = inTensorData.shape[1];
    int32_t lp     = inTensorData.buffer.strided.strides[0] / sizeof(T);
    int32_t iwp    = width + kwr * 2;
    int32_t ihp    = height + khr * 2;

    std::vector<T> src_pad(iwp * ihp, borderValue);
    T *src_pad_ofst = src_pad.data() + khr * iwp + kwr;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            src_pad_ofst[y * iwp + x] = inTensorPtr[y * lp + x];
        }
    }

    if (borderMode == NVCV_BORDER_REPLICATE)
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < kwr; x++)
            {
                // Replicate left border
                src_pad[(y + khr) * iwp + x] = inTensorPtr[y * lp];
                // Replicate right border
                src_pad[(y + khr) * iwp + iwp - 1 - x] = inTensorPtr[y * lp + width - 1];
            }
        }

        for (int y = 0; y < khr; y++)
        {
            // Replicate top border
            memcpy(&src_pad[y * iwp], &src_pad[khr * iwp], iwp * sizeof(T));
            // Replicate bottom border
            memcpy(&src_pad[(height + 2 * khr - 1 - y) * iwp], &src_pad[(height + 2 * khr - 1 - khr) * iwp],
                   iwp * sizeof(T));
        }
    }

    using KT = std::conditional_t<sizeof(T) == 1, uint8_t, uint16_t>;
    std::vector<KT> knl_ref(kw * kh);
    int qbits = sizeof(T) == 1 ? QBITS_8 : QBITS_16;
    get_2D_box_kernel(knl_ref.data(), kernelSize, qbits);
    convolution(src_pad.data(), knl_ref.data(), outTensorPtr, width, height, lp, kw, kh, qbits);
}
} // namespace

void generateRandomBoxFilterInput(NVCVTensorHandle tensorHandle)
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

int32_t checkBoxFilterOutput(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle)
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

void boxFilterRef(NVCVTensorHandle inTensorHandle, NVCVTensorHandle outTensorHandle, int32_t kernelSize,
                  const NVCVBorderType borderMode, int32_t borderValue)
{
    NVCVTensorData inTensorData;
    nvcvTensorExportData(inTensorHandle, &inTensorData);

    switch (inTensorData.dtype)
    {
    case NVCV_DATA_TYPE_U8:
        boxFilterRefImpl<uint8_t>(inTensorHandle, outTensorHandle, kernelSize, borderMode, borderValue);
        break;
    case NVCV_DATA_TYPE_S8:
        boxFilterRefImpl<int8_t>(inTensorHandle, outTensorHandle, kernelSize, borderMode, borderValue);
        break;
    case NVCV_DATA_TYPE_U16:
        boxFilterRefImpl<uint16_t>(inTensorHandle, outTensorHandle, kernelSize, borderMode, borderValue);
        break;
    case NVCV_DATA_TYPE_S16:
        boxFilterRefImpl<int16_t>(inTensorHandle, outTensorHandle, kernelSize, borderMode, borderValue);
        break;
    default:
        break;
    }
}
