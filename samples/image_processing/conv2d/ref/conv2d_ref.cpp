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

#include "conv2d_ref.h"

#include <ImageIO.h>
#include <OpGaussianFilter.h>
#include <PvaAllocator.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>

#define QBITS_8 8
#define QBITS_16 16

// Function to load gaussian filter input data from a file
int32_t loadConv2dInput(NVCVTensorHandle tensorHandle, const char *directory, const char *filename, const int64_t size)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    uint8_t *image_h = NULL;
    CupvaMemGetHostPointer((void **)&image_h, (void *)tensorData.buffer.strided.basePtr);
    // Use ReadImageBuffer with the explicitly provided size
    int32_t err = ReadImageBuffer(filename, directory, image_h, sizeof(uint8_t) * size);
    return err;
}

// Function to save gaussian filter output data to a file
int32_t saveConv2dOutput(NVCVTensorHandle tensorHandle, const char *directory, const char *filename, const int64_t size)
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

    // Initialize C++ random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if constexpr (std::is_signed<T>::value)
            {
                std::uniform_int_distribution<T> dist(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
                tensorPtr_h[i * linePitch + j] = dist(gen);
            }
            else
            {
                std::uniform_int_distribution<T> dist(0, std::numeric_limits<T>::max());
                tensorPtr_h[i * linePitch + j] = dist(gen);
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
    // Scale all coefficients proportionally to achieve target sum while preserving relative weights
    int64_t target_sum = 1 << qbits;
    float multiplier   = static_cast<float>(1U << qbits) / 1.0;
    int64_t upperBound = static_cast<int64_t>(std::numeric_limits<T>::max());
    int64_t lowerBound = static_cast<int64_t>(std::numeric_limits<T>::min());

    // First pass: calculate the sum after scaling and clamping
    int64_t sum = 0;
    for (int32_t i = 0; i < size; ++i)
    {
        int64_t val = static_cast<int64_t>(src[i] * multiplier);
        val         = std::min(val, upperBound);
        val         = std::max(val, lowerBound);
        dst[i]      = static_cast<T>(val);
        sum += dst[i];
    }

    if (sum == 0) // only center coef is not zero
        throw std::runtime_error("sigma value too small");

    // Scale all coefficients proportionally to achieve target sum
    if (sum != target_sum)
    {
        float scale_factor = static_cast<float>(target_sum) / static_cast<float>(sum);
        for (int32_t i = 0; i < size; ++i)
        {
            int64_t val = static_cast<int64_t>(dst[i] * scale_factor);
            val         = std::min(val, upperBound);
            val         = std::max(val, lowerBound);
            dst[i]      = static_cast<T>(val);
        }

        // Recalculate sum after scaling and clamping
        sum = 0;
        for (int32_t i = 0; i < size; ++i)
        {
            sum += dst[i];
        }

        // Final adjustment to ensure exact target sum
        if (sum != target_sum)
        {
            int64_t diff = target_sum - sum;
            // Distribute the difference to minimize impact on relative weights
            // Start with the largest coefficient to minimize relative change
            std::vector<std::pair<int64_t, int32_t>> sorted_coeffs;
            for (int32_t i = 0; i < size; ++i)
            {
                sorted_coeffs.emplace_back(dst[i], i);
            }
            std::sort(sorted_coeffs.begin(), sorted_coeffs.end(),
                      [](const auto &a, const auto &b) { return a.first > b.first; });

            // Apply the difference to the largest coefficient(s)
            int32_t idx = 0;
            while (diff != 0 && idx < size)
            {
                int32_t i           = sorted_coeffs[idx].second;
                int64_t current_val = dst[i];
                int64_t new_val     = current_val + diff;

                if (new_val >= lowerBound && new_val <= upperBound)
                {
                    dst[i] = static_cast<T>(new_val);
                    diff   = 0;
                }
                else if (diff > 0)
                {
                    // Try to add as much as possible
                    int64_t max_add = upperBound - current_val;
                    if (max_add > 0)
                    {
                        int64_t add_amount = std::min(diff, max_add);
                        dst[i]             = static_cast<T>(current_val + add_amount);
                        diff -= add_amount;
                    }
                }
                else
                {
                    // Try to subtract as much as possible
                    int64_t max_sub = current_val - lowerBound;
                    if (max_sub > 0)
                    {
                        int64_t sub_amount = std::min(-diff, max_sub);
                        dst[i]             = static_cast<T>(current_val - sub_amount);
                        diff += sub_amount;
                    }
                }
                idx++;
            }
        }
    }

    // Final verification: ensure coefficients sum to exactly target_sum
    int64_t final_sum = 0;
    for (int32_t i = 0; i < size; ++i)
    {
        final_sum += dst[i];
    }
    assert(final_sum == target_sum && "Float2Fixed: Final coefficients sum does not equal target sum");
}

template<typename T>
void get_conv2d_kernel(T *outKernel, uint32_t kernelSize, int qbits)
{
    // Deterministic random number generator
    std::mt19937 rng(42); // fixed seed for determinism
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    uint32_t kernelElems = kernelSize * kernelSize;
    std::vector<float> kernelFloat(kernelElems);

    // Fill with random values
    for (uint32_t i = 0; i < kernelElems; ++i)
    {
        kernelFloat[i] = dist(rng);
    }

    // Normalize so sum is 1.0 (to mimic a typical conv kernel)
    float sum = std::accumulate(kernelFloat.begin(), kernelFloat.end(), 0.0f);
    if (sum == 0.0f)
        sum = 1.0f;
    for (uint32_t i = 0; i < kernelElems; ++i)
    {
        kernelFloat[i] /= sum;
    }

    // For PVA, coefficients must be strictly < 1.0
    for (uint32_t i = 0; i < kernelElems; ++i)
    {
        if (kernelFloat[i] >= 1.0f)
            kernelFloat[i] = 0.99999f;
    }

    // For float type, just copy the values directly
    if constexpr (std::is_same_v<T, float>)
    {
        memcpy(outKernel, kernelFloat.data(), kernelElems * sizeof(float));
    }
    else
    {
        // For other types, apply quantization
        Float2Fixed<T>(outKernel, kernelFloat.data(), kernelElems, qbits);
    }
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
void conv2dRefWithKernelImpl(NVCVTensorHandle inTensorHandle, NVCVTensorHandle outTensorHandle,
                             NVCVTensorHandle kernelTensorHandle, const NVCVBorderType borderMode, int32_t borderValue)
{
    int32_t kw  = 0;
    int32_t kh  = 0;
    int32_t kwr = 0;
    int32_t khr = 0;

    // Get kernel dimensions from the kernel tensor
    NVCVTensorData kernelData;
    nvcvTensorExportData(kernelTensorHandle, &kernelData);
    kh  = kernelData.shape[0];
    kw  = kernelData.shape[1];
    kwr = kw >> 1;
    khr = kh >> 1;

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

    T *src_pad      = new T[iwp * ihp];
    T *src_pad_ofst = src_pad + khr * iwp + kwr;
    for (int i = 0; i < ihp * iwp; ++i) src_pad[i] = borderValue;

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

    using KT    = std::conditional_t<sizeof(T) == 1, uint8_t, uint16_t>;
    KT *knl_ref = new KT[kw * kh];

    // Copy kernel data directly since it's already in fixed-point format
    KT *kernelPtr = NULL;
    CupvaMemGetHostPointer((void **)&kernelPtr, (void *)kernelData.buffer.strided.basePtr);
    memcpy(knl_ref, kernelPtr, kw * kh * sizeof(KT));

    int qbits = sizeof(T) == 1 ? QBITS_8 : QBITS_16;
    convolution(src_pad, knl_ref, outTensorPtr, width, height, lp, kw, kh, qbits);
    delete[] src_pad;
    delete[] knl_ref;
}
} // namespace

void generateRandomConv2dInput(NVCVTensorHandle tensorHandle)
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

int32_t checkConv2dOutput(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle)
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

extern "C" NVCVTensorHandle createConv2dKernel(NVCVAllocatorHandle allocatorHandle, int32_t kernelSize,
                                               NVCVDataType dataType)
{
    NVCVTensorHandle kernelHandle = NULL;
    NVCVTensorLayout kernelLayout;
    int32_t err = 0;

    // Create kernel tensor layout - kernel should be 3D tensor with HWC layout
    err = nvcvTensorLayoutMake("HWC", &kernelLayout);
    if (err != NVCV_SUCCESS)
    {
        return NULL;
    }

    // Create kernel tensor requirements based on dataType
    int64_t kernelShape[] = {kernelSize, kernelSize, 1}; // HWC layout for 3D tensor with 1 channel
    NVCVTensorRequirements kernelRequirements;
    err = nvcvTensorCalcRequirementsPva(3, kernelShape, dataType, kernelLayout, 0, 0, &kernelRequirements);
    if (err != NVCV_SUCCESS)
    {
        return NULL;
    }

    // Construct kernel tensor
    err = nvcvTensorConstruct(&kernelRequirements, allocatorHandle, &kernelHandle);
    if (err != NVCV_SUCCESS)
    {
        return NULL;
    }

    // Generate kernel coefficients in the specified data type
    NVCVTensorData kernelData;
    err = nvcvTensorExportData(kernelHandle, &kernelData);
    if (err != NVCV_SUCCESS)
    {
        nvcvTensorDecRef(kernelHandle, NULL);
        return NULL;
    }

    // Generate kernel coefficients based on data type
    if (dataType == NVCV_DATA_TYPE_U8 || dataType == NVCV_DATA_TYPE_S8)
    {
        uint8_t *kernelPtr = NULL;
        CupvaMemGetHostPointer((void **)&kernelPtr, (void *)kernelData.buffer.strided.basePtr);
        get_conv2d_kernel<uint8_t>(kernelPtr, kernelSize, QBITS_8);
    }
    else if (dataType == NVCV_DATA_TYPE_U16 || dataType == NVCV_DATA_TYPE_S16)
    {
        uint16_t *kernelPtr = NULL;
        CupvaMemGetHostPointer((void **)&kernelPtr, (void *)kernelData.buffer.strided.basePtr);
        get_conv2d_kernel<uint16_t>(kernelPtr, kernelSize, QBITS_16);
    }
    else
    {
        // Unsupported data type - clean up and return error
        nvcvTensorDecRef(kernelHandle, NULL);
        return NULL;
    }

    return kernelHandle;
}

void conv2dRefWithKernel(NVCVTensorHandle inTensorHandle, NVCVTensorHandle outTensorHandle,
                         NVCVTensorHandle kernelTensorHandle, const NVCVBorderType borderMode, int32_t borderValue)
{
    NVCVTensorData inTensorData;
    nvcvTensorExportData(inTensorHandle, &inTensorData);

    switch (inTensorData.dtype)
    {
    case NVCV_DATA_TYPE_U8:
        conv2dRefWithKernelImpl<uint8_t>(inTensorHandle, outTensorHandle, kernelTensorHandle, borderMode, borderValue);
        break;
    case NVCV_DATA_TYPE_S8:
        conv2dRefWithKernelImpl<int8_t>(inTensorHandle, outTensorHandle, kernelTensorHandle, borderMode, borderValue);
        break;
    case NVCV_DATA_TYPE_U16:
        conv2dRefWithKernelImpl<uint16_t>(inTensorHandle, outTensorHandle, kernelTensorHandle, borderMode, borderValue);
        break;
    case NVCV_DATA_TYPE_S16:
        conv2dRefWithKernelImpl<int16_t>(inTensorHandle, outTensorHandle, kernelTensorHandle, borderMode, borderValue);
        break;
    default:
        break;
    }
}