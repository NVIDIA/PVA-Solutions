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

#include "gaussian_filter_ref.h"

#include <HostPointer.h>
#include <ImageIO.h>
#include <OpGaussianFilter.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <type_traits>

#define QBITS_8 8
#define QBITS_16 16

// Function to load gaussian filter input data from a file
int32_t loadGaussianFilterInput(NVCVTensorHandle tensorHandle, const char *directory, const char *filename,
                                const int64_t size)
{
    uint8_t *image_h = static_cast<uint8_t *>(PvaGetTensorHostPointer(tensorHandle));
    // Use ReadImageBuffer with the explicitly provided size
    int32_t err = ReadImageBuffer(filename, directory, image_h, sizeof(uint8_t) * size);
    return err;
}

// Function to save gaussian filter output data to a file
int32_t saveGaussianFilterOutput(NVCVTensorHandle tensorHandle, const char *directory, const char *filename,
                                 const int64_t size)
{
    uint8_t *image_h = static_cast<uint8_t *>(PvaGetTensorHostPointer(tensorHandle));

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

    T *tensorPtr_h = static_cast<T *>(PvaGetTensorHostPointer(tensorHandle));

    int height    = tensorData.shape[0];
    int width     = tensorData.shape[1];
    int linePitch = tensorData.buffer.strided.strides[0] / sizeof(T);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if constexpr (std::is_signed<T>::value)
            {
                tensorPtr_h[i * linePitch + j] =
                    static_cast<T>(rand() % (std::numeric_limits<T>::max() - std::numeric_limits<T>::min() + 1) +
                                   std::numeric_limits<T>::min());
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

    T *ref = static_cast<T *>(PvaGetTensorHostPointer(refTensorHandle));

    NVCVTensorData pvaTensorData;
    nvcvTensorExportData(pvaTensorHandle, &pvaTensorData);

    T *pva = static_cast<T *>(PvaGetTensorHostPointer(pvaTensorHandle));

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

    if (dst[size / 2] == sum) // only center coef is not zero
        throw std::runtime_error("sigma value too small");

    dst[size / 2] += (1 << qbits) - sum;
}

void get_1D_gaussian_kernel(float *outKernel, uint32_t kernelSize, float sigma)
{
    int len        = kernelSize / 2;
    const double s = 2.0 * sigma * sigma;
    double sum     = 0.0;

    for (int x = -len; x <= len; x++)
    {
        double r       = x * x;
        uint32_t ind   = x + len;
        outKernel[ind] = std::exp(-r / s) / std::sqrt(M_PI * s);
        sum += outKernel[ind];
    }

    for (uint32_t i = 0; i < kernelSize; i++)
    {
        outKernel[i] /= sum;
        // For PVA coefficients must be strictly < 1. So for contrived
        // cases, let's enforce that.
        if (outKernel[i] == 1)
        {
            outKernel[i] -= 1e-5;
        }
    }
}

template<typename T>
void get_2D_gaussian_kernel(T *outKernel, uint32_t kernelSize, float sigmaX, float sigmaY, int qbits)
{
    float *kernelX = new float[kernelSize];
    float *kernelY = new float[kernelSize];

    float *kernelFloat = new float[kernelSize * kernelSize];

    get_1D_gaussian_kernel(kernelX, kernelSize, sigmaX);
    get_1D_gaussian_kernel(kernelY, kernelSize, sigmaY);

    for (size_t i = 0; i < kernelSize; ++i)
    {
        for (size_t j = 0; j < kernelSize; ++j)
        {
            kernelFloat[i * kernelSize + j] = kernelY[i] * kernelX[j];
        }
    }

    Float2Fixed<T>(outKernel, kernelFloat, kernelSize * kernelSize, qbits);

    delete[] kernelX;
    delete[] kernelY;
    delete[] kernelFloat;
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
void gaussianFilterRefImpl(NVCVTensorHandle inTensorHandle, NVCVTensorHandle outTensorHandle, float sigmaX,
                           float sigmaY, int32_t kernelSize, const NVCVBorderType borderMode, int32_t borderValue)
{
    int32_t kw  = kernelSize;
    int32_t kh  = kernelSize;
    int32_t kwr = kw >> 1;
    int32_t khr = kh >> 1;

    NVCVTensorData inTensorData;
    nvcvTensorExportData(inTensorHandle, &inTensorData);
    T *inTensorPtr = static_cast<T *>(PvaGetTensorHostPointer(inTensorHandle));

    NVCVTensorData outTensorData;
    nvcvTensorExportData(outTensorHandle, &outTensorData);
    T *outTensorPtr = static_cast<T *>(PvaGetTensorHostPointer(outTensorHandle));

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
    int qbits   = sizeof(T) == 1 ? QBITS_8 : QBITS_16;
    get_2D_gaussian_kernel(knl_ref, kernelSize, sigmaX, sigmaY, qbits);
    convolution(src_pad, knl_ref, outTensorPtr, width, height, lp, kw, kh, qbits);
    delete[] src_pad;
    delete[] knl_ref;
}
} // namespace

void generateRandomGaussianFilterInput(NVCVTensorHandle tensorHandle)
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

int32_t checkGaussianFilterOutput(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle)
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

void gaussianFilterRef(NVCVTensorHandle inTensorHandle, NVCVTensorHandle outTensorHandle, float sigmaX, float sigmaY,
                       int32_t kernelSize, const NVCVBorderType borderMode, int32_t borderValue)
{
    NVCVTensorData inTensorData;
    nvcvTensorExportData(inTensorHandle, &inTensorData);

    switch (inTensorData.dtype)
    {
    case NVCV_DATA_TYPE_U8:
        gaussianFilterRefImpl<uint8_t>(inTensorHandle, outTensorHandle, sigmaX, sigmaY, kernelSize, borderMode,
                                       borderValue);
        break;
    case NVCV_DATA_TYPE_S8:
        gaussianFilterRefImpl<int8_t>(inTensorHandle, outTensorHandle, sigmaX, sigmaY, kernelSize, borderMode,
                                      borderValue);
        break;
    case NVCV_DATA_TYPE_U16:
        gaussianFilterRefImpl<uint16_t>(inTensorHandle, outTensorHandle, sigmaX, sigmaY, kernelSize, borderMode,
                                        borderValue);
        break;
    case NVCV_DATA_TYPE_S16:
        gaussianFilterRefImpl<int16_t>(inTensorHandle, outTensorHandle, sigmaX, sigmaY, kernelSize, borderMode,
                                       borderValue);
        break;
    default:
        break;
    }
}