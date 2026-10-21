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

#include "PvaAllocator.hpp"

#include <cupva_host.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/Status.hpp>
#include <nvcv/TensorLayout.hpp>

#include <nvcv/alloc/Requirements.h>

#include <numeric>

namespace pva_solutions::priv {

template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
constexpr bool IsPowerOfTwo(T value)
{
    return (value & (value - 1)) == 0;
}

template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
constexpr auto RoundUpNextPowerOfTwo(T x)
{
    assert(x >= 0);

    // Source: Hacker's Delight 1st ed, p.48,
    // adapted for any integer size.

    x = x - 1;
    // all constants, compiler can unroll it
    for (size_t i = 1; i < sizeof(T) * 8; i <<= 1)
    {
        x = x | (x >> i);
    }

    if constexpr (std::is_same_v<T, int32_t>)
    {
        return (int64_t)x + 1;
    }
    else if constexpr (std::is_same_v<T, int64_t>)
    {
        return (uint64_t)x + 1;
    }
    else
    {
        return x + 1;
    }
}

template<class T, class U, class = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<U>>>
constexpr auto RoundUpPowerOfTwo(T value, U multiple)
{
    assert(value >= 0);
    assert(multiple >= 0);

    assert(IsPowerOfTwo(multiple));

    // Source: Hacker's Delight 1st ed, p.45,

    return (value + (multiple - 1)) & -multiple;
}

NVCVTensorRequirements PvaCustomAllocator::CalcRequirements(int32_t rank, const int64_t *shape,
                                                            const nvcv::DataType &dtype, NVCVTensorLayout layout,
                                                            int32_t userBaseAlign, int32_t userRowAlign)
{
    NVCVTensorRequirements reqs;
    reqs.layout = layout;
    reqs.dtype  = static_cast<NVCVDataType>(dtype);

    if (layout.rank > 0 && rank != layout.rank)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Number of shape dimensions %d must be equal to layout dimensions %d", rank, layout.rank);
    }

    if (rank <= 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Number of dimensions must be >= 1, not %d", rank);
    }

    std::copy_n(shape, rank, reqs.shape);
    reqs.rank = rank;

    reqs.mem = {};

    // Calculate row pitch alignment
    int rowAlign;
    {
        if (userRowAlign == 0)
        {
            rowAlign = RoundUpNextPowerOfTwo(dtype.strideBytes());
        }
        else
        {
            if (!IsPowerOfTwo(userRowAlign))
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Invalid pitch alignment of %d, it must be a power-of-two", userRowAlign);
            }
            // must at least satisfy dtype alignment
            rowAlign = std::lcm(userRowAlign, dtype.alignment());
        }
    }

    // Calculate base address alignment
    {
        if (userBaseAlign == 0)
        {
            reqs.alignBytes = rowAlign;
            reqs.alignBytes = RoundUpNextPowerOfTwo(reqs.alignBytes);

            if (reqs.alignBytes > NVCV_MAX_MEM_REQUIREMENTS_BLOCK_SIZE)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Alignment requirement of %d is larger than the maximum allowed %ld",
                                      reqs.alignBytes, NVCV_MAX_MEM_REQUIREMENTS_BLOCK_SIZE);
            }
        }
        else
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Invalid base address alignment of %d, only a value of 0 is supported",
                                  userBaseAlign);
        }
    }

    NVCVTensorLayout layout_last;
    nvcvTensorLayoutMakeLast(reqs.layout, 2, &layout_last);
    int firstPacked = layout_last == NVCV_TENSOR_WC ? std::max(0, rank - 2) : rank - 1;

    reqs.strides[rank - 1] = dtype.strideBytes();
    for (int d = rank - 2; d >= 0; --d)
    {
        if (d == firstPacked - 1)
        {
            reqs.strides[d] = RoundUpPowerOfTwo(reqs.shape[d + 1] * reqs.strides[d + 1], rowAlign);
        }
        else
        {
            reqs.strides[d] = reqs.strides[d + 1] * reqs.shape[d + 1];
        }
    }

    nvcvMemRequirementsAddBuffer(&reqs.mem.cudaMem, reqs.strides[0] * reqs.shape[0], reqs.alignBytes);

    return reqs;
}

NVCVImageRequirements PvaCustomAllocator::CalcRequirements(int32_t width, int32_t height, NVCVImageFormat format,
                                                           int32_t userBaseAlign, int32_t userRowAlign)
{
    NVCVImageRequirements reqs;
    reqs.width  = width;
    reqs.height = height;
    reqs.format = format;
    reqs.mem    = {};

    int rowAlign;

    if (userRowAlign == 0)
    {
        rowAlign = 1;
    }
    else
    {
        if (!IsPowerOfTwo(userRowAlign))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Invalid pitch alignment of %d, it must be a power-of-two", userRowAlign);
        }
        rowAlign = userRowAlign;
    }

    // Pitch alignment must be compatible with each plane's pixel stride.
    int32_t numPlanes;
    nvcvImageFormatGetNumPlanes(format, &numPlanes);
    for (int i = 0; i < numPlanes; ++i)
    {
        int rowAlignPerPlane;
        if (userRowAlign == 0)
        {
            // Safest thing we can do
            int32_t bpp;
            nvcvImageFormatGetPlaneBitsPerPixel(format, i, &bpp);
            rowAlignPerPlane = bpp / 8;
        }
        else
        {
            NVCVDataType dtype;
            nvcvImageFormatGetPlaneDataType(format, i, &dtype);

            nvcv::DataType type{dtype};

            // must at least satisfy dtype alignment
            rowAlignPerPlane = std::lcm(userRowAlign, type.alignment());
        }

        rowAlign = std::lcm(rowAlign, rowAlignPerPlane);
    }

    rowAlign = RoundUpNextPowerOfTwo(rowAlign);

    int baseAlign;
    if (userBaseAlign == 0)
    {
        baseAlign = RoundUpNextPowerOfTwo(rowAlign);
    }
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid base address alignment of %d, only a value of 0 is supported", userBaseAlign);
    }

    // buffer address alignment must be at least the row alignment
    baseAlign = std::lcm(baseAlign, rowAlign);

    reqs.alignBytes = baseAlign;
    if (reqs.alignBytes > NVCV_MAX_MEM_REQUIREMENTS_BLOCK_SIZE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Alignment requirement of %d is larger than the maximum allowed %ld", reqs.alignBytes,
                              NVCV_MAX_MEM_REQUIREMENTS_BLOCK_SIZE);
    }

    // Calculate total device memory needed in blocks
    std::array<int32_t, 2> widths{width, width / 2};
    std::array<int32_t, 2> heights{height, height / 2};
    for (int p = 0; p < numPlanes; ++p)
    {
        int32_t planeWidth, planeHeight;
        nvcvImageFormatGetPlaneSize(format, p, width, height, &planeWidth, &planeHeight);

        assert((size_t)p < sizeof(reqs.planeRowStride) / sizeof(reqs.planeRowStride[0]));

        int32_t bpp;
        nvcvImageFormatGetPlaneBitsPerPixel(format, p, &bpp);
        int32_t bytesPerPixel = bpp / 8;

        reqs.planeRowStride[p] = RoundUpPowerOfTwo((int64_t)planeWidth * bytesPerPixel, rowAlign);

        nvcvMemRequirementsAddBuffer(&reqs.mem.cudaMem, reqs.planeRowStride[p] * planeHeight, baseAlign);
    }

    return reqs;
}

} // namespace pva_solutions::priv