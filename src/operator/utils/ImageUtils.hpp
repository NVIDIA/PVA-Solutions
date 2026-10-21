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

/*
 * @file ImageUtils.hpp
 *
 * @brief Defines utility functions for image type's use.
 */

#ifndef IMAGE_UTILS_PRIV_HPP
#define IMAGE_UTILS_PRIV_HPP

#include <nvcv/Image.hpp>

namespace pva_solutions::utils {

constexpr bool AreImageRequirementsEqual(const NVCVImageRequirements &req, const NVCVImageData &data)
{
    if ((req.format != data.format) || (req.height != data.buffer.strided.planes[0].height) ||
        (req.width != data.buffer.strided.planes[0].width))
    {
        return false;
    }

    for (int i = 0; i < data.buffer.strided.numPlanes; i++)
    {
        if (req.planeRowStride[i] != data.buffer.strided.planes[i].rowStride)
        {
            return false;
        }
    }

    return true;
}

inline bool AreImageRequirementsEqualExcludingLayout(const NVCVImageRequirements &req, const NVCVImageData &data)
{
    // NVCV does not support to create NVCVImage for a BL layout, so needs workaround to pass the PL layout.
    // This will make the req.format has BL layout, while data.format has PL layout.
    // Bypass checking the BL/PL difference, if either "req" or "data" is in BL layout.
    NVCVImageFormat reqFormat = req.format, dataFormat = data.format;
    NVCVMemLayout reqMemLayout, dataMemLayout;
    nvcvImageFormatGetMemLayout(reqFormat, &reqMemLayout);
    nvcvImageFormatGetMemLayout(dataFormat, &dataMemLayout);
    if (reqMemLayout != NVCV_MEM_LAYOUT_PITCH_LINEAR)
    {
        nvcvImageFormatSetMemLayout(&reqFormat, NVCV_MEM_LAYOUT_PL);
    }
    if (dataMemLayout != NVCV_MEM_LAYOUT_PITCH_LINEAR)
    {
        nvcvImageFormatSetMemLayout(&dataFormat, NVCV_MEM_LAYOUT_PL);
    }

    if ((reqFormat != dataFormat) || (req.height != data.buffer.strided.planes[0].height) ||
        (req.width != data.buffer.strided.planes[0].width))
    {
        return false;
    }

    for (int i = 0; i < data.buffer.strided.numPlanes; i++)
    {
        if (req.planeRowStride[i] != data.buffer.strided.planes[i].rowStride)
        {
            return false;
        }
    }

    return true;
}

} // namespace pva_solutions::utils

#endif