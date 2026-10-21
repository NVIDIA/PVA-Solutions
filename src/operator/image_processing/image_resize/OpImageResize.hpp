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

/**
 * @file OpImageResize.hpp
 *
 * @brief Defines the private C++ Class for the ImageResize operation.
 */

#ifndef PVA_SOLUTIONS_PRIV_IMAGE_RESIZE_HPP
#define PVA_SOLUTIONS_PRIV_IMAGE_RESIZE_HPP

#include "IOperator.hpp"

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Image.hpp>

#include <OpImageResize.h>
#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>
#include <nvcv/BorderType.h>

namespace pva_solutions::priv {

template<typename T>
class ImageResize final : public IOperator
{
public:
    using ExecDataKey = std::tuple<NVCVImageFormat, PVAInterpolationType>;

    using ExecKey =
        decltype(std::tuple_cat(std::declval<ExecDataKey>(), std::declval<std::tuple<cupva::impl::Context *>>()));

    explicit ImageResize(NVCVImageRequirements *inImageRequirements, NVCVImageRequirements *outImageRequirements,
                         PVAInterpolationType resizeType);

    void checkImageFormat(NVCVImageFormat format);
    std::tuple<int, int, int, int> findBestTileSize(int srcWidth, int srcHeight, int dstWidth, int dstHeight,
                                                    int channelNum, NVCVImageFormat imageFormat,
                                                    PVAInterpolationType resizeType);

    template<typename S>
    void operator()(S stream, const nvcv::Image &in, const nvcv::Image &out);

    ~ImageResize();

private:
    NVCVImageRequirements m_inImageRequirements;
    NVCVImageRequirements m_outImageRequirements;

    // image could have 2 planes with different w and h
    NVCVImageFormat m_imageFormat;
    int32_t m_planeCount;
    uint8_t m_channelCount[2];

    uint16_t m_srcImageWidth;
    uint16_t m_srcImageHeight;
    uint16_t m_srcLinePitch[2];
    uint16_t m_dstImageWidth;
    uint16_t m_dstImageHeight;
    uint16_t m_dstLinePitch[2];

    PVAInterpolationType m_resizeType;
    // image of at most 2 planes
    cupva::OffsetPointer<T> m_srcOp[2];
    cupva::OffsetPointer<T> m_dstOp[2];
    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_imageResizeExecutables;
    static std::shared_ptr<cupva::Executable> GetImageResizeExec(ExecDataKey execDataKey);
};

} // namespace pva_solutions::priv

// Internal implementation function for submit operations
// Used by macro-generated __cupva/__cuda wrappers
template<typename StreamType>
NVCVStatus ImageResizeSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVImageHandle in, NVCVImageHandle out);

#endif // PVA_SOLUTIONS_PRIV_IMAGE_RESIZE_HPP
