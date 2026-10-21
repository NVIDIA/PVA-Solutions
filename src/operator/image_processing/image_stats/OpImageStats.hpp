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

/**
 * @file OpImageResize.hpp
 *
 * @brief Defines the private C++ Class for the ImageResize operation.
 */

#ifndef PVA_SOLUTIONS_PRIV_IMAGE_STATS_HPP
#define PVA_SOLUTIONS_PRIV_IMAGE_STATS_HPP

#include "IOperator.hpp"

#include "device/image_stats_param.h"

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>

#include <OpImageStats.h>
#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>
#include <nvcv/BorderType.h>

namespace pva_solutions::priv {

template<typename T>
class ImageStats final : public IOperator
{
public:
    using ExecDataKey = std::tuple<NVCVImageFormat, bool, PVAImageStatOpType>;

    using ExecKey =
        decltype(std::tuple_cat(std::declval<ExecDataKey>(), std::declval<std::tuple<cupva::impl::Context *>>()));

    void unpackFlagsToOpType(uint32_t flags, bool hasMask, NVCVImageFormat format);

    void checkImageFormat(NVCVImageFormat format);

    explicit ImageStats(NVCVImageRequirements *imageRequirements, NVCVImageRequirements *maskRequirements,
                        uint32_t flags);

    template<typename S>
    void operator()(S stream, const nvcv::Image &in, const nvcv::Image &mask, const nvcv::Tensor &out);

    ~ImageStats();

private:
    NVCVImageRequirements m_imageRequirements;
    NVCVImageRequirements m_maskRequirements;
    bool m_hasMask;
    PVAImageStatOpType m_opType;

    // flags for statistics to calculate
    bool m_pixelCountFlag;
    bool m_sumFlag;
    bool m_meanFlag;
    bool m_varianceFlag;
    bool m_covarianceFlag;

    NVCVImageFormat m_imageFormat;
    int32_t m_planeCount;
    uint16_t m_srcImageWidth;
    uint16_t m_srcImageHeight;

    // image of at most 2 planes
    cupva::OffsetPointer<T> m_srcOp[MAX_PLANE_COUNT];
    cupva::OffsetPointer<T> m_maskOp;
    cupva::OffsetPointer<uint8_t> m_outOp;

    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_imageStatsExecutables;
    static std::shared_ptr<cupva::Executable> GetImageStatsExec(ExecDataKey execDataKey);
};

} // namespace pva_solutions::priv

// Internal implementation function for submit operations
// Used by macro-generated __cupva/__cuda wrappers
template<typename StreamType>
NVCVStatus ImageStatsSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVImageHandle in, NVCVImageHandle mask,
                                NVCVTensorHandle out);

#endif // PVA_SOLUTIONS_PRIV_IMAGE_STATS_HPP
