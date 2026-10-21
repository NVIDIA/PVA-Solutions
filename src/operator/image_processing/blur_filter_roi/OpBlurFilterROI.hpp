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
 * @file OpBlurFilterROI.hpp
 *
 * @brief Defines the private C++ Class for the BlurFilterROI operation.
 */

#ifndef PVA_SOLUTIONS_PRIV_BLURFILTERROI_HPP
#define PVA_SOLUTIONS_PRIV_BLURFILTERROI_HPP

#include "IOperator.hpp"

#include "device/blur_filter_roi_param.h"

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>

#include <OpBlurFilterROI.h>
#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>

#include <cstdint>

namespace pva_solutions::priv {

template<typename T>
class BlurFilterROI final : public IOperator
{
public:
    using ExecKey = std::tuple<cupva::impl::Context *>;

    explicit BlurFilterROI(NVCVImageRequirements const *const inImageRequirements,
                           NVCVImageRequirements const *const outImageRequirements,
                           NVCVTensorRequirements const *const rectTensorRequirements, const NVCVSize2D blockSize);

    ~BlurFilterROI();

    template<typename S>
    void operator()(S stream, const nvcv::Image &in, const nvcv::Image &out, const nvcv::Tensor &roiRects,
                    const size_t numROIRects);

private:
    NVCVImageRequirements m_inImageRequirements;
    NVCVImageRequirements m_outImageRequirements;
    NVCVTensorRequirements m_rectTensorRequirements;
    int32_t m_width;
    int32_t m_height;
    int32_t m_inLinePitch[MAX_PLANE_COUNT];
    int32_t m_outLinePitch[MAX_PLANE_COUNT];
    NVCVImageFormat m_inFormat;
    NVCVImageFormat m_outFormat;
    NVCVMemLayout m_inLayout;
    NVCVMemLayout m_outLayout;
    int32_t m_numPlanes;
    cupva::OffsetPointer<int32_t> m_rectOp;
    cupva::OffsetPointer<T> m_srcOps[MAX_PLANE_COUNT];
    cupva::OffsetPointer<T> m_dstOps[MAX_PLANE_COUNT];
    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_blurFilterROIExecutables;
    static std::shared_ptr<cupva::Executable> GetBlurFilterROIExec();

    bool isImageFormatNV12(NVCVImageFormat format);
};

} // namespace pva_solutions::priv

// Internal implementation function for submit operations
// Used by macro-generated __cupva/__cuda wrappers
template<typename StreamType>
NVCVStatus BlurFilterROISubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVImageHandle const in,
                                   NVCVImageHandle out, NVCVTensorHandle const rect, const size_t numRects);

#endif // PVA_SOLUTIONS_PRIV_BLURFILTERROI_HPP
