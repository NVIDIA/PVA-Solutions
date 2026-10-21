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
 * @file OpMedianFilter.hpp
 *
 * @brief Defines the private C++ Class for the MedianFilter operation.
 */

#ifndef PVA_SOLUTIONS_PRIV_MEDIAN_FILTER_HPP
#define PVA_SOLUTIONS_PRIV_MEDIAN_FILTER_HPP

#include "IOperator.hpp"

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>

#include <OpMedianFilter.h>
#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>
#include <nvcv/BorderType.h>

namespace pva_solutions::priv {

template<typename T, uint8_t CH>
class MedianFilter final : public IOperator
{
public:
    using ExecDataKey = std::tuple<int32_t, int32_t>;

    using ExecKey =
        decltype(std::tuple_cat(std::declval<ExecDataKey>(), std::declval<std::tuple<cupva::impl::Context *>>()));

    explicit MedianFilter(NVCVImageRequirements *inImageRequirements, NVCVTensorRequirements *kernelRequirements,
                          NVCVImageRequirements *outImageRequirements, NVCVBorderType borderType, int32_t borderValue);

    void checkImageFormat(NVCVImageFormat format);

    template<typename S>
    void operator()(S stream, const nvcv::Image &in, const nvcv::Image &out, const nvcv::Tensor &kernel);

    ~MedianFilter();

private:
    NVCVImageRequirements m_inImageRequirements;
    NVCVImageRequirements m_outImageRequirements;
    NVCVTensorRequirements m_kernelRequirements;

    NVCVImageFormat m_imageFormat;
    int32_t m_imageWidth;
    int32_t m_imageHeight;

    int32_t m_kernelWidth;
    int32_t m_kernelHeight;
    cupva::PadModeType m_padMode;
    int32_t m_padValue;

    // for 2S16 IO, interpret buffer as int32_t so CLAMP support works
    using TSRC = std::conditional_t<std::is_same<T, int16_t>::value && CH == 2, int32_t, T>;
    cupva::OffsetPointer<TSRC> m_srcOp;
    cupva::OffsetPointer<T> m_dstOp;
    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_medianFilterExecutables;
    static std::shared_ptr<cupva::Executable> GetMedianFilterExec(ExecDataKey execDataKey);
};

} // namespace pva_solutions::priv

// Internal implementation function for submit operations
// Used by macro-generated __cupva/__cuda wrappers
template<typename StreamType>
NVCVStatus MedianFilterSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVImageHandle in,
                                  NVCVTensorHandle kernel, NVCVImageHandle out);

#endif // PVA_SOLUTIONS_PRIV_MEDIAN_FILTER_HPP
