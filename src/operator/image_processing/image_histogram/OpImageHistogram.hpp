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
 * @file OpImageHistogram.hpp
 *
 * @brief Defines the private C++ Class for the ImageHistogram operation.
 */

#ifndef PVA_SOLUTIONS_PRIV_IMAGEHISTOGRAM_HPP
#define PVA_SOLUTIONS_PRIV_IMAGEHISTOGRAM_HPP

#include "IOperator.hpp"

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Tensor.hpp>

#include <OpImageHistogram.h>
#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>

#include <cstdint>

namespace pva_solutions::priv {

template<typename T>
class ImageHistogram final : public IOperator
{
public:
    using ExecKey = std::tuple<cupva::impl::Context *, int32_t>;

    explicit ImageHistogram(NVCVTensorRequirements const *const inTensorRequirements,
                            NVCVTensorRequirements const *const outTensorRequirements,
                            PvaImageHistogramParams const *const imageHistogramParams);

    ~ImageHistogram();

    template<typename S>
    void operator()(S stream, const nvcv::Tensor &in, const nvcv::Tensor &out);

private:
    NVCVTensorRequirements m_inTensorRequirements;
    NVCVTensorRequirements m_outTensorRequirements;
    int32_t m_width;
    int32_t m_height;
    int32_t m_inputLinePitch;
    float m_start;
    float m_end;
    int32_t m_numBins;

    cupva::OffsetPointer<T> m_srcOp;
    cupva::OffsetPointer<uint32_t> m_dstOp;
    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_imageHistogramExecutables;
    static std::shared_ptr<cupva::Executable> GetImageHistogramExec(int32_t ways);
};

} // namespace pva_solutions::priv

#endif // PVA_SOLUTIONS_PRIV_IMAGEHISTOGRAM_HPP
