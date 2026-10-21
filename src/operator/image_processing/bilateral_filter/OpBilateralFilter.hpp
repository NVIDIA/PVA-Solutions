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
 * @file OpBilateralFilter.hpp
 *
 * @brief Defines the private C++ Class for the BilateralFilter operation.
 */

#ifndef PVA_SOLUTIONS_PRIV_BILATERALFILTER_HPP
#define PVA_SOLUTIONS_PRIV_BILATERALFILTER_HPP

#include "IOperator.hpp"

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Tensor.hpp>

#include <OpBilateralFilter.h>
#include <cupva_host_scheduling.h>
#include <nvcv/BorderType.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>

namespace pva_solutions::priv {

template<typename T>
class BilateralFilter final : public IOperator
{
public:
    using ExecDataKey = std::tuple<int32_t>;
    using ExecKey =
        decltype(std::tuple_cat(std::declval<ExecDataKey>(), std::declval<std::tuple<cupva::impl::Context *>>()));

    explicit BilateralFilter(NVCVTensorRequirements *tensorRequirements, int32_t kernelSize,
                             const NVCVBorderType borderMode, int32_t borderValue);

    ~BilateralFilter();

    template<typename S>
    void operator()(S stream, const nvcv::Tensor &in, const float sigmaRange, const float sigmaSpace,
                    const nvcv::Tensor &out);

private:
    NVCVTensorRequirements m_tensorRequirements;
    int32_t m_kernelSize;
    cupva::OffsetPointer<T> m_srcOp;
    cupva::OffsetPointer<T> m_dstOp;
    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_bilateralFilterExecutables;
    static std::shared_ptr<cupva::Executable> GetBilateralFilterExec(ExecDataKey execDataKey);
};

} // namespace pva_solutions::priv

// Internal implementation function for submit operations
// Used by macro-generated __cupva/__cuda wrappers
template<typename StreamType>
NVCVStatus BilateralFilterSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle in,
                                     const float sigmaRange, const float sigmaSpace, NVCVTensorHandle out);

#endif // PVA_SOLUTIONS_PRIV_BILATERALFILTER_HPP
