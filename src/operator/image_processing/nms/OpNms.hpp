/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OpNms.hpp
 *
 * @brief Defines the private C++ Class for the Non-Maximum Suppression operation.
 */

#ifndef PVA_SOLUTIONS_PRIV_NMS_HPP
#define PVA_SOLUTIONS_PRIV_NMS_HPP

#include "IOperator.hpp"

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Tensor.hpp>

#include <OpNms.h>
#include <cupva_host_scheduling.h>
#include <nvcv/BorderType.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>

namespace pva_solutions::priv {

template<typename T>
class Nms final : public IOperator
{
public:
    using ExecDataKey = std::tuple<int32_t>;
    using ExecKey =
        decltype(std::tuple_cat(std::declval<ExecDataKey>(), std::declval<std::tuple<cupva::impl::Context *>>()));

    explicit Nms(NVCVTensorRequirements *tensorRequirements, int32_t nmsSize, const NVCVBorderType borderMode,
                 int32_t borderValue);
    ~Nms();

    template<typename S>
    void operator()(S stream, const nvcv::Tensor &in, const nvcv::Tensor &out);

private:
    std::shared_ptr<cupva::Executable> GetExec(ExecDataKey execDataKey);
    void initializeProgram();

    NVCVTensorRequirements m_tensorRequirements;
    cupva::OffsetPointer<T> m_srcOp;
    cupva::OffsetPointer<T> m_dstOp;
    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;
    int32_t m_nmsSize;
    NVCVBorderType m_borderMode;
    int32_t m_borderValue;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_nmsExecutables;
};

} // namespace pva_solutions::priv

#endif // PVA_SOLUTIONS_PRIV_NMS_HPP
