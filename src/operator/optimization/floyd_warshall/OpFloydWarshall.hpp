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
 * @file OpFloydWarshall.hpp
 *
 * @brief Defines the private C++ Class for the Floyd-Warshall operation.
 */

#ifndef PVA_SOLUTIONS_PRIV_FLOYDWARSHALL_HPP
#define PVA_SOLUTIONS_PRIV_FLOYDWARSHALL_HPP

#include "IOperator.hpp"

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Tensor.hpp>

#include <OpFloydWarshall.h>
#include <cupva_host_scheduling.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>

namespace pva_solutions::priv {

class FloydWarshall final : public IOperator
{
public:
    using ExecKey = std::tuple<cupva::impl::Context *>;

    explicit FloydWarshall(NVCVTensorRequirements *tensorRequirements);

    ~FloydWarshall();

    template<typename S>
    void operator()(S stream, const nvcv::Tensor &lenTensor, const nvcv::Tensor &nextTensor);

private:
    NVCVTensorRequirements m_tensorRequirements;
    cupva::OffsetPointer<uint16_t> m_lenTensorOp;
    cupva::OffsetPointer<uint16_t> m_nextTensorOp;
    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_floydWarshallExecutables;
    static std::shared_ptr<cupva::Executable> GetFloydWarshallExec();
};

} // namespace pva_solutions::priv

// Internal implementation function for submit operations
// Used by macro-generated __cupva/__cuda wrappers
template<typename StreamType>
NVCVStatus FloydWarshallSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle lenTensor,
                                   NVCVTensorHandle nextTensor);

#endif // PVA_SOLUTIONS_PRIV_FLOYDWARSHALL_HPP