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
 * @file OpContourTracing.hpp
 *
 * @brief Defines the private C++ Class for the ContourTracing operation.
 */

#ifndef PVA_SOLUTIONS_PRIV_CONTOURTRACING_HPP
#define PVA_SOLUTIONS_PRIV_CONTOURTRACING_HPP

#include "IOperator.hpp"

#include <nvcv/Tensor.hpp>

#include <OpContourTracing.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>

namespace pva_solutions::priv {

template<typename T>
class ContourTracing final : public IOperator
{
public:
    using ExecKey = std::tuple<cupva::impl::Context *>;

    explicit ContourTracing(NVCVTensorRequirements const *const inTensorRequirements,
                            NVCVTensorRequirements const *const rectTensorRequirements,
                            NVCVTensorRequirements const *const outTensorRequirements,
                            NVCVTensorRequirements const *const lenTensorRequirements);

    ~ContourTracing();

    template<typename S>
    void operator()(S stream, const nvcv::Tensor &inTensor, const nvcv::Tensor &rectTensor,
                    const nvcv::Tensor &outTensor, const nvcv::Tensor &lenTensor, const int32_t numRects);

private:
    NVCVTensorRequirements m_inTensorRequirements;
    NVCVTensorRequirements m_rectTensorRequirements;
    NVCVTensorRequirements m_outTensorRequirements;
    NVCVTensorRequirements m_lenTensorRequirements;

    cupva::OffsetPointer<T> m_srcOp;
    cupva::OffsetPointer<int32_t> m_rectOp;
    cupva::OffsetPointer<uint16_t> m_dstOp;
    cupva::OffsetPointer<uint16_t> m_lenOp;
    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_contourTracingExecutables;
    static std::shared_ptr<cupva::Executable> GetContourTracingExec();
};

} // namespace pva_solutions::priv

// Internal implementation function for submit operations
// Used by macro-generated __cupva/__cuda wrappers
template<typename StreamType>
NVCVStatus ContourTracingSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle const inTensor,
                                    NVCVTensorHandle const rectTensor, NVCVTensorHandle outTensor,
                                    NVCVTensorHandle lenTensor, const int32_t numRects);

#endif // PVA_SOLUTIONS_PRIV_CONTOURTRACING_HPP