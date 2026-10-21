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
 * @file OpCCL.hpp
 *
 * @brief Defines the private C++ Class for the CCL operation.
 */

#ifndef PVA_SOLUTIONS_PRIV_CCL_HPP
#define PVA_SOLUTIONS_PRIV_CCL_HPP

#include <IOperator.hpp>
#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Tensor.hpp>

#include <OpCCL.h>
#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>

namespace pva_solutions::priv {

class CCL final : public IOperator
{
public:
    using ExecKey = std::tuple<cupva::impl::Context *>;
    explicit CCL(const NVCVTensorRequirements *imageParams, const NVCVTensorRequirements *labelParams,
                 const PVAConnectivityType connectivity, const uint8_t threshold);
    ~CCL();

    template<typename S>
    void operator()(S stream, const nvcv::Tensor &image, const nvcv::Tensor &labels, const nvcv::Tensor &numLabels);

private:
    NVCVTensorRequirements m_imageTensorRequirements;
    NVCVTensorRequirements m_labelTensorRequirements;

    int32_t m_width;     // image width
    int32_t m_height;    // image height
    int32_t m_linePitch; // image linepitch

    cupva::OffsetPointer<uint8_t> m_imageOp;
    cupva::OffsetPointer<uint16_t> m_labelsOp;
    cupva::OffsetPointer<int32_t> m_numLabelsOp;

    uint16_t *m_bufDevPtr;

    cupvaCmd_t m_prog;
    std::shared_ptr<cupva::Executable> m_exec;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_CCLExecutables;
    static std::shared_ptr<cupva::Executable> GetCCLExec();
};

} // namespace pva_solutions::priv

// Internal implementation function for submit operations
// Used by macro-generated __cupva/__cuda wrappers
template<typename StreamType>
NVCVStatus CCLSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle image, NVCVTensorHandle labels,
                         NVCVTensorHandle numLabels);

#endif // PVA_SOLUTIONS_PRIV_CCL_HPP
