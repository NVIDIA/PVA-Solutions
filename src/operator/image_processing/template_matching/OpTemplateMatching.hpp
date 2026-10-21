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
 * @file OpTemplateMatching.hpp
 *
 * @brief Defines the private C++ class for the TemplateMatching operation.
 */

#ifndef PVA_SOLUTIONS_PRIV_TEMPLATEMATCHING_HPP
#define PVA_SOLUTIONS_PRIV_TEMPLATEMATCHING_HPP

#include "IOperator.hpp"

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Tensor.hpp>

#include <OpTemplateMatching.h>
#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>
#include <nvcv/BorderType.h>

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>

namespace pva_solutions::priv {

class TemplateMatching final : public IOperator
{
public:
    /* One executable per context, shared by every handle built in it. The tile
     * size is a runtime parameter, so one device binary serves every geometry. */
    using ExecKey = std::tuple<cupva::impl::Context *>;

    TemplateMatching(NVCVTensorRequirements const *const inImageTensorRequirements,
                     NVCVTensorRequirements const *const inTemplateTensorRequirements,
                     NVCVTensorRequirements const *const outTensorRequirements);
    ~TemplateMatching();

    template<typename S>
    void operator()(S stream, const nvcv::Tensor &inImage, const nvcv::Tensor &inTemplate, const nvcv::Tensor &out)
    {
        CheckTensorsMatchCreate(inImage, inTemplate, out);
        Submit(stream, inImage, inTemplate, out);
    }

private:
    static std::shared_ptr<cupva::Executable> GetExec();

    void CheckTensorsMatchCreate(const nvcv::Tensor &inImage, const nvcv::Tensor &inTemplate,
                                 const nvcv::Tensor &out) const;

    void Submit(cupvaStream_t stream, const nvcv::Tensor &inImage, const nvcv::Tensor &inTemplate,
                const nvcv::Tensor &out);
    void Submit(cudaStream_t stream, const nvcv::Tensor &inImage, const nvcv::Tensor &inTemplate,
                const nvcv::Tensor &out);

    template<typename S>
    void SubmitImpl(S stream, const nvcv::Tensor &inImage, const nvcv::Tensor &inTemplate, const nvcv::Tensor &out);

    NVCVTensorRequirements m_inImageTensorRequirements;
    NVCVTensorRequirements m_inTemplateTensorRequirements;
    NVCVTensorRequirements m_outTensorRequirements;

    std::shared_ptr<cupva::Executable> m_exec;

    cupva::OffsetPointer<uint8_t> m_imageOp;
    cupva::OffsetPointer<uint8_t> m_templateOp;
    cupva::OffsetPointer<float> m_outOp;

    cupvaCmd_t m_prog;

    static std::mutex s_execMapMutex;
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_execs;
};

} // namespace pva_solutions::priv

// Internal implementation function for submit operations
// Used by macro-generated __cupva/__cuda wrappers
template<typename StreamType>
NVCVStatus TemplateMatchingSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle inImage,
                                      NVCVTensorHandle inTemplate, NVCVTensorHandle out);

#endif // PVA_SOLUTIONS_PRIV_TEMPLATEMATCHING_HPP
