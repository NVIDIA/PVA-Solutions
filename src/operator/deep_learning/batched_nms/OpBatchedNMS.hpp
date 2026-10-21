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

#pragma once

#include <IOperator.hpp>
#include <cupva_host.hpp>
#include <nvcv/Tensor.hpp>

#include <nvcv/Tensor.h>

#include <map>
#include <memory>
#include <mutex>
#include <tuple>

namespace pva_solutions::priv {

class BatchedNMS final : public IOperator
{
public:
    using ExecKey = std::tuple<cupva::impl::Context *>;

    BatchedNMS(const NVCVTensorRequirements *inBoxesReqs, const NVCVTensorRequirements *inScoresReqs,
               const NVCVTensorRequirements *inClassesReqs, const NVCVTensorRequirements *inCountReqs,
               const NVCVTensorRequirements *outBoxesReqs, const NVCVTensorRequirements *outScoresReqs,
               const NVCVTensorRequirements *outClassesReqs, const NVCVTensorRequirements *outCountReqs,
               float iouThreshold);

    template<typename StreamType>
    void operator()(StreamType stream, const nvcv::Tensor &inBoxes, const nvcv::Tensor &inScores,
                    const nvcv::Tensor &inClasses, const nvcv::Tensor &inCount, const nvcv::Tensor &outBoxes,
                    const nvcv::Tensor &outScores, const nvcv::Tensor &outClasses, const nvcv::Tensor &outCount);

    ~BatchedNMS();

private:
    static std::shared_ptr<cupva::Executable> GetBatchedNMSExec();
    static std::mutex s_execMapMutex;
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_batchedNMSExecutables;

    int m_maxInputBoxes;
    int m_maxOutputBoxes;
    float m_iouThreshold;

    NVCVTensorRequirements m_inBoxesTensorRequirements;
    NVCVTensorRequirements m_inScoresTensorRequirements;
    NVCVTensorRequirements m_inClassesTensorRequirements;
    NVCVTensorRequirements m_inCountTensorRequirements;
    NVCVTensorRequirements m_outBoxesTensorRequirements;
    NVCVTensorRequirements m_outScoresTensorRequirements;
    NVCVTensorRequirements m_outClassesTensorRequirements;
    NVCVTensorRequirements m_outCountTensorRequirements;

    std::shared_ptr<cupva::Executable> m_exec;
    cupva::OffsetPointer<float> m_inBoxesOp;
    cupva::OffsetPointer<int16_t> m_inScoresOp;
    cupva::OffsetPointer<int16_t> m_inClassesOp;
    cupva::OffsetPointer<int> m_inCountOp;
    cupva::OffsetPointer<float> m_outBoxesOp;
    cupva::OffsetPointer<float> m_outScoresOp;
    cupva::OffsetPointer<int> m_outClassesOp; // VPU writes 4-byte int labels
    cupva::OffsetPointer<int> m_outCountOp;
    cupvaCmd_t m_prog;
};

} // namespace pva_solutions::priv
