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

#ifndef PVA_SOLUTIONS_PRIV_RADAR_TRANSPOSE_HPP
#define PVA_SOLUTIONS_PRIV_RADAR_TRANSPOSE_HPP

#include "IOperator.hpp"

#include <cupva_host.hpp>
#include <nvcv/Tensor.hpp>

#include <OpRadarTranspose.h>
#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>

namespace pva_solutions::priv {

class RadarTranspose final : public IOperator
{
public:
    using ExecKey = std::tuple<cupva::impl::Context *>;

    explicit RadarTranspose(NVCVTensorRequirements const *const inTensorRequirements,
                            NVCVTensorRequirements const *const outTensorRequirements,
                            PVARadarTransposeParams const *const params);

    ~RadarTranspose();

    template<typename S>
    void operator()(S stream, const nvcv::Tensor &in, const nvcv::Tensor &out);

private:
    NVCVTensorRequirements m_inTensorRequirements;
    NVCVTensorRequirements m_outTensorRequirements;

    cupva::OffsetPointer<int32_t> m_srcOp;
    cupva::OffsetPointer<int32_t> m_dstOp;

    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;

    static std::mutex s_execMapMutex;
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_radarTransposeExecutables;
    static std::shared_ptr<cupva::Executable> GetRadarTransposeExec();
};

} // namespace pva_solutions::priv

template<typename StreamType>
NVCVStatus RadarTransposeSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle const in,
                                    NVCVTensorHandle out);

#endif // PVA_SOLUTIONS_PRIV_RADAR_TRANSPOSE_HPP
