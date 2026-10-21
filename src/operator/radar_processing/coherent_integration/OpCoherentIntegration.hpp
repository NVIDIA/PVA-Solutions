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
 * @file OpCoherentIntegration.hpp
 *
 * @brief Private C++ implementation of radar coherent integration.
 */

#ifndef PVA_SOLUTIONS_PRIV_COHERENT_INTEGRATION_HPP
#define PVA_SOLUTIONS_PRIV_COHERENT_INTEGRATION_HPP

#include "IOperator.hpp"

#include "device/coherent_integration_param.h"

#include <cupva_host.hpp>
#include <nvcv/Tensor.hpp>

#include <OpCoherentIntegration.h>

#include <map>
#include <memory>
#include <mutex>
#include <tuple>

namespace pva_solutions::priv {

class CoherentIntegration final : public IOperator
{
public:
    /* The first key element is the fixed Rx count embedded in the selected
     * VPU executable. */
    using ExecKey = std::tuple<uint32_t, cupva::impl::Context *>;

    CoherentIntegration(NVCVTensorRequirements const *rangeDopplerMapTensorRequirements,
                        NVCVTensorRequirements const *coherentIntegrationRxTensorRequirements,
                        NVCVTensorRequirements const *coherentIntegrationFinalTensorRequirements,
                        PVARadarCoherentIntegrationParams const *params);
    ~CoherentIntegration() override;

    template<typename StreamType>
    void operator()(StreamType stream, nvcv::Tensor const &rangeDopplerMap, nvcv::Tensor const &steeringVectors,
                    nvcv::Tensor const &coherentIntegrationRx, nvcv::Tensor const &coherentIntegrationFinal);

private:
    static std::shared_ptr<cupva::Executable> GetExecutable(uint32_t numRx);

    NVCVTensorRequirements m_rangeDopplerMapTensorRequirements;
    NVCVTensorRequirements m_coherentIntegrationRxTensorRequirements;
    NVCVTensorRequirements m_coherentIntegrationFinalTensorRequirements;
    CoherentIntegrationParams m_globalParams;

    cupva::OffsetPointer<uint8_t> m_inputOp;
    cupva::OffsetPointer<uint8_t> m_steeringOp;
    cupva::OffsetPointer<uint8_t> m_coherentIntegrationRxOp;
    cupva::OffsetPointer<uint8_t> m_coherentIntegrationFinalOp;

    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;

    static std::mutex s_execMapMutex;
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_executables;
};

} // namespace pva_solutions::priv

template<typename StreamType>
NVCVStatus CoherentIntegrationSubmitImpl(NVCVOperatorHandle handle, StreamType stream,
                                         NVCVTensorHandle const rangeDopplerMap, NVCVTensorHandle const steeringVectors,
                                         NVCVTensorHandle coherentIntegrationRx,
                                         NVCVTensorHandle coherentIntegrationFinal);

#endif // PVA_SOLUTIONS_PRIV_COHERENT_INTEGRATION_HPP
