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
 * @file OpInference.hpp
 *
 * @brief Defines the private C++ Class for the DL Inference operation.
 */

#pragma once

#include "IOperator.hpp"

#include <cupva_host.hpp>
#include <nvcv/Tensor.hpp>

#include <OpDLInference.h>
#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>
#include <dlfcn.h>
#include <nvcv/Status.h>
#include <pva_lib_defs.h>

#include <string>
#include <vector>

namespace pva_solutions::priv {

class DLInference : public IOperator
{
public:
    explicit DLInference(PvaDLInferenceCreateParams *params);
    ~DLInference();

    template<typename S>
    void operator()(S stream, NVCVTensorHandle *inTensors, NVCVTensorHandle *outTensors);

    // Public const getters for tensor requirements
    const std::vector<NVCVTensorRequirements> &getInTensorRequirements() const
    {
        return m_inTensorRequirements;
    }
    const std::vector<NVCVTensorRequirements> &getOutTensorRequirements() const
    {
        return m_outTensorRequirements;
    }

private:
    std::vector<void *> m_intermediatePtrs;

    std::string m_networkName;
    std::vector<NVCVTensorRequirements> m_inTensorRequirements;
    std::vector<NVCVTensorRequirements> m_outTensorRequirements;

    std::vector<void *> m_inProxyTensorPtrs;
    std::vector<size_t> m_inProxyTensorSizes;
    std::vector<void *> m_outProxyTensorPtrs;
    std::vector<size_t> m_outProxyTensorSizes;

    template<typename S>
    void submitCopyCmds(S stream, NVCVTensorHandle *inTensors, bool isInput);

    pva_dl_lib_engine_t m_engine;
    void *m_networkFunc; // Function pointer to the network entry point

    template<typename S>
    void submit(S stream, int numOnnxCmds);
    void *getNetworkFunction(const std::string &networkName);
};

} // namespace pva_solutions::priv

// Internal implementation function for submit operations
// Used by macro-generated __cupva/__cuda wrappers
template<typename StreamType>
NVCVStatus DLInferenceSubmitImpl(NVCVOperatorHandle handle, StreamType stream, PvaDLInferenceSubmitParams *params);
