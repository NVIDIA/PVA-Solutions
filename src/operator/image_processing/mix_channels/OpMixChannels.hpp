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
 * @file OpMixChannels.hpp
 *
 * @brief Defines the private C++ Class for the MixChannels operation.
 */

#ifndef PVA_SOLUTIONS_PRIV_MIXCHANNELS_HPP
#define PVA_SOLUTIONS_PRIV_MIXCHANNELS_HPP

#include "IOperator.hpp"

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Tensor.hpp>

#include <OpMixChannels.h>
#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

#define MAX_CASE_NUM (9)

namespace pva_solutions::priv {

class MixChannels final : public IOperator
{
public:
    using ExecDataKey = std::tuple<PVAMixChannelsCode>;
    using ExecKey =
        decltype(std::tuple_cat(std::declval<ExecDataKey>(), std::declval<std::tuple<cupva::impl::Context *>>()));

    explicit MixChannels(NVCVTensorRequirements **inTensorReqs, NVCVTensorRequirements **outTensorReqs,
                         int inTensorCount, int outTensorCount, PVAMixChannelsCode code);

    ~MixChannels();

    template<typename S>
    void operator()(S stream, NVCVTensorHandle *inHandles, NVCVTensorHandle *outHandles, int inTensorCount,
                    int outTensorCount);

private:
    void createParamCheck(const char *prefix, int inTensorNumRef, int outTensorNumRef, int srcImgCRef, int dstImgCRef,
                          NVCVDataType srcDataTypeRef, NVCVDataType dstDataTypeRef);
    std::vector<NVCVTensorRequirements> m_inputTensorReqs;
    std::vector<NVCVTensorRequirements> m_outputTensorReqs;
    std::vector<cupva::OffsetPointer<uint8_t>> m_srcOps;
    std::vector<cupva::OffsetPointer<uint8_t>> m_dstOps;
    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_mixChannelsExecutables;
    static std::shared_ptr<cupva::Executable> GetMixChannelsExec(ExecDataKey execDataKey);
};

} // namespace pva_solutions::priv

#endif // PVA_SOLUTIONS_PRIV_MIXCHANNELS_HPP