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
 * @file OpRadarCFAR.hpp
 *
 * @brief Defines the private C++ class for the Radar CFAR operation.
 */

#pragma once

#include "IOperator.hpp"

#include <cupva_host.hpp>
#include <nvcv/Tensor.hpp>

#include <OpRadarCFAR.h>
#include <PvaOperatorTypes.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>

namespace pva_solutions::priv {

class RadarCFAR final : public IOperator
{
public:
    /** One executable per CFAR algorithm, keyed by the algorithm enum. */
    using ExecDataKey = std::tuple<int32_t>;
    using ExecKey =
        decltype(std::tuple_cat(std::declval<ExecDataKey>(), std::declval<std::tuple<cupva::impl::Context *>>()));

    explicit RadarCFAR(NVCVTensorRequirements const *const inTensorRequirements,
                       NVCVTensorRequirements const *const outDetectionListTensorRequirements,
                       PVARadarCFARType const algorithm, PvaRadarCFARParams const *const radarCFARParams);

    template<typename S>
    void operator()(S stream, nvcv::Tensor const &input, nvcv::Tensor const &outDetectionList,
                    nvcv::Tensor const &outDetectionCount);

    ~RadarCFAR();

private:
    NVCVTensorRequirements m_inTensorRequirements;
    NVCVTensorRequirements m_outDetectionListTensorRequirements;

    cupva::OffsetPointer<uint32_t> m_inOp;
    cupva::OffsetPointer<int32_t> m_detectionListOp;
    cupva::OffsetPointer<int32_t> m_detectionCountOp;

    /** Operator-owned DRAM scratch holding the horizontal decision mask. */
    int32_t *m_maskDevPtr;

    /** Operator-owned DRAM staging for the per-position reciprocal and factor tables. */
    int32_t *m_tableDevPtr;

    cupvaCmd_t m_prog;
    std::shared_ptr<cupva::Executable> m_exec;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_radarCFARExecutables;
    static std::shared_ptr<cupva::Executable> GetRadarCFARExec(ExecDataKey execDataKey);
};
} // namespace pva_solutions::priv

// Internal implementation function for submit operations
// Used by macro-generated __cupva/__cuda wrappers
template<typename StreamType>
NVCVStatus RadarCFARSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle const inputTensor,
                               NVCVTensorHandle outDetectionListTensor, NVCVTensorHandle outDetectionCountTensor);
