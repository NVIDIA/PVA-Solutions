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

#pragma once

#include "IOperator.hpp"

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Tensor.hpp>

#include <OpFastCorner.h>
#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>
#include <nvcv/BorderType.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <tuple>

namespace pva_solutions::priv {

template<typename T>
class FastCornerDetector final : public IOperator
{
public:
    using ExecDataKey = std::tuple<uint8_t, uint8_t>;
    using ExecKey =
        decltype(std::tuple_cat(std::declval<ExecDataKey>(), std::declval<std::tuple<cupva::impl::Context *>>()));

    static constexpr int32_t kMaxImageWidthHeight{32767};

    explicit FastCornerDetector(NVCVTensorRequirements const *const tensorRequirements,
                                PvaFastCornerDetectorParams const *const params, NVCVBorderType borderMode,
                                int32_t borderValue);

    template<typename S>
    void operator()(S stream, nvcv::Tensor const &in, int32_t intensityThreshold, nvcv::Tensor const &out,
                    nvcv::Tensor const &numLoc);

    ~FastCornerDetector();

private:
    NVCVTensorRequirements m_inTensorRequirements;

    cupva::OffsetPointer<T> m_srcOp;
    std::optional<T *> m_scoreBufDevPtr;
    cupva::OffsetPointer<float> m_locOp;
    cupva::OffsetPointer<int32_t> m_numLocOp;

    // TopK intermediate DRAM buffers
    std::optional<uint32_t *> m_packedLocsDevPtr;
    std::optional<uint16_t *> m_packedScoresDevPtr;
    bool m_enableTopK{false};

    cupvaCmd_t m_prog;
    std::shared_ptr<cupva::Executable> m_exec;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_fastCornerExecutables;
    static std::shared_ptr<cupva::Executable> GetFastCornerExec(ExecDataKey execDataKey);
};

} // namespace pva_solutions::priv

// Internal implementation function for submit operations
// Used by macro-generated __cupva/__cuda wrappers
template<typename StreamType>
NVCVStatus FastCornerDetectorSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle in,
                                        int32_t intensityThreshold, NVCVTensorHandle loc, NVCVTensorHandle numLoc);
