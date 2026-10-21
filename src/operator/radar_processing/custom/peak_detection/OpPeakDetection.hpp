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
 * @file OpPeakDetection.hpp
 *
 * @brief Defines the private C++ Class for the Peak Detection operation.
 */

#pragma once

#include "IOperator.hpp"

#include <cupva_host.hpp>
#include <nvcv/Tensor.hpp>

#include <OpPeakDetection.h>
#include <PvaOperatorTypes.h>

#include <map>
#include <mutex>
#include <optional>
#include <tuple>

namespace pva_solutions::priv {

class PeakDetection final : public IOperator
{
public:
    using ExecDataKey = std::tuple<int32_t>;
    using ExecKey =
        decltype(std::tuple_cat(std::declval<ExecDataKey>(), std::declval<std::tuple<cupva::impl::Context *>>()));

    explicit PeakDetection(NVCVTensorRequirements const *const inTensorRequirements,
                           NVCVTensorRequirements const *const outTensorRequirements, int32_t NofTx,
                           int32_t repeatFold);

    template<typename S>
    void operator()(S stream, nvcv::Tensor const &FT2D, nvcv::Tensor const &nciRx, nvcv::Tensor const &nciFinal,
                    nvcv::Tensor const &noiseEst, nvcv::Tensor const &peakCount, nvcv::Tensor const &peakIndex,
                    nvcv::Tensor const &snap);

    ~PeakDetection();

private:
    NVCVTensorRequirements m_FT2DTensorRequirements;
    NVCVTensorRequirements m_nciRxTensorRequirements;
    NVCVTensorRequirements m_nciFinalTensorRequirements;
    NVCVTensorRequirements m_noiseEstTensorRequirements;
    NVCVTensorRequirements m_peakCountTensorRequirements;
    NVCVTensorRequirements m_peakIndexTensorRequirements;
    NVCVTensorRequirements m_snapTensorRequirements;

    cupva::OffsetPointer<uint32_t> m_nciFinalOp;
    cupva::OffsetPointer<uint32_t> m_nciRxOp;
    cupva::OffsetPointer<int32_t> m_FT2DOp;
    cupva::OffsetPointer<uint32_t> m_noiseEstOp;

    cupva::OffsetPointer<uint32_t> m_peakCountOp;
    cupva::OffsetPointer<uint32_t> m_rangeIdxOp;
    cupva::OffsetPointer<uint32_t> m_dopplerFoldIdxOp;
    cupva::OffsetPointer<uint32_t> m_dopplerIdxOp;
    cupva::OffsetPointer<int32_t> m_snapOp;

    cupvaCmd_t m_prog;
    std::shared_ptr<cupva::Executable> m_exec;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_peakDetectionExecutables;
    static std::shared_ptr<cupva::Executable> GetPeakDetectionExec(ExecDataKey execDataKey);
};
} // namespace pva_solutions::priv

// Internal implementation function for submit operations
// Used by macro-generated __cupva/__cuda wrappers
template<typename StreamType>
NVCVStatus PeakDetectionSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle *inHandles,
                                   NVCVTensorHandle *outHandles, int inTensorCount, int outTensorCount);
