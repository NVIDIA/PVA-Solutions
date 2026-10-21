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
 * @file OpBruteForceMatcher.hpp
 *
 * @brief Defines the private C++ Class for the BruteForceMatcher operation.
 */

#pragma once

#include "IOperator.hpp"

//#include "device/brute_force_matcher_param.h"

#include <cupva_host.hpp>
#include <nvcv/Tensor.hpp>

#include <OpBruteForceMatcher.h>
#include <PvaOperatorTypes.h>

#include <map>
#include <mutex>
#include <optional>
#include <tuple>

namespace pva_solutions::priv {

class BruteForceMatcher final : public IOperator
{
public:
    using ExecKey = std::tuple<cupva::impl::Context *>;

    explicit BruteForceMatcher();

    template<typename S>
    void operator()(S stream, PVABriefDescriptor *query, int32_t queryCount, PVABriefDescriptor *reference,
                    int32_t referenceCount, int32_t maxMatchesPerQuery, PVABFMatchesType *matches,
                    uint8_t enableCrossCheck, uint8_t enableDistanceRatioTest, int32_t lowesTestThresholdNumerator,
                    int32_t lowesTestThresholdDenominator);

    ~BruteForceMatcher();

private:
    cupva::OffsetPointer<uint8_t> m_queryOp;
    cupva::OffsetPointer<uint8_t> m_referenceOp;
    cupva::OffsetPointer<uint32_t> m_matchesOp;

    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_bruteForceMatcherExecutables;
    static std::shared_ptr<cupva::Executable> GetBruteForceMatcherExec();
};
} // namespace pva_solutions::priv

// Internal implementation function for submit operations
// Used by macro-generated __cupva/__cuda wrappers
template<typename StreamType>
NVCVStatus BruteForceMatcherSubmitImpl(NVCVOperatorHandle handle, StreamType stream, PVABriefDescriptor *query,
                                       int32_t queryCount, PVABriefDescriptor *reference, int32_t referenceCount,
                                       int32_t maxMatchesPerQuery, PVABFMatchesType *matches, uint8_t enableCrossCheck,
                                       uint8_t enableDistanceRatioTest, int32_t lowesTestThresholdNumerator,
                                       int32_t lowesTestThresholdDenominator);
