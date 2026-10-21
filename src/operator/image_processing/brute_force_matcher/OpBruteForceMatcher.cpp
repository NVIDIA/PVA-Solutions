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

#include "OpBruteForceMatcher.hpp"

#include "device/brute_force_matcher_param.h"

#include <TensorUtils.hpp>
#include <cupva_host_conversions.hpp>
#include <nvcv/Tensor.hpp>

#include <OpBruteForceMatcher.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>
#include <nvcv/TensorData.h>
#include <nvcv/TensorShape.h>

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <utility>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_brute_force_matcher_dev);

namespace pva_solutions::priv {

std::mutex BruteForceMatcher::s_execMapMutex;

std::map<BruteForceMatcher::ExecKey, std::weak_ptr<cupva::Executable>>
    BruteForceMatcher::s_bruteForceMatcherExecutables;

std::shared_ptr<cupva::Executable> BruteForceMatcher::GetBruteForceMatcherExec()
{
    static IOperator::ExecutableData sBruteForceMatcherExecData = {CUPVA_EXECUTABLE_DATA(pva_brute_force_matcher_dev),
                                                                   CUPVA_EXECUTABLE_SIZE(pva_brute_force_matcher_dev)};

    ExecKey execKey = std::make_tuple(cupva::Context::GetCurrent());
    return IOperator::GetExecutable<ExecKey>(sBruteForceMatcherExecData, execKey, s_bruteForceMatcherExecutables,
                                             s_execMapMutex);
}

BruteForceMatcher::BruteForceMatcher()
    : m_queryOp(cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM))
    , m_referenceOp(cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM))
    , m_matchesOp(cupva::OffsetPointer<uint32_t>::Create(cupva::MemType::DRAM))
    , m_prog{}
{
    m_exec    = GetBruteForceMatcherExec();
    auto prog = CmdProgram::Create(*m_exec.get());

    auto querySQDFHdl     = prog["querySQDFHdl"];
    auto referenceSQDFHdl = prog["referenceSQDFHdl"];
    auto matchesSQDFHdl   = prog["matchesSQDFHdl"];

    uint8_t *queryPtr     = (uint8_t *)prog["queryBuf"].ptr<uint8_t>();
    uint8_t *referencePtr = (uint8_t *)prog["referenceBuf"].ptr<uint8_t>();
    uint32_t *matchesPtr  = (uint32_t *)prog["matches_source_buf"].ptr<uint32_t>();

    SequenceDataFlow &queryDataFlow = prog.addDataFlowHead<SequenceDataFlow>();
    queryDataFlow.handler(querySQDFHdl)
        .addTransfer()
        .src(m_queryOp, PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH)
        .dst(queryPtr, PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH)
        .tile(PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH, 1);

    SequenceDataFlow &referenceDataFlow = prog.addDataFlowHead<SequenceDataFlow>();
    referenceDataFlow.handler(referenceSQDFHdl)
        .addTransfer()
        .src(m_referenceOp, PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH)
        .dst(referencePtr, PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH)
        .tile(PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH, 1);

    SequenceDataFlow &matchesDataFlow = prog.addDataFlowHead<SequenceDataFlow>();
    matchesDataFlow.handler(matchesSQDFHdl)
        .addTransfer()
        .src(matchesPtr, PVA_MAX_BF_DESCRIPTOR_COUNT * sizeof(PVABFMatchesType))
        .dst(m_matchesOp, sizeof(PVABFMatchesType))
        .tile(sizeof(PVABFMatchesType), 1);

    prog.compileDataFlows();

    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename S>
void BruteForceMatcher::operator()(S stream, PVABriefDescriptor *query, int32_t queryCount,
                                   PVABriefDescriptor *reference, int32_t referenceCount, int32_t maxMatchesPerQuery,
                                   PVABFMatchesType *matches, uint8_t enableCrossCheck, uint8_t enableDistanceRatioTest,
                                   int32_t lowesTestThresholdNumerator, int32_t lowesTestThresholdDenominator)
{
    if (maxMatchesPerQuery <= 0 || maxMatchesPerQuery > PVA_MAX_BF_MATCHES_PER_DESCRIPTOR)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "maxMatchesPerQuery must be between 1 and PVA_MAX_BF_MATCHES_PER_DESCRIPTOR");
    }
    if (enableCrossCheck != 0 && enableCrossCheck != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "enableCrossCheck must be 0 or 1");
    }
    if (enableCrossCheck == 1 && maxMatchesPerQuery != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "enableCrossCheck is 1, maxMatchesPerQuery must be 1");
    }
    if (enableDistanceRatioTest != 0 && enableDistanceRatioTest != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "enableDistanceRatioTest must be 0 or 1");
    }
    if (enableDistanceRatioTest == 1 && maxMatchesPerQuery != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "enableDistanceRatioTest is 1, maxMatchesPerQuery must be 1");
    }
    if (enableDistanceRatioTest == 1 && (lowesTestThresholdDenominator <= 0 || lowesTestThresholdNumerator <= 0))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "enableDistanceRatioTest is 1, lowesTestThresholdDenominator and "
                              "lowesTestThresholdNumerator must be larger than 0");
    }
    if (enableDistanceRatioTest == 1 && (queryCount <= 1 || referenceCount <= 1))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "enableDistanceRatioTest is 1, queryCount and referenceCount must be larger than 1");
    }

    if ((query == nullptr) || (reference == nullptr) || (matches == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Pointers must not be NULL");
    }
    if (queryCount <= 0 || queryCount > PVA_MAX_BF_DESCRIPTOR_COUNT)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "queryCount must be between 1 and PVA_MAX_BF_DESCRIPTOR_COUNT");
    }
    if (referenceCount <= 0 || referenceCount > PVA_MAX_BF_DESCRIPTOR_COUNT)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "referenceCount must be between 1 and PVA_MAX_BF_DESCRIPTOR_COUNT");
    }
    if (enableCrossCheck == 0 && enableDistanceRatioTest == 0 && maxMatchesPerQuery > referenceCount)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "enableCrossCheck is 0 and enableDistanceRatioTest is 0, maxMatchesPerQuery must be less "
                              "than or equal to referenceCount");
    }

    m_queryOp.setBase(reinterpret_cast<uint8_t *>(query));
    m_referenceOp.setBase(reinterpret_cast<uint8_t *>(reference));
    m_matchesOp.setBase(reinterpret_cast<uint32_t *>(matches));

    {
        cupvaParameter_t param;
        CupvaCmdProgramGetParameter(&m_prog, &param, "queryCount");
        CupvaParameterSetValueScalar(&param, &queryCount, sizeof(int32_t));
    }
    {
        cupvaParameter_t param;
        CupvaCmdProgramGetParameter(&m_prog, &param, "referenceCount");
        CupvaParameterSetValueScalar(&param, &referenceCount, sizeof(int32_t));
    }
    {
        cupvaParameter_t param;
        CupvaCmdProgramGetParameter(&m_prog, &param, "maxMatchesPerQuery");
        CupvaParameterSetValueScalar(&param, &maxMatchesPerQuery, sizeof(int32_t));
    }
    {
        cupvaParameter_t param;
        CupvaCmdProgramGetParameter(&m_prog, &param, "enableCrossCheck");
        CupvaParameterSetValueScalar(&param, &enableCrossCheck, sizeof(uint8_t));
    }
    {
        cupvaParameter_t param;
        CupvaCmdProgramGetParameter(&m_prog, &param, "enableDistanceRatioTest");
        CupvaParameterSetValueScalar(&param, &enableDistanceRatioTest, sizeof(uint8_t));
    }
    {
        cupvaParameter_t param;
        CupvaCmdProgramGetParameter(&m_prog, &param, "lowesTestThresholdNumerator");
        CupvaParameterSetValueScalar(&param, &lowesTestThresholdNumerator, sizeof(int32_t));
    }
    {
        cupvaParameter_t param;
        CupvaCmdProgramGetParameter(&m_prog, &param, "lowesTestThresholdDenominator");
        CupvaParameterSetValueScalar(&param, &lowesTestThresholdDenominator, sizeof(int32_t));
    }

    cupvaError_t err                = CupvaCmdProgramUpdateDataFlows(&m_prog);
    cupvaCmd_t const *submitCmds[1] = {&m_prog};

    if constexpr (std::is_same_v<S, cupvaStream_t>)
    {
        err = CupvaStreamSubmit(stream, submitCmds, nullptr, 1, CUPVA_IN_ORDER, -1, -1);
    }
    else if constexpr (std::is_same_v<S, cudaStream_t>)
    {
        if constexpr (kCupvaHasCudaStreamSupport)
        {
            err = CupvaCudaStreamSubmit(stream, submitCmds, nullptr, 1, CUPVA_IN_ORDER, -1, -1);
        }
        else
        {
            ThrowCudaStreamNotSupported();
        }
    }
    else
    {
        static_assert(!sizeof(S), "Unsupported stream type for BruteForceMatcher operator");
    }

    if (!(err == CUPVA_ERROR_NONE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Submit failed");
    }
}

BruteForceMatcher::~BruteForceMatcher()
{
    CupvaCmdDestroy(&m_prog);
}

} // namespace pva_solutions::priv

NVCVStatus pvaBruteForceMatcherCreate(NVCVOperatorHandle *handle)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::BruteForceMatcher());
    });
}

template<typename StreamType>
NVCVStatus BruteForceMatcherSubmitImpl(NVCVOperatorHandle handle, StreamType stream, PVABriefDescriptor *query,
                                       int32_t queryCount, PVABriefDescriptor *reference, int32_t referenceCount,
                                       int32_t maxMatchesPerQuery, PVABFMatchesType *matches, uint8_t enableCrossCheck,
                                       uint8_t enableDistanceRatioTest, int32_t lowesTestThresholdNumerator,
                                       int32_t lowesTestThresholdDenominator)
{
    return nvcv::ProtectCall([&] {
        pva_solutions::priv::ToDynamicRef<pva_solutions::priv::BruteForceMatcher>(handle)(
            stream, query, queryCount, reference, referenceCount, maxMatchesPerQuery, matches, enableCrossCheck,
            enableDistanceRatioTest, lowesTestThresholdNumerator, lowesTestThresholdDenominator);
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus BruteForceMatcherSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, PVABriefDescriptor *,
                                                               int32_t, PVABriefDescriptor *, int32_t, int32_t,
                                                               PVABFMatchesType *, uint8_t, uint8_t, int32_t, int32_t);
template NVCVStatus BruteForceMatcherSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, PVABriefDescriptor *,
                                                              int32_t, PVABriefDescriptor *, int32_t, int32_t,
                                                              PVABFMatchesType *, uint8_t, uint8_t, int32_t, int32_t);

// C++ overload implementations
NVCVStatus pvaBruteForceMatcherSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, PVABriefDescriptor *query,
                                      int32_t queryCount, PVABriefDescriptor *reference, int32_t referenceCount,
                                      int32_t maxMatchesPerQuery, PVABFMatchesType *matches, uint8_t enableCrossCheck,
                                      uint8_t enableDistanceRatioTest, int32_t lowesTestThresholdNumerator,
                                      int32_t lowesTestThresholdDenominator)
{
    return BruteForceMatcherSubmitImpl(handle, stream, query, queryCount, reference, referenceCount, maxMatchesPerQuery,
                                       matches, enableCrossCheck, enableDistanceRatioTest, lowesTestThresholdNumerator,
                                       lowesTestThresholdDenominator);
}

NVCVStatus pvaBruteForceMatcherSubmit(NVCVOperatorHandle handle, cudaStream_t stream, PVABriefDescriptor *query,
                                      int32_t queryCount, PVABriefDescriptor *reference, int32_t referenceCount,
                                      int32_t maxMatchesPerQuery, PVABFMatchesType *matches, uint8_t enableCrossCheck,
                                      uint8_t enableDistanceRatioTest, int32_t lowesTestThresholdNumerator,
                                      int32_t lowesTestThresholdDenominator)
{
    return BruteForceMatcherSubmitImpl(handle, stream, query, queryCount, reference, referenceCount, maxMatchesPerQuery,
                                       matches, enableCrossCheck, enableDistanceRatioTest, lowesTestThresholdNumerator,
                                       lowesTestThresholdDenominator);
}
