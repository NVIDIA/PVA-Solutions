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
 * @file OpDoaDAFFT.hpp
 *
 * @brief Private C++ implementation of the DoA DA FFT operator.
 */

#ifndef PVA_SOLUTIONS_PRIV_DOA_DA_FFT_HPP
#define PVA_SOLUTIONS_PRIV_DOA_DA_FFT_HPP

#include "IOperator.hpp"

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Tensor.hpp>

#include <OpDoaDAFFT.h>
#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>

#include <map>
#include <mutex>
#include <vector>

namespace pva_solutions::priv {

template<typename TI, typename TO>
class DOA final : public IOperator
{
public:
    static constexpr int32_t kDefaultTensorCount{5};
    static constexpr int32_t kDefaultTensorRank{2};
    static constexpr int32_t kNciCpyCmdCount{3};

    using ExecKey = std::tuple<cupva::impl::Context *>;

    explicit DOA(const NVCVTensorRequirements **inTensorRequirements,
                 const NVCVTensorRequirements *outTensorRequirements, int inTensorCount, const PVARadarGP *GP);

    ~DOA();

    template<typename S>
    void operator()(S stream, NVCVTensorHandle *inHandles, NVCVTensorHandle outHandle, int32_t inTensorCount);

private:
    std::vector<NVCVTensorRequirements> m_inTensorRequirements;
    NVCVTensorRequirements m_outTensorRequirements;
    std::vector<cupva::OffsetPointer<TI>> m_srcOps;
    cupva::OffsetPointer<TO> m_dstOp;
    uint32_t *m_nciFinalPaddedDev;
    int32_t m_nciFinalWidth;
    int32_t m_nciFinalHeight;
    int32_t m_nciFinalWidthInBytes;
    int32_t m_nciFinalPaddedWidth;
    int32_t m_nciFinalPaddedHeight;
    int32_t m_nciFinalPaddedWidthInBytes;

    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;
    cupvaCmd_t m_nciCpyCmd[kNciCpyCmdCount];

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_DOAExecutables;
    static std::shared_ptr<cupva::Executable> GetDOAExec();
};

} // namespace pva_solutions::priv

// Internal implementation function for submit operations
// Used by macro-generated __cupva/__cuda wrappers
template<typename StreamType>
NVCVStatus DoaDAFFTSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle *inHandles,
                              NVCVTensorHandle outHandle, int inTensorCount);

#endif // PVA_SOLUTIONS_PRIV_DOA_DA_FFT_HPP
