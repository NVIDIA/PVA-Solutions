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
 * @file OpNci.hpp
 *
 * @brief Defines the private C++ Class for the NCI operation.
 */

#ifndef PVA_SOLUTIONS_PRIV_NCI_HPP
#define PVA_SOLUTIONS_PRIV_NCI_HPP

#include "IOperator.hpp"

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Tensor.hpp>

#include <OpNci.h>
#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>

namespace pva_solutions::priv {

template<typename TI, typename TO>
class Nci final : public IOperator
{
public:
    using ExecKey = std::tuple<cupva::impl::Context *>;

    explicit Nci(NVCVTensorRequirements const *const inTensorRequirements,
                 NVCVTensorRequirements const *const outNciRxTensorRequirements,
                 NVCVTensorRequirements const *const outNciFinalTensorRequirements,
                 NVCVTensorRequirements const *const outNoiseEstimateTensorRequirements, int32_t repeatFold,
                 uint32_t noiseEstimateEnabled, PVAFFTLayout inputLayout);

    ~Nci();

    template<typename S>
    void operator()(S stream, const nvcv::Tensor &in, const nvcv::Tensor &outNciRx, const nvcv::Tensor &outNciFinal,
                    const nvcv::Tensor &outNoiseEstimate);

    bool isNoiseEstimateEnabled() const;

private:
    NVCVTensorRequirements m_inTensorRequirements;
    NVCVTensorRequirements m_outNciRxTensorRequirements;
    NVCVTensorRequirements m_outNciFinalTensorRequirements;
    NVCVTensorRequirements m_outNoiseEstimateTensorRequirements;

    // NCI buffers
    cupva::OffsetPointer<TI> m_fft2dOp;
    cupva::OffsetPointer<TO> m_NciRxOp;
    cupva::OffsetPointer<TO> m_NciFinalOp;
    cupva::OffsetPointer<TO> m_noiseEstimateOp;

    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_nciExecutables;
    static std::shared_ptr<cupva::Executable> GetNciExec();

    bool isNumChannelsValid(int32_t numChannels) const;
    bool isNumSamplesValid(int32_t numSamples, int32_t numChannels, int32_t numRangeBins, int32_t NrPerTile) const;
    bool isNumRangeBinsValid(int32_t numRangeBins, int32_t NrPerTile) const;
    bool isNumDopplerFoldsValid(int32_t numSamples) const;

    std::pair<int32_t, int32_t> getTileSize(int32_t numChannels, int32_t inWidth, int32_t inHeight) const;

    uint32_t m_numDopplerFolds;
    bool m_noiseEstimateEnabled;
    PVAFFTLayout m_inputLayout;
};

} // namespace pva_solutions::priv

// Internal implementation function for submit operations
// Used by macro-generated __cupva/__cuda wrappers
template<typename StreamType>
NVCVStatus NciSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle const *inHandles,
                         NVCVTensorHandle *outHandles);

#endif // PVA_SOLUTIONS_PRIV_NCI_HPP
