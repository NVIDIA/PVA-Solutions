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
 * @file OpRangeFFT.hpp
 *
 * @brief Defines the private C++ Class for the Range_FFT operation.
 */

#ifndef PVA_SOLUTIONS_PRIV_RANGEFFT_HPP
#define PVA_SOLUTIONS_PRIV_RANGEFFT_HPP

#include "IOperator.hpp"

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Tensor.hpp>

#include <OpRangeFFT.h>
#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>

namespace pva_solutions::priv {

template<typename TI, typename TO>
class RangeFFT final : public IOperator
{
public:
    static constexpr PVARangeFFTParams kDefaultParams{PVA_BATCH_FFT_WINDOW_HANNING};

    using ExecKey = std::tuple<cupva::impl::Context *>;

    explicit RangeFFT(NVCVTensorRequirements const *const inTensorRequirements,
                      NVCVTensorRequirements const *const winTensorRequirements,
                      NVCVTensorRequirements const *const outTensorRequirements, PVARangeFFTParams const *const params);

    ~RangeFFT();

    template<typename S>
    void operator()(S stream, const nvcv::Tensor &in, const nvcv::Tensor &win, const nvcv::Tensor &out);

private:
    NVCVTensorRequirements m_inTensorRequirements;
    NVCVTensorRequirements m_winTensorRequirements;
    NVCVTensorRequirements m_outTensorRequirements;
    PVARangeFFTParams m_params;
    bool m_inputIsComplex;

    // FFT buffers
    cupva::OffsetPointer<TI> m_srcOp;
    cupva::OffsetPointer<TI> m_winOp;
    cupva::OffsetPointer<TO> m_dstOp;
    int32_t *m_twiddle_d;

    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_rangeFFTExecutables;
    static std::shared_ptr<cupva::Executable> GetRangeFFTExec();

    bool isSupportedFFTSize(int32_t size);
    void generateRealFFTTwiddle(int32_t n, int32_t *twiddle);
};

} // namespace pva_solutions::priv

// Internal implementation function for submit operations
// Used by macro-generated __cupva/__cuda wrappers
template<typename StreamType>
NVCVStatus RangeFFTSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle const in,
                              NVCVTensorHandle const win, NVCVTensorHandle out);

#endif // PVA_SOLUTIONS_PRIV_RANGEFFT_HPP
