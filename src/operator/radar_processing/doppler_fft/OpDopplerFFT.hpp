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
 * @file OpDopplerFFT.hpp
 *
 * @brief Defines the private C++ Class for the Doppler_FFT operation.
 */

#ifndef PVA_SOLUTIONS_PRIV_DOPPLERFFT_HPP
#define PVA_SOLUTIONS_PRIV_DOPPLERFFT_HPP

#include "IOperator.hpp"

#include "device/doppler_fft_param.h"

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Tensor.hpp>

#include <OpDopplerFFT.h>
#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>

namespace pva_solutions::priv {

template<typename TI, typename TO>
class DopplerFFT final : public IOperator
{
public:
    static constexpr PVADopplerFFTParams kDefaultParams{PVA_BATCH_FFT_WINDOW_HANNING, PVA_FFT_LAYOUT_DOPPLER_RX_RANGE,
                                                        PVA_FFT_LAYOUT_RANGE_RX_DOPPLER};

    using ExecKey = std::tuple<bool, cupva::impl::Context *>;

    explicit DopplerFFT(NVCVTensorRequirements const *const inTensorRequirements,
                        NVCVTensorRequirements const *const winTensorRequirements,
                        NVCVTensorRequirements const *const outTensorRequirements,
                        PVADopplerFFTParams const *const params);

    ~DopplerFFT();

    template<typename S>
    void operator()(S stream, const nvcv::Tensor &in, const nvcv::Tensor &win, const nvcv::Tensor &out);

private:
    NVCVTensorRequirements m_inTensorRequirements;
    NVCVTensorRequirements m_winTensorRequirements;
    NVCVTensorRequirements m_outTensorRequirements;
    PVADopplerFFTParams m_params;

    // FFT buffers
    cupva::OffsetPointer<TI> m_srcOp;
    cupva::OffsetPointer<TI> m_winOp;
    cupva::OffsetPointer<TO> m_dstOp;
    int32_t *m_twiddle_d;
    int32_t *m_transposeBuffer_d; // DRAM fallback when L2 insufficient for RANGE_DOPPLER_RX

    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting, keyed by (isRxPackedLayout, context)
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_dopplerFFTExecutables;
    static std::shared_ptr<cupva::Executable> GetDopplerFFTExec(bool isRxPackedLayout);

    bool isSupportedFFTSize(int32_t size);
    void generateComplexFFTTwiddleIntraStage(int32_t *twiddle, std::vector<int32_t> &twiddle_interleaved,
                                             FFTFactorization &factorization, int32_t radix);
    void generateComplexFFTTwiddleRadix5(int32_t *twiddle, std::vector<int32_t> &twiddle_interleaved,
                                         FFTFactorization &factorization);
    void generateComplexFFTTwiddleRadix2(int32_t *twiddle, std::vector<int32_t> &twiddle_interleaved,
                                         FFTFactorization &factorization);
    void generateComplexFFTTwiddleRadix3(int32_t *twiddle, std::vector<int32_t> &twiddle_interleaved,
                                         FFTFactorization &factorization);
};

} // namespace pva_solutions::priv

// Internal implementation function for submit operations
// Used by macro-generated __cupva/__cuda wrappers
template<typename StreamType>
NVCVStatus DopplerFFTSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle const in,
                                NVCVTensorHandle const win, NVCVTensorHandle out);

#endif // PVA_SOLUTIONS_PRIV_DOPPLERFFT_HPP
