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
 * @file OpDoaAngleFFT.hpp
 *
 * @brief Defines the private C++ Class for the DOA Angle FFT operation.
 *
 * This operator performs DoA estimation and outputs target angles and target index maps as tensors.
 *
 */

#ifndef PVA_SOLUTIONS_PRIV_DOA_ANGLE_FFT_HPP
#define PVA_SOLUTIONS_PRIV_DOA_ANGLE_FFT_HPP

#include "IOperator.hpp"

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Tensor.hpp>

#include <OpDoaAngleFFT.h>
#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>

namespace pva_solutions::priv {

template<typename TI, typename TO>
class DoaAngleFFT final : public IOperator
{
public:
    static constexpr PVADoaAngleFFTParams kDefaultParams{
        16,                                     // numVirtualAzimuthElements
        4,                                      // numVirtualElevationElements
        PVA_ANGLE_ESTIMATION_PRECISION_DEFAULT, // angleEstimationPrecision
        false,                                  // enableLocalPeakDetection
    };

    using ExecKey = std::tuple<cupva::impl::Context *>;

    explicit DoaAngleFFT(NVCVTensorRequirements const *const inSnapshotsTensorRequirements,
                         NVCVTensorRequirements const *const inDetectionListTensorRequirements,
                         NVCVTensorRequirements const *const inDdmDopplerOffsetsTensorRequirements,
                         NVCVTensorRequirements const *const outTargetAnglesTensorRequirements,
                         NVCVTensorRequirements const *const outTargetIndexMapTensorRequirements,
                         PVADoaAngleFFTParams const *const params);

    ~DoaAngleFFT();

    template<typename S>
    void operator()(S stream, const nvcv::Tensor &inDetectionCountTensor, const nvcv::Tensor &inSnapshotsTensor,
                    const nvcv::Tensor &inDetectionListTensor, const nvcv::Tensor &inDdmDopplerOffsetsTensor,
                    const nvcv::Tensor &outTargetCountTensor, const nvcv::Tensor &outTargetAnglesTensor,
                    const nvcv::Tensor &outTargetIndexMapTensor);

private:
    NVCVTensorRequirements m_inSnapshotsTensorRequirements;
    PVADoaAngleFFTParams m_params;

    // FFT buffers
    cupva::OffsetPointer<int32_t> m_inDetectionCountOp;
    cupva::OffsetPointer<TI> m_inSnapshotsOp;
    cupva::OffsetPointer<int32_t> m_outTargetCountOp;
    int32_t *m_twiddle_d;
    int32_t *m_window_d;
    int32_t *m_angleBins_d;
    cupva::OffsetPointer<float> m_outTargetAnglesOp;
    cupva::OffsetPointer<int32_t> m_outTargetIndexMapOp;

    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_DoaAngleFFTExecutables;
    static std::shared_ptr<cupva::Executable> GetDoaAngleFFTExec();

    bool isSupportedFFTSize(int32_t size, bool isAzimuth);
    void generateComplexFFTTwiddle(int32_t n, int32_t *twiddle);
    void generateHannningWindowCoefficients(int32_t numVirtualAzimuthElements, int32_t numVirtualElevationElements,
                                            int32_t pitch, int32_t *window);
};

} // namespace pva_solutions::priv

// Internal implementation function for submit operations
// Used by macro-generated __cupva/__cuda wrappers
template<typename StreamType>
NVCVStatus DoaAngleFFTSubmitImpl(NVCVOperatorHandle handle, StreamType stream,
                                 NVCVTensorHandle const inDetectionCountTensor,
                                 NVCVTensorHandle const inSnapshotsTensor, NVCVTensorHandle const inDetectionListTensor,
                                 NVCVTensorHandle const inDdmDopplerOffsetsTensor,
                                 NVCVTensorHandle outTargetCountTensor, NVCVTensorHandle outTargetAnglesTensor,
                                 NVCVTensorHandle outTargetIndexMapTensor);
#endif // PVA_SOLUTIONS_PRIV_DOA_ANGLE_FFT_HPP
