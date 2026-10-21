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
 * @file TargetProcessing.hpp
 *
 * @brief Defines the private C++ Class for the TARGET_PROCESSING component.
 */

#ifndef PVA_SOLUTIONS_PRIV_TARGET_PROCESSING_HPP
#define PVA_SOLUTIONS_PRIV_TARGET_PROCESSING_HPP

#include "IOperator.hpp"

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Tensor.hpp>

#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>

namespace pva_solutions::priv {

template<typename TI, typename TO>
class TargetProcessing final : public IOperator
{
public:
    // Default params Target Processing
    // Note: RV decoupling parameters should be computed from radar configuration on the host
    static constexpr PVATargetProcessingParams kDefaultParams{
        // Standard parameters
        1.0f,  // wavelength (meters)
        1.0f,  // sweepTime (seconds)
        1.0f,  // rangeResolution (meters)
        256,   // numChirps
        false, // fftShiftedDoppler
        false, // enableRvDecoupling (disabled by default, requires proper constants)
        false, // enableQuadraticInterpolation (disabled - requires NCI tensor input)
        // Base FMCW parameters for RV decoupling (derived params computed internally)
        10e6f,   // fs - sampling frequency (Hz)
        256,     // nofSamples - samples per chirp
        100e-6f, // PRI - pulse repetition interval (seconds)
        3.6f,    // deltaV - velocity resolution (km/h)
        0.0f,    // contR - range coupling constant
        0.0f,    // contVFast - velocity fast coupling constant
        0.0f,    // contVSlow - velocity slow coupling constant
        // Coordinate system options (automotive radar convention)
        1, // longitudinalAxis: 1 = Y_FORWARD (Y is forward, X is lateral)
        1  // groundProjection: 1 = GROUND_PLANE (scale by cos(elevation))
    };

    using ExecKey = std::tuple<cupva::impl::Context *>;

    explicit TargetProcessing(NVCVTensorRequirements const *const detectionListTensorRequirements,
                              NVCVTensorRequirements const *const targetIndexMapTensorRequirements,
                              NVCVTensorRequirements const *const targetAnglesTensorRequirements,
                              NVCVTensorRequirements const *const ddmDopplerOffsetsTensorRequirements,
                              NVCVTensorRequirements const *const targetListTensorRequirements,
                              NVCVTensorRequirements const *const nciTensorRequirements,
                              PVATargetProcessingParams const *const params);

    ~TargetProcessing();

    void updateDataFlows(const nvcv::Tensor &targetCountTensor, const nvcv::Tensor &detectionCountTensor,
                         const nvcv::Tensor &detectionListTensor, const nvcv::Tensor &targetIndexMapTensor,
                         const nvcv::Tensor &targetAnglesTensor, const nvcv::Tensor &ddmDopplerOffsetsTensor,
                         const nvcv::Tensor &targetListTensor, const nvcv::Tensor *nciTensor);

    cupvaCmd_t m_prog;

private:
    NVCVTensorRequirements m_detectionListTensorRequirements;
    NVCVTensorRequirements m_ddmDopplerOffsetsTensorRequirements;
    NVCVTensorRequirements m_targetListTensorRequirements;
    PVATargetProcessingParams m_params;

    int32_t m_maxNumOfTargets;

    cupva::OffsetPointer<int32_t> m_targetIndexMapOp;
    cupva::OffsetPointer<float> m_targetAnglesOp;
    NVCVTensorRequirements m_targetIndexMapTensorRequirements;
    NVCVTensorRequirements m_targetAnglesTensorRequirements;

    // Device memory pointers
    cupva::OffsetPointer<int32_t> m_detectionCountOp;
    cupva::OffsetPointer<TI> m_detectionListOp;
    cupva::OffsetPointer<float> m_ddmDopplerOffsetsOp;
    cupva::OffsetPointer<TO> m_targetListOp;
    cupva::OffsetPointer<int32_t> m_targetCountOp;

    // NCI source pointer for SQDF DRAM→L2 transfer
    cupva::OffsetPointer<uint8_t> m_nciSrcOp;

    // NCI L2 buffer for quadratic interpolation
    uint32_t *m_nciL2;         // Pointer to padded NCI buffer in L2
    int32_t m_nciL2Size;       // L2 buffer size in bytes
    int32_t m_nciWidth;        // NCI width (Doppler bins)
    int32_t m_nciHeight;       // NCI height (Range bins)
    int32_t m_nciPaddedWidth;  // Padded width (width + 2)
    int32_t m_nciPaddedHeight; // Padded height (height + 2)

    std::shared_ptr<cupva::Executable> m_exec;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_TargetProcessingExecutables;
    static std::shared_ptr<cupva::Executable> GetTargetProcessingExec();
};

} // namespace pva_solutions::priv
#endif // PVA_SOLUTIONS_PRIV_TARGET_PROCESSING_HPP
