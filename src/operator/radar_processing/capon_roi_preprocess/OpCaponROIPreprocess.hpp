/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PVA_SOLUTIONS_PRIV_CAPON_ROI_PREPROCESS_HPP
#define PVA_SOLUTIONS_PRIV_CAPON_ROI_PREPROCESS_HPP

#include "IOperator.hpp"

#include <cupva_host.hpp>
#include <nvcv/Tensor.hpp>

#include <OpCaponROIPreprocess.h>
#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>

#include <map>
#include <mutex>
#include <tuple>

namespace pva_solutions::priv {

class CaponROIPreprocess final : public IOperator
{
public:
    /// Device kernels: direct-DFT coarse DoA with the dot-product projection, direct-DFT with the general
    /// projection paths, batched FFT-64 coarse DoA (dot-product projection). The constructor picks one from
    /// UseFftCoarseDoA and UseDotProductProjection (OpCaponROIPreprocess.cpp).
    enum class Kernel : int32_t
    {
        kDft,
        kDftGeneral,
        kFft
    };
    using ExecKey = std::tuple<cupva::impl::Context *, Kernel>; ///< context, kernel

    CaponROIPreprocess(const NVCVTensorRequirements *wazReqs, const NVCVTensorRequirements *welReqs,
                       const NVCVTensorRequirements *detectionCountReqs, const NVCVTensorRequirements *detDopplersReqs,
                       const NVCVTensorRequirements *elementSnapshotsReqs,
                       const NVCVTensorRequirements *rangeDopplerPlanesReqs,
                       const NVCVTensorRequirements *steeringLutReqs, const NVCVTensorRequirements *snapshotsROIReqs,
                       const NVCVTensorRequirements *steeringVectorsROIReqs, const PVACaponROIPreprocessParams *params);

    ~CaponROIPreprocess();

    template<typename S>
    void operator()(S stream, NVCVTensorHandle wazHandle, NVCVTensorHandle welHandle,
                    NVCVTensorHandle detectionCountHandle, NVCVTensorHandle detDopplersHandle,
                    NVCVTensorHandle elementSnapshotsHandle, NVCVTensorHandle rangeDopplerPlanesHandle,
                    NVCVTensorHandle steeringLutHandle, NVCVTensorHandle snapshotsROIHandle,
                    NVCVTensorHandle steeringVectorsROIHandle);

private:
    cupva::OffsetPointer<float> m_wazOp;
    cupva::OffsetPointer<float> m_elemOp;
    cupva::OffsetPointer<int32_t> m_lutOp;
    cupva::OffsetPointer<int32_t> m_snapOp;

    cupva::OffsetPointer<int32_t> m_snapOp1;
    cupva::OffsetPointer<int32_t> m_snapOp2;
    cupva::OffsetPointer<int32_t> m_snapOp3;
    cupva::OffsetPointer<int32_t> m_svOp;
    cupva::OffsetPointer<float> m_planeOp;    ///< rangeDopplerPlanes, one row per detection
    cupva::OffsetPointer<int32_t> m_detDopOp; ///< detDopplers

    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;
    bool m_useFftCoarseDoA  = false; ///< kernel chosen by UseFftCoarseDoA: batched FFT-64 or direct DFT
    Kernel m_kernel         = Kernel::kDft;
    bool m_vdFold           = false; ///< DDM decode folded into the projection twiddles (BuildHostTwiddle16)
    bool m_useDotProjection = false; ///< dot-product projection (UseDotProductProjection)

    int16_t *m_waz16Scratch  = nullptr;
    int16_t *m_wazT16Scratch = nullptr;
    int16_t *m_wel16Scratch  = nullptr;
    float *m_wazGenScratch   = nullptr; ///< generated fp32 DFT az table (useCustomTwiddles == false)

    PVACaponROIPreprocessParams m_params;

    /* The eight tensors in API order, and the requirements Create validated for each. */
    enum TensorIndex : int32_t
    {
        kWaz,
        kWel,
        kCount,
        kDetDop,
        kElem,
        kPlane,
        kLut,
        kSnap,
        kSv,
        kNumTensors
    };
    static constexpr const char *kTensorNames[kNumTensors] = {
        "waz",         "wel",          "detectionCount",    "detDopplers", "elementSnapshots", "rangeDopplerPlanes",
        "steeringLut", "snapshotsROI", "steeringVectorsROI"};
    NVCVTensorRequirements m_reqs[kNumTensors]{};

    void ValidateParams();
    void ValidateRequirements(const NVCVTensorRequirements *const reqs[kNumTensors]);

    void FreeScratch() noexcept;

    void BuildHostTwiddle16(const float *wazHost, const float *welHost);

    static std::mutex s_execMapMutex;
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_execs;
    static std::shared_ptr<cupva::Executable> GetExec(Kernel kernel);
};

} // namespace pva_solutions::priv

template<typename StreamType>
NVCVStatus CaponROIPreprocessSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle waz,
                                        NVCVTensorHandle wel, NVCVTensorHandle detectionCount,
                                        NVCVTensorHandle detDopplers, NVCVTensorHandle elementSnapshots,
                                        NVCVTensorHandle rangeDopplerPlanes, NVCVTensorHandle steeringLut,
                                        NVCVTensorHandle snapshotsROI, NVCVTensorHandle steeringVectorsROI);

#endif // PVA_SOLUTIONS_PRIV_CAPON_ROI_PREPROCESS_HPP
