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

#ifndef PVA_SOLUTIONS_PRIV_DOA_CAPON_ROI_HPP
#define PVA_SOLUTIONS_PRIV_DOA_CAPON_ROI_HPP

#include "IOperator.hpp"

#include <cupva_host.hpp>
#include <nvcv/Tensor.hpp>

#include <OpDoaCaponROI.h>
#include <PvaOperatorTypes.h>

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>

namespace pva_solutions::priv {

class DoaCaponROI final : public IOperator
{
public:
    using ExecKey = std::tuple<cupva::impl::Context *, bool, bool>;

    DoaCaponROI(const NVCVTensorRequirements *snapshotsReq, const NVCVTensorRequirements *steeringReq,
                const NVCVTensorRequirements *peakPowerReq, const NVCVTensorRequirements *azBinReq,
                const NVCVTensorRequirements *elBinReq, const PVADoaCaponROIParams &params);
    ~DoaCaponROI() override;

    template<typename S>
    void operator()(S stream, NVCVTensorHandle snapshotsHandle, NVCVTensorHandle steeringHandle,
                    NVCVTensorHandle peakPowerHandle, NVCVTensorHandle azBinHandle, NVCVTensorHandle elBinHandle);

private:
    struct DeviceFree
    {
        void operator()(float *ptr) const
        {
            cupva::mem::Free(ptr);
        }
    };

    void Bind(const NVCVTensorData &snapshots, const NVCVTensorData &steering, const NVCVTensorData &peakPower,
              const NVCVTensorData &azBin, const NVCVTensorData &elBin);
    cupvaCmd_t const *Command() const;
    static std::shared_ptr<cupva::Executable> GetExec(bool useFpCovMat, bool enableSubBinInterp);

    NVCVTensorRequirements m_snapshotsReq;
    NVCVTensorRequirements m_steeringReq;
    NVCVTensorRequirements m_peakPowerReq;
    NVCVTensorRequirements m_azBinReq;
    NVCVTensorRequirements m_elBinReq;

    cupva::OffsetPointer<int32_t> m_snapshotsOp;
    cupva::OffsetPointer<float> m_LinvRealOp;
    cupva::OffsetPointer<float> m_LinvImagOp;
    cupva::OffsetPointer<int32_t> m_svOp;
    cupva::OffsetPointer<float> m_peakPowerOp;
    cupva::OffsetPointer<float> m_azBinOp;
    cupva::OffsetPointer<float> m_elBinOp;
    // Part 1 -> Part 2 handoff of the packed L_inv real/imaginary components; operator-owned scratch.
    std::unique_ptr<float, DeviceFree> m_LinvReal;
    std::unique_ptr<float, DeviceFree> m_LinvImag;
    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;

    static std::mutex s_execMutex;
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_execs;
};

} // namespace pva_solutions::priv

template<typename StreamType>
NVCVStatus DoaCaponROISubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle snapshotsROI,
                                 NVCVTensorHandle steeringVectorsROI, NVCVTensorHandle peakPower,
                                 NVCVTensorHandle azimuthBinIndices, NVCVTensorHandle elevationBinIndices);

#endif // PVA_SOLUTIONS_PRIV_DOA_CAPON_ROI_HPP
