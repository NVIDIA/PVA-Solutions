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
 * @file OpDoaBartlettBeamforming.hpp
 *
 * @brief Defines the private C++ Class for the DOA Bartlett Beamforming operation.
 *
 * This operator performs DoA estimation using Bartlett beamforming and outputs target angles
 * and target index map tensors.
 *
 */

#ifndef PVA_SOLUTIONS_PRIV_BARTLETT_BEAMFORMING_HPP
#define PVA_SOLUTIONS_PRIV_BARTLETT_BEAMFORMING_HPP

#include "IOperator.hpp"

#include <cupva_host.hpp>
#include <nvcv/Tensor.hpp>

#include <OpDoaBartlettBeamforming.h>
#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>

#include <map>
#include <mutex>
#include <vector>

namespace pva_solutions::priv {

template<typename TI, typename TO>
class DoaBartlettBeamforming final : public IOperator
{
public:
    static constexpr PVADoaBartlettBeamformingParams kDefaultParams{
        true, // enableInterpolation
        true, // enablePowerOutput
        {
            // separableParams (disabled by default for backward compatibility)
            false,                            // enableSeparable
            PVA_BARTLETT_AZ_NEIGHBORHOOD_2,   // azNeighborhood (±2 bins, max allowed)
            PVA_BARTLETT_AZ_FROM_NEIGHBORHOOD // azSource
        }};

    using ExecDataKey = std::tuple<bool>;                         // (enableSeparable)
    using ExecKey     = std::tuple<bool, cupva::impl::Context *>; // (enableSeparable, context)

    explicit DoaBartlettBeamforming(const NVCVTensorRequirements *snapshotsTensorRequirements,
                                    const NVCVTensorRequirements *detectionListTensorRequirements,
                                    const NVCVTensorRequirements *ddmDopplerOffsetsTensorRequirements,
                                    const NVCVTensorRequirements *targetAnglesTensorRequirements,
                                    const NVCVTensorRequirements *targetIndexMapTensorRequirements,
                                    const PVADoaBartlettBeamformingParams *params);

    ~DoaBartlettBeamforming();

    template<typename S>
    void operator()(S stream, NVCVTensorHandle detectionCountHandle, NVCVTensorHandle snapshotsHandle,
                    NVCVTensorHandle steeringVectorsHandle, NVCVTensorHandle azimuthBinsHandle,
                    NVCVTensorHandle elevationBinsHandle, NVCVTensorHandle detectionListHandle,
                    NVCVTensorHandle ddmDopplerOffsetsHandle, NVCVTensorHandle targetCountHandle,
                    NVCVTensorHandle targetAnglesHandle, NVCVTensorHandle targetIndexMapHandle);

private:
    cupva::OffsetPointer<TI> m_snapshotsOp;
    cupva::OffsetPointer<int32_t> m_detectionCountOp;
    cupva::OffsetPointer<int32_t> m_targetCountOp;

    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;

    PVADoaBartlettBeamformingParams m_params;

    int16_t *m_steeringVectorsL2;
    int32_t m_steeringVectorsSize;

    // Output OffsetPointers for target angles and target index map
    cupva::OffsetPointer<float> m_outTargetAnglesOp;
    cupva::OffsetPointer<int32_t> m_outTargetIndexMapOp;
    int32_t m_maxDetections; // Maximum detections capacity

    // Stored tensor requirements for validation
    NVCVTensorRequirements m_snapshotsTensorRequirements;

    static std::mutex s_execMapMutex;
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_BartlettExecutables;
    static std::shared_ptr<cupva::Executable> GetBartlettExec(bool enableSeparable);
};

} // namespace pva_solutions::priv

template<typename StreamType>
NVCVStatus DoaBartlettBeamformingSubmitImpl(NVCVOperatorHandle handle, StreamType stream,
                                            NVCVTensorHandle detectionCountHandle, NVCVTensorHandle snapshotsHandle,
                                            NVCVTensorHandle steeringVectorsHandle, NVCVTensorHandle azimuthBinsHandle,
                                            NVCVTensorHandle elevationBinsHandle, NVCVTensorHandle detectionListHandle,
                                            NVCVTensorHandle ddmDopplerOffsetsHandle,
                                            NVCVTensorHandle targetCountHandle, NVCVTensorHandle targetAnglesHandle,
                                            NVCVTensorHandle targetIndexMapHandle);

#endif // PVA_SOLUTIONS_PRIV_BARTLETT_BEAMFORMING_HPP
