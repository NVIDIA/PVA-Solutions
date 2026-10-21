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

/**
 * @file OpBundleAdjustment.hpp
 *
 * @brief Defines the private C++ Class for the Bundle Adjustment operation.
 */

#ifndef PVA_SOLUTIONS_PRIV_BUNDLE_ADJUSTMENT_HPP
#define PVA_SOLUTIONS_PRIV_BUNDLE_ADJUSTMENT_HPP

#include "IOperator.hpp"

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Tensor.hpp>

#include <OpBundleAdjustment.h>
#include <cupva_host_scheduling.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>

namespace pva_solutions::priv {

class BundleAdjustment final : public IOperator
{
public:
    using ExecKey = std::tuple<cupva::impl::Context *>;

    explicit BundleAdjustment(NVCVTensorRequirements const *rigPoses, NVCVTensorRequirements const *points3d,
                              NVCVTensorRequirements const *measurements, NVCVTensorRequirements const *cameraBitmask,
                              NVCVTensorRequirements const *rig2cam, NVCVTensorRequirements const *updatedRigPoses,
                              NVCVTensorRequirements const *updatedPoints3d, NVCVTensorRequirements const *finalCost,
                              NVCVTensorRequirements const *iterationsUsed, PvaBundleAdjustmentConfig const *config);

    ~BundleAdjustment();

    template<typename StreamType>
    void operator()(StreamType stream, const nvcv::Tensor &rigPoses, const nvcv::Tensor &points3d,
                    const nvcv::Tensor &measurements, const nvcv::Tensor &cameraBitmask, const nvcv::Tensor &rig2cam,
                    const nvcv::Tensor &updatedRigPoses, const nvcv::Tensor &updatedPoints3d,
                    const nvcv::Tensor &finalCost, const nvcv::Tensor &iterationsUsed,
                    PvaBundleAdjustmentParams const *params);

private:
    NVCVTensorRequirements m_rigPosesTensorRequirements;
    NVCVTensorRequirements m_points3dTensorRequirements;
    NVCVTensorRequirements m_measurementsTensorRequirements;
    NVCVTensorRequirements m_cameraBitmaskTensorRequirements;
    NVCVTensorRequirements m_rig2camTensorRequirements;
    NVCVTensorRequirements m_updatedRigPosesTensorRequirements;
    NVCVTensorRequirements m_updatedPoints3dTensorRequirements;
    NVCVTensorRequirements m_finalCostTensorRequirements;
    NVCVTensorRequirements m_iterationsUsedTensorRequirements;

    int32_t m_activeCameraCount;
    int32_t m_activeCameraBegin;

    cupva::OffsetPointer<float> m_rigPosesOp;
    cupva::OffsetPointer<float> m_points3dOp;
    cupva::OffsetPointer<float> m_measurementsOp;
    cupva::OffsetPointer<int32_t> m_cameraBitmaskOp;
    cupva::OffsetPointer<float> m_rig2camOp;
    cupva::OffsetPointer<float> m_updatedRigPosesOp;
    cupva::OffsetPointer<float> m_updatedPoints3dOp;
    cupva::OffsetPointer<float> m_finalCostOp;
    cupva::OffsetPointer<int32_t> m_iterationsUsedOp;

    // Device-owned scratch sized for kMaxTrackCount; trackCount is only known at Submit.
    void *m_tpcScratch;
    void *m_tpScratch;
    // SoA LM point state (current/tentative); the device swaps them on an accepted step.
    void *m_points3dScratch;
    void *m_points3dScratch2;

    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_bundleAdjustmentExecutables;
    static std::shared_ptr<cupva::Executable> GetBundleAdjustmentExec();
};

} // namespace pva_solutions::priv

// Internal implementation function for submit operations
// Used by macro-generated __cupva/__cuda wrappers
template<typename StreamType>
NVCVStatus BundleAdjustmentSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle rigPoses,
                                      NVCVTensorHandle points3d, NVCVTensorHandle measurements,
                                      NVCVTensorHandle cameraBitmask, NVCVTensorHandle rig2cam,
                                      NVCVTensorHandle updatedRigPoses, NVCVTensorHandle updatedPoints3d,
                                      NVCVTensorHandle finalCost, NVCVTensorHandle iterationsUsed,
                                      PvaBundleAdjustmentParams const *params);

#endif // PVA_SOLUTIONS_PRIV_BUNDLE_ADJUSTMENT_HPP
