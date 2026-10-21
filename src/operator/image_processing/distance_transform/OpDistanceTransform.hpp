/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OpDistanceTransform.hpp
 *
 * @brief Defines the private C++ Class for the DistanceTransform operation.
 */

#ifndef PVA_SOLUTIONS_PRIV_DISTANCETRANSFORM_HPP
#define PVA_SOLUTIONS_PRIV_DISTANCETRANSFORM_HPP

#include "IOperator.hpp"

#include "device/distance_transform_dev_param.h"

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Tensor.hpp>

#include <OpDistanceTransform.h>
#include <cupva_host_scheduling.h>
#include <nvcv/BorderType.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>

namespace pva_solutions::priv {

class DistanceTransform final : public IOperator
{
public:
    using ExecKey = std::tuple<cupva::impl::Context *>;

    explicit DistanceTransform(NVCVTensorRequirements *tensorRequirements, PVADistanceType distanceType);

    ~DistanceTransform();

    template<typename S>
    void operator()(S stream, const nvcv::Tensor &inImage, uint16_t maxDistance, const nvcv::Tensor &outDistance,
                    const nvcv::Tensor &outLabel);

private:
    NVCVTensorRequirements m_tensorRequirements;

    PVADistanceType m_distType;
    uint16_t m_maxDist;
    int32_t m_height;
    int32_t m_width;
    nvcv::TensorShape m_shape;

    uint16_t *m_hd;
    cupva::OffsetPointer<uint16_t> m_imgOp;
    cupva::OffsetPointer<uint16_t> m_distOp;
    cupva::OffsetPointer<uint16_t> m_labelOp;

    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_distanceTransformExecutables;
    static std::shared_ptr<cupva::Executable> GetDistanceTransformExec();

    static constexpr int32_t kMaxImageWidth{MAX_IMAGE_WIDTH};
    static constexpr int32_t kMaxImageHeight{MAX_IMAGE_HEIGHT};
    static constexpr int32_t kMinImageWidth{MIN_IMAGE_WIDTH};
    static constexpr int32_t kMinImageHeight{MIN_IMAGE_HEIGHT};
};

} // namespace pva_solutions::priv

// Internal implementation function for submit operations
// Used by macro-generated __cupva/__cuda wrappers
template<typename StreamType>
NVCVStatus DistanceTransformSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle inImage,
                                       uint16_t maxDistance, NVCVTensorHandle outDistance, NVCVTensorHandle outLabel);

#endif // PVA_SOLUTIONS_PRIV_DISTANCETRANSFORM_HPP
