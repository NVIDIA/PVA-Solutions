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
 * @file OpCannyEdgeDetector.hpp
 *
 * @brief Defines the private C++ Class for the CannyEdgeDetector operation.
 */

#ifndef PVA_SOLUTIONS_PRIV_CANNY_EDGE_DETECTOR_HPP
#define PVA_SOLUTIONS_PRIV_CANNY_EDGE_DETECTOR_HPP

#include <IOperator.hpp>
#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Tensor.hpp>

#include <OpCannyEdgeDetector.h>
#include <cupva_host_scheduling.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>

namespace pva_solutions::priv {

class CannyEdgeDetector final : public IOperator
{
public:
    using ExecKey = std::tuple<cupva::impl::Context *>;
    explicit CannyEdgeDetector(const NVCVTensorRequirements *imageParams, const int32_t gradientSize);
    ~CannyEdgeDetector();

    template<typename S>
    void operator()(S stream, cupvaCmdStatus_t *cmdStatus, const nvcv::Tensor &image, const nvcv::Tensor &edgeMap,
                    const int32_t thresholdStrong, const int32_t thresholdWeak);

private:
    NVCVTensorRequirements m_imageTensorRequirements;

    int32_t m_width;     // image width
    int32_t m_height;    // image height
    int32_t m_linePitch; // image linepitch

    cupva::OffsetPointer<uint8_t> m_imageOp;
    cupva::OffsetPointer<uint8_t> m_edgeMapOp;
    int32_t *m_gradientIntensityDevPtr;
    int8_t *m_edgeDirDevPtr;
    uint8_t *m_edgeTypeDevPtr;
    uint16_t *m_bufDevPtr;

    cupvaCmd_t m_prog;
    std::shared_ptr<cupva::Executable> m_exec;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_CannyEdgeDetectorExecutables;
    static std::shared_ptr<cupva::Executable> GetCannyEdgeDetectorExec();
};

} // namespace pva_solutions::priv

// Internal implementation function for submit operations
// Used by macro-generated __cupva/__cuda wrappers
template<typename StreamType>
NVCVStatus CannyEdgeDetectorSubmitImpl(NVCVOperatorHandle handle, StreamType stream, cupvaCmdStatus_t *cmdStatus,
                                       NVCVTensorHandle image, NVCVTensorHandle edgeMap, const int32_t thresholdStrong,
                                       const int32_t thresholdWeak);

#endif // PVA_SOLUTIONS_PRIV_CANNY_EDGE_DETECTOR_HPP
