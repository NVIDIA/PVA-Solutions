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
 * @file OpLidarHistogramConversion.hpp
 *
 * @brief Defines the private C++ Class for the LidarHistogramConversion operation.
 */

#ifndef PVA_SOLUTIONS_PRIV_LIDARHISTOGRAMCONVERSION_HPP
#define PVA_SOLUTIONS_PRIV_LIDARHISTOGRAMCONVERSION_HPP

#include "IOperator.hpp"

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Tensor.hpp>

#include <OpLidarHistogramConversion.h>
#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>

#include <cstdint>

namespace pva_solutions::priv {

class LidarHistogramConversion final : public IOperator
{
public:
    using ExecKey = std::tuple<cupva::impl::Context *>;

    explicit LidarHistogramConversion(NVCVTensorRequirements const *const histogramRequirements,
                                      PvaLidarHistogramSpec const *const histogramSpec);

    ~LidarHistogramConversion();

    template<typename S>
    void operator()(S stream, const nvcv::Tensor &histogram, const nvcv::Tensor &rangeBias,
                    const nvcv::Tensor &xyzCalibration, const nvcv::Tensor &range, const nvcv::Tensor &xyz,
                    const nvcv::Tensor &reflectance, PvaLidarHistogramParams const *const params);

    void setParameters(PvaLidarHistogramParams const *const params);

private:
    NVCVTensorRequirements m_histogramRequirements;
    PvaLidarHistogramSpec m_histogramSpec;
    int32_t m_height;
    int32_t m_width;
    int32_t m_channels;
    int32_t m_histogramLinePitch;
    cupva::OffsetPointer<uint8_t> m_histogramOp;
    cupva::OffsetPointer<float> m_rangeBiasOp;
    cupva::OffsetPointer<float> m_xyzCalibrationOp;
    cupva::OffsetPointer<float> m_rangeOp;
    cupva::OffsetPointer<float> m_xyzOp;
    cupva::OffsetPointer<float> m_reflectanceOp;
    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_lidarHistogramConversionExecutables;
    static std::shared_ptr<cupva::Executable> GetLidarHistogramConversionExec();
};

// Internal implementation function for submit operations
// Used by macro-generated __cupva/__cuda wrappers
template<typename StreamType>
NVCVStatus LidarHistogramConversionSubmitImpl(NVCVOperatorHandle handle, StreamType stream,
                                              PvaLidarHistogramParams const *const params,
                                              NVCVTensorHandle const histogram, NVCVTensorHandle const rangeBias,
                                              NVCVTensorHandle const xyzCalibration, NVCVTensorHandle range,
                                              NVCVTensorHandle xyz, NVCVTensorHandle reflectance);

} // namespace pva_solutions::priv

#endif // PVA_SOLUTIONS_PRIV_LIDARHISTOGRAMCONVERSION_HPP
