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
 * @file OpCornerSubPix.hpp
 *
 * @brief Defines the private C++ Class for the CornerSubPix operation.
 */

#ifndef PVA_SOLUTIONS_PRIV_CORNER_SUB_PIX_HPP
#define PVA_SOLUTIONS_PRIV_CORNER_SUB_PIX_HPP

#include <IOperator.hpp>
#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Tensor.hpp>

#include <OpCornerSubPix.h>
#include <cupva_host_scheduling.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>

namespace pva_solutions::priv {

class cornerSubPix final : public IOperator
{
public:
    using ExecKey = std::tuple<cupva::impl::Context *>;
    explicit cornerSubPix(NVCVTensorRequirements *cornerParams, NVCVTensorRequirements *imageParams,
                          const NVCVSize2D winSize, const NVCVSize2D zeroZone);
    ~cornerSubPix();

    void generateMaskInput(int32_t *mask);

    template<typename S>
    void operator()(S stream, const nvcv::Tensor &inCorners, const nvcv::Tensor &outCorners, const nvcv::Tensor &image,
                    const int32_t maxIters, const float eps, const int32_t numCorners);

private:
    NVCVTensorRequirements m_cornerTensorRequirements;
    NVCVTensorRequirements m_imageTensorRequirements;
    int32_t m_width;     // image width
    int32_t m_height;    // image height
    int32_t m_linePitch; // image linepitch
    int32_t m_winWidth;
    int32_t m_winHeight;
    int32_t m_zeroZoneWidth;
    int32_t m_zeroZoneHeight;

    cupva::OffsetPointer<float> m_inCornerOp;
    cupva::OffsetPointer<float> m_outCornerOp;

    cupva::OffsetPointer<uint8_t> m_imageOp;

    cupvaCmd_t m_prog;
    std::shared_ptr<cupva::Executable> m_exec;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_cornerSubPixExecutables;
    static std::shared_ptr<cupva::Executable> GetCornerSubPixExec();
};

} // namespace pva_solutions::priv

#endif // PVA_SOLUTIONS_PRIV_CORNER_SUB_PIX_HPP
