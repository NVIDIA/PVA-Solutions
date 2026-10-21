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
 * @file OpGaussianPyramid.hpp
 *
 * @brief Defines the private C++ Class for the GaussianPyramid operation.
 */

#ifndef PVA_SOLUTIONS_PRIV_GAUSSIANPYRAMID_HPP
#define PVA_SOLUTIONS_PRIV_GAUSSIANPYRAMID_HPP

#include "IOperator.hpp"

#include "device/gaussian_pyramid_param.h"

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Image.hpp>

#include <OpGaussianPyramid.h>
#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>

namespace pva_solutions::priv {

template<typename T>
class GaussianPyramid final : public IOperator
{
public:
    using ExecKey = std::tuple<cupva::impl::Context *>;

    explicit GaussianPyramid(NVCVImageRequirements const *const inputImageRequirements, const int32_t numLevels,
                             const NVCVBorderType borderType, const int32_t borderValue);

    ~GaussianPyramid();

    template<typename S>
    void operator()(S stream, NVCVImageHandle inHandle, NVCVImageHandle *outHandles);

private:
    int32_t m_totalLevels;
    NVCVImageRequirements m_inputImageRequirements;
    std::vector<NVCVImageRequirements> m_outputImagesRequirements;

    cupva::OffsetPointer<T> m_srcOp;
    std::vector<cupva::OffsetPointer<T>> m_dstOps;

    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;
    bool m_progInitialized;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_gaussianPyramidExecutables;
    static std::shared_ptr<cupva::Executable> GetGaussianPyramidExec();
};

} // namespace pva_solutions::priv

#endif // PVA_SOLUTIONS_PRIV_GAUSSIANPYRAMID_HPP
