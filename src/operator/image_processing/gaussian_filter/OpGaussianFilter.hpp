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
 * @file OpGaussianFilter.hpp
 *
 * @brief Defines the private C++ Class for the GaussianFilter operation.
 */

#ifndef PVA_SOLUTIONS_PRIV_GAUSSIANFILTER_HPP
#define PVA_SOLUTIONS_PRIV_GAUSSIANFILTER_HPP

#include "../conv2d/OpConv2d.hpp"
#include "IOperator.hpp"

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Tensor.hpp>

#include <OpGaussianFilter.h>
#include <cupva_host_scheduling.h>
#include <nvcv/BorderType.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>

namespace pva_solutions::priv {

template<typename T>
class GaussianFilter final : public BaseConv2d<T, GaussianFilter<T>>
{
public:
    using ExecDataKey = std::tuple<int32_t>;
    using ExecKey =
        decltype(std::tuple_cat(std::declval<ExecDataKey>(), std::declval<std::tuple<cupva::impl::Context *>>()));

    explicit GaussianFilter(NVCVTensorRequirements *tensorRequirements, float sigmaX, float sigmaY, int32_t kernelSize,
                            const NVCVBorderType borderMode, int32_t borderValue);

    ~GaussianFilter();

    static auto &GetExecMap()
    {
        return s_gaussianFilterExecutables;
    }
    static auto &GetExecMutex()
    {
        return s_execMapMutex;
    }
    void prepareKernel(void *outKernel, uint32_t kernelSize, int qbits);

protected:
    float m_sigmaX;
    float m_sigmaY;

private:
    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_gaussianFilterExecutables;
};

} // namespace pva_solutions::priv

#endif // PVA_SOLUTIONS_PRIV_GAUSSIANFILTER_HPP
