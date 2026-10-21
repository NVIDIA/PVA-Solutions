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
 * @file OpRemap.hpp
 *
 * @brief Defines the private C++ Class for the Remap operation.
 */

#ifndef PVA_SOLUTIONS_PRIV_REMAP_HPP
#define PVA_SOLUTIONS_PRIV_REMAP_HPP

#include "IOperator.hpp"

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>

#include <OpRemap.h>
#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>
#include <nvcv/BorderType.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>

namespace pva_solutions::priv {

class Remap final : public IOperator
{
public:
    using ExecDataKey = std::tuple<NVCVImageFormat>;
    using ExecKey =
        decltype(std::tuple_cat(std::declval<ExecDataKey>(), std::declval<std::tuple<cupva::impl::Context *>>()));

    explicit Remap(NVCVImageRequirements const *const inRequirements,
                   NVCVImageRequirements const *const outRequirements,
                   NVCVTensorRequirements const *const warpMapRequirements, PVAInterpolationType const interpType,
                   NVCVBorderType const borderMode);
    ~Remap();

    template<typename S>
    void operator()(S stream, const nvcv::Image &in, const nvcv::Image &out, const nvcv::Tensor &warpMap);

private:
    NVCVImageRequirements m_inRequirements;
    NVCVImageRequirements m_outRequirements;
    NVCVTensorRequirements m_warpMapRequirements;

    // at most 2 planes
    cupva::OffsetPointer<uint8_t> m_srcOp[2];
    cupva::OffsetPointer<uint8_t> m_dstOp[2];
    cupva::OffsetPointer<float> m_warpMapOp;
    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_remapExecutables;
    static std::shared_ptr<cupva::Executable> GetRemapExec(ExecDataKey execDataKey);
};

} // namespace pva_solutions::priv

// Internal implementation function for submit operations
// Used by macro-generated __cupva/__cuda wrappers
template<typename StreamType>
NVCVStatus RemapSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVImageHandle in, NVCVImageHandle out,
                           NVCVTensorHandle warpMap);

#endif // PVA_SOLUTIONS_PRIV_REMAP_HPP
