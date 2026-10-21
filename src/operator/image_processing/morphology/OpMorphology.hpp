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
 * @file OpMorphology.hpp
 *
 * @brief Defines the private C++ Class for the Morphology operation.
 */

#ifndef PVA_SOLUTIONS_PRIV_MORPHOLOGY_HPP
#define PVA_SOLUTIONS_PRIV_MORPHOLOGY_HPP

#include "IOperator.hpp"

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Tensor.hpp>

#include <OpMorphology.h>
#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>
#include <nvcv/BorderType.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>

namespace pva_solutions::priv {

template<typename T>
class Morphology final : public IOperator
{
public:
    using ExecDataKey = std::tuple<PVAMorphologyType, int, MorphologyMaskType>;
    using ExecKey =
        decltype(std::tuple_cat(std::declval<ExecDataKey>(), std::declval<std::tuple<cupva::impl::Context *>>()));

    explicit Morphology(NVCVTensorRequirements *tensorRequirements, PVAMorphologyType morphType,
                        PvaMorphologyMaskParams *maskParams, const NVCVBorderType borderMode, int32_t borderValue);

    ~Morphology();

    template<typename S>
    void operator()(S stream, const nvcv::Tensor &in, const nvcv::Tensor &out);

private:
    NVCVTensorRequirements m_tensorRequirements;
    int32_t m_width;
    int32_t m_height;
    int32_t m_inputLinePitch;
    int32_t m_outputLinePitch;
    int32_t m_maskWidth;
    int32_t m_maskHeight;
    NVCVDataType m_dataType;
    nvcv::TensorShape m_shape;

    cupva::OffsetPointer<T> m_srcOp;
    cupva::OffsetPointer<T> m_dstOp;
    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_morphologyExecutables;
    static std::shared_ptr<cupva::Executable> GetMorphologyExec(ExecDataKey execDataKey);
};

} // namespace pva_solutions::priv

// Internal implementation function for submit operations
// Used by macro-generated __cupva/__cuda wrappers
template<typename StreamType>
NVCVStatus MorphologySubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle in,
                                NVCVTensorHandle out);

#endif // PVA_SOLUTIONS_PRIV_MORPHOLOGY_HPP
