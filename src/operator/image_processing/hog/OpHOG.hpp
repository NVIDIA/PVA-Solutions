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
 * @file OpHOG.hpp
 *
 * @brief Defines the private C++ Class for the HOG operation.
 */

#ifndef PVA_SOLUTIONS_PRIV_HOG_HPP
#define PVA_SOLUTIONS_PRIV_HOG_HPP

#include "IOperator.hpp"

#include "device/hog_tdata.h"

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>

#include <OpHOG.h>
#include <cupva_host_scheduling.h>
#include <nvcv/BorderType.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>

namespace pva_solutions::priv {

template<typename T>
class HOG final : public IOperator
{
public:
    using ExecDataKey = std::tuple<NVCVImageFormat>;
    using ExecKey =
        decltype(std::tuple_cat(std::declval<ExecDataKey>(), std::declval<std::tuple<cupva::impl::Context *>>()));

    explicit HOG(NVCVImageRequirements *imageRequirements, NVCVTensorRequirements *tensorRequirements);

    ~HOG();

    template<typename S>
    void operator()(S stream, const nvcv::Image &in, const nvcv::Tensor &out);

private:
    NVCVImageRequirements m_imageRequirements;
    int32_t m_imagePlaneCount;

    NVCVTensorRequirements m_tensorRequirements;
    cupva::OffsetPointer<T> m_srcOp[MAX_CHANNEL_COUNT];
    cupva::OffsetPointer<T> m_dstOp;
    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_HOGExecutables;
    static std::shared_ptr<cupva::Executable> GetHOGExec(ExecDataKey execDataKey);
};

} // namespace pva_solutions::priv

// Internal implementation function for submit operations
// Used by macro-generated __cupva/__cuda wrappers
template<typename StreamType>
NVCVStatus HOGSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVImageHandle const in,
                         NVCVTensorHandle const out);

#endif // PVA_SOLUTIONS_PRIV_HOG_HPP
