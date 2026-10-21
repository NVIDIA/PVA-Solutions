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
 * @file OpBatchSVD.hpp
 *
 * @brief Defines the private C++ Class for the BatchSVD operation.
 */

#ifndef PVA_SOLUTIONS_PRIV_BATCH_SVD_HPP
#define PVA_SOLUTIONS_PRIV_BATCH_SVD_HPP

#include <IOperator.hpp>
#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Tensor.hpp>

#include <OpBatchSVD.h>
#include <cupva_host_scheduling.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>

namespace pva_solutions::priv {

class BatchSVD final : public IOperator
{
public:
    using ExecKey = std::tuple<cupva::impl::Context *>;
    explicit BatchSVD(const NVCVTensorRequirements *srcParams);
    ~BatchSVD();

    template<typename StreamType>
    void operator()(StreamType stream, const nvcv::Tensor &src, const nvcv::Tensor &U, const nvcv::Tensor &S,
                    const nvcv::Tensor &V, const int32_t maxIters, const float tol);

private:
    NVCVTensorRequirements m_srcTensorRequirements;
    NVCVTensorRequirements m_uTensorRequirements;
    NVCVTensorRequirements m_sTensorRequirements;
    NVCVTensorRequirements m_vTensorRequirements;
    int32_t m_channel;
    int32_t m_width;
    int32_t m_height;
    int32_t m_linePitch;
    NVCVDataType m_srcDataType;
    nvcv::TensorShape m_srcShape;

    cupva::OffsetPointer<float> m_srcOp;
    cupva::OffsetPointer<float> m_uOp;
    cupva::OffsetPointer<float> m_sOp;
    cupva::OffsetPointer<float> m_vOp;

    cupvaCmd_t m_prog;
    std::shared_ptr<cupva::Executable> m_exec;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_BatchSVDExecutables;
    static std::shared_ptr<cupva::Executable> GetBatchSVDExec();
};

} // namespace pva_solutions::priv

// Internal implementation function for submit operations
// Used by macro-generated __cupva/__cuda wrappers
template<typename StreamType>
NVCVStatus BatchSVDSubmitImpl(NVCVOperatorHandle handle, StreamType stream, const NVCVTensorHandle src,
                              const NVCVTensorHandle U, const NVCVTensorHandle S, const NVCVTensorHandle V,
                              const int32_t maxIters, const float tol);

#endif // PVA_SOLUTIONS_PRIV_BATCH_SVD_HPP
