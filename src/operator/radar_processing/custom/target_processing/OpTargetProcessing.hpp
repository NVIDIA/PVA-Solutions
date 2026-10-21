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
 * @file OpTargetProcessing.hpp
 *
 * @brief Defines the private C++ Class for the standalone Target Processing operation.
 *
 * This operator performs Range-Velocity (RV) decoupling and Cartesian coordinate computation
 * for detected radar targets as a standalone operator (independent of DOA operators).
 */

#ifndef PVA_SOLUTIONS_PRIV_TARGET_PROCESSING_OP_HPP
#define PVA_SOLUTIONS_PRIV_TARGET_PROCESSING_OP_HPP

#include "IOperator.hpp"
#include "TargetProcessing.hpp"

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Tensor.hpp>

#include <OpTargetProcessing.h>
#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>

namespace pva_solutions::priv {

template<typename TI, typename TO>
class TargetProcessingOp final : public IOperator
{
public:
    explicit TargetProcessingOp(NVCVTensorRequirements const *const detectionListTensorRequirements,
                                NVCVTensorRequirements const *const targetIndexMapTensorRequirements,
                                NVCVTensorRequirements const *const targetAnglesTensorRequirements,
                                NVCVTensorRequirements const *const ddmDopplerOffsetsTensorRequirements,
                                NVCVTensorRequirements const *const targetListTensorRequirements,
                                NVCVTensorRequirements const *const nciTensorRequirements,
                                PVATargetProcessingParams const *const params);

    ~TargetProcessingOp();

    template<typename S>
    void operator()(S stream, const nvcv::Tensor &targetCountTensor, const nvcv::Tensor &detectionCountTensor,
                    const nvcv::Tensor &detectionListTensor, const nvcv::Tensor &targetIndexMapTensor,
                    const nvcv::Tensor &targetAnglesTensor, const nvcv::Tensor &ddmDopplerOffsetsTensor,
                    const nvcv::Tensor *nciTensor, const nvcv::Tensor &targetListTensor);

private:
    // Target processing component (from common)
    std::unique_ptr<TargetProcessing<TI, TO>> m_targetProcessing;
};

} // namespace pva_solutions::priv

// Internal implementation function for submit operations
template<typename StreamType>
NVCVStatus TargetProcessingSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle const targetCount,
                                      NVCVTensorHandle const detectionCount, NVCVTensorHandle const detectionList,
                                      NVCVTensorHandle const targetIndexMap, NVCVTensorHandle const targetAngles,
                                      NVCVTensorHandle const ddmDopplerOffsets, NVCVTensorHandle const nci,
                                      NVCVTensorHandle targetList);
#endif // PVA_SOLUTIONS_PRIV_TARGET_PROCESSING_OP_HPP
