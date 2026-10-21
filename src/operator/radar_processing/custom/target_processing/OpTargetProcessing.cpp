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
 * @file OpTargetProcessing.cpp
 * @brief Host-side implementation for the standalone Target Processing operator.
 */

#include "OpTargetProcessing.hpp"

#include <IOperator.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>

#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>

namespace pva_solutions::priv {

template<typename TI, typename TO>
TargetProcessingOp<TI, TO>::TargetProcessingOp(NVCVTensorRequirements const *const detectionListTensorRequirements,
                                               NVCVTensorRequirements const *const targetIndexMapTensorRequirements,
                                               NVCVTensorRequirements const *const targetAnglesTensorRequirements,
                                               NVCVTensorRequirements const *const ddmDopplerOffsetsTensorRequirements,
                                               NVCVTensorRequirements const *const targetListTensorRequirements,
                                               NVCVTensorRequirements const *const nciTensorRequirements,
                                               PVATargetProcessingParams const *const params)
{
    m_targetProcessing = std::make_unique<TargetProcessing<TI, TO>>(
        detectionListTensorRequirements, targetIndexMapTensorRequirements, targetAnglesTensorRequirements,
        ddmDopplerOffsetsTensorRequirements, targetListTensorRequirements, nciTensorRequirements, params);
}

template<typename TI, typename TO>
TargetProcessingOp<TI, TO>::~TargetProcessingOp()
{
    m_targetProcessing.reset();
}

template<typename TI, typename TO>
template<typename S>
void TargetProcessingOp<TI, TO>::operator()(S stream, const nvcv::Tensor &targetCountTensor,
                                            const nvcv::Tensor &detectionCountTensor,
                                            const nvcv::Tensor &detectionListTensor,
                                            const nvcv::Tensor &targetIndexMapTensor,
                                            const nvcv::Tensor &targetAnglesTensor,
                                            const nvcv::Tensor &ddmDopplerOffsetsTensor, const nvcv::Tensor *nciTensor,
                                            const nvcv::Tensor &targetListTensor)
{
    // Update data flows with all tensor handles (standalone mode)
    m_targetProcessing->updateDataFlows(targetCountTensor, detectionCountTensor, detectionListTensor,
                                        targetIndexMapTensor, targetAnglesTensor, ddmDopplerOffsetsTensor,
                                        targetListTensor, nciTensor);

    // Submit target processing program
    cupvaCmd_t const *submitCmds[1] = {&m_targetProcessing->m_prog};
    cupvaCmdStatus_t cmdStatus[1]   = {nullptr};

    cupvaError_t err = CUPVA_ERROR_NONE;
    if constexpr (std::is_same_v<S, cupvaStream_t>)
    {
        err = CupvaStreamSubmit(stream, submitCmds, cmdStatus, 1, CUPVA_IN_ORDER, -1, -1);
    }
    else if constexpr (std::is_same_v<S, cudaStream_t>)
    {
        if constexpr (kCupvaHasCudaStreamSupport)
        {
            err = CupvaCudaStreamSubmit(stream, submitCmds, cmdStatus, 1, CUPVA_IN_ORDER, -1, -1);
        }
        else
        {
            ThrowCudaStreamNotSupported();
        }
    }
    else
    {
        static_assert(!sizeof(S), "Unsupported stream type for TargetProcessing operator");
    }

    if (err != CUPVA_ERROR_NONE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Submit failed");
    }
}

// Explicit instantiation of the template
template class TargetProcessingOp<int32_t, float>;

} // namespace pva_solutions::priv

// Create function
NVCVStatus pvaTargetProcessingCreate(NVCVOperatorHandle *handle,
                                     NVCVTensorRequirements const *const detectionListTensorRequirements,
                                     NVCVTensorRequirements const *const targetIndexMapTensorRequirements,
                                     NVCVTensorRequirements const *const targetAnglesTensorRequirements,
                                     NVCVTensorRequirements const *const ddmDopplerOffsetsTensorRequirements,
                                     NVCVTensorRequirements const *const targetListTensorRequirements,
                                     NVCVTensorRequirements const *const nciTensorRequirements,
                                     PVATargetProcessingParams const *const params)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        if (detectionListTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to detection list NVCVTensorRequirements must not be NULL");
        }
        if (targetIndexMapTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to target index map NVCVTensorRequirements must not be NULL");
        }
        if (targetAnglesTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to target angles NVCVTensorRequirements must not be NULL");
        }
        if (ddmDopplerOffsetsTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to DDM Doppler offsets NVCVTensorRequirements must not be NULL");
        }
        if (targetListTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to target list NVCVTensorRequirements must not be NULL");
        }
        // Note: nciTensorRequirements can be nullptr (quadratic interpolation disabled)

        if ((detectionListTensorRequirements->dtype == NVCV_DATA_TYPE_S32) &&
            (targetIndexMapTensorRequirements->dtype == NVCV_DATA_TYPE_S32) &&
            (targetAnglesTensorRequirements->dtype == NVCV_DATA_TYPE_F32) &&
            (ddmDopplerOffsetsTensorRequirements->dtype == NVCV_DATA_TYPE_F32) &&
            (targetListTensorRequirements->dtype == NVCV_DATA_TYPE_F32))
        {
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::TargetProcessingOp<int32_t, float>(
                detectionListTensorRequirements, targetIndexMapTensorRequirements, targetAnglesTensorRequirements,
                ddmDopplerOffsetsTensorRequirements, targetListTensorRequirements, nciTensorRequirements, params));
        }
        else
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input/output data type");
        }
    });
}

// Submit implementation
template<typename StreamType>
NVCVStatus TargetProcessingSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle const targetCount,
                                      NVCVTensorHandle const detectionCount, NVCVTensorHandle const detectionList,
                                      NVCVTensorHandle const targetIndexMap, NVCVTensorHandle const targetAngles,
                                      NVCVTensorHandle const ddmDopplerOffsets, NVCVTensorHandle const nci,
                                      NVCVTensorHandle targetList)
{
    return nvcv::ProtectCall([&] {
        nvcv::TensorWrapHandle wrTargetCount(targetCount), wrDetectionCount(detectionCount),
            wrDetectionList(detectionList), wrTargetIndexMap(targetIndexMap), wrTargetAngles(targetAngles),
            wrDdmDopplerOffsets(ddmDopplerOffsets), wrTargetList(targetList);

        // NCI tensor is optional
        std::unique_ptr<nvcv::TensorWrapHandle> nciWrapper;
        const nvcv::Tensor *nciPtr = nullptr;
        if (nci != nullptr)
        {
            nciWrapper = std::make_unique<nvcv::TensorWrapHandle>(nci);
            nciPtr     = &static_cast<const nvcv::Tensor &>(*nciWrapper);
        }

        NVCVDataType detectionListDataType, targetIndexMapDataType, targetAnglesDataType, ddmDopplerOffsetsDataType,
            targetListDataType;
        nvcvTensorGetDataType(detectionList, &detectionListDataType);
        nvcvTensorGetDataType(targetIndexMap, &targetIndexMapDataType);
        nvcvTensorGetDataType(targetAngles, &targetAnglesDataType);
        nvcvTensorGetDataType(ddmDopplerOffsets, &ddmDopplerOffsetsDataType);
        nvcvTensorGetDataType(targetList, &targetListDataType);

        if ((detectionListDataType == NVCV_DATA_TYPE_S32) && (targetIndexMapDataType == NVCV_DATA_TYPE_S32) &&
            (targetAnglesDataType == NVCV_DATA_TYPE_F32) && (ddmDopplerOffsetsDataType == NVCV_DATA_TYPE_F32) &&
            (targetListDataType == NVCV_DATA_TYPE_F32))
        {
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::TargetProcessingOp<int32_t, float>>(handle)(
                stream, wrTargetCount, wrDetectionCount, wrDetectionList, wrTargetIndexMap, wrTargetAngles,
                wrDdmDopplerOffsets, nciPtr, wrTargetList);
        }
        else
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported tensor data types");
        }
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus TargetProcessingSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle const,
                                                              NVCVTensorHandle const, NVCVTensorHandle const,
                                                              NVCVTensorHandle const, NVCVTensorHandle const,
                                                              NVCVTensorHandle const, NVCVTensorHandle const,
                                                              NVCVTensorHandle);
template NVCVStatus TargetProcessingSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle const,
                                                             NVCVTensorHandle const, NVCVTensorHandle const,
                                                             NVCVTensorHandle const, NVCVTensorHandle const,
                                                             NVCVTensorHandle const, NVCVTensorHandle const,
                                                             NVCVTensorHandle);

// C++ overload implementations
NVCVStatus pvaTargetProcessingSubmit(NVCVOperatorHandle handle, cupvaStream_t stream,
                                     NVCVTensorHandle const targetCount, NVCVTensorHandle const detectionCount,
                                     NVCVTensorHandle const detectionList, NVCVTensorHandle const targetIndexMap,
                                     NVCVTensorHandle const targetAngles, NVCVTensorHandle const ddmDopplerOffsets,
                                     NVCVTensorHandle const nci, NVCVTensorHandle targetList)
{
    return TargetProcessingSubmitImpl(handle, stream, targetCount, detectionCount, detectionList, targetIndexMap,
                                      targetAngles, ddmDopplerOffsets, nci, targetList);
}

NVCVStatus pvaTargetProcessingSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle const targetCount,
                                     NVCVTensorHandle const detectionCount, NVCVTensorHandle const detectionList,
                                     NVCVTensorHandle const targetIndexMap, NVCVTensorHandle const targetAngles,
                                     NVCVTensorHandle const ddmDopplerOffsets, NVCVTensorHandle const nci,
                                     NVCVTensorHandle targetList)
{
    return TargetProcessingSubmitImpl(handle, stream, targetCount, detectionCount, detectionList, targetIndexMap,
                                      targetAngles, ddmDopplerOffsets, nci, targetList);
}
