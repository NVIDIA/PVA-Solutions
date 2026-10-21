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

#include "OpMorphology.hpp"

#include "device/morphological_param.h"

#include <IOperator.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <nvcv/Tensor.hpp>

#include <ErrorCheckMacros.h>
#include <OpMorphology.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <cstdint>

using namespace cupva;

// supports erode and dilate for U8,S8,U16,S16 inputs with kernel sizes of 3x3,5x5 and kernel shapes of rectangle and cross (3x3 only)
// Erode K3
CUPVA_DECLARE_EXECUTABLE(filter_ERODE_U8_K3_RECT_dev)
CUPVA_DECLARE_EXECUTABLE(filter_ERODE_S8_K3_RECT_dev)
CUPVA_DECLARE_EXECUTABLE(filter_ERODE_U16_K3_RECT_dev)
CUPVA_DECLARE_EXECUTABLE(filter_ERODE_S16_K3_RECT_dev)

CUPVA_DECLARE_EXECUTABLE(filter_ERODE_U8_K3_CROSS_dev)
CUPVA_DECLARE_EXECUTABLE(filter_ERODE_S8_K3_CROSS_dev)
CUPVA_DECLARE_EXECUTABLE(filter_ERODE_U16_K3_CROSS_dev)
CUPVA_DECLARE_EXECUTABLE(filter_ERODE_S16_K3_CROSS_dev)

// Erode K5
CUPVA_DECLARE_EXECUTABLE(filter_ERODE_U8_K5_RECT_dev)
CUPVA_DECLARE_EXECUTABLE(filter_ERODE_S8_K5_RECT_dev)
CUPVA_DECLARE_EXECUTABLE(filter_ERODE_U16_K5_RECT_dev)
CUPVA_DECLARE_EXECUTABLE(filter_ERODE_S16_K5_RECT_dev)

// Dilate K3
CUPVA_DECLARE_EXECUTABLE(filter_DILATE_U8_K3_RECT_dev)
CUPVA_DECLARE_EXECUTABLE(filter_DILATE_S8_K3_RECT_dev)
CUPVA_DECLARE_EXECUTABLE(filter_DILATE_U16_K3_RECT_dev)
CUPVA_DECLARE_EXECUTABLE(filter_DILATE_S16_K3_RECT_dev)

CUPVA_DECLARE_EXECUTABLE(filter_DILATE_U8_K3_CROSS_dev)
CUPVA_DECLARE_EXECUTABLE(filter_DILATE_S8_K3_CROSS_dev)
CUPVA_DECLARE_EXECUTABLE(filter_DILATE_U16_K3_CROSS_dev)
CUPVA_DECLARE_EXECUTABLE(filter_DILATE_S16_K3_CROSS_dev)

// Dilate K5
CUPVA_DECLARE_EXECUTABLE(filter_DILATE_U8_K5_RECT_dev)
CUPVA_DECLARE_EXECUTABLE(filter_DILATE_S8_K5_RECT_dev)
CUPVA_DECLARE_EXECUTABLE(filter_DILATE_U16_K5_RECT_dev)
CUPVA_DECLARE_EXECUTABLE(filter_DILATE_S16_K5_RECT_dev)

namespace pva_solutions::priv {

template<typename T>
std::mutex Morphology<T>::s_execMapMutex;

template<typename T>
std::map<typename Morphology<T>::ExecKey, std::weak_ptr<cupva::Executable>> Morphology<T>::s_morphologyExecutables;

template<typename T>
std::map<typename Morphology<T>::ExecDataKey, IOperator::ExecutableData> createExecDataMap();

#define CREATE_EXEC_DATA_MAP(TYPE, TYPE_SUFFIX, PREFIX)                                                   \
    template<>                                                                                            \
    std::map<typename Morphology<TYPE>::ExecDataKey, IOperator::ExecutableData> createExecDataMap<TYPE>() \
    {                                                                                                     \
        return {                                                                                          \
            {{PVA_ERODE, 3, RECTANGLE_MASK},                                                              \
             {CUPVA_EXECUTABLE_DATA(PREFIX##_ERODE_##TYPE_SUFFIX##_K3_RECT_dev),                          \
              CUPVA_EXECUTABLE_SIZE(PREFIX##_ERODE_##TYPE_SUFFIX##_K3_RECT_dev)}},                        \
            {{PVA_ERODE, 3, CROSS_MASK},                                                                  \
             {CUPVA_EXECUTABLE_DATA(PREFIX##_ERODE_##TYPE_SUFFIX##_K3_CROSS_dev),                         \
              CUPVA_EXECUTABLE_SIZE(PREFIX##_ERODE_##TYPE_SUFFIX##_K3_CROSS_dev)}},                       \
            {{PVA_ERODE, 5, RECTANGLE_MASK},                                                              \
             {CUPVA_EXECUTABLE_DATA(PREFIX##_ERODE_##TYPE_SUFFIX##_K5_RECT_dev),                          \
              CUPVA_EXECUTABLE_SIZE(PREFIX##_ERODE_##TYPE_SUFFIX##_K5_RECT_dev)}},                        \
            {{PVA_DILATE, 3, RECTANGLE_MASK},                                                             \
             {CUPVA_EXECUTABLE_DATA(PREFIX##_DILATE_##TYPE_SUFFIX##_K3_RECT_dev),                         \
              CUPVA_EXECUTABLE_SIZE(PREFIX##_DILATE_##TYPE_SUFFIX##_K3_RECT_dev)}},                       \
            {{PVA_DILATE, 3, CROSS_MASK},                                                                 \
             {CUPVA_EXECUTABLE_DATA(PREFIX##_DILATE_##TYPE_SUFFIX##_K3_CROSS_dev),                        \
              CUPVA_EXECUTABLE_SIZE(PREFIX##_DILATE_##TYPE_SUFFIX##_K3_CROSS_dev)}},                      \
            {{PVA_DILATE, 5, RECTANGLE_MASK},                                                             \
             {CUPVA_EXECUTABLE_DATA(PREFIX##_DILATE_##TYPE_SUFFIX##_K5_RECT_dev),                         \
              CUPVA_EXECUTABLE_SIZE(PREFIX##_DILATE_##TYPE_SUFFIX##_K5_RECT_dev)}},                       \
        };                                                                                                \
    }

CREATE_EXEC_DATA_MAP(uint8_t, U8, filter)
CREATE_EXEC_DATA_MAP(int8_t, S8, filter)
CREATE_EXEC_DATA_MAP(uint16_t, U16, filter)
CREATE_EXEC_DATA_MAP(int16_t, S16, filter)

template<typename T>
std::shared_ptr<cupva::Executable> Morphology<T>::GetMorphologyExec(ExecDataKey execDataKey)
{
    static auto execDataMap = createExecDataMap<T>();

    auto execDataIt = execDataMap.find(execDataKey);
    if (execDataIt != execDataMap.end())
    {
        // Create execKey by combining execDataKey and context
        ExecKey execKey = std::tuple_cat(execDataKey, std::make_tuple(cupva::Context::GetCurrent()));
        return GetExecutable(execDataIt->second, execKey, s_morphologyExecutables, s_execMapMutex);
    }
    else
    {
        throw std::invalid_argument("Unsupported type or parameters");
    }
}

template<typename T>
Morphology<T>::Morphology(NVCVTensorRequirements *tensorRequirements, PVAMorphologyType morphType,
                          PvaMorphologyMaskParams *maskParams, const NVCVBorderType borderMode, int32_t borderValue)
    : m_tensorRequirements{*tensorRequirements}
    , m_width(tensorRequirements->shape[1])
    , m_height(tensorRequirements->shape[0])
    , m_inputLinePitch(tensorRequirements->strides[0] / sizeof(T))
    , m_outputLinePitch(tensorRequirements->strides[0] / sizeof(T))
    , m_maskWidth(maskParams->maskWidth)
    , m_maskHeight(maskParams->maskHeight)
    , m_dataType(tensorRequirements->dtype)
    , m_shape(tensorRequirements->shape, tensorRequirements->rank, tensorRequirements->layout)
    , m_srcOp(cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr))
    , m_dstOp(cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr))
    , m_prog{}
{
    if (!(m_shape.layout() == nvcv::TENSOR_HWC))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor must have either HWC layout");
    }
    if (!(m_shape[2] == 1))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor must have 1 channel");
    }
    if (m_maskWidth != m_maskHeight)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Mask must be square");
    }
    if (!(m_maskWidth == 3 || m_maskWidth == 5))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Mask must be 3x3 or 5x5");
    }
    if (m_maskWidth == 5 && maskParams->maskShape == CROSS_MASK)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Cross mask is only supported for 3x3 mask");
    }
    if (!(borderMode == NVCV_BORDER_CONSTANT || borderMode == NVCV_BORDER_REPLICATE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Only NVCV_BORDER_CONSTANT and NVCV_BORDER_REPLICATE are supported");
    }

    m_exec = GetMorphologyExec(std::make_tuple(morphType, maskParams->maskWidth, maskParams->maskShape));

    auto prog = CmdProgram::Create(*m_exec.get());

    const int32_t niterX = (m_width + TILE_W - 1) / TILE_W;
    const int32_t niterY = (m_height + TILE_H - 1) / TILE_H;

    prog["niterOuter"] = niterX * niterY;

    T *src_v = prog["src_v"].ptr<T>();
    T *dst_v = prog["dst_v"].ptr<T>();

    auto srcHdl = prog["srcHdl"];
    auto dstHdl = prog["dstHdl"];

    RasterDataFlow &srcDataFlow = prog.addDataFlowHead<RasterDataFlow>();
    srcDataFlow.handler(srcHdl)
        .src(m_srcOp, m_width, m_height, m_inputLinePitch)
        .tileBuffer(src_v)
        .halo(m_maskWidth / 2, PAD_CONST, borderValue)
        .roi(0, 0, m_width, m_height)
        .tile(TILE_W, TILE_H);
    if (borderMode == NVCV_BORDER_REPLICATE)
    {
        srcDataFlow.halo(m_maskWidth / 2, PAD_BPE);
    }
    else
    {
        srcDataFlow.halo(m_maskWidth / 2, PAD_CONST, borderValue);
    }

    RasterDataFlow &dstDataFlow = prog.addDataFlowHead<RasterDataFlow>();
    dstDataFlow.handler(dstHdl)
        .tileBuffer(dst_v)
        .dst(m_dstOp, m_width, m_height, m_outputLinePitch)
        .tile(TILE_W, TILE_H);

    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename T>
template<typename S>
void Morphology<T>::operator()(S stream, const nvcv::Tensor &in, const nvcv::Tensor &out)
{
    auto inTensorData  = in.exportData<nvcv::TensorDataStrided>();
    auto outTensorData = out.exportData<nvcv::TensorDataStrided>();

    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_tensorRequirements, inTensorData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_tensorRequirements, outTensorData->cdata()))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Input and output tensors must have the same requirements that was used to create the operator");
    }
    if ((static_cast<void *>(inTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(outTensorData->basePtr()) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor base pointers must not be NULL");
    }

    m_srcOp.setBase(static_cast<T *>(static_cast<void *>(inTensorData->basePtr())));
    m_dstOp.setBase(static_cast<T *>(static_cast<void *>(outTensorData->basePtr())));
    CUPVA_CHECK_ERROR(CupvaCmdProgramUpdateDataFlows(&m_prog));

    cupvaCmd_t const *submitCmds[1] = {&m_prog};
    cupvaCmdStatus_t cmdStatus[1]   = {NULL};

    if constexpr (std::is_same_v<S, cupvaStream_t>)
    {
        CUPVA_CHECK_ERROR(CupvaStreamSubmit(stream, submitCmds, cmdStatus, 1, CUPVA_IN_ORDER, -1, -1));
    }
    else if constexpr (std::is_same_v<S, cudaStream_t>)
    {
        if constexpr (kCupvaHasCudaStreamSupport)
        {
            CUPVA_CHECK_ERROR(CupvaCudaStreamSubmit(stream, submitCmds, cmdStatus, 1, CUPVA_IN_ORDER, -1, -1));
        }
        else
        {
            ThrowCudaStreamNotSupported();
        }
    }
    else
    {
        static_assert(!sizeof(S), "Unsupported stream type for Morphology operator");
    }
}

template<typename T>
Morphology<T>::~Morphology()
{
    CUPVA_CHECK_ERROR(CupvaCmdDestroy(&m_prog));
}

// Explicit instantiation of the template to support the types we care about
template class Morphology<uint8_t>;
template class Morphology<int8_t>;
template class Morphology<uint16_t>;
template class Morphology<int16_t>;

} // namespace pva_solutions::priv

NVCVStatus pvaMorphologyCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements *tensorRequirements,
                               PVAMorphologyType morphType, PvaMorphologyMaskParams *maskParams,
                               const NVCVBorderType borderMode, int32_t borderValue)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        if (tensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to TensorRequirements must not be NULL");
        }
        if (maskParams == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Pointer to maskParams must not be NULL");
        }
        switch (tensorRequirements->dtype)
        {
        case NVCV_DATA_TYPE_U8:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::Morphology<uint8_t>(
                tensorRequirements, morphType, maskParams, borderMode, borderValue));
            break;
        case NVCV_DATA_TYPE_S8:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::Morphology<int8_t>(
                tensorRequirements, morphType, maskParams, borderMode, borderValue));
            break;
        case NVCV_DATA_TYPE_U16:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::Morphology<uint16_t>(
                tensorRequirements, morphType, maskParams, borderMode, borderValue));
            break;
        case NVCV_DATA_TYPE_S16:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::Morphology<int16_t>(
                tensorRequirements, morphType, maskParams, borderMode, borderValue));
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }
    });
}

template<typename StreamType>
NVCVStatus MorphologySubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle in, NVCVTensorHandle out)
{
    return nvcv::ProtectCall([&] {
        nvcv::TensorWrapHandle input(in), output(out);

        NVCVDataType inDataType, outDataType;
        nvcvTensorGetDataType(in, &inDataType);
        nvcvTensorGetDataType(out, &outDataType);

        switch (inDataType)
        {
        case NVCV_DATA_TYPE_U8:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::Morphology<uint8_t>>(handle)(stream, input, output);
            break;
        case NVCV_DATA_TYPE_S8:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::Morphology<int8_t>>(handle)(stream, input, output);
            break;
        case NVCV_DATA_TYPE_U16:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::Morphology<uint16_t>>(handle)(stream, input, output);
            break;
        case NVCV_DATA_TYPE_S16:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::Morphology<int16_t>>(handle)(stream, input, output);
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus MorphologySubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle,
                                                        NVCVTensorHandle);
template NVCVStatus MorphologySubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle,
                                                       NVCVTensorHandle);

// C++ overload implementations
NVCVStatus pvaMorphologySubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle in,
                               NVCVTensorHandle out)
{
    return MorphologySubmitImpl(handle, stream, in, out);
}

NVCVStatus pvaMorphologySubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                               NVCVTensorHandle out)
{
    return MorphologySubmitImpl(handle, stream, in, out);
}
