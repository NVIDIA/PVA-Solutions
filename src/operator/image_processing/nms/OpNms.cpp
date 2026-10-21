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

#include "OpNms.hpp"

#include "device/nms_param.h"

#include <IOperator.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <nvcv/Tensor.hpp>

#include <ErrorCheckMacros.h>
#include <cupva_host.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>
#include <nvcv/Tensor.h>

#include <cstdint>
#include <stdexcept>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_nms_K3_S32_dev)
CUPVA_DECLARE_EXECUTABLE(pva_nms_K5_S32_dev)

namespace pva_solutions::priv {

template<typename T>
std::mutex Nms<T>::s_execMapMutex;

template<typename T>
std::map<typename Nms<T>::ExecKey, std::weak_ptr<cupva::Executable>> Nms<T>::s_nmsExecutables;

template<typename T>
std::shared_ptr<cupva::Executable> Nms<T>::GetExec(ExecDataKey execDataKey)
{
    static std::map<ExecDataKey, IOperator::ExecutableData> execDataMap{
        {{3}, {CUPVA_EXECUTABLE_DATA(pva_nms_K3_S32_dev), CUPVA_EXECUTABLE_SIZE(pva_nms_K3_S32_dev)}},
        {{5}, {CUPVA_EXECUTABLE_DATA(pva_nms_K5_S32_dev), CUPVA_EXECUTABLE_SIZE(pva_nms_K5_S32_dev)}},
    };

    auto execDataIt = execDataMap.find(execDataKey);
    if (execDataIt == execDataMap.end())
    {
        throw std::invalid_argument("Unsupported type or parameters");
    }

    ExecKey execKey = std::tuple_cat(execDataKey, std::make_tuple(cupva::Context::GetCurrent()));
    return GetExecutable(execDataIt->second, execKey, s_nmsExecutables, s_execMapMutex);
}

template<typename T>
Nms<T>::Nms(NVCVTensorRequirements *tensorRequirements, int32_t nmsSize, const NVCVBorderType borderMode,
            int32_t borderValue)
    : m_tensorRequirements(*tensorRequirements)
    , m_srcOp(cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr))
    , m_dstOp(cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr))
    , m_prog{}
    , m_nmsSize(nmsSize)
    , m_borderMode(borderMode)
    , m_borderValue(borderValue)
{
    NVCVTensorLayout refLayout;
    nvcvTensorLayoutMake("HWC", &refLayout);

    if (nvcvTensorLayoutCompare(m_tensorRequirements.layout, refLayout) != 0)
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor must have HWC layout");
    if (!(m_tensorRequirements.shape[2] == 1))
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor must have 1 channel");
    if (!(m_tensorRequirements.dtype == NVCV_DATA_TYPE_S32))
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Only NVCV_DATA_TYPE_S32 is supported");
    if (!(nmsSize == 3 || nmsSize == 5))
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Suppression window size must be 3x3 or 5x5");
    if (!(borderMode == NVCV_BORDER_CONSTANT || borderMode == NVCV_BORDER_REPLICATE))
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Only NVCV_BORDER_CONSTANT and NVCV_BORDER_REPLICATE are supported");

    m_exec = GetExec(std::make_tuple(nmsSize));

    initializeProgram();
}

template<typename T>
void Nms<T>::initializeProgram()
{
    auto prog = CmdProgram::Create(*m_exec.get());

    const int iw = m_tensorRequirements.shape[1];
    const int ih = m_tensorRequirements.shape[0];
    const int lp = m_tensorRequirements.strides[0] / sizeof(T);

    const int radius = m_nmsSize >> 1;

    const int niterX = (iw + TILE_WIDTH - 1) / TILE_WIDTH;
    const int niterY = (ih + TILE_HEIGHT - 1) / TILE_HEIGHT;

    T *src_v = prog["srcBuffer"].ptr<T>();
    T *dst_v = prog["dstBuffer"].ptr<T>();

    auto srcHdl = prog["srcHandler"];
    auto dstHdl = prog["dstHandler"];

    prog["niterX"] = niterX;
    prog["niterY"] = niterY;

    // The device kernel initializes the primitive once with the compile-time tile geometry, so the
    // source tile must always be full size. For images smaller than a tile the ROI is expanded and
    // the out-of-image region is supplied by the halo padding. The destination tile is clamped to
    // the image instead, with a full-size arena so the VMEM layout the kernel writes into is
    // unchanged.
    const int roiWidth      = (TILE_WIDTH > iw) ? TILE_WIDTH : iw;
    const int roiHeight     = (TILE_HEIGHT > ih) ? TILE_HEIGHT : ih;
    const int dstTileWidth  = (TILE_WIDTH < iw) ? TILE_WIDTH : iw;
    const int dstTileHeight = (TILE_HEIGHT < ih) ? TILE_HEIGHT : ih;

    RasterDataFlow &srcDataFlow = prog.addDataFlowHead<RasterDataFlow>();
    srcDataFlow.handler(srcHdl)
        .src(m_srcOp, iw, ih, lp)
        .tileBuffer(src_v)
        .roi(0, 0, roiWidth, roiHeight)
        .tile(TILE_WIDTH, TILE_HEIGHT);

    if (m_borderMode == NVCV_BORDER_REPLICATE)
    {
        srcDataFlow.halo(radius, radius, PAD_BPE);
    }
    else
    {
        srcDataFlow.halo(radius, radius, PAD_CONST, m_borderValue);
    }

    RasterDataFlow &dstDataFlow = prog.addDataFlowHead<RasterDataFlow>();
    dstDataFlow.handler(dstHdl)
        .tileBuffer(dst_v)
        .dst(m_dstOp, iw, ih, lp)
        .tile(dstTileWidth, dstTileHeight)
        .tileArena(TILE_WIDTH, TILE_HEIGHT);

    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename T>
template<typename S>
void Nms<T>::operator()(S stream, const nvcv::Tensor &in, const nvcv::Tensor &out)
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
        static_assert(!sizeof(S), "Unsupported stream type for NMS operator");
    }
}

template<typename T>
Nms<T>::~Nms()
{
    CupvaCmdDestroy(&m_prog);
}

// Explicit instantiation of the template to support the types we care about
template class Nms<int32_t>;

template void Nms<int32_t>::operator()<cupvaStream_t>(cupvaStream_t, const nvcv::Tensor &, const nvcv::Tensor &);
template void Nms<int32_t>::operator()<cudaStream_t>(cudaStream_t, const nvcv::Tensor &, const nvcv::Tensor &);

} // namespace pva_solutions::priv

NVCVStatus pvaNmsCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements *tensorRequirements, int32_t nmsSize,
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

        switch (tensorRequirements->dtype)
        {
        case NVCV_DATA_TYPE_S32:
            *handle = reinterpret_cast<NVCVOperatorHandle>(
                new pva_solutions::priv::Nms<int32_t>(tensorRequirements, nmsSize, borderMode, borderValue));
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }
    });
}

template<typename StreamType>
NVCVStatus NmsSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle in, NVCVTensorHandle out)
{
    return nvcv::ProtectCall([&] {
        nvcv::TensorWrapHandle input(in), output(out);

        NVCVDataType inDataType;
        nvcvTensorGetDataType(in, &inDataType);

        switch (inDataType)
        {
        case NVCV_DATA_TYPE_S32:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::Nms<int32_t>>(handle)(stream, input, output);
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus NmsSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle, NVCVTensorHandle);
template NVCVStatus NmsSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle, NVCVTensorHandle);

// C++ overload implementations
NVCVStatus pvaNmsSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out)
{
    return NmsSubmitImpl(handle, stream, in, out);
}

NVCVStatus pvaNmsSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out)
{
    return NmsSubmitImpl(handle, stream, in, out);
}
