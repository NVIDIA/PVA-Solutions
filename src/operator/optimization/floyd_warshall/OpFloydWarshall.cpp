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

#include "OpFloydWarshall.hpp"

#include "device/floyd_warshall_param.h"

#include <IOperator.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <cupva_host_utils.hpp>
#include <nvcv/Tensor.hpp>

#include <OpFloydWarshall.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <cstdint>
#include <iostream>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(floyd_warshall_dev);

namespace pva_solutions::priv {

std::mutex FloydWarshall::s_execMapMutex;

std::map<FloydWarshall::ExecKey, std::weak_ptr<cupva::Executable>> FloydWarshall::s_floydWarshallExecutables;

std::shared_ptr<cupva::Executable> FloydWarshall::GetFloydWarshallExec()
{
    static IOperator::ExecutableData sFloydWarshallExecData = {CUPVA_EXECUTABLE_DATA(floyd_warshall_dev),
                                                               CUPVA_EXECUTABLE_SIZE(floyd_warshall_dev)};

    ExecKey execKey = std::make_tuple(cupva::Context::GetCurrent());
    return IOperator::GetExecutable<ExecKey>(sFloydWarshallExecData, execKey, s_floydWarshallExecutables,
                                             s_execMapMutex);
}

FloydWarshall::FloydWarshall(NVCVTensorRequirements *tensorRequirements)
    : m_tensorRequirements(*tensorRequirements)
    , m_lenTensorOp(cupva::OffsetPointer<uint16_t>::Create(cupva::MemType::DRAM))
    , m_nextTensorOp(cupva::OffsetPointer<uint16_t>::Create(cupva::MemType::DRAM))
    , m_exec{}
    , m_prog{}
{
    if (!(tensorRequirements->dtype == NVCV_DATA_TYPE_U16))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "dtype should be NVCV_DATA_TYPE_U16");
    }

    NVCVTensorLayout refLayout;
    nvcvTensorLayoutMake("HWC", &refLayout);

    if (nvcvTensorLayoutCompare(tensorRequirements->layout, refLayout) != 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor must have HWC layout");
    }

    if (!(tensorRequirements->shape[2] == 1))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor must have 1 channel");
    }

    const int32_t height = static_cast<int32_t>(m_tensorRequirements.shape[0]);
    const int32_t width  = static_cast<int32_t>(m_tensorRequirements.shape[1]);
    const int32_t pitch  = static_cast<int32_t>(m_tensorRequirements.strides[0] / sizeof(uint16_t));

    if (!(width <= MAX_WIDTH && height <= MAX_HEIGHT))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Tensor must have width and height less than or equal to MAX_WIDTH and MAX_HEIGHT respectively");
    }

    if (!(width == height))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "width and height must be equal to node number");
    }

    m_exec    = GetFloydWarshallExec();
    auto prog = CmdProgram::Create(*m_exec.get());

    auto inLenHdl   = prog["inLenHdl"];
    auto inNextHdl  = prog["inNextHdl"];
    auto outLenHdl  = prog["outLenHdl"];
    auto outNextHdl = prog["outNextHdl"];

    prog["num"] = width;

    uint16_t *lenBuf   = prog["len_v"].ptr<uint16_t>();
    uint16_t *lenBufCp = prog["len_cp"].ptr<uint16_t>();
    uint16_t *nextBuf  = prog["next_v"].ptr<uint16_t>();

    RasterDataFlow &inLenDataFlow = prog.addDataFlowHead<RasterDataFlow>();
    inLenDataFlow.handler(inLenHdl)
        .src(m_lenTensorOp, width, height, pitch)
        .tileBuffer(lenBuf, lenBufCp)
        .tile(width, height);

    RasterDataFlow &inNextDataFlow = prog.addDataFlowHead<RasterDataFlow>();
    inNextDataFlow.handler(inNextHdl).src(m_nextTensorOp, width, height, pitch).tileBuffer(nextBuf).tile(width, height);

    RasterDataFlow &outLenDataFlow = prog.addDataFlowHead<RasterDataFlow>();
    outLenDataFlow.handler(outLenHdl).dst(m_lenTensorOp, width, height, pitch).tileBuffer(lenBuf).tile(width, height);

    RasterDataFlow &outNextDataFlow = prog.addDataFlowHead<RasterDataFlow>();
    outNextDataFlow.handler(outNextHdl)
        .dst(m_nextTensorOp, width, height, pitch)
        .tileBuffer(nextBuf)
        .tile(width, height);

    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename S>
void FloydWarshall::operator()(S stream, const nvcv::Tensor &lenTensor, const nvcv::Tensor &nextTensor)
{
    auto lenTensorData  = lenTensor.exportData<nvcv::TensorDataStrided>();
    auto nextTensorData = nextTensor.exportData<nvcv::TensorDataStrided>();

    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_tensorRequirements, lenTensorData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_tensorRequirements, nextTensorData->cdata()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Tensors must have the same requirements that were used to create the operator");
    }

    if ((static_cast<void *>(lenTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(nextTensorData->basePtr()) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor base pointer must not be null");
    }

    m_lenTensorOp.setBase(static_cast<uint16_t *>(static_cast<void *>(lenTensorData->basePtr())));
    m_nextTensorOp.setBase(static_cast<uint16_t *>(static_cast<void *>(nextTensorData->basePtr())));

    cupvaError_t err                = CupvaCmdProgramUpdateDataFlows(&m_prog);
    cupvaCmd_t const *submitCmds[1] = {&m_prog};
    cupvaCmdStatus_t cmdStatus[1]   = {NULL};

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
        static_assert(!sizeof(S), "Unsupported stream type for FloydWarshall operator");
    }

    if (!(err == CUPVA_ERROR_NONE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Submit failed");
    }
}

FloydWarshall::~FloydWarshall()
{
    CupvaCmdDestroy(&m_prog);
}

} // namespace pva_solutions::priv

NVCVStatus pvaFloydWarshallCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements *tensorRequirements)
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
                                  "Pointer to tensorRequirements must not be NULL");
        }
        *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::FloydWarshall(tensorRequirements));
    });
}

template<typename StreamType>
NVCVStatus FloydWarshallSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle lenTensor,
                                   NVCVTensorHandle nextTensor)
{
    return nvcv::ProtectCall([&] {
        nvcv::TensorWrapHandle len(lenTensor), next(nextTensor);
        pva_solutions::priv::ToDynamicRef<pva_solutions::priv::FloydWarshall>(handle)(stream, len, next);
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus FloydWarshallSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle,
                                                           NVCVTensorHandle);
template NVCVStatus FloydWarshallSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle,
                                                          NVCVTensorHandle);

// C++ overload implementations
NVCVStatus pvaFloydWarshallSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle lenTensor,
                                  NVCVTensorHandle nextTensor)
{
    return FloydWarshallSubmitImpl(handle, stream, lenTensor, nextTensor);
}

NVCVStatus pvaFloydWarshallSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle lenTensor,
                                  NVCVTensorHandle nextTensor)
{
    return FloydWarshallSubmitImpl(handle, stream, lenTensor, nextTensor);
}