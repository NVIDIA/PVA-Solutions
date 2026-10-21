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

#include "OpBatchSVD.hpp"

#include "device/batch_svd_params.h"

#include <IOperator.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <cupva_host_utils.hpp>
#include <nvcv/Tensor.hpp>

#include <OpBatchSVD.h>
#include <PvaAllocator.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>
#include <nvcv/Fwd.h>
#include <nvcv/Tensor.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_batch_svd_dev);

namespace pva_solutions::priv {

std::mutex BatchSVD::s_execMapMutex;
std::map<BatchSVD::ExecKey, std::weak_ptr<cupva::Executable>> BatchSVD::s_BatchSVDExecutables;

std::shared_ptr<cupva::Executable> BatchSVD::GetBatchSVDExec()
{

    static IOperator::ExecutableData BatchSVDExecData = {CUPVA_EXECUTABLE_DATA(pva_batch_svd_dev),
                                                         CUPVA_EXECUTABLE_SIZE(pva_batch_svd_dev)};

    ExecKey execKey = std::make_tuple(cupva::Context::GetCurrent());
    return IOperator::GetExecutable<ExecKey>(BatchSVDExecData, execKey, s_BatchSVDExecutables, s_execMapMutex);
}

BatchSVD::BatchSVD(const NVCVTensorRequirements *srcParams)
    : m_srcTensorRequirements(*srcParams)
    , m_channel(srcParams->shape[2])
    , m_width(srcParams->shape[1])
    , m_height(srcParams->shape[0])
    , m_linePitch(srcParams->strides[1])
    , m_srcDataType(srcParams->dtype)
    , m_srcShape(srcParams->shape, srcParams->rank, srcParams->layout)
    , m_srcOp(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr))
    , m_uOp(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr))
    , m_sOp(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr))
    , m_vOp(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr))
    , m_prog{}
{
    if (!(m_srcShape.layout() == nvcv::TENSOR_HWC))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Src tensor is expected to have HWC layout");
    }

    if (!(m_srcDataType == NVCV_DATA_TYPE_F32))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Src tensor only supports 32-bit float data type");
    }

    if (m_width > m_height)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "The matrix width should be less than or equal to height");
    }

    if (m_width > 31)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "The largest matrix width supported is 31");
    }

    if ((m_width * m_height * PITCH * 2 + (m_width + 1) * BATCH) > 1024 * 128)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "The source matrix exceeds the supported size");
    }

    int32_t tensorRank = 3;
    NVCVTensorLayout tensorLayout;
    nvcvTensorLayoutMake("HWC", &tensorLayout);
    int64_t uTensorShape[] = {m_height, m_width, m_channel};
    int64_t sTensorShape[] = {1, m_width, m_channel};
    int64_t vTensorShape[] = {m_width, m_width, m_channel};

    nvcvTensorCalcRequirementsPva(tensorRank, uTensorShape, NVCV_DATA_TYPE_F32, tensorLayout, 0, 0,
                                  &m_uTensorRequirements);
    nvcvTensorCalcRequirementsPva(tensorRank, sTensorShape, NVCV_DATA_TYPE_F32, tensorLayout, 0, 0,
                                  &m_sTensorRequirements);
    nvcvTensorCalcRequirementsPva(tensorRank, vTensorShape, NVCV_DATA_TYPE_F32, tensorLayout, 0, 0,
                                  &m_vTensorRequirements);

    m_exec    = GetBatchSVDExec();
    auto prog = CmdProgram::Create(*m_exec.get());

    float *U_vmem = prog["U"].ptr<float>();

    float *U_sorted_vmem = prog["sorted_U"].ptr<float>();
    float *S_sorted_vmem = prog["sorted_S"].ptr<float>();
    float *V_sorted_vmem = prog["sorted_V"].ptr<float>();

    auto src_handler = prog["src_handler"];
    auto dst_handler = prog["dst_handler"];

    SequenceDataFlow &srcDataFlow = prog.addDataFlowHead<SequenceDataFlow>().handler(src_handler);
    SequenceDataFlow &dstDataFlow = prog.addDataFlowHead<SequenceDataFlow>().handler(dst_handler);

    auto count        = (m_channel + BATCH - 1) / BATCH;
    int32_t last_tile = m_channel - BATCH * (count - 1);
    if (count > 2)
    {
        srcDataFlow.addTransfer()
            .src(m_srcOp, m_linePitch)
            .dst(U_vmem, PITCH * sizeof(float))
            .srcDim1((count - 1) / 2 * 2, BATCH * sizeof(float))
            .dstDim1(2, PITCH * m_width * m_height * sizeof(float))
            .dstDim2((count - 1) / 2, 0)
            .tile(BATCH * sizeof(float), m_width * m_height)
            .mode(TransferModeType::TILE);
    }
    if (count % 2 == 1)
    {
        srcDataFlow.addTransfer()
            .src(m_srcOp + (count - 1) * BATCH, m_linePitch)
            .dst(U_vmem, PITCH * sizeof(float))
            .tile(last_tile * sizeof(float), m_width * m_height);
    }
    else
    {
        srcDataFlow.addTransfer()
            .src(m_srcOp + (count - 2) * BATCH, m_linePitch)
            .dst(U_vmem, PITCH * sizeof(float))
            .tile(BATCH * sizeof(float), m_width * m_height)
            .mode(TransferModeType::TILE);
        srcDataFlow.addTransfer()
            .src(m_srcOp + (count - 1) * BATCH, m_linePitch)
            .dst(U_vmem + PITCH * m_width * m_height, PITCH * sizeof(float))
            .tile(last_tile * sizeof(float), m_width * m_height);
    }

    int32_t tx = (m_channel > BATCH) ? BATCH * sizeof(float) : m_channel * sizeof(float);

    dstDataFlow.addTransfer()
        .src(U_sorted_vmem, BATCH * sizeof(float))
        .dst(m_uOp, m_linePitch)
        .tile(tx, m_width * m_height)
        .mode(TransferModeType::CONTINUOUS);
    dstDataFlow.addTransfer()
        .src(S_sorted_vmem, BATCH * sizeof(float))
        .dst(m_sOp, m_linePitch)
        .tile(tx, m_width)
        .mode(TransferModeType::CONTINUOUS);
    dstDataFlow.addTransfer()
        .src(V_sorted_vmem, BATCH * sizeof(float))
        .dst(m_vOp, m_linePitch)
        .tile(tx, m_width * m_width)
        .mode(TransferModeType::CONTINUOUS);

    prog["n"]           = m_width;
    prog["m"]           = m_height;
    prog["num_channel"] = m_channel;
    prog["batch_count"] = count;

    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename StreamType>
void BatchSVD::operator()(StreamType stream, const nvcv::Tensor &src, const nvcv::Tensor &U, const nvcv::Tensor &S,
                          const nvcv::Tensor &V, const int32_t maxIters, const float tol)
{
    auto srcTensorData = src.exportData<nvcv::TensorDataStrided>();
    auto uTensorData   = U.exportData<nvcv::TensorDataStrided>();
    auto sTensorData   = S.exportData<nvcv::TensorDataStrided>();
    auto vTensorData   = V.exportData<nvcv::TensorDataStrided>();

    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_srcTensorRequirements, srcTensorData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_uTensorRequirements, uTensorData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_sTensorRequirements, sTensorData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_vTensorRequirements, vTensorData->cdata()))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Input and output tensors must have the same requirements that were used to create the operator");
    }
    if ((static_cast<void *>(srcTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(uTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(sTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(vTensorData->basePtr()) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor base pointers must not be NULL");
    }

    cupvaParameter_t param;

    CupvaCmdProgramGetParameter(&m_prog, &param, "max_iters");
    CupvaParameterSetValueScalar(&param, &maxIters, sizeof(int32_t));
    CupvaCmdProgramGetParameter(&m_prog, &param, "tol");
    CupvaParameterSetValueScalar(&param, &tol, sizeof(float));

    m_uOp.setBase(static_cast<float *>(static_cast<void *>(uTensorData->basePtr())));
    m_sOp.setBase(static_cast<float *>(static_cast<void *>(sTensorData->basePtr())));
    m_vOp.setBase(static_cast<float *>(static_cast<void *>(vTensorData->basePtr())));
    m_srcOp.setBase(static_cast<float *>(static_cast<void *>(srcTensorData->basePtr())));

    void *addr_U = static_cast<void *>(uTensorData->basePtr());
    CupvaCmdProgramGetParameter(&m_prog, &param, "dram_U");
    CupvaParameterSetValuePointer(&param, addr_U);

    void *addr_S = static_cast<void *>(sTensorData->basePtr());
    CupvaCmdProgramGetParameter(&m_prog, &param, "dram_S");
    CupvaParameterSetValuePointer(&param, addr_S);

    void *addr_V = static_cast<void *>(vTensorData->basePtr());
    CupvaCmdProgramGetParameter(&m_prog, &param, "dram_V");
    CupvaParameterSetValuePointer(&param, addr_V);

    if (CupvaCmdProgramUpdateDataFlows(&m_prog) != CUPVA_ERROR_NONE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Update DataFlow failed");
    }

    cupvaCmd_t const *submitCmds[1] = {&m_prog};
    cupvaCmdStatus_t cmdStatus[1]   = {NULL};

    if constexpr (std::is_same_v<StreamType, cupvaStream_t>)
    {
        if (CupvaStreamSubmit(stream, submitCmds, cmdStatus, 1, CUPVA_IN_ORDER, -1, -1) != CUPVA_ERROR_NONE)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Submit failed");
        }
    }
    else if constexpr (std::is_same_v<StreamType, cudaStream_t>)
    {
        if constexpr (kCupvaHasCudaStreamSupport)
        {
            if (CupvaCudaStreamSubmit(stream, submitCmds, cmdStatus, 1, CUPVA_IN_ORDER, -1, -1) != CUPVA_ERROR_NONE)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Submit failed");
            }
        }
        else
        {
            ThrowCudaStreamNotSupported();
        }
    }
    else
    {
        static_assert(!sizeof(StreamType), "Unsupported stream type for BatchSVD operator");
    }
}

BatchSVD::~BatchSVD()
{
    CupvaCmdDestroy(&m_prog);
}

} // namespace pva_solutions::priv

NVCVStatus pvaBatchSVDCreate(NVCVOperatorHandle *handle, const NVCVTensorRequirements *srcParams)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        if (srcParams == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to src TensorRequirements must not be NULL");
        }

        *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::BatchSVD(srcParams));
    });
}

template<typename StreamType>
NVCVStatus BatchSVDSubmitImpl(NVCVOperatorHandle handle, StreamType stream, const NVCVTensorHandle src,
                              const NVCVTensorHandle U, const NVCVTensorHandle S, const NVCVTensorHandle V,
                              const int32_t maxIters, const float tol)
{
    return nvcv::ProtectCall([&] {
        nvcv::TensorWrapHandle in(src), u(U), s(S), v(V);
        pva_solutions::priv::ToDynamicRef<pva_solutions::priv::BatchSVD>(handle)(stream, in, u, s, v, maxIters, tol);
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus BatchSVDSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, const NVCVTensorHandle,
                                                      const NVCVTensorHandle, const NVCVTensorHandle,
                                                      const NVCVTensorHandle, const int32_t, const float);
template NVCVStatus BatchSVDSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, const NVCVTensorHandle,
                                                     const NVCVTensorHandle, const NVCVTensorHandle,
                                                     const NVCVTensorHandle, const int32_t, const float);

// C++ overload implementations
NVCVStatus pvaBatchSVDSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, const NVCVTensorHandle src,
                             const NVCVTensorHandle U, const NVCVTensorHandle S, const NVCVTensorHandle V,
                             const int32_t maxIters, const float tol)
{
    return BatchSVDSubmitImpl(handle, stream, src, U, S, V, maxIters, tol);
}

NVCVStatus pvaBatchSVDSubmit(NVCVOperatorHandle handle, cudaStream_t stream, const NVCVTensorHandle src,
                             const NVCVTensorHandle U, const NVCVTensorHandle S, const NVCVTensorHandle V,
                             const int32_t maxIters, const float tol)
{
    return BatchSVDSubmitImpl(handle, stream, src, U, S, V, maxIters, tol);
}
