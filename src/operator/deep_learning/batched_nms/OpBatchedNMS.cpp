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

#include "OpBatchedNMS.hpp"

#include "device/batched_nms_param.h"

#include <IOperator.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <cupva_host_utils.hpp>
#include <nvcv/Tensor.hpp>

#include <OpBatchedNMS.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <cstdint>

using namespace cupva;

// Declare the PVA executable (compiled from device code)
CUPVA_DECLARE_EXECUTABLE(pva_batched_nms_dev);

namespace pva_solutions::priv {

std::mutex BatchedNMS::s_execMapMutex;
std::map<BatchedNMS::ExecKey, std::weak_ptr<cupva::Executable>> BatchedNMS::s_batchedNMSExecutables;

std::shared_ptr<cupva::Executable> BatchedNMS::GetBatchedNMSExec()
{
    static IOperator::ExecutableData batchedNMSExecData = {CUPVA_EXECUTABLE_DATA(pva_batched_nms_dev),
                                                           CUPVA_EXECUTABLE_SIZE(pva_batched_nms_dev)};

    ExecKey execKey = std::make_tuple(cupva::Context::GetCurrent());
    return IOperator::GetExecutable<ExecKey>(batchedNMSExecData, execKey, s_batchedNMSExecutables, s_execMapMutex);
}

BatchedNMS::BatchedNMS(const NVCVTensorRequirements *inBoxesReqs, const NVCVTensorRequirements *inScoresReqs,
                       const NVCVTensorRequirements *inClassesReqs, const NVCVTensorRequirements *inCountReqs,
                       const NVCVTensorRequirements *outBoxesReqs, const NVCVTensorRequirements *outScoresReqs,
                       const NVCVTensorRequirements *outClassesReqs, const NVCVTensorRequirements *outCountReqs,
                       float iouThreshold)
    : m_iouThreshold(iouThreshold)
    , m_inBoxesTensorRequirements(inBoxesReqs ? *inBoxesReqs : NVCVTensorRequirements{})
    , m_inScoresTensorRequirements(inScoresReqs ? *inScoresReqs : NVCVTensorRequirements{})
    , m_inClassesTensorRequirements(inClassesReqs ? *inClassesReqs : NVCVTensorRequirements{})
    , m_inCountTensorRequirements(inCountReqs ? *inCountReqs : NVCVTensorRequirements{})
    , m_outBoxesTensorRequirements(outBoxesReqs ? *outBoxesReqs : NVCVTensorRequirements{})
    , m_outScoresTensorRequirements(outScoresReqs ? *outScoresReqs : NVCVTensorRequirements{})
    , m_outClassesTensorRequirements(outClassesReqs ? *outClassesReqs : NVCVTensorRequirements{})
    , m_outCountTensorRequirements(outCountReqs ? *outCountReqs : NVCVTensorRequirements{})
    , m_inBoxesOp(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr))
    , m_inScoresOp(cupva::OffsetPointer<int16_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_inClassesOp(cupva::OffsetPointer<int16_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_inCountOp(cupva::OffsetPointer<int>::Create(cupva::MemType::DRAM, nullptr))
    , m_outBoxesOp(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr))
    , m_outScoresOp(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr))
    , m_outClassesOp(cupva::OffsetPointer<int>::Create(cupva::MemType::DRAM, nullptr))
    , m_outCountOp(cupva::OffsetPointer<int>::Create(cupva::MemType::DRAM, nullptr))
    , m_prog{}
{
    if (!inBoxesReqs || !inScoresReqs || !inClassesReqs || !inCountReqs || !outBoxesReqs || !outScoresReqs ||
        !outClassesReqs || !outCountReqs)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor requirements cannot be null");
    }
    if (!(iouThreshold >= 0.0f && iouThreshold <= 1.0f))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "IoU threshold must be in [0, 1]");
    }

    if (inBoxesReqs->rank != 2 || inBoxesReqs->dtype != NVCV_DATA_TYPE_F32 || inBoxesReqs->shape[0] != 4 ||
        outBoxesReqs->rank != 2 || outBoxesReqs->dtype != NVCV_DATA_TYPE_F32 || outBoxesReqs->shape[0] != 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Boxes must use F32 SoA layout [4, N]");
    }
    m_maxInputBoxes  = inBoxesReqs->shape[1];
    m_maxOutputBoxes = outBoxesReqs->shape[1];

    if (inBoxesReqs->strides[1] != sizeof(float) ||
        inBoxesReqs->strides[0] < m_maxInputBoxes * static_cast<int64_t>(sizeof(float)) ||
        outBoxesReqs->strides[1] != sizeof(float) ||
        outBoxesReqs->strides[0] < m_maxOutputBoxes * static_cast<int64_t>(sizeof(float)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "NMS box tensors must have contiguous rows in SoA layout");
    }

    if (inScoresReqs->rank != 1 || inScoresReqs->dtype != NVCV_DATA_TYPE_F16 ||
        inScoresReqs->shape[0] != m_maxInputBoxes || inClassesReqs->rank != 1 ||
        inClassesReqs->dtype != NVCV_DATA_TYPE_S16 || inClassesReqs->shape[0] != m_maxInputBoxes ||
        inCountReqs->rank != 1 || inCountReqs->dtype != NVCV_DATA_TYPE_S32 || inCountReqs->shape[0] != 1 ||
        outScoresReqs->rank != 1 || outScoresReqs->dtype != NVCV_DATA_TYPE_F32 ||
        outScoresReqs->shape[0] != m_maxOutputBoxes || outClassesReqs->rank != 1 ||
        outClassesReqs->dtype != NVCV_DATA_TYPE_S32 || outClassesReqs->shape[0] != m_maxOutputBoxes ||
        outCountReqs->rank != 1 || outCountReqs->dtype != NVCV_DATA_TYPE_S32 || outCountReqs->shape[0] != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "NMS tensor types or extents are inconsistent");
    }

    if (inScoresReqs->strides[0] != sizeof(uint16_t) || inClassesReqs->strides[0] != sizeof(int16_t) ||
        inCountReqs->strides[0] != sizeof(int32_t) || outScoresReqs->strides[0] != sizeof(float) ||
        outClassesReqs->strides[0] != sizeof(int32_t) || outCountReqs->strides[0] != sizeof(int32_t))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "NMS vector tensors must be tightly packed");
    }

    // Validate dimensions
    if (m_maxInputBoxes <= 0 || m_maxInputBoxes > MAX_BOX_NUM_BEFORE_NMS_F32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Max input boxes exceeds VPU limit");
    }
    if (m_maxOutputBoxes <= 0 || m_maxOutputBoxes > MAX_BOX_NUM_AFTER_NMS)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Max output boxes exceeds VPU limit");
    }

    // Create PVA executable
    m_exec = GetBatchedNMSExec();

    // Create command program
    auto prog = CmdProgram::Create(*m_exec.get());

    // Set VPU parameters using struct
    char vpu_param_a[sizeof(batched_nms_vpu_param_t)];
    batched_nms_vpu_param_t *vpu_param = (batched_nms_vpu_param_t *)vpu_param_a;
    vpu_param->topk                    = m_maxOutputBoxes;
    vpu_param->threshold               = m_iouThreshold;
    vpu_param->max_input_boxes         = m_maxInputBoxes;
    prog["vpu_param"].set(vpu_param_a, sizeof(batched_nms_vpu_param_t));

    // Fetch the runtime count independently so payload DMA sizes can be set
    // before the payload data flow is triggered.
    SequenceDataFlow &inputCountDf = prog.addDataFlowHead<SequenceDataFlow>().handler(prog["inputCountTrig"]);

    inputCountDf.addTransfer()
        .tile(sizeof(int32_t))
        .src(m_inCountOp, 0)
        .srcDim1(1, 0)
        .dst(prog["numBuf"].ptr<int32_t>(), 0)
        .dstDim1(1, 0)
        .mode(TransferModeType::TILE);

    SequenceDataFlow &inputPayloadDf = prog.addDataFlowHead<SequenceDataFlow>().handler(prog["inputPayloadTrig"]);

    inputPayloadDf.addTransfer()
        .tile(m_maxInputBoxes * sizeof(float), 4)
        .src(m_inBoxesOp, inBoxesReqs->strides[0])
        .srcDim1(1, 0)
        .dst(prog["boxesBuf"].ptr<float>(), MAX_BOX_NUM_BEFORE_NMS_F32 * sizeof(float))
        .dstDim1(1, 0);

    inputPayloadDf.addTransfer()
        .tile(m_maxInputBoxes * sizeof(uint16_t))
        .src(m_inScoresOp, 0)
        .srcDim1(1, 0)
        .dst(prog["scoresBuf"].ptr<short>(), 0)
        .dstDim1(1, 0);

    inputPayloadDf.addTransfer()
        .tile(m_maxInputBoxes * sizeof(int16_t))
        .src(m_inClassesOp, 0)
        .srcDim1(1, 0)
        .dst(prog["labelsBuf"].ptr<short>(), 0)
        .dstDim1(1, 0);

    // Configure output data flow - all transfers CONTINUOUS (default, no .mode() call).
    // A single Trig/FlushAndTrig runs all transfers in sequence automatically.
    SequenceDataFlow &outputDf = prog.addDataFlowHead<SequenceDataFlow>().handler(prog["outputTrig"]);

    // Transfer 0: Output boxes (4 lines: left, top, right, bottom)
    // VPU stores: [left*N, top*N, right*N, bottom*N, score*N, class*N] where N=MAX_BOX_NUM_AFTER_NMS
    outputDf.addTransfer()
        .tile(m_maxOutputBoxes * sizeof(float), 4)
        .src(prog["outputBuf"].ptr<float>(), MAX_BOX_NUM_AFTER_NMS * sizeof(float))
        .srcDim1(1, 0)
        .dst(m_outBoxesOp, outBoxesReqs->strides[0])
        .dstDim1(1, 0);

    // Transfer 1: Output scores (5th array from outputBuf)
    outputDf.addTransfer()
        .tile(m_maxOutputBoxes * sizeof(float))
        .src(prog["outputBuf"].ptr<float>() + (4 * MAX_BOX_NUM_AFTER_NMS), 0)
        .srcDim1(1, 0)
        .dst(m_outScoresOp, 0)
        .dstDim1(1, 0);

    // Transfer 2: Output classes (6th array from outputBuf), S32 not F32
    outputDf.addTransfer()
        .tile(m_maxOutputBoxes * sizeof(int32_t))
        .src(prog["outputBuf"].ptr<int32_t>() + (5 * MAX_BOX_NUM_AFTER_NMS), 0)
        .srcDim1(1, 0)
        .dst(m_outClassesOp, 0)
        .dstDim1(1, 0);

    // Transfer 3: Output count
    outputDf.addTransfer()
        .tile(sizeof(int32_t))
        .src(prog["numBuf"].ptr<int32_t>(), 0)
        .srcDim1(1, 0)
        .dst(m_outCountOp, 0)
        .dstDim1(1, 0);

    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename StreamType>
void BatchedNMS::operator()(StreamType stream, const nvcv::Tensor &inBoxes, const nvcv::Tensor &inScores,
                            const nvcv::Tensor &inClasses, const nvcv::Tensor &inCount, const nvcv::Tensor &outBoxes,
                            const nvcv::Tensor &outScores, const nvcv::Tensor &outClasses, const nvcv::Tensor &outCount)
{
    auto inBoxesData    = inBoxes.exportData<nvcv::TensorDataStrided>();
    auto inScoresData   = inScores.exportData<nvcv::TensorDataStrided>();
    auto inClassesData  = inClasses.exportData<nvcv::TensorDataStrided>();
    auto inCountData    = inCount.exportData<nvcv::TensorDataStrided>();
    auto outBoxesData   = outBoxes.exportData<nvcv::TensorDataStrided>();
    auto outScoresData  = outScores.exportData<nvcv::TensorDataStrided>();
    auto outClassesData = outClasses.exportData<nvcv::TensorDataStrided>();
    auto outCountData   = outCount.exportData<nvcv::TensorDataStrided>();

    // Validate tensor requirements
    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_inBoxesTensorRequirements, inBoxesData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_inScoresTensorRequirements, inScoresData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_inClassesTensorRequirements, inClassesData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_inCountTensorRequirements, inCountData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_outBoxesTensorRequirements, outBoxesData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_outScoresTensorRequirements, outScoresData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_outClassesTensorRequirements, outClassesData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_outCountTensorRequirements, outCountData->cdata()))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Input and output tensors must have the same requirements that were used to create the operator");
    }

    // Validate base pointers
    if ((static_cast<void *>(inBoxesData->basePtr()) == nullptr) ||
        (static_cast<void *>(inScoresData->basePtr()) == nullptr) ||
        (static_cast<void *>(inClassesData->basePtr()) == nullptr) ||
        (static_cast<void *>(inCountData->basePtr()) == nullptr) ||
        (static_cast<void *>(outBoxesData->basePtr()) == nullptr) ||
        (static_cast<void *>(outScoresData->basePtr()) == nullptr) ||
        (static_cast<void *>(outClassesData->basePtr()) == nullptr) ||
        (static_cast<void *>(outCountData->basePtr()) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor base pointers must not be NULL");
    }

    // Set base pointers
    m_inBoxesOp.setBase(static_cast<float *>(static_cast<void *>(inBoxesData->basePtr())));
    m_inScoresOp.setBase(static_cast<int16_t *>(static_cast<void *>(inScoresData->basePtr())));
    m_inClassesOp.setBase(static_cast<int16_t *>(static_cast<void *>(inClassesData->basePtr())));
    m_inCountOp.setBase(static_cast<int *>(static_cast<void *>(inCountData->basePtr())));
    m_outBoxesOp.setBase(static_cast<float *>(static_cast<void *>(outBoxesData->basePtr())));
    m_outScoresOp.setBase(static_cast<float *>(static_cast<void *>(outScoresData->basePtr())));
    m_outClassesOp.setBase(static_cast<int *>(static_cast<void *>(outClassesData->basePtr())));
    m_outCountOp.setBase(static_cast<int *>(static_cast<void *>(outCountData->basePtr())));

    // Update data flows
    if (CupvaCmdProgramUpdateDataFlows(&m_prog) != CUPVA_ERROR_NONE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Update DataFlow failed");
    }

    // Submit to PVA
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
            const cupvaError_t submitStatus =
                CupvaCudaStreamSubmit(stream, submitCmds, cmdStatus, 1, CUPVA_IN_ORDER, -1, -1);
            if (submitStatus != CUPVA_ERROR_NONE)
            {
                const char *detail = nullptr;
                CupvaGetLastError(&detail);
                throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL,
                                      "PVA CUDA-stream Cmd Submit failed: CUPVA error %d%s%s", submitStatus,
                                      detail ? ": " : "", detail ? detail : "");
            }
        }
        else
        {
            ThrowCudaStreamNotSupported();
        }
    }
    else
    {
        static_assert(!sizeof(StreamType), "Unsupported stream type for BatchedNMS operator");
    }
}

BatchedNMS::~BatchedNMS()
{
    CupvaCmdDestroy(&m_prog);
}

} // namespace pva_solutions::priv

NVCVStatus pvaBatchedNMSCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements const *inBoxesReqs,
                               NVCVTensorRequirements const *inScoresReqs, NVCVTensorRequirements const *inClassesReqs,
                               NVCVTensorRequirements const *inCountReqs, NVCVTensorRequirements const *outBoxesReqs,
                               NVCVTensorRequirements const *outScoresReqs,
                               NVCVTensorRequirements const *outClassesReqs, NVCVTensorRequirements const *outCountReqs,
                               float iouThreshold)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        if (!inBoxesReqs || !inScoresReqs || !inClassesReqs || !inCountReqs || !outBoxesReqs || !outScoresReqs ||
            !outClassesReqs || !outCountReqs)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor requirements cannot be null");
        }

        *handle = reinterpret_cast<NVCVOperatorHandle>(
            new pva_solutions::priv::BatchedNMS(inBoxesReqs, inScoresReqs, inClassesReqs, inCountReqs, outBoxesReqs,
                                                outScoresReqs, outClassesReqs, outCountReqs, iouThreshold));
    });
}

template<typename StreamType>
NVCVStatus BatchedNMSSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle inBoxes,
                                NVCVTensorHandle inScores, NVCVTensorHandle inClasses, NVCVTensorHandle inCount,
                                NVCVTensorHandle outBoxes, NVCVTensorHandle outScores, NVCVTensorHandle outClasses,
                                NVCVTensorHandle outCount)
{
    return nvcv::ProtectCall([&] {
        nvcv::TensorWrapHandle inBox(inBoxes), inScore(inScores), inCls(inClasses), inCnt(inCount), outBox(outBoxes),
            outScore(outScores), outCls(outClasses), outCnt(outCount);
        pva_solutions::priv::ToDynamicRef<pva_solutions::priv::BatchedNMS>(handle)(stream, inBox, inScore, inCls, inCnt,
                                                                                   outBox, outScore, outCls, outCnt);
    });
}

// Explicit template instantiation
template NVCVStatus BatchedNMSSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle,
                                                        NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle,
                                                        NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle,
                                                        NVCVTensorHandle);
template NVCVStatus BatchedNMSSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle,
                                                       NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle,
                                                       NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle,
                                                       NVCVTensorHandle);

// SubmitImpl owns ProtectCall because the generated plain-C wrappers in
// PvaSubmitDefinitions.cpp call it directly as well as these C++ overloads.
NVCVStatus pvaBatchedNMSSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle inBoxes,
                               NVCVTensorHandle inScores, NVCVTensorHandle inClasses, NVCVTensorHandle inCount,
                               NVCVTensorHandle outBoxes, NVCVTensorHandle outScores, NVCVTensorHandle outClasses,
                               NVCVTensorHandle outCount)
{
    return BatchedNMSSubmitImpl(handle, stream, inBoxes, inScores, inClasses, inCount, outBoxes, outScores, outClasses,
                                outCount);
}

NVCVStatus pvaBatchedNMSSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle inBoxes,
                               NVCVTensorHandle inScores, NVCVTensorHandle inClasses, NVCVTensorHandle inCount,
                               NVCVTensorHandle outBoxes, NVCVTensorHandle outScores, NVCVTensorHandle outClasses,
                               NVCVTensorHandle outCount)
{
    return BatchedNMSSubmitImpl(handle, stream, inBoxes, inScores, inClasses, inCount, outBoxes, outScores, outClasses,
                                outCount);
}
