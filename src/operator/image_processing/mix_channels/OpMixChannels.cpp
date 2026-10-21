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

#include "OpMixChannels.hpp"

#include "device/mix_channels_param.h"

#include <IOperator.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <nvcv/Tensor.hpp>

#include <ErrorCheckMacros.h>
#include <OpMixChannels.h>
#include <PvaOperatorTypes.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <cstdint>
#include <vector>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_split_rgba8_to_u8_dev);
CUPVA_DECLARE_EXECUTABLE(pva_merge_u8_to_rgba8_dev);

namespace pva_solutions::priv {

std::mutex MixChannels::s_execMapMutex;

std::map<MixChannels::ExecKey, std::weak_ptr<cupva::Executable>> MixChannels::s_mixChannelsExecutables;

std::map<MixChannels::ExecDataKey, IOperator::ExecutableData> createExecDataMap()
{
    return {
        {{PVA_SPLIT_RGBA8_TO_U8},
         {CUPVA_EXECUTABLE_DATA(pva_split_rgba8_to_u8_dev), CUPVA_EXECUTABLE_SIZE(pva_split_rgba8_to_u8_dev)}},
        {{PVA_MERGE_U8_TO_RGBA8},
         {CUPVA_EXECUTABLE_DATA(pva_merge_u8_to_rgba8_dev), CUPVA_EXECUTABLE_SIZE(pva_merge_u8_to_rgba8_dev)}},
    };
}

std::shared_ptr<cupva::Executable> MixChannels::GetMixChannelsExec(ExecDataKey execDataKey)
{
    static auto execDataMap = createExecDataMap();

    auto execDataIt = execDataMap.find(execDataKey);
    if (execDataIt != execDataMap.end())
    {
        // Create execKey by combining execDataKey and context
        ExecKey execKey = std::tuple_cat(execDataKey, std::make_tuple(cupva::Context::GetCurrent()));
        return IOperator::GetExecutable<ExecKey>(execDataIt->second, execKey, s_mixChannelsExecutables, s_execMapMutex);
    }
    else
    {
        throw std::invalid_argument("Unsupported type or parameters");
    }
}

void MixChannels::createParamCheck(const char *prefix, int inTensorNumRef, int outTensorNumRef, int srcImgCRef,
                                   int dstImgCRef, NVCVDataType srcDataTypeRef, NVCVDataType dstDataTypeRef)
{
    if (m_inputTensorReqs.size() != inTensorNumRef)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s should have %d input tensor(s).", prefix,
                              inTensorNumRef);
    }

    if (m_outputTensorReqs.size() != outTensorNumRef)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s should have %d output tensor(s).", prefix,
                              outTensorNumRef);
    }

    for (int i = 0; i < inTensorNumRef; i++)
    {
        if (m_inputTensorReqs[i].shape[2] != srcImgCRef)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s input tensor(s) should be %d channel(s).",
                                  prefix, srcImgCRef);
        }
        if (m_inputTensorReqs[i].dtype != srcDataTypeRef)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "%s input tensor(s) contain unsupported data type.", prefix);
        }
    }

    for (int i = 0; i < outTensorNumRef; i++)
    {
        if (m_outputTensorReqs[i].shape[2] != dstImgCRef)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s output tensor(s) should be %d channel(s).",
                                  prefix, dstImgCRef);
        }
        if (m_outputTensorReqs[i].dtype != dstDataTypeRef)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "%s output tensor(s) contain unsupported data type.", prefix);
        }
    }
}

MixChannels::MixChannels(NVCVTensorRequirements **inTensorReqs, NVCVTensorRequirements **outTensorReqs,
                         int inTensorCount, int outTensorCount, PVAMixChannelsCode code)
{
    // Initialize input tensor requirements vector
    for (size_t i = 0; i < inTensorCount; ++i)
    {
        m_inputTensorReqs.push_back(*inTensorReqs[i]);
    }

    // Initialize output tensor requirements vector
    for (size_t i = 0; i < outTensorCount; ++i)
    {
        m_outputTensorReqs.push_back(*outTensorReqs[i]);
    }

    const int32_t imgW    = inTensorReqs[0]->shape[1];
    const int32_t imgH    = inTensorReqs[0]->shape[0];
    const int32_t srcImgC = inTensorReqs[0]->shape[2];
    const int32_t dstImgC = outTensorReqs[0]->shape[2];

    NVCVTensorLayout refLayout;
    nvcvTensorLayoutMake("HWC", &refLayout);

    for (int i = 0; i < inTensorCount; i++)
    {
        if (nvcvTensorLayoutCompare(m_inputTensorReqs[i].layout, refLayout) != 0)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor(s) must have HWC layout");
        }

        if (!(m_inputTensorReqs[i].shape[1] == imgW && m_inputTensorReqs[i].shape[0] == imgH))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor(s) must have same width and height");
        }
    }
    for (int i = 0; i < outTensorCount; i++)
    {
        if (nvcvTensorLayoutCompare(m_outputTensorReqs[i].layout, refLayout) != 0)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor(s) must have HWC layout");
        }

        if (!(m_outputTensorReqs[i].shape[1] == imgW && m_outputTensorReqs[i].shape[0] == imgH))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor(s) must have same width and height");
        }
    }

    m_exec    = GetMixChannelsExec(std::make_tuple(code));
    auto prog = CmdProgram::Create(*m_exec.get());

    const int32_t srcImgH  = imgH;
    const int32_t srcImgW  = imgW * srcImgC;
    const int32_t srcTileW = TILE_W * srcImgC;
    const int32_t srcTileH = TILE_H;

    const int32_t dstImgH  = imgH;
    const int32_t dstImgW  = imgW * dstImgC;
    const int32_t dstTileW = TILE_W * dstImgC;
    const int32_t dstTileH = TILE_H;

    const int32_t niterX = (imgW + TILE_W - 1) / TILE_W;
    const int32_t niterY = (imgH + TILE_H - 1) / TILE_H;
    prog["niter"]        = niterX * niterY;

    std::vector<uint8_t *> srcBufVec;
    std::vector<uint8_t *> dstBufVec;
    switch (code)
    {
    case PVA_SPLIT_RGBA8_TO_U8:
    {
        createParamCheck("PVA_SPLIT_RGBA8_TO_U8", 1, 4, 4, 1, NVCV_DATA_TYPE_U8, NVCV_DATA_TYPE_U8);

        srcBufVec = {prog["srcBuf"].ptr<uint8_t>()};
        dstBufVec = {prog["dst0Buf"].ptr<uint8_t>(), prog["dst1Buf"].ptr<uint8_t>(), prog["dst2Buf"].ptr<uint8_t>(),
                     prog["dst3Buf"].ptr<uint8_t>()};

        break;
    }

    case PVA_MERGE_U8_TO_RGBA8:
    {
        createParamCheck("PVA_MERGE_U8_TO_RGBA8", 4, 1, 1, 4, NVCV_DATA_TYPE_U8, NVCV_DATA_TYPE_U8);

        srcBufVec = {prog["src0Buf"].ptr<uint8_t>(), prog["src1Buf"].ptr<uint8_t>(), prog["src2Buf"].ptr<uint8_t>(),
                     prog["src3Buf"].ptr<uint8_t>()};
        dstBufVec = {prog["dstBuf"].ptr<uint8_t>()};

        break;
    }
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported case code.");
    }

    for (int i = 0; i < srcBufVec.size(); i++)
    {
        const int32_t inputLinePitch = m_inputTensorReqs[i].strides[0] / sizeof(uint8_t);
        m_srcOps.push_back(cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM, nullptr));
        prog.addDataFlowHead<RasterDataFlow>()
            .handler(prog["srcDFHandler"])
            .src(m_srcOps[i], srcImgW, srcImgH, inputLinePitch)
            .tileBuffer(srcBufVec[i])
            .tile(srcTileW, srcTileH)
            .transpose(TRANS_MODE_1);
    }
    for (int i = 0; i < dstBufVec.size(); i++)
    {
        const int32_t outputLinePitch = m_outputTensorReqs[i].strides[0] / sizeof(uint8_t);
        m_dstOps.push_back(cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM, nullptr));
        prog.addDataFlowHead<RasterDataFlow>()
            .handler(prog["dstDFHandler"])
            .tileBuffer(dstBufVec[i])
            .tile(dstTileW, dstTileH)
            .transpose(TRANS_MODE_1)
            .dst(m_dstOps[i], dstImgW, dstImgH, outputLinePitch);
    }

    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename S>
void MixChannels::operator()(S stream, NVCVTensorHandle *inHandles, NVCVTensorHandle *outHandles, int inTensorCount,
                             int outTensorCount)
{

    if ((inTensorCount != m_inputTensorReqs.size()) || (outTensorCount != m_outputTensorReqs.size()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "The number of input/output tensor(s) must be equal to the number of tensor(s) "
                              "requirements that was used to create the operator.");
    }

    for (int i = 0; i < inTensorCount; ++i)
    {
        nvcv::Tensor input = nvcv::TensorWrapHandle(inHandles[i]);
        auto inTensorData  = input.exportData<nvcv::TensorDataStrided>();

        if (!pva_solutions::utils::AreTensorRequirementsEqual(m_inputTensorReqs[i], inTensorData->cdata()))
        {
            throw nvcv::Exception(
                nvcv::Status::ERROR_INVALID_ARGUMENT,
                "Input tensor %d must have the same requirements that was used to create the operator.", i);
        }
        if (static_cast<void *>(inTensorData->basePtr()) == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor base pointers must not be NULL");
        }

        m_srcOps[i].setBase(static_cast<uint8_t *>(static_cast<void *>(inTensorData->basePtr())));
    }

    for (int i = 0; i < outTensorCount; ++i)
    {
        nvcv::Tensor output = nvcv::TensorWrapHandle(outHandles[i]);
        auto outTensorData  = output.exportData<nvcv::TensorDataStrided>();

        if (!pva_solutions::utils::AreTensorRequirementsEqual(m_outputTensorReqs[i], outTensorData->cdata()))
        {
            throw nvcv::Exception(
                nvcv::Status::ERROR_INVALID_ARGUMENT,
                "Output tensor %d must have the same requirements that was used to create the operator.", i);
        }
        if (static_cast<void *>(outTensorData->basePtr()) == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor base pointers must not be NULL");
        }

        m_dstOps[i].setBase(static_cast<uint8_t *>(static_cast<void *>(outTensorData->basePtr())));
    }

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
        static_assert(!sizeof(S), "Unsupported stream type for MixChannels operator");
    }
}

// Explicit template instantiation
template void MixChannels::operator()<cupvaStream_t>(cupvaStream_t, NVCVTensorHandle *, NVCVTensorHandle *, int, int);
template void MixChannels::operator()<cudaStream_t>(cudaStream_t, NVCVTensorHandle *, NVCVTensorHandle *, int, int);

MixChannels::~MixChannels()
{
    CupvaCmdDestroy(&m_prog);
}

} // namespace pva_solutions::priv

NVCVStatus pvaMixChannelsCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements **inTensorReqPtrs,
                                NVCVTensorRequirements **outTensorReqPtrs, int inTensorCount, int outTensorCount,
                                PVAMixChannelsCode code)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }

        for (int i = 0; i < inTensorCount; ++i)
        {
            if (inTensorReqPtrs[i] == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to input TensorRequirements must not be NULL");
            }
        }

        for (int i = 0; i < outTensorCount; ++i)
        {
            if (outTensorReqPtrs[i] == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to output TensorRequirements must not be NULL");
            }
        }

        *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::MixChannels(
            inTensorReqPtrs, outTensorReqPtrs, inTensorCount, outTensorCount, code));
    });
}

template<typename StreamType>
NVCVStatus MixChannelsSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle *inHandles,
                                 NVCVTensorHandle *outHandles, int inTensorCount, int outTensorCount)
{
    return nvcv::ProtectCall([&] {
        pva_solutions::priv::ToDynamicRef<pva_solutions::priv::MixChannels>(handle)(stream, inHandles, outHandles,
                                                                                    inTensorCount, outTensorCount);
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus MixChannelsSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle *,
                                                         NVCVTensorHandle *, int, int);
template NVCVStatus MixChannelsSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle *,
                                                        NVCVTensorHandle *, int, int);

// C++ overload implementations
NVCVStatus pvaMixChannelsSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle *inHandles,
                                NVCVTensorHandle *outHandles, int inTensorCount, int outTensorCount)
{
    return MixChannelsSubmitImpl(handle, stream, inHandles, outHandles, inTensorCount, outTensorCount);
}

NVCVStatus pvaMixChannelsSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle *inHandles,
                                NVCVTensorHandle *outHandles, int inTensorCount, int outTensorCount)
{
    return MixChannelsSubmitImpl(handle, stream, inHandles, outHandles, inTensorCount, outTensorCount);
}