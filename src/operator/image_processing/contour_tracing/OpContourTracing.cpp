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

#include "OpContourTracing.hpp"

#include "device/contour_tracing_param.h"

#include <TensorUtils.hpp>
#include <cupva_host_conversions.hpp>

#include <cupva_host.h>
#include <cupva_host_cuda.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <cstdint>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(contour_tracing_dev);

namespace pva_solutions::priv {

template<typename T>
std::mutex ContourTracing<T>::s_execMapMutex;

template<typename T>
std::map<typename ContourTracing<T>::ExecKey, std::weak_ptr<cupva::Executable>>
    ContourTracing<T>::s_contourTracingExecutables;

template<typename T>
std::shared_ptr<cupva::Executable> ContourTracing<T>::GetContourTracingExec()
{
    static IOperator::ExecutableData sExecData = {CUPVA_EXECUTABLE_DATA(contour_tracing_dev),
                                                  CUPVA_EXECUTABLE_SIZE(contour_tracing_dev)};

    ExecKey execKey = std::make_tuple(cupva::Context::GetCurrent());
    return IOperator::GetExecutable<ExecKey>(sExecData, execKey, s_contourTracingExecutables, s_execMapMutex);
}

template<typename T>
ContourTracing<T>::ContourTracing(NVCVTensorRequirements const *const inTensorRequirements,
                                  NVCVTensorRequirements const *const rectTensorRequirements,
                                  NVCVTensorRequirements const *const outTensorRequirements,
                                  NVCVTensorRequirements const *const lenTensorRequirements)
    : m_inTensorRequirements{*inTensorRequirements}
    , m_rectTensorRequirements{*rectTensorRequirements}
    , m_outTensorRequirements{*outTensorRequirements}
    , m_lenTensorRequirements{*lenTensorRequirements}
    , m_srcOp(cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr))
    , m_rectOp(cupva::OffsetPointer<int32_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_dstOp(cupva::OffsetPointer<uint16_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_lenOp(cupva::OffsetPointer<uint16_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_prog{}
{
    NVCVTensorLayout HWCLayout;
    nvcvTensorLayoutMake("HWC", &HWCLayout);
    NVCVTensorLayout WLayout;
    nvcvTensorLayoutMake("W", &WLayout);

    if (nvcvTensorLayoutCompare(inTensorRequirements->layout, HWCLayout) != 0 ||
        nvcvTensorLayoutCompare(outTensorRequirements->layout, HWCLayout) != 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output tensors must have HWC layout");
    }
    if (nvcvTensorLayoutCompare(rectTensorRequirements->layout, WLayout) != 0 ||
        nvcvTensorLayoutCompare(lenTensorRequirements->layout, WLayout) != 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Rect and length tensors must have W layout");
    }

    if (rectTensorRequirements->shape[0] > MAX_RECT_NUM || outTensorRequirements->shape[0] > MAX_RECT_NUM ||
        lenTensorRequirements->shape[0] > MAX_RECT_NUM)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Rect, output and length tensors must have no more than MAX_RECT_NUM ROIs");
    }
    if (outTensorRequirements->shape[1] > MAX_RECT_AREA)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Each ROI in output tensor must have no more than MAX_RECT_AREA contour pixels");
    }
    if (inTensorRequirements->shape[2] != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input tensor must have 1 channel");
    }
    if (outTensorRequirements->shape[2] != 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output tensor must have 2 channels");
    }

    if (rectTensorRequirements->dtype != NVCV_DATA_TYPE_4S32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Rect tensor must have NVCV_DATA_TYPE_4S32 data type");
    }
    if (outTensorRequirements->dtype != NVCV_DATA_TYPE_U16)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output tensor must have NVCV_DATA_TYPE_U16 data type");
    }
    if (lenTensorRequirements->dtype != NVCV_DATA_TYPE_U16)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Length tensor must have NVCV_DATA_TYPE_U16 data type");
    }

    m_exec    = GetContourTracingExec();
    auto prog = CmdProgram::Create(*m_exec.get());

    auto inHdl   = prog["inHdl"];
    auto rectHdl = prog["rectHdl"];
    auto outHdl  = prog["outHdl"];
    auto lenHdl  = prog["lenHdl"];

    const int32_t inLinePitchBytes  = static_cast<int32_t>(m_inTensorRequirements.strides[0]);
    const int32_t inLinePitch       = static_cast<int32_t>(inLinePitchBytes / sizeof(T));
    const int32_t maxContourPixels  = static_cast<int32_t>(m_outTensorRequirements.shape[1]);
    const int32_t maxRectWidthBytes = static_cast<int32_t>(MAX_RECT_WIDTH * sizeof(T));
    const int32_t typicalOutputNum  = 1024;

    prog["inLinePitch"]      = inLinePitch;
    prog["maxContourPixels"] = maxContourPixels;

    uint16_t dirTable[PVA_CONTOUR_TRACING_DIRECTION_TABLE_ENTRIES];
    pvaContourTracingBuildDirectionTable(dirTable);
    prog["dirTable"].set(dirTable, NUM_NEIGHBOR_PERM);

    T *inBuf         = prog["inBuf"].ptr<T>();
    int32_t *rectBuf = prog["rectBuf"].ptr<int32_t>();
    uint16_t *outBuf = prog["outBuf"].ptr<uint16_t>();
    uint16_t *lenBuf = prog["lenBuf"].ptr<uint16_t>();

    const int32_t rectNum  = m_rectTensorRequirements.shape[0];
    const int32_t rectSize = rectNum * 4;

    // Transfer bounding rectangles from DRAM to VMEM
    RasterDataFlow &rectDataFlow = prog.addDataFlowHead<RasterDataFlow>(0);
    rectDataFlow.handler(rectHdl).src(m_rectOp, rectSize, 1, rectSize).tileBuffer(rectBuf).tile(rectSize, 1);

    // Transfer input ROI from DRAM to VMEM
    SequenceDataFlow &inDataFlow = prog.addDataFlowHead<SequenceDataFlow>(1);
    inDataFlow.handler(inHdl)
        .addTransfer()
        .src(m_srcOp, inLinePitchBytes)
        .dst(inBuf, maxRectWidthBytes)
        .tile(maxRectWidthBytes, MAX_RECT_HEIGHT);

    // Transfer output contour from VMEM to DRAM
    SequenceDataFlow &outDataFlow = prog.addDataFlowHead<SequenceDataFlow>(1);
    outDataFlow.handler(outHdl)
        .addTransfer()
        .src(outBuf, 0)
        .dst(m_dstOp, 0)
        .tile(typicalOutputNum * 2 * sizeof(uint16_t), 1);

    // Transfer contour lengths from VMEM to DRAM
    RasterDataFlow &lenDataFlow = prog.addDataFlowHead<RasterDataFlow>(2);
    lenDataFlow.handler(lenHdl).tileBuffer(lenBuf).dst(m_lenOp, rectNum, 1, rectNum).tile(rectNum, 1);

    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename T>
template<typename S>
void ContourTracing<T>::operator()(S stream, const nvcv::Tensor &inTensor, const nvcv::Tensor &rectTensor,
                                   const nvcv::Tensor &outTensor, const nvcv::Tensor &lenTensor, const int32_t numRects)
{
    auto inTensorData   = inTensor.exportData<nvcv::TensorDataStrided>();
    auto rectTensorData = rectTensor.exportData<nvcv::TensorDataStrided>();
    auto outTensorData  = outTensor.exportData<nvcv::TensorDataStrided>();
    auto lenTensorData  = lenTensor.exportData<nvcv::TensorDataStrided>();

    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_inTensorRequirements, inTensorData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_rectTensorRequirements, rectTensorData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_outTensorRequirements, outTensorData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_lenTensorRequirements, lenTensorData->cdata()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Tensors must have the same requirements that were used to create the operator");
    }
    if ((static_cast<void *>(inTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(rectTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(outTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(lenTensorData->basePtr()) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor base pointers must not be NULL");
    }
    if (numRects <= 0 || numRects > m_rectTensorRequirements.shape[0] || numRects > m_outTensorRequirements.shape[0] ||
        numRects > m_lenTensorRequirements.shape[0])
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Number of ROIs is out of range");
    }

    int32_t *rectPtr = NULL;
    CupvaMemGetHostPointer((void **)&rectPtr, (void *)rectTensorData->basePtr());
    int32_t imageHeight = static_cast<int32_t>(m_inTensorRequirements.shape[0]);
    int32_t imageWidth  = static_cast<int32_t>(m_inTensorRequirements.shape[1]);
    for (int32_t i = 0; i < numRects; i++)
    {
        // Rect tensor stores NVCVRectI as {x, y, width, height}.
        int32_t x      = rectPtr[i * 4];
        int32_t y      = rectPtr[i * 4 + 1];
        int32_t width  = rectPtr[i * 4 + 2];
        int32_t height = rectPtr[i * 4 + 3];

        bool isValidSize   = width > 0 && height > 0 && width <= MAX_RECT_WIDTH && height <= MAX_RECT_HEIGHT;
        bool isEvenSize    = (width % 2) == 0 && (height % 2) == 0;
        bool isInsideImage = x >= 0 && y >= 0 && x + width <= imageWidth && y + height <= imageHeight;

        if (!isValidSize || !isEvenSize || !isInsideImage)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "ROI parameter is invalid");
        }
    }

    // Set number of ROIs to process
    cupvaParameter_t numRectsParam;
    CupvaCmdProgramGetParameter(&m_prog, &numRectsParam, "numRects");
    CupvaParameterSetValueScalar(&numRectsParam, &numRects, sizeof(int32_t));

    // Set external memory pointers for SQDF.
    cupvaParameter_t inExtPtr;
    CupvaCmdProgramGetParameter(&m_prog, &inExtPtr, "inExtPtr");
    CupvaParameterSetValuePointer(&inExtPtr, static_cast<void *>(inTensorData->basePtr()));
    cupvaParameter_t outExtPtr;
    CupvaCmdProgramGetParameter(&m_prog, &outExtPtr, "outExtPtr");
    CupvaParameterSetValuePointer(&outExtPtr, static_cast<void *>(outTensorData->basePtr()));

    m_srcOp.setBase(static_cast<T *>(static_cast<void *>(inTensorData->basePtr())));
    m_rectOp.setBase(static_cast<int32_t *>(static_cast<void *>(rectTensorData->basePtr())));
    m_dstOp.setBase(static_cast<uint16_t *>(static_cast<void *>(outTensorData->basePtr())));
    m_lenOp.setBase(static_cast<uint16_t *>(static_cast<void *>(lenTensorData->basePtr())));

    cupvaError_t err = CupvaCmdProgramUpdateDataFlows(&m_prog);
    if (err != CUPVA_ERROR_NONE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "Failed to update data flows");
    }

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
        static_assert(!sizeof(S), "Unsupported stream type for ContourTracing operator");
    }

    if (!(err == CUPVA_ERROR_NONE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Submit failed");
    }
}

template<typename T>
ContourTracing<T>::~ContourTracing()
{
    CupvaCmdDestroy(&m_prog);
}

// Explicit instantiation of the template to support the types we care about
template class ContourTracing<uint8_t>;

} // namespace pva_solutions::priv

NVCVStatus pvaContourTracingCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements const *const inTensorRequirements,
                                   NVCVTensorRequirements const *const rectTensorRequirements,
                                   NVCVTensorRequirements const *const outTensorRequirements,
                                   NVCVTensorRequirements const *const lenTensorRequirements)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        if (inTensorRequirements == nullptr || rectTensorRequirements == nullptr || outTensorRequirements == nullptr ||
            lenTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVTensorRequirements must not be NULL");
        }

        switch (inTensorRequirements->dtype)
        {
        case NVCV_DATA_TYPE_U8:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::ContourTracing<uint8_t>(
                inTensorRequirements, rectTensorRequirements, outTensorRequirements, lenTensorRequirements));
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input data type");
        }
    });
}

template<typename StreamType>
NVCVStatus ContourTracingSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle const inTensor,
                                    NVCVTensorHandle const rectTensor, NVCVTensorHandle outTensor,
                                    NVCVTensorHandle lenTensor, const int32_t numRects)
{
    return nvcv::ProtectCall([&] {
        nvcv::TensorWrapHandle input(inTensor), rect(rectTensor), output(outTensor), len(lenTensor);

        NVCVDataType inDataType;
        nvcvTensorGetDataType(inTensor, &inDataType);

        switch (inDataType)
        {
        case NVCV_DATA_TYPE_U8:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::ContourTracing<uint8_t>>(handle)(
                stream, input, rect, output, len, numRects);
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input data type");
        }
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus ContourTracingSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle const,
                                                            NVCVTensorHandle const, NVCVTensorHandle, NVCVTensorHandle,
                                                            const int32_t);
template NVCVStatus ContourTracingSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle const,
                                                           NVCVTensorHandle const, NVCVTensorHandle, NVCVTensorHandle,
                                                           const int32_t);

// C++ overload implementations
NVCVStatus pvaContourTracingSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle const inTensor,
                                   NVCVTensorHandle const rectTensor, NVCVTensorHandle outTensor,
                                   NVCVTensorHandle lenTensor, const int32_t numRects)
{
    return ContourTracingSubmitImpl(handle, stream, inTensor, rectTensor, outTensor, lenTensor, numRects);
}

NVCVStatus pvaContourTracingSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle const inTensor,
                                   NVCVTensorHandle const rectTensor, NVCVTensorHandle outTensor,
                                   NVCVTensorHandle lenTensor, const int32_t numRects)
{
    return ContourTracingSubmitImpl(handle, stream, inTensor, rectTensor, outTensor, lenTensor, numRects);
}

NVCVStatus pvaContourTracingBuildDirectionTable(uint16_t *dirTable)
{
    return nvcv::ProtectCall([&] {
        if (dirTable == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Direction table must not be NULL");
        }

        for (uint32_t bitmap = 0; bitmap < NUM_NEIGHBOR_PERM; bitmap++)
        {
            uint16_t entry = 0;
            for (uint8_t prevDir = 0; prevDir < NUM_NEIGHBOR; prevDir += 2)
            {
                uint8_t nextDir = DIR_NONE;
                if ((bitmap & 0x55) != 0x55) // at least one 4-connectivity neighbor is zero
                {
                    uint8_t backDir = (prevDir + NUM_NEIGHBOR / 2) % NUM_NEIGHBOR;

                    // make sure backDir points to a zero neighbor
                    while (bitmap & (1U << backDir))
                    {
                        backDir = (backDir + 1) % NUM_NEIGHBOR;
                    }

                    for (uint8_t i = 1; i < NUM_NEIGHBOR; i++)
                    {
                        uint8_t d = (backDir + i) % NUM_NEIGHBOR;
                        if (bitmap & (1U << d))
                        {
                            nextDir = d;
                            break;
                        }
                    }
                }
                entry |= static_cast<uint16_t>(nextDir) << (2 * prevDir);
            }
            dirTable[bitmap] = entry;
        }
    });
}
