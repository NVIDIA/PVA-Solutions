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

#include "OpHOG.hpp"

#include "device/hog_tdata.h"

#include <IOperator.hpp>
#include <ImageUtils.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>

#include <OpHOG.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <cmath>
#include <cstdint>
#include <vector>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_hog_dev)

namespace pva_solutions::priv {

template<typename T>
std::mutex HOG<T>::s_execMapMutex;

template<typename T>
std::map<typename HOG<T>::ExecKey, std::weak_ptr<cupva::Executable>> HOG<T>::s_HOGExecutables;

template<typename T>
std::map<typename HOG<T>::ExecDataKey, IOperator::ExecutableData> createExecDataMap();

template<>
std::map<typename HOG<uint8_t>::ExecDataKey, IOperator::ExecutableData> createExecDataMap<uint8_t>()
{
    return {
        {{NVCV_IMAGE_FORMAT_RGB8p}, {CUPVA_EXECUTABLE_DATA(pva_hog_dev), CUPVA_EXECUTABLE_SIZE(pva_hog_dev)}},
    };
}

template<typename T>
std::shared_ptr<cupva::Executable> HOG<T>::GetHOGExec(ExecDataKey execDataKey)
{
    static auto execDataMap = createExecDataMap<T>();

    auto execDataIt = execDataMap.find(execDataKey);
    if (execDataIt != execDataMap.end())
    {
        // Create execKey by combining execDataKey and context
        ExecKey execKey = std::tuple_cat(execDataKey, std::make_tuple(cupva::Context::GetCurrent()));
        return GetExecutable(execDataIt->second, execKey, s_HOGExecutables, s_execMapMutex);
    }
    else
    {
        throw std::invalid_argument("Unsupported type or parameters");
    }
}

template<typename T>
HOG<T>::HOG(NVCVImageRequirements *imageRequirements, NVCVTensorRequirements *tensorRequirements)
    : m_imageRequirements{*imageRequirements}
    , m_tensorRequirements{*tensorRequirements}
    , m_srcOp{cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr),
              cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr),
              cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr),
              cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr)}
    , m_dstOp(cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr))
    , m_prog{}
{
    int32_t imageWidth     = imageRequirements->width;
    int32_t imageHeight    = imageRequirements->height;
    int32_t imageLinePitch = imageRequirements->planeRowStride[0] / sizeof(T);

    NVCVImageFormat imageFormat = imageRequirements->format;

    switch (imageFormat)
    {
    case NVCV_IMAGE_FORMAT_RGB8p:
        break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported image format");
    }

    if (imageWidth != INPUT_WIDTH * B_WIDTH || imageHeight != INPUT_HEIGHT * B_HEIGHT)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Image width and height must be equal to %d and %d",
                              INPUT_WIDTH * B_WIDTH, INPUT_HEIGHT * B_HEIGHT);
    }

    // Select executable based on image format instead of kernel size
    m_exec    = GetHOGExec(std::make_tuple(imageFormat));
    auto prog = CmdProgram::Create(*m_exec.get());

    nvcvImageFormatGetNumPlanes(imageFormat, &m_imagePlaneCount);

    T *input_r_v = prog["input_r_v"].ptr<T>();
    T *input_g_v = prog["input_g_v"].ptr<T>();
    T *input_b_v = prog["input_b_v"].ptr<T>();
    T *output_v  = prog["output_v"].ptr<T>();

    auto inRHdl = prog["inRHdl"];
    auto inGHdl = prog["inGHdl"];
    auto inBHdl = prog["inBHdl"];
    auto outHdl = prog["outHdl"];

    const int32_t inputWidth  = FEATURE_WIDTH * CELL_SIZE;
    const int32_t inputHeight = FEATURE_HEIGHT * CELL_SIZE;
    const int32_t planeSize   = INPUT_SIZE * B_SIZE;
    const int32_t rgbOfst     = CELL_SIZE * RGB_LP + CELL_SIZE;

    SequenceDataFlow &inputRDataFlow = prog.addDataFlowHead<SequenceDataFlow>();
    inputRDataFlow.handler(inRHdl)
        .addTransfer()
        .src(m_srcOp[0], inputWidth * B_WIDTH)
        .dst(input_r_v + rgbOfst, RGB_LP)
        .tile(inputWidth, inputHeight)
        .srcDim1(B_WIDTH, inputWidth)
        .srcDim2(B_HEIGHT, INPUT_SIZE * B_WIDTH)
        .dstDim1(2, RGB_BUF_SIZE)
        .dstDim2(B_WIDTH * B_HEIGHT / 2, 0)
        .mode(TransferModeType::TILE);
    SequenceDataFlow &inputGDataFlow = prog.addDataFlowHead<SequenceDataFlow>();
    inputGDataFlow.handler(inGHdl)
        .addTransfer()
        .src(m_srcOp[1], inputWidth * B_WIDTH)
        .dst(input_g_v + rgbOfst, RGB_LP)
        .tile(inputWidth, inputHeight)
        .srcDim1(B_WIDTH, inputWidth)
        .srcDim2(B_HEIGHT, INPUT_SIZE * B_WIDTH)
        .dstDim1(2, RGB_BUF_SIZE)
        .dstDim2(B_WIDTH * B_HEIGHT / 2, 0)
        .mode(TransferModeType::TILE);
    SequenceDataFlow &inputBDataFlow = prog.addDataFlowHead<SequenceDataFlow>();
    inputBDataFlow.handler(inBHdl)
        .addTransfer()
        .src(m_srcOp[2], inputWidth * B_WIDTH)
        .dst(input_b_v + rgbOfst, RGB_LP)
        .tile(inputWidth, inputHeight)
        .srcDim1(B_WIDTH, inputWidth)
        .srcDim2(B_HEIGHT, INPUT_SIZE * B_WIDTH)
        .dstDim1(2, RGB_BUF_SIZE)
        .dstDim2(B_WIDTH * B_HEIGHT / 2, 0)
        .mode(TransferModeType::TILE);

    SequenceDataFlow &outDataFlow = prog.addDataFlowHead<SequenceDataFlow>();
    outDataFlow.handler(outHdl)
        .addTransfer()
        .src(output_v, FEATURE_WIDTH)
        .dst(m_dstOp, FULL_OUT_WIDTH)
        .tile(FEATURE_WIDTH, FEATURE_HEIGHT)
        .srcDim1(ORIENTATIONS_NUM, FEATURE_SIZE)
        .srcDim2(2, (FEATURE_SIZE * ORIENTATIONS_NUM + 64))
        .srcDim3(B_WIDTH * B_HEIGHT / 2, 0)
        .dstDim1(ORIENTATIONS_NUM, FULL_OUT_SIZE)
        .dstDim2(B_WIDTH, FEATURE_WIDTH)
        .dstDim3(B_HEIGHT, FULL_OUT_WIDTH * FEATURE_HEIGHT)
        .mode(TransferModeType::DIM1);

    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename T>
template<typename S>
void HOG<T>::operator()(S stream, const nvcv::Image &in, const nvcv::Tensor &out)
{
    auto inImageData   = in.exportData<nvcv::ImageDataStrided>();
    auto outTensorData = out.exportData<nvcv::TensorDataStrided>();

    if (!pva_solutions::utils::AreImageRequirementsEqual(m_imageRequirements, inImageData->cdata()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input image must have the same requirements that was used to create the operator");
    }

    if ((static_cast<void *>(inImageData->plane(0).basePtr) == nullptr) ||
        (static_cast<void *>(outTensorData->basePtr()) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Image/Tensor base pointers must not be NULL");
    }

    for (int32_t i = 0; i < m_imagePlaneCount; i++)
    {
        m_srcOp[i].setBase(static_cast<T *>(static_cast<void *>(inImageData->plane(i).basePtr)));
    }
    m_dstOp.setBase(static_cast<T *>(static_cast<void *>(outTensorData->basePtr())));

    cupvaError_t err = CupvaCmdProgramUpdateDataFlows(&m_prog);

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
        static_assert(!sizeof(S), "Unsupported stream type for HOG operator");
    }

    if (!(err == CUPVA_ERROR_NONE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Submit failed");
    }
}

template<typename T>
HOG<T>::~HOG()
{
    CupvaCmdDestroy(&m_prog);
}

// Explicit instantiation of the template to support the types we care about
template class HOG<uint8_t>;

} // namespace pva_solutions::priv

NVCVStatus pvaHOGCreate(NVCVOperatorHandle *handle, NVCVImageRequirements *imageRequirements,
                        NVCVTensorRequirements *tensorRequirements)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        if (imageRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to ImageRequirements must not be NULL");
        }
        if (tensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to TensorRequirements must not be NULL");
        }

        NVCVDataType imageDataType;
        nvcvImageFormatGetPlaneDataType(imageRequirements->format, 0, &imageDataType);

        switch (imageDataType)
        {
        case NVCV_DATA_TYPE_U8:
        case NVCV_DATA_TYPE_2U8:
        case NVCV_DATA_TYPE_3U8:
        case NVCV_DATA_TYPE_4U8:
            *handle = reinterpret_cast<NVCVOperatorHandle>(
                new pva_solutions::priv::HOG<uint8_t>(imageRequirements, tensorRequirements));
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }
    });
}

template<typename StreamType>
NVCVStatus HOGSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVImageHandle const in,
                         NVCVTensorHandle const out)
{
    return nvcv::ProtectCall([&] {
        nvcv::ImageWrapHandle input(in);
        nvcv::TensorWrapHandle output(out);

        NVCVImageFormat inFormat;
        nvcvImageGetFormat(in, &inFormat);

        NVCVDataType imageDataType;
        nvcvImageFormatGetPlaneDataType(inFormat, 0, &imageDataType);

        switch (imageDataType)
        {
        case NVCV_DATA_TYPE_U8:
        case NVCV_DATA_TYPE_2U8:
        case NVCV_DATA_TYPE_3U8:
        case NVCV_DATA_TYPE_4U8:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::HOG<uint8_t>>(handle)(stream, input, output);
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus HOGSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVImageHandle const,
                                                 NVCVTensorHandle const);
template NVCVStatus HOGSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVImageHandle const,
                                                NVCVTensorHandle const);

// C++ overload implementations
NVCVStatus pvaHOGSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVImageHandle const in,
                        NVCVTensorHandle const out)
{
    return HOGSubmitImpl(handle, stream, in, out);
}

NVCVStatus pvaHOGSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageHandle const in,
                        NVCVTensorHandle const out)
{
    return HOGSubmitImpl(handle, stream, in, out);
}
