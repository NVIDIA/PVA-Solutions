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

#include "OpCannyEdgeDetector.hpp"

#include "device/canny_edge_detector_params.h"

#include <IOperator.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <cupva_host_nonsafety.hpp>
#include <cupva_host_utils.hpp>
#include <nvcv/Tensor.hpp>

#include <OpCannyEdgeDetector.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>
#include <nvcv/Fwd.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>

#include <array>
#include <cmath>
#include <iostream>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_canny_edge_detector_dev);

namespace pva_solutions::priv {

std::mutex CannyEdgeDetector::s_execMapMutex;
std::map<CannyEdgeDetector::ExecKey, std::weak_ptr<cupva::Executable>>
    CannyEdgeDetector::s_CannyEdgeDetectorExecutables;

std::shared_ptr<cupva::Executable> CannyEdgeDetector::GetCannyEdgeDetectorExec()
{

    static IOperator::ExecutableData CannyEdgeDetectorExecData = {CUPVA_EXECUTABLE_DATA(pva_canny_edge_detector_dev),
                                                                  CUPVA_EXECUTABLE_SIZE(pva_canny_edge_detector_dev)};

    ExecKey execKey = std::make_tuple(cupva::Context::GetCurrent());
    return IOperator::GetExecutable<ExecKey>(CannyEdgeDetectorExecData, execKey, s_CannyEdgeDetectorExecutables,
                                             s_execMapMutex);
}

CannyEdgeDetector::CannyEdgeDetector(const NVCVTensorRequirements *imageParams, const int32_t gradientSize)
    : m_imageTensorRequirements(*imageParams)
    , m_imageOp(cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_edgeMapOp(cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_prog{}
{

    // Get dimensions for input image
    int64_t imageShapeNCHW[4];
    int64_t imageStridesNCHW[4];
    nvcvTensorShapePermute(imageParams->layout, imageParams->shape, NVCV_TENSOR_NCHW, imageShapeNCHW);
    nvcvTensorShapePermute(imageParams->layout, imageParams->strides, NVCV_TENSOR_NCHW, imageStridesNCHW);

    int32_t imageN = imageShapeNCHW[0];
    int32_t imageC = imageShapeNCHW[1];
    m_height       = imageShapeNCHW[2];
    m_width        = imageShapeNCHW[3];
    m_linePitch    = imageStridesNCHW[2] / (nvcv::DataType(imageParams->dtype).bitsPerPixel() / 8);

    if (!((imageParams->layout == NVCV_TENSOR_HW) || (imageParams->layout == NVCV_TENSOR_CHW) ||
          (imageParams->layout == NVCV_TENSOR_NCHW)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Image tensor layout is not supported");
    }

    if ((imageParams->layout == NVCV_TENSOR_CHW) && imageC != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Only support C = 1 for image Tensor");
    }

    if ((imageParams->layout == NVCV_TENSOR_NCHW) && (imageN != 1 || imageC != 1))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Only support C = 1 and N = 1 for image Tensor");
    }

    if (imageParams->dtype != NVCV_DATA_TYPE_U8)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Only U8 is supported for image tensor");
    }
    if (!(gradientSize == 3 || gradientSize == 5 || gradientSize == 7))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Gradient windowsize must be 3, 5, or 7");
    }

    m_gradientIntensityDevPtr = static_cast<int32_t *>(cupva::mem::Alloc(m_height * m_width * sizeof(int32_t)));
    if (m_gradientIntensityDevPtr == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "Failed to allocate memory for gradient intensity buffer");
    }

    m_edgeDirDevPtr = static_cast<int8_t *>(cupva::mem::Alloc(m_height * m_width * sizeof(int8_t)));
    if (m_edgeDirDevPtr == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "Failed to allocate memory for edge direction buffer");
    }

    m_edgeTypeDevPtr = static_cast<uint8_t *>(cupva::mem::Alloc(m_height * m_width * sizeof(uint8_t)));
    if (m_edgeTypeDevPtr == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "Failed to allocate memory for edge type buffer");
    }

    int32_t nIterX   = (m_width + CCL_TILEW - 1) / CCL_TILEW;
    int32_t bufPitch = (nIterX * CCL_TILEW) + 2 * CCL_HALOX;

    m_bufDevPtr = static_cast<uint16_t *>(cupva::mem::Alloc(m_height * bufPitch * sizeof(uint16_t)));
    if (m_bufDevPtr == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "Failed to allocate memory for intermediate buffer");
    }

    m_exec    = GetCannyEdgeDetectorExec();
    auto prog = CmdProgram::Create(*m_exec.get());

    uint8_t *image        = prog["image"].ptr<uint8_t>();
    int8_t *edgeDirOut    = prog["edge_dir_out"].ptr<int8_t>();
    int32_t *intensityOut = prog["intensity_out"].ptr<int32_t>();
    int8_t *edgeDirIn     = prog["edge_dir_in"].ptr<int8_t>();
    int32_t *intensityIn  = prog["intensity_in"].ptr<int32_t>();
    uint8_t *edgeType     = prog["edge_type"].ptr<uint8_t>();

    auto imageHandler        = prog["image_handler"];
    auto intensityOutHandler = prog["intensity_out_handler"];
    auto edgeDirOutHandler   = prog["edge_dir_out_handler"];
    auto intensityInHandler  = prog["intensity_in_handler"];
    auto edgeDirInHandler    = prog["edge_dir_in_handler"];
    auto edgeTypeOutHandler  = prog["edge_type_out_handler"];

    const int32_t p1{1};
    const int32_t p2{2};
    const int32_t p3{3};
    const int32_t p4{4};
    const int32_t p5{5};

    RasterDataFlow &imageDataFlow = prog.addDataFlowHead<RasterDataFlow>(p1);
    imageDataFlow.handler(imageHandler)
        .src(m_imageOp, m_width, m_height, m_linePitch)
        .tileBuffer(image)
        .tile(TW, TH)
        .roi(0, 0, m_width, m_height)
        .halo(gradientSize / 2);

    RasterDataFlow &intensityOutDataFlow = prog.addDataFlowHead<RasterDataFlow>(p1);
    intensityOutDataFlow.handler(intensityOutHandler)
        .tileBuffer(intensityOut)
        .tile(TW, TH)
        .roi(0, 0, m_width, m_height)
        .dst(m_gradientIntensityDevPtr, m_width, m_height, m_width);

    RasterDataFlow &edgeDirOutDataFlow = prog.addDataFlowHead<RasterDataFlow>(p1);
    edgeDirOutDataFlow.handler(edgeDirOutHandler)
        .tileBuffer(edgeDirOut)
        .tile(TW, TH)
        .roi(0, 0, m_width, m_height)
        .dst(m_edgeDirDevPtr, m_width, m_height, m_width);

    int32_t widthPadded  = (m_width + TW_NMS - 1) / TW_NMS * TW_NMS;
    int32_t heightPadded = (m_height + TH_NMS - 1) / TH_NMS * TH_NMS;

    RasterDataFlow &intensityInDataFlow = prog.addDataFlowHead<RasterDataFlow>(p2);
    intensityInDataFlow.handler(intensityInHandler)
        .src(m_gradientIntensityDevPtr, m_width, m_height, m_width)
        .tileBuffer(intensityIn)
        .tile(TW_NMS, TH_NMS)
        .roi(0, 0, m_width, m_height)
        .halo(1, PAD_BPE);

    RasterDataFlow &edgeDirInDataFlow = prog.addDataFlowHead<RasterDataFlow>(p2);
    edgeDirInDataFlow.handler(edgeDirInHandler)
        .src(m_edgeDirDevPtr, m_width, m_height, m_width)
        .tileBuffer(edgeDirIn)
        .tile(TW_NMS, TH_NMS)
        .roi(0, 0, widthPadded, heightPadded)
        .halo(0);

    RasterDataFlow &edgeTypeDataFlow = prog.addDataFlowHead<RasterDataFlow>(p5);
    edgeTypeDataFlow.handler(edgeTypeOutHandler)
        .tileBuffer(edgeType)
        .tile(TW_NMS, TH_NMS)
        .roi(0, 0, m_width, m_height)
        .dst(m_edgeTypeDevPtr, m_width, m_height, m_width);

    uint8_t *inEdgeType             = prog["in_edge_type"].ptr<uint8_t>();
    uint16_t *outIntermediateBuffer = prog["out_intermediate_buffer"].ptr<uint16_t>();
    uint16_t *inIntermediateBuffer  = prog["in_intermediate_buffer"].ptr<uint16_t>();
    uint8_t *outEdgeMap             = prog["out_edge_map"].ptr<uint8_t>();

    auto inEdgeTypeHdl      = prog["inEdgeTypeHdl"];
    auto outIntermediateHdl = prog["outIntermediateHdl"];
    auto inIntermediateHdl  = prog["inIntermediateHdl"];
    auto outHdl             = prog["outHdl"];

    prog.addDataFlowHead<cupva::RasterDataFlow>(p3)
        .handler(inEdgeTypeHdl)
        .src(m_edgeTypeDevPtr, m_width, m_height, m_width)
        .tileBuffer(inEdgeType)
        .halo(CCL_HALOX, 0)
        .tile(CCL_TILEW, CCL_TILEH);

    RasterDataFlow &intermediateDataFlow = prog.addDataFlow<cupva::RasterDataFlow>();
    intermediateDataFlow.handler(outIntermediateHdl)
        .dst(m_bufDevPtr, (nIterX * CCL_TILEW) + CCL_HALOX, m_height, bufPitch)
        .tileBuffer(outIntermediateBuffer)
        .transpose(TRANS_MODE_1)
        .tile(CCL_TILEW, CCL_TILEH);

    prog.addDataFlowHead<cupva::RasterDataFlow>(p4)
        .handler(inIntermediateHdl)
        .src(m_bufDevPtr, (nIterX * CCL_TILEW) + CCL_HALOX, m_height, bufPitch)
        .roi(CCL_HALOX, 0, m_width, m_height)
        .tileBuffer(inIntermediateBuffer)
        .halo(CCL_HALOX, 0)
        .tile(CCL_TILEW, CCL_TILEH);

    RasterDataFlow &outDataFlow = prog.addDataFlow<cupva::RasterDataFlow>();
    outDataFlow.handler(outHdl)
        .dst(m_edgeMapOp, m_width, m_height, m_linePitch)
        .tileBuffer(outEdgeMap)
        .tile(CCL_TILEW, CCL_TILEH);
    edgeTypeDataFlow.link(intermediateDataFlow);
    intermediateDataFlow.link(outDataFlow);

    prog["img_w"]         = m_width;
    prog["img_h"]         = m_height;
    prog["gradient_size"] = gradientSize;

    prog.compileDataFlows();

    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename S>
void CannyEdgeDetector::operator()(S stream, cupvaCmdStatus_t *cmdStatus, const nvcv::Tensor &image,
                                   const nvcv::Tensor &edgeMap, const int32_t thresholdStrong,
                                   const int32_t thresholdWeak)
{
    auto imageTensorData   = image.exportData<nvcv::TensorDataStrided>();
    auto edgeMapTensorData = edgeMap.exportData<nvcv::TensorDataStrided>();

    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_imageTensorRequirements, imageTensorData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_imageTensorRequirements, edgeMapTensorData->cdata()))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Input and output tensors must have the same requirements that was used to create the operator");
    }
    if ((static_cast<void *>(imageTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(edgeMapTensorData->basePtr()) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor base pointers must not be NULL");
    }

    if (thresholdStrong < thresholdWeak)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Strong threshold must be greater than or equal to weak threshold");
    }

    if (thresholdStrong < 0 || thresholdWeak < 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Threshold values must be non-negative");
    }

    cupvaParameter_t param;
    CupvaCmdProgramGetParameter(&m_prog, &param, "threshold_strong");
    CupvaParameterSetValueScalar(&param, &thresholdStrong, sizeof(int32_t));
    CupvaCmdProgramGetParameter(&m_prog, &param, "threshold_weak");
    CupvaParameterSetValueScalar(&param, &thresholdWeak, sizeof(int32_t));

    m_imageOp.setBase(static_cast<uint8_t *>(static_cast<void *>(imageTensorData->basePtr())));
    m_edgeMapOp.setBase(static_cast<uint8_t *>(static_cast<void *>(edgeMapTensorData->basePtr())));

    if (CupvaCmdProgramUpdateDataFlows(&m_prog) != CUPVA_ERROR_NONE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Update DataFlow failed");
    }

    cupvaCmd_t const *submitCmds[1] = {&m_prog};

    if constexpr (std::is_same_v<S, cupvaStream_t>)
    {
        if (CupvaStreamSubmit(stream, submitCmds, cmdStatus, 1, CUPVA_IN_ORDER, -1, -1) != CUPVA_ERROR_NONE)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Submit failed");
        }
    }
    else if constexpr (std::is_same_v<S, cudaStream_t>)
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
        static_assert(!sizeof(S), "Unsupported stream type for CannyEdgeDetector operator");
    }
}

CannyEdgeDetector::~CannyEdgeDetector()
{
    CupvaCmdDestroy(&m_prog);
    cupva::mem::Free(m_gradientIntensityDevPtr);
    cupva::mem::Free(m_edgeDirDevPtr);
    cupva::mem::Free(m_edgeTypeDevPtr);
    cupva::mem::Free(m_bufDevPtr);
}

} // namespace pva_solutions::priv

NVCVStatus pvaCannyEdgeDetectorCreate(NVCVOperatorHandle *handle, const NVCVTensorRequirements *imageParams,
                                      const int32_t gradientSize)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        if (imageParams == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to image TensorRequirements must not be NULL");
        }

        *handle =
            reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::CannyEdgeDetector(imageParams, gradientSize));
    });
}

template<typename StreamType>
NVCVStatus CannyEdgeDetectorSubmitImpl(NVCVOperatorHandle handle, StreamType stream, cupvaCmdStatus_t *cmdStatus,
                                       NVCVTensorHandle image, NVCVTensorHandle edgeMap, const int32_t thresholdStrong,
                                       const int32_t thresholdWeak)
{
    return nvcv::ProtectCall([&] {
        nvcv::TensorWrapHandle in(image), out(edgeMap);
        pva_solutions::priv::ToDynamicRef<pva_solutions::priv::CannyEdgeDetector>(handle)(
            stream, cmdStatus, in, out, thresholdStrong, thresholdWeak);
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus CannyEdgeDetectorSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, cupvaCmdStatus_t *,
                                                               NVCVTensorHandle, NVCVTensorHandle, const int32_t,
                                                               const int32_t);
template NVCVStatus CannyEdgeDetectorSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, cupvaCmdStatus_t *,
                                                              NVCVTensorHandle, NVCVTensorHandle, const int32_t,
                                                              const int32_t);

// C++ overload implementations
NVCVStatus pvaCannyEdgeDetectorSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, cupvaCmdStatus_t *cmdStatus,
                                      NVCVTensorHandle image, NVCVTensorHandle edgeMap, const int32_t thresholdStrong,
                                      const int32_t thresholdWeak)
{
    return CannyEdgeDetectorSubmitImpl(handle, stream, cmdStatus, image, edgeMap, thresholdStrong, thresholdWeak);
}

NVCVStatus pvaCannyEdgeDetectorSubmit(NVCVOperatorHandle handle, cudaStream_t stream, cupvaCmdStatus_t *cmdStatus,
                                      NVCVTensorHandle image, NVCVTensorHandle edgeMap, const int32_t thresholdStrong,
                                      const int32_t thresholdWeak)
{
    return CannyEdgeDetectorSubmitImpl(handle, stream, cmdStatus, image, edgeMap, thresholdStrong, thresholdWeak);
}
