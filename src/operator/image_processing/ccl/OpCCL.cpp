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

#include "OpCCL.hpp"

#include "device/ccl_params.h"

#include <IOperator.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <cupva_host_nonsafety.hpp>
#include <cupva_host_utils.hpp>
#include <nvcv/Tensor.hpp>

#include <OpCCL.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>
#include <nvcv/Fwd.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>

#include <iostream>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_ccl_dev);

namespace pva_solutions::priv {

std::mutex CCL::s_execMapMutex;
std::map<CCL::ExecKey, std::weak_ptr<cupva::Executable>> CCL::s_CCLExecutables;

std::shared_ptr<cupva::Executable> CCL::GetCCLExec()
{

    static IOperator::ExecutableData CCLExecData = {CUPVA_EXECUTABLE_DATA(pva_ccl_dev),
                                                    CUPVA_EXECUTABLE_SIZE(pva_ccl_dev)};

    ExecKey execKey = std::make_tuple(cupva::Context::GetCurrent());
    return IOperator::GetExecutable<ExecKey>(CCLExecData, execKey, s_CCLExecutables, s_execMapMutex);
}

CCL::CCL(const NVCVTensorRequirements *imageParams, const NVCVTensorRequirements *labelParams,
         const PVAConnectivityType connectivity, const uint8_t threshold)
    : m_imageTensorRequirements(*imageParams)
    , m_labelTensorRequirements(*labelParams)
    , m_imageOp(cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_labelsOp(cupva::OffsetPointer<uint16_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_numLabelsOp(cupva::OffsetPointer<int32_t>::Create(cupva::MemType::DRAM, nullptr))
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

    // Get dimensions for output labels
    int64_t labelShapeNCHW[4];
    int64_t labelStridesNCHW[4];
    nvcvTensorShapePermute(labelParams->layout, labelParams->shape, NVCV_TENSOR_NCHW, labelShapeNCHW);
    nvcvTensorShapePermute(labelParams->layout, labelParams->strides, NVCV_TENSOR_NCHW, labelStridesNCHW);

    int32_t labelN         = labelShapeNCHW[0];
    int32_t labelC         = labelShapeNCHW[1];
    int32_t labelH         = labelShapeNCHW[2];
    int32_t labelW         = labelShapeNCHW[3];
    int32_t labelLinePitch = labelStridesNCHW[2] / (nvcv::DataType(labelParams->dtype).bitsPerPixel() / 8);

    if (!((imageParams->layout == NVCV_TENSOR_HW) || (imageParams->layout == NVCV_TENSOR_CHW) ||
          (imageParams->layout == NVCV_TENSOR_NCHW)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Image tensor layout is not supported");
    }

    if (!(labelParams->layout == imageParams->layout))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Label tensor layout must be the same as image tensor layout");
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

    if (labelParams->dtype != NVCV_DATA_TYPE_U16)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Only U16 is supported for label tensor");
    }

    if (labelW != m_width || labelH != m_height)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Label tensor must have the same width and height as image tensor");
    }

    if (!(connectivity == PVA_4_CONNECTIVITY || connectivity == PVA_8_CONNECTIVITY))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Connectivity must be 4 or 8");
    }

    int32_t nIterX   = (m_width + CCL_TILEW - 1) / CCL_TILEW;
    int32_t nIterY   = (m_height + CCL_TILEH - 1) / CCL_TILEH;
    int32_t bufPitch = (nIterX * CCL_TILEW) + 2 * CCL_HALOX;

    m_bufDevPtr = static_cast<uint16_t *>(cupva::mem::Alloc(m_height * bufPitch * sizeof(uint16_t)));
    if (m_bufDevPtr == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "Failed to allocate memory for intermediate buffer");
    }

    m_exec    = GetCCLExec();
    auto prog = CmdProgram::Create(*m_exec.get());

    // Input - read image
    auto inHdl    = prog["inHdl"];
    auto inBuffer = prog["inBuffer"];
    prog.addDataFlowHead<cupva::RasterDataFlow>(0)
        .handler(inHdl)
        .src(m_imageOp, m_width, m_height, m_linePitch)
        .tileBuffer(inBuffer.ptr<uint8_t>())
        .halo(CCL_HALOX, 0)
        .tile(CCL_TILEW, CCL_TILEH);

    // Write intermediate labels. At this point, pairs still need to be merged...
    auto outIntermediateHdl    = prog["outIntermediateHdl"];
    auto outIntermediateBuffer = prog["outIntermediateBuffer"];
    prog.addDataFlowHead<cupva::RasterDataFlow>(0)
        .handler(outIntermediateHdl)
        .dst(m_bufDevPtr, (nIterX * CCL_TILEW) + CCL_HALOX, m_height, bufPitch)
        .tileBuffer(outIntermediateBuffer.ptr<uint16_t>())
        .transpose(TRANS_MODE_1)
        .tile(CCL_TILEW, CCL_TILEH);

    // Phase 2 output - write number of labels
    auto numLabelsHdl = prog["numLabelsHdl"];
    auto numLabels    = prog["numLabels"];
    prog.addDataFlowHead<cupva::SequenceDataFlow>(1)
        .handler(numLabelsHdl)
        .addTransfer()
        .src(numLabels.ptr<int32_t>(), sizeof(int32_t))
        .dst(m_numLabelsOp, sizeof(int32_t))
        .tile(sizeof(int32_t));

    // Phase 3 input - read intermediate labels
    auto inIntHdl    = prog["inIntermediateHdl"];
    auto inIntBuffer = prog["inIntermediateBuffer"];
    prog.addDataFlowHead<cupva::RasterDataFlow>(2)
        .handler(inIntHdl)
        .src(m_bufDevPtr, (nIterX * CCL_TILEW) + CCL_HALOX, m_height, bufPitch)
        .roi(CCL_HALOX, 0, m_width, m_height)
        .tileBuffer(inIntBuffer.ptr<uint16_t>())
        .halo(CCL_HALOX, 0)
        .tile(CCL_TILEW, CCL_TILEH);

    // Phase 3 output - write final labels
    auto outHdl    = prog["outLabelsHdl"];
    auto outBuffer = prog["outLabels"];
    prog.addDataFlowHead<cupva::RasterDataFlow>(2)
        .handler(outHdl)
        .dst(m_labelsOp, m_width, m_height, labelLinePitch)
        .tileBuffer(outBuffer.ptr<uint16_t>())
        .tile(CCL_TILEW, CCL_TILEH);

    // Params
    param_ccl params = {
        .nIterX    = nIterX,
        .nIterY    = nIterY,
        .isFourWay = static_cast<uint8_t>(connectivity == PVA_4_CONNECTIVITY ? 1 : 0),
        .threshold = threshold,
    };

    prog["algoParams"].set(&params, 1);

    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename S>
void CCL::operator()(S stream, const nvcv::Tensor &image, const nvcv::Tensor &labels, const nvcv::Tensor &numLabels)
{
    auto imageTensorData     = image.exportData<nvcv::TensorDataStrided>();
    auto labelTensorData     = labels.exportData<nvcv::TensorDataStrided>();
    auto numLabelsTensorData = numLabels.exportData<nvcv::TensorDataStrided>();
    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_imageTensorRequirements, imageTensorData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_labelTensorRequirements, labelTensorData->cdata()))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Input and output tensors must have the same requirements that were used to create the operator");
    }
    if ((static_cast<void *>(imageTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(labelTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(numLabelsTensorData->basePtr()) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor base pointers must not be NULL");
    }

    m_imageOp.setBase(static_cast<uint8_t *>(static_cast<void *>(imageTensorData->basePtr())));
    m_labelsOp.setBase(static_cast<uint16_t *>(static_cast<void *>(labelTensorData->basePtr())));
    m_numLabelsOp.setBase(static_cast<int32_t *>(static_cast<void *>(numLabelsTensorData->basePtr())));

    if (CupvaCmdProgramUpdateDataFlows(&m_prog) != CUPVA_ERROR_NONE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Update DataFlow failed");
    }

    cupvaCmd_t const *submitCmds[1] = {&m_prog};
    cupvaCmdStatus_t cmdStatus[1]   = {NULL};

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
        static_assert(!sizeof(S), "Unsupported stream type for CCL operator");
    }
}

CCL::~CCL()
{
    CupvaCmdDestroy(&m_prog);
    cupva::mem::Free(m_bufDevPtr);
}

} // namespace pva_solutions::priv

NVCVStatus pvaCCLCreate(NVCVOperatorHandle *handle, const NVCVTensorRequirements *imageParams,
                        const NVCVTensorRequirements *labelParams, const PVAConnectivityType connectivity,
                        const uint8_t threshold)
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
        if (labelParams == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to label TensorRequirements must not be NULL");
        }

        *handle = reinterpret_cast<NVCVOperatorHandle>(
            new pva_solutions::priv::CCL(imageParams, labelParams, connectivity, threshold));
    });
}

template<typename StreamType>
NVCVStatus CCLSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle image, NVCVTensorHandle labels,
                         NVCVTensorHandle numLabels)
{
    return nvcv::ProtectCall([&] {
        nvcv::TensorWrapHandle in(image), out(labels), count(numLabels);
        pva_solutions::priv::ToDynamicRef<pva_solutions::priv::CCL>(handle)(stream, in, out, count);
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus CCLSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle, NVCVTensorHandle,
                                                 NVCVTensorHandle);
template NVCVStatus CCLSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle, NVCVTensorHandle,
                                                NVCVTensorHandle);

// C++ overload implementations
NVCVStatus pvaCCLSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle image,
                        NVCVTensorHandle labels, NVCVTensorHandle numLabels)
{
    return CCLSubmitImpl(handle, stream, image, labels, numLabels);
}

NVCVStatus pvaCCLSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle image, NVCVTensorHandle labels,
                        NVCVTensorHandle numLabels)
{
    return CCLSubmitImpl(handle, stream, image, labels, numLabels);
}
