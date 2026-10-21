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

#include "OpWarpPerspective.hpp"

#include "device/warpPerspective_param.h"

#include <IOperator.hpp>
#include <ImageUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <cupva_host_utils.hpp>
#include <nvcv/Image.hpp>

#include <OpWarpPerspective.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <cstdint>
#include <iostream>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(warp_perspective_u8_dev);
CUPVA_DECLARE_EXECUTABLE(warp_perspective_nv12_dev);

namespace pva_solutions::priv {

std::mutex WarpPerspective::s_execMapMutex;

std::map<typename WarpPerspective::ExecKey, std::weak_ptr<cupva::Executable>>
    WarpPerspective::s_warpPerspectiveExecutables;

static std::map<typename WarpPerspective::ExecDataKey, IOperator::ExecutableData> createExecDataMap()
{
    return {{{NVCV_IMAGE_FORMAT_U8},
             {CUPVA_EXECUTABLE_DATA(warp_perspective_u8_dev), CUPVA_EXECUTABLE_SIZE(warp_perspective_u8_dev)}},
            {{NVCV_IMAGE_FORMAT_NV12_ER},
             {CUPVA_EXECUTABLE_DATA(warp_perspective_nv12_dev), CUPVA_EXECUTABLE_SIZE(warp_perspective_nv12_dev)}},
            {{NVCV_IMAGE_FORMAT_NV12},
             {CUPVA_EXECUTABLE_DATA(warp_perspective_nv12_dev), CUPVA_EXECUTABLE_SIZE(warp_perspective_nv12_dev)}}};
}

std::shared_ptr<cupva::Executable> WarpPerspective::GetWarpPerspectiveExec(ExecDataKey execDataKey)
{
    static auto execDataMap = createExecDataMap();

    auto execDataIt = execDataMap.find(execDataKey);
    if (execDataIt != execDataMap.end())
    {
        // Create execKey by combining execDataKey and context
        ExecKey execKey = std::tuple_cat(execDataKey, std::make_tuple(cupva::Context::GetCurrent()));
        return GetExecutable(execDataIt->second, execKey, s_warpPerspectiveExecutables, s_execMapMutex);
    }
    else
    {
        throw std::invalid_argument("Unsupported type or parameters");
    }
}

WarpPerspective::WarpPerspective(NVCVImageRequirements *inRequirements, NVCVImageRequirements *outRequirements,
                                 PVAInterpolationType interpType, PVAWarpTransformationType transformType)
    : m_inRequirements{*inRequirements}
    , m_outRequirements{*outRequirements}
    , m_srcOp{cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM, nullptr),
              cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM, nullptr)}
    , m_dstOp{cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM, nullptr),
              cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM, nullptr)}
    , m_interpType{interpType}
    , m_transformType{transformType}
    , m_exec{}
    , m_prog{}
{
    if (!(m_inRequirements.format == NVCV_IMAGE_FORMAT_U8 || m_inRequirements.format == NVCV_IMAGE_FORMAT_NV12_ER ||
          m_inRequirements.format == NVCV_IMAGE_FORMAT_NV12))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Image must have NVCV_IMAGE_FORMAT_U8, NVCV_IMAGE_FORMAT_NV12, or "
                              "NVCV_IMAGE_FORMAT_NV12_ER format");
    }

    const int32_t outImageWidth  = m_outRequirements.width;
    const int32_t outImageHeight = m_outRequirements.height;
    if (outImageWidth % TILE_W != 0 || outImageHeight % TILE_H != 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output image width and height must be divisible by TILE_W and TILE_H respectively");
    }

    m_exec = GetWarpPerspectiveExec(std::make_tuple(m_inRequirements.format));

    auto prog = CmdProgram::Create(*m_exec.get());

    // dataflow constructions
    prog["outImageWidth"]       = outImageWidth;
    prog["outImageHeight"]      = outImageHeight;
    prog["inImageWidth"]        = m_inRequirements.width;
    prog["inImageHeight"]       = m_inRequirements.height;
    prog["outLinePitch"]        = (int32_t)(m_outRequirements.planeRowStride[0] / sizeof(uint8_t));
    uint8_t *dstBuf             = prog["dst_v"].ptr<uint8_t>();
    auto srcHdl                 = prog["srcHdl"];
    auto dstHdl                 = prog["dstHdl"];
    InterpTypeDev devInterpType = (m_interpType == PVAInterpolationType::PVA_INTERPOLATION_NN) ? NN : BILINEAR;
    prog["interpType_v"]        = static_cast<uint32_t>(devInterpType);
    TransformTypeDev devTransformType =
        (m_transformType == PVAWarpTransformationType::PVA_WARP_TRANSFORMATION_AFFINE) ? AFFINE : PERSPECTIVE;
    prog["transformType_v"] = static_cast<uint32_t>(devTransformType);

    GatherScatterDataFlow &imgInDataFlow = prog.addDataFlowHead<GatherScatterDataFlow>();
    SequenceDataFlow &imgOutDataFlow     = prog.addDataFlowHead<SequenceDataFlow>();

    const int32_t inLinePitch = m_inRequirements.planeRowStride[0] / sizeof(uint8_t);
    imgInDataFlow.handler(srcHdl)
        .numTilesPerTrigger(1)
        // will configure the exact tile size at VPU side
        .tile(TILE_W, TILE_H)
        .src(m_srcOp[0], m_inRequirements.width, m_inRequirements.height, inLinePitch);
    imgInDataFlow.padVal(cupva::PadModeType::PAD_CONST);

    const int32_t outLinePitch = m_outRequirements.planeRowStride[0] / sizeof(uint8_t);
    imgOutDataFlow.handler(dstHdl)
        .addTransfer()
        .src(dstBuf, TILE_W * sizeof(uint8_t))
        .dst(m_dstOp[0], outLinePitch)
        .tile(TILE_W * sizeof(uint8_t), TILE_H)
        .mode(TransferModeType::CONTINUOUS);

    // additional dataflow for semiplanar format
    if (m_inRequirements.format == NVCV_IMAGE_FORMAT_NV12_ER || m_inRequirements.format == NVCV_IMAGE_FORMAT_NV12)
    {
        auto srcSPHdl                          = prog["srcSPHdl"];
        GatherScatterDataFlow &imgSPInDataFlow = prog.addDataFlowHead<GatherScatterDataFlow>();
        const int32_t inLinePitchSP            = m_inRequirements.planeRowStride[1] / sizeof(uint8_t);
        imgSPInDataFlow.handler(srcSPHdl)
            .numTilesPerTrigger(1)
            .tile(TILE_W, TILE_H / 2)
            .src(m_srcOp[1], m_inRequirements.width, m_inRequirements.height / 2, inLinePitchSP);
        imgSPInDataFlow.padVal(cupva::PadModeType::PAD_CONST);

        auto dstSPHdl                      = prog["dstSPHdl"];
        uint8_t *dstSPBuf                  = prog["dstSP_v"].ptr<uint8_t>();
        const int32_t outLinePitchSP       = m_outRequirements.planeRowStride[1];
        SequenceDataFlow &imgOutSPDataFlow = prog.addDataFlowHead<SequenceDataFlow>();
        imgOutSPDataFlow.handler(dstSPHdl)
            .addTransfer()
            .src(dstSPBuf, TILE_W * sizeof(uint8_t))
            .dst(m_dstOp[1], outLinePitchSP)
            .tile(TILE_W * sizeof(uint8_t), TILE_H / 2)
            .mode(TransferModeType::CONTINUOUS);

        prog["outLinePitchSP"] = (int32_t)(outLinePitchSP / sizeof(uint8_t));
    }

    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename S>
void WarpPerspective::operator()(S stream, const nvcv::Image &in, WarpMatrixType invWarpMatrix, const nvcv::Image &out)
{
    auto inImageData  = in.exportData<nvcv::ImageDataStrided>();
    auto outImageData = out.exportData<nvcv::ImageDataStrided>();

    if (!pva_solutions::utils::AreImageRequirementsEqual(m_inRequirements, inImageData->cdata()) ||
        !pva_solutions::utils::AreImageRequirementsEqual(m_outRequirements, outImageData->cdata()))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Input and output images must have the same requirements that was used to create the operator");
    }
    if ((static_cast<void *>(inImageData->plane(0).basePtr) == nullptr) ||
        (static_cast<void *>(outImageData->plane(0).basePtr) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Image base pointers must not be NULL");
    }

    if (m_inRequirements.format == NVCV_IMAGE_FORMAT_NV12_ER || m_inRequirements.format == NVCV_IMAGE_FORMAT_NV12)
    {
        if ((static_cast<void *>(inImageData->plane(1).basePtr) == nullptr) ||
            (static_cast<void *>(outImageData->plane(1).basePtr) == nullptr))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Image base pointers must not be NULL");
        }
    }

    // Set external memory pointers for SQDF.
    cupvaParameter_t dstExtPtr;
    CupvaCmdProgramGetParameter(&m_prog, &dstExtPtr, "dstExtPtr");
    CupvaParameterSetValuePointer(&dstExtPtr, static_cast<void *>(outImageData->plane(0).basePtr));

    m_srcOp[0].setBase(static_cast<uint8_t *>(static_cast<void *>(inImageData->plane(0).basePtr)));
    m_dstOp[0].setBase(static_cast<uint8_t *>(static_cast<void *>(outImageData->plane(0).basePtr)));
    // clear memory for Y plane as VPU will omit transfer if total tile is out of range
    uint8_t *dstYBuf;
    CupvaMemGetHostPointer((void **)&dstYBuf, (void *)outImageData->plane(0).basePtr);
    const int32_t dstYSize = m_outRequirements.planeRowStride[0] * m_outRequirements.height;
    memset(dstYBuf, 0, dstYSize);

    if (m_inRequirements.format == NVCV_IMAGE_FORMAT_NV12_ER || m_inRequirements.format == NVCV_IMAGE_FORMAT_NV12)
    {
        cupvaParameter_t dstSPExtPtr;
        CupvaCmdProgramGetParameter(&m_prog, &dstSPExtPtr, "dstSPExtPtr");
        CupvaParameterSetValuePointer(&dstSPExtPtr, static_cast<void *>(outImageData->plane(1).basePtr));
        m_srcOp[1].setBase(static_cast<uint8_t *>(static_cast<void *>(inImageData->plane(1).basePtr)));
        m_dstOp[1].setBase(static_cast<uint8_t *>(static_cast<void *>(outImageData->plane(1).basePtr)));

        // clear memory for UV plane as VPU will omit transfer if total tile is out of range
        uint8_t *dstUVBuf = dstYBuf + dstYSize;
        //CupvaMemGetHostPointer((void **)&dstUVBuf, (void *)outImageData->plane(1).basePtr);
        const int32_t dstUVSize = m_outRequirements.planeRowStride[1] * m_outRequirements.height / 2;
        memset(dstUVBuf, 128, dstUVSize);
    }

    cupvaParameter_t param;
    CupvaCmdProgramGetParameter(&m_prog, &param, "invWarpMat_v");
    CupvaParameterSetValueArray(&param, &invWarpMatrix[0][0], sizeof(float) * 9);
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
        static_assert(!sizeof(S), "Unsupported stream type for WarpPerspective operator");
    }

    if (!(err == CUPVA_ERROR_NONE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Submit failed");
    }
}

WarpPerspective::~WarpPerspective()
{
    CupvaCmdDestroy(&m_prog);
}

} // namespace pva_solutions::priv

NVCVStatus pvaWarpPerspectiveCreate(NVCVOperatorHandle *handle, NVCVImageRequirements *inRequirements,
                                    NVCVImageRequirements *outRequirements, PVAInterpolationType interpType,
                                    PVAWarpTransformationType transformType)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        if (inRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to ImageRequirements must not be NULL");
        }
        if (outRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Pointer to outRequirements must not be NULL");
        }
        *handle = reinterpret_cast<NVCVOperatorHandle>(
            new pva_solutions::priv::WarpPerspective(inRequirements, outRequirements, interpType, transformType));
    });
}

template<typename StreamType>
NVCVStatus WarpPerspectiveSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVImageHandle in,
                                     WarpMatrixType invWarpMatrix, NVCVImageHandle out)
{
    return nvcv::ProtectCall([&] {
        nvcv::ImageWrapHandle input(in), output(out);
        pva_solutions::priv::ToDynamicRef<pva_solutions::priv::WarpPerspective>(handle)(stream, input, invWarpMatrix,
                                                                                        output);
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus WarpPerspectiveSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVImageHandle,
                                                             WarpMatrixType, NVCVImageHandle);
template NVCVStatus WarpPerspectiveSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVImageHandle,
                                                            WarpMatrixType, NVCVImageHandle);

// C++ overload implementations
NVCVStatus pvaWarpPerspectiveSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVImageHandle in,
                                    WarpMatrixType invWarpMatrix, NVCVImageHandle out)
{
    return WarpPerspectiveSubmitImpl(handle, stream, in, invWarpMatrix, out);
}

NVCVStatus pvaWarpPerspectiveSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageHandle in,
                                    WarpMatrixType invWarpMatrix, NVCVImageHandle out)
{
    return WarpPerspectiveSubmitImpl(handle, stream, in, invWarpMatrix, out);
}