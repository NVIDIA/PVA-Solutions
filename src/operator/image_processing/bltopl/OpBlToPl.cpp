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

#include "OpBlToPl.hpp"

#include "device/bltopl_param.h"

#include <IOperator.hpp>
#include <ImageUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <nvcv/Image.hpp>

#include <ErrorCheckMacros.h>
#include <OpBlToPl.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <cmath>
#include <cstdint>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_bltopl_dev);

namespace pva_solutions::priv {

std::mutex Bltopl::s_execMapMutex;

std::map<Bltopl::ExecKey, std::weak_ptr<cupva::Executable>> Bltopl::s_bltoplExecutables;

std::shared_ptr<cupva::Executable> Bltopl::GetBltoplExec()
{
    static IOperator::ExecutableData sBltoplExecData = {CUPVA_EXECUTABLE_DATA(pva_bltopl_dev),
                                                        CUPVA_EXECUTABLE_SIZE(pva_bltopl_dev)};

    ExecKey execKey = std::make_tuple(cupva::Context::GetCurrent());
    return IOperator::GetExecutable<ExecKey>(sBltoplExecData, execKey, s_bltoplExecutables, s_execMapMutex);
}

Bltopl::Bltopl(NVCVImageRequirements const *const imageRequirements)
    : m_inImageRequirements(*imageRequirements)
    , m_outImageRequirements(*imageRequirements)
    , m_srcOp{cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM),
              cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM)}
    , m_dstOp{cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM),
              cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM)}
{
    if (!(imageRequirements->format == NVCV_IMAGE_FORMAT_NV12 ||
          imageRequirements->format == NVCV_IMAGE_FORMAT_NV12_ER))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Image's format should be either NVCV_IMAGE_FORMAT_NV12 or NVCV_IMAGE_FORMAT_NV12_ER");
    }
    if (!(imageRequirements->height % 2 == 0))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Image height should be even");
    }
    if (!(imageRequirements->width % 2 == 0))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Image width should be even");
    }
    if (!(imageRequirements->width >= kMinImageWidth))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Image width should be larger than the minimum value %d", kMinImageWidth);
    }
    if (!(imageRequirements->height >= kMinImageHeight))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Image height should be larger than the minimum value %d", kMinImageHeight);
    }

    int const tileX{TILE_X};
    int const tileY{TILE_Y};
    // NVCVImageRequirements does not have per plane width and height, so need to derive the UV plane's
    // value from that of Y plane's. If only considering the layout, a UV plane can be thought of having
    // "width/2"x"height/2" with bytes per pixel to be 2, or equivalently having "width"x"height/2" with
    // bytes per pixel to be 1. The former is used in NVCVImage and the latter is used to configured the
    // DMA dataflow.
    int32_t const width          = imageRequirements->width;
    int32_t const heightY        = imageRequirements->height;
    int32_t const heightUv       = imageRequirements->height / 2;
    int32_t const outLinePitchY  = imageRequirements->planeRowStride[0];
    int32_t const outLinePitchUv = imageRequirements->planeRowStride[1];
    // BL layout's linepitch is determined by the image width, which cannot be explicitly specified
    int32_t const srcLinePitch              = (width + kBL_ALIGN - 1) / kBL_ALIGN * kBL_ALIGN;
    m_inImageRequirements.planeRowStride[0] = srcLinePitch;
    m_inImageRequirements.planeRowStride[1] = srcLinePitch;

    m_exec    = GetBltoplExec();
    auto prog = CmdProgram::Create(*m_exec.get());

    int8_t vpu_param_a[sizeof(bl_to_pl_host2dev_param_t)];
    bl_to_pl_host2dev_param_t *vpu_param = reinterpret_cast<bl_to_pl_host2dev_param_t *>(vpu_param_a);
    vpu_param->num_tiles_y               = ((width + tileX - 1) / tileX) * ((heightY + tileY - 1) / tileY);
    vpu_param->num_tiles_uv              = ((width + tileX - 1) / tileX) * ((heightUv + tileY - 1) / tileY);

    prog["vpu_param"].set(vpu_param_a, sizeof(bl_to_pl_host2dev_param_t));

    uint8_t *inoutTileBufferVMEM = prog["inoutTileBufferVMEM"].ptr<uint8_t>();

    const int32_t roiWidth   = std::max(tileX, width);
    const int32_t roiHeightY = std::max(tileY, heightY);
    prog.addDataFlowHead<RasterDataFlow>()
        .handler(prog["srcHdlY"])
        .src(m_srcOp[0], width, heightY, srcLinePitch)
        .tileBuffer(inoutTileBufferVMEM)
        .roi(0, 0, roiWidth, roiHeightY)
        .tile(tileX, tileY);

    prog.addDataFlowHead<RasterDataFlow>()
        .handler(prog["dstHdlY"])
        .dst(m_dstOp[0], width, heightY, outLinePitchY)
        .roi(0, 0, roiWidth, roiHeightY)
        .tileBuffer(inoutTileBufferVMEM)
        .tile(tileX, tileY);

    // UV plane
    const int32_t roiHeightUv = std::max(tileY, heightUv);
    prog.addDataFlowHead<RasterDataFlow>()
        .handler(prog["srcHdlUv"])
        .src(m_srcOp[1], width, heightUv, srcLinePitch)
        .roi(0, 0, roiWidth, roiHeightUv)
        .tileBuffer(inoutTileBufferVMEM)
        .tile(tileX, tileY);

    prog.addDataFlowHead<RasterDataFlow>()
        .handler(prog["dstHdlUv"])
        .dst(m_dstOp[1], width, heightUv, outLinePitchUv)
        .roi(0, 0, roiWidth, roiHeightUv)
        .tileBuffer(inoutTileBufferVMEM)
        .tile(tileX, tileY);

    prog.compileDataFlows();

    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename S>
void Bltopl::operator()(S stream, nvcv::Image const &in, nvcv::Image const &out)
{
    auto inImageData  = in.exportData<nvcv::ImageDataStrided>();
    auto outImageData = out.exportData<nvcv::ImageDataStrided>();

    if (!pva_solutions::utils::AreImageRequirementsEqual(m_inImageRequirements, inImageData->cdata()) ||
        !pva_solutions::utils::AreImageRequirementsEqual(m_outImageRequirements, outImageData->cdata()))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Input and output images must have the same requirements that was used to create the operator");
    }
    if (inImageData->cdata().buffer.strided.planes[0].basePtr == nullptr ||
        inImageData->cdata().buffer.strided.planes[1].basePtr == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "inImageData->basePtr() returns nullptr");
    }
    if (outImageData->cdata().buffer.strided.planes[0].basePtr == nullptr ||
        outImageData->cdata().buffer.strided.planes[1].basePtr == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "outImageData->basePtr() returns nullptr");
    }

    for (int i = 0; i < inImageData->numPlanes(); i++)
    {
        m_srcOp[i].setBase(reinterpret_cast<uint8_t *>(inImageData->cdata().buffer.strided.planes[i].basePtr));
        m_dstOp[i].setBase(reinterpret_cast<uint8_t *>(outImageData->cdata().buffer.strided.planes[i].basePtr));
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
        static_assert(!sizeof(S), "Unsupported stream type for BlToPl operator");
    }
}

// Explicit template instantiation
template void Bltopl::operator()<cupvaStream_t>(cupvaStream_t, nvcv::Image const &, nvcv::Image const &);
template void Bltopl::operator()<cudaStream_t>(cudaStream_t, nvcv::Image const &, nvcv::Image const &);

Bltopl::~Bltopl()
{
    CupvaCmdDestroy(&m_prog);
}

} // namespace pva_solutions::priv

NVCVStatus pvaBlToPlCreate(NVCVOperatorHandle *handle, NVCVImageRequirements const *const imageRequirements)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::Bltopl(imageRequirements));
    });
}

template<typename StreamType>
NVCVStatus BlToPlSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVImageHandle in, NVCVImageHandle out)
{
    return nvcv::ProtectCall([&] {
        nvcv::ImageWrapHandle input(in), output(out);

        pva_solutions::priv::ToDynamicRef<pva_solutions::priv::Bltopl>(handle)(stream, input, output);
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus BlToPlSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVImageHandle,
                                                    NVCVImageHandle);
template NVCVStatus BlToPlSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVImageHandle, NVCVImageHandle);

// C++ overload implementations
NVCVStatus pvaBlToPlSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVImageHandle in, NVCVImageHandle out)
{
    return BlToPlSubmitImpl(handle, stream, in, out);
}

NVCVStatus pvaBlToPlSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageHandle in, NVCVImageHandle out)
{
    return BlToPlSubmitImpl(handle, stream, in, out);
}
