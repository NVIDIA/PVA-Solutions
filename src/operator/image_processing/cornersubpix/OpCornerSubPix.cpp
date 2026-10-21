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

#include "OpCornerSubPix.hpp"

#include "device/corner_subpix_params.h"

#include <IOperator.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <cupva_host_utils.hpp>
#include <nvcv/Tensor.hpp>

#include <ErrorCheckMacros.h>
#include <OpCornerSubPix.h>
#include <PvaAllocator.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>
#include <nvcv/Fwd.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_corner_subpix_dev);

namespace pva_solutions::priv {

std::mutex cornerSubPix::s_execMapMutex;
std::map<cornerSubPix::ExecKey, std::weak_ptr<cupva::Executable>> cornerSubPix::s_cornerSubPixExecutables;

std::shared_ptr<cupva::Executable> cornerSubPix::GetCornerSubPixExec()
{

    static IOperator::ExecutableData cornerSubPixExecData = {CUPVA_EXECUTABLE_DATA(pva_corner_subpix_dev),
                                                             CUPVA_EXECUTABLE_SIZE(pva_corner_subpix_dev)};

    ExecKey execKey = std::make_tuple(cupva::Context::GetCurrent());
    return IOperator::GetExecutable<ExecKey>(cornerSubPixExecData, execKey, s_cornerSubPixExecutables, s_execMapMutex);
}

cornerSubPix::cornerSubPix(NVCVTensorRequirements *cornerParams, NVCVTensorRequirements *imageParams,
                           const NVCVSize2D winSize, const NVCVSize2D zeroZone)
    : m_cornerTensorRequirements(*cornerParams)
    , m_imageTensorRequirements(*imageParams)
    , m_inCornerOp(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr))
    , m_outCornerOp(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr))
    , m_imageOp(cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_winWidth(winSize.w)
    , m_winHeight(winSize.h)
    , m_zeroZoneWidth(zeroZone.w)
    , m_zeroZoneHeight(zeroZone.h)
    , m_prog{}
{
    // Get dimensions for input image
    int64_t imageShapeNCHW[4];
    int64_t imageStridesNCHW[4];
    nvcvTensorShapePermute(imageParams->layout, imageParams->shape, NVCV_TENSOR_NCHW, imageShapeNCHW);
    nvcvTensorShapePermute(imageParams->layout, imageParams->strides, NVCV_TENSOR_NCHW, imageStridesNCHW);

    int32_t imageBatch    = imageShapeNCHW[0];
    int32_t imageChannels = imageShapeNCHW[1];
    m_height              = imageShapeNCHW[2];
    m_width               = imageShapeNCHW[3];
    m_linePitch           = imageStridesNCHW[2] / (nvcv::DataType(imageParams->dtype).bitsPerPixel() / 8);

    if (imageParams->dtype != NVCV_DATA_TYPE_U8)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Only U8 is supported for image tensor");
    }

    if (imageParams->layout != nvcv::TENSOR_HW && imageParams->layout != nvcv::TENSOR_CHW &&
        imageParams->layout != nvcv::TENSOR_HWC && imageParams->layout != nvcv::TENSOR_NCHW &&
        imageParams->layout != nvcv::TENSOR_NHWC)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Image tensor layout is not supported");
    }

    if ((imageBatch != 0 && imageBatch != 1) || (imageChannels != 0 && imageChannels != 1))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Image tensor must have N=1, C=1");
    }

    if (cornerParams->layout != nvcv::TENSOR_W && cornerParams->layout != nvcv::TENSOR_HW &&
        cornerParams->layout != nvcv::TENSOR_CHW && cornerParams->layout != nvcv::TENSOR_NCHW &&
        cornerParams->layout != nvcv::TENSOR_NHWC)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Corners tensor layout is not supported");
    }
    if (cornerParams->dtype != NVCV_DATA_TYPE_F32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Only 32-bit float is supported for corners tensor");
    }

    if (zeroZone.w >= winSize.w || zeroZone.h >= winSize.h)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Zero-zone must be within window");
    }
    if (winSize.w != winSize.h)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Window width must equal window height");
    }
    if (!(winSize.w == 4 || winSize.w == 3))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Only winSize = 4 or winSize = 3 is supported");
    }

    // Generate mask
    const int32_t mask_w = 2 * winSize.w + 1;
    const int32_t mask_h = 2 * winSize.h + 1;

    std::array<int32_t, MAX_WIN_SIZE * MAX_WIN_SIZE> mask;

    generateMaskInput(mask.data());

    m_exec    = GetCornerSubPixExec();
    auto prog = CmdProgram::Create(*m_exec.get());

    float *corners_vmem = prog["corners"].ptr<float>();
    auto gsdf_handler   = prog["gsdf_handler"];
    auto in_handler     = prog["in_handler"];
    auto out_handler    = prog["out_handler"];

    const int32_t p1{1};
    const int32_t p2{2};
    const int32_t p3{3};

    GatherScatterDataFlow &imgInput = prog.addDataFlowHead<GatherScatterDataFlow>(p1);

    imgInput.handler(gsdf_handler)
        .numTilesPerTrigger(BATCH)
        .src(m_imageOp, m_width, m_height, m_linePitch)
        .tile(TILE_SIZE(mask_w), TILE_SIZE(mask_h))
        .padVal(PAD_BPE);

    SequenceDataFlow &cornersInput  = prog.addDataFlowHead<SequenceDataFlow>(p2).handler(in_handler);
    SequenceDataFlow &cornersOutput = prog.addDataFlowHead<SequenceDataFlow>(p3).handler(out_handler);

    cornersInput.addTransfer()
        .src(m_inCornerOp, 2 * sizeof(float))
        .dst(corners_vmem, 2 * sizeof(float))
        .tile(2 * sizeof(float), 1);
    cornersOutput.addTransfer()
        .dst(m_outCornerOp, 2 * sizeof(float))
        .src(corners_vmem, 2 * sizeof(float))
        .tile(2 * sizeof(float), 1);

    prog["img_w"]    = m_width;
    prog["img_h"]    = m_height;
    prog["win_size"] = mask_w;
    prog["mask"].set(mask.data(), MAX_WIN_SIZE * MAX_WIN_SIZE);

    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
}

void cornerSubPix::generateMaskInput(int32_t *mask)
{
    int32_t win_w      = m_winWidth;
    int32_t win_h      = m_winHeight;
    int32_t zeroZone_w = m_zeroZoneWidth;
    int32_t zeroZone_h = m_zeroZoneHeight;

    for (int32_t i = 0; i < (2 * win_h + 1); i++)
    {
        float y  = (float)(i - win_h) / win_h;
        float vy = std::exp(-y * y);

        for (int32_t j = 0; j < (2 * win_w + 1); j++)
        {
            float x = (float)(j - win_w) / win_w;
            float v = vy * std::exp(-x * x);

            mask[i * (2 * win_w + 1) + j] = (int32_t)(v * (1 << QBITS) + 0.5f);
        }
    }

    if (zeroZone_w >= 0 && zeroZone_h >= 0 && zeroZone_w < win_w && zeroZone_h < win_h)
    {
        for (int32_t i = win_h - zeroZone_h; i <= win_h + zeroZone_h; i++)
        {
            for (int32_t j = win_w - zeroZone_w; j <= win_w + zeroZone_w; j++)
            {
                mask[i * (2 * win_w + 1) + j] = 0;
            }
        }
    }
}

template<typename S>
void cornerSubPix::operator()(S stream, const nvcv::Tensor &inCorners, const nvcv::Tensor &outCorners,
                              const nvcv::Tensor &image, const int32_t maxIters, const float eps,
                              const int32_t numCorners)
{
    auto inCornerTensorData  = inCorners.exportData<nvcv::TensorDataStrided>();
    auto outCornerTensorData = outCorners.exportData<nvcv::TensorDataStrided>();
    auto imageTensorData     = image.exportData<nvcv::TensorDataStrided>();

    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_cornerTensorRequirements, inCornerTensorData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_cornerTensorRequirements, outCornerTensorData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_imageTensorRequirements, imageTensorData->cdata()))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Input and output tensors must have the same requirements that was used to create the operator");
    }
    if ((static_cast<void *>(inCornerTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(outCornerTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(imageTensorData->basePtr()) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor base pointers must not be NULL");
    }

    if (numCorners > 4096)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Max numCorners supported is 4096");
    }

    if (maxIters < 6)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "We don't support maxIters less than 6");
    }

    cupvaParameter_t param;
    CupvaCmdProgramGetParameter(&m_prog, &param, "num_corners");
    CupvaParameterSetValueScalar(&param, &numCorners, sizeof(int32_t));
    CupvaCmdProgramGetParameter(&m_prog, &param, "max_iters");
    CupvaParameterSetValueScalar(&param, &maxIters, sizeof(int32_t));
    CupvaCmdProgramGetParameter(&m_prog, &param, "eps");
    CupvaParameterSetValueScalar(&param, &eps, sizeof(float));

    m_inCornerOp.setBase(static_cast<float *>(static_cast<void *>(inCornerTensorData->basePtr())));
    m_outCornerOp.setBase(static_cast<float *>(static_cast<void *>(outCornerTensorData->basePtr())));
    m_imageOp.setBase(static_cast<uint8_t *>(static_cast<void *>(imageTensorData->basePtr())));

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
        static_assert(!sizeof(S), "Unsupported stream type for CornerSubPix operator");
    }
}

// Explicit template instantiation
template void cornerSubPix::operator()<cupvaStream_t>(cupvaStream_t, const nvcv::Tensor &, const nvcv::Tensor &,
                                                      const nvcv::Tensor &, const int32_t, const float, const int32_t);
template void cornerSubPix::operator()<cudaStream_t>(cudaStream_t, const nvcv::Tensor &, const nvcv::Tensor &,
                                                     const nvcv::Tensor &, const int32_t, const float, const int32_t);

cornerSubPix::~cornerSubPix()
{
    CupvaCmdDestroy(&m_prog);
}

} // namespace pva_solutions::priv

NVCVStatus pvaCornerSubPixCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements *cornerParams,
                                 NVCVTensorRequirements *imageParams, const NVCVSize2D winSize,
                                 const NVCVSize2D zeroZone)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        if (cornerParams == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to corners TensorRequirements must not be NULL");
        }
        if (imageParams == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to image TensorRequirements must not be NULL");
        }

        *handle = reinterpret_cast<NVCVOperatorHandle>(
            new pva_solutions::priv::cornerSubPix(cornerParams, imageParams, winSize, zeroZone));
    });
}

template<typename StreamType>
NVCVStatus CornerSubPixSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle inCorners,
                                  NVCVTensorHandle outCorners, NVCVTensorHandle image, const int32_t maxIters,
                                  const float eps, const int32_t numCorners)
{
    return nvcv::ProtectCall([&] {
        nvcv::TensorWrapHandle inCornersH(inCorners), outCornersH(outCorners), inImageH(image);
        pva_solutions::priv::ToDynamicRef<pva_solutions::priv::cornerSubPix>(handle)(
            stream, inCornersH, outCornersH, inImageH, maxIters, eps, numCorners);
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus CornerSubPixSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle,
                                                          NVCVTensorHandle, NVCVTensorHandle, const int32_t,
                                                          const float, const int32_t);
template NVCVStatus CornerSubPixSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle,
                                                         NVCVTensorHandle, NVCVTensorHandle, const int32_t, const float,
                                                         const int32_t);

// C++ overload implementations
NVCVStatus pvaCornerSubPixSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle inCorners,
                                 NVCVTensorHandle outCorners, NVCVTensorHandle image, const int32_t maxIters,
                                 const float eps, const int32_t numCorners)
{
    return CornerSubPixSubmitImpl(handle, stream, inCorners, outCorners, image, maxIters, eps, numCorners);
}

NVCVStatus pvaCornerSubPixSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle inCorners,
                                 NVCVTensorHandle outCorners, NVCVTensorHandle image, const int32_t maxIters,
                                 const float eps, const int32_t numCorners)
{
    return CornerSubPixSubmitImpl(handle, stream, inCorners, outCorners, image, maxIters, eps, numCorners);
}
