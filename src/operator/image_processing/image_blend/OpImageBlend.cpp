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

#include "OpImageBlend.hpp"

#include <IOperator.hpp>
#include <ImageUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <nvcv/Image.hpp>

#include <OpImageBlend.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <climits>
#include <cmath>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_image_blend_u8_dev);

namespace pva_solutions::priv {

template<typename T>
std::mutex ImageBlend<T>::s_execMapMutex;

template<typename T>
std::map<typename ImageBlend<T>::ExecKey, std::weak_ptr<cupva::Executable>> ImageBlend<T>::s_imageBlendExecutables;

template<typename T>
std::shared_ptr<cupva::Executable> ImageBlend<T>::GetImageBlendExec()
{
    static IOperator::ExecutableData sImageBlendExecData;

    sImageBlendExecData = {CUPVA_EXECUTABLE_DATA(pva_image_blend_u8_dev),
                           CUPVA_EXECUTABLE_SIZE(pva_image_blend_u8_dev)};

    ExecKey execKey = std::make_tuple(cupva::Context::GetCurrent());
    return IOperator::GetExecutable<ExecKey>(sImageBlendExecData, execKey, s_imageBlendExecutables, s_execMapMutex);
}

template<typename T>
ImageBlend<T>::ImageBlend(NVCVImageRequirements const *const imageRequirements)
    : m_imageRequirements{*imageRequirements}
    , m_src0Ops{cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr),
                cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr),
                cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr),
                cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr)}
    , m_src1Ops{cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr),
                cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr),
                cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr),
                cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr)}
    , m_dstOps{cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr),
               cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr),
               cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr),
               cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr)}
    , m_prog{}
{
    int32_t imageWidth     = imageRequirements->width;
    int32_t imageHeight    = imageRequirements->height;
    int32_t imageLinePitch = imageRequirements->planeRowStride[0] / sizeof(T);

    NVCVImageFormat imageFormat = imageRequirements->format;

    switch (imageFormat)
    {
    case NVCV_IMAGE_FORMAT_U8:
    case NVCV_IMAGE_FORMAT_YUYV:
    case NVCV_IMAGE_FORMAT_UYVY:
    case NVCV_IMAGE_FORMAT_VYUY:
    case NVCV_IMAGE_FORMAT_YUV8p:
    case NVCV_IMAGE_FORMAT_BGR8:
    case NVCV_IMAGE_FORMAT_RGB8:
    case NVCV_IMAGE_FORMAT_BGRA8:
    case NVCV_IMAGE_FORMAT_RGBA8:
    case NVCV_IMAGE_FORMAT_BGR8p:
    case NVCV_IMAGE_FORMAT_RGB8p:
        break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported image format");
    }

    m_exec    = GetImageBlendExec();
    auto prog = CmdProgram::Create(*m_exec.get());

    nvcvImageFormatGetNumPlanes(imageFormat, &m_imagePlaneCount);

    int32_t imageBitsPerPixel;
    nvcvImageFormatGetPlaneBitsPerPixel(imageFormat, 0, &imageBitsPerPixel);

    // channel count used in VPU kernels represents the number of channels per pixel
    int32_t imageChannelCount = (imageBitsPerPixel / CHAR_BIT) / sizeof(T);

    imageWidth *= imageChannelCount;
    const int32_t niterX = (imageWidth + TILE_WIDTH - 1) / TILE_WIDTH;
    const int32_t niterY = (imageHeight + TILE_HEIGHT - 1) / TILE_HEIGHT;

    prog["niterX"] = niterX;
    prog["niterY"] = niterY;

    prog["imagePlaneCount"] = m_imagePlaneCount;

    RasterDataFlow *src0DataFlow[MAX_CHANNEL_COUNT];
    RasterDataFlow *src1DataFlow[MAX_CHANNEL_COUNT];
    RasterDataFlow *dstDataFlow[MAX_CHANNEL_COUNT];

    int32_t roiWidth      = TILE_WIDTH > imageWidth ? TILE_WIDTH : imageWidth;
    int32_t roiHeight     = TILE_HEIGHT > imageHeight ? TILE_HEIGHT : imageHeight;
    int32_t dstTileWidth  = TILE_WIDTH < imageWidth ? TILE_WIDTH : imageWidth;
    int32_t dstTileHeight = TILE_HEIGHT < imageHeight ? TILE_HEIGHT : imageHeight;

    for (int i = 0; i < m_imagePlaneCount; i++)
    {
        src0DataFlow[i] = &prog.addDataFlowHead<RasterDataFlow>();
        src0DataFlow[i]
            ->handler(prog["src0Handler"])
            .src(m_src0Ops[i], imageWidth, imageHeight, imageLinePitch)
            .tileBuffer(prog[std::string("src0Buffer" + std::to_string(i)).c_str()].ptr<T>())
            .tile(TILE_WIDTH, TILE_HEIGHT)
            .roi(0, 0, roiWidth, roiHeight);

        src1DataFlow[i] = &prog.addDataFlowHead<RasterDataFlow>();
        src1DataFlow[i]
            ->handler(prog["src1Handler"])
            .src(m_src1Ops[i], imageWidth, imageHeight, imageLinePitch)
            .tileBuffer(prog[std::string("src1Buffer" + std::to_string(i)).c_str()].ptr<T>())
            .tile(TILE_WIDTH, TILE_HEIGHT)
            .roi(0, 0, roiWidth, roiHeight);

        dstDataFlow[i] = &prog.addDataFlowHead<RasterDataFlow>();
        dstDataFlow[i]
            ->handler(prog["dstHandler"])
            .tileBuffer(prog[std::string("dstBuffer" + std::to_string(i)).c_str()].ptr<T>())
            .dst(m_dstOps[i], imageWidth, imageHeight, imageLinePitch)
            .tile(dstTileWidth, dstTileHeight)
            .tileArena(TILE_WIDTH, TILE_HEIGHT);
    }

    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename T>
template<typename S>
void ImageBlend<T>::operator()(S stream, const nvcv::Image &in0, const nvcv::Image &in1, const float alpha,
                               const nvcv::Image &out)
{
    auto src0ImageData = in0.exportData<nvcv::ImageDataStrided>();
    auto src1ImageData = in1.exportData<nvcv::ImageDataStrided>();
    auto dstImageData  = out.exportData<nvcv::ImageDataStrided>();

    if ((static_cast<void *>(src0ImageData->plane(0).basePtr) == nullptr) ||
        (static_cast<void *>(src1ImageData->plane(0).basePtr) == nullptr) ||
        (static_cast<void *>(dstImageData->plane(0).basePtr) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Image base pointers must not be NULL");
    }

    for (int32_t i = 0; i < m_imagePlaneCount; i++)
    {
        m_src0Ops[i].setBase(static_cast<T *>(static_cast<void *>(src0ImageData->plane(i).basePtr)));
        m_src1Ops[i].setBase(static_cast<T *>(static_cast<void *>(src1ImageData->plane(i).basePtr)));
        m_dstOps[i].setBase(static_cast<T *>(static_cast<void *>(dstImageData->plane(i).basePtr)));
    }

    if (alpha < 0.f || alpha > 1.0f)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Alpha should be in range [0.0, 1.0]");
    }

    uint32_t fixedPointAlpha = std::round(alpha * QBITS_VALUE_U8);

    cupvaParameter_t param;
    CupvaCmdProgramGetParameter(&m_prog, &param, "alpha");
    CupvaParameterSetValueScalar(&param, &fixedPointAlpha, sizeof(uint32_t));

    cupvaError_t err = CupvaCmdProgramUpdateDataFlows(&m_prog);

    cupvaCmd_t const *submitCmds[1] = {&m_prog};
    cupvaCmdStatus_t cmdStatus[1]   = {nullptr};

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
        static_assert(!sizeof(S), "Unsupported stream type for ImageBlend operator");
    }

    if (!(err == CUPVA_ERROR_NONE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Submit failed");
    }
}

template<typename T>
ImageBlend<T>::~ImageBlend()
{
    CupvaCmdDestroy(&m_prog);
}

// Explicit instantiation of the template to support the types we care about
template class ImageBlend<uint8_t>;

} // namespace pva_solutions::priv

NVCVStatus pvaImageBlendCreate(NVCVOperatorHandle *handle, NVCVImageRequirements const *const imageRequirements)
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
        NVCVDataType dataType;
        nvcvImageFormatGetPlaneDataType(imageRequirements->format, 0, &dataType);
        switch (dataType)
        {
        case NVCV_DATA_TYPE_U8:
        case NVCV_DATA_TYPE_2U8:
        case NVCV_DATA_TYPE_3U8:
        case NVCV_DATA_TYPE_4U8:
            *handle =
                reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::ImageBlend<uint8_t>(imageRequirements));
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }
    });
}

template<typename StreamType>
NVCVStatus ImageBlendSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVImageHandle const src0,
                                NVCVImageHandle const src1, const float alpha, NVCVImageHandle dst)
{
    return nvcv::ProtectCall([&] {
        nvcv::ImageWrapHandle src0WrapHandle(src0), src1WrapHandle(src1), dstWrapHandle(dst);

        NVCVImageFormat imageFormat;
        nvcvImageGetFormat(src0, &imageFormat);

        NVCVDataType imageDataType;
        nvcvImageFormatGetPlaneDataType(imageFormat, 0, &imageDataType);
        switch (imageDataType)
        {
        case NVCV_DATA_TYPE_U8:
        case NVCV_DATA_TYPE_2U8:
        case NVCV_DATA_TYPE_3U8:
        case NVCV_DATA_TYPE_4U8:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::ImageBlend<uint8_t>>(handle)(
                stream, src0WrapHandle, src1WrapHandle, alpha, dstWrapHandle);
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus ImageBlendSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVImageHandle const,
                                                        NVCVImageHandle const, const float, NVCVImageHandle);
template NVCVStatus ImageBlendSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVImageHandle const,
                                                       NVCVImageHandle const, const float, NVCVImageHandle);

// C++ overload implementations
NVCVStatus pvaImageBlendSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVImageHandle const src0,
                               NVCVImageHandle const src1, const float alpha, NVCVImageHandle dst)
{
    return ImageBlendSubmitImpl(handle, stream, src0, src1, alpha, dst);
}

NVCVStatus pvaImageBlendSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageHandle const src0,
                               NVCVImageHandle const src1, const float alpha, NVCVImageHandle dst)
{
    return ImageBlendSubmitImpl(handle, stream, src0, src1, alpha, dst);
}
