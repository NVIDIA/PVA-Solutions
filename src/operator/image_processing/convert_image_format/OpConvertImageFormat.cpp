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

#include "OpConvertImageFormat.hpp"

#include "device/convert_image_format_param.h"

#include <IOperator.hpp>
#include <ImageUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <nvcv/Image.hpp>

#include <OpConvertImageFormat.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <climits>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_convert_image_format_u8_dev);

namespace pva_solutions::priv {

template<typename T>
std::mutex ConvertImageFormat<T>::s_execMapMutex;

template<typename T>
std::map<typename ConvertImageFormat<T>::ExecKey, std::weak_ptr<cupva::Executable>>
    ConvertImageFormat<T>::s_convertImageFormatExecutables;

template<typename T>
std::shared_ptr<cupva::Executable> ConvertImageFormat<T>::GetConvertImageFormatExec()
{
    static IOperator::ExecutableData sConvertImageFormatExecData;

    sConvertImageFormatExecData = {CUPVA_EXECUTABLE_DATA(pva_convert_image_format_u8_dev),
                                   CUPVA_EXECUTABLE_SIZE(pva_convert_image_format_u8_dev)};

    ExecKey execKey = std::make_tuple(cupva::Context::GetCurrent());
    return IOperator::GetExecutable<ExecKey>(sConvertImageFormatExecData, execKey, s_convertImageFormatExecutables,
                                             s_execMapMutex);
}

template<typename T>
ImageFormatConversionCode ConvertImageFormat<T>::getImageFormatConversionCode(NVCVImageFormat inFormat,
                                                                              NVCVImageFormat outFormat)
{
    std::map<std::tuple<NVCVImageFormat, NVCVImageFormat>, ImageFormatConversionCode> codeMap = {
        {{NVCV_IMAGE_FORMAT_YUYV, NVCV_IMAGE_FORMAT_NV12}, CONVERT_IMAGE_FORMAT_YUYV2NV12},
        {{NVCV_IMAGE_FORMAT_UYVY, NVCV_IMAGE_FORMAT_NV12}, CONVERT_IMAGE_FORMAT_UYVY2NV12},
        {{NVCV_IMAGE_FORMAT_VYUY, NVCV_IMAGE_FORMAT_NV12}, CONVERT_IMAGE_FORMAT_VYUY2NV12},
        {{NVCV_IMAGE_FORMAT_YUV8p, NVCV_IMAGE_FORMAT_NV12}, CONVERT_IMAGE_FORMAT_YUV8p2NV12},
        {{NVCV_IMAGE_FORMAT_YUV8p, NVCV_IMAGE_FORMAT_YUYV}, CONVERT_IMAGE_FORMAT_YUV8p2YUYV},
        {{NVCV_IMAGE_FORMAT_YUV8p, NVCV_IMAGE_FORMAT_UYVY}, CONVERT_IMAGE_FORMAT_YUV8p2UYVY},
        {{NVCV_IMAGE_FORMAT_YUV8p, NVCV_IMAGE_FORMAT_VYUY}, CONVERT_IMAGE_FORMAT_YUV8p2VYUY},
        {{NVCV_IMAGE_FORMAT_NV12, NVCV_IMAGE_FORMAT_BGR8}, CONVERT_IMAGE_FORMAT_NV122BGR8},
        {{NVCV_IMAGE_FORMAT_NV12, NVCV_IMAGE_FORMAT_RGB8}, CONVERT_IMAGE_FORMAT_NV122RGB8},
        {{NVCV_IMAGE_FORMAT_NV12, NVCV_IMAGE_FORMAT_BGR8p}, CONVERT_IMAGE_FORMAT_NV122BGR8p},
        {{NVCV_IMAGE_FORMAT_NV12, NVCV_IMAGE_FORMAT_RGB8p}, CONVERT_IMAGE_FORMAT_NV122RGB8p},
        {{NVCV_IMAGE_FORMAT_YUYV, NVCV_IMAGE_FORMAT_BGR8}, CONVERT_IMAGE_FORMAT_YUYV2BGR8},
        {{NVCV_IMAGE_FORMAT_YUYV, NVCV_IMAGE_FORMAT_RGB8}, CONVERT_IMAGE_FORMAT_YUYV2RGB8},
        {{NVCV_IMAGE_FORMAT_UYVY, NVCV_IMAGE_FORMAT_BGR8}, CONVERT_IMAGE_FORMAT_UYVY2BGR8},
        {{NVCV_IMAGE_FORMAT_UYVY, NVCV_IMAGE_FORMAT_RGB8}, CONVERT_IMAGE_FORMAT_UYVY2RGB8},
        {{NVCV_IMAGE_FORMAT_VYUY, NVCV_IMAGE_FORMAT_BGR8}, CONVERT_IMAGE_FORMAT_VYUY2BGR8},
        {{NVCV_IMAGE_FORMAT_VYUY, NVCV_IMAGE_FORMAT_RGB8}, CONVERT_IMAGE_FORMAT_VYUY2RGB8},
        {{NVCV_IMAGE_FORMAT_YUYV, NVCV_IMAGE_FORMAT_BGR8p}, CONVERT_IMAGE_FORMAT_YUYV2BGR8p},
        {{NVCV_IMAGE_FORMAT_YUYV, NVCV_IMAGE_FORMAT_RGB8p}, CONVERT_IMAGE_FORMAT_YUYV2RGB8p},
        {{NVCV_IMAGE_FORMAT_UYVY, NVCV_IMAGE_FORMAT_BGR8p}, CONVERT_IMAGE_FORMAT_UYVY2BGR8p},
        {{NVCV_IMAGE_FORMAT_UYVY, NVCV_IMAGE_FORMAT_RGB8p}, CONVERT_IMAGE_FORMAT_UYVY2RGB8p},
        {{NVCV_IMAGE_FORMAT_VYUY, NVCV_IMAGE_FORMAT_BGR8p}, CONVERT_IMAGE_FORMAT_VYUY2BGR8p},
        {{NVCV_IMAGE_FORMAT_VYUY, NVCV_IMAGE_FORMAT_RGB8p}, CONVERT_IMAGE_FORMAT_VYUY2RGB8p},
        {{NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRA8}, CONVERT_IMAGE_FORMAT_RGB82BGRA8},
        {{NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_NV12}, CONVERT_IMAGE_FORMAT_BGR82NV12},
        {{NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_NV12}, CONVERT_IMAGE_FORMAT_RGB82NV12},
        {{NVCV_IMAGE_FORMAT_BGR8p, NVCV_IMAGE_FORMAT_NV12}, CONVERT_IMAGE_FORMAT_BGR8p2NV12},
        {{NVCV_IMAGE_FORMAT_RGB8p, NVCV_IMAGE_FORMAT_NV12}, CONVERT_IMAGE_FORMAT_RGB8p2NV12},
        {{NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_YUYV}, CONVERT_IMAGE_FORMAT_BGR82YUYV},
        {{NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_YUYV}, CONVERT_IMAGE_FORMAT_RGB82YUYV},
        {{NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_UYVY}, CONVERT_IMAGE_FORMAT_BGR82UYVY},
        {{NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_UYVY}, CONVERT_IMAGE_FORMAT_RGB82UYVY},
        {{NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_VYUY}, CONVERT_IMAGE_FORMAT_BGR82VYUY},
        {{NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_VYUY}, CONVERT_IMAGE_FORMAT_RGB82VYUY},
        {{NVCV_IMAGE_FORMAT_BGR8p, NVCV_IMAGE_FORMAT_YUYV}, CONVERT_IMAGE_FORMAT_BGR8p2YUYV},
        {{NVCV_IMAGE_FORMAT_RGB8p, NVCV_IMAGE_FORMAT_YUYV}, CONVERT_IMAGE_FORMAT_RGB8p2YUYV},
        {{NVCV_IMAGE_FORMAT_BGR8p, NVCV_IMAGE_FORMAT_UYVY}, CONVERT_IMAGE_FORMAT_BGR8p2UYVY},
        {{NVCV_IMAGE_FORMAT_RGB8p, NVCV_IMAGE_FORMAT_UYVY}, CONVERT_IMAGE_FORMAT_RGB8p2UYVY},
        {{NVCV_IMAGE_FORMAT_BGR8p, NVCV_IMAGE_FORMAT_VYUY}, CONVERT_IMAGE_FORMAT_BGR8p2VYUY},
        {{NVCV_IMAGE_FORMAT_RGB8p, NVCV_IMAGE_FORMAT_VYUY}, CONVERT_IMAGE_FORMAT_RGB8p2VYUY},
        /* YUV → NV12_ER: chroma resampling only — no color-matrix conversion,
         * so the studio/ER coefficient distinction doesn't apply. Safe to
         * enable now. */
        {{NVCV_IMAGE_FORMAT_YUYV, NVCV_IMAGE_FORMAT_NV12_ER}, CONVERT_IMAGE_FORMAT_YUYV2NV12_ER},
        {{NVCV_IMAGE_FORMAT_UYVY, NVCV_IMAGE_FORMAT_NV12_ER}, CONVERT_IMAGE_FORMAT_UYVY2NV12_ER},
        {{NVCV_IMAGE_FORMAT_VYUY, NVCV_IMAGE_FORMAT_NV12_ER}, CONVERT_IMAGE_FORMAT_VYUY2NV12_ER},
        {{NVCV_IMAGE_FORMAT_YUV8p, NVCV_IMAGE_FORMAT_NV12_ER}, CONVERT_IMAGE_FORMAT_YUV8p2NV12_ER},
        /* RGB8 / BGR8 / planar → NV12_ER (full-range RGB→YUV). The kernel uses
         * cfg.lumaOffset = 0 (no +16 Y bias) and the ER color matrix. */
        {{NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_NV12_ER}, CONVERT_IMAGE_FORMAT_BGR82NV12_ER},
        {{NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_NV12_ER}, CONVERT_IMAGE_FORMAT_RGB82NV12_ER},
        {{NVCV_IMAGE_FORMAT_BGR8p, NVCV_IMAGE_FORMAT_NV12_ER}, CONVERT_IMAGE_FORMAT_BGR8p2NV12_ER},
        {{NVCV_IMAGE_FORMAT_RGB8p, NVCV_IMAGE_FORMAT_NV12_ER}, CONVERT_IMAGE_FORMAT_RGB8p2NV12_ER},
        /* NV12_ER (full range) → packed/planar RGB/BGR. The kernel skips the Y-16
         * luma offset via cfg.lumaOffset = 0 and loads the ER color matrix. */
        {{NVCV_IMAGE_FORMAT_NV12_ER, NVCV_IMAGE_FORMAT_RGB8}, CONVERT_IMAGE_FORMAT_NV12_ER2RGB8},
        {{NVCV_IMAGE_FORMAT_NV12_ER, NVCV_IMAGE_FORMAT_BGR8}, CONVERT_IMAGE_FORMAT_NV12_ER2BGR8},
        {{NVCV_IMAGE_FORMAT_NV12_ER, NVCV_IMAGE_FORMAT_RGB8p}, CONVERT_IMAGE_FORMAT_NV12_ER2RGB8p},
        {{NVCV_IMAGE_FORMAT_NV12_ER, NVCV_IMAGE_FORMAT_BGR8p}, CONVERT_IMAGE_FORMAT_NV12_ER2BGR8p},
        /* NV12_ER → packed RGBA8: as above with channel_out = 4; alpha = 255. */
        {{NVCV_IMAGE_FORMAT_NV12_ER, NVCV_IMAGE_FORMAT_RGBA8}, CONVERT_IMAGE_FORMAT_NV12_ER2RGBA8},
        /* RGB → RGBA / BGR → BGRA (3→4, no R/B swap; alpha = 255).
         * Reuses the existing rgba2rgba kernel with in-order R/G/B output
         * pointers (RGB→BGRA reverses them). */
        {{NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGBA8}, CONVERT_IMAGE_FORMAT_RGB82RGBA8},
        {{NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_BGRA8}, CONVERT_IMAGE_FORMAT_BGR82BGRA8},
        /* RGBA → RGB / BGRA → BGR (4→3; alpha lane dropped). */
        {{NVCV_IMAGE_FORMAT_RGBA8, NVCV_IMAGE_FORMAT_RGB8}, CONVERT_IMAGE_FORMAT_RGBA82RGB8},
        {{NVCV_IMAGE_FORMAT_BGRA8, NVCV_IMAGE_FORMAT_BGR8}, CONVERT_IMAGE_FORMAT_BGRA82BGR8},
        /* NOTE: single-pass studio-range NV12 → RGBA8 still chains via RGB8
         * (NV12→RGB8 + RGB8→RGBA8); only the NV12_ER RGBA path is direct. */
    };

    auto codeIt = codeMap.find(std::tuple<NVCVImageFormat, NVCVImageFormat>(inFormat, outFormat));
    if (codeIt != codeMap.end())
    {
        return codeIt->second;
    }
    else
    {
        throw std::invalid_argument("Unsupported input/output image format conversion");
    }
}

template<typename T>
ConvertImageFormat<T>::ConvertImageFormat(NVCVImageRequirements const *const inImageRequirements,
                                          NVCVImageRequirements const *const outImageRequirements)
    : m_inImageRequirements{*inImageRequirements}
    , m_outImageRequirements{*outImageRequirements}
    , m_width(inImageRequirements->width)
    , m_height(inImageRequirements->height)
    , m_inputLinePitch(inImageRequirements->planeRowStride[0] / sizeof(T))
    , m_outputLinePitch(outImageRequirements->planeRowStride[0] / sizeof(T))
    , m_inFormat(inImageRequirements->format)
    , m_outFormat(outImageRequirements->format)
    , m_srcOps{cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr),
               cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr),
               cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr),
               cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr)}
    , m_dstOps{cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr),
               cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr),
               cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr),
               cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr)}
    , m_prog{}
{
    const int supported_tile_size = std::max(TILE_W, TILE_H);
    if (m_width < supported_tile_size)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "The smallest supported image width is %d",
                              supported_tile_size);
    }
    if (m_height < supported_tile_size)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "The smallest supported image height is %d",
                              supported_tile_size);
    }

    ImageFormatConversionCode code = getImageFormatConversionCode(m_inFormat, m_outFormat);

    m_exec    = GetConvertImageFormatExec();
    auto prog = CmdProgram::Create(*m_exec.get());

    nvcvImageFormatGetNumPlanes(m_inFormat, &m_inPlaneCount);
    nvcvImageFormatGetNumPlanes(m_outFormat, &m_outPlaneCount);

    int32_t inBitsPerPixel, outBitsPerPixel;
    nvcvImageFormatGetPlaneBitsPerPixel(m_inFormat, 0, &inBitsPerPixel);
    nvcvImageFormatGetPlaneBitsPerPixel(m_outFormat, 0, &outBitsPerPixel);

    // channel count used in VPU kernels represents the number of channels per pixel
    int32_t inChannelCount  = (inBitsPerPixel / CHAR_BIT) / sizeof(T);
    int32_t outChannelCount = (outBitsPerPixel / CHAR_BIT) / sizeof(T);

    int tileWidth  = TILE_W;
    int tileHeight = TILE_H;
    if (((m_inPlaneCount >= 3) && (outChannelCount <= 2)) || ((m_outPlaneCount >= 3) && (inChannelCount <= 2)))
    {
        // set tile width = 128 for performance reason
        tileWidth  = TILE_H;
        tileHeight = TILE_W;
    }

    // calculate the largest tile size
    int32_t scale        = std::max(1, std::min(MAX_CHANNEL_COUNT / m_inPlaneCount / inChannelCount,
                                                MAX_CHANNEL_COUNT / m_outPlaneCount / outChannelCount));
    int32_t inTileWidth  = tileWidth * inChannelCount * scale;
    int32_t outTileWidth = tileWidth * outChannelCount * scale;

    const int32_t inImageWidth  = m_width * inChannelCount;
    const int32_t outImageWidth = m_width * outChannelCount;
    const int32_t niterX        = (inImageWidth + inTileWidth - 1) / inTileWidth;
    const int32_t niterY        = (m_height + tileHeight - 1) / tileHeight;

    prog["niterX"] = niterX;
    prog["niterY"] = niterY;
    prog["code"]   = static_cast<int32_t>(code);

    prog["inPlaneCount"]    = m_inPlaneCount;
    prog["inChannelCount"]  = inChannelCount;
    prog["inTileWidth"]     = inTileWidth;
    prog["outPlaneCount"]   = m_outPlaneCount;
    prog["outChannelCount"] = outChannelCount;
    prog["outTileWidth"]    = outTileWidth;
    prog["tileHeight"]      = tileHeight;

    for (int32_t i = 0; i < m_inPlaneCount; i++)
    {
        int32_t inPlaneWidth  = inImageWidth;
        int32_t inPlaneHeight = m_height;
        int32_t inTileHeight  = tileHeight;
        if ((m_inFormat == NVCV_IMAGE_FORMAT_NV12 || m_inFormat == NVCV_IMAGE_FORMAT_NV12_ER) && (i == 1))
        {
            inPlaneWidth  = (inImageWidth + 1) / 2 * 2;
            inPlaneHeight = (m_height + 1) / 2;
            inTileHeight  = tileHeight / 2;
        }

        prog.addDataFlowHead<RasterDataFlow>()
            .handler(prog[std::string("srcHdl" + std::to_string(i)).c_str()])
            .src(m_srcOps[i], inPlaneWidth, inPlaneHeight, m_inputLinePitch)
            .tileBuffer(prog[std::string("srcBuf" + std::to_string(i)).c_str()].ptr<T>())
            .tile(inTileWidth, inTileHeight)
            .transpose(TRANS_MODE_1);
    }

    for (int32_t i = 0; i < m_outPlaneCount; i++)
    {
        int32_t outPlaneWidth  = outImageWidth;
        int32_t outPlaneHeight = m_height;
        int32_t outTileHeight  = tileHeight;
        if ((m_outFormat == NVCV_IMAGE_FORMAT_NV12 || m_outFormat == NVCV_IMAGE_FORMAT_NV12_ER) && (i == 1))
        {
            outPlaneWidth  = (outImageWidth + 1) / 2 * 2;
            outPlaneHeight = (m_height + 1) / 2;
            outTileHeight  = tileHeight / 2;
        }

        prog.addDataFlowHead<RasterDataFlow>()
            .handler(prog[std::string("dstHdl" + std::to_string(i)).c_str()])
            .tileBuffer(prog[std::string("dstBuf" + std::to_string(i)).c_str()].ptr<T>())
            .dst(m_dstOps[i], outPlaneWidth, outPlaneHeight, m_outputLinePitch)
            .tile(outTileWidth, outTileHeight)
            .transpose(TRANS_MODE_1);
    }

    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename T>
template<typename S>
void ConvertImageFormat<T>::operator()(S stream, const nvcv::Image &in, const nvcv::Image &out)
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
    if ((static_cast<void *>(inImageData->plane(0).basePtr) == nullptr) ||
        (static_cast<void *>(outImageData->plane(0).basePtr) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Image base pointers must not be NULL");
    }

    for (int32_t i = 0; i < m_inPlaneCount; i++)
    {
        m_srcOps[i].setBase(static_cast<T *>(static_cast<void *>(inImageData->plane(i).basePtr)));
    }
    for (int32_t i = 0; i < m_outPlaneCount; i++)
    {
        m_dstOps[i].setBase(static_cast<T *>(static_cast<void *>(outImageData->plane(i).basePtr)));
    }
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
        static_assert(!sizeof(S), "Unsupported stream type for ConvertImageFormat operator");
    }

    if (!(err == CUPVA_ERROR_NONE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Submit failed");
    }
}

template<typename T>
ConvertImageFormat<T>::~ConvertImageFormat()
{
    CupvaCmdDestroy(&m_prog);
}

// Explicit instantiation of the template to support the types we care about
template class ConvertImageFormat<uint8_t>;

} // namespace pva_solutions::priv

NVCVStatus pvaConvertImageFormatCreate(NVCVOperatorHandle *handle,
                                       NVCVImageRequirements const *const inImageRequirements,
                                       NVCVImageRequirements const *const outImageRequirements)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        if (inImageRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to input ImageRequirements must not be NULL");
        }
        if (outImageRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to output ImageRequirements must not be NULL");
        }
        NVCVDataType dataType;
        nvcvImageFormatGetPlaneDataType(inImageRequirements->format, 0, &dataType);
        switch (dataType)
        {
        case NVCV_DATA_TYPE_U8:
        case NVCV_DATA_TYPE_2U8:
        case NVCV_DATA_TYPE_3U8:
        case NVCV_DATA_TYPE_4U8:
            *handle = reinterpret_cast<NVCVOperatorHandle>(
                new pva_solutions::priv::ConvertImageFormat<uint8_t>(inImageRequirements, outImageRequirements));
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }
    });
}

template<typename StreamType>
NVCVStatus ConvertImageFormatSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVImageHandle const in,
                                        NVCVImageHandle out)
{
    return nvcv::ProtectCall([&] {
        nvcv::ImageWrapHandle input(in), output(out);

        NVCVImageFormat inFormat;
        nvcvImageGetFormat(in, &inFormat);

        NVCVDataType inDataType;
        nvcvImageFormatGetPlaneDataType(inFormat, 0, &inDataType);
        switch (inDataType)
        {
        case NVCV_DATA_TYPE_U8:
        case NVCV_DATA_TYPE_2U8:
        case NVCV_DATA_TYPE_3U8:
        case NVCV_DATA_TYPE_4U8:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::ConvertImageFormat<uint8_t>>(handle)(stream, input,
                                                                                                        output);
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus ConvertImageFormatSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t,
                                                                NVCVImageHandle const, NVCVImageHandle);
template NVCVStatus ConvertImageFormatSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVImageHandle const,
                                                               NVCVImageHandle);

// C++ overload implementations
NVCVStatus pvaConvertImageFormatSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVImageHandle const in,
                                       NVCVImageHandle out)
{
    return ConvertImageFormatSubmitImpl(handle, stream, in, out);
}

NVCVStatus pvaConvertImageFormatSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageHandle const in,
                                       NVCVImageHandle out)
{
    return ConvertImageFormatSubmitImpl(handle, stream, in, out);
}
