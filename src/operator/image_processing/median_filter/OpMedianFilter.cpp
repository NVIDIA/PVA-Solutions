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

#include "OpMedianFilter.hpp"

#include "device/median_filter_param.h"

#include <ImageUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <cupva_host_utils.hpp>
#include <nvcv/Image.hpp>

#include <OpMedianFilter.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>

#include <array>
#include <cassert>
#include <iostream>

using namespace cupva;

// no reuse sorting networks for 3x3 kernels and interleaved data format 2S16
CUPVA_DECLARE_EXECUTABLE(pva_median_filter_u8_3x3_dev);
CUPVA_DECLARE_EXECUTABLE(pva_median_filter_s8_3x3_dev);
CUPVA_DECLARE_EXECUTABLE(pva_median_filter_u16_3x3_dev);
CUPVA_DECLARE_EXECUTABLE(pva_median_filter_s16_3x3_dev);
CUPVA_DECLARE_EXECUTABLE(pva_median_filter_i2s16_3x3_dev);
CUPVA_DECLARE_EXECUTABLE(pva_median_filter_u32_3x3_dev);
CUPVA_DECLARE_EXECUTABLE(pva_median_filter_s32_3x3_dev);

// separable 2x2 via channel planes for interleaved 2S16
CUPVA_DECLARE_EXECUTABLE(pva_median_filter_sepa_2x2_i2s16_5x5_dev);
CUPVA_DECLARE_EXECUTABLE(pva_median_filter_sepa_2x2_i2s16_5x7_dev);
CUPVA_DECLARE_EXECUTABLE(pva_median_filter_sepa_2x2_i2s16_7x7_dev);

// separable 2x2 sorting networks for kernel sizes 5x5, 5x7, 7x7
CUPVA_DECLARE_EXECUTABLE(pva_median_filter_sepa_2x2_u8_5x5_dev);
CUPVA_DECLARE_EXECUTABLE(pva_median_filter_sepa_2x2_u8_5x7_dev);
CUPVA_DECLARE_EXECUTABLE(pva_median_filter_sepa_2x2_u8_7x7_dev);
CUPVA_DECLARE_EXECUTABLE(pva_median_filter_sepa_2x2_s8_5x5_dev);
CUPVA_DECLARE_EXECUTABLE(pva_median_filter_sepa_2x2_s8_5x7_dev);
CUPVA_DECLARE_EXECUTABLE(pva_median_filter_sepa_2x2_s8_7x7_dev);
CUPVA_DECLARE_EXECUTABLE(pva_median_filter_sepa_2x2_u16_5x5_dev);
CUPVA_DECLARE_EXECUTABLE(pva_median_filter_sepa_2x2_u16_5x7_dev);
CUPVA_DECLARE_EXECUTABLE(pva_median_filter_sepa_2x2_u16_7x7_dev);
CUPVA_DECLARE_EXECUTABLE(pva_median_filter_sepa_2x2_s16_5x5_dev);
CUPVA_DECLARE_EXECUTABLE(pva_median_filter_sepa_2x2_s16_5x7_dev);
CUPVA_DECLARE_EXECUTABLE(pva_median_filter_sepa_2x2_s16_7x7_dev);
CUPVA_DECLARE_EXECUTABLE(pva_median_filter_sepa_2x2_u32_5x5_dev);
CUPVA_DECLARE_EXECUTABLE(pva_median_filter_sepa_2x2_u32_5x7_dev);
CUPVA_DECLARE_EXECUTABLE(pva_median_filter_sepa_2x2_u32_7x7_dev);
CUPVA_DECLARE_EXECUTABLE(pva_median_filter_sepa_2x2_s32_5x5_dev);
CUPVA_DECLARE_EXECUTABLE(pva_median_filter_sepa_2x2_s32_5x7_dev);
CUPVA_DECLARE_EXECUTABLE(pva_median_filter_sepa_2x2_s32_7x7_dev);

namespace pva_solutions::priv {

template<typename T, uint8_t CH>
std::mutex MedianFilter<T, CH>::s_execMapMutex;

template<typename T, uint8_t CH>
std::map<typename MedianFilter<T, CH>::ExecKey, std::weak_ptr<cupva::Executable>>
    MedianFilter<T, CH>::s_medianFilterExecutables;

template<typename T, uint8_t CH>
using ExecDataMap = std::map<typename MedianFilter<T, CH>::ExecDataKey, IOperator::ExecutableData>;

template<typename T, uint8_t CH>
ExecDataMap<T, CH> createExecDataMap();

#define MEDIAN_FILTER_EXECUTABLE_DATA(EXECUTABLE)                            \
    {                                                                        \
        CUPVA_EXECUTABLE_DATA(EXECUTABLE), CUPVA_EXECUTABLE_SIZE(EXECUTABLE) \
    }

#define DEFINE_SCALAR_EXEC_DATA_MAP(TYPE, TYPE_SUFFIX)                                                        \
    template<>                                                                                                \
    ExecDataMap<TYPE, 1> createExecDataMap<TYPE, 1>()                                                         \
    {                                                                                                         \
        return {{{3, 3}, MEDIAN_FILTER_EXECUTABLE_DATA(pva_median_filter_##TYPE_SUFFIX##_3x3_dev)},           \
                {{5, 5}, MEDIAN_FILTER_EXECUTABLE_DATA(pva_median_filter_sepa_2x2_##TYPE_SUFFIX##_5x5_dev)},  \
                {{5, 7}, MEDIAN_FILTER_EXECUTABLE_DATA(pva_median_filter_sepa_2x2_##TYPE_SUFFIX##_5x7_dev)},  \
                {{7, 7}, MEDIAN_FILTER_EXECUTABLE_DATA(pva_median_filter_sepa_2x2_##TYPE_SUFFIX##_7x7_dev)}}; \
    }

DEFINE_SCALAR_EXEC_DATA_MAP(uint8_t, u8)
DEFINE_SCALAR_EXEC_DATA_MAP(int8_t, s8)
DEFINE_SCALAR_EXEC_DATA_MAP(uint16_t, u16)
DEFINE_SCALAR_EXEC_DATA_MAP(int16_t, s16)
DEFINE_SCALAR_EXEC_DATA_MAP(uint32_t, u32)
DEFINE_SCALAR_EXEC_DATA_MAP(int32_t, s32)

template<>
ExecDataMap<int16_t, 2> createExecDataMap<int16_t, 2>()
{
    return {{{3, 3}, MEDIAN_FILTER_EXECUTABLE_DATA(pva_median_filter_i2s16_3x3_dev)},
            {{5, 5}, MEDIAN_FILTER_EXECUTABLE_DATA(pva_median_filter_sepa_2x2_i2s16_5x5_dev)},
            {{5, 7}, MEDIAN_FILTER_EXECUTABLE_DATA(pva_median_filter_sepa_2x2_i2s16_5x7_dev)},
            {{7, 7}, MEDIAN_FILTER_EXECUTABLE_DATA(pva_median_filter_sepa_2x2_i2s16_7x7_dev)}};
}

#undef DEFINE_SCALAR_EXEC_DATA_MAP
#undef MEDIAN_FILTER_EXECUTABLE_DATA

template<typename T, uint8_t CH>
std::shared_ptr<cupva::Executable> MedianFilter<T, CH>::GetMedianFilterExec(ExecDataKey execDataKey)
{
    static const auto execDataMap = createExecDataMap<T, CH>();

    auto execDataIt = execDataMap.find(execDataKey);
    if (execDataIt == execDataMap.end())
    {
        throw std::invalid_argument("Unsupported type or parameters");
    }

    // Create execKey by combining execDataKey and context
    ExecKey execKey = std::tuple_cat(execDataKey, std::make_tuple(cupva::Context::GetCurrent()));
    return GetExecutable(execDataIt->second, execKey, s_medianFilterExecutables, s_execMapMutex);
};

template<typename T, uint8_t CH>
void MedianFilter<T, CH>::checkImageFormat(NVCVImageFormat format)
{
    bool isCompatible = false;
    if constexpr (std::is_same_v<T, uint8_t> && CH == 1)
    {
        isCompatible =
            format == NVCV_IMAGE_FORMAT_U8 || format == NVCV_IMAGE_FORMAT_Y8 || format == NVCV_IMAGE_FORMAT_Y8_ER;
    }
    else if constexpr (std::is_same_v<T, uint16_t> && CH == 1)
    {
        isCompatible =
            format == NVCV_IMAGE_FORMAT_U16 || format == NVCV_IMAGE_FORMAT_Y16 || format == NVCV_IMAGE_FORMAT_Y16_ER;
    }
    else if constexpr (std::is_same_v<T, int8_t> && CH == 1)
    {
        isCompatible = format == NVCV_IMAGE_FORMAT_S8;
    }
    else if constexpr (std::is_same_v<T, int16_t> && CH == 1)
    {
        isCompatible = format == NVCV_IMAGE_FORMAT_S16;
    }
    else if constexpr (std::is_same_v<T, int16_t> && CH == 2)
    {
        isCompatible = format == NVCV_IMAGE_FORMAT_2S16;
    }
    else if constexpr (std::is_same_v<T, uint32_t> && CH == 1)
    {
        isCompatible = format == NVCV_IMAGE_FORMAT_U32;
    }
    else if constexpr (std::is_same_v<T, int32_t> && CH == 1)
    {
        isCompatible = format == NVCV_IMAGE_FORMAT_S32;
    }

    if (!isCompatible)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported image format");
    }
};

template<typename T, uint8_t CH>
MedianFilter<T, CH>::MedianFilter(NVCVImageRequirements *inImageRequirements,
                                  NVCVTensorRequirements *kernelRequirements,
                                  NVCVImageRequirements *outImageRequirements, NVCVBorderType borderType,
                                  int32_t borderValue)
    : m_inImageRequirements(*inImageRequirements)
    , m_outImageRequirements(*outImageRequirements)
    , m_kernelRequirements(*kernelRequirements)
    , m_imageWidth(inImageRequirements->width)
    , m_imageHeight(inImageRequirements->height)
    , m_srcOp(cupva::OffsetPointer<TSRC>::Create(cupva::MemType::DRAM, nullptr))
    , m_dstOp(cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr))
    , m_prog{}
{
    // check input == output size and format
    if (inImageRequirements->width != outImageRequirements->width ||
        inImageRequirements->height != outImageRequirements->height)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output image must have the same size");
    }
    if (inImageRequirements->format != outImageRequirements->format)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input image and output image must have the same format");
    }
    // check image format
    m_imageFormat = inImageRequirements->format;
    checkImageFormat(m_imageFormat);

    // check kernel layout
    if (kernelRequirements->layout != nvcv::TENSOR_HW)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Kernel data tensor must be of layout HW");
    }
    if (kernelRequirements->dtype != NVCV_DATA_TYPE_S8)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Kernel data tensor must be of type S8");
    }
    m_kernelHeight = kernelRequirements->shape[0];
    m_kernelWidth  = kernelRequirements->shape[1];

    m_exec    = GetMedianFilterExec(std::make_tuple(m_kernelHeight, m_kernelWidth));
    auto prog = CmdProgram::Create(*m_exec.get());

    // get padMode from borderType
    cupva::PadModeType padMode = (borderType == NVCV_BORDER_REPLICATE) ? PAD_BPE : PAD_CONST;

    // get image line pitch
    int srcLp = m_inImageRequirements.planeRowStride[0] / sizeof(TSRC);
    int dstLp = m_outImageRequirements.planeRowStride[0] / sizeof(T);

    // check tile size: must be multiple of channels per pixel
    if (TILE_W % CH != 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tile size must be multiple of channels per pixel");
    }

    int32_t tileW = TILE_W;
    int32_t tileH = MEDIAN_TILE_H_DF(TILE_H / sizeof(T), CH, m_kernelHeight, m_kernelWidth);

    // tile size in terms of pixels in TSRC
    int32_t kTileInPixelsX = tileW / CH;
    int32_t kTileInPixelsY = tileH;
    // number of tiles
    int32_t nTileX = ((m_imageWidth - 1) / kTileInPixelsX) + 1;
    int32_t nTileY = ((m_imageHeight - 1) / kTileInPixelsY) + 1;

    prog["nTileX"] = nTileX;
    prog["nTileY"] = nTileY;
    TSRC *srcBuf   = prog["srcBuf"].ptr<TSRC>();
    T *dstBuf      = prog["dstBuf"].ptr<T>();

    auto srcHdl = prog["srcHdl"];
    auto dstHdl = prog["dstHdl"];

    int32_t const haloX = m_kernelWidth >> 1;
    int32_t const haloY = m_kernelHeight >> 1;

    RasterDataFlow &srcDataFlow = prog.addDataFlowHead<RasterDataFlow>();
    srcDataFlow.handler(srcHdl)
        .src(m_srcOp, m_imageWidth, m_imageHeight, srcLp)
        .tileBuffer(srcBuf)
        .halo(haloX, haloY, padMode, borderValue)
        .roi(0, 0, std::max(m_imageWidth, kTileInPixelsX), std::max(m_imageHeight, kTileInPixelsY))
        .tile(kTileInPixelsX, kTileInPixelsY);

    RasterDataFlow &dstDataFlow = prog.addDataFlowHead<RasterDataFlow>();
    dstDataFlow.handler(dstHdl)
        .dst(m_dstOp, m_imageWidth * CH, m_imageHeight, dstLp)
        .roi(0, 0, std::max(m_imageWidth * CH, tileW), std::max(m_imageHeight, tileH))
        .tileBuffer(dstBuf)
        .tile(tileW, tileH);

    prog.compileDataFlows();

    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename T, uint8_t CH>
template<typename S>
void MedianFilter<T, CH>::operator()(S stream, const nvcv::Image &in, const nvcv::Image &out,
                                     const nvcv::Tensor &kernel)
{
    auto inImageData      = in.exportData<nvcv::ImageDataStrided>();
    auto outImageData     = out.exportData<nvcv::ImageDataStrided>();
    auto kernelTensorData = kernel.exportData<nvcv::TensorDataStrided>();

    if (!pva_solutions::utils::AreImageRequirementsEqual(m_inImageRequirements, inImageData->cdata()))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Input image tensor must have the same requirements that was used to create the operator");
    }
    if (!pva_solutions::utils::AreImageRequirementsEqual(m_outImageRequirements, outImageData->cdata()))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Output image tensor must have the same requirements that was used to create the operator");
    }
    if ((static_cast<void *>(inImageData->plane(0).basePtr) == nullptr) ||
        (static_cast<void *>(outImageData->plane(0).basePtr) == nullptr) ||
        (static_cast<void *>(kernelTensorData->basePtr()) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Image or tensor base pointers must not be NULL");
    }

    // check kernel data: box filter
    int8_t *kernelPtr_h = NULL;
    CupvaMemGetHostPointer((void **)&kernelPtr_h, (void *)kernelTensorData->basePtr());

    for (int i = 0; i < m_kernelHeight; i++)
    {
        for (int j = 0; j < m_kernelWidth; j++)
        {
            if (kernelPtr_h[i * m_kernelWidth + j] == 0)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Kernel tensor must be a box filter");
            }
        }
    }

    m_srcOp.setBase(static_cast<TSRC *>(static_cast<void *>(inImageData->plane(0).basePtr)));
    m_dstOp.setBase(static_cast<T *>(static_cast<void *>(outImageData->plane(0).basePtr)));

    cupvaError_t err                = CupvaCmdProgramUpdateDataFlows(&m_prog);
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
        static_assert(!sizeof(S), "Unsupported stream type for MedianFilter operator");
    }

    if (!(err == CUPVA_ERROR_NONE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Submit failed");
    }
}

template<typename T, uint8_t CH>
MedianFilter<T, CH>::~MedianFilter()
{
    CupvaCmdDestroy(&m_prog);
}

// Explicit instantiation of the template to support the types we care about
template class MedianFilter<uint8_t, 1>;
template class MedianFilter<int8_t, 1>;
template class MedianFilter<uint16_t, 1>;
template class MedianFilter<int16_t, 1>;
template class MedianFilter<int16_t, 2>;
template class MedianFilter<uint32_t, 1>;
template class MedianFilter<int32_t, 1>;

} // namespace pva_solutions::priv

NVCVStatus pvaMedianFilterCreate(NVCVOperatorHandle *handle, NVCVImageRequirements *inImageRequirements,
                                 NVCVTensorRequirements *kernelRequirements,
                                 NVCVImageRequirements *outImageRequirements, NVCVBorderType borderType,
                                 int32_t borderValue)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        switch (inImageRequirements->format)
        {
        case NVCV_IMAGE_FORMAT_U8:
        case NVCV_IMAGE_FORMAT_Y8:
        case NVCV_IMAGE_FORMAT_Y8_ER:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::MedianFilter<uint8_t, 1>(
                inImageRequirements, kernelRequirements, outImageRequirements, borderType, borderValue));
            break;
        case NVCV_IMAGE_FORMAT_S8:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::MedianFilter<int8_t, 1>(
                inImageRequirements, kernelRequirements, outImageRequirements, borderType, borderValue));
            break;
        case NVCV_IMAGE_FORMAT_U16:
        case NVCV_IMAGE_FORMAT_Y16:
        case NVCV_IMAGE_FORMAT_Y16_ER:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::MedianFilter<uint16_t, 1>(
                inImageRequirements, kernelRequirements, outImageRequirements, borderType, borderValue));
            break;
        case NVCV_IMAGE_FORMAT_S16:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::MedianFilter<int16_t, 1>(
                inImageRequirements, kernelRequirements, outImageRequirements, borderType, borderValue));
            break;
        case NVCV_IMAGE_FORMAT_2S16:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::MedianFilter<int16_t, 2>(
                inImageRequirements, kernelRequirements, outImageRequirements, borderType, borderValue));
            break;
        case NVCV_IMAGE_FORMAT_U32:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::MedianFilter<uint32_t, 1>(
                inImageRequirements, kernelRequirements, outImageRequirements, borderType, borderValue));
            break;
        case NVCV_IMAGE_FORMAT_S32:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::MedianFilter<int32_t, 1>(
                inImageRequirements, kernelRequirements, outImageRequirements, borderType, borderValue));
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported image format!");
        }
    });
}

template<typename StreamType>
NVCVStatus MedianFilterSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVImageHandle in,
                                  NVCVTensorHandle kernel, NVCVImageHandle out)
{
    return nvcv::ProtectCall([&] {
        nvcv::ImageWrapHandle input(in), output(out);
        nvcv::TensorWrapHandle kernel_t(kernel);

        NVCVImageFormat inFormat;
        nvcvImageGetFormat(in, &inFormat);

        switch (inFormat)
        {
        case NVCV_IMAGE_FORMAT_U8:
        case NVCV_IMAGE_FORMAT_Y8:
        case NVCV_IMAGE_FORMAT_Y8_ER:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::MedianFilter<uint8_t, 1>>(handle)(stream, input,
                                                                                                     output, kernel_t);
            break;
        case NVCV_IMAGE_FORMAT_S8:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::MedianFilter<int8_t, 1>>(handle)(stream, input,
                                                                                                    output, kernel_t);
            break;
        case NVCV_IMAGE_FORMAT_U16:
        case NVCV_IMAGE_FORMAT_Y16:
        case NVCV_IMAGE_FORMAT_Y16_ER:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::MedianFilter<uint16_t, 1>>(handle)(stream, input,
                                                                                                      output, kernel_t);
            break;
        case NVCV_IMAGE_FORMAT_S16:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::MedianFilter<int16_t, 1>>(handle)(stream, input,
                                                                                                     output, kernel_t);
            break;
        case NVCV_IMAGE_FORMAT_2S16:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::MedianFilter<int16_t, 2>>(handle)(stream, input,
                                                                                                     output, kernel_t);
            break;
        case NVCV_IMAGE_FORMAT_U32:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::MedianFilter<uint32_t, 1>>(handle)(stream, input,
                                                                                                      output, kernel_t);
            break;
        case NVCV_IMAGE_FORMAT_S32:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::MedianFilter<int32_t, 1>>(handle)(stream, input,
                                                                                                     output, kernel_t);
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported image format!");
        }
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus MedianFilterSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVImageHandle,
                                                          NVCVTensorHandle, NVCVImageHandle);
template NVCVStatus MedianFilterSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVImageHandle,
                                                         NVCVTensorHandle, NVCVImageHandle);

// C++ overload implementations
NVCVStatus pvaMedianFilterSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVImageHandle in,
                                 NVCVTensorHandle kernel, NVCVImageHandle out)
{
    return MedianFilterSubmitImpl(handle, stream, in, kernel, out);
}

NVCVStatus pvaMedianFilterSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageHandle in,
                                 NVCVTensorHandle kernel, NVCVImageHandle out)
{
    return MedianFilterSubmitImpl(handle, stream, in, kernel, out);
}
