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

#include "OpImageFlip.hpp"

#include "device/image_flip_param.h"

#include <IOperator.hpp>
#include <ImageUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <cupva_host_utils.hpp>
#include <nvcv/Image.hpp>

#include <OpImageFlip.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <cstdint>
#include <limits>
#include <set>

using namespace cupva;

// Declare all executables for runtime selection
CUPVA_DECLARE_EXECUTABLE(pva_image_flip_horizontal_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_flip_vertical_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_flip_both_dev);

namespace pva_solutions::priv {

// Define supported image formats for ImageFlip
static std::set<NVCVImageFormat> formatSet = {NVCV_IMAGE_FORMAT_U8, NVCV_IMAGE_FORMAT_Y8, NVCV_IMAGE_FORMAT_Y8_ER};

template<typename T>
void ImageFlip<T>::checkImageFormat(NVCVImageFormat format)
{
    auto formatIt = formatSet.find(format);
    if (formatIt == formatSet.end())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported image format");
    }
}

template<typename T>
std::mutex ImageFlip<T>::s_execMapMutex;

template<typename T>
std::map<typename ImageFlip<T>::ExecKey, std::weak_ptr<cupva::Executable>> ImageFlip<T>::s_ImageFlipExecutables;

template<typename T>
std::shared_ptr<cupva::Executable> ImageFlip<T>::GetImageFlipExec(PVAFlipDirection flipDirection)
{
    static IOperator::ExecutableData sImageFlipHorizontalExecData = {
        CUPVA_EXECUTABLE_DATA(pva_image_flip_horizontal_dev), CUPVA_EXECUTABLE_SIZE(pva_image_flip_horizontal_dev)};

    static IOperator::ExecutableData sImageFlipVerticalExecData = {CUPVA_EXECUTABLE_DATA(pva_image_flip_vertical_dev),
                                                                   CUPVA_EXECUTABLE_SIZE(pva_image_flip_vertical_dev)};

    static IOperator::ExecutableData sImageFlipBothExecData = {CUPVA_EXECUTABLE_DATA(pva_image_flip_both_dev),
                                                               CUPVA_EXECUTABLE_SIZE(pva_image_flip_both_dev)};

    // Select executable data based on flip direction
    IOperator::ExecutableData *execData = nullptr;
    switch (flipDirection)
    {
    case PVA_FLIP_HORIZONTAL:
        execData = &sImageFlipHorizontalExecData;
        break;
    case PVA_FLIP_VERTICAL:
        execData = &sImageFlipVerticalExecData;
        break;
    case PVA_FLIP_BOTH:
        execData = &sImageFlipBothExecData;
        break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid flip direction");
    }

    // Create execKey combining flip direction and context
    ExecKey execKey = std::make_tuple(cupva::Context::GetCurrent(), static_cast<int>(flipDirection));
    return IOperator::GetExecutable<ExecKey>(*execData, execKey, s_ImageFlipExecutables, s_execMapMutex);
}

template<typename T>
ImageFlip<T>::ImageFlip(NVCVImageRequirements *imageRequirements)
    : m_imageRequirements(*imageRequirements)
    , m_srcOp(cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr))
    , m_dstOp(cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr))
{
    // Extract image dimensions from image requirements
    m_imageWidth  = static_cast<int32_t>(m_imageRequirements.width);
    m_imageHeight = static_cast<int32_t>(m_imageRequirements.height);
    // Use the actual line pitch from image requirements to account for row alignment
    m_inputLinePitch = static_cast<int32_t>(m_imageRequirements.planeRowStride[0]);

    // Validate image format
    NVCVImageFormat format = m_imageRequirements.format;
    checkImageFormat(format);

    // Get number of channels from image format (ImageFlip only supports grayscale)
    int32_t channels = 1; // Default for grayscale formats

    const int32_t minImageWidth  = 256; // Must match VPU tile width for memory and processing constraints
    const int32_t minImageHeight = 128; // Must match VPU tile height for memory and processing constraints
    const int32_t vectorWidth    = 32;  // Vector processing width - image width must be divisible by this

    if (((m_imageWidth < minImageWidth)) || ((m_imageHeight < minImageHeight)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Image width and/or image height must be at least as large as %d x %d", minImageWidth,
                              minImageHeight);
    }

    // Enforce vector width alignment constraint
    if (m_imageWidth % vectorWidth != 0)
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Image width (%d) must be divisible by vector width (%d) for proper horizontal flip processing",
            m_imageWidth, vectorWidth);
    }

    // Temporarily enforce zero row alignment constraint until device kernel is fixed for row alignment support
    if (m_inputLinePitch != m_imageWidth)
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Row alignment not currently supported - image line pitch (%d) must equal image width (%d)",
            m_inputLinePitch, m_imageWidth);
    }

    if (!(channels == 1))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid Channel Count - only grayscale images are supported");
    }

    // Create persistent programs for all flip directions
    m_execHorizontal = GetImageFlipExec(PVA_FLIP_HORIZONTAL);
    m_execVertical   = GetImageFlipExec(PVA_FLIP_VERTICAL);
    m_execCombined   = GetImageFlipExec(PVA_FLIP_BOTH);

    // Create and compile programs for horizontal flip
    {
        auto prog = CmdProgram::Create(*m_execHorizontal.get());

        uint8_t *input_v  = prog["input_v"].ptr<uint8_t>();
        uint8_t *output_v = prog["output_v"].ptr<uint8_t>();
        auto srcHdl       = prog["srcHdl"];
        auto dstHdl       = prog["dstHdl"];

        setupHorizontalFlipDataFlows(prog, input_v, output_v, srcHdl, dstHdl);
        prog.compileDataFlows();
        cupva::ConvertTo(std::move(prog), m_progHorizontal);
    }

    // Create and compile programs for vertical flip
    {
        auto prog = CmdProgram::Create(*m_execVertical.get());

        uint8_t *input_v  = prog["input_v"].ptr<uint8_t>();
        uint8_t *output_v = prog["output_v"].ptr<uint8_t>();
        auto srcHdl1      = prog["srcHdl1"];
        auto dstHdl1      = prog["dstHdl1"];

        setupVerticalFlipDataFlows(prog, input_v, output_v, srcHdl1, dstHdl1);
        prog.compileDataFlows();
        cupva::ConvertTo(std::move(prog), m_progVertical);
    }

    // Create and compile programs for combined flip
    {
        auto prog = CmdProgram::Create(*m_execCombined.get());

        uint8_t *input_v  = prog["input_v"].ptr<uint8_t>();
        uint8_t *output_v = prog["output_v"].ptr<uint8_t>();
        auto srcHdl2      = prog["srcHdl2"];
        auto dstHdl2      = prog["dstHdl2"];

        setupCombinedFlipDataFlows(prog, input_v, output_v, srcHdl2, dstHdl2);
        prog.compileDataFlows();
        cupva::ConvertTo(std::move(prog), m_progCombined);
    }
}

template<typename T>
template<typename TriggerType>
void ImageFlip<T>::setupHorizontalFlipDataFlows(cupva::CmdProgram &prog, uint8_t *input_v, uint8_t *output_v,
                                                TriggerType &srcHdl, TriggerType &dstHdl)
{
    // Use effective tile sizes that don't exceed image dimensions
    int32_t effectiveTW = std::min(TW, m_imageWidth);  // Don't exceed image width
    int32_t effectiveTH = std::min(TH, m_imageHeight); // Don't exceed image height

    // ROI should match actual image size, not tile size
    int32_t roiWidth      = m_imageWidth;
    int32_t roiHeight     = m_imageHeight;
    int32_t dstTileWidth  = effectiveTW;
    int32_t dstTileHeight = effectiveTH;
    int32_t niterW        = (m_imageWidth + effectiveTW - 1) / effectiveTW;
    int32_t niterH        = (m_imageHeight + effectiveTH - 1) / effectiveTH;

    prog["niterW"]    = niterW;
    prog["niterH"]    = niterH;
    prog["IMW_param"] = m_imageWidth;
    prog["IMH_param"] = m_imageHeight;
    prog["TW_param"]  = dstTileWidth;
    prog["TH_param"]  = dstTileHeight;

    RasterDataFlow &sourceDataFlow = prog.addDataFlowHead<RasterDataFlow>();
    sourceDataFlow.handler(srcHdl)
        .src(m_srcOp, m_imageWidth, m_imageHeight, m_inputLinePitch)
        .roi(0, 0, roiWidth, roiHeight)
        .tileBuffer(input_v)
        .scanOrder(cupva::HORIZONTAL_REVERSED)
        .tile(effectiveTW, effectiveTH); // Use effective tile sizes

    RasterDataFlow &destinationDataFlow = prog.addDataFlowHead<RasterDataFlow>();
    destinationDataFlow.handler(dstHdl)
        .tileBuffer(output_v)
        .tile(dstTileWidth, dstTileHeight)
        .tileArena(effectiveTW, effectiveTH) // Use effective tile sizes for arena
        .dst(m_dstOp, m_imageWidth, m_imageHeight, m_inputLinePitch);
}

template<typename T>
template<typename TriggerType>
void ImageFlip<T>::setupVerticalFlipDataFlows(cupva::CmdProgram &prog, uint8_t *input_v, uint8_t *output_v,
                                              TriggerType &srcHdl, TriggerType &dstHdl)
{

    // Use effective tile sizes that don't exceed image dimensions
    int32_t effectiveTW = std::min(TW, m_imageWidth);  // Don't exceed image width
    int32_t effectiveTH = std::min(TH, m_imageHeight); // Don't exceed image height

    // ROI should match actual image size, not tile size
    int32_t roiWidth      = m_imageWidth;
    int32_t roiHeight     = m_imageHeight;
    int32_t dstTileWidth  = effectiveTW;
    int32_t dstTileHeight = effectiveTH;
    int32_t niterW        = (m_imageWidth + effectiveTW - 1) / effectiveTW;
    int32_t niterH        = (m_imageHeight + effectiveTH - 1) / effectiveTH;

    prog["niterW"]    = niterW;
    prog["niterH"]    = niterH;
    prog["IMW_param"] = m_imageWidth;
    prog["IMH_param"] = m_imageHeight;
    prog["TW_param"]  = dstTileWidth;
    prog["TH_param"]  = dstTileHeight;

    RasterDataFlow &sourceDataFlow = prog.addDataFlowHead<RasterDataFlow>();
    sourceDataFlow.handler(srcHdl)
        .src(m_srcOp, m_imageWidth, m_imageHeight, m_inputLinePitch)
        .roi(0, 0, roiWidth, roiHeight)
        .tileBuffer(input_v)
        .scanOrder(cupva::VERTICAL_REVERSED)
        .tile(effectiveTW, effectiveTH); // Use effective tile sizes

    RasterDataFlow &destinationDataFlow = prog.addDataFlowHead<RasterDataFlow>();
    destinationDataFlow.handler(dstHdl)
        .tileBuffer(output_v)
        .tile(dstTileWidth, dstTileHeight)
        .tileArena(effectiveTW, effectiveTH) // Use effective tile sizes for arena
        .dst(m_dstOp, m_imageWidth, m_imageHeight, m_inputLinePitch);
}

template<typename T>
template<typename TriggerType>
void ImageFlip<T>::setupCombinedFlipDataFlows(cupva::CmdProgram &prog, uint8_t *input_v, uint8_t *output_v,
                                              TriggerType &srcHdl2, TriggerType &dstHdl2)
{
    // Use effective tile sizes that don't exceed image dimensions
    int32_t effectiveTW = std::min(TW, m_imageWidth);  // Don't exceed image width
    int32_t effectiveTH = std::min(TH, m_imageHeight); // Don't exceed image height

    // ROI should match actual image size, not tile size
    int32_t roiWidth      = m_imageWidth;
    int32_t roiHeight     = m_imageHeight;
    int32_t dstTileWidth  = effectiveTW;
    int32_t dstTileHeight = effectiveTH;
    int32_t niterW        = (m_imageWidth + effectiveTW - 1) / effectiveTW;
    int32_t niterH        = (m_imageHeight + effectiveTH - 1) / effectiveTH;

    prog["niterW"]    = niterW;
    prog["niterH"]    = niterH;
    prog["IMW_param"] = m_imageWidth;
    prog["IMH_param"] = m_imageHeight;
    prog["TW_param"]  = dstTileWidth;
    prog["TH_param"]  = dstTileHeight;

    RasterDataFlow &sourceDataFlow = prog.addDataFlowHead<RasterDataFlow>();
    sourceDataFlow.handler(srcHdl2)
        .src(m_srcOp, m_imageWidth, m_imageHeight, m_inputLinePitch)
        .roi(0, 0, roiWidth, roiHeight)
        .tileBuffer(input_v)
        .scanOrder(cupva::HORIZONTAL_REVERSED | cupva::VERTICAL_REVERSED)
        .tile(effectiveTW, effectiveTH); // Use effective tile sizes

    RasterDataFlow &destinationDataFlow = prog.addDataFlowHead<RasterDataFlow>();
    destinationDataFlow.handler(dstHdl2)
        .tileBuffer(output_v)
        .tile(dstTileWidth, dstTileHeight)
        .tileArena(effectiveTW, effectiveTH) // Use effective tile sizes for arena
        .dst(m_dstOp, m_imageWidth, m_imageHeight, m_inputLinePitch);
}

template<typename T>
template<typename S>
void ImageFlip<T>::operator()(S stream, const nvcv::Image &in, const nvcv::Image &out, int32_t flipDirection)
{
    // Validate flip direction parameter
    if (flipDirection < PVA_FLIP_HORIZONTAL || flipDirection > PVA_FLIP_BOTH)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid flip direction parameter");
    }

    auto inImageData  = in.exportData<nvcv::ImageDataStrided>();
    auto outImageData = out.exportData<nvcv::ImageDataStrided>();

    // Validate input image requirements match what was used to create the operator
    if (!pva_solutions::utils::AreImageRequirementsEqual(m_imageRequirements, inImageData->cdata()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input image must have the same requirements that was used to create the operator");
    }

    // Validate output image dimensions match input (ImageFlip produces same size output)
    if (!pva_solutions::utils::AreImageRequirementsEqual(m_imageRequirements, outImageData->cdata()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output image must have the same requirements as input image");
    }

    if (static_cast<void *>(inImageData->plane(0).basePtr) == nullptr ||
        static_cast<void *>(outImageData->plane(0).basePtr) == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Image base pointers must not be NULL");
    }

    m_srcOp.setBase(static_cast<T *>(static_cast<void *>(inImageData->plane(0).basePtr)));
    m_dstOp.setBase(static_cast<T *>(static_cast<void *>(outImageData->plane(0).basePtr)));

    // Select the appropriate persistent program based on flip direction
    cupvaCmd_t *selectedProgram = nullptr;
    switch (static_cast<PVAFlipDirection>(flipDirection))
    {
    case PVA_FLIP_HORIZONTAL:
        selectedProgram = &m_progHorizontal;
        break;
    case PVA_FLIP_VERTICAL:
        selectedProgram = &m_progVertical;
        break;
    case PVA_FLIP_BOTH:
        selectedProgram = &m_progCombined;
        break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid flip direction parameter");
    }

    // Update DataFlow pointers and submit the persistent program
    cupvaError_t err = CupvaCmdProgramUpdateDataFlows(selectedProgram);

    if (!(err == CUPVA_ERROR_NONE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Program Update DataFlows failed");
    }

    cupvaCmd_t const *submitCmds[1] = {selectedProgram};
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
        static_assert(!sizeof(S), "Unsupported stream type for ImageFlip operator");
    }

    if (!(err == CUPVA_ERROR_NONE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Submit failed");
    }
}

template<typename T>
ImageFlip<T>::~ImageFlip()
{
    // Clean up persistent command programs
    CupvaCmdDestroy(&m_progHorizontal);
    CupvaCmdDestroy(&m_progVertical);
    CupvaCmdDestroy(&m_progCombined);
}

// Explicit instantiation of the template to support the types we care about
template class ImageFlip<uint8_t>;

} // namespace pva_solutions::priv

NVCVStatus pvaImageFlipCreate(NVCVOperatorHandle *handle, NVCVImageRequirements *imageRequirements)
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
                                  "Pointer to Input or output ImageRequirements must not be NULL");
        }

        // For U8, Y8 and Y8_ER formats, we use uint8_t data type
        NVCVImageFormat format = imageRequirements->format;
        if (format == NVCV_IMAGE_FORMAT_U8 || format == NVCV_IMAGE_FORMAT_Y8 || format == NVCV_IMAGE_FORMAT_Y8_ER)
        {
            *handle =
                reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::ImageFlip<uint8_t>(imageRequirements));
        }
        else
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported image format");
        }
    });
}

template<typename StreamType>
NVCVStatus ImageFlipSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVImageHandle in, NVCVImageHandle out,
                               PVAFlipDirection flipDirection)
{
    return nvcv::ProtectCall([&] {
        // Validate flip direction parameter
        if (flipDirection < PVA_FLIP_HORIZONTAL || flipDirection > PVA_FLIP_BOTH)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid flip direction parameter");
        }

        // Wrap image handles for processing
        nvcv::ImageWrapHandle input(in);
        nvcv::ImageWrapHandle output(out);

        NVCVImageFormat inFormat;
        nvcvImageGetFormat(in, &inFormat);

        // For U8, Y8 and Y8_ER formats, we use uint8_t data type
        if (inFormat == NVCV_IMAGE_FORMAT_U8 || inFormat == NVCV_IMAGE_FORMAT_Y8 || inFormat == NVCV_IMAGE_FORMAT_Y8_ER)
        {
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::ImageFlip<uint8_t>>(handle)(
                stream, input, output, static_cast<int32_t>(flipDirection));
        }
        else
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported image format");
        }
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus ImageFlipSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVImageHandle,
                                                       NVCVImageHandle, PVAFlipDirection);
template NVCVStatus ImageFlipSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVImageHandle,
                                                      NVCVImageHandle, PVAFlipDirection);

// C++ overload implementations
NVCVStatus pvaImageFlipSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVImageHandle in, NVCVImageHandle out,
                              PVAFlipDirection flipDirection)
{
    return ImageFlipSubmitImpl(handle, stream, in, out, flipDirection);
}

NVCVStatus pvaImageFlipSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageHandle in, NVCVImageHandle out,
                              PVAFlipDirection flipDirection)
{
    return ImageFlipSubmitImpl(handle, stream, in, out, flipDirection);
}
