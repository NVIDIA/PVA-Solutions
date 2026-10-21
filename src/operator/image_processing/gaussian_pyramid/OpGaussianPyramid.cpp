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

#include "OpGaussianPyramid.hpp"

#include <IOperator.hpp>
#include <ImageUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <nvcv/Image.hpp>

#include <ErrorCheckMacros.h>
#include <OpGaussianPyramid.h>
#include <PvaAllocator.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>
#include <nvcv/Fwd.h>
#include <nvcv/alloc/Allocator.h>

#include <climits>
#include <cmath>
#include <cstring>
#include <limits>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_gaussian_pyramid_u8_dev);
CUPVA_DECLARE_EXECUTABLE(pva_gaussian_pyramid_s8_dev);
CUPVA_DECLARE_EXECUTABLE(pva_gaussian_pyramid_u16_dev);
CUPVA_DECLARE_EXECUTABLE(pva_gaussian_pyramid_s16_dev);

namespace pva_solutions::priv {

constexpr int32_t MIN_PENULTIMATE_EVEN_LEVEL_HEIGHT = 18;
constexpr int32_t MIN_LEVELS                        = 2;
constexpr int32_t MAX_LEVELS                        = 3;
constexpr int32_t MIN_IMAGE_WIDTH                   = 2;
constexpr int32_t MIN_IMAGE_HEIGHT                  = 4;

namespace {

template<typename T>
int32_t GetImageLinePitchElems(const nvcv::ImageDataStrided &data)
{
    const int64_t rowStrideBytes = data.plane(0).rowStride;
    if ((rowStrideBytes % static_cast<int64_t>(sizeof(T))) != 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Image row stride must be divisible by element size");
    }
    const int64_t linePitchElems = rowStrideBytes / static_cast<int64_t>(sizeof(T));
    if (linePitchElems > std::numeric_limits<int32_t>::max())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Image row stride is too large");
    }
    return static_cast<int32_t>(linePitchElems);
}

template<typename T>
int32_t ValidateOutputImageAndGetPitch(const nvcv::ImageDataStrided &data, const NVCVImageRequirements &req)
{
    if ((req.format != data.format()) || (req.width != data.plane(0).width) || (req.height != data.plane(0).height))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output image must match the expected format and dimensions");
    }

    int32_t linePitchElems = GetImageLinePitchElems<T>(data);
    if (linePitchElems < req.width)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output image line pitch must be at least the image width");
    }

    return linePitchElems;
}

} // namespace

template<typename T>
std::mutex GaussianPyramid<T>::s_execMapMutex;

template<typename T>
std::map<typename GaussianPyramid<T>::ExecKey, std::weak_ptr<cupva::Executable>>
    GaussianPyramid<T>::s_gaussianPyramidExecutables;

template<typename T>
std::shared_ptr<cupva::Executable> GaussianPyramid<T>::GetGaussianPyramidExec()
{
    static IOperator::ExecutableData sGaussianPyramidExecData;

    if constexpr (std::is_same<T, uint8_t>::value)
    {
        sGaussianPyramidExecData = {CUPVA_EXECUTABLE_DATA(pva_gaussian_pyramid_u8_dev),
                                    CUPVA_EXECUTABLE_SIZE(pva_gaussian_pyramid_u8_dev)};
    }
    else if constexpr (std::is_same<T, int8_t>::value)
    {
        sGaussianPyramidExecData = {CUPVA_EXECUTABLE_DATA(pva_gaussian_pyramid_s8_dev),
                                    CUPVA_EXECUTABLE_SIZE(pva_gaussian_pyramid_s8_dev)};
    }
    else if constexpr (std::is_same<T, uint16_t>::value)
    {
        sGaussianPyramidExecData = {CUPVA_EXECUTABLE_DATA(pva_gaussian_pyramid_u16_dev),
                                    CUPVA_EXECUTABLE_SIZE(pva_gaussian_pyramid_u16_dev)};
    }
    else if constexpr (std::is_same<T, int16_t>::value)
    {
        sGaussianPyramidExecData = {CUPVA_EXECUTABLE_DATA(pva_gaussian_pyramid_s16_dev),
                                    CUPVA_EXECUTABLE_SIZE(pva_gaussian_pyramid_s16_dev)};
    }

    ExecKey execKey = std::make_tuple(cupva::Context::GetCurrent());
    return IOperator::GetExecutable<ExecKey>(sGaussianPyramidExecData, execKey, s_gaussianPyramidExecutables,
                                             s_execMapMutex);
}

template<typename T>
GaussianPyramid<T>::GaussianPyramid(NVCVImageRequirements const *const inputImageRequirements, const int32_t numLevels,
                                    const NVCVBorderType borderType, const int32_t borderValue)
    : m_totalLevels{numLevels}
    , m_inputImageRequirements{*inputImageRequirements}
    , m_srcOp(cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr))
    , m_prog{}
    , m_progInitialized{false}
{
    int32_t imageWidth  = inputImageRequirements->width;
    int32_t imageHeight = inputImageRequirements->height;

    NVCVImageFormat imageFormat = inputImageRequirements->format;

    if (borderType != NVCV_BORDER_CONSTANT)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported border type");
    }

    if (borderValue != 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Border value must be 0, got %d", borderValue);
    }

    int32_t imagePlaneCount;
    nvcvImageFormatGetNumPlanes(imageFormat, &imagePlaneCount);
    if (imagePlaneCount != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Pyramid format must have only one plane");
    }

    switch (imageFormat)
    {
    case NVCV_IMAGE_FORMAT_U8:
    case NVCV_IMAGE_FORMAT_S8:
    case NVCV_IMAGE_FORMAT_U16:
    case NVCV_IMAGE_FORMAT_S16:
        break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported image format");
    }

    if (m_totalLevels < MIN_LEVELS)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Pyramid must have at least %d levels", MIN_LEVELS);
    }

    if (m_totalLevels > MAX_LEVELS)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Pyramid must have at most %d levels", MAX_LEVELS);
    }

    if (imageHeight % TH0 == 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Image height mod %d cannot be 1", TH0);
    }

    if ((imageWidth < MIN_IMAGE_WIDTH) || (imageHeight < MIN_IMAGE_HEIGHT))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Minimum input image size is [%dx%d], [%dx%d] is too small", MIN_IMAGE_WIDTH,
                              MIN_IMAGE_HEIGHT, imageWidth, imageHeight);
    }

    int32_t outputImageWidth  = imageWidth;
    int32_t outputImageHeight = imageHeight;
    for (int32_t i = 0; i < m_totalLevels; ++i)
    {
        NVCVImageRequirements outputImageRequirements;
        nvcvImageCalcRequirementsPva(outputImageWidth, outputImageHeight, imageFormat, 0, 0, &outputImageRequirements);

        m_outputImagesRequirements.push_back(outputImageRequirements);
        m_dstOps.push_back(cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr));

        outputImageWidth  = (outputImageWidth + 1) / 2;
        outputImageHeight = (outputImageHeight + 1) / 2;
    }

    m_exec = GetGaussianPyramidExec();
}

template<typename T>
template<typename S>
void GaussianPyramid<T>::operator()(S stream, NVCVImageHandle inHandle, NVCVImageHandle *outHandles)
{
    nvcv::Image input          = nvcv::ImageWrapHandle(inHandle);
    auto inputImageDataStrided = input.exportData<nvcv::ImageDataStrided>();
    if (!inputImageDataStrided)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input image data must not be empty");
    }
    const auto &inputImageDataView = inputImageDataStrided.value();

    if (static_cast<void *>(inputImageDataView.plane(0).basePtr) == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Image base pointers must not be NULL");
    }

    int32_t inputLinePitch = GetImageLinePitchElems<T>(inputImageDataView);
    m_srcOp.setBase(static_cast<T *>(static_cast<void *>(inputImageDataView.plane(0).basePtr)));

    std::array<int32_t, MAX_LEVELS> outputLinePitches = {};
    for (int32_t i = 0; i < m_totalLevels; i++)
    {
        nvcv::Image output          = nvcv::ImageWrapHandle(outHandles[i]);
        auto outputImageDataStrided = output.exportData<nvcv::ImageDataStrided>();
        if (!outputImageDataStrided)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output image data must not be empty");
        }
        const auto &outputImageDataView = outputImageDataStrided.value();
        if (static_cast<void *>(outputImageDataView.plane(0).basePtr) == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Image base pointers must not be NULL");
        }
        outputLinePitches[i] = ValidateOutputImageAndGetPitch<T>(outputImageDataView, m_outputImagesRequirements[i]);
        m_dstOps[i].setBase(static_cast<T *>(static_cast<void *>(outputImageDataView.plane(0).basePtr)));
    }

    if (m_progInitialized)
    {
        CUPVA_CHECK_ERROR(CupvaCmdDestroy(&m_prog));
        m_progInitialized = false;
    }

    int32_t imageWidth  = m_inputImageRequirements.width;
    int32_t imageHeight = m_inputImageRequirements.height;

    auto prog = CmdProgram::Create(*m_exec.get());

    prog["imgWidth"]    = imageWidth;
    prog["imgHeight"]   = imageHeight;
    prog["totalLevels"] = m_totalLevels;
    prog["padValue"]    = 0;

    uint16_t gaussianKernel[] = {1, 4,  6,  4,  1, 0, 4, 16, 24, 16, 4, 0, 6, 24, 36, 24, 6, 0,
                                 4, 16, 24, 16, 4, 0, 1, 4,  6,  4,  1, 0, 0, 0,  0,  0,  0, 0};
    prog["gaussianKernel"].set(gaussianKernel, (KW + 1) * (KW + 1));

    T *level0EvenBuffer = prog["level0EvenBuffer"].ptr<T>();
    T *level0OddBuffer  = prog["level0OddBuffer"].ptr<T>();
    T *level1EvenBuffer = prog["level1EvenBuffer"].ptr<T>();
    T *level1OddBuffer  = prog["level1OddBuffer"].ptr<T>();
    T *level2Buffer     = prog["level2Buffer"].ptr<T>();

    bool hasLevel2 = (m_totalLevels > 2);

    int32_t level0ImageWidth  = imageWidth;
    int32_t level0ImageHeight = imageHeight;
    int32_t level1ImageWidth  = (level0ImageWidth + 1) / 2;
    int32_t level1ImageHeight = (level0ImageHeight + 1) / 2;
    int32_t level2ImageWidth  = (level1ImageWidth + 1) / 2;
    int32_t level2ImageHeight = (level1ImageHeight + 1) / 2;

    int32_t level0OddImageHeight  = level0ImageHeight >> 1;
    int32_t level0EvenImageHeight = level0ImageHeight - level0OddImageHeight;
    int32_t level1OddImageHeight  = level1ImageHeight >> 1;
    int32_t level1EvenImageHeight = level1ImageHeight - level1OddImageHeight;

    int32_t level2LinePitchValue = hasLevel2 ? outputLinePitches[2] : 0;

    int32_t level0LinePitch       = outputLinePitches[0];
    int32_t level1LinePitch       = outputLinePitches[1];
    int32_t level0DoubleLinePitch = level0LinePitch << 1;
    int32_t level1DoubleLinePitch = level1LinePitch << 1;

    auto level0ImagePtr = m_dstOps[0];
    auto level1ImagePtr = m_dstOps[1];

    int32_t roiWidth             = TW0 > level0ImageWidth ? TW0 : level0ImageWidth;
    int32_t roiHeightEven        = TH0_SPLIT_BUF > level0EvenImageHeight ? TH0_SPLIT_BUF : level0EvenImageHeight;
    int32_t roiHeightOdd         = TH0_SPLIT_BUF > level0OddImageHeight ? TH0_SPLIT_BUF : level0OddImageHeight;
    int32_t level1TileWidth      = TW1 < level1ImageWidth ? TW1 : level1ImageWidth;
    int32_t level1TileHeightEven = TH1_SPLIT_BUF < level1EvenImageHeight ? TH1_SPLIT_BUF : level1EvenImageHeight;
    int32_t level1TileHeightOdd  = TH1_SPLIT_BUF < level1OddImageHeight ? TH1_SPLIT_BUF : level1OddImageHeight;
    int32_t level2TileWidth      = TW2 < level2ImageWidth ? TW2 : level2ImageWidth;
    int32_t level2TileHeight     = TH2 < level2ImageHeight ? TH2 : level2ImageHeight;

    uint32_t niterY = ((level0ImageHeight + TH0 - 1) / TH0);
    uint32_t niterX = ((level0ImageWidth + TW0 - 1) / TW0);
    prog["niterY"]  = niterY;
    prog["niterX"]  = niterX;

    if (m_totalLevels == 2)
    {
        // level0 DataFlows
        RasterDataFlow &level0EvenDataFlow = prog.addDataFlowHead<RasterDataFlow>();
        level0EvenDataFlow.handler(prog["level0HandlerEven"])
            .src(level0ImagePtr, level0ImageWidth, level0EvenImageHeight, level0DoubleLinePitch)
            .tileBuffer(level0EvenBuffer)
            .tile(TW0, TH0_SPLIT_BUF)
            .halo(HALO_L1, HALO_L1_Y_SPLIT_BUF)
            .roi(0, 0, roiWidth, roiHeightEven);

        RasterDataFlow &level0OddDataFlow = prog.addDataFlowHead<RasterDataFlow>();
        level0OddDataFlow.handler(prog["level0HandlerOdd"])
            .src(level0ImagePtr + level0LinePitch, level0ImageWidth, level0OddImageHeight, level0DoubleLinePitch)
            .tileBuffer(level0OddBuffer)
            .tile(TW0, TH0_SPLIT_BUF)
            .halo(HALO_L1, HALO_L1_Y_SPLIT_BUF)
            .roi(0, 0, roiWidth, roiHeightOdd);

        // level1 DataFlows
        RasterDataFlow &level1EvenDataFlow = prog.addDataFlowHead<RasterDataFlow>();
        level1EvenDataFlow.handler(prog["level1Handler"])
            .tileBuffer(level1EvenBuffer)
            .dst(level1ImagePtr, level1ImageWidth, level1EvenImageHeight, level1DoubleLinePitch)
            .tile(level1TileWidth, level1TileHeightEven)
            .tileArena(L1_TILE_ARENA_X, L1_TILE_ARENA_Y);

        RasterDataFlow &level1OddDataFlow = prog.addDataFlowHead<RasterDataFlow>();
        level1OddDataFlow.handler(prog["level1Handler"])
            .dst(level1ImagePtr + level1LinePitch, level1ImageWidth, level1OddImageHeight, level1DoubleLinePitch)
            .tileBuffer(level1OddBuffer)
            .tile(level1TileWidth, level1TileHeightOdd)
            .tileArena(L1_TILE_ARENA_X, L1_TILE_ARENA_Y);
    }
    else if (m_totalLevels == 3)
    {
        auto level2ImagePtr = m_dstOps[2];
        // level0 DataFlows
        RasterDataFlow &level0EvenDataFlow = prog.addDataFlowHead<RasterDataFlow>();
        level0EvenDataFlow.handler(prog["level0HandlerEven"])
            .src(level0ImagePtr, level0ImageWidth, level0EvenImageHeight, level0DoubleLinePitch)
            .tileBuffer(level0EvenBuffer)
            .tile(TW0, TH0_SPLIT_BUF)
            .halo(HALO_L0, HALO_L0_Y_SPLIT_BUF)
            .roi(0, 0, roiWidth, roiHeightEven);

        RasterDataFlow &level0OddDataFlow = prog.addDataFlowHead<RasterDataFlow>();
        level0OddDataFlow.handler(prog["level0HandlerOdd"])
            .src(level0ImagePtr + level0LinePitch, level0ImageWidth, level0OddImageHeight, level0DoubleLinePitch)
            .tileBuffer(level0OddBuffer)
            .tile(TW0, TH0_SPLIT_BUF)
            .halo(HALO_L0, HALO_L0_Y_SPLIT_BUF)
            .roi(0, 0, roiWidth, roiHeightOdd);

        // level1 DataFlows
        RasterDataFlow &level1EvenDataFlow = prog.addDataFlowHead<RasterDataFlow>();
        level1EvenDataFlow.handler(prog["level1Handler"])
            .tileBuffer(level1EvenBuffer + L1_TILE_ADJUSTMENT_MAX)
            .dst(level1ImagePtr, level1ImageWidth, level1EvenImageHeight, level1DoubleLinePitch)
            .tile(level1TileWidth, level1TileHeightEven)
            .tileArena(L1_TILE_ARENA_X, L1_TILE_ARENA_Y);

        RasterDataFlow &level1OddDataFlow = prog.addDataFlowHead<RasterDataFlow>();
        level1OddDataFlow.handler(prog["level1Handler"])
            .dst(level1ImagePtr + level1LinePitch, level1ImageWidth, level1OddImageHeight, level1DoubleLinePitch)
            .tileBuffer(level1OddBuffer + L1_TILE_ADJUSTMENT_MAX)
            .tile(level1TileWidth, level1TileHeightOdd)
            .tileArena(L1_TILE_ARENA_X, L1_TILE_ARENA_Y);

        // level2 DataFlows
        RasterDataFlow &level2DataFlow = prog.addDataFlowHead<RasterDataFlow>();

        level2DataFlow.handler(prog["level2Handler"])
            .dst(level2ImagePtr, level2ImageWidth, level2ImageHeight, level2LinePitchValue)
            .tileBuffer(level2Buffer)
            .tile(level2TileWidth, level2TileHeight)
            .tileArena(L2_TILE_ARENA_X, L2_TILE_ARENA_Y);
    }

    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
    m_progInitialized = true;

    // Copy the input image to the output pyramid first level.
    NVCVImageData inputImageData;
    nvcvImageExportData(inHandle, &inputImageData);
    T *inputImagePtr = NULL;
    CupvaMemGetHostPointer((void **)&inputImagePtr, (void *)inputImageData.buffer.strided.planes[0].basePtr);

    NVCVImageData outputImageLevel0Data;
    nvcvImageExportData(outHandles[0], &outputImageLevel0Data);
    T *outputImageLevel0Ptr = NULL;
    CupvaMemGetHostPointer((void **)&outputImageLevel0Ptr,
                           (void *)outputImageLevel0Data.buffer.strided.planes[0].basePtr);

    int32_t height                = inputImageData.buffer.strided.planes[0].height;
    int32_t width                 = inputImageData.buffer.strided.planes[0].width;
    int32_t outputLevel0LinePitch = outputImageLevel0Data.buffer.strided.planes[0].rowStride / sizeof(T);

    for (int32_t j = 0; j < height; j++)
    {
        // Copy entire row in one operation
        std::memcpy(&outputImageLevel0Ptr[j * outputLevel0LinePitch], &inputImagePtr[j * inputLinePitch],
                    width * sizeof(T));
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
        static_assert(!sizeof(S), "Unsupported stream type for GaussianPyramid operator");
    }
}

// Explicit template instantiation
template void GaussianPyramid<uint8_t>::operator()<cupvaStream_t>(cupvaStream_t, NVCVImageHandle, NVCVImageHandle *);
template void GaussianPyramid<uint8_t>::operator()<cudaStream_t>(cudaStream_t, NVCVImageHandle, NVCVImageHandle *);
template void GaussianPyramid<int8_t>::operator()<cupvaStream_t>(cupvaStream_t, NVCVImageHandle, NVCVImageHandle *);
template void GaussianPyramid<int8_t>::operator()<cudaStream_t>(cudaStream_t, NVCVImageHandle, NVCVImageHandle *);
template void GaussianPyramid<uint16_t>::operator()<cupvaStream_t>(cupvaStream_t, NVCVImageHandle, NVCVImageHandle *);
template void GaussianPyramid<uint16_t>::operator()<cudaStream_t>(cudaStream_t, NVCVImageHandle, NVCVImageHandle *);
template void GaussianPyramid<int16_t>::operator()<cupvaStream_t>(cupvaStream_t, NVCVImageHandle, NVCVImageHandle *);
template void GaussianPyramid<int16_t>::operator()<cudaStream_t>(cudaStream_t, NVCVImageHandle, NVCVImageHandle *);

template<typename T>
GaussianPyramid<T>::~GaussianPyramid()
{
    if (m_progInitialized)
    {
        CupvaCmdDestroy(&m_prog);
    }
}

// Explicit instantiation of the template to support the types we care about
template class GaussianPyramid<uint8_t>;
template class GaussianPyramid<int8_t>;
template class GaussianPyramid<uint16_t>;
template class GaussianPyramid<int16_t>;

} // namespace pva_solutions::priv

NVCVStatus pvaGaussianPyramidCreate(NVCVOperatorHandle *handle,
                                    NVCVImageRequirements const *const inputImageRequirements, const int32_t numLevels,
                                    const NVCVBorderType borderType, const int32_t borderValue)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        if (inputImageRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to ImageRequirements must not be NULL");
        }
        NVCVDataType dataType;
        nvcvImageFormatGetPlaneDataType(inputImageRequirements->format, 0, &dataType);
        switch (dataType)
        {
        case NVCV_DATA_TYPE_U8:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::GaussianPyramid<uint8_t>(
                inputImageRequirements, numLevels, borderType, borderValue));
            break;
        case NVCV_DATA_TYPE_S8:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::GaussianPyramid<int8_t>(
                inputImageRequirements, numLevels, borderType, borderValue));
            break;
        case NVCV_DATA_TYPE_U16:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::GaussianPyramid<uint16_t>(
                inputImageRequirements, numLevels, borderType, borderValue));
            break;
        case NVCV_DATA_TYPE_S16:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::GaussianPyramid<int16_t>(
                inputImageRequirements, numLevels, borderType, borderValue));
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }
    });
}

template<typename StreamType>
NVCVStatus GaussianPyramidSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVImageHandle inImageHandle,
                                     NVCVImageHandle *outImageHandles)
{
    return nvcv::ProtectCall([&] {
        NVCVImageFormat imageFormat;
        nvcvImageGetFormat(inImageHandle, &imageFormat);

        NVCVDataType imageDataType;
        nvcvImageFormatGetPlaneDataType(imageFormat, 0, &imageDataType);
        switch (imageDataType)
        {
        case NVCV_DATA_TYPE_U8:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::GaussianPyramid<uint8_t>>(handle)(
                stream, inImageHandle, outImageHandles);
            break;
        case NVCV_DATA_TYPE_S8:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::GaussianPyramid<int8_t>>(handle)(
                stream, inImageHandle, outImageHandles);
            break;
        case NVCV_DATA_TYPE_U16:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::GaussianPyramid<uint16_t>>(handle)(
                stream, inImageHandle, outImageHandles);
            break;
        case NVCV_DATA_TYPE_S16:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::GaussianPyramid<int16_t>>(handle)(
                stream, inImageHandle, outImageHandles);
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus GaussianPyramidSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVImageHandle,
                                                             NVCVImageHandle *);
template NVCVStatus GaussianPyramidSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVImageHandle,
                                                            NVCVImageHandle *);

// C++ overload implementations
NVCVStatus pvaGaussianPyramidSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVImageHandle inImageHandle,
                                    NVCVImageHandle *outImageHandles)
{
    return GaussianPyramidSubmitImpl(handle, stream, inImageHandle, outImageHandles);
}

NVCVStatus pvaGaussianPyramidSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageHandle inImageHandle,
                                    NVCVImageHandle *outImageHandles)
{
    return GaussianPyramidSubmitImpl(handle, stream, inImageHandle, outImageHandles);
}
