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

#include "OpBlurFilterROI.hpp"

#include <IOperator.hpp>
#include <ImageUtils.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>

#include <OpBlurFilterROI.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <climits>
#include <cmath>
#include <set>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_blur_filter_roi_u8_dev);

namespace pva_solutions::priv {

template<typename T>
std::mutex BlurFilterROI<T>::s_execMapMutex;

template<typename T>
std::map<typename BlurFilterROI<T>::ExecKey, std::weak_ptr<cupva::Executable>>
    BlurFilterROI<T>::s_blurFilterROIExecutables;

template<typename T>
std::shared_ptr<cupva::Executable> BlurFilterROI<T>::GetBlurFilterROIExec()
{
    static IOperator::ExecutableData sBlurFilterROIExecData;

    sBlurFilterROIExecData = {CUPVA_EXECUTABLE_DATA(pva_blur_filter_roi_u8_dev),
                              CUPVA_EXECUTABLE_SIZE(pva_blur_filter_roi_u8_dev)};

    ExecKey execKey = std::make_tuple(cupva::Context::GetCurrent());
    return IOperator::GetExecutable<ExecKey>(sBlurFilterROIExecData, execKey, s_blurFilterROIExecutables,
                                             s_execMapMutex);
}

template<typename T>
bool BlurFilterROI<T>::isImageFormatNV12(NVCVImageFormat format)
{
    NVCVChromaSubsampling chromaSubsampling;
    nvcvImageFormatGetChromaSubsampling(format, &chromaSubsampling);

    return chromaSubsampling == NVCV_CSS_420;
}

template<typename T>
BlurFilterROI<T>::BlurFilterROI(NVCVImageRequirements const *const inImageRequirements,
                                NVCVImageRequirements const *const outImageRequirements,
                                NVCVTensorRequirements const *const rectTensorRequirements, const NVCVSize2D blockSize)
    : m_inImageRequirements{*inImageRequirements}
    , m_outImageRequirements{*outImageRequirements}
    , m_rectTensorRequirements{*rectTensorRequirements}
    , m_width(inImageRequirements->width)
    , m_height(inImageRequirements->height)
    , m_inLinePitch{}
    , m_outLinePitch{}
    , m_inFormat(inImageRequirements->format)
    , m_outFormat(outImageRequirements->format)
    , m_rectOp(cupva::OffsetPointer<int32_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_srcOps{}
    , m_dstOps{}
    , m_prog{}
{
    if (!isImageFormatNV12(m_inFormat) || !isImageFormatNV12(m_outFormat))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input/output image format");
    }

    const size_t numRects = rectTensorRequirements->shape[0];
    if (numRects > MAX_RECT_COUNT)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Number of rectangles is larger than maximum supported size");
    }

    nvcvImageFormatGetMemLayout(m_inFormat, &m_inLayout);
    nvcvImageFormatGetMemLayout(m_outFormat, &m_outLayout);

    nvcvImageFormatGetNumPlanes(m_inFormat, &m_numPlanes);

    m_exec    = GetBlurFilterROIExec();
    auto prog = CmdProgram::Create(*m_exec.get());

    int32_t phase = 0;

    // execute blur filter operation only on Y plane

    int32_t numTiles[MAX_PLANE_COUNT];

    for (int32_t i = 0; i < m_numPlanes; i++)
    {
        T *inBuf            = prog["inBuf"].ptr<T>();
        T *outBuf           = prog["outBuf"].ptr<T>();
        int32_t planeWidth  = m_width;
        int32_t planeHeight = m_height;
        int32_t tileWidth   = planeWidth;
        int32_t tileHeight  = std::pow(2, std::floor(std::log2(MAX_BUF_SIZE / tileWidth)));
        tileHeight          = std::min(std::min(tileHeight, planeHeight), 256);
        if (isImageFormatNV12(m_inFormat) && (i == 1))
        {
            // UV plane is half the size of Y plane
            inBuf       = prog["inBuf"].ptr<T>();
            outBuf      = prog["inBuf"].ptr<T>();
            planeWidth  = (m_width + 1) / 2 * 2;
            planeHeight = (m_height + 1) / 2;
            tileWidth   = std::min(TILE_W, planeWidth);
            tileHeight  = std::min(TILE_H, planeHeight);
        }

        m_inLinePitch[i] = inImageRequirements->planeRowStride[i] / sizeof(T);
        m_srcOps[i]      = cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM, nullptr);
        prog.addDataFlowHead<RasterDataFlow>(phase)
            .handler(prog[std::string("srcHdl" + std::to_string(i)).c_str()])
            .src(m_srcOps[i], planeWidth, planeHeight, m_inLinePitch[i])
            .tileBuffer(inBuf)
            .tile(tileWidth, tileHeight);

        m_outLinePitch[i] = outImageRequirements->planeRowStride[i] / sizeof(T);
        m_dstOps[i]       = cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM, nullptr);
        prog.addDataFlowHead<RasterDataFlow>(phase)
            .handler(prog[std::string("dstHdl" + std::to_string(i)).c_str()])
            .tileBuffer(outBuf)
            .tile(tileWidth, tileHeight)
            .dst(m_dstOps[i], planeWidth, planeHeight, m_outLinePitch[i]);

        phase++;

        const int32_t nTileX = (planeWidth + tileWidth - 1) / tileWidth;
        const int32_t nTileY = (planeHeight + tileHeight - 1) / tileHeight;
        numTiles[i]          = nTileX * nTileY;

        if (i == 0)
        {
            prog["tileW"] = tileWidth;
            prog["tileH"] = tileHeight;
        }
    }

    prog["numPlanes"] = m_numPlanes;
    prog["numTiles"].set(numTiles, MAX_PLANE_COUNT);

    prog.addDataFlowHead<SequenceDataFlow>(phase)
        .handler(prog["rectHdl"])
        .addTransfer()
        .src(m_rectOp, sizeof(NVCVRectI))
        .dst(prog["rectBuf"].ptr<int32_t>(), sizeof(NVCVRectI))
        .tile(sizeof(NVCVRectI), numRects)
        .mode(TransferModeType::CONTINUOUS);

    phase++;

    prog["blockW"] = blockSize.w;
    prog["blockH"] = blockSize.h;
    prog["imageH"] = m_height;

    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename T>
template<typename S>
void BlurFilterROI<T>::operator()(S stream, const nvcv::Image &in, const nvcv::Image &out, const nvcv::Tensor &rect,
                                  const size_t numRects)
{
    auto inImageData    = in.exportData<nvcv::ImageDataStrided>();
    auto outImageData   = out.exportData<nvcv::ImageDataStrided>();
    auto rectTensorData = rect.exportData<nvcv::TensorDataStrided>();

    if (!pva_solutions::utils::AreImageRequirementsEqualExcludingLayout(m_inImageRequirements, inImageData->cdata()) ||
        !pva_solutions::utils::AreImageRequirementsEqualExcludingLayout(m_outImageRequirements, outImageData->cdata()))
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
    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_rectTensorRequirements, rectTensorData->cdata()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Rectangle tensor must have the same requirements that was used to create the operator");
    }
    if (static_cast<void *>(rectTensorData->basePtr()) == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor base pointers must not be NULL");
    }

    for (int32_t i = 0; i < m_numPlanes; i++)
    {
        m_srcOps[i].setBase(static_cast<T *>(static_cast<void *>(inImageData->plane(i).basePtr)));
    }
    for (int32_t i = 0; i < m_numPlanes; i++)
    {
        m_dstOps[i].setBase(static_cast<T *>(static_cast<void *>(outImageData->plane(i).basePtr)));
    }
    m_rectOp.setBase(static_cast<int32_t *>(static_cast<void *>(rectTensorData->basePtr())));

    cupvaParameter_t param;
    CupvaCmdProgramGetParameter(&m_prog, &param, "numRects");
    CupvaParameterSetValueScalar(&param, &numRects, sizeof(int32_t));

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
        static_assert(!sizeof(S), "Unsupported stream type for BlurFilterROI operator");
    }

    if (!(err == CUPVA_ERROR_NONE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Submit failed");
    }
}

template<typename T>
BlurFilterROI<T>::~BlurFilterROI()
{
    CupvaCmdDestroy(&m_prog);
}

// Explicit instantiation of the template to support the types we care about
template class BlurFilterROI<uint8_t>;

} // namespace pva_solutions::priv

NVCVStatus pvaBlurFilterROICreate(NVCVOperatorHandle *handle, NVCVImageRequirements const *const inImageRequirements,
                                  NVCVImageRequirements const *const outImageRequirements,
                                  NVCVTensorRequirements const *const rectTensorRequirements,
                                  const NVCVSize2D blockSize)
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
        if (rectTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to rectangle TensorRequirements must not be NULL");
        }

        NVCVDataType dataType;
        nvcvImageFormatGetPlaneDataType(inImageRequirements->format, 0, &dataType);

        switch (dataType)
        {
        case NVCV_DATA_TYPE_U8:
        case NVCV_DATA_TYPE_2U8:
        case NVCV_DATA_TYPE_3U8:
        case NVCV_DATA_TYPE_4U8:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::BlurFilterROI<uint8_t>(
                inImageRequirements, outImageRequirements, rectTensorRequirements, blockSize));
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }
    });
}

template<typename StreamType>
NVCVStatus BlurFilterROISubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVImageHandle const in,
                                   NVCVImageHandle out, NVCVTensorHandle const rect, const size_t numRects)
{
    return nvcv::ProtectCall([&] {
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
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::BlurFilterROI<uint8_t>>(handle)(
                stream, nvcv::ImageWrapHandle(in), nvcv::ImageWrapHandle(out), nvcv::TensorWrapHandle(rect), numRects);
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus BlurFilterROISubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVImageHandle const,
                                                           NVCVImageHandle, NVCVTensorHandle const, const size_t);
template NVCVStatus BlurFilterROISubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVImageHandle const,
                                                          NVCVImageHandle, NVCVTensorHandle const, const size_t);

// C++ overload implementations
NVCVStatus pvaBlurFilterROISubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVImageHandle const in,
                                  NVCVImageHandle out, NVCVTensorHandle const rect, const size_t numRects)
{
    return BlurFilterROISubmitImpl(handle, stream, in, out, rect, numRects);
}

NVCVStatus pvaBlurFilterROISubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageHandle const in,
                                  NVCVImageHandle out, NVCVTensorHandle const rect, const size_t numRects)
{
    return BlurFilterROISubmitImpl(handle, stream, in, out, rect, numRects);
}
