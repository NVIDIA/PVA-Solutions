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

#include "OpHistogramEqualization.hpp"

#include "device/histogram_equalization_param.h"

#include <IOperator.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <nvcv/Tensor.hpp>

#include <ErrorCheckMacros.h>
#include <OpHistogramEqualization.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <cstdint>
#include <limits>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_histogram_equalization_u8_dev);

// Geometry invariants the device kernels rely on. Mirrors the device-side
// asserts in device/histogram_equalization_param.h, and the Limitations block in
// OpHistogramEqualization.h.
static_assert(HEQ_TW_MAX % HEQ_WAYS == 0, "max tile width must be a whole number of 32-lane groups");
static_assert(HEQ_TH_MAX >= 1, "tile byte budget must admit at least one row at the max tile width");
static_assert(HEQ_MIN_WIDTH >= HEQ_WAYS, "the minimum supported width must fill at least one vector group");
static_assert(HEQ_NBINS == 256, "a U8 intensity domain has exactly 256 bins");

namespace pva_solutions::priv {

std::mutex HistogramEqualization::s_execMapMutex;

std::map<HistogramEqualization::ExecKey, std::weak_ptr<cupva::Executable>>
    HistogramEqualization::s_histogramEqualizationExecutables;

std::shared_ptr<cupva::Executable> HistogramEqualization::GetHistogramEqualizationExec()
{
    static IOperator::ExecutableData sHistogramEqualizationExecData = {
        CUPVA_EXECUTABLE_DATA(pva_histogram_equalization_u8_dev),
        CUPVA_EXECUTABLE_SIZE(pva_histogram_equalization_u8_dev)};

    ExecKey execKey = std::make_tuple(cupva::Context::GetCurrent());
    return IOperator::GetExecutable<ExecKey>(sHistogramEqualizationExecData, execKey,
                                             s_histogramEqualizationExecutables, s_execMapMutex);
}

HistogramEqualization::HistogramEqualization(NVCVTensorRequirements const *const tensorRequirements)
    : m_tensorRequirements(*tensorRequirements)
    /* Bases stay null until submit rebases them. */
    , m_srcOp{cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM, nullptr)}
    , m_dstOp{cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM, nullptr)}
    , m_prog{}
{
    // Permute shape and strides into NCHW so H, W and the line pitch are read
    // from fixed positions regardless of the declared layout.
    int64_t imageShapeNCHW[4];
    int64_t imageStridesNCHW[4];
    nvcvTensorShapePermute(tensorRequirements->layout, tensorRequirements->shape, NVCV_TENSOR_NCHW, imageShapeNCHW);
    nvcvTensorShapePermute(tensorRequirements->layout, tensorRequirements->strides, NVCV_TENSOR_NCHW, imageStridesNCHW);

    int32_t imageBatch    = imageShapeNCHW[0];
    int32_t imageChannels = imageShapeNCHW[1];
    m_height              = imageShapeNCHW[2];
    m_width               = imageShapeNCHW[3];
    m_linePitch           = imageStridesNCHW[2] / (nvcv::DataType(tensorRequirements->dtype).bitsPerPixel() / 8);

    if (tensorRequirements->layout != nvcv::TENSOR_HWC)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor must have the HWC layout");
    }
    if ((imageBatch != 0 && imageBatch != 1) || (imageChannels != 1))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Image tensor must be grayscale: channel count must be 1");
    }
    if (tensorRequirements->dtype != NVCV_DATA_TYPE_U8)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type: only U8 is supported");
    }
    if (m_width < HEQ_MIN_WIDTH)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Image width must be at least 129 pixels");
    }
    if (m_height < HEQ_MIN_HEIGHT)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Image height must be at least 17 pixels");
    }

    m_exec    = GetHistogramEqualizationExec();
    auto prog = CmdProgram::Create(*m_exec.get());

    /*
     * Tile shape: the widest tile is not the best one. Pick the shape that covers the image in the
     * fewest tiles, subject to
     *   - a whole number of vector groups wide, since the device loops consume HEQ_TILE_WIDTH_GRAIN
     *     pixels per issue,
     *   - at least HEQ_TW_MIN bytes per row, so a transfer still has enough contiguous
     *     bytes to keep the DMA busy. An image narrower than that is covered by one tile whose
     *     roi() runs past the image; the DMA pads the difference,
     *   - no more right-hand padding on the last column than HEQ_RDF_MAX_PAD, which is the
     *     largest the DMA engine accepts.
     * The scan runs narrow to wide and replaces only on a strict improvement, so a tie keeps the
     * narrower shape; the full-width shape seeds the search and is displaced only by a strictly
     * smaller tile count.
     */
    const int32_t twMax =
        std::max(std::min(m_width / HEQ_TILE_WIDTH_GRAIN * HEQ_TILE_WIDTH_GRAIN, static_cast<int32_t>(HEQ_TW_MAX)),
                 static_cast<int32_t>(HEQ_TW_MIN));

    auto padding = [this](int32_t w) {
        return ((m_width + w - 1) / w) * w - m_width;
    };
    auto tileRows = [this](int32_t w) {
        return std::min(m_height, static_cast<int32_t>(HEQ_TILE_BYTES) / w);
    };
    auto tileCount = [this](int32_t w, int32_t h) {
        return ((m_width + w - 1) / w) * ((m_height + h - 1) / h);
    };

    int32_t tw        = 0;
    int32_t th        = 0;
    int32_t bestTiles = std::numeric_limits<int32_t>::max();

    if (padding(twMax) <= HEQ_RDF_MAX_PAD)
    {
        tw        = twMax;
        th        = tileRows(twMax);
        bestTiles = tileCount(tw, th);
    }

    for (int32_t w = HEQ_TILE_WIDTH_GRAIN; w <= twMax; w += HEQ_TILE_WIDTH_GRAIN)
    {
        if ((w < HEQ_TW_MIN) || (padding(w) > HEQ_RDF_MAX_PAD))
        {
            continue;
        }
        const int32_t h = tileRows(w);
        if (h < 1)
        {
            continue;
        }
        const int32_t tiles = tileCount(w, h);
        if (tiles < bestTiles)
        {
            bestTiles = tiles;
            tw        = w;
            th        = h;
        }
    }

    const int32_t niterX = (m_width + tw - 1) / tw;
    const int32_t niterY = (m_height + th - 1) / th;

    prog["niterX"]      = niterX;
    prog["niterY"]      = niterY;
    prog["imageWidth"]  = m_width;
    prog["imageHeight"] = m_height;
    prog["tileWidth"]   = tw;
    prog["tileHeight"]  = th;

    // One input tile buffer serves both read passes: their dataflows are opened and closed in
    // disjoint phases.
    uint8_t *src_v = prog["src_v"].ptr<uint8_t>();
    uint8_t *dst_v = prog["dst_v"].ptr<uint8_t>();

    /*
     * The image is read once to build the histogram and again to apply the lookup table, and
     * written once. The two read passes never run at the same time, so they are linked and share
     * one head's DMA resources instead of each holding their own. The write is a separate head:
     * it has to overlap the apply pass, where the write-out of tile N runs against the
     * equalization of tile N+1.
     */
    RasterDataFlow &srcDataFlow = prog.addDataFlowHead<RasterDataFlow>();
    srcDataFlow.handler(prog["srcHdl"])
        .src(m_srcOp, m_width, m_height, m_linePitch)
        .tileBuffer(src_v)
        .tile(tw, th)
        .roi(0, 0, std::max(tw, m_width), m_height);

    RasterDataFlow &src2DataFlow = prog.addDataFlow<RasterDataFlow>();
    src2DataFlow.handler(prog["src2Hdl"])
        .src(m_srcOp, m_width, m_height, m_linePitch)
        .tileBuffer(src_v)
        .tile(tw, th)
        .roi(0, 0, std::max(tw, m_width), m_height);

    srcDataFlow.link(src2DataFlow);

    RasterDataFlow &dstDataFlow = prog.addDataFlowHead<RasterDataFlow>();
    dstDataFlow.handler(prog["dstHdl"])
        .tileBuffer(dst_v)
        .dst(m_dstOp, m_width, m_height, m_linePitch)
        .tile(tw, th)
        .roi(0, 0, std::max(tw, m_width), m_height);

    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename S>
void HistogramEqualization::operator()(S stream, const nvcv::Tensor &in, const nvcv::Tensor &out)
{
    auto inTensorData  = in.exportData<nvcv::TensorDataStrided>();
    auto outTensorData = out.exportData<nvcv::TensorDataStrided>();

    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_tensorRequirements, inTensorData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_tensorRequirements, outTensorData->cdata()))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Input and output tensors must have the same requirements that was used to create the operator");
    }
    if ((static_cast<void *>(inTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(outTensorData->basePtr()) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor base pointers must not be NULL");
    }

    m_srcOp.setBase(static_cast<uint8_t *>(static_cast<void *>(inTensorData->basePtr())));
    m_dstOp.setBase(static_cast<uint8_t *>(static_cast<void *>(outTensorData->basePtr())));
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
        static_assert(!sizeof(S), "Unsupported stream type for HistogramEqualization operator");
    }
}

// Explicit template instantiation
template void HistogramEqualization::operator()<cupvaStream_t>(cupvaStream_t, const nvcv::Tensor &,
                                                               const nvcv::Tensor &);
template void HistogramEqualization::operator()<cudaStream_t>(cudaStream_t, const nvcv::Tensor &, const nvcv::Tensor &);

HistogramEqualization::~HistogramEqualization()
{
    CupvaCmdDestroy(&m_prog);
}

} // namespace pva_solutions::priv

NVCVStatus pvaHistogramEqualizationCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements *tensorRequirements)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        if (tensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to TensorRequirements must not be NULL");
        }
        *handle =
            reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::HistogramEqualization(tensorRequirements));
    });
}

template<typename StreamType>
NVCVStatus HistogramEqualizationSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle in,
                                           NVCVTensorHandle out)
{
    return nvcv::ProtectCall([&] {
        nvcv::TensorWrapHandle input(in), output(out);

        NVCVDataType inDataType;
        nvcvTensorGetDataType(in, &inDataType);

        if (inDataType != NVCV_DATA_TYPE_U8)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }

        pva_solutions::priv::ToDynamicRef<pva_solutions::priv::HistogramEqualization>(handle)(stream, input, output);
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus HistogramEqualizationSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle,
                                                                   NVCVTensorHandle);
template NVCVStatus HistogramEqualizationSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle,
                                                                  NVCVTensorHandle);

// C++ overload implementations
NVCVStatus pvaHistogramEqualizationSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle in,
                                          NVCVTensorHandle out)
{
    return HistogramEqualizationSubmitImpl(handle, stream, in, out);
}

NVCVStatus pvaHistogramEqualizationSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                          NVCVTensorHandle out)
{
    return HistogramEqualizationSubmitImpl(handle, stream, in, out);
}
