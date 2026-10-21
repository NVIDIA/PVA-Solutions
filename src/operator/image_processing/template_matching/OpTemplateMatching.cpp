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

#include "OpTemplateMatching.hpp"

#include "device/template_matching_param.h"

#include <IOperator.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <nvcv/Tensor.hpp>

#include <ErrorCheckMacros.h>
#include <OpTemplateMatching.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <type_traits>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_template_matching_dev);

namespace pva_solutions::priv {

std::mutex TemplateMatching::s_execMapMutex;
std::map<TemplateMatching::ExecKey, std::weak_ptr<cupva::Executable>> TemplateMatching::s_execs;

std::shared_ptr<cupva::Executable> TemplateMatching::GetExec()
{
    static IOperator::ExecutableData const execData{CUPVA_EXECUTABLE_DATA(pva_template_matching_dev),
                                                    CUPVA_EXECUTABLE_SIZE(pva_template_matching_dev)};

    return IOperator::GetExecutable<ExecKey>(execData, std::make_tuple(cupva::Context::GetCurrent()), s_execs,
                                             s_execMapMutex);
}

namespace {

/* Layouts accepted for every tensor: N and C, when present, must be 1. */
void ValidateLayout(NVCVTensorLayout layout, const char *what)
{
    if ((layout != NVCV_TENSOR_HWC) && (layout != NVCV_TENSOR_CHW) && (layout != NVCV_TENSOR_NHWC) &&
        (layout != NVCV_TENSOR_NCHW))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s tensor must have HWC, CHW, NHWC or NCHW layout",
                              what);
    }
}

struct TensorGeometry
{
    int32_t width;
    int32_t height;
    int32_t linePitch;
};

TensorGeometry DescribeTensor(NVCVTensorRequirements const *const req, const char *what)
{
    ValidateLayout(req->layout, what);

    int64_t shapeNCHW[4];
    int64_t stridesNCHW[4];
    nvcvTensorShapePermute(req->layout, req->shape, NVCV_TENSOR_NCHW, shapeNCHW);
    nvcvTensorShapePermute(req->layout, req->strides, NVCV_TENSOR_NCHW, stridesNCHW);

    int32_t const batch    = static_cast<int32_t>(shapeNCHW[0]);
    int32_t const channels = static_cast<int32_t>(shapeNCHW[1]);

    if (((batch != 0) && (batch != 1)) || ((channels != 0) && (channels != 1)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s tensor must have N = 1 and C = 1", what);
    }

    int32_t const bytesPerSample = nvcv::DataType(req->dtype).bitsPerPixel() / 8;

    TensorGeometry geom;
    geom.height    = static_cast<int32_t>(shapeNCHW[2]);
    geom.width     = static_cast<int32_t>(shapeNCHW[3]);
    geom.linePitch = static_cast<int32_t>(stridesNCHW[2]) / bytesPerSample;
    return geom;
}

/* The asserts keep these bounds pinned to the device header rather than
 * drifting from it silently. */
constexpr int32_t kMinImageDim    = 94;
constexpr int32_t kMaxTemplateDim = 31;

static_assert(kMinImageDim == TM_MIN_IMAGE_DIM);
static_assert(kMaxTemplateDim == TM_MAX_TDIM);
/* Guards the groupPairs=1 fallback of the accumulator-group search below: a
 * single tap pair of the widest template spans at most 2*31 samples, so it
 * always fits one 24-bit accumulator group and the search can never fail. */
static_assert(2 * TM_MAX_TDIM <= TM_ACC_MAX_SAMPLES);
/* Smallest output a valid request can produce, in each dimension. */
constexpr int32_t kMinOutputDim = kMinImageDim - kMaxTemplateDim + 1;
static_assert(kMinOutputDim >= TM_TILE_HEIGHT);
/* Staging buffer bounds the device packing relies on: a template row has to fit
 * the pitch, and the rows the packing reads plus the zero row have to fit the
 * buffer. */
static_assert(TM_MAX_TDIM <= TM_STAGE_PITCH);
static_assert(TM_STAGE_ROWS >= (2 * TM_MAX_IB) + 1);

/* Tile widths the host may pick from, widest first, and the padding the DMA can
 * synthesise for one edge tile (RasterDataFlow rules, cupva_host.hpp). */
constexpr int32_t kTileWidths[] = {TM_TILE_WIDTH, 128, 64};
constexpr int32_t kMaxDmaPad    = 255;

constexpr int32_t kNarrowestTile = kTileWidths[std::size(kTileWidths) - 1];

/* ChunkCount() divides by TM_CHUNK and rounds up to TM_SCORE_UNROLL, so every
 * candidate must be a whole number of chunks and the rounded-up count must
 * still fit the buffers, which are sized for TM_TILE_WIDTH. */
static_assert(kNarrowestTile <= kMinOutputDim);
static_assert(TM_SCORE_UNROLL * TM_CHUNK <= TM_TILE_WIDTH);
static_assert([] {
    for (int32_t const candidate : kTileWidths)
    {
        if (((candidate % TM_CHUNK) != 0) || (candidate > TM_TILE_WIDTH))
        {
            return false;
        }
    }
    return true;
}());

/* Columns the DMA must synthesise for the right-most tile of a row: the tiles
 * cover ceil(roi / tile) whole tiles plus the halo, and the surface stops at
 * imageWidth. */
int32_t EdgeTilePadding(int32_t const roiWidth, int32_t const tileWidth, int32_t const haloX, int32_t const imageWidth)
{
    int32_t const tiles = (roiWidth + tileWidth - 1) / tileWidth;
    return (tiles * tileWidth) + haloX - imageWidth;
}

/* Widest tile satisfying RasterDataFlow's tile() <= roi() and the DMA padding
 * limit. The narrowest candidate satisfies tile() <= roi() for every request
 * the parameter range admits, so the search always returns. It does not bound
 * the tile count, which RasterDataFlow caps at 256 per dimension -- an output
 * past ~32k columns that falls off the widest tile exceeds that and is rejected
 * by compileDataFlows(). */
int32_t SelectTileWidth(int32_t const roiWidth, int32_t const haloX, int32_t const imageWidth)
{
    for (int32_t const candidate : kTileWidths)
    {
        if ((candidate <= roiWidth) && (EdgeTilePadding(roiWidth, candidate, haloX, imageWidth) <= kMaxDmaPad))
        {
            return candidate;
        }
    }
    return kNarrowestTile;
}

/* Chunks the kernel evaluates per tile, rounded up so the score loop's unroll
 * divides evenly. The surplus computes into VMEM sized for TM_TILE_WIDTH and is
 * discarded by the output tile. */
int32_t ChunkCount(int32_t const tileWidth)
{
    int32_t const chunks = tileWidth / TM_CHUNK;
    return ((chunks + TM_SCORE_UNROLL - 1) / TM_SCORE_UNROLL) * TM_SCORE_UNROLL;
}

} // namespace

TemplateMatching::TemplateMatching(NVCVTensorRequirements const *const inImageTensorRequirements,
                                   NVCVTensorRequirements const *const inTemplateTensorRequirements,
                                   NVCVTensorRequirements const *const outTensorRequirements)
    : m_inImageTensorRequirements(*inImageTensorRequirements)
    , m_inTemplateTensorRequirements(*inTemplateTensorRequirements)
    , m_outTensorRequirements(*outTensorRequirements)
    , m_imageOp(cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_templateOp(cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_outOp(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr))
    , m_prog{}
{
    if (inImageTensorRequirements->dtype != NVCV_DATA_TYPE_U8)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input image tensor must be U8");
    }
    if (inTemplateTensorRequirements->dtype != NVCV_DATA_TYPE_U8)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input template tensor must be U8");
    }
    if (outTensorRequirements->dtype != NVCV_DATA_TYPE_F32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output tensor must be F32");
    }

    TensorGeometry const image = DescribeTensor(inImageTensorRequirements, "Input image");
    TensorGeometry const tmpl  = DescribeTensor(inTemplateTensorRequirements, "Input template");
    TensorGeometry const out   = DescribeTensor(outTensorRequirements, "Output");

    if ((image.width < kMinImageDim) || (image.height < kMinImageDim))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input image width and height must both be at least %d", kMinImageDim);
    }
    if ((tmpl.width < 1) || (tmpl.width > kMaxTemplateDim) || (tmpl.height < 1) || (tmpl.height > kMaxTemplateDim))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Template width and height must be in [1, %d]",
                              kMaxTemplateDim);
    }
    if ((out.width != (image.width - tmpl.width + 1)) || (out.height != (image.height - tmpl.height + 1)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output size must be (imageWidth - templateWidth + 1) x (imageHeight - templateHeight "
                              "+ 1)");
    }

    /* DVLDPermHBU forces its address even, so the tile-buffer column
     * origin -- the requested x halo -- must be even too. */
    int32_t const haloX = tmpl.width & ~1;
    int32_t const haloY = tmpl.height - 1;

    /* The ROI is exactly the output extent: taps reaching past a tile are
     * covered by the halo, which is padded where it leaves the surface. The
     * 32-row tile needs no candidate search -- its edge padding is at most 31
     * rows for any request. */
    int32_t const inRoiWidth  = out.width;
    int32_t const inRoiHeight = out.height;
    int32_t const tileWidth   = SelectTileWidth(inRoiWidth, haloX, image.width);

    /* Built only after the checks above pin the template to [1, kMaxTemplateDim],
     * so neither the executable nor any size-derived geometry is touched for an
     * invalid request. */
    m_exec = GetExec();

    /* Set into the program below, which copies the contents, so this does
     * not have to outlive the constructor. */
    TemplateMatchingParams params{};

    params.haloX          = haloX;
    params.haloY          = haloY;
    params.imageWidth     = image.width;
    params.imageHeight    = image.height;
    params.templateWidth  = tmpl.width;
    params.templateHeight = tmpl.height;
    params.outWidth       = out.width;
    params.outHeight      = out.height;
    params.niterX         = (inRoiWidth + tileWidth - 1) / tileWidth;
    params.niterY         = (inRoiHeight + TM_TILE_HEIGHT - 1) / TM_TILE_HEIGHT;
    params.tapGroupsX     = (tmpl.width + 3) / 4;

    /* At most TM_ACC_MAX_SAMPLES template samples may share one 24-bit
     * accumulator, so a group of G vertical tap pairs is safe whenever
     * min(2*G, h) * w <= TM_ACC_MAX_SAMPLES. G is the largest divisor of
     * IB meeting that bound, so groups tile IB exactly. */
    int32_t tapPairs = (tmpl.height / 2) + 1;
    /* The correlation tap loop is compiled for a trip count of at least one,
     * so a group needs two taps. Only a one-row template four columns or
     * narrower falls short. Padding its tap pairs to two adds one tap built
     * from staging rows the transfer never writes, so its coefficients are zero
     * and it contributes nothing. That tap also advances the image agens two
     * rows past the window, which stays in bounds because they carry the tile
     * ring and wrap in hardware. */
    if ((tapPairs * params.tapGroupsX) < 2)
    {
        tapPairs = 2;
    }
    int32_t groupPairs = 1;
    for (int32_t candidate = tapPairs; candidate >= 1; candidate--)
    {
        if ((tapPairs % candidate) != 0)
        {
            continue;
        }
        if ((std::min(2 * candidate, tmpl.height) * tmpl.width) <= TM_ACC_MAX_SAMPLES)
        {
            groupPairs = candidate;
            break;
        }
    }
    int32_t const groupCount = tapPairs / groupPairs;

    params.tapPairsY = tapPairs;
    params.groupTaps = groupPairs * params.tapGroupsX;
    params.numGroups = groupCount;
    params.numTaps   = params.tapPairsY * params.tapGroupsX;
    params.sqPairs   = (tmpl.width + 1) / 2;
    params.vRows     = tmpl.height + 2;
    params.nChunk    = ChunkCount(tileWidth);

    auto prog = CmdProgram::Create(*m_exec.get());

    /* The device packs the coefficient blocks and the template energy from
     * this copy, so the host never reads the template. It is closed before
     * the image dataflows open, which is why it gets its own phase. */
    SequenceDataFlow &tmplDataFlow = prog.addDataFlowHead<SequenceDataFlow>(0);
    tmplDataFlow.handler(prog["tmplHandler"])
        .addTransfer()
        .src(m_templateOp, tmpl.linePitch)
        .dst(prog["tmplStage"].ptr<uint8_t>(), TM_STAGE_PITCH)
        .tile(tmpl.width, tmpl.height);

    /* VFilt4x2x2 needs an even and an odd image row in the same cycle, so the
     * tile has to be in both superbank A and B. Full replication gives the
     * second copy from the same DMA transfer, so DRAM is read once. */
    RasterDataFlow &imgDataFlow = prog.addDataFlowHead<RasterDataFlow>(1);
    imgDataFlow.handler(prog["imgHandler"])
        .src(m_imageOp, image.width, image.height, image.linePitch)
        .tileBuffer(prog["srcTileA"].ptr<uint8_t>(), prog["srcTileB"].ptr<uint8_t>())
        .roi(0, 0, inRoiWidth, inRoiHeight)
        .tile(tileWidth, TM_TILE_HEIGHT)
        .halo(haloX, haloY, PAD_BPE);

    RasterDataFlow &outDataFlow = prog.addDataFlowHead<RasterDataFlow>(1);
    outDataFlow.handler(prog["dstHandler"])
        .tileBuffer(prog["outTile"].ptr<float>())
        .dst(m_outOp, out.width, out.height, out.linePitch)
        .tile(tileWidth, TM_TILE_HEIGHT)
        /* Arena is the widest tile, not the selected one: ChunkCount may round
         * past the selected width, and the surplus must land inside it. */
        .tileArena(TM_TILE_WIDTH, TM_TILE_HEIGHT);

    prog["algoParams"].set(&params, 1);

    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
}

TemplateMatching::~TemplateMatching()
{
    CupvaCmdDestroy(&m_prog);
}

void TemplateMatching::Submit(cupvaStream_t stream, const nvcv::Tensor &inImage, const nvcv::Tensor &inTemplate,
                              const nvcv::Tensor &out)
{
    SubmitImpl(stream, inImage, inTemplate, out);
}

void TemplateMatching::Submit(cudaStream_t stream, const nvcv::Tensor &inImage, const nvcv::Tensor &inTemplate,
                              const nvcv::Tensor &out)
{
    SubmitImpl(stream, inImage, inTemplate, out);
}

template<typename S>
void TemplateMatching::SubmitImpl(S stream, const nvcv::Tensor &inImage, const nvcv::Tensor &inTemplate,
                                  const nvcv::Tensor &out)
{
    auto imageTensorData    = inImage.exportData<nvcv::TensorDataStrided>();
    auto templateTensorData = inTemplate.exportData<nvcv::TensorDataStrided>();
    auto outTensorData      = out.exportData<nvcv::TensorDataStrided>();

    if ((static_cast<void *>(imageTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(templateTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(outTensorData->basePtr()) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor base pointers must not be NULL");
    }

    m_imageOp.setBase(static_cast<uint8_t *>(static_cast<void *>(imageTensorData->basePtr())));
    m_templateOp.setBase(static_cast<uint8_t *>(static_cast<void *>(templateTensorData->basePtr())));
    m_outOp.setBase(static_cast<float *>(static_cast<void *>(outTensorData->basePtr())));
    CUPVA_CHECK_ERROR(CupvaCmdProgramUpdateDataFlows(&m_prog));

    cupvaCmd_t const *submitCmds[1] = {&m_prog};
    cupvaCmdStatus_t cmdStatus[1]   = {nullptr};

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
        static_assert(!sizeof(S), "Unsupported stream type for TemplateMatching operator");
    }
}

void TemplateMatching::CheckTensorsMatchCreate(const nvcv::Tensor &inImage, const nvcv::Tensor &inTemplate,
                                               const nvcv::Tensor &out) const
{
    auto inImageTensorData    = inImage.exportData<nvcv::TensorDataStrided>();
    auto inTemplateTensorData = inTemplate.exportData<nvcv::TensorDataStrided>();
    auto outTensorData        = out.exportData<nvcv::TensorDataStrided>();

    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_inImageTensorRequirements, inImageTensorData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_inTemplateTensorRequirements,
                                                          inTemplateTensorData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_outTensorRequirements, outTensorData->cdata()))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Input and output tensors must have the same requirements that were used to create the operator");
    }
}

} // namespace pva_solutions::priv

NVCVStatus pvaTemplateMatchingCreate(NVCVOperatorHandle *handle,
                                     NVCVTensorRequirements const *const inImageTensorRequirements,
                                     NVCVTensorRequirements const *const inTemplateTensorRequirements,
                                     NVCVTensorRequirements const *const outTensorRequirements)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        if (inImageTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to input image TensorRequirements must not be NULL");
        }
        if (inTemplateTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to input template TensorRequirements must not be NULL");
        }
        if (outTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to output TensorRequirements must not be NULL");
        }

        *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::TemplateMatching(
            inImageTensorRequirements, inTemplateTensorRequirements, outTensorRequirements));
    });
}

template<typename StreamType>
NVCVStatus TemplateMatchingSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle inImage,
                                      NVCVTensorHandle inTemplate, NVCVTensorHandle out)
{
    return nvcv::ProtectCall([&] {
        nvcv::TensorWrapHandle image(inImage), tmpl(inTemplate), score(out);
        pva_solutions::priv::ToDynamicRef<pva_solutions::priv::TemplateMatching>(handle)(stream, image, tmpl, score);
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus TemplateMatchingSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle,
                                                              NVCVTensorHandle, NVCVTensorHandle);
template NVCVStatus TemplateMatchingSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle,
                                                             NVCVTensorHandle, NVCVTensorHandle);

// C++ overload implementations
NVCVStatus pvaTemplateMatchingSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle inImage,
                                     NVCVTensorHandle inTemplate, NVCVTensorHandle out)
{
    return TemplateMatchingSubmitImpl(handle, stream, inImage, inTemplate, out);
}

NVCVStatus pvaTemplateMatchingSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle inImage,
                                     NVCVTensorHandle inTemplate, NVCVTensorHandle out)
{
    return TemplateMatchingSubmitImpl(handle, stream, inImage, inTemplate, out);
}
