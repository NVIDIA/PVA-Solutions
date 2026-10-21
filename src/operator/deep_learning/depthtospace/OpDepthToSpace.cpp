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

#include "OpDepthToSpace.hpp"

#include "TileSizeEnumeration.hpp"

/*
 * Included at one arbitrary BLK_SZ purely for the BLK_SZ-independent constants and the
 * TILE_W_FOR_BLK rule; the host evaluates that rule per block size itself.
 */
#define BLK_SZ 4
#include "device/depthtospace_rdf_param.h"
#undef BLK_SZ

#include <IOperator.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <nvcv/Tensor.hpp>

#include <ErrorCheckMacros.h>
#include <OpDepthToSpace.h>
#include <PvaAllocator.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>
#include <nvcv/TensorData.h>
#include <nvcv/TensorShape.h>

#include <cstdint>
#include <functional>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_depthtospace_gsdf_dev);
CUPVA_DECLARE_EXECUTABLE(pva_depthtospace_rdf_b2_dev);
CUPVA_DECLARE_EXECUTABLE(pva_depthtospace_rdf_b3_dev);
CUPVA_DECLARE_EXECUTABLE(pva_depthtospace_rdf_b4_dev);

namespace pva_solutions::priv {

/*
 * Geometry of the RDF path's dataflows, derived once so the path selection and the dataflow
 * configuration cannot disagree about what the dataflows would actually be.
 */
struct RdfGeometry
{
    /* Flattened input row count r = c * H + h, i.e. the raster height of one row offset. */
    int32_t flatRows;
    /* Tile counts of the input raster, bounded by RasterDataFlow constraint 7. */
    int32_t niterX;
    int32_t niterY;
    /* Full VMEM output tile the kernel fills, before clamping to the output surface. */
    int32_t outTileW;
    /* Tile dimensions the output dataflow is configured with. */
    int32_t dstTileW;
    int32_t dstTileH;
    /* Line pitch of the output surface, bounded by constraint 8. */
    int64_t outLinePitch;
    /* The tensor's row stride, not the width; it reaches only the DMA engine. */
    int64_t srcLinePitch;
    /* Fixed base offset separating the blkSz * blkSz input dataflows. Empty when the raster
     * cannot express the layout at all -- see DeriveRdfGeometry(). */
    std::optional<int64_t> chanGroupStride;
};

namespace {

/* Input tile width, from the device header's own rule, so host and device cannot drift. */
constexpr int32_t TileWForBlkSz(int32_t blkSz)
{
    return TILE_W_FOR_BLK(blkSz);
}

constexpr int32_t kTileH{32};

/* DMA limit: at most 256 tiles per raster dimension (RasterDataFlow constraint 7). */
constexpr int32_t kMaxTilesPerDim{256};

/* DMA limit: a surface line pitch is a uint16 (RasterDataFlow constraint 8). */
constexpr int32_t kMaxLinePitch{65535};

/* DMA limit: surface width and height are 16-bit device fields, and the output surface is
 * blkSz times the input in each, so the output is what binds. */
constexpr int32_t kMaxSurfaceExtent{65535};

/*
 * Address limit, on both paths. cupvaGSDFUpdateOffsets1D does not range-check the int32_t
 * tile offsets the GSDF kernel builds, so overrunning them is undefined behaviour rather
 * than an exception.
 */
constexpr int64_t kMaxAddressableOffsetExclusive{2147483648};

/* Caller-supplied shape[] and strides[] are int64_t; bounding each to int32_t range keeps the
 * pairwise products SelectPath() forms afterwards well inside int64. */
constexpr int64_t kMaxRequirementFieldInclusive{kMaxAddressableOffsetExclusive - 1};

/* Largest supported block size; sets the derived limits asserted below. */
constexpr int32_t kMaxBlkSz{4};

/* Host/device tile-geometry invariants: a mismatch would make the DMA move a different
 * tile than the kernel indexes -- silently, and only on some configs. */
static_assert(kTileH == TILE_H, "host kTileH must match device TILE_H");
/* The invariant the RDF path rests on: an output tile of at most 256 columns cannot need
 * more than 255 columns of trailing padding, at any width and block size. */
static_assert(TileWForBlkSz(2) * 2 <= 256, "output tile must fit the 8-bit DMA pad field");
static_assert(TileWForBlkSz(3) * 3 <= 256, "output tile must fit the 8-bit DMA pad field");
static_assert(TileWForBlkSz(4) * 4 <= 256, "output tile must fit the 8-bit DMA pad field");
static_assert(TileWForBlkSz(2) % TRANSP_ROWS == 0, "tile width must be a whole transposed-access block");
static_assert(TileWForBlkSz(3) % TRANSP_ROWS == 0, "tile width must be a whole transposed-access block");
static_assert(TileWForBlkSz(4) % TRANSP_ROWS == 0, "tile width must be a whole transposed-access block");
static_assert(kTileH % TRANSP_ROWS == 0, "TILE_H must be a whole number of 32-row transposed-access blocks");
static_assert(kMaxBlkSz * TileWForBlkSz(kMaxBlkSz) <= 4096, "output tile row must stay within the bank-C budget");

RdfGeometry DeriveRdfGeometry(NVCVTensorRequirements const &reqs, int32_t blkSz)
{
    int32_t const height     = static_cast<int32_t>(reqs.shape[1]);
    int32_t const width      = static_cast<int32_t>(reqs.shape[2]);
    int32_t const outChannel = static_cast<int32_t>(reqs.shape[0]) / (blkSz * blkSz);
    int32_t const outWidth   = width * blkSz;

    RdfGeometry geom{};

    /* Flattened input row index r = c * H + h covers Cout * H rows of W bytes; each row offset
     * turns those into Cout * H output rows of W * B bytes. */
    geom.flatRows = outChannel * height;

    int32_t const tileW = TileWForBlkSz(blkSz);

    geom.niterX = (width + tileW - 1) / tileW;
    geom.niterY = (geom.flatRows + kTileH - 1) / kTileH;

    /* dst() cannot grow its ROI past the surface, so the output tile is clamped to it while
     * tileArena declares the full VMEM tile the kernel fills. */
    geom.outTileW = tileW * blkSz;
    geom.dstTileW = (geom.outTileW < outWidth) ? geom.outTileW : outWidth;
    geom.dstTileH = (kTileH < geom.flatRows) ? kTileH : geom.flatRows;

    /* The output dataflow views the surface as blkSz * outWidth-wide rows. */
    geom.outLinePitch = static_cast<int64_t>(outWidth) * blkSz;

    /* src() must be handed the row stride, not the width: on a padded tensor every row
     * after the first would otherwise be read from the wrong offset. */
    geom.srcLinePitch = reqs.strides[1];

    /*
     * The raster flattens a channel group's channels into ONE raster, so it can express the
     * channel stride only when a group holds a single channel or the stride is exactly height row
     * pitches. Anything else walks inter-channel padding as image rows -- inside the
     * allocation, so silently wrong rather than a fault. Left empty routes the shape to GSDF.
     */
    int64_t const chanStride = reqs.strides[0];
    if ((outChannel == 1) || (chanStride == static_cast<int64_t>(height) * geom.srcLinePitch))
    {
        geom.chanGroupStride = static_cast<int64_t>(outChannel) * chanStride;
    }

    return geom;
}

/* SelectPath()'s answer; the geometry is meaningful only when the path is Rdf. */
struct PathDecision
{
    DepthToSpace::DataFlowPath path;
    RdfGeometry geom;
};

/*
 * The whole layout decision, in one pass. The fields are bounded on their raw int64 values
 * first, so nothing after that narrows or multiplies a field not yet shown to fit; rejections
 * are ordered most-specific-first so a caller is told the one property to change. What survives
 * is exactly what the gather/scatter path can carry, so routing is only whether the faster
 * raster path can carry it too.
 */
PathDecision SelectPath(NVCVTensorRequirements const &reqs, int32_t blkSz)
{
    for (int32_t dim = 0; dim < 3; ++dim)
    {
        if ((reqs.shape[dim] < 1) || (reqs.shape[dim] > kMaxRequirementFieldInclusive))
        {
            throw nvcv::Exception(
                nvcv::Status::ERROR_INVALID_ARGUMENT,
                "Input tensor shape[0..2] and strides[0..1] must each be positive and below 2147483648.");
        }
    }

    /* Neither dataflow can skip elements within a row: both walk it a byte at a time. */
    if (reqs.strides[2] != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input tensor strides[2] must be 1; a non-unit element stride is not supported.");
    }

    if ((reqs.strides[0] < 1) || (reqs.strides[0] > kMaxRequirementFieldInclusive) || (reqs.strides[1] < 1) ||
        (reqs.strides[1] > kMaxRequirementFieldInclusive))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input tensor shape[0..2] and strides[0..1] must each be positive and below 2147483648.");
    }

    /* A row pitch below the row width would make consecutive rows overlap. */
    if (reqs.strides[1] < reqs.shape[2])
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Input tensor strides[1] must be at least shape[2]; rows may be padded but must not overlap.");
    }

    /* The same rule one dimension out; neither path fails on an overlapping channel pitch, both
     * read the wrong bytes silently. */
    if (reqs.strides[0] < (reqs.shape[1] * reqs.strides[1]))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Input tensor strides[0] must be at least shape[1] * strides[1]; channels may be padded but "
            "must not overlap.");
    }

    /* The span both paths reach over. The no-overlap rules above put the tightly packed output
     * inside it too, and the raster path offsets in 64 bits so nothing else bounds it. */
    if ((reqs.shape[0] * reqs.strides[0]) >= kMaxAddressableOffsetExclusive)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input tensor is too large: shape[0] * strides[0] must be below 2147483648.");
    }

    /* A surface line pitch is a 16-bit DMA field on both paths. */
    if (reqs.strides[1] > kMaxLinePitch)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input tensor strides[1] must not exceed 65535; the row pitch is a 16-bit DMA field.");
    }

    /* The output surface is blkSz times the input in each dimension, in 16-bit fields. */
    if ((reqs.shape[2] * blkSz) > kMaxSurfaceExtent)
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Input tensor shape[2] * blkSz must not exceed 65535; the output row pitch is a 16-bit DMA field.");
    }

    if ((reqs.shape[1] * blkSz) > kMaxSurfaceExtent)
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Input tensor shape[1] * blkSz must not exceed 65535; the output surface height is a 16-bit "
            "DMA field.");
    }

    if ((reqs.shape[0] % (blkSz * blkSz)) != 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input tensor shape[0] must be a multiple of blkSz * blkSz.");
    }

    /* Safe only here: the checks above bound every field the geometry narrows and every product
     * it forms. */
    RdfGeometry const geom = DeriveRdfGeometry(reqs, blkSz);

    /*
     * The three RasterDataFlow limits the raster cannot work around; everything else it needs is
     * already established above. A false answer routes to the gather/scatter path, not an error.
     */
    bool const rasterServes = geom.chanGroupStride.has_value()         /* expressible channel stride */
                              && (geom.niterY <= kMaxTilesPerDim)      /* constraint 7 */
                              && (geom.outLinePitch <= kMaxLinePitch); /* constraint 8 */

    return {rasterServes ? DepthToSpace::DataFlowPath::Rdf : DepthToSpace::DataFlowPath::Gsdf, geom};
}

NVCVTensorRequirements DeriveOutputRequirements(NVCVTensorRequirements const &reqs, int32_t blkSz)
{
    int64_t outShape[] = {reqs.shape[0] / (blkSz * blkSz), reqs.shape[1] * blkSz, reqs.shape[2] * blkSz};

    NVCVTensorRequirements outReqs;
    if (nvcvTensorCalcRequirementsPva(3, outShape, reqs.dtype, reqs.layout, 0, 0, &outReqs) != NVCV_SUCCESS)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Failed to derive the output tensor requirements.");
    }

    return outReqs;
}

std::map<DepthToSpace::ExecDataKey, IOperator::ExecutableData> createRdfExecDataMap()
{
    return {
        {{2}, {CUPVA_EXECUTABLE_DATA(pva_depthtospace_rdf_b2_dev), CUPVA_EXECUTABLE_SIZE(pva_depthtospace_rdf_b2_dev)}},
        {{3}, {CUPVA_EXECUTABLE_DATA(pva_depthtospace_rdf_b3_dev), CUPVA_EXECUTABLE_SIZE(pva_depthtospace_rdf_b3_dev)}},
        {{4}, {CUPVA_EXECUTABLE_DATA(pva_depthtospace_rdf_b4_dev), CUPVA_EXECUTABLE_SIZE(pva_depthtospace_rdf_b4_dev)}},
    };
}

} // namespace

// ceil<2>()  is used for output pointer of DLUT, needs to be 2 byte aligned
// ceil<64>() is used for table pointer of DLUT, needs to be 64 byte aligned
template<int N>
int ceil(int input)
{
    static_assert(N > 1);
    return ((input + N - 1) / N) * N;
}

std::mutex DepthToSpace::s_execMapMutex;

std::map<DepthToSpace::ExecKey, std::weak_ptr<cupva::Executable>> DepthToSpace::s_rdfExecutables;

std::map<DepthToSpace::GsdfExecKey, std::weak_ptr<cupva::Executable>> DepthToSpace::s_gsdfExecutables;

std::shared_ptr<cupva::Executable> DepthToSpace::GetRdfExec(ExecDataKey execDataKey)
{
    static auto execDataMap = createRdfExecDataMap();

    auto execDataIt = execDataMap.find(execDataKey);
    if (execDataIt != execDataMap.end())
    {
        ExecKey execKey = std::tuple_cat(execDataKey, std::make_tuple(cupva::Context::GetCurrent()));
        return IOperator::GetExecutable<ExecKey>(execDataIt->second, execKey, s_rdfExecutables, s_execMapMutex);
    }
    else
    {
        /* The constructor admits only the block sizes createRdfExecDataMap() holds, so a miss is
         * the two disagreeing rather than anything the caller passed. */
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL,
                              "No DepthToSpace raster executable is built for block size %d", std::get<0>(execDataKey));
    }
}

std::shared_ptr<cupva::Executable> DepthToSpace::GetGsdfExec()
{
    static IOperator::ExecutableData sDepthToSpaceExecData = {CUPVA_EXECUTABLE_DATA(pva_depthtospace_gsdf_dev),
                                                              CUPVA_EXECUTABLE_SIZE(pva_depthtospace_gsdf_dev)};

    GsdfExecKey execKey = std::make_tuple(cupva::Context::GetCurrent());
    return IOperator::GetExecutable<GsdfExecKey>(sDepthToSpaceExecData, execKey, s_gsdfExecutables, s_execMapMutex);
}

DepthToSpace::DepthToSpace(NVCVTensorRequirements const *const tensorRequirements, int32_t blkSz)
    : m_path{DataFlowPath::Gsdf}
    , m_blkSz{blkSz}
    , m_srcOp{cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM)}
    , m_dstOp{cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM)}
    , m_prog{}
    , m_mappingTblDevPtr{nullptr}
{
    /* Parameter contract, independent of the tensor's layout and extents -- SelectPath()'s question. */
    if ((blkSz != 2) && (blkSz != 3) && (blkSz != 4))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Block size must be 2, 3 or 4.");
    }

    m_inputTensorReqs = *tensorRequirements;

    if (m_inputTensorReqs.rank != 3)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input tensor must have rank 3.");
    }

    NVCVTensorLayout refLayout;
    nvcvTensorLayoutMake("CHW", &refLayout);
    if (nvcvTensorLayoutCompare(m_inputTensorReqs.layout, refLayout) != 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input tensor must have CHW layout.");
    }

    if ((m_inputTensorReqs.dtype != NVCV_DATA_TYPE_U8) && (m_inputTensorReqs.dtype != NVCV_DATA_TYPE_S8))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input tensor contains an unsupported data type; expected U8 or S8.");
    }

    /* Throws when no path can carry the shape; the message names the one field to change. */
    PathDecision const decision = SelectPath(m_inputTensorReqs, blkSz);

    m_outputTensorReqs = DeriveOutputRequirements(m_inputTensorReqs, blkSz);
    m_path             = decision.path;

    if (m_path == DataFlowPath::Rdf)
    {
        CreateRdf(decision.geom);
    }
    else
    {
        CreateGsdf();
    }
}

/*
 * RasterDataFlow fast path: one program per output row offset, all submitted together. The
 * decomposition is described in doc/impl/operator/depthtospace.rst.
 */
void DepthToSpace::CreateRdf(RdfGeometry const &geom)
{
    int32_t const width = static_cast<int32_t>(m_inputTensorReqs.shape[2]);

    int32_t const numChanGroups = m_blkSz * m_blkSz;
    int32_t const outWidth      = width * m_blkSz;

    m_exec = GetRdfExec(std::make_tuple(m_blkSz));

    int32_t const numPrograms = m_blkSz;

    /* Grow the ROI to a whole tile when the tile is larger than the surface, as image_blend
     * does, so the RDF constraint tile() <= roi() holds for tiny inputs. */
    int32_t const tileW   = TileWForBlkSz(m_blkSz);
    int32_t const srcRoiW = (tileW > width) ? tileW : width;
    int32_t const srcRoiH = (kTileH > geom.flatRows) ? kTileH : geom.flatRows;

    m_srcChanGroupOps.reserve(numChanGroups);
    m_dstRowOfsOps.reserve(m_blkSz);
    m_progs.resize(numPrograms);

    /* Eligibility is the same object as the value, so .value() cannot throw: a layout the
     * raster cannot reach was routed to GSDF. */
    int64_t const chanGroupStride = geom.chanGroupStride.value();

    for (int32_t by = 0; by < numPrograms; by++)
    {
        auto prog = CmdProgram::Create(*m_exec.get());

        prog["niterX"] = geom.niterX;
        prog["niterY"] = geom.niterY;

        /*
         * Input: one RDF per channel group. All blkSz share one (W x flatRows) geometry and differ
         * only in their group's fixed base offset, so they alias onto one handler and the kernel
         * recovers the siblings by pointer offset.
         */
        for (int32_t bx = 0; bx < m_blkSz; bx++)
        {
            int32_t const g = by * m_blkSz + bx;

            /* Offset copies share m_srcOp's base, so Submit rebases all of them at once. */
            m_srcChanGroupOps.push_back(m_srcOp + g * chanGroupStride);

            std::string const bufName = "srcBuf" + std::to_string(bx);

            prog.addDataFlowHead<RasterDataFlow>()
                .handler(prog["srcHandler"])
                .src(m_srcChanGroupOps.back(), width, geom.flatRows, geom.srcLinePitch)
                .tileBuffer(prog[bufName.c_str()].ptr<uint8_t>())
                .tile(tileW, kTileH)
                .roi(0, 0, srcRoiW, srcRoiH)
                .transpose(TRANS_MODE_1);
        }

        /*
         * Output: viewing the output as a surface of blkSz*outWidth-wide rows makes row offset by's
         * rows those of a plain raster based at by*outWidth, so that stride lives entirely in the
         * dataflow's line pitch. dst() cannot grow its ROI past the surface, so the tile is clamped
         * and tileArena declares the full VMEM tile.
         */
        m_dstRowOfsOps.push_back(m_dstOp + static_cast<int64_t>(by) * outWidth);

        prog.addDataFlowHead<RasterDataFlow>()
            .handler(prog["dstHandler"])
            .tileBuffer(prog["dstBuf"].ptr<uint8_t>())
            .dst(m_dstRowOfsOps.back(), outWidth, geom.flatRows, geom.outLinePitch)
            .tile(geom.dstTileW, geom.dstTileH)
            .tileArena(geom.outTileW, kTileH)
            .transpose(TRANS_MODE_1);

        prog.compileDataFlows();
        cupva::ConvertTo(std::move(prog), m_progs[by]);
    }
}

/*
 * GatherScatterDataFlow general path. Tiles come from the prime factorization of the extents,
 * so every tile is whole and no DMA padding is needed -- which is why it serves what the RDF
 * path cannot. The rearrangement is done by the DLUT through a host-built mapping table.
 */
void DepthToSpace::CreateGsdf()
{
    int32_t const channel = static_cast<int32_t>(m_inputTensorReqs.shape[0]);
    int32_t const height  = static_cast<int32_t>(m_inputTensorReqs.shape[1]);
    int32_t const width   = static_cast<int32_t>(m_inputTensorReqs.shape[2]);
    NVCVDataType const dtype{m_inputTensorReqs.dtype};

    int32_t const blkSz = m_blkSz;

    /* ADDRESS strides of the input surface, as opposed to its extents. SelectPath() has
     * already bounded both, so a padded row or channel needs nothing else here. */
    int32_t const inputRowStride  = static_cast<int32_t>(m_inputTensorReqs.strides[1]);
    int32_t const inputChanStride = static_cast<int32_t>(m_inputTensorReqs.strides[0]);

    m_exec    = GetGsdfExec();
    auto prog = CmdProgram::Create(*m_exec.get());

    // Firstly try to use the pre-calculated tile size.
    std::optional<std::pair<int32_t, int32_t>> tileSizePair =
        pva_solutions::priv::preCalTileSz(width, height, blkSz, kMaxTileSz);
    if (!tileSizePair)
    {
        // If not, then try to do the auto tiling.
        tileSizePair = pva_solutions::priv::enumerateTileSz(width, height, blkSz, kMaxTileSz);
    }

    if (!tileSizePair)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Automatic tiling fails. Consider to modify the heuristic logic");
    }

    const int tileX{tileSizePair.value().first};
    const int tileY{tileSizePair.value().second};

    // For GEN2:  blk_sz=2, numGsdfHdl=4, with tile size to be 1.
    //            blk_sz=3, numGsdfHdl=3, with tile size to be 3.
    //            blk_sz=4, numGsdfHdl=1, since numGsdfHdl=2 will result in lack of DMA descriptors.
    // For GEN3:  numGsdfHdl is always set to 1.
    const int numGsdfHdl = [](int32_t blockSz) {
        if (GetHardwareInfo().pvaGen == PVA_GEN2)
        {
            if (blockSz == 2)
            {
                return 4;
            }
            else if (blockSz == 3)
            {
                return 3;
            }
            else // if (blockSz == 4)
            {
                return 1;
            }
        }
        else
        {
            return 1;
        }
    }(blkSz);

    m_mappingTblDevPtr = static_cast<uint16_t *>(cupva::mem::Alloc(blkSz * blkSz * tileY * tileX * sizeof(uint16_t)));
    if (m_mappingTblDevPtr == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_OUT_OF_MEMORY,
                              "Cannot allocate the memory using cupva::mem::Alloc()");
    }

    uint16_t *mappingTblDevPtrHost{nullptr};
    CupvaMemGetHostPointer((void **)&mappingTblDevPtrHost, (void *)m_mappingTblDevPtr);

    if (tileX == 1 && tileY == 1)
    {
        // In this case, the mapping table is an iota sequence: 0, 1, 2,...,((blkSz^2)*(tileY)*(tileX)-1)
        std::iota(mappingTblDevPtrHost, mappingTblDevPtrHost + blkSz * blkSz * tileY * tileX, 0);
    }
    else
    {
        // The mapping table can be seen as a DepthToSpace operation on a (blkSz^2)*(tileY)*(tileX) tensor within
        // the current implementation. Then reuse the generation of the reference output, whose only difference
        // is using uint16_t as the element type.
        const int32_t rank = 5;
        const std::vector<int64_t> shape{blkSz, blkSz, 1, tileY, tileX};
        const int numel = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int64_t>());
        std::vector<uint16_t> baseBuf(numel);
        std::iota(baseBuf.begin(), baseBuf.end(), 0);

        NVCVTensorBuffer inBuf;
        // construct the stride from the shape vector by assuming the initial layout is contiguous
        int32_t strideBytes;
        nvcvDataTypeGetStrideBytes(dtype, &strideBytes);
        inBuf.strided.strides[rank - 1] = strideBytes; // in the unit of element size
        for (int i = rank - 2; i >= 0; i--)
        {
            inBuf.strided.strides[i] = shape[i + 1] * inBuf.strided.strides[i + 1];
        }
        inBuf.strided.basePtr = (NVCVByte *)baseBuf.data();

        NVCVTensorLayout layout;
        // need to permute the dimension from [0, 1, 2, 3, 4] to [2, 3, 0, 4, 1]
        nvcvTensorLayoutMake("01234", &layout);

        // uint16_t is large enough to hold a maximum DLUT's size of ~60,000 elements
        NVCVTensorData inData{NVCV_DATA_TYPE_U16,      layout, rank, {blkSz, blkSz, 1, tileY, tileX},
                              NVCV_TENSOR_BUFFER_NONE, inBuf};

        NVCVTensorLayout newTensorLayout;
        nvcvTensorLayoutMake("23041", &newTensorLayout);

        std::vector<int64_t> newShape(rank), newStrides(rank);
        // there is no NVCV API to permute the shape and strides simultaneously
        nvcvTensorShapePermute(layout, inData.shape, newTensorLayout, newShape.data());
        nvcvTensorShapePermute(layout, inBuf.strided.strides, newTensorLayout, newStrides.data());

        pva_solutions::utils::MakeContiguous<uint16_t>(reinterpret_cast<uint16_t *>(inBuf.strided.basePtr), newShape,
                                                       newStrides, mappingTblDevPtrHost);
    }

    int8_t *scratch_mems[3] = {prog["scratch_a"].ptr<int8_t>(), prog["scratch_b"].ptr<int8_t>(),
                               prog["scratch_c"].ptr<int8_t>()};

    int8_t buffer_info_a[sizeof(depth_to_space_buffer_info_t)];
    depth_to_space_buffer_info_t *buffer_info = (depth_to_space_buffer_info_t *)buffer_info_a;
    buffer_info->tbl_id                       = 2;
    buffer_info->tbl_ofst                     = 0;
    buffer_info->tbl_sz                       = (tileX * blkSz) * (tileY * blkSz) * sizeof(uint16_t);

    buffer_info->dut_in_id   = 0;
    buffer_info->dut_in_ofst = 0;
    buffer_info->dut_in_sz   = 2 * ceil<64>(blkSz * blkSz * tileX * tileY); // 2 is for pingpong

    buffer_info->dut_out_id   = 1;
    buffer_info->dut_out_ofst = 0;
    buffer_info->dut_out_sz   = 2 * ceil<2>(blkSz * blkSz * tileX * tileY); // the first 2 is for pingpong

    prog["buffer_info"].set(buffer_info_a, sizeof(depth_to_space_buffer_info_t));

    int8_t vpu_param_a[sizeof(depth_to_space_host2dev_param_t)];
    depth_to_space_host2dev_param_t *vpu_param = (depth_to_space_host2dev_param_t *)vpu_param_a;
    const int ow{width * blkSz};
    const int oh{height * blkSz};
    const int oc{channel / (blkSz * blkSz)};

    vpu_param->tile_x     = tileX;
    vpu_param->tile_y     = tileY;
    vpu_param->blk_sz     = blkSz;
    vpu_param->iw         = width;
    vpu_param->ih         = height;
    vpu_param->ic         = channel;
    vpu_param->ow         = ow;
    vpu_param->oh         = oh;
    vpu_param->oc         = oc;
    vpu_param->num_tiles  = oc * ((width / tileX) * (height / tileY));
    vpu_param->numGsdfHdl = numGsdfHdl;
    /* iw/ih above are the extents the tile grid is counted in; these two are the strides
     * the surface is walked with. They coincide only on a tightly packed tensor. */
    vpu_param->in_row_stride  = inputRowStride;
    vpu_param->in_chan_stride = inputChanStride;

    prog["vpu_param"].set(vpu_param_a, sizeof(depth_to_space_host2dev_param_t));

    // The mapping table is of 1D, but configured in 2D style in order to not exceed some max values
    // of the DMA dataflow's fields.
    const int32_t linePitch{tileX * blkSz};
    prog.addDataFlowHead<RasterDataFlow>()
        .handler(prog["tblHdl"])
        .tile(tileX * blkSz, tileY * blkSz)
        .tileBuffer(scratch_mems[buffer_info->tbl_id] + buffer_info->tbl_ofst)
        .src(m_mappingTblDevPtr, tileX * blkSz, tileY * blkSz, linePitch);

    // Keep the vectors local but store Parameters directly
    std::vector<cupva::Parameter> srcHdlVec;
    std::vector<int> blksPerGsdfVec;

    // Store Parameters directly instead of references
    if (numGsdfHdl == 4)
    {
        srcHdlVec      = {prog["srcHdlT16"], prog["srcHdlT8"], prog["srcHdlT4_0"], prog["srcHdlT4_1"]};
        blksPerGsdfVec = {1, 1, 1, 1};
    }
    else if (numGsdfHdl == 3)
    {
        srcHdlVec      = {prog["srcHdlT8"], prog["srcHdlT4_0"], prog["srcHdlT4_1"]};
        blksPerGsdfVec = {3, 3, 3};
    }
    else
    {
        srcHdlVec      = {prog["srcHdlT16"]};
        blksPerGsdfVec = {blkSz * blkSz};
    }

    /* The surface line pitch, i.e. the tensor's row stride and not its width. */
    const int32_t inputLinePitch{inputRowStride};
    for (size_t i = 0; i < srcHdlVec.size(); i++)
    {
        prog.addDataFlowHead<GatherScatterDataFlow>()
            .handler(srcHdlVec[i])
            .numTilesPerTrigger(blksPerGsdfVec[i])
            .tile(tileX, tileY)
            .src(m_srcOp, width, height, inputLinePitch);
    }

    const int32_t outputLinePitch{ow};
    prog.addDataFlowHead<GatherScatterDataFlow>()
        .bufferType(mem::BufferType::RAW)
        .handler(prog["dstHdl"])
        .numTilesPerTrigger(1)
        .tile(tileX * blkSz, tileY * blkSz)
        .dst(m_dstOp, ow, oh, outputLinePitch);

    prog.compileDataFlows();

    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename S>
void DepthToSpace::SubmitRdf(S stream)
{
    /* CreateRdf() sizes m_progs to the block size, which the constructor admits only as 2, 3
     * or 4, so the count never exceeds kMaxBlkSz and the arrays are sized at compile time
     * rather than allocated on every submit. */
    int32_t const numPrograms = static_cast<int32_t>(m_progs.size());

    cupvaCmd_t const *submitCmds[kMaxBlkSz] = {};
    cupvaCmdStatus_t cmdStatus[kMaxBlkSz]   = {};

    for (int32_t i = 0; i < numPrograms; i++)
    {
        CUPVA_CHECK_ERROR(CupvaCmdProgramUpdateDataFlows(&m_progs[i]));
        submitCmds[i] = &m_progs[i];
    }

    /* One submit for all programs: the row offsets write disjoint output rows, so they may run
     * in any order, and the per-submit overhead is paid once per operator call. */
    if constexpr (std::is_same_v<S, cupvaStream_t>)
    {
        CUPVA_CHECK_ERROR(CupvaStreamSubmit(stream, submitCmds, cmdStatus, numPrograms, CUPVA_IN_ORDER, -1, -1));
    }
    else if constexpr (std::is_same_v<S, cudaStream_t>)
    {
        if constexpr (kCupvaHasCudaStreamSupport)
        {
            CUPVA_CHECK_ERROR(
                CupvaCudaStreamSubmit(stream, submitCmds, cmdStatus, numPrograms, CUPVA_IN_ORDER, -1, -1));
        }
        else
        {
            ThrowCudaStreamNotSupported();
        }
    }
    else
    {
        static_assert(!sizeof(S), "Unsupported stream type for DepthToSpace operator");
    }
}

template<typename S>
void DepthToSpace::SubmitGsdf(S stream)
{
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
        static_assert(!sizeof(S), "Unsupported stream type for DepthToSpace operator");
    }
}

template<typename S>
void DepthToSpace::operator()(S stream, NVCVTensorHandle in, NVCVTensorHandle out)
{
    nvcv::Tensor input = nvcv::TensorWrapHandle(in);
    auto inTensorData  = input.exportData<nvcv::TensorDataStrided>();

    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_inputTensorReqs, inTensorData->cdata()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input tensor must have the same requirements that was used to create the operator.");
    }

    nvcv::Tensor output = nvcv::TensorWrapHandle(out);
    auto outTensorData  = output.exportData<nvcv::TensorDataStrided>();

    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_outputTensorReqs, outTensorData->cdata()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output tensor must have the requirements derived when the operator was created.");
    }

    if ((static_cast<void *>(inTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(outTensorData->basePtr()) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor base pointers must not be NULL");
    }

    /* Rebase the ROOT pointers only: every m_srcChanGroupOps[]/m_dstRowOfsOps[] entry shares
     * these bases and keeps its own offset. */
    m_srcOp.setBase(static_cast<uint8_t *>(static_cast<void *>(inTensorData->basePtr())));
    m_dstOp.setBase(static_cast<uint8_t *>(static_cast<void *>(outTensorData->basePtr())));

    if (m_path == DataFlowPath::Rdf)
    {
        SubmitRdf(stream);
    }
    else
    {
        SubmitGsdf(stream);
    }
}

// Explicit template instantiation
template void DepthToSpace::operator()<cupvaStream_t>(cupvaStream_t, NVCVTensorHandle, NVCVTensorHandle);
template void DepthToSpace::operator()<cudaStream_t>(cudaStream_t, NVCVTensorHandle, NVCVTensorHandle);

DepthToSpace::~DepthToSpace()
{
    if (m_path == DataFlowPath::Rdf)
    {
        for (auto &prog : m_progs)
        {
            CupvaCmdDestroy(&prog);
        }
    }
    else
    {
        CupvaCmdDestroy(&m_prog);
        cupva::mem::Free(m_mappingTblDevPtr);
    }
}

auto preCalTileSz(int32_t width, int32_t height, int32_t blkSz, int32_t maxTileSz)
    -> std::optional<std::pair<int32_t, int32_t>>
{
    int32_t tileX{0}, tileY{0};

    if (blkSz * blkSz * height * width <= maxTileSz)
    {
        // no tiling needed if the image size is small enough
        tileX = width;
        tileY = height;
    }
    else if ((blkSz == 2 && height == 256 && width == 256) || (blkSz == 2 && height == 512 && width == 512))
    {
        tileX = 64;
        tileY = 64;
    }
    else if (blkSz == 2 && height == 1024 && width == 1024)
    {
        tileX = 128;
        tileY = 64;
    }
    else if (blkSz == 2 && height == 128 && width == 128)
    {
        tileX = 32;
        tileY = 32;
    }
    else if (blkSz == 3 && height == 368 && width == 640)
    {
        tileX = 640;
        tileY = 8;
    }

    if (tileX == 0 || tileY == 0)
    {
        return {};
    }
    else
    {
        return std::make_pair(tileX, tileY);
    }
}

} // namespace pva_solutions::priv

NVCVStatus pvaDepthToSpaceCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements const *const tensorRequirements,
                                 int32_t blkSz)
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
            reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::DepthToSpace(tensorRequirements, blkSz));
    });
}

template<typename StreamType>
NVCVStatus DepthToSpaceSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle in,
                                  NVCVTensorHandle out)
{
    return nvcv::ProtectCall(
        [&] { pva_solutions::priv::ToDynamicRef<pva_solutions::priv::DepthToSpace>(handle)(stream, in, out); });
}

// Explicit template instantiation for supported stream types
template NVCVStatus DepthToSpaceSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle,
                                                          NVCVTensorHandle);
template NVCVStatus DepthToSpaceSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle,
                                                         NVCVTensorHandle);

// C++ overload implementations
NVCVStatus pvaDepthToSpaceSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle in,
                                 NVCVTensorHandle out)
{
    return DepthToSpaceSubmitImpl(handle, stream, in, out);
}

NVCVStatus pvaDepthToSpaceSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                 NVCVTensorHandle out)
{
    return DepthToSpaceSubmitImpl(handle, stream, in, out);
}
