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

#include "OpDopplerFFT.hpp"

#include "device/doppler_fft_param.h"

#include <IOperator.hpp>
#include <RadarUtils.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <cupva_host_nonsafety.hpp>

#include <OpDopplerFFT.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <cupva_types.h>
#include <detail/cupva_host_types_detail.h>

#include <algorithm>
#include <cmath>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_doppler_fft_dev);
CUPVA_DECLARE_EXECUTABLE(pva_doppler_fft_dev_direct);

namespace pva_solutions::priv {

template<typename TI, typename TO>
std::mutex DopplerFFT<TI, TO>::s_execMapMutex;

template<typename TI, typename TO>
std::map<typename DopplerFFT<TI, TO>::ExecKey, std::weak_ptr<cupva::Executable>>
    DopplerFFT<TI, TO>::s_dopplerFFTExecutables;

// RANGE_DOPPLER_RX output runs the RxPacked kernel; other layouts run the direct kernel.
template<typename TI, typename TO>
std::shared_ptr<cupva::Executable> DopplerFFT<TI, TO>::GetDopplerFFTExec(bool isRxPackedLayout)
{
    IOperator::ExecutableData execData;
    if (isRxPackedLayout)
    {
        execData = {CUPVA_EXECUTABLE_DATA(pva_doppler_fft_dev), CUPVA_EXECUTABLE_SIZE(pva_doppler_fft_dev)};
    }
    else
    {
        execData = {CUPVA_EXECUTABLE_DATA(pva_doppler_fft_dev_direct),
                    CUPVA_EXECUTABLE_SIZE(pva_doppler_fft_dev_direct)};
    }

    ExecKey execKey = std::make_tuple(isRxPackedLayout, cupva::Context::GetCurrent());
    return IOperator::GetExecutable<ExecKey>(execData, execKey, s_dopplerFFTExecutables, s_execMapMutex);
}

namespace {

// Words one ping-pong half of dstBuf must hold: the output tile or the FFT scratch, whichever is larger.
int32_t DstBufHalfSize(int32_t chirps, int32_t groupsPerTile, int32_t outTileHeight, int32_t outTileLinePitch,
                       int32_t dstGroupStride)
{
    const int32_t tilePitch  = IN_TILE_WIDTH * groupsPerTile;
    const int32_t fftScratch = (groupsPerTile - 1) * dstGroupStride + (chirps - 1) * tilePitch + IN_TILE_WIDTH;
    const int32_t outTile    = outTileHeight * outTileLinePitch;

    // Keep the second half aligned to the 64-byte superbank for the DMA.
    return ((std::max(fftScratch, outTile) + 15) / 16) * 16;
}

constexpr int32_t kDstBufHalfBudget = OUT_TILE_WIDTH * OUT_TILE_HEIGHT;
constexpr int32_t kSrcBufHalfBudget = IN_TILE_WIDTH * IN_TILE_HEIGHT;

// Layout- and tile-dependent quantities, resolved once for the constructor and the transfer emitters.

/* What the input layout decides. */
struct InputPlan
{
    bool chirpInnermost; ///< RANGE_RX_DOPPLER: HCW[Nb, NofRx, Nr], chirps in W and bins in H.
    bool rxDopplerRange; ///< RX_DOPPLER_RANGE: HCW[NofRx, Nr, Nb].
    int32_t channels;
    int32_t chirps;
    int32_t rangeBins;
    int32_t tensorWidth;    ///< Row length in int32.
    int32_t imageLinePitch; ///< DRAM row stride in int32.
    int32_t rxStride;       ///< Distance between channels in int32.
};

/* What the output layout decides. */
struct OutputPlan
{
    bool rxPacked;       ///< RANGE_DOPPLER_RX: the L2 transpose path rather than the direct one.
    bool transposed;     ///< Anything but DOPPLER_RX_RANGE, whose tile is Doppler-major.
    bool rxRangeDoppler; ///< RX_RANGE_DOPPLER: channels are a plane stride, not part of the row.
    int32_t dopplerBins;
    int32_t rangeBins;
    int32_t tensorHeight;
    int32_t tensorWidth;
    int32_t imageLinePitch;
    int32_t rxStride;
};

/* Tile geometry, which both layouts and the group count decide together. */
struct TilePlan
{
    int32_t inWidth;
    int32_t inHeight;
    int32_t inLinePitch; ///< VMEM row stride, transpose-aligned where the kernel gathers lanes.
    int32_t inLaneOfst;  ///< Lane step of the transposed load; 0 where the load is contiguous.
    int32_t advSrcDim2;  ///< Advance between consecutive tiles along the range axis.
    int32_t outWidth;
    int32_t outHeight;
    int32_t outLinePitch;
    int32_t outLaneOfst;
    int32_t advDstDim1;
    int32_t dstGroupStride; ///< Where group g sits inside the output tile.
};

InputPlan ResolveInputPlan(PVADopplerFFTParams const &params, int64_t const *const shapeNCHW)
{
    InputPlan p{};
    if (params.inputLayout != PVA_FFT_LAYOUT_DOPPLER_RX_RANGE &&
        params.inputLayout != PVA_FFT_LAYOUT_RANGE_RX_DOPPLER && params.inputLayout != PVA_FFT_LAYOUT_RX_DOPPLER_RANGE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Unsupported input layout %d; DopplerFFT supports only "
                              "PVA_FFT_LAYOUT_DOPPLER_RX_RANGE (%d), PVA_FFT_LAYOUT_RANGE_RX_DOPPLER (%d) and "
                              "PVA_FFT_LAYOUT_RX_DOPPLER_RANGE (%d)",
                              static_cast<int32_t>(params.inputLayout),
                              static_cast<int32_t>(PVA_FFT_LAYOUT_DOPPLER_RX_RANGE),
                              static_cast<int32_t>(PVA_FFT_LAYOUT_RANGE_RX_DOPPLER),
                              static_cast<int32_t>(PVA_FFT_LAYOUT_RX_DOPPLER_RANGE));
    }

    p.chirpInnermost = (params.inputLayout == PVA_FFT_LAYOUT_RANGE_RX_DOPPLER);
    p.rxDopplerRange = (params.inputLayout == PVA_FFT_LAYOUT_RX_DOPPLER_RANGE);

    p.channels  = p.chirpInnermost ? shapeNCHW[1] : (p.rxDopplerRange ? shapeNCHW[2] : shapeNCHW[1]);
    p.chirps    = p.chirpInnermost ? shapeNCHW[3] : (p.rxDopplerRange ? shapeNCHW[1] : shapeNCHW[2]);
    p.rangeBins = p.chirpInnermost ? static_cast<int32_t>(shapeNCHW[2]) : static_cast<int32_t>(shapeNCHW[3]);

    p.tensorWidth = (p.chirpInnermost ? p.chirps : p.rangeBins) * COMPLEX_ELEMENT_NUM;

    p.imageLinePitch = p.rxDopplerRange ? p.tensorWidth : (p.channels * p.tensorWidth);
    p.rxStride       = p.rxDopplerRange ? (p.chirps * p.tensorWidth) : p.tensorWidth;
    if (p.chirpInnermost)
    {
        p.imageLinePitch = p.channels * p.tensorWidth;
        p.rxStride       = p.tensorWidth;
    }

    if (p.chirpInnermost)
    {
        // This input is a RasterDataFlow, whose line pitch is counted in elements.
        const int64_t srcLinePitchElems = static_cast<int64_t>(p.channels) * p.chirps * COMPLEX_ELEMENT_NUM;
        if (srcLinePitchElems > 65535)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "RANGE_RX_DOPPLER input needs NofRx * Nr * %d <= 65535 elements; got %ld",
                                  COMPLEX_ELEMENT_NUM, srcLinePitchElems);
        }
        if (p.rangeBins % TILE_BATCH != 0)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "RANGE_RX_DOPPLER input requires range bins to be a multiple of %d", TILE_BATCH);
        }
        if (params.outputLayout == PVA_FFT_LAYOUT_RANGE_DOPPLER_RX)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "RANGE_RX_DOPPLER input is not supported with RANGE_DOPPLER_RX output");
        }
    }

    return p;
}

OutputPlan ResolveOutputPlan(PVAFFTLayout const outputLayout, int64_t const *const shapeNCHW, InputPlan const &in)
{
    OutputPlan p{};
    int64_t outChannels = 0;

    p.rxPacked       = (outputLayout == PVA_FFT_LAYOUT_RANGE_DOPPLER_RX);
    p.transposed     = (outputLayout != PVA_FFT_LAYOUT_DOPPLER_RX_RANGE);
    p.rxRangeDoppler = (outputLayout == PVA_FFT_LAYOUT_RX_RANGE_DOPPLER);

    switch (outputLayout)
    {
    case PVA_FFT_LAYOUT_DOPPLER_RX_RANGE:
        p.dopplerBins  = shapeNCHW[2];
        p.rangeBins    = shapeNCHW[3];
        p.tensorHeight = shapeNCHW[2];
        p.tensorWidth  = shapeNCHW[3] * COMPLEX_ELEMENT_NUM;
        outChannels    = shapeNCHW[1];
        break;
    case PVA_FFT_LAYOUT_RANGE_RX_DOPPLER:
        p.dopplerBins  = shapeNCHW[3];
        p.rangeBins    = shapeNCHW[2];
        p.tensorHeight = shapeNCHW[2];
        p.tensorWidth  = shapeNCHW[3] * COMPLEX_ELEMENT_NUM;
        outChannels    = shapeNCHW[1];
        break;
    case PVA_FFT_LAYOUT_RANGE_DOPPLER_RX:
        p.dopplerBins  = shapeNCHW[1];
        p.rangeBins    = shapeNCHW[2];
        p.tensorHeight = shapeNCHW[2];
        p.tensorWidth  = shapeNCHW[1] * COMPLEX_ELEMENT_NUM;
        outChannels    = shapeNCHW[3];
        break;
    case PVA_FFT_LAYOUT_RX_RANGE_DOPPLER:
        p.dopplerBins  = shapeNCHW[3];
        p.rangeBins    = shapeNCHW[1];
        p.tensorHeight = shapeNCHW[1];
        p.tensorWidth  = shapeNCHW[3] * COMPLEX_ELEMENT_NUM;
        outChannels    = shapeNCHW[2];
        break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Unsupported output layout %d; DopplerFFT supports only "
                              "PVA_FFT_LAYOUT_DOPPLER_RX_RANGE (%d), PVA_FFT_LAYOUT_RANGE_RX_DOPPLER (%d), "
                              "PVA_FFT_LAYOUT_RANGE_DOPPLER_RX (%d) and PVA_FFT_LAYOUT_RX_RANGE_DOPPLER (%d)",
                              static_cast<int32_t>(outputLayout), static_cast<int32_t>(PVA_FFT_LAYOUT_DOPPLER_RX_RANGE),
                              static_cast<int32_t>(PVA_FFT_LAYOUT_RANGE_RX_DOPPLER),
                              static_cast<int32_t>(PVA_FFT_LAYOUT_RANGE_DOPPLER_RX),
                              static_cast<int32_t>(PVA_FFT_LAYOUT_RX_RANGE_DOPPLER));
    }

    if (p.dopplerBins > in.chirps)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output Doppler bins can't be greater than %d",
                              in.chirps);
    }
    if (p.rangeBins != in.rangeBins)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output range bins must be equal to input range bins");
    }
    if (p.rxPacked)
    {
        if (in.channels != 8)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "RANGE_DOPPLER_RX layout only supports 8 Rx channels, got %d", in.channels);
        }
        if (in.chirps > 900)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "chirps must be less than or equal to 900");
        }
        if (in.chirps < 32)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "chirps must be greater than or equal to 32");
        }
    }
    // The output DMA strides by the input channel count.
    if (outChannels != in.channels)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output Rx channels %ld must match input Rx channels %d", outChannels, in.channels);
    }

    p.imageLinePitch = p.rxRangeDoppler ? p.tensorWidth : (in.channels * p.tensorWidth);
    p.rxStride       = p.rxRangeDoppler ? (p.rangeBins * p.tensorWidth) : p.tensorWidth;

    return p;
}

// Range-bin groups to pack into one tile (up to maxGroups), or 1 when packing does not apply.
int32_t SelectGroupsPerTile(InputPlan const &in, PVAFFTLayout const outputLayout, int32_t const maxGroups)
{
    for (int32_t g = maxGroups; g > 1; g--)
    {
        const int32_t tiles = (in.rangeBins % (TILE_BATCH * g) == 0) ? in.rangeBins / (TILE_BATCH * g) : 0;

        const bool transposedOut = (outputLayout != PVA_FFT_LAYOUT_DOPPLER_RX_RANGE);
        const int32_t transPitch =
            TRANSPOSE_LINE_PITCH_ALIGNED(in.chirps * COMPLEX_ELEMENT_NUM * static_cast<int32_t>(sizeof(int32_t)),
                                         TRANS_MODE_2, sizeof(int32_t)) /
            static_cast<int32_t>(sizeof(int32_t));
        const int32_t dstGroupStride = transposedOut ? (OUT_TILE_HEIGHT_TRANS * transPitch) : OUT_TILE_WIDTH;
        const int32_t outTileHeightG = transposedOut ? (OUT_TILE_HEIGHT_TRANS * g) : in.chirps;
        const int32_t outTilePitchG  = transposedOut ? transPitch : (OUT_TILE_WIDTH * g);
        const bool outFits =
            DstBufHalfSize(in.chirps, g, outTileHeightG, outTilePitchG, dstGroupStride) <= kDstBufHalfBudget;
        // A chirp-innermost input tile is TILE_BATCH * g rows at the transposed chirp pitch.
        const bool inFits = !in.chirpInnermost || ((TILE_BATCH * g * transPitch) <= kSrcBufHalfBudget);

        if (outputLayout != PVA_FFT_LAYOUT_RANGE_DOPPLER_RX && tiles >= 2 && (tiles % 2) == 0 &&
            (g * in.chirps) <= FFT_SIZE_1024 && outFits && inFits)
        {
            return g;
        }
    }
    return 1;
}

TilePlan SelectTilePlan(InputPlan const &in, OutputPlan const &out, int32_t const groupsPerTile)
{
    TilePlan t{};

    t.inWidth     = IN_TILE_WIDTH * groupsPerTile;
    t.inHeight    = in.chirps;
    t.inLinePitch = t.inWidth;
    t.advSrcDim2  = t.inWidth;
    t.inLaneOfst  = 0;
    if (in.chirpInnermost)
    {
        // TILE_BATCH * G range-bin rows by all chirps, read back with a transposed load.
        t.inWidth  = in.chirps * COMPLEX_ELEMENT_NUM;
        t.inHeight = TILE_BATCH * groupsPerTile;

        const int32_t pitchInBytes =
            TRANSPOSE_LINE_PITCH_ALIGNED(t.inWidth * sizeof(int32_t), TRANS_MODE_2, sizeof(int32_t));
        t.inLaneOfst =
            (pitchInBytes - TRANSPOSE_HORIZONTAL_OFFSET(TRANS_MODE_2, sizeof(int32_t))) / VMEM_SUPERBANK_WIDTH_IN_BYTES;
        t.inLinePitch = pitchInBytes / sizeof(int32_t);
        t.advSrcDim2  = t.inHeight * in.imageLinePitch;
    }

    t.outWidth     = OUT_TILE_WIDTH * groupsPerTile;
    t.outHeight    = out.tensorHeight;
    t.outLinePitch = t.outWidth;
    t.advDstDim1   = t.outWidth;
    t.outLaneOfst  = 0;
    if (out.transposed)
    {
        t.outWidth  = out.tensorWidth;
        t.outHeight = OUT_TILE_HEIGHT_TRANS * groupsPerTile;

        const int32_t pitchInBytes =
            TRANSPOSE_LINE_PITCH_ALIGNED(t.outWidth * sizeof(int32_t), TRANS_MODE_2, sizeof(int32_t));
        t.outLaneOfst =
            (pitchInBytes - TRANSPOSE_HORIZONTAL_OFFSET(TRANS_MODE_2, sizeof(int32_t))) / VMEM_SUPERBANK_WIDTH_IN_BYTES;
        t.outLinePitch = pitchInBytes / sizeof(int32_t);
        t.advDstDim1   = t.outHeight * out.imageLinePitch;
    }

    // Group stride in the output tile: one vector, or one block of OUT_TILE_HEIGHT_TRANS rows when transposed.
    t.dstGroupStride = out.transposed ? (OUT_TILE_HEIGHT_TRANS * t.outLinePitch) : OUT_TILE_WIDTH;

    return t;
}

// Bulk/middle/last iteration counts and offsets of the tile walk.
struct TransferPlan
{
    int32_t niterX;
    int32_t niterY;
    int32_t niterXEven; ///< Channels roundable to pairs; the odd one goes single-buffered.
    int32_t niterYBulk;
    bool needMiddle;
    bool needLast;
    int32_t srcOffsetMiddle;
    int32_t srcOffsetLast;
    int32_t inTileWidthLast;
    int32_t outTileWidthLast;
    int32_t outTileHeightLast;
    int32_t inTileBufSize;
    int32_t outTileBufSize;
};

// Geometry of the Rx-packed output tile.
struct RxPackedPlan
{
    int32_t tileWidth;
    int32_t tileLinePitch;
    int32_t tileHeight;
    int32_t l2TileSize;
    int32_t transposeWidth;
    int32_t bufferSizeBytes; ///< Double-buffered across niterX tiles; L2 if it fits, else DRAM.
    int32_t outTileBufSize;
};

RxPackedPlan SelectRxPackedPlan(TilePlan const &tile, int32_t const niterX)
{
    RxPackedPlan p{};
    p.tileWidth       = niterX * 2;
    p.tileLinePitch   = ((p.tileWidth + 15) & ~15) + 2;
    p.tileHeight      = tile.outWidth * tile.outHeight / (niterX * 2);
    p.l2TileSize      = tile.outWidth * tile.outHeight;
    p.transposeWidth  = p.l2TileSize / niterX;
    p.bufferSizeBytes = 2 * niterX * p.l2TileSize * static_cast<int32_t>(sizeof(int32_t));
    p.outTileBufSize  = p.tileHeight * p.tileLinePitch;
    return p;
}

// Bulk: an even count of full tiles; middle: one leftover full tile; last: one narrower tile.
TransferPlan SelectTransferPlan(InputPlan const &in, OutputPlan const &out, TilePlan const &tile,
                                int32_t const groupsPerTile)
{
    TransferPlan xf{};
    const int32_t tileRangeBins = TILE_BATCH * groupsPerTile;

    xf.niterX     = in.channels;
    xf.niterY     = (in.rangeBins + tileRangeBins - 1) / tileRangeBins;
    xf.niterXEven = (xf.niterX / 2) * 2;

    xf.inTileWidthLast  = ((in.tensorWidth % tile.inWidth) == 0) ? tile.inWidth : (in.tensorWidth % tile.inWidth);
    xf.outTileWidthLast = ((out.tensorWidth % tile.outWidth) == 0) ? tile.outWidth : (out.tensorWidth % tile.outWidth);
    xf.outTileHeightLast =
        ((out.tensorHeight % tile.outHeight) == 0) ? tile.outHeight : (out.tensorHeight % tile.outHeight);

    xf.needLast   = (xf.inTileWidthLast != tile.inWidth);
    xf.niterYBulk = xf.needLast ? ((xf.niterY - 1) / 2) * 2 : (xf.niterY / 2) * 2;
    xf.needMiddle = (xf.niterY - xf.niterYBulk - (xf.needLast ? 1 : 0)) > 0;

    xf.srcOffsetMiddle = xf.niterYBulk * tile.inWidth;
    xf.srcOffsetLast   = (xf.niterY - 1) * tile.inWidth;

    xf.inTileBufSize = tile.inHeight * tile.inLinePitch;
    xf.outTileBufSize =
        out.rxPacked ? SelectRxPackedPlan(tile, xf.niterX).outTileBufSize
                     : DstBufHalfSize(in.chirps, groupsPerTile, tile.outHeight, tile.outLinePitch, tile.dstGroupStride);
    return xf;
}

const char *LayoutName(PVAFFTLayout const layout)
{
    switch (layout)
    {
    case PVA_FFT_LAYOUT_DOPPLER_RX_RANGE:
        return "DOPPLER_RX_RANGE";
    case PVA_FFT_LAYOUT_RANGE_RX_DOPPLER:
        return "RANGE_RX_DOPPLER";
    case PVA_FFT_LAYOUT_RANGE_DOPPLER_RX:
        return "RANGE_DOPPLER_RX";
    case PVA_FFT_LAYOUT_RX_RANGE_DOPPLER:
        return "RX_RANGE_DOPPLER";
    case PVA_FFT_LAYOUT_RX_DOPPLER_RANGE:
        return "RX_DOPPLER_RANGE";
    default:
        return "unknown";
    }
}

// Throws "<layout> <side> needs <what> <= <limit>; got <a> * <b> = <a*b>", or "got <a>" when b is 1.
[[noreturn]] void ThrowSizeLimit(PVAFFTLayout const layout, char const *side, char const *what, int64_t const limit,
                                 int64_t const a, int64_t const b)
{
    if (b == 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s %s needs %s <= %ld; got %ld",
                              LayoutName(layout), side, what, static_cast<long>(limit), static_cast<long>(a));
    }
    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s %s needs %s <= %ld; got %ld * %ld = %ld",
                          LayoutName(layout), side, what, static_cast<long>(limit), static_cast<long>(a),
                          static_cast<long>(b), static_cast<long>(a * b));
}

// Dataflow and VMEM limits a tensor shape can reach, checked before anything is allocated.
void CheckDataflowLimits(PVADopplerFFTParams const &params, InputPlan const &in, OutputPlan const &out,
                         TilePlan const &tile, TransferPlan const &xf, int32_t const groupsPerTile)
{
    constexpr int64_t kMaxLinePitchBytes = 65535;
    constexpr int32_t kMaxIterations     = 256;
    constexpr int64_t kMaxAdvBytes       = 8388607;
    constexpr int64_t kComplexBytes      = COMPLEX_ELEMENT_NUM * sizeof(int32_t);
    // Line pitch and advance limits in complex elements.
    constexpr int64_t kMaxRowElems = kMaxLinePitchBytes / kComplexBytes;
    constexpr int64_t kMaxAdvElems = kMaxAdvBytes / kComplexBytes;
    const int32_t tileRangeBins    = TILE_BATCH * groupsPerTile;

    if (xf.inTileBufSize > kSrcBufHalfBudget)
    {
        if (in.chirpInnermost)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "RANGE_RX_DOPPLER input supports fewer than %d chirps; got %d", FFT_SIZE_1024,
                                  in.chirps);
        }
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s input does not support %d chirps",
                              LayoutName(params.inputLayout), in.chirps);
    }
    if (!out.rxPacked && (xf.outTileBufSize > kDstBufHalfBudget))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s output does not support %d chirps",
                              LayoutName(params.outputLayout), in.chirps);
    }
    if (xf.niterX > kMaxIterations)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Rx channels must not exceed %d; got %d",
                              kMaxIterations, in.channels);
    }
    if (in.rangeBins > kMaxIterations * tileRangeBins)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "%s input supports at most %d range bins for this configuration; got %d",
                              LayoutName(params.inputLayout), kMaxIterations * tileRangeBins, in.rangeBins);
    }

    /* The chirp-innermost input is an RDF, whose limits ResolveInputPlan checks. */
    if (!in.chirpInnermost)
    {
        if (in.imageLinePitch * static_cast<int64_t>(sizeof(int32_t)) > kMaxLinePitchBytes)
        {
            if (in.rxDopplerRange)
            {
                ThrowSizeLimit(params.inputLayout, "input", "range bins", kMaxRowElems, in.rangeBins, 1);
            }
            ThrowSizeLimit(params.inputLayout, "input", "Rx channels * range bins", kMaxRowElems, in.channels,
                           in.rangeBins);
        }
        const int64_t inAdvBytes = std::max(in.rxStride, tile.advSrcDim2) * static_cast<int64_t>(sizeof(int32_t));
        if (inAdvBytes > kMaxAdvBytes)
        {
            if (in.rxDopplerRange)
            {
                ThrowSizeLimit(params.inputLayout, "input", "chirps * range bins", kMaxAdvElems, in.chirps,
                               in.rangeBins);
            }
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "%s input with %d Rx channels, %d chirps and %d range bins needs a %ld-byte DMA "
                                  "advance, more than the %ld-byte limit",
                                  LayoutName(params.inputLayout), in.channels, in.chirps, in.rangeBins,
                                  static_cast<long>(inAdvBytes), static_cast<long>(kMaxAdvBytes));
        }
    }

    const int64_t outPitchBytes =
        (out.rxPacked ? (xf.niterX * COMPLEX_ELEMENT_NUM) : out.imageLinePitch) * static_cast<int64_t>(sizeof(int32_t));
    if (outPitchBytes > kMaxLinePitchBytes)
    {
        if (out.rxPacked)
        {
            ThrowSizeLimit(params.outputLayout, "output", "Rx channels", kMaxRowElems, in.channels, 1);
        }
        if (out.rxRangeDoppler)
        {
            ThrowSizeLimit(params.outputLayout, "output", "Doppler bins", kMaxRowElems, out.dopplerBins, 1);
        }
        const bool rangeInRow = !out.transposed;
        ThrowSizeLimit(params.outputLayout, "output",
                       rangeInRow ? "Rx channels * range bins" : "Rx channels * Doppler bins", kMaxRowElems,
                       in.channels, rangeInRow ? out.rangeBins : out.dopplerBins);
    }

    const int64_t outAdvBytes = out.rxPacked ? (static_cast<int64_t>(xf.niterX) * out.imageLinePitch * sizeof(int32_t))
                                             : std::max<int64_t>(out.rxStride, tile.advDstDim1) * sizeof(int32_t);
    if (outAdvBytes > kMaxAdvBytes)
    {
        if (out.rxRangeDoppler)
        {
            ThrowSizeLimit(params.outputLayout, "output", "range bins * Doppler bins", kMaxAdvElems, out.rangeBins,
                           out.dopplerBins);
        }
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "%s output with %d Rx channels, %d range bins and %d Doppler bins needs a %ld-byte DMA "
                              "advance, more than the %ld-byte limit",
                              LayoutName(params.outputLayout), in.channels, out.rangeBins, out.dopplerBins,
                              static_cast<long>(outAdvBytes), static_cast<long>(kMaxAdvBytes));
    }
}

template<typename SrcOp>
void EmitInputTransfers(SequenceDataFlow &df, SrcOp const &srcOp, int32_t *const src_v, InputPlan const &in,
                        TilePlan const &tile, TransferPlan const &xf)
{
    if (xf.niterYBulk > 0)
    {
        df.addTransfer()
            .src(srcOp, in.imageLinePitch * sizeof(int32_t))
            .dst(src_v, tile.inLinePitch * sizeof(int32_t))
            .srcDim1(xf.niterX, in.rxStride * sizeof(int32_t))
            .srcDim2(xf.niterYBulk, tile.advSrcDim2 * sizeof(int32_t))
            .dstDim1(2, xf.inTileBufSize * sizeof(int32_t))
            .dstDim2(xf.niterYBulk / 2, 0)
            .dstDim3(xf.niterX, 0)
            .tile(tile.inWidth * sizeof(int32_t), tile.inHeight)
            .mode(TransferModeType::TILE);
    }

    if (xf.needMiddle)
    {
        if (xf.niterXEven > 0)
        {
            df.addTransfer()
                .src(srcOp + xf.srcOffsetMiddle, in.imageLinePitch * sizeof(int32_t))
                .dst(src_v, tile.inLinePitch * sizeof(int32_t))
                .srcDim1(xf.niterXEven, in.rxStride * sizeof(int32_t))
                .dstDim1(2, xf.inTileBufSize * sizeof(int32_t))
                .dstDim2(xf.niterXEven / 2, 0)
                .tile(tile.inWidth * sizeof(int32_t), tile.inHeight)
                .mode(TransferModeType::TILE);
        }
        if (xf.niterX & 1)
        {
            df.addTransfer()
                .src(srcOp + (xf.srcOffsetMiddle + xf.niterXEven * in.rxStride), in.imageLinePitch * sizeof(int32_t))
                .dst(src_v, tile.inLinePitch * sizeof(int32_t))
                .srcDim1(1, in.rxStride * sizeof(int32_t))
                .dstDim1(1, xf.inTileBufSize * sizeof(int32_t))
                .tile(tile.inWidth * sizeof(int32_t), tile.inHeight)
                .mode(TransferModeType::TILE);
        }
    }

    if (xf.needLast)
    {
        if (xf.niterXEven > 0)
        {
            df.addTransfer()
                .src(srcOp + xf.srcOffsetLast, in.imageLinePitch * sizeof(int32_t))
                .dst(src_v, tile.inLinePitch * sizeof(int32_t))
                .srcDim1(xf.niterXEven, in.rxStride * sizeof(int32_t))
                .dstDim1(2, xf.inTileBufSize * sizeof(int32_t))
                .dstDim2(xf.niterXEven / 2, 0)
                .tile(xf.inTileWidthLast * sizeof(int32_t), tile.inHeight)
                .mode(TransferModeType::TILE);
        }
        if (xf.niterX & 1)
        {
            df.addTransfer()
                .src(srcOp + (xf.srcOffsetLast + xf.niterXEven * in.rxStride), in.imageLinePitch * sizeof(int32_t))
                .dst(src_v, tile.inLinePitch * sizeof(int32_t))
                .srcDim1(1, in.rxStride * sizeof(int32_t))
                .dstDim1(1, xf.inTileBufSize * sizeof(int32_t))
                .tile(xf.inTileWidthLast * sizeof(int32_t), tile.inHeight)
                .mode(TransferModeType::TILE);
        }
    }
}

/* Output transfers for the direct layouts: VMEM tile straight to DRAM, no transpose buffer. */
template<typename DstOp>
void EmitDirectOutputTransfers(SequenceDataFlow &df, int32_t *const dst_v, DstOp const &dstOp, OutputPlan const &out,
                               TilePlan const &tile, TransferPlan const &xf)
{
    const int32_t dstOffsetMiddle = xf.niterYBulk * tile.advDstDim1;
    const int32_t dstOffsetLast   = (xf.niterY - 1) * tile.advDstDim1;

    if (xf.niterYBulk > 0)
    {
        df.addTransfer()
            .src(dst_v, tile.outLinePitch * sizeof(int32_t))
            .dst(dstOp, out.imageLinePitch * sizeof(int32_t))
            .srcDim1(2, xf.outTileBufSize * sizeof(int32_t))
            .srcDim2(xf.niterYBulk / 2, 0)
            .srcDim3(xf.niterX, 0)
            .dstDim1(xf.niterX, out.rxStride * sizeof(int32_t))
            .dstDim2(xf.niterYBulk, tile.advDstDim1 * sizeof(int32_t))
            .tile(tile.outWidth * sizeof(int32_t), tile.outHeight)
            .mode(TransferModeType::TILE);
    }

    if (xf.needMiddle)
    {
        if (xf.niterXEven > 0)
        {
            df.addTransfer()
                .src(dst_v, tile.outLinePitch * sizeof(int32_t))
                .dst(dstOp + dstOffsetMiddle, out.imageLinePitch * sizeof(int32_t))
                .srcDim1(2, xf.outTileBufSize * sizeof(int32_t))
                .srcDim2(xf.niterXEven / 2, 0)
                .dstDim1(xf.niterXEven, out.rxStride * sizeof(int32_t))
                .tile(tile.outWidth * sizeof(int32_t), tile.outHeight)
                .mode(TransferModeType::TILE);
        }
        if (xf.niterX & 1)
        {
            df.addTransfer()
                .src(dst_v, tile.outLinePitch * sizeof(int32_t))
                .dst(dstOp + (dstOffsetMiddle + xf.niterXEven * out.rxStride), out.imageLinePitch * sizeof(int32_t))
                .srcDim1(1, xf.outTileBufSize * sizeof(int32_t))
                .dstDim1(1, out.rxStride * sizeof(int32_t))
                .tile(tile.outWidth * sizeof(int32_t), tile.outHeight)
                .mode(TransferModeType::TILE);
        }
    }

    if (xf.needLast)
    {
        if (xf.niterXEven > 0)
        {
            df.addTransfer()
                .src(dst_v, tile.outLinePitch * sizeof(int32_t))
                .dst(dstOp + dstOffsetLast, out.imageLinePitch * sizeof(int32_t))
                .srcDim1(2, xf.outTileBufSize * sizeof(int32_t))
                .srcDim2(xf.niterXEven / 2, 0)
                .dstDim1(xf.niterXEven, out.rxStride * sizeof(int32_t))
                .tile(xf.outTileWidthLast * sizeof(int32_t), xf.outTileHeightLast)
                .mode(TransferModeType::TILE);
        }
        if (xf.niterX & 1)
        {
            df.addTransfer()
                .src(dst_v, tile.outLinePitch * sizeof(int32_t))
                .dst(dstOp + (dstOffsetLast + xf.niterXEven * out.rxStride), out.imageLinePitch * sizeof(int32_t))
                .srcDim1(1, xf.outTileBufSize * sizeof(int32_t))
                .dstDim1(1, out.rxStride * sizeof(int32_t))
                .tile(xf.outTileWidthLast * sizeof(int32_t), xf.outTileHeightLast)
                .mode(TransferModeType::TILE);
        }
    }
}

template<typename DstOp>
void EmitRxPackedOutputTransfers(SequenceDataFlow &dstDf, SequenceDataFlow &toL2Df, SequenceDataFlow &fromL2Df,
                                 int32_t *const dst_v, DstOp const &dstOp, uint32_t *const transposeBuffer,
                                 int32_t *const tmpBuf, OutputPlan const &out, TilePlan const &tile,
                                 TransferPlan const &xf, RxPackedPlan const &rx)
{
    const int32_t dstOffsetMiddle = xf.niterYBulk * xf.niterX * out.imageLinePitch;
    const int32_t dstOffsetLast   = (xf.niterY - 1) * xf.niterX * out.imageLinePitch;

    if (xf.niterYBulk > 0)
    {
        dstDf.addTransfer()
            .src(dst_v, rx.tileLinePitch * sizeof(int32_t))
            .dst(dstOp, rx.tileWidth * sizeof(int32_t))
            .srcDim1(2, xf.outTileBufSize * sizeof(int32_t))
            .srcDim2(xf.niterYBulk / 2, 0)
            .srcDim3(xf.niterX, 0)
            .dstDim1(xf.niterX, out.imageLinePitch * sizeof(int32_t))
            .dstDim2(xf.niterYBulk, xf.niterX * out.imageLinePitch * sizeof(int32_t))
            .tile(rx.tileWidth * sizeof(int32_t), rx.tileHeight)
            .mode(TransferModeType::TILE);

        toL2Df.addTransfer()
            .src(dst_v, tile.outLinePitch * sizeof(int32_t))
            .dst(transposeBuffer, tile.outWidth * sizeof(int32_t))
            .srcDim1(2, xf.outTileBufSize * sizeof(int32_t))
            .srcDim2(xf.niterYBulk / 2, 0)
            .srcDim3(xf.niterX, 0)
            .dstDim1(2 * xf.niterX, rx.l2TileSize * sizeof(int32_t))
            .dstDim2(xf.niterYBulk / 2, 0)
            .tile(tile.outWidth * sizeof(int32_t), tile.outHeight)
            .mode(TransferModeType::TILE);

        fromL2Df.addTransfer()
            .src(transposeBuffer, rx.l2TileSize * sizeof(int32_t))
            .dst(tmpBuf, rx.transposeWidth * sizeof(int32_t))
            .srcDim1(xf.niterX, rx.transposeWidth * sizeof(int32_t))
            .srcDim2(2, rx.l2TileSize * xf.niterX * sizeof(int32_t))
            .srcDim3(xf.niterYBulk / 2, 0)
            .dstDim1(2, 0)
            .dstDim2(xf.niterYBulk / 2, 0)
            .dstDim3(xf.niterX, 0)
            .tile(rx.transposeWidth * sizeof(int32_t), xf.niterX)
            .mode(TransferModeType::TILE);
    }

    if (xf.needMiddle)
    {
        dstDf.addTransfer()
            .src(dst_v, rx.tileLinePitch * sizeof(int32_t))
            .dst(dstOp + dstOffsetMiddle, rx.tileWidth * sizeof(int32_t))
            .srcDim1(2, xf.outTileBufSize * sizeof(int32_t))
            .srcDim2(xf.niterX / 2, 0)
            .dstDim1(xf.niterX, out.imageLinePitch * sizeof(int32_t))
            .tile(rx.tileWidth * sizeof(int32_t), rx.tileHeight)
            .mode(TransferModeType::TILE);

        toL2Df.addTransfer()
            .src(dst_v, tile.outLinePitch * sizeof(int32_t))
            .dst(transposeBuffer, tile.outWidth * sizeof(int32_t))
            .srcDim1(2, xf.outTileBufSize * sizeof(int32_t))
            .srcDim2(xf.niterX / 2, 0)
            .dstDim1(xf.niterX, rx.l2TileSize * sizeof(int32_t))
            .tile(tile.outWidth * sizeof(int32_t), tile.outHeight)
            .mode(TransferModeType::TILE);

        fromL2Df.addTransfer()
            .src(transposeBuffer, rx.l2TileSize * sizeof(int32_t))
            .dst(tmpBuf, rx.transposeWidth * sizeof(int32_t))
            .srcDim1(xf.niterX, rx.transposeWidth * sizeof(int32_t))
            .dstDim1(2, 0)
            .dstDim2(xf.niterX / 2, 0)
            .tile(rx.transposeWidth * sizeof(int32_t), xf.niterX)
            .mode(TransferModeType::TILE);
    }

    if (xf.needLast)
    {
        const int32_t lastRangeBins = xf.outTileHeightLast;
        const int32_t lastIterEven  = (lastRangeBins / 2) * 2;
        const int32_t lastIterOdd   = lastRangeBins - lastIterEven;
        const int32_t l2OffsetLast  = xf.needMiddle ? xf.niterX * rx.l2TileSize : 0;

        toL2Df.addTransfer()
            .src(dst_v, tile.outLinePitch * sizeof(int32_t))
            .dst(transposeBuffer + l2OffsetLast, tile.outWidth * sizeof(int32_t))
            .srcDim1(2, xf.outTileBufSize * sizeof(int32_t))
            .srcDim2(xf.niterX / 2, 0)
            .dstDim1(xf.niterX, rx.l2TileSize * sizeof(int32_t))
            .tile(tile.outWidth * sizeof(int32_t), tile.outHeight)
            .mode(TransferModeType::TILE);

        if (lastIterEven > 0)
        {
            dstDf.addTransfer()
                .src(dst_v, rx.tileLinePitch * sizeof(int32_t))
                .dst(dstOp + dstOffsetLast, rx.tileWidth * sizeof(int32_t))
                .srcDim1(2, xf.outTileBufSize * sizeof(int32_t))
                .srcDim2(lastIterEven / 2, 0)
                .dstDim1(lastIterEven, out.imageLinePitch * sizeof(int32_t))
                .tile(rx.tileWidth * sizeof(int32_t), rx.tileHeight)
                .mode(TransferModeType::TILE);

            fromL2Df.addTransfer()
                .src(transposeBuffer + l2OffsetLast, rx.l2TileSize * sizeof(int32_t))
                .dst(tmpBuf, rx.transposeWidth * sizeof(int32_t))
                .srcDim1(lastIterEven, rx.transposeWidth * sizeof(int32_t))
                .dstDim1(2, 0)
                .dstDim2(lastIterEven / 2, 0)
                .tile(rx.transposeWidth * sizeof(int32_t), xf.niterX)
                .mode(TransferModeType::TILE);
        }

        if (lastIterOdd > 0)
        {
            dstDf.addTransfer()
                .src(dst_v, rx.tileLinePitch * sizeof(int32_t))
                .dst(dstOp + (dstOffsetLast + lastIterEven * out.imageLinePitch), rx.tileWidth * sizeof(int32_t))
                .srcDim1(1, 0)
                .dstDim1(1, 0)
                .tile(rx.tileWidth * sizeof(int32_t), rx.tileHeight)
                .mode(TransferModeType::TILE);

            fromL2Df.addTransfer()
                .src(transposeBuffer + (l2OffsetLast + lastIterEven * rx.transposeWidth),
                     rx.l2TileSize * sizeof(int32_t))
                .dst(tmpBuf, rx.transposeWidth * sizeof(int32_t))
                .srcDim1(1, 0)
                .dstDim1(1, 0)
                .tile(rx.transposeWidth * sizeof(int32_t), xf.niterX)
                .mode(TransferModeType::TILE);
        }
    }
}

} // namespace

template<typename TI, typename TO>
DopplerFFT<TI, TO>::DopplerFFT(NVCVTensorRequirements const *const inTensorRequirements,
                               NVCVTensorRequirements const *const winTensorRequirements,
                               NVCVTensorRequirements const *const outTensorRequirements,
                               PVADopplerFFTParams const *const params)
    : m_inTensorRequirements{*inTensorRequirements}
    , m_winTensorRequirements{*winTensorRequirements}
    , m_outTensorRequirements{*outTensorRequirements}
    , m_params{params ? *params : kDefaultParams}
    , m_srcOp(cupva::OffsetPointer<TI>::Create(cupva::MemType::DRAM, nullptr))
    , m_winOp(cupva::OffsetPointer<TI>::Create(cupva::MemType::DRAM, nullptr))
    , m_dstOp(cupva::OffsetPointer<TO>::Create(cupva::MemType::DRAM, nullptr))
    , m_twiddle_d(nullptr)
    , m_transposeBuffer_d(nullptr)
    , m_prog{}
{
    pva_solutions::utils::CheckDenselyPacked(*inTensorRequirements, "Input");
    pva_solutions::utils::CheckDenselyPacked(*winTensorRequirements, "Window");
    pva_solutions::utils::CheckDenselyPacked(*outTensorRequirements, "Output");

    int64_t inTensorShapeNCHW[4];
    nvcvTensorShapePermute(inTensorRequirements->layout, inTensorRequirements->shape, NVCV_TENSOR_NCHW,
                           inTensorShapeNCHW);

    const InputPlan inPlan = ResolveInputPlan(m_params, inTensorShapeNCHW);

    const int32_t channels  = inPlan.channels;
    const int32_t chirps    = inPlan.chirps;
    const int32_t rangeBins = inPlan.rangeBins;

    const bool needTranspose = (m_params.outputLayout == PVA_FFT_LAYOUT_RANGE_RX_DOPPLER ||
                                m_params.outputLayout == PVA_FFT_LAYOUT_RANGE_DOPPLER_RX ||
                                m_params.outputLayout == PVA_FFT_LAYOUT_RX_RANGE_DOPPLER);

    const FFTFactorization probe{priv::FactorizeFFTSize(chirps, needTranspose, false, 1)};
    const int32_t probePow2 = probe.groups[RADIX4_GROUP_IDX].numStages + probe.groups[RADIX4_GROUP_IDX].isOdd;
    const int32_t probeStages =
        probe.groups[RADIX5_GROUP_IDX].numStages + probe.groups[RADIX3_GROUP_IDX].numStages + probePow2;

    const bool canFuse      = (probe.isValid != 0) && (probeStages <= FFT_MAX_FUSED_STAGES);
    const int32_t maxGroups = canFuse ? TILE_GROUPS_MAX : (inPlan.chirpInnermost ? 1 : TILE_GROUPS_MAX_UNFUSED);

    const int32_t groupsPerTile = SelectGroupsPerTile(inPlan, m_params.outputLayout, maxGroups);

    FFTFactorization factorization{priv::FactorizeFFTSize(chirps, needTranspose, false, groupsPerTile)};

    if (!factorization.isValid)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported chirp number %d", chirps);
    }

    if (factorization.twiddleSize > TWIDDLE_SIZE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Twiddle size is too large");
    }

    const int32_t cfgsRequired = FFT_BATCHED_CFGS_REQUIRED(
        factorization.groups[RADIX5_GROUP_IDX].numStages, factorization.groups[RADIX3_GROUP_IDX].numStages,
        factorization.groups[RADIX4_GROUP_IDX].numStages + factorization.groups[RADIX4_GROUP_IDX].isOdd);
    if (cfgsRequired > FFT_BATCHED_MAX_CFGS)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Chirp number %d needs %d FFT agen configs, only %d are available", chirps, cfgsRequired,
                              FFT_BATCHED_MAX_CFGS);
    }

    int64_t winTensorShapeNCHW[4];
    nvcvTensorShapePermute(winTensorRequirements->layout, winTensorRequirements->shape, NVCV_TENSOR_NCHW,
                           winTensorShapeNCHW);

    const int32_t windowSize = winTensorShapeNCHW[3];

    if (windowSize != chirps)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Window size must be equal to chirp number");
    }

    int64_t outTensorShapeNCHW[4];
    nvcvTensorShapePermute(outTensorRequirements->layout, outTensorRequirements->shape, NVCV_TENSOR_NCHW,
                           outTensorShapeNCHW);

    const OutputPlan outPlan = ResolveOutputPlan(m_params.outputLayout, outTensorShapeNCHW, inPlan);
    const TilePlan tile      = SelectTilePlan(inPlan, outPlan, groupsPerTile);
    const TransferPlan xf    = SelectTransferPlan(inPlan, outPlan, tile, groupsPerTile);

    // Checks run before the first allocation: a throwing constructor skips the destructor.
    CheckDataflowLimits(m_params, inPlan, outPlan, tile, xf, groupsPerTile);

    m_twiddle_d        = (int32_t *)mem::Alloc(factorization.twiddleSize * sizeof(int32_t));
    int32_t *twiddle_h = (int32_t *)mem::GetHostPointer(m_twiddle_d);

    std::vector<int32_t> twiddle_interleaved(2 * chirps);
    for (int32_t i = 0; i < chirps; i++)
    {
        double angle                   = 2.0f * M_PI * i / chirps;
        twiddle_interleaved[2 * i]     = (int32_t)round(cos(angle) * (1UL << TWIDDLE_QBITS));
        twiddle_interleaved[2 * i + 1] = (int32_t)round(-sin(angle) * (1UL << TWIDDLE_QBITS));
    }

    if (factorization.groups[RADIX3_GROUP_IDX].numStages > 0)
    {
        pva_solutions::priv::generateComplexFFTTwiddleRadix3(twiddle_h, twiddle_interleaved, factorization);
        pva_solutions::priv::generateComplexFFTTwiddleIntraStage(twiddle_h + 2 * COMPLEX_ELEMENT_NUM,
                                                                 twiddle_interleaved, factorization, RADIX3_NUM);
    }

    if (factorization.groups[RADIX4_GROUP_IDX].numStages > 0)
    {
        pva_solutions::priv::generateComplexFFTTwiddleIntraStage(twiddle_h + factorization.twiddleOffsets[0],
                                                                 twiddle_interleaved, factorization, RADIX4_NUM);
    }

    if (factorization.groups[RADIX5_GROUP_IDX].numStages > 0)
    {
        pva_solutions::priv::generateComplexFFTTwiddleRadix5(twiddle_h + factorization.twiddleOffsets[1],
                                                             twiddle_interleaved, factorization);
        pva_solutions::priv::generateComplexFFTTwiddleIntraStage(
            twiddle_h + factorization.twiddleOffsets[1] + 4 * COMPLEX_ELEMENT_NUM, twiddle_interleaved, factorization,
            RADIX5_NUM);
    }

    m_exec    = GetDopplerFFTExec(outPlan.rxPacked);
    auto prog = CmdProgram::Create(*m_exec.get());

    int32_t phase = 0;

    auto coeffHdl = prog["coeffHdl"];

    int32_t *twiddle_v              = prog["twiddleBuf"].ptr<int32_t>();
    TI *window_v                    = prog["windowBuf"].ptr<TI>();
    SequenceDataFlow &coeffDataFlow = prog.addDataFlowHead<SequenceDataFlow>(phase).handler(coeffHdl);
    coeffDataFlow.addTransfer()
        .src(m_twiddle_d, factorization.twiddleSize * sizeof(int32_t))
        .dst(twiddle_v, factorization.twiddleSize * sizeof(int32_t))
        .tile(factorization.twiddleSize * sizeof(int32_t), 1)
        .mode(TransferModeType::CONTINUOUS);
    coeffDataFlow.addTransfer()
        .src(m_winOp, windowSize * sizeof(TI))
        .dst(window_v, windowSize * sizeof(TI))
        .tile(windowSize * sizeof(TI), 1)
        .mode(TransferModeType::CONTINUOUS);

    phase++;

    if (outPlan.transposed)
    {
        prog["outLaneOfst"] = tile.outLaneOfst;
    }

    /* Only the direct top declares the wide-tile parameters. */
    if (!outPlan.rxPacked)
    {
        /* canFuse depends only on the stage structure, which does not change with G. */
        prog["fuseGroups"]    = static_cast<int32_t>((canFuse && (groupsPerTile > 1)) ? 1 : 0);
        prog["groupsPerTile"] = groupsPerTile;
        // Chirp stride of the [chirp][bin] intermediates, whatever the input layout.
        prog["tilePitch"]        = IN_TILE_WIDTH * groupsPerTile;
        prog["srcGroupStride"]   = IN_TILE_WIDTH;
        prog["inChirpInnermost"] = static_cast<int32_t>(inPlan.chirpInnermost ? 1 : 0);
        prog["inLaneOfst"]       = tile.inLaneOfst;
        prog["inRowPitch"]       = tile.inLinePitch;
        prog["dstGroupStride"]   = tile.dstGroupStride;
    }

    // Set up data flow handlers and buffers
    auto srcHdl    = prog["srcHdl"];
    auto dstHdl    = prog["dstHdl"];
    int32_t *src_v = prog["srcBuf"].ptr<int32_t>();
    int32_t *dst_v = prog["dstBuf"].ptr<int32_t>();

    // The chirp-innermost input is a RasterDataFlow; the others use the SequenceDataFlow walk.
    if (inPlan.chirpInnermost)
    {
        auto srcRdfHdl = prog["srcRdfHdl"];

        RasterDataFlow &srcRdf = prog.addDataFlowHead<RasterDataFlow>(phase);
        srcRdf.handler(srcRdfHdl)
            .src(m_srcOp, inPlan.imageLinePitch, rangeBins, inPlan.imageLinePitch)
            .tileBuffer(src_v)
            .tile(tile.inWidth, tile.inHeight)
            .transpose(TRANS_MODE_2);
    }
    else
    {
        SequenceDataFlow &srcDataFlow = prog.addDataFlowHead<SequenceDataFlow>(phase).handler(srcHdl);
        EmitInputTransfers(srcDataFlow, m_srcOp, src_v, inPlan, tile, xf);
    }
    SequenceDataFlow &dstDataFlow = prog.addDataFlowHead<SequenceDataFlow>(phase).handler(dstHdl);

    if (outPlan.rxPacked)
    {
        // RANGE_DOPPLER_RX transposes Rx against Doppler through an L2 (or DRAM) staging buffer.
        const RxPackedPlan rx = SelectRxPackedPlan(tile, xf.niterX);

        phase++;

        /* The staging buffer lives in L2 when it fits and in DRAM otherwise. */
        uint32_t *transposeBufferAddr;
        constexpr int32_t kL2MaxSize = 512 * 1024;
        if (rx.bufferSizeBytes <= kL2MaxSize)
        {
            transposeBufferAddr = (uint32_t *)cupva::mem::GetL2BaseAddress();
            prog.setL2Size(rx.bufferSizeBytes);
        }
        else
        {
            m_transposeBuffer_d = (int32_t *)mem::Alloc(rx.bufferSizeBytes);
            transposeBufferAddr = (uint32_t *)m_transposeBuffer_d;
        }

        auto toL2Hdl                     = prog["toL2Hdl"];
        auto fromL2Hdl                   = prog["fromL2Hdl"];
        int32_t *tmpBuf                  = prog["tmpBuf"].ptr<int32_t>();
        SequenceDataFlow &toL2DataFlow   = prog.addDataFlowHead<SequenceDataFlow>(phase).handler(toL2Hdl);
        SequenceDataFlow &fromL2DataFlow = prog.addDataFlowHead<SequenceDataFlow>(phase).handler(fromL2Hdl);

        EmitRxPackedOutputTransfers(dstDataFlow, toL2DataFlow, fromL2DataFlow, dst_v, m_dstOp, transposeBufferAddr,
                                    tmpBuf, outPlan, tile, xf, rx);

        // The drain covers the last, partial tile when there is one.
        prog["drainCount"]    = static_cast<int32_t>(xf.needLast ? xf.outTileHeightLast : xf.niterX);
        prog["outTileWidth"]  = static_cast<int32_t>(tile.outWidth);
        prog["outTileHeight"] = static_cast<int32_t>(tile.outHeight);
    }
    else
    {
        // The other output layouts write each tile straight out.
        EmitDirectOutputTransfers(dstDataFlow, dst_v, m_dstOp, outPlan, tile, xf);
    }

    // Configure program parameters
    prog["niterX"]         = static_cast<int32_t>(xf.niterX);
    prog["niterY"]         = static_cast<int32_t>(xf.niterY);
    prog["inTileBufSize"]  = static_cast<int32_t>(xf.inTileBufSize);
    prog["outTileBufSize"] = static_cast<int32_t>(xf.outTileBufSize);
    prog["factorization"].set(&factorization, 1);
    // RX_RANGE_DOPPLER uses the same VPU transpose tile as RANGE_RX_DOPPLER; only DRAM addressing differs
    prog["outputLayout"] = (m_params.outputLayout == PVA_FFT_LAYOUT_RX_RANGE_DOPPLER)
                               ? static_cast<int32_t>(PVA_FFT_LAYOUT_RANGE_RX_DOPPLER)
                               : static_cast<int32_t>(m_params.outputLayout);

    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename TI, typename TO>
template<typename S>
void DopplerFFT<TI, TO>::operator()(S stream, const nvcv::Tensor &in, const nvcv::Tensor &win, const nvcv::Tensor &out)
{
    auto inTensorData  = in.exportData<nvcv::TensorDataStrided>();
    auto winTensorData = win.exportData<nvcv::TensorDataStrided>();
    auto outTensorData = out.exportData<nvcv::TensorDataStrided>();

    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_inTensorRequirements, inTensorData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_winTensorRequirements, winTensorData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_outTensorRequirements, outTensorData->cdata()))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Input and output images must have the same requirements that was used to create the operator");
    }
    if ((static_cast<void *>(inTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(winTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(outTensorData->basePtr()) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor base pointers must not be NULL");
    }

    m_srcOp.setBase(static_cast<TI *>(static_cast<void *>(inTensorData->basePtr())));
    m_winOp.setBase(static_cast<TI *>(static_cast<void *>(winTensorData->basePtr())));
    m_dstOp.setBase(static_cast<TO *>(static_cast<void *>(outTensorData->basePtr())));

    // Set external memory pointers for SQDF
    cupvaParameter_t srcExtPtr, dstExtPtr;
    CupvaCmdProgramGetParameter(&m_prog, &srcExtPtr, "srcExtPtr");
    CupvaCmdProgramGetParameter(&m_prog, &dstExtPtr, "dstExtPtr");

    CupvaParameterSetValuePointer(&srcExtPtr, static_cast<void *>(inTensorData->basePtr()));
    CupvaParameterSetValuePointer(&dstExtPtr, static_cast<void *>(outTensorData->basePtr()));

    cupvaError_t err = CupvaCmdProgramUpdateDataFlows(&m_prog);
    if (err != CUPVA_ERROR_NONE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "Failed to update data flows");
    }

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
        static_assert(!sizeof(S), "Unsupported stream type for DopplerFFT operator");
    }

    if (!(err == CUPVA_ERROR_NONE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Submit failed");
    }
}

template<typename TI, typename TO>
DopplerFFT<TI, TO>::~DopplerFFT()
{
    mem::Free(m_twiddle_d);
    mem::Free(m_transposeBuffer_d);
    CupvaCmdDestroy(&m_prog);
}

// Explicit instantiation of the template to support the types we care about
template class DopplerFFT<int32_t, int32_t>;

} // namespace pva_solutions::priv

NVCVStatus pvaDopplerFFTCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements const *const inTensorRequirements,
                               NVCVTensorRequirements const *const winTensorRequirements,
                               NVCVTensorRequirements const *const outTensorRequirements,
                               PVADopplerFFTParams const *const params)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        if (inTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to input NVCVTensorRequirements must not be NULL");
        }
        if (winTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to window NVCVTensorRequirements must not be NULL");
        }
        if (outTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to output NVCVTensorRequirements must not be NULL");
        }

        if ((inTensorRequirements->dtype == NVCV_DATA_TYPE_2S32) &&
            (outTensorRequirements->dtype == NVCV_DATA_TYPE_2S32))
        {
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::DopplerFFT<int32_t, int32_t>(
                inTensorRequirements, winTensorRequirements, outTensorRequirements, params));
        }
        else
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input/output data type");
        }
    });
}

template<typename StreamType>
NVCVStatus DopplerFFTSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle const in,
                                NVCVTensorHandle const win, NVCVTensorHandle out)
{
    return nvcv::ProtectCall([&] {
        nvcv::TensorWrapHandle input(in), window(win), output(out);

        NVCVDataType inDataType, outDataType;
        nvcvTensorGetDataType(in, &inDataType);
        nvcvTensorGetDataType(out, &outDataType);

        if ((inDataType == NVCV_DATA_TYPE_2S32) && (outDataType == NVCV_DATA_TYPE_2S32))
        {
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::DopplerFFT<int32_t, int32_t>>(handle)(
                stream, input, window, output);
        }
        else
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input/output data type");
        }
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus DopplerFFTSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle const,
                                                        NVCVTensorHandle const, NVCVTensorHandle);
template NVCVStatus DopplerFFTSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle const,
                                                       NVCVTensorHandle const, NVCVTensorHandle);

// C++ overload implementations
NVCVStatus pvaDopplerFFTSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle const in,
                               NVCVTensorHandle const win, NVCVTensorHandle out)
{
    return DopplerFFTSubmitImpl(handle, stream, in, win, out);
}

NVCVStatus pvaDopplerFFTSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle const in,
                               NVCVTensorHandle const win, NVCVTensorHandle out)
{
    return DopplerFFTSubmitImpl(handle, stream, in, win, out);
}
