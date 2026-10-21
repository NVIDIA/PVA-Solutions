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

#include "OpRangeFFT.hpp"

#include "device/range_fft_param.h"

#include <IOperator.hpp>
#include <RadarUtils.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>

#include <OpRangeFFT.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <cmath>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_range_fft_dev);

namespace pva_solutions::priv {

template<typename TI, typename TO>
std::mutex RangeFFT<TI, TO>::s_execMapMutex;

template<typename TI, typename TO>
std::map<typename RangeFFT<TI, TO>::ExecKey, std::weak_ptr<cupva::Executable>> RangeFFT<TI, TO>::s_rangeFFTExecutables;

template<typename TI, typename TO>
std::shared_ptr<cupva::Executable> RangeFFT<TI, TO>::GetRangeFFTExec()
{
    static IOperator::ExecutableData sRangeFFTExecData;

    sRangeFFTExecData = {CUPVA_EXECUTABLE_DATA(pva_range_fft_dev), CUPVA_EXECUTABLE_SIZE(pva_range_fft_dev)};

    ExecKey execKey = std::make_tuple(cupva::Context::GetCurrent());
    return IOperator::GetExecutable<ExecKey>(sRangeFFTExecData, execKey, s_rangeFFTExecutables, s_execMapMutex);
}

template<typename TI, typename TO>
RangeFFT<TI, TO>::RangeFFT(NVCVTensorRequirements const *const inTensorRequirements,
                           NVCVTensorRequirements const *const winTensorRequirements,
                           NVCVTensorRequirements const *const outTensorRequirements,
                           PVARangeFFTParams const *const params)
    : m_inTensorRequirements{*inTensorRequirements}
    , m_winTensorRequirements{*winTensorRequirements}
    , m_outTensorRequirements{*outTensorRequirements}
    , m_params{params ? *params : kDefaultParams}
    , m_inputIsComplex{inTensorRequirements->dtype == NVCV_DATA_TYPE_2S32}
    , m_srcOp(cupva::OffsetPointer<TI>::Create(cupva::MemType::DRAM, nullptr))
    , m_winOp(cupva::OffsetPointer<TI>::Create(cupva::MemType::DRAM, nullptr))
    , m_dstOp(cupva::OffsetPointer<TO>::Create(cupva::MemType::DRAM, nullptr))
    , m_twiddle_d(nullptr)
    , m_prog{}
{
    // The DMA strides are derived from the shapes.
    pva_solutions::utils::CheckDenselyPacked(*inTensorRequirements, "Input");
    pva_solutions::utils::CheckDenselyPacked(*winTensorRequirements, "Window");
    pva_solutions::utils::CheckDenselyPacked(*outTensorRequirements, "Output");

    auto isSupportedInputLayout = [](PVAFFTLayout const layout) {
        return (layout == PVA_FFT_LAYOUT_DOPPLER_RX_RANGE) || (layout == PVA_FFT_LAYOUT_RX_DOPPLER_RANGE);
    };
    auto isSupportedOutputLayout = [&](PVAFFTLayout const layout) {
        return isSupportedInputLayout(layout) || (layout == PVA_FFT_LAYOUT_RANGE_RX_DOPPLER);
    };
    if (!isSupportedInputLayout(m_params.inputLayout))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Unsupported input layout %d; RangeFFT supports only PVA_FFT_LAYOUT_DOPPLER_RX_RANGE "
                              "(%d) and PVA_FFT_LAYOUT_RX_DOPPLER_RANGE (%d)",
                              static_cast<int32_t>(m_params.inputLayout),
                              static_cast<int32_t>(PVA_FFT_LAYOUT_DOPPLER_RX_RANGE),
                              static_cast<int32_t>(PVA_FFT_LAYOUT_RX_DOPPLER_RANGE));
    }
    if (!isSupportedOutputLayout(m_params.outputLayout))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Unsupported output layout %d; RangeFFT supports only PVA_FFT_LAYOUT_DOPPLER_RX_RANGE "
                              "(%d), PVA_FFT_LAYOUT_RX_DOPPLER_RANGE (%d) and PVA_FFT_LAYOUT_RANGE_RX_DOPPLER (%d)",
                              static_cast<int32_t>(m_params.outputLayout),
                              static_cast<int32_t>(PVA_FFT_LAYOUT_DOPPLER_RX_RANGE),
                              static_cast<int32_t>(PVA_FFT_LAYOUT_RX_DOPPLER_RANGE),
                              static_cast<int32_t>(PVA_FFT_LAYOUT_RANGE_RX_DOPPLER));
    }

    int64_t inTensorShapeNCHW[4];
    nvcvTensorShapePermute(inTensorRequirements->layout, inTensorRequirements->shape, NVCV_TENSOR_NCHW,
                           inTensorShapeNCHW);

    // Tiles a dataflow can repeat along one dimension.
    static constexpr int32_t kMaxTilesPerDim = 256;
    // Largest byte stride a dataflow dimension can carry.
    static constexpr int64_t kMaxAdvBytes = 8388607LL;
    // Largest RasterDataFlow line pitch, in elements.
    static constexpr int64_t kMaxLinePitchElems = 65535LL;

    const bool isRxDopplerRangeInputLayout = (m_params.inputLayout == PVA_FFT_LAYOUT_RX_DOPPLER_RANGE);
    const int32_t channels                 = isRxDopplerRangeInputLayout ? inTensorShapeNCHW[2] : inTensorShapeNCHW[1];
    const int32_t chirps                   = isRxDopplerRangeInputLayout ? inTensorShapeNCHW[1] : inTensorShapeNCHW[2];
    const int32_t samples                  = inTensorShapeNCHW[3];

    const int32_t inTensorHeight = chirps;
    const int32_t inTensorWidth  = m_inputIsComplex ? (samples * COMPLEX_ELEMENT_NUM) : samples;

    if (chirps % TILE_BATCH != 0 &&
        (isRxDopplerRangeInputLayout || m_params.outputLayout == PVA_FFT_LAYOUT_RX_DOPPLER_RANGE ||
         m_params.outputLayout == PVA_FFT_LAYOUT_RANGE_RX_DOPPLER))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Chirps must be a multiple of %d", TILE_BATCH);
    }

    if (isRxDopplerRangeInputLayout && (channels * chirps) > (kMaxTilesPerDim * TILE_BATCH))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "RX_DOPPLER_RANGE input supports at most %d Rx channels * chirps; got %d",
                              kMaxTilesPerDim * TILE_BATCH, channels * chirps);
    }

    int64_t outTensorShapeNCHW[4];
    nvcvTensorShapePermute(outTensorRequirements->layout, outTensorRequirements->shape, NVCV_TENSOR_NCHW,
                           outTensorShapeNCHW);

    const bool isRxDopplerRangeOutputLayout = (m_params.outputLayout == PVA_FFT_LAYOUT_RX_DOPPLER_RANGE);
    // RANGE_RX_DOPPLER is HCW[Nb, NofRx, Nr]: chirps in W, range bins in H.
    const bool isRangeRxDopplerOutputLayout = (m_params.outputLayout == PVA_FFT_LAYOUT_RANGE_RX_DOPPLER);
    const int32_t outTensorWidth =
        isRangeRxDopplerOutputLayout ? (chirps * COMPLEX_ELEMENT_NUM) : (outTensorShapeNCHW[3] * COMPLEX_ELEMENT_NUM);
    // The chirp count in every layout; only which NCHW axis holds it differs.
    const int32_t outTensorHeight = isRangeRxDopplerOutputLayout   ? outTensorShapeNCHW[3]
                                    : isRxDopplerRangeOutputLayout ? outTensorShapeNCHW[1]
                                                                   : outTensorShapeNCHW[2];
    const int32_t outRangeBins =
        isRangeRxDopplerOutputLayout ? outTensorShapeNCHW[2] : (outTensorWidth / COMPLEX_ELEMENT_NUM);

    if (isRangeRxDopplerOutputLayout)
    {
        // Only the real path's post process writes the chirp-innermost tile.
        if (m_inputIsComplex)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "RANGE_RX_DOPPLER output is not supported for complex input");
        }
        // The output row is NofRx * Nr complex; the tile limit checked below keeps it within the line pitch.
        static_assert(kMaxTilesPerDim * TILE_BATCH * TILE_GROUPS_MAX * COMPLEX_ELEMENT_NUM <= kMaxLinePitchElems,
                      "RANGE_RX_DOPPLER tile limit must imply the line-pitch limit");
    }

    // Real-input path: pack pairs of real samples as complex, factorize N/2, post-process to recover N/2+1 bins.
    // Complex-input path: factorize N directly, no post-process, output N bins.
    if (m_inputIsComplex && samples > MAX_COMPLEX_FFT_SIZE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Complex-input range FFT supports at most %d samples per chirp; got %d",
                              MAX_COMPLEX_FFT_SIZE, samples);
    }
    if (!m_inputIsComplex && samples > IN_TILE_WIDTH)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Real-input range FFT supports at most %d samples per chirp; got %d", IN_TILE_WIDTH,
                              samples);
    }
    if (!m_inputIsComplex && samples % 2 != 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Real-input range FFT supports only even number of samples; got %d", samples);
    }
    // Complex path: transposeOutput=true so the digit_reverse_transpose stage's 6D wrapper iters/strides are
    // configured for the transposed output layout required by the RDF/TRANS_MODE_2 output DMA.
    FFTFactorization factorization{m_inputIsComplex ? priv::FactorizeFFTSize(samples, true, false)
                                                    : priv::FactorizeFFTSize(samples / 2, false, true)};

    if (!factorization.isValid)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported sample number %d", samples);
    }

    if (factorization.twiddleSize > TWIDDLE_SIZE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Twiddle size is too large");
    }

    int64_t winTensorShapeNCHW[4];
    nvcvTensorShapePermute(winTensorRequirements->layout, winTensorRequirements->shape, NVCV_TENSOR_NCHW,
                           winTensorShapeNCHW);

    const int32_t windowSize = winTensorShapeNCHW[3];

    if (windowSize != samples)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Window size must be equal to sample number");
    }

    const int32_t maxRangeBins = m_inputIsComplex ? samples : RANGE_BINS_REAL_FFT_CALCULATE(samples);
    if (outRangeBins > maxRangeBins)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output range bins can't be greater than %d",
                              maxRangeBins);
    }

    const int32_t outNofRx  = isRxDopplerRangeOutputLayout ? outTensorShapeNCHW[2] : outTensorShapeNCHW[1];
    const int32_t outChirps = isRangeRxDopplerOutputLayout   ? outTensorShapeNCHW[3]
                              : isRxDopplerRangeOutputLayout ? outTensorShapeNCHW[1]
                                                             : outTensorShapeNCHW[2];
    if (outNofRx != channels)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output H dimension must equal number of Rx channels (%d)", channels);
    }
    if (outChirps != chirps)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output chirps must be equal to input chirps");
    }

    // Tile arena width for the destination DMA: real path caps at (N/2)+1 complex bins, complex path uses full N.
    const int32_t dstArenaWidth = maxRangeBins * COMPLEX_ELEMENT_NUM;

    // Wide-tile packing: the largest G whose chirp blocks divide evenly and whose padded tiles fit VMEM.
    auto transposePitchInts = [](int32_t widthInts) {
        return static_cast<int32_t>(TRANSPOSE_LINE_PITCH_ALIGNED(widthInts * static_cast<int32_t>(sizeof(int32_t)),
                                                                 TRANS_MODE_2, sizeof(int32_t))) /
               static_cast<int32_t>(sizeof(int32_t));
    };

    const int32_t fftStages =
        factorization.groups[RADIX5_GROUP_IDX].numStages + factorization.groups[RADIX3_GROUP_IDX].numStages +
        factorization.groups[RADIX4_GROUP_IDX].numStages + factorization.groups[RADIX4_GROUP_IDX].isOdd;
    const bool canFuse = (fftStages <= FFT_MAX_FUSED_STAGES);

    // The RX_DOPPLER_RANGE SQDF drain (past kMaxTilesPerDim tiles) needs every bin, unpadded rows and a tile count
    // that splits into an even count of at most kMaxTilesPerDim times a count of at most kMaxTilesPerDim. Returns
    // the even count, or 0 if there is none.
    const int32_t sqdfRowBytes = outTensorWidth * static_cast<int32_t>(sizeof(TO));
    const bool sqdfRowUnpadded =
        (static_cast<int32_t>(TRANSPOSE_LINE_PITCH_ALIGNED(sqdfRowBytes, TRANS_MODE_2, sizeof(TO))) == sqdfRowBytes);
    auto sqdfSplit = [&](int32_t nTiles, int32_t tileChirpCount) {
        const int64_t tileBytes = static_cast<int64_t>(sqdfRowBytes) * tileChirpCount;
        for (int32_t d = kMaxTilesPerDim; d >= 2; d -= 2)
        {
            if ((nTiles % d == 0) && (nTiles / d <= kMaxTilesPerDim) && (d * tileBytes <= kMaxAdvBytes))
            {
                return d;
            }
        }
        return 0;
    };

    int32_t groupsPerTile = 1;
    if (canFuse)
    {
        for (int32_t g = TILE_GROUPS_MAX; g > 1; g--)
        {
            const bool tilesDivide = (chirps % (TILE_BATCH * g) == 0);
            const bool srcFits     = (g * transposePitchInts(inTensorWidth) <= transposePitchInts(IN_TILE_WIDTH));
            // The chirp-innermost tile, TILE_BATCH * g chirps by every bin, must fit a ping-pong half.
            const bool dstFits = isRangeRxDopplerOutputLayout
                                     ? ((maxRangeBins * TILE_BATCH * g * COMPLEX_ELEMENT_NUM) <=
                                        (OUT_TILE_HEIGHT * transposePitchInts(OUT_TILE_WIDTH)))
                                     : (g * transposePitchInts(dstArenaWidth) <= transposePitchInts(OUT_TILE_WIDTH));
            /* The real path carries n + 1 elements per group; see midGroupStride in range_fft_knl.c. */
            const int32_t midPerGroup = m_inputIsComplex ? factorization.n : (factorization.n + 1);
            const bool midFits        = (g * midPerGroup <= TILE_HEIGHT);

            const int32_t gTiles = (channels * chirps) / (TILE_BATCH * g);
            const bool drainOk =
                !isRxDopplerRangeOutputLayout || (gTiles <= kMaxTilesPerDim) ||
                ((outRangeBins == maxRangeBins) && sqdfRowUnpadded && (sqdfSplit(gTiles, TILE_BATCH * g) != 0));

            if (tilesDivide && srcFits && dstFits && midFits && drainOk)
            {
                groupsPerTile = g;
                break;
            }
        }
    }
    // Chirps carried by one DMA tile.
    const int32_t tileChirps = TILE_BATCH * groupsPerTile;
    // Output tile rows: a block of chirps, or every bin for the chirp-innermost RANGE_RX_DOPPLER.
    const int32_t outTileRows = isRangeRxDopplerOutputLayout ? maxRangeBins : tileChirps;

    // A DOPPLER_RX_RANGE tensor is tiled tileChirps chirps at a time along one dataflow dimension.
    const bool chirpMajorEnd = (m_params.inputLayout == PVA_FFT_LAYOUT_DOPPLER_RX_RANGE) ||
                               (m_params.outputLayout == PVA_FFT_LAYOUT_DOPPLER_RX_RANGE);
    if (chirpMajorEnd && (channels > kMaxTilesPerDim))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "DOPPLER_RX_RANGE input or output supports at most %d Rx channels; got %d",
                              kMaxTilesPerDim, channels);
    }
    // A DOPPLER_RX_RANGE row holds every channel's samples or range bins.
    if ((m_params.inputLayout == PVA_FFT_LAYOUT_DOPPLER_RX_RANGE) &&
        (static_cast<int64_t>(channels) * inTensorWidth > kMaxLinePitchElems))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "DOPPLER_RX_RANGE input supports at most %lld Rx channels * samples; got %d * %d",
                              static_cast<long long>(kMaxLinePitchElems / (inTensorWidth / samples)), channels,
                              samples);
    }
    if ((m_params.outputLayout == PVA_FFT_LAYOUT_DOPPLER_RX_RANGE) &&
        (static_cast<int64_t>(channels) * outTensorWidth > kMaxLinePitchElems))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "DOPPLER_RX_RANGE output supports at most %lld Rx channels * range bins; got %d * %d",
                              static_cast<long long>(kMaxLinePitchElems / COMPLEX_ELEMENT_NUM), channels, outRangeBins);
    }
    if (chirpMajorEnd && (chirps > kMaxTilesPerDim * tileChirps))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "DOPPLER_RX_RANGE input or output supports at most %d chirps for this configuration; got %d",
            kMaxTilesPerDim * tileChirps, chirps);
    }

    if (isRangeRxDopplerOutputLayout)
    {
        // The output tiles all advance along the dataflow's first dimension.
        const int32_t outTiles = (channels * chirps) / tileChirps;
        if (outTiles > kMaxTilesPerDim)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "RANGE_RX_DOPPLER output supports at most %d Rx channels * chirps for this "
                                  "configuration; got %d * %d",
                                  kMaxTilesPerDim * tileChirps, channels, chirps);
        }
    }

    const int32_t nOutTiles = (channels * chirps) / tileChirps;
    const bool useSqdf      = isRxDopplerRangeOutputLayout && (nOutTiles > kMaxTilesPerDim);
    int32_t sqdfDim1        = 0;
    // One double-buffer slot of the SQDF drain: tileChirps unpadded rows of outTensorWidth elements.
    int32_t sqdfTileBytes = 0;
    if (useSqdf)
    {
        if (outRangeBins != maxRangeBins)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "RX_DOPPLER_RANGE output with more than %d tiles (Rx channels * chirps / %d) "
                                  "requires all %d range bins; got %d",
                                  kMaxTilesPerDim, tileChirps, maxRangeBins, outRangeBins);
        }

        if (!sqdfRowUnpadded)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "RX_DOPPLER_RANGE output with more than %d tiles (Rx channels * chirps / %d) "
                                  "requires an output row with no transpose padding, so the number of samples must "
                                  "be %s; got %d",
                                  kMaxTilesPerDim, tileChirps,
                                  m_inputIsComplex ? "one more than a multiple of 8" : "a multiple of 16", samples);
        }

        sqdfTileBytes = sqdfRowBytes * tileChirps;
        sqdfDim1      = sqdfSplit(nOutTiles, tileChirps);
        if (sqdfDim1 == 0)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "RX_DOPPLER_RANGE output of %d tiles (Rx channels * chirps / %d) cannot be split "
                                  "into an even count of at most %d times a count of at most %d, with the even count "
                                  "times the %d-byte tile no greater than %lld bytes",
                                  nOutTiles, tileChirps, kMaxTilesPerDim, kMaxTilesPerDim, sqdfTileBytes,
                                  static_cast<long long>(kMaxAdvBytes));
        }
    }

    m_twiddle_d        = (int32_t *)mem::Alloc(factorization.twiddleSize * sizeof(int32_t));
    int32_t *twiddle_h = (int32_t *)mem::GetHostPointer(m_twiddle_d);

    // Real path uses samples/2 complex points (pairs packed). Complex path uses N complex points directly.
    const int32_t fftN = m_inputIsComplex ? samples : (samples / 2);
    std::vector<int32_t> twiddle_interleaved(2 * fftN);
    for (int32_t i = 0; i < fftN; i++)
    {
        double angle                   = 2.0f * M_PI * i / fftN;
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

    // Real path needs an extra post-process twiddle stage to recover N/2+1 distinct bins from the N/2-point complex
    // FFT.  Complex path skips this — the N-point complex FFT already produces the final N bins.
    if (!m_inputIsComplex)
    {
        generateComplexFFTTwiddlePostProcess(twiddle_h + factorization.twiddleOffsets[2], TWIDDLE_QBITS, TRUNCATE_QBITS,
                                             factorization);
    }

    m_exec    = GetRangeFFTExec();
    auto prog = CmdProgram::Create(*m_exec.get());

    int32_t phase = 0;

    auto coeffHdl = prog["coeffHdl"];

    int32_t *twiddle_v = prog["twiddleBuf"].ptr<int32_t>();
    TI *window_v       = prog["windowBuf"].ptr<TI>();

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

    const int32_t inImageWidth     = isRxDopplerRangeInputLayout ? inTensorWidth : (channels * inTensorWidth);
    const int32_t inImageHeight    = isRxDopplerRangeInputLayout ? (channels * inTensorHeight) : inTensorHeight;
    const int32_t inImageLinePitch = inImageWidth;

    const int32_t inTileWidth  = inTensorWidth;
    const int32_t inTileHeight = IN_TILE_HEIGHT * groupsPerTile;

    auto srcHdl = prog["srcHdl"];
    TI *src_v   = prog["srcBuf"].ptr<TI>();

    RasterDataFlow &srcDataFlow = prog.addDataFlowHead<RasterDataFlow>(phase);
    srcDataFlow.handler(srcHdl)
        .src(m_srcOp, inImageWidth, inImageHeight, inImageLinePitch)
        .tileBuffer(src_v)
        .tile(inTileWidth, inTileHeight)
        .transpose(TRANS_MODE_2)
        .scanOrder(cupva::COLUMN_MAJOR);

    const int32_t niterX = channels;
    const int32_t niterY = (chirps + tileChirps - 1) / tileChirps;

    if (isRangeRxDopplerOutputLayout)
    {
        // One row per range bin; a tile is one channel's block of tileChirps chirps by every bin.
        const int32_t outImageWidth = channels * chirps * COMPLEX_ELEMENT_NUM;
        const int32_t outTileWidth  = tileChirps * COMPLEX_ELEMENT_NUM;
        const int32_t outTileHeight = outRangeBins;
        // The post process writes every bin, so the arena holds all; the DMA ships the requested rows.
        const int32_t outArenaHeight = outTileRows;

        auto dstHdl = prog["dstHdl"];
        TO *dst_v   = prog["dstBuf"].ptr<TO>();

        /* Default row-major scan: one row band, so tiles advance across columns as (ch0, blk0),
         * (ch0, blk1), ... (ch1, blk0), which is the order the input's COLUMN_MAJOR scan produces. */
        RasterDataFlow &dstDataFlow = prog.addDataFlowHead<RasterDataFlow>(phase);
        dstDataFlow.handler(dstHdl)
            .tileBuffer(dst_v)
            .dst(m_dstOp, outImageWidth, outRangeBins, outImageWidth)
            .tile(outTileWidth, outTileHeight)
            .tileArena(outTileWidth, outArenaHeight);

        prog["useRxDrSqdf"] = static_cast<int32_t>(0);
    }
    else if (isRxDopplerRangeOutputLayout)
    {
        const int32_t outTileWidth  = outTensorWidth;
        const int32_t outTileHeight = OUT_TILE_HEIGHT * groupsPerTile;

        auto dstHdl = prog["dstHdl"];
        TO *dst_v   = prog["dstBuf"].ptr<TO>();

        if (useSqdf)
        {
            const int32_t dstLinePitchVal = outTileWidth;
            const int32_t outLaneOfstVal  = (outTileWidth * static_cast<int32_t>(sizeof(TO)) - 8) / 64;
            prog["dstLinePitchVar"]       = dstLinePitchVal;
            prog["outLaneOfstVar"]        = outLaneOfstVal;

            const int32_t dim1_n = sqdfDim1;
            const int32_t dim2_n = nOutTiles / dim1_n;

            auto dstSqdfHdl = prog["dstSqdfHdl"];

            SequenceDataFlow &dstSqdfFlow = prog.addDataFlowHead<SequenceDataFlow>(phase);
            auto &xfer =
                dstSqdfFlow.handler(dstSqdfHdl)
                    .addTransfer()
                    .src(dst_v, sqdfTileBytes) // flat copy: pitch = sqdfTileBytes (height=1)
                    .dst(m_dstOp, sqdfTileBytes)
                    .tile(sqdfTileBytes, 1) // flat: sqdfTileBytes wide × 1 row
                    .mode(TransferModeType::TILE)
                    .srcDim1(2, sqdfTileBytes)       // ping-pong: slot 0 (offset 0) and slot 1 (offset sqdfTileBytes)
                    .srcDim2(dim1_n / 2, 0)          // inner wrap: repeat ping-pong dim1_n/2 times
                    .dstDim1(dim1_n, sqdfTileBytes); // dim1_n consecutive tiles in DRAM

            if (dim2_n > 1)
            {
                xfer.srcDim3(dim2_n, 0)                                             // outer VMEM repeat
                    .dstDim2(dim2_n, static_cast<int64_t>(dim1_n) * sqdfTileBytes); // outer DRAM advance
            }

            prog["useRxDrSqdf"] = static_cast<int32_t>(1);
        }
        else
        {
            // Standard path: full-image RDF (nTiles ≤ kMaxTilesPerDim=256).
            const int32_t outImageHeight = channels * outTensorHeight;

            RasterDataFlow &dstDataFlow = prog.addDataFlowHead<RasterDataFlow>(phase);
            dstDataFlow.handler(dstHdl)
                .tileBuffer(dst_v)
                .dst(m_dstOp, outTileWidth, outImageHeight, outTileWidth)
                .tile(outTileWidth, outTileHeight)
                .tileArena(dstArenaWidth, outTileHeight)
                .transpose(TRANS_MODE_2)
                .scanOrder(cupva::COLUMN_MAJOR);

            prog["useRxDrSqdf"] = static_cast<int32_t>(0);
        }
    }
    else
    {
        const int32_t outImageWidth     = channels * outTensorWidth;
        const int32_t outImageHeight    = outTensorHeight;
        const int32_t outImageLinePitch = outImageWidth;

        // For default layouts: output is [chirps] rows × [NofRx * rangeBins] cols (short, wide).
        const int32_t outTileWidth  = outTensorWidth;
        const int32_t outTileHeight = OUT_TILE_HEIGHT * groupsPerTile;

        // Set up data flow
        auto dstHdl = prog["dstHdl"];

        TO *dst_v = prog["dstBuf"].ptr<TO>();

        RasterDataFlow &dstDataFlow = prog.addDataFlowHead<RasterDataFlow>(phase);
        dstDataFlow.handler(dstHdl)
            .tileBuffer(dst_v)
            .dst(m_dstOp, outImageWidth, outImageHeight, outImageLinePitch)
            .tile(outTileWidth, outTileHeight)
            .tileArena(dstArenaWidth, outTileHeight)
            .transpose(TRANS_MODE_2)
            .scanOrder(cupva::COLUMN_MAJOR);

        prog["useRxDrSqdf"] = static_cast<int32_t>(0);
    }

    // Configure program parameters
    prog["groupsPerTile"]     = static_cast<int32_t>(groupsPerTile);
    prog["outChirpInnermost"] = static_cast<int32_t>(isRangeRxDopplerOutputLayout ? 1 : 0);
    prog["outTileRows"]       = static_cast<int32_t>(outTileRows);
    prog["niterX"]            = static_cast<int32_t>(niterX);
    prog["niterY"]            = static_cast<int32_t>(niterY);
    prog["inputIsComplex"]    = static_cast<int32_t>(m_inputIsComplex ? 1 : 0);
    prog["factorization"].set(&factorization, 1);

    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename TI, typename TO>
template<typename S>
void RangeFFT<TI, TO>::operator()(S stream, const nvcv::Tensor &in, const nvcv::Tensor &win, const nvcv::Tensor &out)
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
            "Input, window and output tensors must have the same requirements that was used to create the operator");
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
        static_assert(!sizeof(S), "Unsupported stream type for RangeFFT operator");
    }

    if (!(err == CUPVA_ERROR_NONE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Submit failed");
    }
}

template<typename TI, typename TO>
RangeFFT<TI, TO>::~RangeFFT()
{
    CupvaCmdDestroy(&m_prog);
    mem::Free(m_twiddle_d);
}

// Explicit instantiation of the template to support the types we care about
template class RangeFFT<int32_t, int32_t>;

} // namespace pva_solutions::priv

NVCVStatus pvaRangeFFTCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements const *const inTensorRequirements,
                             NVCVTensorRequirements const *const winTensorRequirements,
                             NVCVTensorRequirements const *const outTensorRequirements,
                             PVARangeFFTParams const *const params)
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

        const bool isRealInput    = (inTensorRequirements->dtype == NVCV_DATA_TYPE_S32);
        const bool isComplexInput = (inTensorRequirements->dtype == NVCV_DATA_TYPE_2S32);
        const bool outIsComplex   = (outTensorRequirements->dtype == NVCV_DATA_TYPE_2S32);

        if ((isRealInput || isComplexInput) && outIsComplex)
        {
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::RangeFFT<int32_t, int32_t>(
                inTensorRequirements, winTensorRequirements, outTensorRequirements, params));
        }
        else
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input/output data type");
        }
    });
}

template<typename StreamType>
NVCVStatus RangeFFTSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle const in,
                              NVCVTensorHandle const win, NVCVTensorHandle out)
{
    return nvcv::ProtectCall([&] {
        nvcv::TensorWrapHandle input(in), window(win), output(out);

        NVCVDataType inDataType, outDataType;
        nvcvTensorGetDataType(in, &inDataType);
        nvcvTensorGetDataType(out, &outDataType);

        const bool isRealInput    = (inDataType == NVCV_DATA_TYPE_S32);
        const bool isComplexInput = (inDataType == NVCV_DATA_TYPE_2S32);
        const bool outIsComplex   = (outDataType == NVCV_DATA_TYPE_2S32);

        if ((isRealInput || isComplexInput) && outIsComplex)
        {
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::RangeFFT<int32_t, int32_t>>(handle)(stream, input,
                                                                                                       window, output);
        }
        else
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input/output data type");
        }
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus RangeFFTSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle const,
                                                      NVCVTensorHandle const, NVCVTensorHandle);
template NVCVStatus RangeFFTSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle const,
                                                     NVCVTensorHandle const, NVCVTensorHandle);

// C++ overload implementations
NVCVStatus pvaRangeFFTSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle const in,
                             NVCVTensorHandle const win, NVCVTensorHandle out)
{
    return RangeFFTSubmitImpl(handle, stream, in, win, out);
}

NVCVStatus pvaRangeFFTSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle const in,
                             NVCVTensorHandle const win, NVCVTensorHandle out)
{
    return RangeFFTSubmitImpl(handle, stream, in, win, out);
}
