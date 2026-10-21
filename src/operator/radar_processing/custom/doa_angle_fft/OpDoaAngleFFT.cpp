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

/**
 * @file OpDoaAngleFFT.cpp
 * @brief Host-side operator implementation for DOA Angle FFT
 *
 * This operator performs DoA (Direction of Arrival) angle estimation and outputs
 * target angles and target index maps as tensors.
 *
 * This file implements the host-side interface for the DOA (Direction of Arrival)
 * Angle FFT operator, which estimates target azimuth and elevation angles from
 * radar detection snapshots using 2D FFT-based beamforming on the PVA.
 *
 * Main Responsibilities:
 * 1. Tensor requirement calculation and validation
 * 2. Pre-computation and upload of coefficient data:
 *    - FFT twiddle factors (complex exponentials)
 *    - 2D Hanning window coefficients
 *    - Angle bins lookup table (bin index to angle mapping)
 * 3. VPU executable management (loading and caching)
 * 4. Task submission to PVA via cuPVA runtime
 * 5. Stream synchronization for CUDA and cuPVA
 *
 * Input Tensors:
 * - Detection count: [1] int32 scalar
 * - Snapshots: [N, H, W] complex int16 (N detections, H elevation, W azimuth)
 * - Detection list: [N, 2] int32 (range bin, Doppler bin for each detection)
 * - DDM Doppler offsets: [N] float32 (Doppler offset for each detection)
 *
 * Output Tensors:
 * - Target count: [1] int32 scalar (number of targets detected)
 * - Target angles: [H, W] float32 (azimuth and elevation angles for each target)
 * - Target index map: [H, W] int32 (index mapping for each target)
 */

#include "OpDoaAngleFFT.hpp"

#include "device/doa_angle_fft_param.h"

#include <IOperator.hpp>
#include <RadarUtils.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>

#include <OpDoaAngleFFT.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>
#include <nvcv/Tensor.h>

#include <algorithm>
#include <cmath>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_doa_angle_fft_dev);

namespace pva_solutions::priv {

/**
 * @brief Convert angle estimation precision enum to scaling factor.
 *
 * Maps the precision enum value to the corresponding FFT scaling factor:
 *   - LOW (0)     -> 2
 *   - DEFAULT (1) -> 4
 *   - HIGH (2)    -> 8
 *
 * @param precision The angle estimation precision level
 * @return The scaling factor (2, 4, or 8), defaults to 4 for invalid input
 */
static inline int32_t precisionToScalingFactor(PVADoaAngleFFTPrecision precision)
{
    switch (precision)
    {
    case PVA_ANGLE_ESTIMATION_PRECISION_LOW:
        return 2;
    case PVA_ANGLE_ESTIMATION_PRECISION_DEFAULT:
        return 4;
    case PVA_ANGLE_ESTIMATION_PRECISION_HIGH:
        return 8;
    default:
        return 4;
    }
}

template<typename TI, typename TO>
std::mutex DoaAngleFFT<TI, TO>::s_execMapMutex;

template<typename TI, typename TO>
std::map<typename DoaAngleFFT<TI, TO>::ExecKey, std::weak_ptr<cupva::Executable>>
    DoaAngleFFT<TI, TO>::s_DoaAngleFFTExecutables;

template<typename TI, typename TO>
std::shared_ptr<cupva::Executable> DoaAngleFFT<TI, TO>::GetDoaAngleFFTExec()
{
    static IOperator::ExecutableData sDoaAngleFFTExecData;

    sDoaAngleFFTExecData = {CUPVA_EXECUTABLE_DATA(pva_doa_angle_fft_dev), CUPVA_EXECUTABLE_SIZE(pva_doa_angle_fft_dev)};

    ExecKey execKey = std::make_tuple(cupva::Context::GetCurrent());
    return IOperator::GetExecutable<ExecKey>(sDoaAngleFFTExecData, execKey, s_DoaAngleFFTExecutables, s_execMapMutex);
}

/**
  * @brief Check if FFT size is supported for azimuth or elevation dimension
  *
  * @param size FFT size to check
  * @param isAzimuth true for azimuth dimension, false for elevation
  * @return true if size is supported, false otherwise
  *
  * Currently supported sizes:
  * - Azimuth: 32, 64, 128
  * - Elevation: 8, 16, 32
  */
template<typename TI, typename TO>
bool DoaAngleFFT<TI, TO>::isSupportedFFTSize(int32_t size, bool isAzimuth)
{
    // The shared power digit reverse handles sizes 8 to 256 (log2N in [3, 8]).
    static_assert((MAX_AZIMUTH_FFT_SIZE <= 256) && (MAX_ELEVATION_FFT_SIZE <= 256), "Angle FFT size above 256");
    if (isAzimuth)
    {
        switch (size)
        {
        case ANGLE_FFT_SIZE_32:
        case ANGLE_FFT_SIZE_64:
        case ANGLE_FFT_SIZE_128:
            return true;
        default:
            return false;
        }
    }
    else
    {
        switch (size)
        {
        case ANGLE_FFT_SIZE_8:
        case ANGLE_FFT_SIZE_16:
        case ANGLE_FFT_SIZE_32:
            return true;
        default:
            return false;
        }
    }
}

/**
  * @brief Round a fixed-point value by shifting right with rounding
  *
  * Performs: round(x / 2^shift) = (x + 2^(shift-1)) >> shift
  *
  * @param x Fixed-point value to round
  * @param shift Number of fractional bits to remove
  * @return Rounded and scaled value
  */
static inline int64_t my_round(int64_t x, int32_t shift)
{
    int64_t x_rounded = x;
    if (shift != 0)
    {
        int64_t half = 1 << (shift - 1);
        x_rounded    = (x + half) >> shift;
    }
    return x_rounded;
}

/**
  * @brief Initialize angle bins lookup table
  *
  * Pre-computes the mapping from FFT bin indices to physical angles in degrees.
  * The angles are stored in Q16 fixed-point format (16 fractional bits).
  *
  * Formula:
  * - Azimuth: angle[i] = arcsin((i / N) * 2 - 1) * (180 / π)
  * - Elevation: angle[i] = arcsin((i / N) * 2 - 1) * (180 / π)
  *
  * @param paddedAzimuthSize Padded azimuth FFT size (for alignment)
  * @param paddedElevationSize Padded elevation FFT size (for alignment)
  * @param angleBins [out] Pre-allocated buffer for angle bins (azimuth + elevation)
  */
void InitializeAngleBins(int32_t paddedAzimuthSize, int32_t paddedElevationSize, int32_t *angleBins)
{
    int32_t *azimuthBins   = angleBins;
    int32_t *elevationBins = angleBins + paddedAzimuthSize;

    // Generate azimuth angle bins: map FFT bin indices to angles in Q16 format (adjusted for removed conjugate)
    for (int32_t i = 0; i < paddedAzimuthSize; i++)
    {
        float normalizedIndex = 1.0f - 2.0f * (static_cast<float>(i) / static_cast<float>(paddedAzimuthSize));
        float angleDeg        = std::asin(std::max(-1.0f, std::min(1.0f, normalizedIndex))) * 180.0f / M_PI;
        azimuthBins[i]        = static_cast<int32_t>(angleDeg * ANGLE_SCALE);
    }

    // Generate elevation angle bins: map FFT bin indices to angles in Q16 format (adjusted for removed conjugate)
    for (int32_t i = 0; i < paddedElevationSize; i++)
    {
        float normalizedIndex = 1.0f - 2.0f * (static_cast<float>(i) / static_cast<float>(paddedElevationSize));
        float angleDeg        = std::asin(std::max(-1.0f, std::min(1.0f, normalizedIndex))) * 180.0f / M_PI;
        elevationBins[i]      = static_cast<int32_t>(angleDeg * ANGLE_SCALE);
    }
}

template<typename TI, typename TO>
void DoaAngleFFT<TI, TO>::generateComplexFFTTwiddle(int32_t n, int32_t *twiddle)
{
    std::vector<int32_t> twiddle_interleaved(2 * n);

    for (int32_t i = 0; i < n; i++)
    {
        double angle                   = -2.0 * M_PI * i / n;
        twiddle_interleaved[2 * i]     = (int32_t)round(cos(angle) * (1UL << TWIDDLE_QBITS));
        twiddle_interleaved[2 * i + 1] = (int32_t)round(sin(angle) * (1UL << TWIDDLE_QBITS));
    }

    const int32_t dist = n / RADIX_NUM;

    for (int32_t j = 0; j < dist; j++)
    {
        const int32_t t1idx = j;
        const int32_t t2idx = 2 * t1idx;
        const int32_t t3idx = 3 * t1idx;

        // Read twiddle factors
        *twiddle++ = twiddle_interleaved[2 * t1idx];
        *twiddle++ = twiddle_interleaved[2 * t1idx + 1];
        *twiddle++ = twiddle_interleaved[2 * t2idx];
        *twiddle++ = twiddle_interleaved[2 * t2idx + 1];
        *twiddle++ = twiddle_interleaved[2 * t3idx];
        *twiddle++ = twiddle_interleaved[2 * t3idx + 1];
    }
}

template<typename TI, typename TO>
void DoaAngleFFT<TI, TO>::generateHannningWindowCoefficients(int32_t numVirtualAzimuthElements,
                                                             int32_t numVirtualElevationElements, int32_t pitch,
                                                             int32_t *window)
{
    const int64_t WINDOW_SCALE = (1UL << WINDOW_QBITS);

    for (int32_t h = 0; h < numVirtualElevationElements; h++)
    {
        for (int32_t w = 0; w < numVirtualAzimuthElements; w++)
        {
            float hanningValueAz = 0.5f * (1.0f - std::cos(2.0f * M_PI * static_cast<float>(w + 1) /
                                                           static_cast<float>(numVirtualAzimuthElements + 1)));
            int32_t azWindow     = static_cast<int32_t>(hanningValueAz * WINDOW_SCALE);

            float hanningValueEl = 0.5f * (1.0f - std::cos(2.0f * M_PI * static_cast<float>(h + 1) /
                                                           static_cast<float>(numVirtualElevationElements + 1)));
            int32_t elWindow     = static_cast<int32_t>(hanningValueEl * WINDOW_SCALE);

            int32_t jointWindow     = my_round((int64_t)azWindow * (int64_t)elWindow, WINDOW_QBITS);
            window[(h * pitch) + w] = static_cast<int32_t>(jointWindow);
        }
    }
}

template<typename TI, typename TO>
DoaAngleFFT<TI, TO>::DoaAngleFFT(NVCVTensorRequirements const *const inSnapshotsTensorRequirements,
                                 NVCVTensorRequirements const *const inDetectionListTensorRequirements,
                                 NVCVTensorRequirements const *const inDdmDopplerOffsetsTensorRequirements,
                                 NVCVTensorRequirements const *const outTargetAnglesTensorRequirements,
                                 NVCVTensorRequirements const *const outTargetIndexMapTensorRequirements,
                                 PVADoaAngleFFTParams const *const params)
    : m_inSnapshotsTensorRequirements{*inSnapshotsTensorRequirements}
    , m_params{params ? *params : kDefaultParams}
    , m_inDetectionCountOp(cupva::OffsetPointer<int32_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_inSnapshotsOp(cupva::OffsetPointer<TI>::Create(cupva::MemType::DRAM, nullptr))
    , m_outTargetCountOp(cupva::OffsetPointer<int32_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_outTargetAnglesOp(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr))
    , m_outTargetIndexMapOp(cupva::OffsetPointer<int32_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_twiddle_d(nullptr)
    , m_window_d(nullptr)
    , m_prog{}
{
    // The DMA strides are derived from the shapes.
    pva_solutions::utils::CheckDenselyPacked(*inSnapshotsTensorRequirements, "Snapshots");
    pva_solutions::utils::CheckDenselyPacked(*inDetectionListTensorRequirements, "Detection list");
    pva_solutions::utils::CheckDenselyPacked(*inDdmDopplerOffsetsTensorRequirements, "DDM Doppler offsets");
    pva_solutions::utils::CheckDenselyPacked(*outTargetAnglesTensorRequirements, "Target angles");
    pva_solutions::utils::CheckDenselyPacked(*outTargetIndexMapTensorRequirements, "Target index map");

    int64_t inSnapshotsTensorShapeNCHW[4];
    nvcvTensorShapePermute(inSnapshotsTensorRequirements->layout, inSnapshotsTensorRequirements->shape,
                           NVCV_TENSOR_NCHW, inSnapshotsTensorShapeNCHW);

    const int32_t maxNumOfTargets = inSnapshotsTensorShapeNCHW[1];
    const int32_t numOfTx         = inSnapshotsTensorShapeNCHW[2];
    const int32_t numOfRx         = inSnapshotsTensorShapeNCHW[3];

    const int32_t inSnapshotsTensorHeight = inSnapshotsTensorShapeNCHW[2];
    const int32_t inSnapshotsTensorWidth  = inSnapshotsTensorShapeNCHW[3] * COMPLEX_ELEMENT_NUM;

    // Compute FFT sizes from the parameters using the precision-to-scaling-factor conversion
    const int32_t scalingFactor       = precisionToScalingFactor(m_params.angleEstimationPrecision);
    const int32_t paddedAzimuthSize   = m_params.numVirtualAzimuthElements * scalingFactor;
    const int32_t paddedElevationSize = m_params.numVirtualElevationElements * scalingFactor;
    const int32_t windowSize          = paddedAzimuthSize * paddedElevationSize;

    /* input snapshots verification */
    if (maxNumOfTargets > MAX_SUPPORTED_TARGET_COUNT)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "maximum number of targets must be less than or equal to 8192");
    }

    if (8 != numOfTx)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Number of Tx antennas must be equal to 8");
    }

    if (8 != numOfRx)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Number of Rx antennas must be equal to 8");
    }

    /* FFT parameters verification */
    if (m_params.numVirtualAzimuthElements != 16)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Number of virtual azimuth elements must be equal to 16");
    }

    if (m_params.numVirtualElevationElements != 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Number of virtual elevation elements must be equal to 4");
    }

    if (!isSupportedFFTSize(paddedAzimuthSize, true))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Computed azimuth FFT size (numVirtualAzimuthElements * scalingFactor) must be 32 or 64 or 128");
    }
    if (!isSupportedFFTSize(paddedElevationSize, false))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Computed elevation FFT size (numVirtualElevationElements * scalingFactor) must be 8 or 16 or 32");
    }

    // The device runs the shared radix-4 engine from these and cannot report a bad factorization.
    const FFTFactorization azimuthFactorization   = priv::FactorizeFFTSize(paddedAzimuthSize);
    const FFTFactorization elevationFactorization = priv::FactorizeFFTSize(paddedElevationSize);
    if ((azimuthFactorization.isValid == 0) || (elevationFactorization.isValid == 0))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Failed to factorize the angle FFT sizes (azimuth %d, elevation %d)", paddedAzimuthSize,
                              paddedElevationSize);
    }
    // The device implements power-of-two sizes only.
    if ((azimuthFactorization.groups[RADIX3_GROUP_IDX].numStages != 0) ||
        (azimuthFactorization.groups[RADIX5_GROUP_IDX].numStages != 0) ||
        (elevationFactorization.groups[RADIX3_GROUP_IDX].numStages != 0) ||
        (elevationFactorization.groups[RADIX5_GROUP_IDX].numStages != 0))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Angle FFT sizes must be powers of two (azimuth %d, elevation %d)", paddedAzimuthSize,
                              paddedElevationSize);
    }
    const int32_t fftGroups           = fft_groups_for(paddedAzimuthSize, paddedElevationSize);
    const int32_t inSnapshotGroupSize = std::min(fftGroups, maxNumOfTargets);

    // The last packed slot's windowing reads past its own rows; the input group buffer must hold that span.
    const int32_t lastSlotReadEnd =
        ((fftGroups - 1) * IN_SLOT_SIZE) + (IN_SLOT_HEIGHT * IN_TILE_WIDTH_PAD * (paddedAzimuthSize / 16));
    if (lastSlotReadEnd > IN_GROUP_BUF_SIZE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "angleEstimationPrecision %d (%dx%d angle FFT, %d detections per batch) is not supported",
                              static_cast<int32_t>(m_params.angleEstimationPrecision), paddedAzimuthSize,
                              paddedElevationSize, fftGroups);
    }

    const int32_t twiddleSize_azimuth   = DOA_TWIDDLE_SIZE_CALCULATE(paddedAzimuthSize);
    const int32_t twiddleSize_elevation = DOA_TWIDDLE_SIZE_CALCULATE(paddedElevationSize);
    const int32_t twiddleSize           = twiddleSize_azimuth + twiddleSize_elevation;

    m_twiddle_d        = (int32_t *)mem::Alloc(twiddleSize * sizeof(int32_t));
    int32_t *twiddle_h = (int32_t *)mem::GetHostPointer(m_twiddle_d);
    if (twiddle_h == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_OUT_OF_MEMORY, "Failed to get host pointer for twiddle coefficients");
    }

    generateComplexFFTTwiddle(paddedAzimuthSize, twiddle_h);
    generateComplexFFTTwiddle(paddedElevationSize, twiddle_h + twiddleSize_azimuth);

    // Allocate and generate window coefficients
    m_window_d        = (int32_t *)mem::Alloc(windowSize * sizeof(int32_t));
    int32_t *window_h = (int32_t *)mem::GetHostPointer(m_window_d);
    if (window_h == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_OUT_OF_MEMORY, "Failed to get host pointer for window coefficients");
    }
    memset(window_h, 0, windowSize * sizeof(int32_t));
    generateHannningWindowCoefficients(m_params.numVirtualAzimuthElements, m_params.numVirtualElevationElements,
                                       paddedAzimuthSize, window_h);

    m_angleBins_d        = (int32_t *)mem::Alloc((paddedAzimuthSize + paddedElevationSize) * sizeof(int32_t));
    int32_t *angleBins_h = (int32_t *)mem::GetHostPointer(m_angleBins_d);
    if (angleBins_h == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_OUT_OF_MEMORY, "Failed to get host pointer for angle bins");
    }
    InitializeAngleBins(paddedAzimuthSize, paddedElevationSize, angleBins_h);

    m_exec    = GetDoaAngleFFTExec();
    auto prog = CmdProgram::Create(*m_exec.get());

    int32_t phase = 0;

    auto coeffHdl = prog["coeffHdl"];

    int32_t *twiddle_v        = prog["twiddleBuf"].ptr<int32_t>();
    TI *window_v              = prog["windowBuf"].ptr<TI>();
    int32_t *angleBins_v      = prog["angleBinsBuf"].ptr<int32_t>();
    int32_t *detectionCount_v = prog["detectionCount"].ptr<int32_t>();

    SequenceDataFlow &coeffDataFlow = prog.addDataFlowHead<SequenceDataFlow>(phase).handler(coeffHdl);
    coeffDataFlow.addTransfer()
        .src(m_twiddle_d, twiddleSize * sizeof(int32_t))
        .dst(twiddle_v, twiddleSize * sizeof(int32_t))
        .tile(twiddleSize * sizeof(int32_t), 1)
        .mode(TransferModeType::CONTINUOUS);
    coeffDataFlow.addTransfer()
        .src(m_window_d, windowSize * sizeof(int32_t))
        .dst(window_v, windowSize * sizeof(TI))
        .tile(windowSize * sizeof(TI), 1)
        .mode(TransferModeType::CONTINUOUS);
    coeffDataFlow.addTransfer()
        .src(m_angleBins_d, (paddedAzimuthSize + paddedElevationSize) * sizeof(int32_t))
        .dst(angleBins_v, (paddedAzimuthSize + paddedElevationSize) * sizeof(int32_t))
        .tile((paddedAzimuthSize + paddedElevationSize) * sizeof(int32_t), 1)
        .mode(TransferModeType::CONTINUOUS);
    coeffDataFlow.addTransfer()
        .src(m_inDetectionCountOp, sizeof(int32_t))
        .dst(detectionCount_v, sizeof(int32_t))
        .tile(1 * sizeof(int32_t), 1)
        .mode(TransferModeType::CONTINUOUS);

    phase++;

    const int32_t inSnapshotImageLinePitch = inSnapshotsTensorWidth;

    const int32_t outTargetAnglesImageWidth     = maxNumOfTargets;
    const int32_t outTargetAnglesImageLinePitch = maxNumOfTargets;

    const int32_t outTargetIndexMapImageWidth     = maxNumOfTargets;
    const int32_t outTargetIndexMapImageLinePitch = maxNumOfTargets;

    const int32_t inSnapshotTileWidth     = inSnapshotsTensorWidth;
    const int32_t inSnapshotTileLinePitch = IN_TILE_WIDTH_PAD;
    const int32_t inSnapshotTileHeight    = inSnapshotsTensorHeight;

    int32_t outTargetAnglesTileWidth     = OUT_TARGET_ANGLE_TILE_WIDTH;
    int32_t outTargetAnglesTileLinePitch = OUT_TARGET_ANGLE_TILE_PITCH;
    int32_t outTargetAnglesTileHeight    = OUT_TARGET_ANGLE_TILE_HEIGHT;

    int32_t outTargetIndexMapTileWidth     = OUT_TARGET_INDEX_MAP_TILE_WIDTH;
    int32_t outTargetIndexMapTileLinePitch = OUT_TARGET_INDEX_MAP_TILE_PITCH;
    int32_t outTargetIndexMapTileHeight    = OUT_TARGET_INDEX_MAP_TILE_HEIGHT;

    // Set up data flow
    auto srcSnapshotsHdl      = prog["srcSnapshotsHdl"];
    auto dstTargetAnglesHdl   = prog["dstTargetAnglesHdl"];
    auto dstTargetIndexMapHdl = prog["dstTargetIndexMapHdl"];
    auto dstTargetCountHdl    = prog["dstTargetCountHdl"];

    int32_t *srcSnapshots_v      = prog["srcSnapshotsBuf0"].ptr<int32_t>();
    float *dstTargetAngles_v     = prog["dstTargetAnglesBuf"].ptr<float>();
    int32_t *dstTargetIndexMap_v = prog["dstTargetIndexMapBuf"].ptr<int32_t>();
    int32_t *dstTargetCount_v    = prog["dstTargetCountBuf"].ptr<int32_t>();

    SequenceDataFlow &srcSnapshotsDataFlow = prog.addDataFlowHead<SequenceDataFlow>(phase).handler(srcSnapshotsHdl);
    SequenceDataFlow &dstTargetAnglesDataFlow =
        prog.addDataFlowHead<SequenceDataFlow>(phase).handler(dstTargetAnglesHdl);
    SequenceDataFlow &dstTargetIndexMapDataFlow =
        prog.addDataFlowHead<SequenceDataFlow>(phase).handler(dstTargetIndexMapHdl);
    phase++;
    SequenceDataFlow &dstTargetCountDataFlow = prog.addDataFlowHead<SequenceDataFlow>(phase).handler(dstTargetCountHdl);

    srcSnapshotsDataFlow.addTransfer()
        .src(m_inSnapshotsOp, inSnapshotImageLinePitch * sizeof(int32_t))
        .dst(srcSnapshots_v, inSnapshotTileLinePitch * sizeof(int32_t))
        .tile(inSnapshotTileWidth * sizeof(int32_t), inSnapshotTileHeight * inSnapshotGroupSize);

    int32_t outAnglesWidth =
        (outTargetAnglesImageWidth >= outTargetAnglesTileWidth) ? outTargetAnglesTileWidth : outTargetAnglesImageWidth;
    int32_t outIndexMapWidth = (outTargetIndexMapImageWidth >= outTargetIndexMapTileWidth)
                                   ? outTargetIndexMapTileWidth
                                   : outTargetIndexMapImageWidth;

    dstTargetAnglesDataFlow.addTransfer()
        .src(dstTargetAngles_v, outTargetAnglesTileLinePitch * sizeof(float))
        .dst(m_outTargetAnglesOp, outTargetAnglesImageLinePitch * sizeof(float))
        .tile(outAnglesWidth * sizeof(float), outTargetAnglesTileHeight);

    dstTargetIndexMapDataFlow.addTransfer()
        .src(dstTargetIndexMap_v, outTargetIndexMapTileLinePitch * sizeof(int32_t))
        .dst(m_outTargetIndexMapOp, outTargetIndexMapImageLinePitch * sizeof(int32_t))
        .tile(outIndexMapWidth * sizeof(int32_t), outTargetIndexMapTileHeight);

    // Configure target count data flow
    dstTargetCountDataFlow.addTransfer()
        .src(dstTargetCount_v, OUT_TARGET_COUNT_TILE_PITCH * sizeof(int32_t))
        .dst(m_outTargetCountOp, 1 * sizeof(int32_t))
        .tile(OUT_TARGET_COUNT_TILE_WIDTH * sizeof(int32_t), OUT_TARGET_COUNT_TILE_HEIGHT)
        .mode(TransferModeType::CONTINUOUS);

    // Configure program parameters
    prog["detectionCount"] =
        static_cast<int32_t>(maxNumOfTargets); // Just the initial value, will be updated later in the submit function
    prog["maxTargetCount"]      = static_cast<int32_t>(maxNumOfTargets);
    prog["inTileWidthBytes"]    = static_cast<int32_t>(inSnapshotTileWidth * sizeof(int32_t));
    prog["log2N_AzimuthSize"]   = static_cast<int32_t>(std::log2(paddedAzimuthSize));
    prog["log2N_ElevationSize"] = static_cast<int32_t>(std::log2(paddedElevationSize));

    prog["factorizationAzimuth"].set(&azimuthFactorization, 1);
    prog["factorizationElevation"].set(&elevationFactorization, 1);
    prog["epsilon"] = static_cast<float>(std::numeric_limits<float>::epsilon());

    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename TI, typename TO>
template<typename S>
void DoaAngleFFT<TI, TO>::operator()(S stream, const nvcv::Tensor &inDetectionCountTensor,
                                     const nvcv::Tensor &inSnapshotsTensor, const nvcv::Tensor &inDetectionListTensor,
                                     const nvcv::Tensor &inDdmDopplerOffsetsTensor,
                                     const nvcv::Tensor &outTargetCountTensor,
                                     const nvcv::Tensor &outTargetAnglesTensor,
                                     const nvcv::Tensor &outTargetIndexMapTensor)
{
    auto inDetectionCountTensorData  = inDetectionCountTensor.exportData<nvcv::TensorDataStrided>();
    auto inSnapshotsTensorData       = inSnapshotsTensor.exportData<nvcv::TensorDataStrided>();
    auto outTargetCountTensorData    = outTargetCountTensor.exportData<nvcv::TensorDataStrided>();
    auto outTargetAnglesTensorData   = outTargetAnglesTensor.exportData<nvcv::TensorDataStrided>();
    auto outTargetIndexMapTensorData = outTargetIndexMapTensor.exportData<nvcv::TensorDataStrided>();

    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_inSnapshotsTensorRequirements,
                                                          inSnapshotsTensorData->cdata()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Snapshots tensor must have the same requirements that was used to create the operator");
    }
    if ((static_cast<void *>(inDetectionCountTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(inSnapshotsTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(outTargetCountTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(outTargetAnglesTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(outTargetIndexMapTensorData->basePtr()) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor base pointers must not be NULL");
    }

    m_inDetectionCountOp.setBase(static_cast<int32_t *>(static_cast<void *>(inDetectionCountTensorData->basePtr())));
    m_inSnapshotsOp.setBase(static_cast<TI *>(static_cast<void *>(inSnapshotsTensorData->basePtr())));
    m_outTargetCountOp.setBase(static_cast<int32_t *>(static_cast<void *>(outTargetCountTensorData->basePtr())));
    m_outTargetAnglesOp.setBase(static_cast<float *>(static_cast<void *>(outTargetAnglesTensorData->basePtr())));
    m_outTargetIndexMapOp.setBase(static_cast<int32_t *>(static_cast<void *>(outTargetIndexMapTensorData->basePtr())));

    // Set external memory pointers for SQDF
    cupvaParameter_t srcSnapshotsExtPtr, dstTargetAnglesExtPtr, dstTargetIndexMapExtPtr;
    CupvaCmdProgramGetParameter(&m_prog, &srcSnapshotsExtPtr, "srcSnapshotsExtPtr");
    CupvaCmdProgramGetParameter(&m_prog, &dstTargetAnglesExtPtr, "dstTargetAnglesExtPtr");
    CupvaCmdProgramGetParameter(&m_prog, &dstTargetIndexMapExtPtr, "dstTargetIndexMapExtPtr");

    {
        CupvaParameterSetValuePointer(&srcSnapshotsExtPtr, static_cast<void *>(inSnapshotsTensorData->basePtr()));
        CupvaParameterSetValuePointer(&dstTargetAnglesExtPtr,
                                      static_cast<void *>(outTargetAnglesTensorData->basePtr()));
        CupvaParameterSetValuePointer(&dstTargetIndexMapExtPtr,
                                      static_cast<void *>(outTargetIndexMapTensorData->basePtr()));

        cupvaError_t err = CupvaCmdProgramUpdateDataFlows(&m_prog);
        if (err != CUPVA_ERROR_NONE)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "Failed to update data flows");
        }

        // Submit command: main prog only
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
            static_assert(!sizeof(S), "Unsupported stream type for DoaAngleFFT operator");
        }

        if (err != CUPVA_ERROR_NONE)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Submit failed");
        }
    }
}

template<typename TI, typename TO>
DoaAngleFFT<TI, TO>::~DoaAngleFFT()
{
    CupvaCmdDestroy(&m_prog);

    mem::Free(m_twiddle_d);
    mem::Free(m_window_d);
    mem::Free(m_angleBins_d);
}

// Explicit instantiation of the template to support the types we care about
template class DoaAngleFFT<int32_t, float>;

} // namespace pva_solutions::priv

// Create function
NVCVStatus pvaDoaAngleFFTCreate(NVCVOperatorHandle *handle,
                                NVCVTensorRequirements const *const inSnapshotsTensorRequirements,
                                NVCVTensorRequirements const *const inDetectionListTensorRequirements,
                                NVCVTensorRequirements const *const inDdmDopplerOffsetsTensorRequirements,
                                NVCVTensorRequirements const *const outTargetAnglesTensorRequirements,
                                NVCVTensorRequirements const *const outTargetIndexMapTensorRequirements,
                                PVADoaAngleFFTParams const *const params)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        if (inSnapshotsTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to snapshots NVCVTensorRequirements must not be NULL");
        }
        if (inDetectionListTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to detection list NVCVTensorRequirements must not be NULL");
        }
        if (inDdmDopplerOffsetsTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to DDM Doppler offsets NVCVTensorRequirements must not be NULL");
        }
        if (outTargetAnglesTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to target angles NVCVTensorRequirements must not be NULL");
        }
        if (outTargetIndexMapTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to target index map NVCVTensorRequirements must not be NULL");
        }

        if ((inSnapshotsTensorRequirements->dtype == NVCV_DATA_TYPE_2S32) &&
            (inDetectionListTensorRequirements->dtype == NVCV_DATA_TYPE_S32) &&
            (inDdmDopplerOffsetsTensorRequirements->dtype == NVCV_DATA_TYPE_F32) &&
            (outTargetAnglesTensorRequirements->dtype == NVCV_DATA_TYPE_F32) &&
            (outTargetIndexMapTensorRequirements->dtype == NVCV_DATA_TYPE_S32))
        {
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::DoaAngleFFT<int32_t, float>(
                inSnapshotsTensorRequirements, inDetectionListTensorRequirements, inDdmDopplerOffsetsTensorRequirements,
                outTargetAnglesTensorRequirements, outTargetIndexMapTensorRequirements, params));
        }
        else
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input/output data type");
        }
    });
}

// Submit implementation
template<typename StreamType>
NVCVStatus DoaAngleFFTSubmitImpl(NVCVOperatorHandle handle, StreamType stream,
                                 NVCVTensorHandle const inDetectionCountTensor,
                                 NVCVTensorHandle const inSnapshotsTensor, NVCVTensorHandle const inDetectionListTensor,
                                 NVCVTensorHandle const inDdmDopplerOffsetsTensor,
                                 NVCVTensorHandle outTargetCountTensor, NVCVTensorHandle outTargetAnglesTensor,
                                 NVCVTensorHandle outTargetIndexMapTensor)
{
    return nvcv::ProtectCall([&] {
        nvcv::TensorWrapHandle inDetectionCount(inDetectionCountTensor), inSnapshots(inSnapshotsTensor),
            inDetectionList(inDetectionListTensor), inDdmDopplerOffsets(inDdmDopplerOffsetsTensor),
            outTargetCount(outTargetCountTensor), outTargetAngles(outTargetAnglesTensor),
            outTargetIndexMap(outTargetIndexMapTensor);

        NVCVDataType inDetectionCountTensorDataType, inSnapshotsTensorDataType, inDetectionListTensorDataType,
            inDdmDopplerOffsetsTensorDataType, outTargetCountTensorDataType, outTargetAnglesTensorDataType,
            outTargetIndexMapTensorDataType;
        nvcvTensorGetDataType(inDetectionCountTensor, &inDetectionCountTensorDataType);
        nvcvTensorGetDataType(inSnapshotsTensor, &inSnapshotsTensorDataType);
        nvcvTensorGetDataType(inDetectionListTensor, &inDetectionListTensorDataType);
        nvcvTensorGetDataType(inDdmDopplerOffsetsTensor, &inDdmDopplerOffsetsTensorDataType);
        nvcvTensorGetDataType(outTargetCountTensor, &outTargetCountTensorDataType);
        nvcvTensorGetDataType(outTargetAnglesTensor, &outTargetAnglesTensorDataType);
        nvcvTensorGetDataType(outTargetIndexMapTensor, &outTargetIndexMapTensorDataType);

        if ((inDetectionCountTensorDataType == NVCV_DATA_TYPE_S32) &&
            (inSnapshotsTensorDataType == NVCV_DATA_TYPE_2S32) &&
            (inDetectionListTensorDataType == NVCV_DATA_TYPE_S32) &&
            (inDdmDopplerOffsetsTensorDataType == NVCV_DATA_TYPE_F32) &&
            (outTargetCountTensorDataType == NVCV_DATA_TYPE_S32) &&
            (outTargetAnglesTensorDataType == NVCV_DATA_TYPE_F32) &&
            (outTargetIndexMapTensorDataType == NVCV_DATA_TYPE_S32))
        {
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::DoaAngleFFT<int32_t, float>>(handle)(
                stream, inDetectionCount, inSnapshots, inDetectionList, inDdmDopplerOffsets, outTargetCount,
                outTargetAngles, outTargetIndexMap);
        }
        else
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported tensor data types");
        }
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus DoaAngleFFTSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle const,
                                                         NVCVTensorHandle const, NVCVTensorHandle const,
                                                         NVCVTensorHandle const, NVCVTensorHandle, NVCVTensorHandle,
                                                         NVCVTensorHandle);
template NVCVStatus DoaAngleFFTSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle const,
                                                        NVCVTensorHandle const, NVCVTensorHandle const,
                                                        NVCVTensorHandle const, NVCVTensorHandle, NVCVTensorHandle,
                                                        NVCVTensorHandle);

// C++ overload implementations
NVCVStatus pvaDoaAngleFFTSubmit(NVCVOperatorHandle handle, cupvaStream_t stream,
                                NVCVTensorHandle const inDetectionCountTensor, NVCVTensorHandle const inSnapshotsTensor,
                                NVCVTensorHandle const inDetectionListTensor,
                                NVCVTensorHandle const inDdmDopplerOffsetsTensor, NVCVTensorHandle outTargetCountTensor,
                                NVCVTensorHandle outTargetAnglesTensor, NVCVTensorHandle outTargetIndexMapTensor)
{
    return DoaAngleFFTSubmitImpl(handle, stream, inDetectionCountTensor, inSnapshotsTensor, inDetectionListTensor,
                                 inDdmDopplerOffsetsTensor, outTargetCountTensor, outTargetAnglesTensor,
                                 outTargetIndexMapTensor);
}

NVCVStatus pvaDoaAngleFFTSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                NVCVTensorHandle const inDetectionCountTensor, NVCVTensorHandle const inSnapshotsTensor,
                                NVCVTensorHandle const inDetectionListTensor,
                                NVCVTensorHandle const inDdmDopplerOffsetsTensor, NVCVTensorHandle outTargetCountTensor,
                                NVCVTensorHandle outTargetAnglesTensor, NVCVTensorHandle outTargetIndexMapTensor)
{
    return DoaAngleFFTSubmitImpl(handle, stream, inDetectionCountTensor, inSnapshotsTensor, inDetectionListTensor,
                                 inDdmDopplerOffsetsTensor, outTargetCountTensor, outTargetAnglesTensor,
                                 outTargetIndexMapTensor);
}
