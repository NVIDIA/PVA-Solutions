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
 * @file OpDoaBartlettBeamforming.cpp
 * @brief Host-side operator implementation for DOA Bartlett Beamforming
 *
 * This operator performs DoA (Direction of Arrival) angle estimation using Bartlett beamforming
 * and outputs target angles and target index map as tensors.
 */

#include "OpDoaBartlettBeamforming.hpp"

#include "device/bartlett_params.h"

#include <IOperator.hpp>
#include <SteeringVectorGenerator.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <nvcv/Tensor.hpp>

#include <OpDoaBartlettBeamforming.h>
#include <PvaAllocator.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_nonsafety.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>

using namespace cupva;

// Declare both Bartlett ELFs - separable and full 2D modes
CUPVA_DECLARE_EXECUTABLE(pva_bartlett_separable_dev);
CUPVA_DECLARE_EXECUTABLE(pva_bartlett_full2d_dev);

namespace pva_solutions::priv {

template<typename TI, typename TO>
std::mutex DoaBartlettBeamforming<TI, TO>::s_execMapMutex;

template<typename TI, typename TO>
std::map<typename DoaBartlettBeamforming<TI, TO>::ExecKey, std::weak_ptr<cupva::Executable>>
    DoaBartlettBeamforming<TI, TO>::s_BartlettExecutables;

template<typename TI, typename TO>
std::map<typename DoaBartlettBeamforming<TI, TO>::ExecDataKey, IOperator::ExecutableData> createBartlettExecDataMap()
{
    return {
        {std::make_tuple(true), // Separable mode
         {CUPVA_EXECUTABLE_DATA(pva_bartlett_separable_dev), CUPVA_EXECUTABLE_SIZE(pva_bartlett_separable_dev)}},
        {std::make_tuple(false), // Full 2D mode
         {CUPVA_EXECUTABLE_DATA(pva_bartlett_full2d_dev), CUPVA_EXECUTABLE_SIZE(pva_bartlett_full2d_dev)}},
    };
}

template<typename TI, typename TO>
std::shared_ptr<cupva::Executable> DoaBartlettBeamforming<TI, TO>::GetBartlettExec(bool enableSeparable)
{
    static auto execDataMap = createBartlettExecDataMap<TI, TO>();

    ExecDataKey execDataKey = std::make_tuple(enableSeparable);
    auto execDataIt         = execDataMap.find(execDataKey);
    if (execDataIt != execDataMap.end())
    {
        ExecKey execKey = std::tuple_cat(execDataKey, std::make_tuple(cupva::Context::GetCurrent()));
        return IOperator::GetExecutable<ExecKey>(execDataIt->second, execKey, s_BartlettExecutables, s_execMapMutex);
    }
    else
    {
        throw std::invalid_argument("Unsupported Bartlett beamforming mode");
    }
}

template<typename TI, typename TO>
DoaBartlettBeamforming<TI, TO>::DoaBartlettBeamforming(
    const NVCVTensorRequirements *snapshotsTensorRequirements,
    const NVCVTensorRequirements *detectionListTensorRequirements,
    const NVCVTensorRequirements *ddmDopplerOffsetsTensorRequirements,
    const NVCVTensorRequirements *targetAnglesTensorRequirements,
    const NVCVTensorRequirements *targetIndexMapTensorRequirements, const PVADoaBartlettBeamformingParams *params)
    : m_snapshotsOp(cupva::OffsetPointer<TI>::Create(cupva::MemType::DRAM, nullptr))
    , m_prog{}
    , m_params{params ? *params : kDefaultParams}
    , m_steeringVectorsL2(nullptr)
    , m_steeringVectorsSize(0)
    , m_outTargetAnglesOp(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr))
    , m_outTargetIndexMapOp(cupva::OffsetPointer<int32_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_maxDetections(0)
    , m_detectionCountOp(cupva::OffsetPointer<int32_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_targetCountOp(cupva::OffsetPointer<int32_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_snapshotsTensorRequirements{*snapshotsTensorRequirements}
{
    int64_t snapshotsShape[3];
    nvcvTensorShapePermute(snapshotsTensorRequirements->layout, snapshotsTensorRequirements->shape, NVCV_TENSOR_CHW,
                           snapshotsShape);

    int32_t numDetections      = static_cast<int32_t>(snapshotsShape[0]);
    int32_t numTx              = static_cast<int32_t>(snapshotsShape[1]);
    int32_t numRx              = static_cast<int32_t>(snapshotsShape[2]);
    int32_t numVirtualChannels = numTx * numRx;

    // Validate parameters at creation time
    if (numDetections > MAX_DETECTIONS)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Number of detections exceeds maximum supported (8192)");
    }
    if (numVirtualChannels > MAX_VIRTUAL_CHANNELS)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Number of virtual channels (numTx * numRx) exceeds maximum supported (64)");
    }
    if ((numVirtualChannels % 8) != 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Number of virtual channels (numTx * numRx) must be divisible by 8");
    }

    m_maxDetections = numDetections; // Store capacity for runtime validation

    // Select the appropriate ELF based on mode
    m_exec    = GetBartlettExec(m_params.separableParams.enableSeparable);
    auto prog = CmdProgram::Create(*m_exec.get());

    m_steeringVectorsL2 = (int16_t *)cupva::mem::GetL2BaseAddress();
    if (m_params.separableParams.enableSeparable)
    {
        m_steeringVectorsSize =
            numVirtualChannels * STEERING_ANGLE_BATCH_SIZE * ELEMENT_NUM_OF_COMPLEX * sizeof(int16_t);
    }
    else
    {
        // Full 2D mode: all steering vectors are copied to L2
        m_steeringVectorsSize =
            numVirtualChannels * FULL2D_POWER_SPECTRUM_SIZE * ELEMENT_NUM_OF_COMPLEX * sizeof(int16_t);
    }

    auto snapshotsHdl             = prog["snapshotsSqdfHdl"];
    auto anglesOutputHdl          = prog["anglesOutputSqdfHdl"];
    auto steeringVectorsHdl       = prog["steeringVectorsSqdfHdl"];
    auto detectionCntSqdfHdl      = prog["detectionCntSqdfHdl"];
    auto targetCountAndMapSqdfHdl = prog["targetCountAndMapSqdfHdl"];

    auto snapshotBufferV        = prog["snapshotBuffer"].ptr<int32_t>();
    auto anglesOutputBufferV    = prog["anglesOutputBuffer"].ptr<float>();
    auto steeringVectorsBufferV = prog["steeringVectorsBuffer"].ptr<int16_t>();
    int32_t *detectionCount_v   = prog["detectionCount"].ptr<int32_t>();
    int32_t *targetCount_v      = prog["targetCount"].ptr<int32_t>();
    int32_t *targetMap_v        = prog["targetMap"].ptr<int32_t>();

    BartlettGlobalParams GP{};
    GP.numTx               = numTx;
    GP.numRx               = numRx;
    GP.numAzimuthBins      = 0;
    GP.numElevationBins    = 0;
    GP.enableInterpolation = m_params.enableInterpolation ? 1 : 0;
    GP.enablePowerOutput   = m_params.enablePowerOutput ? 1 : 0;

    constexpr int32_t p0{0};
    constexpr int32_t p1{1};
    constexpr int32_t p2{2};

    int32_t complexBytes     = ELEMENT_NUM_OF_COMPLEX * sizeof(int32_t);
    int32_t snapshotRowBytes = numVirtualChannels * complexBytes;

    SequenceDataFlow &detectionCntSQDF = prog.addDataFlowHead<SequenceDataFlow>(p0).handler(detectionCntSqdfHdl);
    detectionCntSQDF.addTransfer()
        .src(m_detectionCountOp, sizeof(int32_t))
        .dst(detectionCount_v, sizeof(int32_t))
        .tile(1 * sizeof(int32_t), 1)
        .mode(TransferModeType::CONTINUOUS);

    SequenceDataFlow &snapshotsSQDF = prog.addDataFlowHead<SequenceDataFlow>(p1).handler(snapshotsHdl);
    snapshotsSQDF.addTransfer()
        .tile(snapshotRowBytes, SNAPSHOT_BATCH_SIZE)
        .src(m_snapshotsOp, snapshotRowBytes)
        .dst(snapshotBufferV, snapshotRowBytes)
        .mode(TransferModeType::CONTINUOUS);

    // Configure 2D tile for SOA output: [numRows][batchSize] in VMEM -> [numRows][maxTargets] in DRAM
    // VMEM layout: [numRows][SNAPSHOT_BATCH_SIZE] - rows are SNAPSHOT_BATCH_SIZE apart
    // DRAM layout: [numRows][numDetections] - rows are numDetections apart
    int32_t anglesNumRows = m_params.enablePowerOutput ? ANGLES_NUM_ROWS_WITH_POWER : ANGLES_NUM_ROWS_WITHOUT_POWER;
    int32_t anglesRowTileBytes         = SNAPSHOT_BATCH_SIZE * sizeof(float);
    int32_t anglesVmemLinePitch        = SNAPSHOT_BATCH_SIZE * sizeof(float);
    int32_t anglesDramLinePitch        = numDetections * sizeof(float); // maxTargets = numDetections from tensor shape
    SequenceDataFlow &anglesOutputSQDF = prog.addDataFlowHead<SequenceDataFlow>(p1).handler(anglesOutputHdl);
    anglesOutputSQDF.addTransfer()
        .src(anglesOutputBufferV, anglesVmemLinePitch)
        .dst(m_outTargetAnglesOp, anglesDramLinePitch) // Output to target angles tensor
        .tile(anglesRowTileBytes, anglesNumRows)       // 2D tile: batchSize x numRows
        .mode(TransferModeType::CONTINUOUS);

    int32_t steeringBatchBytes =
        numVirtualChannels * STEERING_ANGLE_BATCH_SIZE * ELEMENT_NUM_OF_COMPLEX * sizeof(int16_t);
    SequenceDataFlow &steeringVectorsSQDF = prog.addDataFlowHead<SequenceDataFlow>(p1).handler(steeringVectorsHdl);
    steeringVectorsSQDF.addTransfer()
        .src((void *)m_steeringVectorsL2, steeringBatchBytes)
        .dst(steeringVectorsBufferV, steeringBatchBytes)
        .tile(steeringBatchBytes, 1)
        .mode(TransferModeType::CONTINUOUS);

    // Step 1 strided DMA for separable mode - VPU kernel updates addresses at runtime
    if (m_params.separableParams.enableSeparable)
    {
        auto step1SteeringHdl     = prog["step1SteeringSqdfHdl"];
        auto step1SteeringBufferV = prog["step1SteeringBuffer"].ptr<int16_t>();

        // Use minimal tile height (1) to keep L2 placeholder small; VPU sets actual height.
        const int32_t vcStride = numVirtualChannels * ELEMENT_NUM_OF_COMPLEX * sizeof(int16_t);

        SequenceDataFlow &step1SteeringSQDF = prog.addDataFlowHead<SequenceDataFlow>(p1).handler(step1SteeringHdl);
        step1SteeringSQDF.addTransfer()
            .src((void *)m_steeringVectorsL2, vcStride) // Placeholder src, updated at runtime
            .dst(step1SteeringBufferV, vcStride)        // Placeholder dst, updated at runtime
            .tile(vcStride, 1)                          // Minimal placeholder; VPU sets actual tile
            .mode(TransferModeType::CONTINUOUS);
    }

    // Configure target count data flow
    SequenceDataFlow &TargetCountAndMapSQDF =
        prog.addDataFlowHead<SequenceDataFlow>(p2).handler(targetCountAndMapSqdfHdl);
    TargetCountAndMapSQDF.addTransfer()
        .src(targetCount_v, 1 * sizeof(int32_t))
        .dst(m_targetCountOp, 1 * sizeof(int32_t))
        .tile(1 * sizeof(int32_t), 1)
        .mode(TransferModeType::CONTINUOUS);
    TargetCountAndMapSQDF.addTransfer()
        .src(targetMap_v, sizeof(int32_t))
        .dst(m_outTargetIndexMapOp, sizeof(int32_t))
        .tile(sizeof(int32_t), m_maxDetections)
        .mode(TransferModeType::CONTINUOUS);

    prog["GP"].set(&GP, 1);
    prog["maxTargetCount"] = static_cast<int32_t>(m_maxDetections);

    prog.compileDataFlows();
    prog.setL2Size(m_steeringVectorsSize);

    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename TI, typename TO>
DoaBartlettBeamforming<TI, TO>::~DoaBartlettBeamforming()
{
    CupvaCmdDestroy(&m_prog);
}

template<typename TI, typename TO>
template<typename S>
void DoaBartlettBeamforming<TI, TO>::operator()(
    S stream, NVCVTensorHandle detectionCountHandle, NVCVTensorHandle snapshotsHandle,
    NVCVTensorHandle steeringVectorsHandle, NVCVTensorHandle azimuthBinsHandle, NVCVTensorHandle elevationBinsHandle,
    NVCVTensorHandle detectionListHandle, NVCVTensorHandle ddmDopplerOffsetsHandle, NVCVTensorHandle targetCountHandle,
    NVCVTensorHandle targetAnglesHandle, NVCVTensorHandle targetIndexMapHandle)
{
    NVCVTensorData detectionCountData;
    nvcvTensorExportData(detectionCountHandle, &detectionCountData);

    NVCVTensorData snapshotsData;
    nvcvTensorExportData(snapshotsHandle, &snapshotsData);

    NVCVTensorData steeringVectorsData;
    nvcvTensorExportData(steeringVectorsHandle, &steeringVectorsData);

    NVCVTensorData azimuthBinsData;
    nvcvTensorExportData(azimuthBinsHandle, &azimuthBinsData);

    NVCVTensorData elevationBinsData;
    nvcvTensorExportData(elevationBinsHandle, &elevationBinsData);

    NVCVTensorData targetCountData;
    nvcvTensorExportData(targetCountHandle, &targetCountData);

    NVCVTensorData targetAnglesData;
    nvcvTensorExportData(targetAnglesHandle, &targetAnglesData);

    NVCVTensorData targetIndexMapData;
    nvcvTensorExportData(targetIndexMapHandle, &targetIndexMapData);

    // Validate tensor base pointers
    if (snapshotsData.buffer.strided.basePtr == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Snapshots tensor base pointer must not be NULL");
    }
    if (steeringVectorsData.buffer.strided.basePtr == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Steering vectors tensor base pointer must not be NULL");
    }

    // Validate tensor requirements match what was used to create the operator
    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_snapshotsTensorRequirements, snapshotsData))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Snapshots tensor must have the same requirements used to create the operator");
    }

    int32_t numAzimuthBins   = static_cast<int32_t>(azimuthBinsData.shape[0]);
    int32_t numElevationBins = static_cast<int32_t>(elevationBinsData.shape[0]);

    // Validate angle bins range
    if (numAzimuthBins < 1 || numElevationBins < 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Azimuth/elevation bins must be at least 1");
    }
    if (!m_params.separableParams.enableSeparable)
    {
        // Full 2D mode: each dimension limited by FULL2D_POWER_SPECTRUM_SIZE
        if (numAzimuthBins > FULL2D_POWER_SPECTRUM_SIZE || numElevationBins > FULL2D_POWER_SPECTRUM_SIZE)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Azimuth/elevation bins size out of supported range (max: %d)",
                                  FULL2D_POWER_SPECTRUM_SIZE);
        }
    }

    // Validate total angle bins doesn't exceed maximum (full 2D mode only)
    // Separable mode has no combined angle bin limit - it streams from DRAM
    if (!m_params.separableParams.enableSeparable)
    {
        if (numAzimuthBins * numElevationBins > FULL2D_POWER_SPECTRUM_SIZE)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Total angle bins (azimuth * elevation) exceeds maximum supported");
        }
    }

    // Get min/max angles from the bins array (assuming uniform spacing)
    float *azimuthBinsPtr   = nullptr;
    float *elevationBinsPtr = nullptr;
    CupvaMemGetHostPointer((void **)&azimuthBinsPtr, (void *)azimuthBinsData.buffer.strided.basePtr);
    CupvaMemGetHostPointer((void **)&elevationBinsPtr, (void *)elevationBinsData.buffer.strided.basePtr);

    float azimuthMin   = azimuthBinsPtr[0];
    float azimuthMax   = azimuthBinsPtr[numAzimuthBins - 1];
    float elevationMin = elevationBinsPtr[0];
    float elevationMax = elevationBinsPtr[numElevationBins - 1];

    // Compute elZeroIdx: find elevation bin closest to 0.0 degrees
    int32_t elZeroIdx = 0;
    float minAbsEl    = std::abs(elevationBinsPtr[0]);
    for (int32_t i = 1; i < numElevationBins; i++)
    {
        float absEl = std::abs(elevationBinsPtr[i]);
        if (absEl < minAbsEl)
        {
            minAbsEl  = absEl;
            elZeroIdx = i;
        }
    }

    // Validate separable Bartlett parameters
    if (m_params.separableParams.enableSeparable)
    {
        const int32_t N = static_cast<int32_t>(m_params.separableParams.azNeighborhood);

        // VPU constraint: powerSpectrum is 2*numAz floats (azimuth half, double buffered for
        // the step-1 lookahead) followed by (2*N+1)*numEl for the Step 2 local spectrum.
        const int32_t step2Size    = (2 * N + 1) * numElevationBins;
        const int32_t spectrumSize = 2 * numAzimuthBins + step2Size;

        if (spectrumSize > SEPARABLE_POWER_SPECTRUM_SIZE)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Separable spectrum too large for VMEM buffer (required: %d, available: %d)",
                                  spectrumSize, SEPARABLE_POWER_SPECTRUM_SIZE);
        }

        if (numAzimuthBins < 2 * N + 1)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "numAzimuthBins too small for specified azNeighborhood");
        }

        if (numElevationBins > 255)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Separable mode: numElevationBins must be <= 255 (got %d)", numElevationBins);
        }
    }

    cupvaParameter_t gpParam;
    CupvaCmdProgramGetParameter(&m_prog, &gpParam, "GP");
    BartlettGlobalParams GP{};
    int64_t snapshotsShape[3];
    nvcvTensorShapePermute(snapshotsData.layout, snapshotsData.shape, NVCV_TENSOR_CHW, snapshotsShape);
    GP.numTx               = static_cast<int32_t>(snapshotsShape[1]);
    GP.numRx               = static_cast<int32_t>(snapshotsShape[2]);
    GP.numAzimuthBins      = numAzimuthBins;
    GP.numElevationBins    = numElevationBins;
    GP.enableInterpolation = m_params.enableInterpolation ? 1 : 0;
    GP.enablePowerOutput   = m_params.enablePowerOutput ? 1 : 0;
    GP.azimuthMin          = azimuthMin;
    GP.azimuthMax          = azimuthMax;
    GP.elevationMin        = elevationMin;
    GP.elevationMax        = elevationMax;
    // Separable Bartlett parameters
    GP.enableSeparable = m_params.separableParams.enableSeparable ? 1 : 0;
    GP.azNeighborhood  = static_cast<int32_t>(m_params.separableParams.azNeighborhood);
    GP.azFromEl0       = (int32_t)m_params.separableParams.azSource;
    GP.elZeroIdx       = elZeroIdx;
    GP.azChannelMaskLo = static_cast<uint32_t>(m_params.separableParams.azChannelMask & 0xFFFFFFFFULL);
    GP.azChannelMaskHi = static_cast<uint32_t>(m_params.separableParams.azChannelMask >> 32);
    CupvaParameterSetValueArray(&gpParam, &GP, sizeof(GP));

    m_snapshotsOp.setBase(static_cast<TI *>(static_cast<void *>(snapshotsData.buffer.strided.basePtr)));
    m_detectionCountOp.setBase(static_cast<int32_t *>(static_cast<void *>(detectionCountData.buffer.strided.basePtr)));
    m_targetCountOp.setBase(static_cast<int32_t *>(static_cast<void *>(targetCountData.buffer.strided.basePtr)));
    m_outTargetAnglesOp.setBase(static_cast<float *>(static_cast<void *>(targetAnglesData.buffer.strided.basePtr)));
    m_outTargetIndexMapOp.setBase(
        static_cast<int32_t *>(static_cast<void *>(targetIndexMapData.buffer.strided.basePtr)));

    cupvaParameter_t snapshotsExtAddrParam;
    CupvaCmdProgramGetParameter(&m_prog, &snapshotsExtAddrParam, "snapshotsExtAddr");
    CupvaParameterSetValuePointer(&snapshotsExtAddrParam, static_cast<void *>(snapshotsData.buffer.strided.basePtr));

    cupvaParameter_t anglesExtAddrParam;
    CupvaCmdProgramGetParameter(&m_prog, &anglesExtAddrParam, "anglesOutputExtAddr");
    CupvaParameterSetValuePointer(&anglesExtAddrParam, static_cast<void *>(targetAnglesData.buffer.strided.basePtr));

    cupvaParameter_t targetCountExtAddrParam;
    CupvaCmdProgramGetParameter(&m_prog, &targetCountExtAddrParam, "targetCountExtAddr");
    CupvaParameterSetValuePointer(&targetCountExtAddrParam,
                                  static_cast<void *>(targetCountData.buffer.strided.basePtr));

    int32_t numVirtualChannels = GP.numTx * GP.numRx;

    int32_t numAngleBins              = numAzimuthBins * numElevationBins;
    int32_t actualSteeringVectorsSize = numVirtualChannels * numAngleBins * ELEMENT_NUM_OF_COMPLEX * sizeof(int16_t);

    const bool useDramSteering = m_params.separableParams.enableSeparable;

    if (!useDramSteering && actualSteeringVectorsSize > m_steeringVectorsSize)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Steering vector size exceeds L2 allocation (check FULL2D_POWER_SPECTRUM_SIZE)");
    }

    // Set DRAM steering vector address for separable mode
    if (useDramSteering)
    {
        cupvaParameter_t steeringDramParam;
        CupvaCmdProgramGetParameter(&m_prog, &steeringDramParam, "steeringDramExtAddr");
        CupvaParameterSetValuePointer(&steeringDramParam, steeringVectorsData.buffer.strided.basePtr);
    }

    {
        cupvaCmd_t steeringVectorsCopyCmd = {};
        if (!useDramSteering)
        {
            // Full 2D mode: create DRAM→L2 memcpy for steering vectors
            cupvaCmdMemcpyBufferParams_t copyParams = {
                .ptrSrc = steeringVectorsData.buffer.strided.basePtr,
                .ptrDst = m_steeringVectorsL2,
            };
            cupvaError_t copyErr =
                CupvaCmdMemcpyCreate(&steeringVectorsCopyCmd, &copyParams, actualSteeringVectorsSize);
            if (copyErr != CUPVA_ERROR_NONE)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_DEVICE, "Failed to create steering vectors copy command");
            }
        }

        cupvaError_t err = CupvaCmdProgramUpdateDataFlows(&m_prog);
        if (err != CUPVA_ERROR_NONE)
        {
            if (!useDramSteering)
                CupvaCmdDestroy(&steeringVectorsCopyCmd);
            throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "Failed to update data flows");
        }

        if (useDramSteering)
        {
            // Separable mode: no DRAM→L2 copy, submit beamforming only
            cupvaCmd_t const *submitCmds[1] = {&m_prog};

            if constexpr (std::is_same_v<S, cupvaStream_t>)
            {
                err = CupvaStreamSubmit(stream, submitCmds, nullptr, 1, CUPVA_IN_ORDER, -1, -1);
            }
            else if constexpr (std::is_same_v<S, cudaStream_t>)
            {
                if constexpr (kCupvaHasCudaStreamSupport)
                {
                    err = CupvaCudaStreamSubmit(stream, submitCmds, nullptr, 1, CUPVA_IN_ORDER, -1, -1);
                }
                else
                {
                    ThrowCudaStreamNotSupported();
                }
            }
            else
            {
                static_assert(!sizeof(S), "Unsupported stream type for Bartlett Beamforming operator");
            }
        }
        else
        {
            // Full 2D mode: submit L2 copy + beamforming
            cupvaCmd_t const *submitCmds[2] = {&steeringVectorsCopyCmd, &m_prog};

            if constexpr (std::is_same_v<S, cupvaStream_t>)
            {
                err = CupvaStreamSubmit(stream, submitCmds, nullptr, 2, CUPVA_IN_ORDER, -1, -1);
            }
            else if constexpr (std::is_same_v<S, cudaStream_t>)
            {
                if constexpr (kCupvaHasCudaStreamSupport)
                {
                    err = CupvaCudaStreamSubmit(stream, submitCmds, nullptr, 2, CUPVA_IN_ORDER, -1, -1);
                }
                else
                {
                    CupvaCmdDestroy(&steeringVectorsCopyCmd);
                    ThrowCudaStreamNotSupported();
                }
            }
            else
            {
                static_assert(!sizeof(S), "Unsupported stream type for Bartlett Beamforming operator");
            }

            CupvaCmdDestroy(&steeringVectorsCopyCmd);
        }

        if (err != CUPVA_ERROR_NONE)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_DEVICE, "PVA Cmd Submit failed");
        }
    }
}

template class DoaBartlettBeamforming<int32_t, float>;

template void DoaBartlettBeamforming<int32_t, float>::operator()(cupvaStream_t, NVCVTensorHandle, NVCVTensorHandle,
                                                                 NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle,
                                                                 NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle,
                                                                 NVCVTensorHandle, NVCVTensorHandle);

template void DoaBartlettBeamforming<int32_t, float>::operator()(cudaStream_t, NVCVTensorHandle, NVCVTensorHandle,
                                                                 NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle,
                                                                 NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle,
                                                                 NVCVTensorHandle, NVCVTensorHandle);

} // namespace pva_solutions::priv

extern "C" {

NVCVStatus pvaDoaBartlettBeamformingCreate(NVCVOperatorHandle *handle,
                                           NVCVTensorRequirements const *const snapshotsTensorRequirements,
                                           NVCVTensorRequirements const *const detectionListTensorRequirements,
                                           NVCVTensorRequirements const *const ddmDopplerOffsetsTensorRequirements,
                                           NVCVTensorRequirements const *const targetAnglesTensorRequirements,
                                           NVCVTensorRequirements const *const targetIndexMapTensorRequirements,
                                           PVADoaBartlettBeamformingParams const *const params)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        if (snapshotsTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to snapshots tensor requirements must not be NULL");
        }
        if (detectionListTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to detection list tensor requirements must not be NULL");
        }
        if (ddmDopplerOffsetsTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to DDM Doppler offsets tensor requirements must not be NULL");
        }
        if (targetAnglesTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to target angles tensor requirements must not be NULL");
        }
        if (targetIndexMapTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to target index map tensor requirements must not be NULL");
        }
        if (params == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Pointer to params must not be NULL");
        }
        *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::DoaBartlettBeamforming<int32_t, float>(
            snapshotsTensorRequirements, detectionListTensorRequirements, ddmDopplerOffsetsTensorRequirements,
            targetAnglesTensorRequirements, targetIndexMapTensorRequirements, params));
    });
}

} // extern "C"

template<typename StreamType>
NVCVStatus DoaBartlettBeamformingSubmitImpl(NVCVOperatorHandle handle, StreamType stream,
                                            NVCVTensorHandle detectionCountHandle, NVCVTensorHandle snapshotsHandle,
                                            NVCVTensorHandle steeringVectorsHandle, NVCVTensorHandle azimuthBinsHandle,
                                            NVCVTensorHandle elevationBinsHandle, NVCVTensorHandle detectionListHandle,
                                            NVCVTensorHandle ddmDopplerOffsetsHandle,
                                            NVCVTensorHandle targetCountHandle, NVCVTensorHandle targetAnglesHandle,
                                            NVCVTensorHandle targetIndexMapHandle)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }

        auto *op = reinterpret_cast<pva_solutions::priv::DoaBartlettBeamforming<int32_t, float> *>(handle);
        (*op)(stream, detectionCountHandle, snapshotsHandle, steeringVectorsHandle, azimuthBinsHandle,
              elevationBinsHandle, detectionListHandle, ddmDopplerOffsetsHandle, targetCountHandle, targetAnglesHandle,
              targetIndexMapHandle);
    });
}

template NVCVStatus DoaBartlettBeamformingSubmitImpl<cupvaStream_t>(
    NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle,
    NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle);

template NVCVStatus DoaBartlettBeamformingSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle,
                                                                   NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle,
                                                                   NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle,
                                                                   NVCVTensorHandle, NVCVTensorHandle,
                                                                   NVCVTensorHandle);

NVCVStatus pvaDoaBartlettBeamformingSubmit(NVCVOperatorHandle handle, cupvaStream_t stream,
                                           NVCVTensorHandle const detectionCount, NVCVTensorHandle const snapshots,
                                           NVCVTensorHandle const steeringVectors, NVCVTensorHandle const azimuthBins,
                                           NVCVTensorHandle const elevationBins, NVCVTensorHandle const detectionList,
                                           NVCVTensorHandle const ddmDopplerOffsets, NVCVTensorHandle targetCount,
                                           NVCVTensorHandle targetAngles, NVCVTensorHandle targetIndexMap)
{
    return DoaBartlettBeamformingSubmitImpl(handle, stream, detectionCount, snapshots, steeringVectors, azimuthBins,
                                            elevationBins, detectionList, ddmDopplerOffsets, targetCount, targetAngles,
                                            targetIndexMap);
}

NVCVStatus pvaDoaBartlettBeamformingSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                           NVCVTensorHandle const detectionCount, NVCVTensorHandle const snapshots,
                                           NVCVTensorHandle const steeringVectors, NVCVTensorHandle const azimuthBins,
                                           NVCVTensorHandle const elevationBins, NVCVTensorHandle const detectionList,
                                           NVCVTensorHandle const ddmDopplerOffsets, NVCVTensorHandle targetCount,
                                           NVCVTensorHandle targetAngles, NVCVTensorHandle targetIndexMap)
{
    return DoaBartlettBeamformingSubmitImpl(handle, stream, detectionCount, snapshots, steeringVectors, azimuthBins,
                                            elevationBins, detectionList, ddmDopplerOffsets, targetCount, targetAngles,
                                            targetIndexMap);
}

extern "C" {

NVCVStatus pvaDoaBartlettBeamformingComputeSteeringVectors(NVCVTensorHandle virtualArrayLocations,
                                                           NVCVTensorHandle azimuthBins, NVCVTensorHandle elevationBins,
                                                           NVCVTensorHandle steeringVectors)
{
    return nvcv::ProtectCall([&] {
        // Read virtual antenna array locations from tensor
        NVCVTensorData vcPosData;
        nvcvTensorExportData(virtualArrayLocations, &vcPosData);
        float *vcPosPtr = NULL;
        CupvaMemGetHostPointer((void **)&vcPosPtr, (void *)vcPosData.buffer.strided.basePtr);

        int64_t vcPosShape[2];
        nvcvTensorShapePermute(vcPosData.layout, vcPosData.shape, NVCV_TENSOR_HW, vcPosShape);
        const int32_t numTx = static_cast<int32_t>(vcPosShape[0]);
        const int32_t numRx = static_cast<int32_t>(vcPosShape[1]);

        // Read azimuth bins from tensor
        NVCVTensorData azimuthBinsData;
        nvcvTensorExportData(azimuthBins, &azimuthBinsData);
        float *azimuthBinsPtr = NULL;
        CupvaMemGetHostPointer((void **)&azimuthBinsPtr, (void *)azimuthBinsData.buffer.strided.basePtr);
        const int32_t numAzimuthBins = static_cast<int32_t>(azimuthBinsData.shape[0]);

        // Read elevation bins from tensor
        NVCVTensorData elevationBinsData;
        nvcvTensorExportData(elevationBins, &elevationBinsData);
        float *elevationBinsPtr = NULL;
        CupvaMemGetHostPointer((void **)&elevationBinsPtr, (void *)elevationBinsData.buffer.strided.basePtr);
        const int32_t numElevationBins = static_cast<int32_t>(elevationBinsData.shape[0]);

        // Write the packed [azimuth][elevation][Tx][Rx] Q15 tensor using the
        // conjugated storage convention consumed by the Bartlett kernels.
        NVCVTensorData steeringVectorsData;
        nvcvTensorExportData(steeringVectors, &steeringVectorsData);
        int16_t *steeringVectorsPtr = NULL;
        CupvaMemGetHostPointer((void **)&steeringVectorsPtr, (void *)steeringVectorsData.buffer.strided.basePtr);

        pva_solutions::priv::steering_vector::generateSteeringVectorsQ15(
            {vcPosPtr, static_cast<int64_t>(2U * sizeof(float))}, {azimuthBinsPtr, static_cast<int64_t>(sizeof(float))},
            {elevationBinsPtr, static_cast<int64_t>(sizeof(float))}, steeringVectorsPtr, numTx * numRx, numAzimuthBins,
            numElevationBins, pva_solutions::priv::steering_vector::OutputConvention::ConjugatedAfterQuantization);
    });
}

} // extern "C"
