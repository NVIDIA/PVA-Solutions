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
 * @file TargetProcessing.cpp
 * @brief Host-side implementation for Target Processing
 *
 * This file implements the host-side interface for the Target Processing component,
 * which performs Range-Velocity (RV) decoupling and Cartesian coordinate computation
 * for detected radar targets on the PVA.
 *
 * Main Responsibilities:
 * 1. Tensor requirement calculation and validation
 * 2. VPU executable management (loading and caching)
 * 3. Data flow management (input and output)
 * 4. Parameter management (program parameters)
 *
 * Input Tensors:
 * - Target count: [1] int32 scalar
 * - Detection list: [N, 2] int32 (range and Doppler indices)
 * - Target index map: [M] int32 (maps targets to detections)
 * - Target angles: [3, M] float32 (azimuth, elevation, power)
 * - DDM Doppler offsets: [K] float32 (Doppler offset per fold)
 *
 * Output Tensors:
 * - Target list: [7 or 8, M] float32 (velocity, range, angles, XYZ coordinates, power(optional))
 */

#include "TargetProcessing.hpp"

#include "device/target_processing_param.h"

#include <IOperator.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>

#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <cmath>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_target_processing_dev);

namespace pva_solutions::priv {

template<typename TI, typename TO>
std::mutex TargetProcessing<TI, TO>::s_execMapMutex;

template<typename TI, typename TO>
std::map<typename TargetProcessing<TI, TO>::ExecKey, std::weak_ptr<cupva::Executable>>
    TargetProcessing<TI, TO>::s_TargetProcessingExecutables;

template<typename TI, typename TO>
std::shared_ptr<cupva::Executable> TargetProcessing<TI, TO>::GetTargetProcessingExec()
{
    static IOperator::ExecutableData sTargetProcessingExecData;

    sTargetProcessingExecData = {CUPVA_EXECUTABLE_DATA(pva_target_processing_dev),
                                 CUPVA_EXECUTABLE_SIZE(pva_target_processing_dev)};

    ExecKey execKey = std::make_tuple(cupva::Context::GetCurrent());
    return IOperator::GetExecutable<ExecKey>(sTargetProcessingExecData, execKey, s_TargetProcessingExecutables,
                                             s_execMapMutex);
}

template<typename TI, typename TO>
TargetProcessing<TI, TO>::TargetProcessing(NVCVTensorRequirements const *const detectionListTensorRequirements,
                                           NVCVTensorRequirements const *const targetIndexMapTensorRequirements,
                                           NVCVTensorRequirements const *const targetAnglesTensorRequirements,
                                           NVCVTensorRequirements const *const ddmDopplerOffsetsTensorRequirements,
                                           NVCVTensorRequirements const *const targetListTensorRequirements,
                                           NVCVTensorRequirements const *const nciTensorRequirements,
                                           PVATargetProcessingParams const *const params)
    : m_detectionListTensorRequirements{*detectionListTensorRequirements}
    , m_ddmDopplerOffsetsTensorRequirements{*ddmDopplerOffsetsTensorRequirements}
    , m_targetListTensorRequirements{*targetListTensorRequirements}
    , m_params{params ? *params : kDefaultParams}
    , m_maxNumOfTargets{0}
    , m_targetIndexMapOp(cupva::OffsetPointer<int32_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_targetAnglesOp(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr))
    , m_targetIndexMapTensorRequirements{*targetIndexMapTensorRequirements}
    , m_targetAnglesTensorRequirements{*targetAnglesTensorRequirements}
    , m_detectionCountOp(cupva::OffsetPointer<int32_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_detectionListOp(cupva::OffsetPointer<TI>::Create(cupva::MemType::DRAM, nullptr))
    , m_ddmDopplerOffsetsOp(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr))
    , m_targetListOp(cupva::OffsetPointer<TO>::Create(cupva::MemType::DRAM, nullptr))
    , m_targetCountOp(cupva::OffsetPointer<int32_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_nciSrcOp(cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_nciL2{nullptr}
    , m_nciL2Size{0}
    , m_nciWidth{0}
    , m_nciHeight{0}
    , m_nciPaddedWidth{0}
    , m_nciPaddedHeight{0}
    , m_prog{}
{
    // Get detection list tensor dimensions
    int64_t detectionListTensorShapeNCHW[4];
    nvcvTensorShapePermute(detectionListTensorRequirements->layout, detectionListTensorRequirements->shape,
                           NVCV_TENSOR_NCHW, detectionListTensorShapeNCHW);

    const int32_t detectionListHeight = detectionListTensorShapeNCHW[2];
    const int32_t detectionListWidth  = detectionListTensorShapeNCHW[3];
    const int32_t maxNumDetections    = detectionListHeight;
    if (maxNumDetections > MAX_SUPPORTED_TARGET_COUNT)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Number of detections must not exceed MAX_SUPPORTED_TARGET_COUNT");
    }

    if (detectionListWidth != 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Detection list must have 2 columns (range, doppler)");
    }

    // Get numTargets from targetIndexMap tensor shape
    int64_t targetIndexMapTensorShapeNCHW[4];
    nvcvTensorShapePermute(targetIndexMapTensorRequirements->layout, targetIndexMapTensorRequirements->shape,
                           NVCV_TENSOR_NCHW, targetIndexMapTensorShapeNCHW);
    m_maxNumOfTargets = static_cast<int32_t>(targetIndexMapTensorShapeNCHW[3]);

    if (m_maxNumOfTargets > MAX_SUPPORTED_TARGET_COUNT)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Number of targets must not exceed MAX_SUPPORTED_TARGET_COUNT");
    }

    // Validate target angles tensor dimensions
    int64_t targetAnglesTensorShapeNCHW[4];
    nvcvTensorShapePermute(targetAnglesTensorRequirements->layout, targetAnglesTensorRequirements->shape,
                           NVCV_TENSOR_NCHW, targetAnglesTensorShapeNCHW);

    const int32_t targetAnglesHeight = targetAnglesTensorShapeNCHW[2];
    const int32_t targetAnglesWidth  = targetAnglesTensorShapeNCHW[3];
    if (targetAnglesHeight != TARGET_ANGLES_TILE_HEIGHT)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Target angles must have 3 rows (azimuth, elevation, power)");
    }
    if (targetAnglesWidth != m_maxNumOfTargets)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Target angles columns must equal number of targets");
    }

    // Get target list tensor dimensions
    int64_t targetListTensorShapeNCHW[4];
    nvcvTensorShapePermute(targetListTensorRequirements->layout, targetListTensorRequirements->shape, NVCV_TENSOR_NCHW,
                           targetListTensorShapeNCHW);

    const int32_t targetListHeight = targetListTensorShapeNCHW[2];
    const int32_t targetListWidth  = targetListTensorShapeNCHW[3];
    if (targetListHeight != TARGET_LIST_TILE_HEIGHT && targetListHeight != (TARGET_LIST_TILE_HEIGHT - 1))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Target list must have 7 or 8 Height (velocity, range, angles, XYZ, power (optional))");
    }

    if (targetListWidth != m_maxNumOfTargets)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Target list columns must equal number of targets");
    }

    // validate ddmDopplerOffsets tensor dimensions
    int64_t ddmDopplerOffsetsTensorShapeNCHW[4];
    nvcvTensorShapePermute(ddmDopplerOffsetsTensorRequirements->layout, ddmDopplerOffsetsTensorRequirements->shape,
                           NVCV_TENSOR_NCHW, ddmDopplerOffsetsTensorShapeNCHW);

    const int32_t ddmDopplerOffsetsHeight = ddmDopplerOffsetsTensorShapeNCHW[2];
    const int32_t ddmDopplerOffsetsWidth  = ddmDopplerOffsetsTensorShapeNCHW[3];

    if (ddmDopplerOffsetsHeight != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "DDM Doppler offsets must have 1 Height (Doppler offsets)");
    }
    if (ddmDopplerOffsetsWidth > 32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "DDM Doppler offsets must be less than or equal to 32 Width (Doppler offsets)");
    }

    // Load VPU executable
    m_exec    = GetTargetProcessingExec();
    auto prog = CmdProgram::Create(*m_exec.get());

    int32_t phase = 0;

    // Set up data flow for inputs and outputs
    const int32_t detectionListLinePitch = detectionListWidth;
    const int32_t targetAnglesLinePitch  = m_maxNumOfTargets;
    const int32_t targetListLinePitch    = m_maxNumOfTargets;

    // Configure input data flows
    auto targetCntDetectionListAndDdmDopplerOffsetsHdl = prog["targetCntDetectionListAndDdmDopplerOffsetsHdl"];
    auto targetIndexMapHdl                             = prog["targetIndexMapHdl"];
    auto targetAnglesHdl                               = prog["targetAnglesHdl"];
    auto targetListHdl                                 = prog["targetListHdl"];

    int32_t *detectionCount_v  = prog["detectionCount"].ptr<int32_t>();
    TI *detectionList_v        = prog["detectionList"].ptr<TI>();
    float *ddmDopplerOffsets_v = prog["ddmDopplerOffsets"].ptr<float>();
    int32_t *targetCount_v     = prog["targetCount"].ptr<int32_t>();
    int32_t *targetIndexMap_v  = prog["targetIndexMap"].ptr<int32_t>();
    float *targetAngles_v      = prog["targetAngles"].ptr<float>();
    TO *targetList_v           = prog["targetList"].ptr<TO>();

    // Load detection list
    SequenceDataFlow &targetCntdetectionListAndDdmDopplerOffsetsDataFlow =
        prog.addDataFlowHead<SequenceDataFlow>(phase).handler(targetCntDetectionListAndDdmDopplerOffsetsHdl);
    phase++;

    // Load target index map and target angles
    SequenceDataFlow &targetIndexMapDataFlow = prog.addDataFlowHead<SequenceDataFlow>(phase).handler(targetIndexMapHdl);
    SequenceDataFlow &targetAnglesDataFlow   = prog.addDataFlowHead<SequenceDataFlow>(phase).handler(targetAnglesHdl);
    // Load target list
    SequenceDataFlow &targetListDataFlow = prog.addDataFlowHead<SequenceDataFlow>(phase).handler(targetListHdl);
    // The next-batch input SQDFs overlap the current batch's NCI GSDF in device code. Dataflows that may be active
    // simultaneously must share a phase so that the DMA compiler does not make them share transfer buffers.
    const int32_t processingPhase = phase;

    int32_t targetWidth =
        (m_maxNumOfTargets >= TARGET_ANGLES_TILE_WIDTH) ? TARGET_ANGLES_TILE_WIDTH : m_maxNumOfTargets;

    targetCntdetectionListAndDdmDopplerOffsetsDataFlow.addTransfer()
        .src(m_detectionCountOp, sizeof(int32_t))
        .dst(detectionCount_v, sizeof(int32_t))
        .tile(1 * sizeof(int32_t), 1)
        .mode(TransferModeType::CONTINUOUS);

    targetCntdetectionListAndDdmDopplerOffsetsDataFlow.addTransfer()
        .src(m_detectionListOp, detectionListLinePitch * sizeof(TI))
        .dst(detectionList_v, DETECTION_LIST_TILE_PITCH * sizeof(TI))
        .tile(DETECTION_LIST_TILE_WIDTH * sizeof(TI), maxNumDetections)
        .mode(TransferModeType::CONTINUOUS);

    targetCntdetectionListAndDdmDopplerOffsetsDataFlow.addTransfer()
        .src(m_ddmDopplerOffsetsOp, sizeof(float))
        .dst(ddmDopplerOffsets_v, sizeof(float))
        .tile(ddmDopplerOffsetsWidth * sizeof(float), DDM_DOPPLER_OFFSETS_TILE_HEIGHT)
        .mode(TransferModeType::CONTINUOUS);

    targetCntdetectionListAndDdmDopplerOffsetsDataFlow.addTransfer()
        .src(m_targetCountOp, sizeof(int32_t))
        .dst(targetCount_v, sizeof(int32_t))
        .tile(1 * sizeof(int32_t), 1)
        .mode(TransferModeType::CONTINUOUS);

    // Use OffsetPointers for tensor-based targetIndexMap/targetAngles
    targetIndexMapDataFlow.addTransfer()
        .src(m_targetIndexMapOp, sizeof(int32_t))
        .dst(targetIndexMap_v, sizeof(int32_t))
        .tile(targetWidth * sizeof(int32_t), TARGET_INDEX_MAP_TILE_HEIGHT);

    targetAnglesDataFlow.addTransfer()
        .src(m_targetAnglesOp, targetAnglesLinePitch * sizeof(float))
        .dst(targetAngles_v, TARGET_ANGLES_TILE_PITCH * sizeof(float))
        .tile(targetWidth * sizeof(float), TARGET_ANGLES_TILE_HEIGHT);

    targetListDataFlow.addTransfer()
        .src(targetList_v, TARGET_LIST_TILE_PITCH * sizeof(TO))
        .dst(m_targetListOp, targetListLinePitch * sizeof(TO))
        .tile(targetWidth * sizeof(TO), targetListHeight);

    // Configure program parameters
    prog["rangeResolution"]      = m_params.rangeResolution;
    prog["sweepTime"]            = m_params.sweepTime;
    prog["wavelength"]           = m_params.wavelength;
    prog["numChirps"]            = m_params.numChirps;
    prog["fftShiftedDoppler"]    = m_params.fftShiftedDoppler ? 1 : 0;
    prog["detectionCount"]       = maxNumDetections;
    prog["maxTargetCount"]       = static_cast<int32_t>(m_maxNumOfTargets);
    prog["power_output_enabled"] = targetListHeight == TARGET_LIST_TILE_HEIGHT ? 1 : 0;
    prog["longitudinalAxis"]     = m_params.longitudinalAxis;
    prog["groundProjection"]     = m_params.groundProjection;

    // Configure RV decoupling parameters (struct defined in target_processing_param.h)
    rv_decoupling_params_t rvParamsHost{};

    rvParamsHost.enableRvDecoupling           = m_params.enableRvDecoupling ? 1 : 0;
    rvParamsHost.enableQuadraticInterpolation = m_params.enableQuadraticInterpolation ? 1 : 0;

    // Default to Infineon RV decoupling mode
    if (m_params.enableRvDecoupling)
    {

        // Configure Infineon-style RV decoupling parameters
        // Compute derived parameters from base FMCW params

        // Compute derived RV decoupling parameters from base FMCW parameters
        rvParamsHost.f_fast_delta    = m_params.fs / static_cast<float>(m_params.nofSamples);
        rvParamsHost.f_slow_delta    = 1.0f / (m_params.PRI * static_cast<float>(m_params.numChirps));
        rvParamsHost.contR           = m_params.contR;
        rvParamsHost.contVFast       = m_params.contVFast;
        rvParamsHost.contVSlow       = m_params.contVSlow;
        rvParamsHost.velocityAmbHigh = -1.0f * m_params.deltaV * static_cast<float>(m_params.numChirps) / 2.0f / 3.6f;
        rvParamsHost.velocityAmbLow  = m_params.deltaV * static_cast<float>(m_params.numChirps) / 2.0f / 3.6f;
        // Compute 5 hypothesis coefficients: (1/PRI) * (i - 3) * contVSlow for i=0..4
        for (int i = 0; i < 5; i++)
        {
            rvParamsHost.tempVCoeff[i] = (1.0f / m_params.PRI) * (static_cast<float>(i) - 3.0f) * m_params.contVSlow;
        }
    }

    prog["rvParams"].set(&rvParamsHost, 1);

    // Set up L2 memory, SQDF for DRAM->L2 padding, and GSDF for L2->VMEM tiles if quadratic interpolation enabled
    if (m_params.enableQuadraticInterpolation && nciTensorRequirements != nullptr)
    {
        int64_t nciTensorShapeHW[2];
        nvcvTensorShapePermute(nciTensorRequirements->layout, nciTensorRequirements->shape, NVCV_TENSOR_HW,
                               nciTensorShapeHW);

        m_nciHeight       = static_cast<int32_t>(nciTensorShapeHW[0]);
        m_nciWidth        = static_cast<int32_t>(nciTensorShapeHW[1]);
        m_nciPaddedWidth  = m_nciWidth + 2;
        m_nciPaddedHeight = m_nciHeight + 2;
        m_nciL2Size       = m_nciPaddedWidth * m_nciPaddedHeight * sizeof(uint32_t);

        int32_t srcPitch = m_nciWidth * sizeof(uint32_t);
        int32_t dstPitch = m_nciPaddedWidth * sizeof(uint32_t);

        m_nciL2 = static_cast<uint32_t *>(cupva::mem::GetL2BaseAddress());

        auto nciPaddingHdl = prog["nciPaddingHdl"];
        phase++;
        SequenceDataFlow &nciPaddingSQDF = prog.addDataFlowHead<SequenceDataFlow>(phase).handler(nciPaddingHdl);

        nciPaddingSQDF.addTransfer()
            .src(m_nciSrcOp + (srcPitch - sizeof(uint32_t)), srcPitch)
            .dst(reinterpret_cast<uint8_t *>(m_nciL2) + dstPitch, dstPitch)
            .tile(sizeof(uint32_t), m_nciHeight)
            .mode(TransferModeType::CONTINUOUS);

        nciPaddingSQDF.addTransfer()
            .src(m_nciSrcOp, srcPitch)
            .dst(reinterpret_cast<uint8_t *>(m_nciL2) + dstPitch + sizeof(uint32_t), dstPitch)
            .tile(srcPitch, m_nciHeight)
            .mode(TransferModeType::CONTINUOUS);

        nciPaddingSQDF.addTransfer()
            .src(m_nciSrcOp, srcPitch)
            .dst(reinterpret_cast<uint8_t *>(m_nciL2) + dstPitch + sizeof(uint32_t) + srcPitch, dstPitch)
            .tile(sizeof(uint32_t), m_nciHeight)
            .mode(TransferModeType::CONTINUOUS);

        // Top row padding: cyclic wrap of NCI last row (NCI[nciHeight-1]) into padded[0][1..nciWidth].
        // The GSDF fetches a 3-row neighbourhood; for rangeBin=0 the "above" neighbour lands in padded[0],
        // which must contain NCI[nciHeight-1] to satisfy the cyclic boundary condition.
        nciPaddingSQDF.addTransfer()
            .src(m_nciSrcOp + srcPitch * (m_nciHeight - 1), srcPitch)
            .dst(reinterpret_cast<uint8_t *>(m_nciL2) + sizeof(uint32_t), dstPitch)
            .tile(srcPitch, 1)
            .mode(TransferModeType::CONTINUOUS);

        // Bottom row padding: cyclic wrap of NCI first row (NCI[0]) into padded[nciHeight+1][1..nciWidth].
        // For rangeBin=nciHeight-1 the "below" neighbour lands in padded[nciHeight+1].
        nciPaddingSQDF.addTransfer()
            .src(m_nciSrcOp, srcPitch)
            .dst(reinterpret_cast<uint8_t *>(m_nciL2) + dstPitch * (m_nciHeight + 1) + sizeof(uint32_t), dstPitch)
            .tile(srcPitch, 1)
            .mode(TransferModeType::CONTINUOUS);

        auto nciHdl                    = prog["nciHdl"];
        GatherScatterDataFlow &nciGSDF = prog.addDataFlowHead<GatherScatterDataFlow>(processingPhase).handler(nciHdl);
        nciGSDF.numTilesPerTrigger(NUM_NCI_NODES)
            .src(m_nciL2, m_nciPaddedWidth, m_nciPaddedHeight, m_nciPaddedWidth)
            .tile(NCI_NODE_WIDTH, NCI_NODE_HEIGHT);

        prog["nciPaddedWidth"]  = m_nciPaddedWidth;
        prog["nciPaddedHeight"] = m_nciPaddedHeight;
    }

    prog.compileDataFlows();

    if (m_nciL2 != nullptr)
    {
        prog.setL2Size(m_nciL2Size);
    }

    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename TI, typename TO>
TargetProcessing<TI, TO>::~TargetProcessing()
{
    CupvaCmdDestroy(&m_prog);
}

template<typename TI, typename TO>
void TargetProcessing<TI, TO>::updateDataFlows(const nvcv::Tensor &targetCountTensor,
                                               const nvcv::Tensor &detectionCountTensor,
                                               const nvcv::Tensor &detectionListTensor,
                                               const nvcv::Tensor &targetIndexMapTensor,
                                               const nvcv::Tensor &targetAnglesTensor,
                                               const nvcv::Tensor &ddmDopplerOffsetsTensor,
                                               const nvcv::Tensor &targetListTensor, const nvcv::Tensor *nciTensor)
{
    auto targetCountTensorData       = targetCountTensor.exportData<nvcv::TensorDataStrided>();
    auto detectionCountTensorData    = detectionCountTensor.exportData<nvcv::TensorDataStrided>();
    auto detectionListTensorData     = detectionListTensor.exportData<nvcv::TensorDataStrided>();
    auto targetIndexMapTensorData    = targetIndexMapTensor.exportData<nvcv::TensorDataStrided>();
    auto targetAnglesTensorData      = targetAnglesTensor.exportData<nvcv::TensorDataStrided>();
    auto ddmDopplerOffsetsTensorData = ddmDopplerOffsetsTensor.exportData<nvcv::TensorDataStrided>();
    auto targetListTensorData        = targetListTensor.exportData<nvcv::TensorDataStrided>();

    // Validate tensor requirements
    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_detectionListTensorRequirements,
                                                          detectionListTensorData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_targetIndexMapTensorRequirements,
                                                          targetIndexMapTensorData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_targetAnglesTensorRequirements,
                                                          targetAnglesTensorData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_ddmDopplerOffsetsTensorRequirements,
                                                          ddmDopplerOffsetsTensorData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_targetListTensorRequirements,
                                                          targetListTensorData->cdata()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Tensors must have the same requirements that were used to create the component");
    }

    // Validate pointers
    if ((static_cast<void *>(targetCountTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(detectionCountTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(detectionListTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(targetIndexMapTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(targetAnglesTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(ddmDopplerOffsetsTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(targetListTensorData->basePtr()) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor base pointers must not be NULL");
    }

    // Set base pointers for all tensors
    m_targetCountOp.setBase(static_cast<int32_t *>(static_cast<void *>(targetCountTensorData->basePtr())));
    m_detectionCountOp.setBase(static_cast<int32_t *>(static_cast<void *>(detectionCountTensorData->basePtr())));
    m_detectionListOp.setBase(static_cast<TI *>(static_cast<void *>(detectionListTensorData->basePtr())));
    m_targetIndexMapOp.setBase(static_cast<int32_t *>(static_cast<void *>(targetIndexMapTensorData->basePtr())));
    m_targetAnglesOp.setBase(static_cast<float *>(static_cast<void *>(targetAnglesTensorData->basePtr())));
    m_ddmDopplerOffsetsOp.setBase(static_cast<float *>(static_cast<void *>(ddmDopplerOffsetsTensorData->basePtr())));
    m_targetListOp.setBase(static_cast<TO *>(static_cast<void *>(targetListTensorData->basePtr())));

    // Set NCI source base pointer for SQDF if quadratic interpolation is enabled
    if (m_nciL2 != nullptr && nciTensor != nullptr)
    {
        auto nciTensorData = nciTensor->exportData<nvcv::TensorDataStrided>();
        if (static_cast<void *>(nciTensorData->basePtr()) != nullptr)
        {
            m_nciSrcOp.setBase(static_cast<uint8_t *>(static_cast<void *>(nciTensorData->basePtr())));
        }
    }

    // Get external memory pointer parameters
    cupvaParameter_t targetAnglesExtPtr, targetIndexMapExtPtr, targetListExtPtr;
    CupvaCmdProgramGetParameter(&m_prog, &targetIndexMapExtPtr, "targetIndexMapExtPtr");
    CupvaCmdProgramGetParameter(&m_prog, &targetAnglesExtPtr, "targetAnglesExtPtr");
    CupvaCmdProgramGetParameter(&m_prog, &targetListExtPtr, "targetListExtPtr");

    // Set pointer values from tensor base pointers
    CupvaParameterSetValuePointer(&targetAnglesExtPtr, static_cast<void *>(targetAnglesTensorData->basePtr()));
    CupvaParameterSetValuePointer(&targetIndexMapExtPtr, static_cast<void *>(targetIndexMapTensorData->basePtr()));
    CupvaParameterSetValuePointer(&targetListExtPtr, static_cast<void *>(targetListTensorData->basePtr()));

    // Update data flows
    cupvaError_t err = CupvaCmdProgramUpdateDataFlows(&m_prog);
    if (err != CUPVA_ERROR_NONE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "Failed to update data flows");
    }
}

// Explicit template instantiation
template class TargetProcessing<int32_t, float>;

} // namespace pva_solutions::priv
