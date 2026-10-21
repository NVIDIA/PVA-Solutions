/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OpSnapshotExtraction.cpp
 * @brief Host-side operator implementation for Snapshot Extraction
 *
 * This file implements the host-side interface for the Snapshot Extraction
 * operator, which performs DDM (Doppler Division Multiplexing) disambiguation
 * and gathers antenna snapshots from the Range-Doppler map for detected targets.
 *
 * Main Responsibilities:
 * 1. Tensor requirement calculation and validation
 * 2. VPU executable management (loading and caching)
 * 3. Task submission to PVA via cuPVA runtime
 * 4. Stream synchronization for CUDA and cuPVA
 *
 * Input Tensors:
 * - Detection count: [1] int32 scalar
 * - Folded detection list: [N, 2] int32 (range index, folded doppler index)
 * - DDM Doppler offsets: [M] float32 (M = numDopplerFolds)
 * - NCI Rx: [H, W] uint32 (H = numRangeBins, W = numDopplerBins)
 * - Range-Doppler map: [H, C, W] complex int32 (H = numRangeBins, C = numRx, W = numDopplerBins)
 *
 * Output Tensors:
 * - Detection list: [N, 2] int32 (range index, unfolded doppler index)
 * - Snapshots: [N, Tx, Rx] complex int32 (N detections, Tx antennas, Rx antennas)
 */

#include "OpSnapshotExtraction.hpp"

#include "device/snapshot_extraction_param.h"

#include <IOperator.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>

#include <OpSnapshotExtraction.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <cmath>
#include <complex>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_snapshot_extraction_gsdf_dev);
CUPVA_DECLARE_EXECUTABLE(pva_snapshot_extraction_sqdf_dev);

namespace pva_solutions::priv {

template<typename TI>
std::mutex SnapshotExtraction<TI>::s_execMapMutex;

template<typename TI>
std::map<typename SnapshotExtraction<TI>::ExecKey, std::weak_ptr<cupva::Executable>>
    SnapshotExtraction<TI>::s_SnapshotExtractionExecutables;

template<typename TI>
std::map<typename SnapshotExtraction<TI>::ExecDataKey, IOperator::ExecutableData> createExecDataMap()
{
    return {
        {std::make_tuple(false), // SQDF
         {CUPVA_EXECUTABLE_DATA(pva_snapshot_extraction_sqdf_dev),
          CUPVA_EXECUTABLE_SIZE(pva_snapshot_extraction_sqdf_dev)}},
        {std::make_tuple(true), // GSDF
         {CUPVA_EXECUTABLE_DATA(pva_snapshot_extraction_gsdf_dev),
          CUPVA_EXECUTABLE_SIZE(pva_snapshot_extraction_gsdf_dev)}},
    };
}

template<typename TI>
std::shared_ptr<cupva::Executable> SnapshotExtraction<TI>::GetSnapshotExtractionExec(ExecDataKey execDataKey)
{
    static auto execDataMap = createExecDataMap<TI>();

    auto execDataIt = execDataMap.find(execDataKey);
    if (execDataIt != execDataMap.end())
    {
        // Create execKey by combining execDataKey and context
        ExecKey execKey = std::tuple_cat(execDataKey, std::make_tuple(cupva::Context::GetCurrent()));
        return IOperator::GetExecutable<ExecKey>(execDataIt->second, execKey, s_SnapshotExtractionExecutables,
                                                 s_execMapMutex);
    }
    else
    {
        throw std::invalid_argument("Unsupported snapshot extraction mode");
    }
}

/**
 * @brief Constructor for SnapshotExtraction operator
 *
 * Initializes the operator by:
 * 1. Validating input tensor requirements
 * 2. Computing derived parameters (numTx from snapshot dimensions)
 * 3. Loading PVA executable
 * 4. Pre-allocating and uploading coefficient data
 */
template<typename TI>
SnapshotExtraction<TI>::SnapshotExtraction(NVCVTensorRequirements const *const foldedDetectionListTensorRequirements,
                                           NVCVTensorRequirements const *const ddmDopplerOffsetsTensorRequirements,
                                           NVCVTensorRequirements const *const nciRxTensorRequirements,
                                           NVCVTensorRequirements const *const rangeDopplerMapTensorRequirements,
                                           NVCVTensorRequirements const *const detectionListTensorRequirements,
                                           NVCVTensorRequirements const *const snapshotsTensorRequirements,
                                           NVCVTensorRequirements const *const calibrationWeightsTensorRequirements,
                                           PVASnapshotExtractionParams const *const params)
    : m_foldedDetectionListTensorRequirements(*foldedDetectionListTensorRequirements)
    , m_ddmDopplerOffsetsTensorRequirements(*ddmDopplerOffsetsTensorRequirements)
    , m_nciRxTensorRequirements(*nciRxTensorRequirements)
    , m_rangeDopplerMapTensorRequirements(*rangeDopplerMapTensorRequirements)
    , m_detectionListTensorRequirements(*detectionListTensorRequirements)
    , m_snapshotsTensorRequirements(*snapshotsTensorRequirements)
    , m_calibrationWeightsTensorRequirements(calibrationWeightsTensorRequirements != nullptr
                                                 ? *calibrationWeightsTensorRequirements
                                                 : NVCVTensorRequirements{})
    , m_params(*params)
    , m_foldedDetectionListOp(cupva::OffsetPointer<int32_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_nciRxOp(cupva::OffsetPointer<uint32_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_rangeDopplerMapOp(cupva::OffsetPointer<TI>::Create(cupva::MemType::DRAM, nullptr))
    , m_detectionListOp(cupva::OffsetPointer<int32_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_snapshotsOp(cupva::OffsetPointer<TI>::Create(cupva::MemType::DRAM, nullptr))
    , m_calibrationWeightsOp(cupva::OffsetPointer<int32_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_DetectionCountOp(cupva::OffsetPointer<int32_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_ddmaWeights_d(nullptr)
    , m_ddmSpace_d(nullptr)
    , m_useCalibrationWeights(false)
    , m_useGSDFForRDM(true) // Will be computed after extracting numTx and numRx
{
    // Validate parameters
    if (m_params.numDopplerFolds <= 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "numDopplerFolds must be positive");
    }

    // Extract tensor dimensions for validation
    // Folded Detection List tensor validation
    int64_t foldedDetShapeNCHW[4];
    nvcvTensorShapePermute(m_foldedDetectionListTensorRequirements.layout,
                           m_foldedDetectionListTensorRequirements.shape, NVCV_TENSOR_NCHW, foldedDetShapeNCHW);
    int32_t maxFoldedDetections = foldedDetShapeNCHW[2];

    // Validate maximum detections limit based on VMEM buffer size (folded detection list)
    if (maxFoldedDetections > MAX_DETECTIONS)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Maximum number of folded detections is %d (VMEM buffer limit), got %d", MAX_DETECTIONS,
                              maxFoldedDetections);
    }

    if (foldedDetShapeNCHW[3] != 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Folded detection list second dimension must be 2, got %d", (int)foldedDetShapeNCHW[3]);
    }

    // Detection List tensor validation (replicated logic)
    int64_t detListShapeNCHW[4];
    nvcvTensorShapePermute(m_detectionListTensorRequirements.layout, m_detectionListTensorRequirements.shape,
                           NVCV_TENSOR_NCHW, detListShapeNCHW);
    int32_t maxDetections = detListShapeNCHW[2];

    if (maxDetections != maxFoldedDetections)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "detection list tensor size must match folded detection list tensor size");
    }

    if (detListShapeNCHW[3] != 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Detection list second dimension must be 2, got %d",
                              (int)detListShapeNCHW[3]);
    }

    int64_t ddmShapeNCHW[4];
    nvcvTensorShapePermute(m_ddmDopplerOffsetsTensorRequirements.layout, m_ddmDopplerOffsetsTensorRequirements.shape,
                           NVCV_TENSOR_NCHW, ddmShapeNCHW);
    int32_t numDopplerFolds = ddmShapeNCHW[3];

    if (numDopplerFolds != m_params.numDopplerFolds)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "DDM Doppler offsets tensor size must match numDopplerFolds parameter");
    }

    int64_t nciShapeNCHW[4];
    nvcvTensorShapePermute(m_nciRxTensorRequirements.layout, m_nciRxTensorRequirements.shape, NVCV_TENSOR_NCHW,
                           nciShapeNCHW);
    int32_t numRangeBins   = nciShapeNCHW[2];
    int32_t numDopplerBins = nciShapeNCHW[3];

    int64_t rdmShapeNCHW[4];
    nvcvTensorShapePermute(m_rangeDopplerMapTensorRequirements.layout, m_rangeDopplerMapTensorRequirements.shape,
                           NVCV_TENSOR_NCHW, rdmShapeNCHW);

    if (m_params.rangeDopplerFormat != PVA_FFT_LAYOUT_RANGE_RX_DOPPLER &&
        m_params.rangeDopplerFormat != PVA_FFT_LAYOUT_RANGE_DOPPLER_RX)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Unsupported rangeDopplerFormat %d; SnapshotExtraction supports only "
                              "PVA_FFT_LAYOUT_RANGE_RX_DOPPLER (%d) and PVA_FFT_LAYOUT_RANGE_DOPPLER_RX (%d)",
                              static_cast<int32_t>(m_params.rangeDopplerFormat),
                              static_cast<int32_t>(PVA_FFT_LAYOUT_RANGE_RX_DOPPLER),
                              static_cast<int32_t>(PVA_FFT_LAYOUT_RANGE_DOPPLER_RX));
    }

    // Extract dimensions based on Range-Doppler format
    int32_t rdmNumRangeBins, numRx, rdmNumDopplerBins;
    if (m_params.rangeDopplerFormat == PVA_FFT_LAYOUT_RANGE_RX_DOPPLER)
    {
        // RANGE_RX_DOPPLER format: [numRangeBins, numRx, numDopplerBins]
        // NCHW mapping: N=1, C=numRx, H=numRangeBins, W=numDopplerBins
        rdmNumRangeBins   = rdmShapeNCHW[2];
        numRx             = rdmShapeNCHW[1];
        rdmNumDopplerBins = rdmShapeNCHW[3];
    }
    else // PVA_FFT_LAYOUT_RANGE_DOPPLER_RX
    {
        // RANGE_DOPPLER_RX format: [numRangeBins, numDopplerBins, numRx]
        // NCHW mapping: N=1, C=numDopplerBins, H=numRangeBins, W=numRx
        rdmNumRangeBins   = rdmShapeNCHW[2];
        numRx             = rdmShapeNCHW[3];
        rdmNumDopplerBins = rdmShapeNCHW[1];
    }

    if (rdmNumRangeBins != numRangeBins || rdmNumDopplerBins != numDopplerBins)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Range-Doppler map dimensions must match NCI Rx dimensions");
    }

    int64_t snapshotsShapeCHW[3];
    nvcvTensorShapePermute(m_snapshotsTensorRequirements.layout, m_snapshotsTensorRequirements.shape, NVCV_TENSOR_CHW,
                           snapshotsShapeCHW);
    int32_t numTx = snapshotsShapeCHW[1]; // Number of physical TX antennas for snapshot output
    int32_t snRx  = snapshotsShapeCHW[2];

    if (snapshotsShapeCHW[0] != maxDetections)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Snapshots tensor channel dimension must match detection list tensor size");
    }

    if (snRx != numRx)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Snapshots Rx dimension must match Range-Doppler map Rx dimension");
    }

    // Validate antenna configuration - only 4x4 and 8x8 configurations are supported
    bool isValid4x4 = (numTx == 4 && numRx == 4);
    bool isValid8x8 = (numTx == 8 && numRx == 8);

    if (!isValid4x4 && !isValid8x8)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Only 4x4 and 8x8 TX×RX antenna configurations are supported. Got: numTx=%d, numRx=%d",
                              numTx, numRx);
    }

    // Validate numDopplerFolds against numTx
    if (numDopplerFolds < numTx)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "numDopplerFolds (%d) cannot be less than numTx (%d)", numDopplerFolds, numTx);
    }

    // Validate numDopplerFolds upper limit
    if (numDopplerFolds > MAX_DDM_FOLDS)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "numDopplerFolds (%d) cannot exceed %d",
                              numDopplerFolds, MAX_DDM_FOLDS);
    }

    // Validate that numDopplerBins is evenly divisible by numDopplerFolds
    if (numDopplerBins % numDopplerFolds != 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "numDopplerBins (%d) must be evenly divisible by numDopplerFolds (%d). Remainder: %d",
                              numDopplerBins, numDopplerFolds, numDopplerBins % numDopplerFolds);
    }

    // Validate calibration weights tensor if provided (optional - can be NULL to use DDMA weights)
    if (calibrationWeightsTensorRequirements != nullptr)
    {
        m_useCalibrationWeights = true;

        int64_t calShapeNCHW[4];
        nvcvTensorShapePermute(m_calibrationWeightsTensorRequirements.layout,
                               m_calibrationWeightsTensorRequirements.shape, NVCV_TENSOR_NCHW, calShapeNCHW);

        if (calShapeNCHW[2] != numTx || calShapeNCHW[3] != numRx)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Calibration weights must be [numTx=%d, numRx=%d], got [%d, %d]", numTx, numRx,
                                  (int)calShapeNCHW[2], (int)calShapeNCHW[3]);
        }
    }

    // Determine whether to use GSDF or SQDF based on antenna configuration and format
    // Logic:
    // - when numTx * numRx <= 16, use GSDF irrespective of format (many tiles, GSDF is more efficient)
    // - when numTx * numRx > 16 and format is RANGE_DOPPLER_RX, use GSDF (optimized: numTx tiles only, fewer tiles per detection)
    // - when numTx * numRx > 16 and format is RANGE_RX_DOPPLER, use SQDF (simple case, SQDF is sufficient)
    if (numTx * numRx <= 16)
    {
        m_useGSDFForRDM = true;
    }
    else if (numTx * numRx > 16 && m_params.rangeDopplerFormat == PVA_FFT_LAYOUT_RANGE_DOPPLER_RX)
    {
        m_useGSDFForRDM = true;
    }
    else
    {
        m_useGSDFForRDM = false;
    }

    // Load PVA executable (GSDF or SQDF variant based on configuration)
    m_exec = GetSnapshotExtractionExec(std::make_tuple(m_useGSDFForRDM));

    if (!m_exec)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_OUT_OF_MEMORY, "Failed to load SnapshotExtraction executable");
    }

    // Allocate and pre-compute coefficient buffers
    // Note: numTx (8) is physical TX antennas, numDopplerFolds (16) is DDM disambiguation parameter

    // Allocate DDMA weights buffer only if calibration weights are not provided
    // For calibration weights, we'll use the user's tensor directly (no intermediate allocation)
    if (calibrationWeightsTensorRequirements == nullptr)
    {
        // DDMA weights: [numTx] complex - one weight per physical TX antenna
        m_ddmaWeights_d        = (int32_t *)mem::Alloc(numTx * 2 * sizeof(int32_t));
        int32_t *ddmaWeights_h = (int32_t *)mem::GetHostPointer(m_ddmaWeights_d);
        if (ddmaWeights_h == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_OUT_OF_MEMORY, "Failed to get host pointer for DDMA weights");
        }
    }

    // DDM space: [numTx] - one value per TX antenna
    m_ddmSpace_d        = (int32_t *)mem::Alloc(numTx * sizeof(int32_t));
    int32_t *ddmSpace_h = (int32_t *)mem::GetHostPointer(m_ddmSpace_d);
    if (ddmSpace_h == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_OUT_OF_MEMORY, "Failed to get host pointer for DDM space");
    }

    // Create PVA program and set static parameters
    auto prog = CmdProgram::Create(*m_exec.get());

    // Set static parameters that don't change between operator calls
    const int32_t rangeBatchSize   = 1; // Process range bins one at a time
    const int32_t numChirpsPerFold = numDopplerBins / m_params.numDopplerFolds;

    prog["numRangeBins"]       = numRangeBins;
    prog["numDopplerBins"]     = numDopplerBins;
    prog["numRx"]              = numRx;
    prog["numTx"]              = numTx;                    // Physical TX antennas
    prog["numDopplerFolds"]    = m_params.numDopplerFolds; // DDM disambiguation folds
    prog["numChirpsPerFold"]   = numChirpsPerFold;
    prog["rangeBatchSize"]     = rangeBatchSize;
    prog["rangeDopplerFormat"] = static_cast<int32_t>(m_params.rangeDopplerFormat); // RDM memory layout format
    prog["useCalibrationWeights"] =
        (calibrationWeightsTensorRequirements != nullptr) ? 1 : 0; // Use calibration if provided
    prog["maxDetections"] = maxDetections;

    // Set up data flows for coefficient loading and I/O transfers
    int32_t phase = 0;

    // Get VMEM buffer pointers and handlers
    auto coeffHdl             = prog["coeffHdl"];
    int32_t *detectionCount_v = prog["numDetections"].ptr<int32_t>();

    // Phase 0: Coefficient data flow (load weights and relative doppler offsets)
    SequenceDataFlow &coeffDataFlow = prog.addDataFlowHead<SequenceDataFlow>(phase).handler(coeffHdl);

    if (calibrationWeightsTensorRequirements != nullptr)
    {
        // Load calibration weights [numTx x numRx] complex from user tensor (DMA directly from user buffer)
        // Transfer entire buffer in one tile (similar to Range FFT windowing buffer pattern)
        int32_t *calibrationWeights_v       = prog["weightsBuf"].ptr<int32_t>();
        int32_t calibrationWeightsTotalSize = numTx * numRx * 2 * sizeof(int32_t);
        coeffDataFlow.addTransfer()
            .src(m_calibrationWeightsOp, calibrationWeightsTotalSize)
            .dst(calibrationWeights_v, calibrationWeightsTotalSize)
            .tile(calibrationWeightsTotalSize, 1)
            .mode(TransferModeType::CONTINUOUS);
    }
    else
    {
        // Load DDMA weights [numTx] complex
        int32_t *ddmaWeights_v = prog["weightsBuf"].ptr<int32_t>();
        coeffDataFlow.addTransfer()
            .src(static_cast<void *>(m_ddmaWeights_d), numTx * 2 * sizeof(int32_t))
            .dst(ddmaWeights_v, numTx * 2 * sizeof(int32_t))
            .tile(numTx * 2 * sizeof(int32_t), 1)
            .mode(TransferModeType::CONTINUOUS);
    }

    // Load precomputed DDM space [numTx]
    int32_t *ddmSpace_v = prog["ddmSpaceBuf"].ptr<int32_t>();
    coeffDataFlow.addTransfer()
        .src(static_cast<void *>(m_ddmSpace_d), numTx * sizeof(int32_t))
        .dst(ddmSpace_v, numTx * sizeof(int32_t))
        .tile(numTx * sizeof(int32_t), 1)
        .mode(TransferModeType::CONTINUOUS);

    coeffDataFlow.addTransfer()
        .src(m_DetectionCountOp, sizeof(int32_t))
        .dst(detectionCount_v, sizeof(int32_t))
        .tile(1 * sizeof(int32_t), 1)
        .mode(TransferModeType::CONTINUOUS);
    phase++;

    // Get handlers for input/output data
    auto detectionListHdl    = prog["detectionListHdl"];
    auto rdmHdl              = prog["rdmHdl"];
    auto nciHdl              = prog["nciHdl"];
    auto detectionListOutHdl = prog["detectionListOutHdl"];
    auto snapshotsOutHdl     = prog["snapshotsOutHdl"];

    // Get VMEM buffer pointers for input/output
    int32_t *foldedDetectionList_v = prog["foldedDetectionListBuf"].ptr<int32_t>();
    int32_t *rdm_v                 = prog["rdmBuf"].ptr<int32_t>();
    uint32_t *nci_v                = prog["nciBuf"].ptr<uint32_t>();
    int32_t *detectionListOut_v    = prog["detectionListOutBuf"].ptr<int32_t>();
    int32_t *snapshotsOut_v        = prog["snapshotsOutBuf"].ptr<int32_t>();

    // Phase 1: Folded detection list input (will transfer actual numDetections, max 8192)
    // Note: Actual transfer size is based on numDetections, known at runtime
    SequenceDataFlow &detectionListDataFlow = prog.addDataFlowHead<SequenceDataFlow>(phase).handler(detectionListHdl);
    detectionListDataFlow.addTransfer()
        .src(m_foldedDetectionListOp, 2 * sizeof(int32_t))
        .dst(foldedDetectionList_v, 2 * sizeof(int32_t))
        .tile(2 * sizeof(int32_t), maxDetections)
        .mode(TransferModeType::CONTINUOUS);

    phase++;

    // Phase 2: RDM and NCI batch input
    if (m_useGSDFForRDM)
    {
        // GSDF path: RDM is gathered based on bestDopplerIdxs computed per detection
        //
        // Optimization for RANGE_DOPPLER_RX format:
        // When rangeDopplerFormat is RANGE_DOPPLER_RX [numRangeBins, numDopplerBins, numRx],
        // all Rx data for a given (range, doppler) are contiguous in memory.
        // This allows us to use only numTx tiles (each tile gathers numRx complex values),
        // instead of numTx * numRx tiles (each tile gathering one complex value).
        // This reduces the number of GSDF gather operations and improves performance.
        //
        // Note: GSDF supports max 32 tiles per trigger, so may need multiple triggers in device code

        int32_t rdmWidth, totalTiles, tileWidth;
        if (m_params.rangeDopplerFormat == PVA_FFT_LAYOUT_RANGE_DOPPLER_RX)
        {
            // RANGE_DOPPLER_RX format: [numRangeBins, numDopplerBins, numRx] complex S32
            // For GSDF, view as 2D: width = numDopplerBins * numRx * 2 (int32), height = numRangeBins
            // Each tile fetches numRx complex values (numRx * 2 int32), total numTx tiles per detection
            rdmWidth   = numDopplerBins * numRx * 2; // 2 for complex (real, imag)
            totalTiles = numTx;                      // One tile per Tx (each tile gathers all Rx)
            tileWidth  = numRx * 2;                  // Each tile: numRx complex values
        }
        else // PVA_FFT_LAYOUT_RANGE_RX_DOPPLER
        {
            // RANGE_RX_DOPPLER format: [numRangeBins, numRx, numDopplerBins] complex S32
            // For GSDF, view as 2D: width = numRx * numDopplerBins * 2 (int32), height = numRangeBins
            // Each tile is one complex value (2 int32), total numTx * numRx tiles per detection
            rdmWidth   = numRx * numDopplerBins * 2; // 2 for complex (real, imag)
            totalTiles = numTx * numRx;              // One tile per Tx-Rx pair
            tileWidth  = 2;                          // Each tile: one complex value
        }

        const int32_t maxTilesPerTrigger = 32; // Hardware limitation
        const int32_t tilesPerTrigger    = (totalTiles > maxTilesPerTrigger) ? maxTilesPerTrigger : totalTiles;

        GatherScatterDataFlow &rdmDataFlow = prog.addDataFlowHead<GatherScatterDataFlow>(phase).handler(rdmHdl);
        rdmDataFlow.numTilesPerTrigger(tilesPerTrigger)
            .tile(tileWidth, 1) // Tile width depends on format
            .src(m_rangeDopplerMapOp, rdmWidth, numRangeBins, rdmWidth);
    }
    else
    {
        // SQDF path: Load full RDM batch (loaded per batch during processing)
        SequenceDataFlow &rdmDataFlow = prog.addDataFlowHead<SequenceDataFlow>(phase).handler(rdmHdl);
        rdmDataFlow.addTransfer()
            .src(m_rangeDopplerMapOp, rangeBatchSize * numRx * numDopplerBins * 2 * sizeof(int32_t))
            .dst(rdm_v, rangeBatchSize * numRx * numDopplerBins * 2 * sizeof(int32_t))
            .tile(rangeBatchSize * numRx * numDopplerBins * 2 * sizeof(int32_t), 1)
            .mode(TransferModeType::CONTINUOUS);
    }

    //phase++;
    SequenceDataFlow &nciDataFlow = prog.addDataFlowHead<SequenceDataFlow>(phase).handler(nciHdl);
    nciDataFlow.addTransfer()
        .src(m_nciRxOp, rangeBatchSize * numDopplerBins * sizeof(uint32_t))
        .dst(nci_v, rangeBatchSize * numDopplerBins * sizeof(uint32_t))
        .tile(rangeBatchSize * numDopplerBins * sizeof(uint32_t), 1)
        .mode(TransferModeType::CONTINUOUS);

    // Snapshot output - transfer one detection's snapshot at a time (numTx * numRx complex values)
    SequenceDataFlow &snapshotsOutDataFlow = prog.addDataFlowHead<SequenceDataFlow>(phase).handler(snapshotsOutHdl);
    snapshotsOutDataFlow.addTransfer()
        .src(snapshotsOut_v, numTx * numRx * 2 * sizeof(int32_t))
        .dst(m_snapshotsOp, numTx * numRx * 2 * sizeof(int32_t))
        .tile(numTx * numRx * 2 * sizeof(int32_t), 1)
        .mode(TransferModeType::CONTINUOUS);

    phase++;

    // Phase 3: Detection list output - transfer all detections in a single transfer
    // Note: Transfer size based on actual numDetections (max 8192)
    SequenceDataFlow &detectionListOutDataFlow =
        prog.addDataFlowHead<SequenceDataFlow>(phase).handler(detectionListOutHdl);
    detectionListOutDataFlow.addTransfer()
        .src(detectionListOut_v, 2 * sizeof(int32_t))
        .dst(m_detectionListOp, 2 * sizeof(int32_t))
        .tile(2 * sizeof(int32_t), maxDetections)
        .mode(TransferModeType::CONTINUOUS);

    // Compile data flows and convert to command
    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename TI>
SnapshotExtraction<TI>::~SnapshotExtraction()
{
    // Free PVA memory buffers
    if (m_ddmaWeights_d)
    {
        mem::Free(m_ddmaWeights_d);
        m_ddmaWeights_d = nullptr;
    }
    if (m_ddmSpace_d)
    {
        mem::Free(m_ddmSpace_d);
        m_ddmSpace_d = nullptr;
    }
}

/**
 * @brief Operator call for PVA stream submission with batched data transfer
 *
 * @param stream cuPVA or CUDA stream for execution
 * @param detectionCountTensor Input: Detection count [1] int32
 * @param foldedDetectionListTensor Input: Folded detection list [N, 2] int32
 * @param ddmDopplerOffsetsTensor Input: DDM Doppler offsets [M] float32
 * @param nciRxTensor Input: NCI Rx [H, W] uint32
 * @param rangeDopplerMapTensor Input: Range-Doppler map [H, C, W] complex int32
 * @param detectionListTensor Output: Unfolded detection list [N, 2] int32
 * @param snapshotsTensor Output: Snapshots [N, Tx, Rx] complex int32
 */
template<typename TI>
template<typename S>
void SnapshotExtraction<TI>::operator()(S stream, const nvcv::Tensor &detectionCountTensor,
                                        const nvcv::Tensor &foldedDetectionListTensor,
                                        const nvcv::Tensor &ddmDopplerOffsetsTensor, const nvcv::Tensor &nciRxTensor,
                                        const nvcv::Tensor &rangeDopplerMapTensor,
                                        const nvcv::Tensor &calibrationWeightsTensor,
                                        const nvcv::Tensor &detectionListTensor, const nvcv::Tensor &snapshotsTensor)
{
    // Extract tensor data pointers and dimensions
    auto detectionCountData      = detectionCountTensor.exportData<nvcv::TensorDataStrided>();
    auto foldedDetectionListData = foldedDetectionListTensor.exportData<nvcv::TensorDataStrided>();
    auto ddmDopplerOffsetsData   = ddmDopplerOffsetsTensor.exportData<nvcv::TensorDataStrided>();
    auto nciRxData               = nciRxTensor.exportData<nvcv::TensorDataStrided>();
    auto rangeDopplerMapData     = rangeDopplerMapTensor.exportData<nvcv::TensorDataStrided>();
    auto detectionListData       = detectionListTensor.exportData<nvcv::TensorDataStrided>();
    auto snapshotsData           = snapshotsTensor.exportData<nvcv::TensorDataStrided>();

    // Export calibration weights if provided (check if tensor handle is not null)
    nvcv::Optional<nvcv::TensorDataStrided> calibrationWeightsData;
    if (m_useCalibrationWeights)
    {
        calibrationWeightsData = calibrationWeightsTensor.exportData<nvcv::TensorDataStrided>();

        if (!pva_solutions::utils::AreTensorRequirementsEqual(m_calibrationWeightsTensorRequirements,
                                                              calibrationWeightsData->cdata()))
        {
            throw nvcv::Exception(
                nvcv::Status::ERROR_INVALID_ARGUMENT,
                "Calibration weights tensor must have the same requirements that were used to create the operator");
        }
        if (static_cast<void *>(calibrationWeightsData->basePtr()) == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Calibration weights tensor base pointer must not be NULL when provided");
        }
    }

    // Validate tensor requirements match those used during create
    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_foldedDetectionListTensorRequirements,
                                                          foldedDetectionListData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_ddmDopplerOffsetsTensorRequirements,
                                                          ddmDopplerOffsetsData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_nciRxTensorRequirements, nciRxData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_rangeDopplerMapTensorRequirements,
                                                          rangeDopplerMapData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_detectionListTensorRequirements,
                                                          detectionListData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_snapshotsTensorRequirements, snapshotsData->cdata()))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Input/output tensors must have the same requirements that were used to create the operator");
    }

    // Validate tensor base pointers are not null
    if ((static_cast<void *>(detectionCountData->basePtr()) == nullptr) ||
        (static_cast<void *>(foldedDetectionListData->basePtr()) == nullptr) ||
        (static_cast<void *>(ddmDopplerOffsetsData->basePtr()) == nullptr) ||
        (static_cast<void *>(nciRxData->basePtr()) == nullptr) ||
        (static_cast<void *>(rangeDopplerMapData->basePtr()) == nullptr) ||
        (static_cast<void *>(detectionListData->basePtr()) == nullptr) ||
        (static_cast<void *>(snapshotsData->basePtr()) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor base pointers must not be NULL");
    }

    // Get tensor dimensions to compute coefficients
    int64_t snapshotsShapeCHW[3];
    nvcvTensorShapePermute(m_snapshotsTensorRequirements.layout, m_snapshotsTensorRequirements.shape, NVCV_TENSOR_CHW,
                           snapshotsShapeCHW);
    int32_t numTx = snapshotsShapeCHW[1];
    int32_t numRx = snapshotsShapeCHW[2];

    int64_t nciShapeNCHW[4];
    nvcvTensorShapePermute(m_nciRxTensorRequirements.layout, m_nciRxTensorRequirements.shape, NVCV_TENSOR_NCHW,
                           nciShapeNCHW);
    int32_t numDopplerBins = nciShapeNCHW[3];

    // Compute coefficient data: weights (DDMA or calibration) and relative doppler offsets
    const float PI_CONST_F = 3.14159265358979323846f;

    // Get DDM doppler offsets from input tensor (need host pointer for PVA memory)
    float *ddmDopplerOffsetsPtr = nullptr;
    CupvaMemGetHostPointer((void **)&ddmDopplerOffsetsPtr, (void *)ddmDopplerOffsetsData->basePtr());
    if (ddmDopplerOffsetsPtr == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Failed to get host pointer for DDM doppler offsets tensor");
    }

    // Compute relative doppler offsets (needed for DDM space computation on host only)
    std::vector<int32_t> relativeDopplerOffsets_h(numTx);
    for (int32_t txIdx = 0; txIdx < numTx; txIdx++)
    {
        relativeDopplerOffsets_h[txIdx] =
            static_cast<int32_t>((ddmDopplerOffsetsPtr[0] - ddmDopplerOffsetsPtr[txIdx]) * numDopplerBins);
    }

    // Compute DDM space on host [numTx]
    // DDMspace stores the full doppler offset for each TX antenna (used in bestDopplerIdxs calculation)
    int32_t *ddmSpace_h = (int32_t *)mem::GetHostPointer(m_ddmSpace_d);
    for (int32_t txIdx = 0; txIdx < numTx; txIdx++)
    {
        ddmSpace_h[txIdx] = ((relativeDopplerOffsets_h[txIdx] + numDopplerBins) % numDopplerBins);
    }

    // Compute DDM space bitmask for device optimization
    // Derive folded DDM space from the full DDM space already computed above
    uint32_t ddmSpaceBitMask = 0;
    {
        int32_t numDopplerFolds = m_params.numDopplerFolds;
        int32_t foldSize        = numDopplerBins / numDopplerFolds;
        std::vector<uint32_t> bitMaskVec(numDopplerFolds * 2, 1);

        for (int32_t txIdx = 0; txIdx < numTx; txIdx++)
        {
            int32_t foldedDdmSpace        = ddmSpace_h[txIdx] / foldSize;
            bitMaskVec.at(foldedDdmSpace) = 0;
        }

        // Copy the first half of bitMaskVec to its second half
        std::copy(bitMaskVec.begin(), bitMaskVec.begin() + numDopplerFolds, bitMaskVec.begin() + numDopplerFolds);

        for (int32_t i = 0; i < 2 * numDopplerFolds - 1; i++)
        {
            ddmSpaceBitMask |= bitMaskVec[1 + i] << i; // start from the second element
        }
    }

    // Compute weights based on whether calibration weights are provided
    if (m_useCalibrationWeights == true)
    {
        // Get PVA-accessible pointer for calibration weights tensor
        // Note: For OffsetPointers used in DMA, we use basePtr() directly (not CupvaMemGetHostPointer)
        // This matches the pattern used for other input tensors (detectionList, rangeDopplerMap, etc.)
        void *calibrationWeightsBasePtr = static_cast<void *>(calibrationWeightsData->basePtr());
        m_calibrationWeightsOp.setBase(static_cast<int32_t *>(calibrationWeightsBasePtr));
    }
    else
    {
        // Compute DDMA weights (fixed-point complex values)
        int32_t *ddmaWeights_h = (int32_t *)mem::GetHostPointer(m_ddmaWeights_d);
        for (int32_t txIdx = 0; txIdx < numTx; txIdx++)
        {
            float dopplerOffset = ddmDopplerOffsetsPtr[txIdx];
            std::complex<float> imaginaryUnit(0.0f, 1.0f);
            std::complex<float> weightFloat = std::conj(std::exp(-imaginaryUnit * 2.0f * PI_CONST_F * dopplerOffset));

            ddmaWeights_h[txIdx * 2]     = static_cast<int32_t>(std::round(weightFloat.real() * DDMA_WEIGHTS_SCALE));
            ddmaWeights_h[txIdx * 2 + 1] = static_cast<int32_t>(std::round(weightFloat.imag() * DDMA_WEIGHTS_SCALE));
        }
    }

    // Update base pointers for offset pointers (tensor addresses may change per call)
    m_foldedDetectionListOp.setBase(static_cast<int32_t *>(static_cast<void *>(foldedDetectionListData->basePtr())));
    m_nciRxOp.setBase(static_cast<uint32_t *>(static_cast<void *>(nciRxData->basePtr())));
    m_rangeDopplerMapOp.setBase(static_cast<TI *>(static_cast<void *>(rangeDopplerMapData->basePtr())));
    m_detectionListOp.setBase(static_cast<int32_t *>(static_cast<void *>(detectionListData->basePtr())));
    m_snapshotsOp.setBase(static_cast<TI *>(static_cast<void *>(snapshotsData->basePtr())));
    m_DetectionCountOp.setBase(static_cast<int32_t *>(static_cast<void *>(detectionCountData->basePtr())));

    cupvaParameter_t param;
    CupvaCmdProgramGetParameter(&m_prog, &param, "ddmSpaceBitMask");
    CupvaParameterSetValueScalar(&param, &ddmSpaceBitMask, sizeof(uint32_t));

    // Set external memory pointers for SQDF and GSDF
    cupvaParameter_t rdmExtPtr, nciExtPtr, snapshotsOutExtPtr;

    CupvaCmdProgramGetParameter(&m_prog, &rdmExtPtr, "rdmExtPtr");
    CupvaCmdProgramGetParameter(&m_prog, &nciExtPtr, "nciExtPtr");
    CupvaCmdProgramGetParameter(&m_prog, &snapshotsOutExtPtr, "snapshotsOutExtPtr");

    CupvaParameterSetValuePointer(&rdmExtPtr, static_cast<void *>(rangeDopplerMapData->basePtr()));
    CupvaParameterSetValuePointer(&nciExtPtr, static_cast<void *>(nciRxData->basePtr()));
    CupvaParameterSetValuePointer(&snapshotsOutExtPtr, static_cast<void *>(snapshotsData->basePtr()));

    cupvaError_t err = CupvaCmdProgramUpdateDataFlows(&m_prog);
    if (err != CUPVA_ERROR_NONE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "Failed to update data flows");
    }

    // Submit program to stream
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
        static_assert(!sizeof(S), "Unsupported stream type for SnapshotExtraction operator");
    }

    if (err != CUPVA_ERROR_NONE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "Failed to submit PVA program");
    }
}

// Explicit template instantiation for supported data types
template class SnapshotExtraction<int32_t>;

} // namespace pva_solutions::priv

/**
 * @brief Create a SnapshotExtraction operator instance
 *
 * This is the C API entry point for creating a SnapshotExtraction operator.
 * It validates input parameters and constructs the operator object.
 */
NVCVStatus pvaSnapshotExtractionCreate(NVCVOperatorHandle *handle,
                                       NVCVTensorRequirements const *const foldedDetectionListTensorRequirements,
                                       NVCVTensorRequirements const *const ddmDopplerOffsetsTensorRequirements,
                                       NVCVTensorRequirements const *const nciRxTensorRequirements,
                                       NVCVTensorRequirements const *const rangeDopplerMapTensorRequirements,
                                       NVCVTensorRequirements const *const detectionListTensorRequirements,
                                       NVCVTensorRequirements const *const snapshotsTensorRequirements,
                                       NVCVTensorRequirements const *const calibrationWeightsTensorRequirements,
                                       PVASnapshotExtractionParams const *const params)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        if (foldedDetectionListTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to folded detection list NVCVTensorRequirements must not be NULL");
        }
        if (ddmDopplerOffsetsTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to DDM Doppler offsets NVCVTensorRequirements must not be NULL");
        }
        if (nciRxTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NCI Rx NVCVTensorRequirements must not be NULL");
        }
        if (rangeDopplerMapTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to Range-Doppler map NVCVTensorRequirements must not be NULL");
        }
        if (detectionListTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to detection list NVCVTensorRequirements must not be NULL");
        }
        if (snapshotsTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to snapshots NVCVTensorRequirements must not be NULL");
        }

        // Validate data types
        bool validDataTypes = (foldedDetectionListTensorRequirements->dtype == NVCV_DATA_TYPE_S32) &&
                              (ddmDopplerOffsetsTensorRequirements->dtype == NVCV_DATA_TYPE_F32) &&
                              (nciRxTensorRequirements->dtype == NVCV_DATA_TYPE_U32) &&
                              (rangeDopplerMapTensorRequirements->dtype == NVCV_DATA_TYPE_2S32) &&
                              (detectionListTensorRequirements->dtype == NVCV_DATA_TYPE_S32) &&
                              (snapshotsTensorRequirements->dtype == NVCV_DATA_TYPE_2S32);

        // Validate calibration weights data type if provided (optional - can be NULL to use DDMA weights)
        if (calibrationWeightsTensorRequirements != nullptr)
        {
            validDataTypes = validDataTypes && (calibrationWeightsTensorRequirements->dtype == NVCV_DATA_TYPE_2S32);
        }

        if (validDataTypes)
        {
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::SnapshotExtraction<int32_t>(
                foldedDetectionListTensorRequirements, ddmDopplerOffsetsTensorRequirements, nciRxTensorRequirements,
                rangeDopplerMapTensorRequirements, detectionListTensorRequirements, snapshotsTensorRequirements,
                calibrationWeightsTensorRequirements, params));
        }
        else
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Unsupported tensor data types for SnapshotExtraction operator");
        }
    });
}

/**
 * @brief Submit implementation for SnapshotExtraction operator
 *
 * This template function handles submission for both cuPVA and CUDA streams.
 */
template<typename StreamType>
NVCVStatus SnapshotExtractionSubmitImpl(NVCVOperatorHandle handle, StreamType stream,
                                        NVCVTensorHandle const detectionCount,
                                        NVCVTensorHandle const foldedDetectionList,
                                        NVCVTensorHandle const ddmDopplerOffsets, NVCVTensorHandle const nciRx,
                                        NVCVTensorHandle const rangeDopplerMap,
                                        NVCVTensorHandle const calibrationWeights, NVCVTensorHandle detectionList,
                                        NVCVTensorHandle snapshots)
{
    return nvcv::ProtectCall([&] {
        nvcv::TensorWrapHandle detectionCountWrap(detectionCount), foldedDetectionListWrap(foldedDetectionList),
            ddmDopplerOffsetsWrap(ddmDopplerOffsets), nciRxWrap(nciRx), rangeDopplerMapWrap(rangeDopplerMap),
            calibrationWeightsWrap(calibrationWeights), detectionListWrap(detectionList), snapshotsWrap(snapshots);

        NVCVDataType detectionCountDataType, foldedDetectionListDataType, ddmDopplerOffsetsDataType, nciRxDataType,
            rangeDopplerMapDataType, detectionListDataType, snapshotsDataType;

        nvcvTensorGetDataType(detectionCount, &detectionCountDataType);
        nvcvTensorGetDataType(foldedDetectionList, &foldedDetectionListDataType);
        nvcvTensorGetDataType(ddmDopplerOffsets, &ddmDopplerOffsetsDataType);
        nvcvTensorGetDataType(nciRx, &nciRxDataType);
        nvcvTensorGetDataType(rangeDopplerMap, &rangeDopplerMapDataType);
        nvcvTensorGetDataType(detectionList, &detectionListDataType);
        nvcvTensorGetDataType(snapshots, &snapshotsDataType);

        if ((detectionCountDataType == NVCV_DATA_TYPE_S32) && (foldedDetectionListDataType == NVCV_DATA_TYPE_S32) &&
            (ddmDopplerOffsetsDataType == NVCV_DATA_TYPE_F32) && (nciRxDataType == NVCV_DATA_TYPE_U32) &&
            (rangeDopplerMapDataType == NVCV_DATA_TYPE_2S32) && (detectionListDataType == NVCV_DATA_TYPE_S32) &&
            (snapshotsDataType == NVCV_DATA_TYPE_2S32))
        {
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::SnapshotExtraction<int32_t>>(handle)(
                stream, detectionCountWrap, foldedDetectionListWrap, ddmDopplerOffsetsWrap, nciRxWrap,
                rangeDopplerMapWrap, calibrationWeightsWrap, detectionListWrap, snapshotsWrap);
        }
        else
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Unsupported tensor data types for SnapshotExtraction operator");
        }
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus SnapshotExtractionSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t,
                                                                NVCVTensorHandle const, NVCVTensorHandle const,
                                                                NVCVTensorHandle const, NVCVTensorHandle const,
                                                                NVCVTensorHandle const, NVCVTensorHandle const,
                                                                NVCVTensorHandle, NVCVTensorHandle);
template NVCVStatus SnapshotExtractionSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle const,
                                                               NVCVTensorHandle const, NVCVTensorHandle const,
                                                               NVCVTensorHandle const, NVCVTensorHandle const,
                                                               NVCVTensorHandle const, NVCVTensorHandle,
                                                               NVCVTensorHandle);

// C++ overload implementations
NVCVStatus pvaSnapshotExtractionSubmit(NVCVOperatorHandle handle, cupvaStream_t stream,
                                       NVCVTensorHandle const detectionCount,
                                       NVCVTensorHandle const foldedDetectionList,
                                       NVCVTensorHandle const ddmDopplerOffsets, NVCVTensorHandle const nciRx,
                                       NVCVTensorHandle const rangeDopplerMap,
                                       NVCVTensorHandle const calibrationWeights, NVCVTensorHandle detectionList,
                                       NVCVTensorHandle snapshots)
{
    return SnapshotExtractionSubmitImpl(handle, stream, detectionCount, foldedDetectionList, ddmDopplerOffsets, nciRx,
                                        rangeDopplerMap, calibrationWeights, detectionList, snapshots);
}

NVCVStatus pvaSnapshotExtractionSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                       NVCVTensorHandle const detectionCount,
                                       NVCVTensorHandle const foldedDetectionList,
                                       NVCVTensorHandle const ddmDopplerOffsets, NVCVTensorHandle const nciRx,
                                       NVCVTensorHandle const rangeDopplerMap,
                                       NVCVTensorHandle const calibrationWeights, NVCVTensorHandle detectionList,
                                       NVCVTensorHandle snapshots)
{
    return SnapshotExtractionSubmitImpl(handle, stream, detectionCount, foldedDetectionList, ddmDopplerOffsets, nciRx,
                                        rangeDopplerMap, calibrationWeights, detectionList, snapshots);
}
