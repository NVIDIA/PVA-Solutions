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

#include "OpRadarCFAR.hpp"

#include "device/radar_cfar_param.h"

#include <IOperator.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <nvcv/Tensor.hpp>

#include <OpRadarCFAR.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>
#include <nvcv/TensorData.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_radar_cfar_ca_dev);
CUPVA_DECLARE_EXECUTABLE(pva_radar_cfar_go_dev);
CUPVA_DECLARE_EXECUTABLE(pva_radar_cfar_so_dev);

namespace pva_solutions::priv {

std::mutex RadarCFAR::s_execMapMutex;
std::map<RadarCFAR::ExecKey, std::weak_ptr<cupva::Executable>> RadarCFAR::s_radarCFARExecutables;

// Add anonymous namespace to avoid duplicate with other operator's createExecDataMap functions.
namespace {

std::map<RadarCFAR::ExecDataKey, IOperator::ExecutableData> createExecDataMap()
{
    return {
        {{static_cast<int32_t>(PVA_CFAR_CA)},
         {CUPVA_EXECUTABLE_DATA(pva_radar_cfar_ca_dev), CUPVA_EXECUTABLE_SIZE(pva_radar_cfar_ca_dev)}},
        {{static_cast<int32_t>(PVA_CFAR_CA_GO)},
         {CUPVA_EXECUTABLE_DATA(pva_radar_cfar_go_dev), CUPVA_EXECUTABLE_SIZE(pva_radar_cfar_go_dev)}},
        {{static_cast<int32_t>(PVA_CFAR_CA_SO)},
         {CUPVA_EXECUTABLE_DATA(pva_radar_cfar_so_dev), CUPVA_EXECUTABLE_SIZE(pva_radar_cfar_so_dev)}},
    };
}

/**
 * Convert a threshold factor to Q14, matching FloatToFixedPoint in the CPU reference:
 * static_cast<int32_t>(std::round(value * DATA_SCALE)).
 */
int32_t floatToQ14(float value)
{
    return static_cast<int32_t>(std::round(value * static_cast<float>(CFAR_FIXED_POINT_SCALE)));
}

/**
 * Fill the per-position reciprocal tables for one axis.
 *
 * Mirrors computeHorizontalCounts / computeVerticalCounts in the CPU reference, then
 * stores 65536/count so the device only multiplies. In cyclic mode the counts do not
 * depend on position; in "full" mode the window is clipped to the image and the trailing
 * and leading counts are floored at one, exactly as the reference does.
 */
void fillReciprocalTables(int32_t numBins, int32_t window, int32_t guard, int32_t train, bool cyclic,
                          std::vector<int32_t> &recipTotal, std::vector<int32_t> &recipTrail,
                          std::vector<int32_t> &recipLead)
{
    recipTotal.assign(static_cast<size_t>(numBins), 0);
    recipTrail.assign(static_cast<size_t>(numBins), 0);
    recipLead.assign(static_cast<size_t>(numBins), 0);

    for (int32_t i = 0; i < numBins; i++)
    {
        int32_t total    = 0;
        int32_t trailing = 0;
        int32_t leading  = 0;

        if (cyclic)
        {
            total    = 2 * train;
            trailing = train;
            leading  = train;
        }
        else
        {
            int32_t const start      = std::max(0, i - window);
            int32_t const end        = std::min(numBins - 1, i + window);
            int32_t const guardStart = std::max(start, i - guard);
            int32_t const guardEnd   = std::min(end, i + guard);

            total    = (end - start + 1) - (guardEnd - guardStart + 1);
            trailing = guardStart - start;
            leading  = end - guardEnd;

            if (trailing == 0)
            {
                trailing = 1;
            }
            if (leading == 0)
            {
                leading = 1;
            }
        }

        // The reference divides by these counts, so a zero would be a division by zero
        // there too; clamp to one defensively rather than emitting a bogus reciprocal.
        recipTotal[static_cast<size_t>(i)] = CFAR_RECIP_ONE / std::max(1, total);
        recipTrail[static_cast<size_t>(i)] = CFAR_RECIP_ONE / std::max(1, trailing);
        recipLead[static_cast<size_t>(i)]  = CFAR_RECIP_ONE / std::max(1, leading);
    }
}

} // namespace

std::shared_ptr<cupva::Executable> RadarCFAR::GetRadarCFARExec(ExecDataKey execDataKey)
{
    static auto execDataMap = createExecDataMap();

    auto execDataIt = execDataMap.find(execDataKey);
    if (execDataIt == execDataMap.end())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported CFAR algorithm");
    }

    ExecKey execKey = std::tuple_cat(execDataKey, std::make_tuple(cupva::Context::GetCurrent()));
    return GetExecutable<ExecKey>(execDataIt->second, execKey, s_radarCFARExecutables, s_execMapMutex);
}

RadarCFAR::RadarCFAR(NVCVTensorRequirements const *const inTensorRequirements,
                     NVCVTensorRequirements const *const outDetectionListTensorRequirements,
                     PVARadarCFARType const algorithm, PvaRadarCFARParams const *const radarCFARParams)
    : m_maskDevPtr(nullptr)
    , m_tableDevPtr(nullptr)
    , m_prog{}
{
    if (inTensorRequirements == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "inputTensorRequirements must not be nullptr");
    }
    if (outDetectionListTensorRequirements == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "outDetectionListTensorRequirements must not be nullptr");
    }
    if (radarCFARParams == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "radarCFARParams must not be nullptr");
    }

    m_inTensorRequirements               = *inTensorRequirements;
    m_outDetectionListTensorRequirements = *outDetectionListTensorRequirements;

    if (algorithm != PVA_CFAR_CA && algorithm != PVA_CFAR_CA_GO && algorithm != PVA_CFAR_CA_SO)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "algorithm must be PVA_CFAR_CA, PVA_CFAR_CA_GO or PVA_CFAR_CA_SO");
    }

    if (m_inTensorRequirements.rank != 2 && m_inTensorRequirements.rank != 3)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "input tensor rank must be 2 or 3");
    }
    if (m_inTensorRequirements.rank == 3 && m_inTensorRequirements.shape[2] != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "input tensor must have a single channel");
    }
    if (m_inTensorRequirements.dtype != NVCV_DATA_TYPE_S32 && m_inTensorRequirements.dtype != NVCV_DATA_TYPE_U32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "input tensor dtype must be S32 or U32");
    }
    if (m_outDetectionListTensorRequirements.dtype != NVCV_DATA_TYPE_S32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "detection list dtype must be S32");
    }

    int32_t const height = static_cast<int32_t>(m_inTensorRequirements.shape[0]);
    int32_t const width  = static_cast<int32_t>(m_inTensorRequirements.shape[1]);

    if (width < 8 || width > CFAR_MAX_DIM || (width % 8) != 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "input width must be in [8, %d] and divisible by 8",
                              CFAR_MAX_DIM);
    }
    if (height < 8 || height > CFAR_MAX_DIM || (height % 8) != 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "input height must be in [8, %d] and divisible by 8", CFAR_MAX_DIM);
    }
    if (width * height > CFAR_MAX_PIXELS)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "width * height must not exceed %d",
                              CFAR_MAX_PIXELS);
    }

    int32_t const maxDetections = static_cast<int32_t>(m_outDetectionListTensorRequirements.shape[0]);
    if (static_cast<int32_t>(m_outDetectionListTensorRequirements.shape[1]) != 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "detection list width must be 2");
    }
    if (maxDetections < 1 || maxDetections > CFAR_MAX_DETECTIONS)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "detection list height must be in [1, %d]",
                              CFAR_MAX_DETECTIONS);
    }

    int32_t const numHorTrain = radarCFARParams->numHorTrain;
    int32_t const numHorGuard = radarCFARParams->numHorGuard;
    int32_t const numVerTrain = radarCFARParams->numVerTrain;
    int32_t const numVerGuard = radarCFARParams->numVerGuard;

    if (numHorTrain <= 0 || numVerTrain <= 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "numHorTrain and numVerTrain must be positive");
    }
    if (numHorGuard < 0 || numHorGuard >= numHorTrain)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "numHorGuard must be in [0, numHorTrain)");
    }
    if (numVerGuard < 0 || numVerGuard >= numVerTrain)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "numVerGuard must be in [0, numVerTrain)");
    }
    if (numHorTrain > (width - 1) / 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "numHorTrain must not exceed (width - 1) / 2");
    }
    if (numVerTrain > (height - 1) / 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "numVerTrain must not exceed (height - 1) / 2");
    }
    if (numHorTrain + numHorGuard > CFAR_MAX_WINDOW || numVerTrain + numVerGuard > CFAR_MAX_WINDOW)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "numTrain + numGuard must not exceed %d",
                              CFAR_MAX_WINDOW);
    }

    if (radarCFARParams->horizontalThreshold.type != PVA_CFAR_HORIZONTAL_THRESHOLD_UNIFORM &&
        radarCFARParams->horizontalThreshold.type != PVA_CFAR_HORIZONTAL_THRESHOLD_PER_ROW)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "horizontalThreshold.type is invalid");
    }
    bool const isUniform = radarCFARParams->horizontalThreshold.type == PVA_CFAR_HORIZONTAL_THRESHOLD_UNIFORM;
    if (!isUniform && radarCFARParams->horizontalThreshold.value.perRow == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "per-row horizontalThreshold array must not be NULL");
    }
    if (radarCFARParams->verticalThresholdFactor <= 0.0F)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "verticalThresholdFactor must be positive");
    }

    // Bound the Q14 factors so noise * factor stays inside the device's 48-bit lane; see
    // cfarThreshold in radar_cfar_knl.c for the derivation.
    int32_t const kMaxFactorQ14 = 1 << 28;

    std::vector<int32_t> horFactorQ14(static_cast<size_t>(height), 0);
    for (int32_t i = 0; i < height; i++)
    {
        float const factor = isUniform ? radarCFARParams->horizontalThreshold.value.uniform
                                       : radarCFARParams->horizontalThreshold.value.perRow[i];
        if (!(factor > 0.0F))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "every horizontal threshold factor must be positive");
        }
        int32_t const q14 = floatToQ14(factor);
        if (q14 <= 0 || q14 > kMaxFactorQ14)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "horizontal threshold factor is outside the supported range");
        }
        horFactorQ14[static_cast<size_t>(i)] = q14;
    }

    int32_t const verFactorQ14 = floatToQ14(radarCFARParams->verticalThresholdFactor);
    if (verFactorQ14 <= 0 || verFactorQ14 > kMaxFactorQ14)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "verticalThresholdFactor is outside the supported range");
    }

    int32_t const horWindow    = numHorTrain + numHorGuard;
    int32_t const verWindow    = numVerTrain + numVerGuard;
    int32_t const numHorStrips = (height + CFAR_LANES - 1) / CFAR_LANES;
    int32_t const numVerStrips = (width + CFAR_LANES - 1) / CFAR_LANES;

    m_inOp             = cupva::OffsetPointer<uint32_t>::Create(cupva::MemType::DRAM);
    m_detectionListOp  = cupva::OffsetPointer<int32_t>::Create(cupva::MemType::DRAM);
    m_detectionCountOp = cupva::OffsetPointer<int32_t>::Create(cupva::MemType::DRAM);

    // Horizontal decision mask, one int32 per cell. Operator-owned; freed in the dtor.
    // The width is rounded to a whole strip so the final column strip reads in bounds.
    int32_t const maskPitch = numVerStrips * CFAR_LANES;
    m_maskDevPtr = static_cast<int32_t *>(cupva::mem::Alloc(static_cast<size_t>(height) * maskPitch * sizeof(int32_t)));
    if (m_maskDevPtr == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_OUT_OF_MEMORY, "Failed to allocate the CFAR mask buffer");
    }

    m_tableDevPtr = static_cast<int32_t *>(
        cupva::mem::Alloc(static_cast<size_t>(CFAR_NUM_TABLES) * CFAR_TABLE_STRIDE * sizeof(int32_t)));
    if (m_tableDevPtr == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_OUT_OF_MEMORY, "Failed to allocate the CFAR table buffer");
    }

    std::vector<int32_t> horRecipTotal, horRecipTrail, horRecipLead;
    std::vector<int32_t> verRecipTotal, verRecipTrail, verRecipLead;
    fillReciprocalTables(width, horWindow, numHorGuard, numHorTrain, radarCFARParams->isHorizontalCyclicPadding,
                         horRecipTotal, horRecipTrail, horRecipLead);
    fillReciprocalTables(height, verWindow, numVerGuard, numVerTrain, radarCFARParams->isVerticalCyclicPadding,
                         verRecipTotal, verRecipTrail, verRecipLead);

    m_exec    = GetRadarCFARExec(std::make_tuple(static_cast<int32_t>(algorithm)));
    auto prog = CmdProgram::Create(*m_exec.get());

    CfarHost2DevParam params{};
    params.width         = width;
    params.height        = height;
    params.numHorTrain   = numHorTrain;
    params.numHorGuard   = numHorGuard;
    params.numVerTrain   = numVerTrain;
    params.numVerGuard   = numVerGuard;
    params.horWindow     = horWindow;
    params.verWindow     = verWindow;
    params.horCyclic     = radarCFARParams->isHorizontalCyclicPadding ? 1 : 0;
    params.verCyclic     = radarCFARParams->isVerticalCyclicPadding ? 1 : 0;
    params.peakGrouping  = radarCFARParams->enablePeakGrouping ? 1 : 0;
    params.numHorStrips  = numHorStrips;
    params.numVerStrips  = numVerStrips;
    params.maxDetections = maxDetections;
    params.verFactorQ14  = verFactorQ14;
    prog["cfarParam"].set(&params, 1);

    // The tables total 28 KB at the widest supported image, which exceeds the 8 KB
    // parameter-block limit, so they are staged in operator-owned DRAM and DMA'd into
    // VMEM once at kernel startup. Entries past width/height are never read.
    int32_t *tablesHost = static_cast<int32_t *>(cupva::mem::GetHostPointer(m_tableDevPtr));
    if (tablesHost == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "Failed to map the CFAR table buffer");
    }
    std::fill_n(tablesHost, static_cast<size_t>(CFAR_NUM_TABLES) * CFAR_TABLE_STRIDE, 0);

    auto stageTable = [tablesHost](int32_t index, std::vector<int32_t> const &table) {
        std::copy(table.begin(), table.end(), tablesHost + static_cast<size_t>(index) * CFAR_TABLE_STRIDE);
    };

    stageTable(CFAR_TBL_HOR_TOTAL, horRecipTotal);
    stageTable(CFAR_TBL_HOR_TRAIL, horRecipTrail);
    stageTable(CFAR_TBL_HOR_LEAD, horRecipLead);
    stageTable(CFAR_TBL_VER_TOTAL, verRecipTotal);
    stageTable(CFAR_TBL_VER_TRAIL, verRecipTrail);
    stageTable(CFAR_TBL_VER_LEAD, verRecipLead);
    stageTable(CFAR_TBL_HOR_FACTOR, horFactorQ14);

    // Each DataFlowPhase holds exactly one open/close group, matching image_processing/ccl:
    // the runtime rejects a phase that contains two sequential groups. Phase 0 loads the
    // tables, phase 1 is the horizontal pass, phase 2 the vertical pass plus the list drain,
    // phase 3 the detection count.
    prog.addDataFlowHead<SequenceDataFlow>(0)
        .handler(prog["tableHdl"])
        .addTransfer()
        .src(m_tableDevPtr, CFAR_NUM_TABLES * CFAR_TABLE_STRIDE * sizeof(int32_t))
        .dst(prog["tableBuf"].ptr<int32_t>(), CFAR_NUM_TABLES * CFAR_TABLE_STRIDE * sizeof(int32_t))
        .tile(CFAR_NUM_TABLES * CFAR_TABLE_STRIDE * sizeof(int32_t), 1);

    int32_t const inLinePitch = static_cast<int32_t>(m_inTensorRequirements.strides[0] / sizeof(uint32_t));

    // ---- Phase 0: horizontal pass ----
    // Row strips spanning the full image width. Padding is materialised on device: RDF
    // has no cyclic pad mode.
    prog.addDataFlowHead<RasterDataFlow>(1)
        .handler(prog["hInHdl"])
        .src(m_inOp, width, height, inLinePitch)
        .tileBuffer(prog["inBuffer"].ptr<uint32_t>())
        .tile(width, CFAR_LANES);

    prog.addDataFlowHead<RasterDataFlow>(1)
        .handler(prog["hMaskOutHdl"])
        .dst(m_maskDevPtr, width, height, maskPitch)
        .tileBuffer(prog["maskBuffer"].ptr<int32_t>())
        .transpose(TRANS_MODE_1)
        .tile(width, CFAR_LANES);

    // ---- Phase 1: vertical pass and emission ----
    // Column strips spanning the full height. Width is a multiple of 8 but not
    // necessarily of CFAR_LANES, so the final strip is partial; lanes past the edge are
    // masked off the detection on device.
    prog.addDataFlowHead<RasterDataFlow>(2)
        .handler(prog["vInHdl"])
        .src(m_inOp, width, height, inLinePitch)
        .tileBuffer(prog["inBuffer"].ptr<uint32_t>())
        .tile(CFAR_LANES, height);

    prog.addDataFlowHead<RasterDataFlow>(2)
        .handler(prog["vMaskInHdl"])
        .src(m_maskDevPtr, maskPitch, height, maskPitch)
        .tileBuffer(prog["maskBuffer"].ptr<int32_t>())
        .tile(CFAR_LANES, height);

    // The detection list is drained at device-computed offsets, so the kernel needs the
    // raw base pointer in addition to the dataflow. The tile size is compiled at the
    // maximum the kernel can ever drain and shrunk at runtime by cupvaSQDFUpdateTileSize:
    // a compiled tile may shrink, never grow.
    //
    // The maximum is the staging buffer capacity or the caller's detection list, whichever
    // is smaller. The kernel already refuses to drain past maxDetections, so the larger
    // value would never be used - but it is still compiled into the descriptor, and a
    // transfer of CFAR_STAGE_CAPACITY rows at a two-int destination pitch reaches past the
    // end of a shorter list. cuPVA 2.5 validates that at submit and fails the command; later
    // runtimes do not range check it, so the overrun is silent there.
    int32_t const listTileRows = (maxDetections < CFAR_STAGE_CAPACITY) ? maxDetections : CFAR_STAGE_CAPACITY;

    SequenceDataFlow &listDataFlow = prog.addDataFlowHead<SequenceDataFlow>(2).handler(prog["listHdl"]);
    listDataFlow.addTransfer()
        .src(prog["stageY"].ptr<int32_t>(), sizeof(int32_t))
        .dst(m_detectionListOp, 2 * sizeof(int32_t))
        .tile(sizeof(int32_t), listTileRows);
    listDataFlow.addTransfer()
        .src(prog["stageX"].ptr<int32_t>(), sizeof(int32_t))
        .dst(m_detectionListOp, 2 * sizeof(int32_t))
        .tile(sizeof(int32_t), listTileRows);

    prog.addDataFlowHead<SequenceDataFlow>(3)
        .handler(prog["countHdl"])
        .addTransfer()
        .src(prog["detectionCountBuf"].ptr<int32_t>(), sizeof(int32_t))
        .dst(m_detectionCountOp, sizeof(int32_t))
        .tile(sizeof(int32_t), 1);

    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename S>
void RadarCFAR::operator()(S stream, nvcv::Tensor const &input, nvcv::Tensor const &outDetectionList,
                           nvcv::Tensor const &outDetectionCount)
{
    auto inTensorData    = input.exportData<nvcv::TensorDataStrided>();
    auto listTensorData  = outDetectionList.exportData<nvcv::TensorDataStrided>();
    auto countTensorData = outDetectionCount.exportData<nvcv::TensorDataStrided>();

    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_inTensorRequirements, inTensorData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_outDetectionListTensorRequirements,
                                                          listTensorData->cdata()))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Input and output tensors must have the same requirements that were used to create the operator");
    }
    if ((static_cast<void *>(inTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(listTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(countTensorData->basePtr()) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor base pointers must not be NULL");
    }

    m_inOp.setBase(static_cast<uint32_t *>(static_cast<void *>(inTensorData->basePtr())));
    m_detectionListOp.setBase(static_cast<int32_t *>(static_cast<void *>(listTensorData->basePtr())));
    m_detectionCountOp.setBase(static_cast<int32_t *>(static_cast<void *>(countTensorData->basePtr())));

    {
        cupvaParameter_t listExtMemPtr;
        if (CupvaCmdProgramGetParameter(&m_prog, &listExtMemPtr, "listExtMemPtr") != CUPVA_ERROR_NONE)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "CupvaCmdProgramGetParameter failed");
        }
        if (CupvaParameterSetValuePointer(&listExtMemPtr, static_cast<void *>(listTensorData->basePtr())) !=
            CUPVA_ERROR_NONE)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "CupvaParameterSetValuePointer failed");
        }
    }

    if (CupvaCmdProgramUpdateDataFlows(&m_prog) != CUPVA_ERROR_NONE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Update DataFlow failed");
    }

    cupvaCmd_t const *submitCmds[1] = {&m_prog};
    cupvaCmdStatus_t cmdStatus[1]   = {nullptr};

    if constexpr (std::is_same_v<S, cupvaStream_t>)
    {
        if (CupvaStreamSubmit(stream, submitCmds, cmdStatus, 1, CUPVA_IN_ORDER, -1, -1) != CUPVA_ERROR_NONE)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Submit failed");
        }
    }
    else if constexpr (std::is_same_v<S, cudaStream_t>)
    {
        if constexpr (kCupvaHasCudaStreamSupport)
        {
            if (CupvaCudaStreamSubmit(stream, submitCmds, cmdStatus, 1, CUPVA_IN_ORDER, -1, -1) != CUPVA_ERROR_NONE)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Submit failed");
            }
        }
        else
        {
            ThrowCudaStreamNotSupported();
        }
    }
    else
    {
        static_assert(!sizeof(S), "Unsupported stream type for RadarCFAR operator");
    }
}

RadarCFAR::~RadarCFAR()
{
    CupvaCmdDestroy(&m_prog);
    if (m_maskDevPtr != nullptr)
    {
        cupva::mem::Free(m_maskDevPtr);
        m_maskDevPtr = nullptr;
    }
    if (m_tableDevPtr != nullptr)
    {
        cupva::mem::Free(m_tableDevPtr);
        m_tableDevPtr = nullptr;
    }
}

} // namespace pva_solutions::priv

NVCVStatus pvaRadarCFARCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements const *const inputTensorRequirements,
                              NVCVTensorRequirements const *const outDetectionListTensorRequirements,
                              PVARadarCFARType const algorithm, PvaRadarCFARParams const *const radarCFARParams)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::RadarCFAR(
            inputTensorRequirements, outDetectionListTensorRequirements, algorithm, radarCFARParams));
    });
}

template<typename StreamType>
NVCVStatus RadarCFARSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle const inputTensor,
                               NVCVTensorHandle outDetectionListTensor, NVCVTensorHandle outDetectionCountTensor)
{
    return nvcv::ProtectCall([&] {
        if (inputTensor == nullptr || outDetectionListTensor == nullptr || outDetectionCountTensor == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid tensor handles provided");
        }

        nvcv::TensorWrapHandle input(inputTensor), detectionList(outDetectionListTensor),
            detectionCount(outDetectionCountTensor);

        pva_solutions::priv::ToDynamicRef<pva_solutions::priv::RadarCFAR>(handle)(stream, input, detectionList,
                                                                                  detectionCount);
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus RadarCFARSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle const,
                                                       NVCVTensorHandle, NVCVTensorHandle);
template NVCVStatus RadarCFARSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle const,
                                                      NVCVTensorHandle, NVCVTensorHandle);

// C++ overload implementations
NVCVStatus pvaRadarCFARSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle const inputTensor,
                              NVCVTensorHandle outDetectionListTensor, NVCVTensorHandle outDetectionCountTensor)
{
    return RadarCFARSubmitImpl(handle, stream, inputTensor, outDetectionListTensor, outDetectionCountTensor);
}

NVCVStatus pvaRadarCFARSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle const inputTensor,
                              NVCVTensorHandle outDetectionListTensor, NVCVTensorHandle outDetectionCountTensor)
{
    return RadarCFARSubmitImpl(handle, stream, inputTensor, outDetectionListTensor, outDetectionCountTensor);
}
