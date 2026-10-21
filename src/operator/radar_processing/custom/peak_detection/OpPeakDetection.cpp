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

#include "OpPeakDetection.hpp"

#include "device/peak_detection_param.h"

#include <TensorUtils.hpp>
#include <cupva_host_conversions.hpp>
#include <nvcv/Tensor.hpp>

#include <OpPeakDetection.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>
#include <nvcv/TensorData.h>
#include <nvcv/TensorShape.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <utility>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_peak_detection_nciw512_dev);
CUPVA_DECLARE_EXECUTABLE(pva_peak_detection_nciw1024_dev);

namespace pva_solutions::priv {

std::mutex PeakDetection::s_execMapMutex;

std::map<PeakDetection::ExecKey, std::weak_ptr<cupva::Executable>> PeakDetection::s_peakDetectionExecutables;

// Add anonymous namespace to avoid duplicate with other operator's createExecDataMap functions.
namespace {
std::map<PeakDetection::ExecDataKey, IOperator::ExecutableData> createExecDataMap()
{
    return {
        {{512},
         {CUPVA_EXECUTABLE_DATA(pva_peak_detection_nciw512_dev),
          CUPVA_EXECUTABLE_SIZE(pva_peak_detection_nciw512_dev)}},
        {{1024},
         {CUPVA_EXECUTABLE_DATA(pva_peak_detection_nciw1024_dev),
          CUPVA_EXECUTABLE_SIZE(pva_peak_detection_nciw1024_dev)}},
    };
}
} // namespace

std::shared_ptr<cupva::Executable> PeakDetection::GetPeakDetectionExec(ExecDataKey execDataKey)
{
    static auto execDataMap = createExecDataMap();

    auto execDataIt = execDataMap.find(execDataKey);
    if (execDataIt != execDataMap.end())
    {
        // Create execKey by combining execDataKey and context
        ExecKey execKey = std::tuple_cat(execDataKey, std::make_tuple(cupva::Context::GetCurrent()));
        return GetExecutable(execDataIt->second, execKey, s_peakDetectionExecutables, s_execMapMutex);
    }
    else
    {
        throw std::invalid_argument("Unsupported parameters");
    }
}

PeakDetection::PeakDetection(NVCVTensorRequirements const *const inTensorRequirements,
                             NVCVTensorRequirements const *const outTensorRequirements, int32_t NofTx,
                             int32_t repeatFold)
    : m_FT2DTensorRequirements(*inTensorRequirements)
    , m_snapTensorRequirements(*outTensorRequirements)
    , m_nciFinalOp(cupva::OffsetPointer<uint32_t>::Create(cupva::MemType::DRAM))
    , m_nciRxOp(cupva::OffsetPointer<uint32_t>::Create(cupva::MemType::DRAM))
    , m_FT2DOp(cupva::OffsetPointer<int32_t>::Create(cupva::MemType::DRAM))
    , m_noiseEstOp(cupva::OffsetPointer<uint32_t>::Create(cupva::MemType::DRAM))
    , m_peakCountOp(cupva::OffsetPointer<uint32_t>::Create(cupva::MemType::DRAM))
    , m_rangeIdxOp(cupva::OffsetPointer<uint32_t>::Create(cupva::MemType::DRAM))
    , m_dopplerFoldIdxOp(cupva::OffsetPointer<uint32_t>::Create(cupva::MemType::DRAM))
    , m_dopplerIdxOp(cupva::OffsetPointer<uint32_t>::Create(cupva::MemType::DRAM))
    , m_snapOp(cupva::OffsetPointer<int32_t>::Create(cupva::MemType::DRAM))
    , m_prog{}
{
    if (inTensorRequirements == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "inTensorRequirements must not be nullptr");
    }

    if (inTensorRequirements->rank != 3)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "inTensorRequirements->rank must be 3");
    }

    if (inTensorRequirements->dtype != NVCV_DATA_TYPE_2S32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "inTensorRequirements->dtype must be 2S32");
    }

    if (outTensorRequirements == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "outTensorRequirements must not be nullptr");
    }

    if (outTensorRequirements->rank != 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "outTensorRequirements->rank must be 2");
    }

    if (outTensorRequirements->dtype != NVCV_DATA_TYPE_2S32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "outTensorRequirements->dtype must be 2S32");
    }

    if (NofTx != PEAKDET_SUPPORTED_TX_COUNT)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "NofTx must be %d", PEAKDET_SUPPORTED_TX_COUNT);
    }

    if (repeatFold != PEAKDET_SUPPORTED_REPEAT_FOLD)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "repeatFold must be %d",
                              PEAKDET_SUPPORTED_REPEAT_FOLD);
    }

    const int32_t numRangeBin{static_cast<int32_t>(inTensorRequirements->shape[0])};
    const int32_t NofRx{static_cast<int32_t>(inTensorRequirements->shape[1])};
    const int32_t numDopplerBin{static_cast<int32_t>(inTensorRequirements->shape[2])};

    if (numRangeBin < TILE_Y || numRangeBin % TILE_Y != 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Number of Doppler bins must be greater than or equal to TILE_Y and divisible by TILE_Y");
    }

    if ((numDopplerBin != 512 && numDopplerBin != 1024) || NofRx != 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Number of ramps must be 512 or 1024, and number of Rx must be 4");
    }

    {
        NVCVTensorLayout layoutHW;
        nvcvTensorLayoutMake("HW", &layoutHW);
        int64_t nciRxTensorShape[] = {numRangeBin, numDopplerBin};
        nvcvTensorCalcRequirementsPva(sizeof(nciRxTensorShape) / sizeof(nciRxTensorShape[0]), nciRxTensorShape,
                                      NVCV_DATA_TYPE_U32, layoutHW, 0, 0, &m_nciRxTensorRequirements);
    }

    {
        NVCVTensorLayout layoutHW;
        nvcvTensorLayoutMake("HW", &layoutHW);
        int64_t nciFinalTensorShape[] = {numRangeBin, numDopplerBin / repeatFold};
        nvcvTensorCalcRequirementsPva(sizeof(nciFinalTensorShape) / sizeof(nciFinalTensorShape[0]), nciFinalTensorShape,
                                      NVCV_DATA_TYPE_U32, layoutHW, 0, 0, &m_nciFinalTensorRequirements);
    }

    {
        NVCVTensorLayout layout;
        nvcvTensorLayoutMake("W", &layout);
        int64_t noiseEstTensorShape[] = {numRangeBin};
        nvcvTensorCalcRequirementsPva(sizeof(noiseEstTensorShape) / sizeof(noiseEstTensorShape[0]), noiseEstTensorShape,
                                      NVCV_DATA_TYPE_U32, layout, 0, 0, &m_noiseEstTensorRequirements);
    }

    {
        NVCVTensorLayout layout;
        nvcvTensorLayoutMake("W", &layout);
        int64_t peakCountTensorShape[] = {1};
        nvcvTensorCalcRequirementsPva(sizeof(peakCountTensorShape) / sizeof(peakCountTensorShape[0]),
                                      peakCountTensorShape, NVCV_DATA_TYPE_U32, layout, 0, 0,
                                      &m_peakCountTensorRequirements);
    }

    {
        NVCVTensorLayout layout;
        nvcvTensorLayoutMake("HW", &layout);
        int64_t peakIndexTensorShape[] = {3, PEAKDET_MAX_PEAK_COUNT};
        nvcvTensorCalcRequirementsPva(sizeof(peakIndexTensorShape) / sizeof(peakIndexTensorShape[0]),
                                      peakIndexTensorShape, NVCV_DATA_TYPE_U32, layout, 0, 0,
                                      &m_peakIndexTensorRequirements);
    }

    const int32_t niterY = numRangeBin / TILE_Y;
    m_exec               = GetPeakDetectionExec(std::make_tuple(numDopplerBin));
    auto prog            = CmdProgram::Create(*m_exec.get());

    const int32_t numIntPerCmplx = 2;
    std::vector<int32_t> tbl1024{
        0, 1 * numDopplerBin * numIntPerCmplx, 2 * numDopplerBin * numIntPerCmplx, 3 * numDopplerBin * numIntPerCmplx,
        0, 1 * numDopplerBin * numIntPerCmplx, 2 * numDopplerBin * numIntPerCmplx, 3 * numDopplerBin * numIntPerCmplx,
        0, 1 * numDopplerBin * numIntPerCmplx, 2 * numDopplerBin * numIntPerCmplx, 3 * numDopplerBin * numIntPerCmplx,
        0, 1 * numDopplerBin * numIntPerCmplx, 2 * numDopplerBin * numIntPerCmplx, 3 * numDopplerBin * numIntPerCmplx};

    const std::vector<int32_t> ddmSpace{0, 7, 6, 4};
    std::vector<int32_t> ddmSpaceTbl{
        ddmSpace[0] * numDopplerBin / repeatFold, ddmSpace[0] * numDopplerBin / repeatFold,
        ddmSpace[0] * numDopplerBin / repeatFold, ddmSpace[0] * numDopplerBin / repeatFold,
        ddmSpace[1] * numDopplerBin / repeatFold, ddmSpace[1] * numDopplerBin / repeatFold,
        ddmSpace[1] * numDopplerBin / repeatFold, ddmSpace[1] * numDopplerBin / repeatFold,
        ddmSpace[2] * numDopplerBin / repeatFold, ddmSpace[2] * numDopplerBin / repeatFold,
        ddmSpace[2] * numDopplerBin / repeatFold, ddmSpace[2] * numDopplerBin / repeatFold,
        ddmSpace[3] * numDopplerBin / repeatFold, ddmSpace[3] * numDopplerBin / repeatFold,
        ddmSpace[3] * numDopplerBin / repeatFold, ddmSpace[3] * numDopplerBin / repeatFold};

    uint32_t ddmSpaceBitMask = 0;
    {
        std::vector<uint32_t> bitMaskVec(repeatFold * 2, 1);
        for (int32_t i = 0; i < ddmSpace.size(); i++)
        {
            bitMaskVec.at(ddmSpace[i]) = 0;
        }

        // Copy the first half of bitMaskVec to its second half
        std::copy(bitMaskVec.begin(), bitMaskVec.begin() + repeatFold, bitMaskVec.begin() + repeatFold);

        for (int32_t i = 0; i < 2 * repeatFold - 1; i++)
        {
            ddmSpaceBitMask |= bitMaskVec[1 + i] << i; // start from the second element
        }
    }

    int8_t vpu_param_a[sizeof(PeakDetectionHost2DevParam)];
    PeakDetectionHost2DevParam *vpuParam = (PeakDetectionHost2DevParam *)vpu_param_a;
    vpuParam->niterY                     = niterY;
    std::copy(tbl1024.begin(), tbl1024.end(), vpuParam->tbl1024);
    std::copy(ddmSpaceTbl.begin(), ddmSpaceTbl.end(), vpuParam->ddmSpaceTbl);
    vpuParam->ddmSpaceBitMask = ddmSpaceBitMask;
    prog["vpuParam"].set(vpu_param_a, sizeof(PeakDetectionHost2DevParam));

    auto nciFinalHdl      = prog["nciFinalHdl"];
    auto noiseEstHdl      = prog["noiseEstHdl"];
    auto nciRxHdl         = prog["nciRxHdl"];
    auto fT2DHdl          = prog["fT2DHdl"];
    uint32_t *nciFinalPtr = (uint32_t *)prog["nciFinalBuf"].ptr<uint32_t>();
    uint32_t *noiseEstPtr = (uint32_t *)prog["noiseEstBuf"].ptr<uint32_t>();
    uint32_t *nciRxPtr    = (uint32_t *)prog["nciRxTileBuf"].ptr<uint32_t>();

    const int32_t nciFinalWidth      = numDopplerBin / repeatFold;
    RasterDataFlow &nciFinalDataFlow = prog.addDataFlowHead<RasterDataFlow>();
    nciFinalDataFlow.handler(nciFinalHdl)
        .tile(nciFinalWidth, TILE_Y)
        .src(m_nciFinalOp, nciFinalWidth, numRangeBin, nciFinalWidth)
        .tileBuffer(nciFinalPtr)
        .halo(1, PAD_CONST, std::numeric_limits<uint32_t>::max());

    RasterDataFlow &noiseEstDataFlow = prog.addDataFlowHead<RasterDataFlow>();
    noiseEstDataFlow.handler(noiseEstHdl).tile(1, TILE_Y).src(m_noiseEstOp, 1, numRangeBin, 1).tileBuffer(noiseEstPtr);

    const int32_t nciRxWidth = numDopplerBin;
    prog.addDataFlowHead<SequenceDataFlow>()
        .handler(nciRxHdl)
        .addTransfer()
        .src(m_nciRxOp, nciRxWidth * sizeof(uint32_t))
        .dst(nciRxPtr, nciRxWidth * sizeof(uint32_t))
        .tile(nciRxWidth * sizeof(uint32_t), 1);

    // nciFinal is viewed as numRangeBin*(NofRx*numDopplerBin*std::complex<int32_t>) for GSDF use.
    // Each tile is of size 2*1 to represent a std::complex<int32_t>.
    const int32_t nciRxFinalWidth = NofRx * numDopplerBin * numIntPerCmplx;
    prog.addDataFlowHead<GatherScatterDataFlow>()
        .handler(fT2DHdl)
        .numTilesPerTrigger(NofTx * NofRx)
        .tile(numIntPerCmplx, 1)
        .src(m_FT2DOp, nciRxFinalWidth, numRangeBin, nciRxFinalWidth);

    auto peakCountHdl             = prog["peakCountHdl"];
    int32_t *peakCountVem2DramPtr = (int32_t *)prog["peakCountVem2Dram"].ptr<int32_t>();
    prog.addDataFlowHead<SequenceDataFlow>()
        .handler(peakCountHdl)
        .addTransfer()
        .src(peakCountVem2DramPtr, sizeof(int32_t))
        .dst(m_peakCountOp, sizeof(int32_t))
        .tile(sizeof(int32_t), 1);

    auto idxSnapHdl             = prog["idxSnapHdl"];
    uint32_t *rangeIdxPtr       = (uint32_t *)prog["rangeIdxExport"].ptr<uint32_t>();
    uint32_t *dopplerFoldIdxPtr = (uint32_t *)prog["dopplerFoldIdxExport"].ptr<uint32_t>();
    uint32_t *dopplerIdxPtr     = (uint32_t *)prog["dopplerIdxExport"].ptr<uint32_t>();
    int32_t *snapPtr            = (int32_t *)prog["snapExport"].ptr<int32_t>();

    const int32_t outputTileHint      = 100;
    SequenceDataFlow &idxSnapDataFlow = prog.addDataFlowHead<SequenceDataFlow>().handler(idxSnapHdl);
    idxSnapDataFlow.addTransfer()
        .src(rangeIdxPtr, outputTileHint * sizeof(uint32_t))
        .dst(m_rangeIdxOp, outputTileHint * sizeof(uint32_t))
        .tile(outputTileHint * sizeof(uint32_t), 1);

    idxSnapDataFlow.addTransfer()
        .src(dopplerFoldIdxPtr, outputTileHint * sizeof(uint32_t))
        .dst(m_dopplerFoldIdxOp, outputTileHint * sizeof(uint32_t))
        .tile(outputTileHint * sizeof(uint32_t), 1);

    idxSnapDataFlow.addTransfer()
        .src(dopplerIdxPtr, outputTileHint * sizeof(uint32_t))
        .dst(m_dopplerIdxOp, outputTileHint * sizeof(uint32_t))
        .tile(outputTileHint * sizeof(uint32_t), 1);

    // Each peak's snapshot is NofTx*NofRx*numIntPerCmplx contiguous int32, so declare
    // it as one wide row rather than that many 4-byte rows. A 4-byte tile width makes
    // the DMA issue one transaction per element.
    const int32_t snapBytes = static_cast<int32_t>(NofTx * NofRx * numIntPerCmplx * sizeof(int32_t));
    idxSnapDataFlow.addTransfer().src(snapPtr, snapBytes).dst(m_snapOp, snapBytes).tile(snapBytes, outputTileHint);

    prog.compileDataFlows();

    cupva::ConvertTo(std::move(prog), m_prog);
}

PeakDetection::~PeakDetection()
{
    CupvaCmdDestroy(&m_prog);
}

template<typename S>
void PeakDetection::operator()(S stream, nvcv::Tensor const &FT2D, nvcv::Tensor const &nciRx,
                               nvcv::Tensor const &nciFinal, nvcv::Tensor const &noiseEst,
                               nvcv::Tensor const &peakCount, nvcv::Tensor const &peakIndex, nvcv::Tensor const &snap)
{
    auto FT2DTensorData      = FT2D.exportData<nvcv::TensorDataStrided>();
    auto nciRxTensorData     = nciRx.exportData<nvcv::TensorDataStrided>();
    auto nciFinalTensorData  = nciFinal.exportData<nvcv::TensorDataStrided>();
    auto noiseEstTensorData  = noiseEst.exportData<nvcv::TensorDataStrided>();
    auto peakCountTensorData = peakCount.exportData<nvcv::TensorDataStrided>();
    auto peakIndexTensorData = peakIndex.exportData<nvcv::TensorDataStrided>();
    auto snapTensorData      = snap.exportData<nvcv::TensorDataStrided>();

    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_FT2DTensorRequirements, FT2DTensorData->cdata()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "FT2D must have the same requirements that was used to create the operator");
    }

    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_nciRxTensorRequirements, nciRxTensorData->cdata()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "nciRx must have the same requirements that was used to create the operator");
    }

    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_nciFinalTensorRequirements, nciFinalTensorData->cdata()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "nciFinal must have the same requirements that was used to create the operator");
    }

    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_noiseEstTensorRequirements, noiseEstTensorData->cdata()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "noiseEst must have the same requirements that was used to create the operator");
    }

    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_peakCountTensorRequirements, peakCountTensorData->cdata()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "peakCount must have the same requirements that was used to create the operator");
    }

    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_peakIndexTensorRequirements, peakIndexTensorData->cdata()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "peakIndex must have the same requirements that was used to create the operator");
    }

    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_snapTensorRequirements, snapTensorData->cdata()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "snap must have the same requirements that was used to create the operator");
    }

    if ((static_cast<void *>(FT2DTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(nciRxTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(nciFinalTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(noiseEstTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(peakCountTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(peakIndexTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(snapTensorData->basePtr()) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor base pointers must not be nullptr");
    }

    m_FT2DOp.setBase(reinterpret_cast<int32_t *>(FT2DTensorData->basePtr()));
    m_nciFinalOp.setBase(reinterpret_cast<uint32_t *>(nciFinalTensorData->basePtr()));
    m_nciRxOp.setBase(reinterpret_cast<uint32_t *>(nciRxTensorData->basePtr()));
    m_noiseEstOp.setBase(reinterpret_cast<uint32_t *>(noiseEstTensorData->basePtr()));

    m_peakCountOp.setBase(reinterpret_cast<uint32_t *>(peakCountTensorData->basePtr()));
    m_rangeIdxOp.setBase(reinterpret_cast<uint32_t *>(peakIndexTensorData->basePtr()));
    m_dopplerFoldIdxOp.setBase(reinterpret_cast<uint32_t *>(peakIndexTensorData->basePtr()) + PEAKDET_MAX_PEAK_COUNT);
    m_dopplerIdxOp.setBase(reinterpret_cast<uint32_t *>(peakIndexTensorData->basePtr()) + 2 * PEAKDET_MAX_PEAK_COUNT);
    m_snapOp.setBase(reinterpret_cast<int32_t *>(snapTensorData->basePtr()));

    // Set external memory pointers for SQDF.
    {
        cupvaParameter_t nciRxBaseExtMemPtr;
        cupvaError_t err = CupvaCmdProgramGetParameter(&m_prog, &nciRxBaseExtMemPtr, "nciRxBaseExtMemPtr");
        if (err != CUPVA_ERROR_NONE)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "CupvaCmdProgramGetParameter failed");
        }
        err = CupvaParameterSetValuePointer(&nciRxBaseExtMemPtr, static_cast<void *>(nciRxTensorData->basePtr()));
        if (err != CUPVA_ERROR_NONE)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "CupvaParameterSetValuePointer failed");
        }
    }

    {
        cupvaParameter_t param;
        if (CupvaCmdProgramGetParameter(&m_prog, &param, "peakIndexExtMemPtr") != CUPVA_ERROR_NONE)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "CupvaCmdProgramGetParameter failed");
        }
        if (CupvaParameterSetValuePointer(&param, static_cast<void *>(peakIndexTensorData->basePtr())) !=
            CUPVA_ERROR_NONE)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "CupvaParameterSetValuePointer failed");
        }
    }

    {
        cupvaParameter_t param;
        if (CupvaCmdProgramGetParameter(&m_prog, &param, "snapExtMemPtr") != CUPVA_ERROR_NONE)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "CupvaCmdProgramGetParameter failed");
        }
        if (CupvaParameterSetValuePointer(&param, static_cast<void *>(snapTensorData->basePtr())) != CUPVA_ERROR_NONE)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "CupvaParameterSetValuePointer failed");
        }
    }

    cupvaError_t err = CupvaCmdProgramUpdateDataFlows(&m_prog);
    if (err != CUPVA_ERROR_NONE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "CupvaCmdProgramUpdateDataFlows failed");
    }

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
        static_assert(!sizeof(S), "Unsupported stream type for PeakDetection operator");
    }

    if (!(err == CUPVA_ERROR_NONE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Submit failed");
    }
}

} // namespace pva_solutions::priv

NVCVStatus pvaPeakDetectionCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements const *const inTensorRequirements,
                                  NVCVTensorRequirements const *const outTensorRequirements, int32_t NofTx,
                                  int32_t repeatFold)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        *handle = reinterpret_cast<NVCVOperatorHandle>(
            new pva_solutions::priv::PeakDetection(inTensorRequirements, outTensorRequirements, NofTx, repeatFold));
    });
}

template<typename StreamType>
NVCVStatus PeakDetectionSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle *inHandles,
                                   NVCVTensorHandle *outHandles, int inTensorCount, int outTensorCount)
{
    return nvcv::ProtectCall([&] {
        if (inTensorCount != 4 || outTensorCount != 3)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "inTensorCount must be 4 and outTensorCount must be 3");
        }

        nvcv::TensorWrapHandle FT2D(inHandles[0]), nciRx(inHandles[1]), nciFinal(inHandles[2]), noiseEst(inHandles[3]);
        nvcv::TensorWrapHandle peakCount(outHandles[0]), peakIndex(outHandles[1]), snap(outHandles[2]);

        pva_solutions::priv::ToDynamicRef<pva_solutions::priv::PeakDetection>(handle)(
            stream, FT2D, nciRx, nciFinal, noiseEst, peakCount, peakIndex, snap);
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus PeakDetectionSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle *,
                                                           NVCVTensorHandle *, int, int);
template NVCVStatus PeakDetectionSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle *,
                                                          NVCVTensorHandle *, int, int);

// C++ overload implementations
NVCVStatus pvaPeakDetectionSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle *inHandles,
                                  NVCVTensorHandle *outHandles, int inTensorCount, int outTensorCount)
{
    return PeakDetectionSubmitImpl(handle, stream, inHandles, outHandles, inTensorCount, outTensorCount);
}

NVCVStatus pvaPeakDetectionSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle *inHandles,
                                  NVCVTensorHandle *outHandles, int inTensorCount, int outTensorCount)
{
    return PeakDetectionSubmitImpl(handle, stream, inHandles, outHandles, inTensorCount, outTensorCount);
}