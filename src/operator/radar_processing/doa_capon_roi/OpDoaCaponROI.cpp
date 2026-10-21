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

#include "OpDoaCaponROI.hpp"

#include "OpDoaCaponROICommon.hpp"

#include "device/capon_roi_vmem_layout.h"

#include <TensorUtils.hpp>
#include <cupva_host_conversions.hpp>

#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_nonsafety.h>
#include <cupva_platform.h>
#include <cupva_types.h>

#include <algorithm>
#include <cstdint>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_capon_roi_combined_s32_interp_dev);
CUPVA_DECLARE_EXECUTABLE(pva_capon_roi_combined_s32_no_interp_dev);
CUPVA_DECLARE_EXECUTABLE(pva_capon_roi_combined_fp_interp_dev);
CUPVA_DECLARE_EXECUTABLE(pva_capon_roi_combined_fp_no_interp_dev);

namespace pva_solutions::priv {

namespace {

PVADoaCaponROIParams ValidateCombinedRequirements(const NVCVTensorRequirements *snapshotsReq,
                                                  const NVCVTensorRequirements *steeringReq,
                                                  const NVCVTensorRequirements *peakPowerReq,
                                                  const NVCVTensorRequirements *azBinReq,
                                                  const NVCVTensorRequirements *elBinReq,
                                                  const PVADoaCaponROIParams *params)
{
    PVADoaCaponROIParams normalized = ValidateCommonParams(params);
    int32_t const numDopplerBins    = normalized.numDopplerBinsCovMat;
    int32_t const numAzBins         = normalized.numAzBins;
    int32_t const numElBins         = normalized.numElBins;
    int32_t const svDim             = normalized.svDim;
    int32_t const numPeaks          = normalized.numPeaksPerROI;

    if (snapshotsReq->shape[0] <= 0 || snapshotsReq->shape[0] > PVA_DOA_CAPON_ROI_MAX_ROIS)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "numROIs must be in the range [1, PVA_DOA_CAPON_ROI_MAX_ROIS]");
    }

    // Validate snapshots are NHW [numROIs][numDopplerBinsCovMat][svDim] with data type 2S32.
    if (nvcv::TensorLayout{snapshotsReq->layout} != nvcv::TENSOR_NHW || snapshotsReq->dtype != NVCV_DATA_TYPE_2S32 ||
        snapshotsReq->shape[1] != numDopplerBins || snapshotsReq->shape[2] != svDim)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid snapshotsROITensorRequirements");
    }

    int32_t const numROIs         = static_cast<int32_t>(snapshotsReq->shape[0]);
    int64_t const snapshotRowSize = static_cast<int64_t>(svDim) * 2 * sizeof(int32_t);
    int64_t const snapshotRoiSize = snapshotsReq->shape[1] * snapshotRowSize;
    // Validate each snapshot ROI has contiguous storage.
    if (snapshotsReq->strides[0] < snapshotRoiSize || snapshotsReq->strides[0] % sizeof(int32_t) != 0 ||
        snapshotsReq->strides[1] != snapshotRowSize || snapshotsReq->strides[2] != 2 * sizeof(int32_t))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid snapshotsROITensorRequirements: each ROI must be contiguous");
    }

    // Validate steering vectors are NCHW [numROIs][svDim][numElBins][numAzBins] with data type 2S32.
    if (nvcv::TensorLayout{steeringReq->layout} != nvcv::TENSOR_NCHW || steeringReq->dtype != NVCV_DATA_TYPE_2S32 ||
        steeringReq->shape[0] != numROIs || steeringReq->shape[1] != svDim || steeringReq->shape[2] != numElBins ||
        steeringReq->shape[3] != numAzBins)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid steeringVectorsROITensorRequirements");
    }

    int32_t const svNum = numAzBins * numElBins;
    // Validate each steering-vector ROI has contiguous storage.
    if (steeringReq->strides[0] != svDim * svNum * 2 * sizeof(int32_t) ||
        steeringReq->strides[1] != svNum * 2 * sizeof(int32_t) ||
        steeringReq->strides[2] != numAzBins * 2 * sizeof(int32_t) || steeringReq->strides[3] != 2 * sizeof(int32_t))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid steeringVectorsROITensorRequirements: data must be contiguous");
    }

    // Validate the output tensors are HW [numROIs][numPeaksPerROI] with data type F32.
    int64_t const outputRowSize                      = static_cast<int64_t>(numPeaks) * sizeof(float);
    NVCVTensorRequirements const *const outputReqs[] = {peakPowerReq, azBinReq, elBinReq};
    for (NVCVTensorRequirements const *outputReq : outputReqs)
    {
        if (nvcv::TensorLayout{outputReq->layout} != nvcv::TENSOR_HW || outputReq->dtype != NVCV_DATA_TYPE_F32 ||
            outputReq->shape[0] != numROIs || outputReq->shape[1] != numPeaks ||
            outputReq->strides[0] < outputRowSize || outputReq->strides[0] % sizeof(float) != 0 ||
            outputReq->strides[1] != sizeof(float))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid Capon ROI output tensor requirements");
        }
    }

    int32_t const numDopplerBits = 31 - __builtin_clz(static_cast<unsigned>(numDopplerBins));
    int32_t const covSumQBits    = normalized.inputQBits * 2 - 16 - numDopplerBits;
    // Validate the derived S32 covariance Q format is in [0, 30].
    if (!normalized.useFpCovMatKnl && (covSumQBits < 0 || covSumQBits > 30))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "inputQBits produces an unsupported S32 covariance Q-format");
    }

    return normalized;
}

} // namespace

std::mutex DoaCaponROI::s_execMutex;
std::map<DoaCaponROI::ExecKey, std::weak_ptr<Executable>> DoaCaponROI::s_execs;

std::shared_ptr<Executable> DoaCaponROI::GetExec(bool useFpCovMat, bool enableSubBinInterp)
{
    ExecutableData data;
    if (useFpCovMat)
    {
        data = enableSubBinInterp ? ExecutableData{CUPVA_EXECUTABLE_DATA(pva_capon_roi_combined_fp_interp_dev),
                                                   CUPVA_EXECUTABLE_SIZE(pva_capon_roi_combined_fp_interp_dev)}
                                  : ExecutableData{CUPVA_EXECUTABLE_DATA(pva_capon_roi_combined_fp_no_interp_dev),
                                                   CUPVA_EXECUTABLE_SIZE(pva_capon_roi_combined_fp_no_interp_dev)};
    }
    else
    {
        data = enableSubBinInterp ? ExecutableData{CUPVA_EXECUTABLE_DATA(pva_capon_roi_combined_s32_interp_dev),
                                                   CUPVA_EXECUTABLE_SIZE(pva_capon_roi_combined_s32_interp_dev)}
                                  : ExecutableData{CUPVA_EXECUTABLE_DATA(pva_capon_roi_combined_s32_no_interp_dev),
                                                   CUPVA_EXECUTABLE_SIZE(pva_capon_roi_combined_s32_no_interp_dev)};
    }

    ExecKey key = std::make_tuple(Context::GetCurrent(), useFpCovMat, enableSubBinInterp);
    return GetExecutable<ExecKey>(data, key, s_execs, s_execMutex);
}

DoaCaponROI::DoaCaponROI(const NVCVTensorRequirements *snapshotsReq, const NVCVTensorRequirements *steeringReq,
                         const NVCVTensorRequirements *peakPowerReq, const NVCVTensorRequirements *azBinReq,
                         const NVCVTensorRequirements *elBinReq, const PVADoaCaponROIParams &params)
    : m_snapshotsReq(*snapshotsReq)
    , m_steeringReq(*steeringReq)
    , m_peakPowerReq(*peakPowerReq)
    , m_azBinReq(*azBinReq)
    , m_elBinReq(*elBinReq)
    , m_snapshotsOp(OffsetPointer<int32_t>::Create(MemType::DRAM, nullptr))
    , m_LinvRealOp(OffsetPointer<float>::Create(MemType::DRAM, nullptr))
    , m_LinvImagOp(OffsetPointer<float>::Create(MemType::DRAM, nullptr))
    , m_svOp(OffsetPointer<int32_t>::Create(MemType::DRAM, nullptr))
    , m_peakPowerOp(OffsetPointer<float>::Create(MemType::DRAM, nullptr))
    , m_azBinOp(OffsetPointer<float>::Create(MemType::DRAM, nullptr))
    , m_elBinOp(OffsetPointer<float>::Create(MemType::DRAM, nullptr))
    , m_prog{}
{
    int32_t const numROIs        = snapshotsReq->shape[0];
    int32_t const svDim          = params.svDim;
    int32_t const svNum          = params.numAzBins * params.numElBins;
    int32_t const numDopplerBins = params.numDopplerBinsCovMat;

    m_exec    = GetExec(params.useFpCovMatKnl, params.enableSubBinInterp);
    auto prog = CmdProgram::Create(*m_exec);
    prog["GP"].set(&params, 1);

    int32_t const snapImageLp      = snapshotsReq->strides[0] / sizeof(int32_t);
    int32_t const dopplerBatchSize = std::min(BUF_MAX_NUM_SAMPLES, numDopplerBins);
    int32_t const snapTileWidth    = svDim * 2 * dopplerBatchSize;
    int32_t const numDopplerTiles  = (numDopplerBins + dopplerBatchSize - 1) / dopplerBatchSize;

    int32_t const roiBatchSize   = std::min(BUF_MAX_ROI_NUM, numROIs);
    int32_t const snapTileHeight = roiBatchSize;
    int32_t const numRoiGroups   = (numROIs + roiBatchSize - 1) / roiBatchSize;

    int32_t const snapTileLp          = ((snapTileWidth - 1 + 15) / 16) * 16 + 1;
    int32_t const lastTileDopplerBins = numDopplerBins % dopplerBatchSize;
    int32_t const hasLastTileDoppler  = lastTileDopplerBins != 0;
    int32_t const lastTileRoiHeight   = numROIs % roiBatchSize;
    int32_t const hasLastTileRoi      = lastTileRoiHeight != 0;
    prog["numTilesROIdv"]             = numRoiGroups;
    prog["numTilesDoppler"]           = numDopplerTiles;
    if (!params.useFpCovMatKnl)
    {
        int32_t const numDopplerBits     = 31 - __builtin_clz(static_cast<unsigned>(numDopplerBins));
        prog["numDopplerBinsCovMatBits"] = numDopplerBits;
    }
    prog["dopplerBatchSize"]     = dopplerBatchSize;
    prog["snapImageLp"]          = snapImageLp;
    prog["snapTileWidth"]        = snapTileWidth;
    prog["snapTileHeight"]       = snapTileHeight;
    prog["hasLastTileDoppler"]   = hasLastTileDoppler;
    prog["lastTileDopplerBins"]  = lastTileDopplerBins;
    prog["lastTileWidthDoppler"] = lastTileDopplerBins * svDim * 2;
    prog["hasLastTileRoi"]       = hasLastTileRoi;
    prog["lastTileRoiHeight"]    = lastTileRoiHeight;
    prog["numROIs"]              = numROIs;

    float *LinvRealBuffer = prog["LinvRealBuffer"].ptr<float>();
    float *LinvImagBuffer = prog["LinvImagBuffer"].ptr<float>();
    int32_t *arenaBPtr    = prog["arenaBPtr"].ptr<int32_t>();
    int32_t *arenaCPtr    = prog["arenaCPtr"].ptr<int32_t>();

    // Part 1 data flows
    int32_t *snapshotBuffer = arenaBPtr;

    SequenceDataFlow &snapshotSQDF = prog.addDataFlowHead<SequenceDataFlow>(0).handler(prog["snapshotSQDFHdl"]);
    snapshotSQDF.addTransfer()
        .src(m_snapshotsOp, snapImageLp * sizeof(int32_t))
        .dst(snapshotBuffer, snapTileLp * sizeof(int32_t))
        .tile(snapTileWidth * sizeof(int32_t), snapTileHeight)
        .mode(TransferModeType::TILE);

    // L_inv scratch layout is packed NHW [ceil(numROIs / 16), svDim * (svDim + 1) / 2, 16] F32.
    int32_t const interWidth       = svDim * (svDim + 1) / 2 * 16;
    int32_t const interWidthBytes  = interWidth * sizeof(float);
    int32_t const interGroupStride = interWidthBytes;
    prog["interGroupStride"]       = interGroupStride;
    size_t const interBytes        = static_cast<size_t>((numROIs + 15) / 16) * interGroupStride;
    m_LinvReal.reset(static_cast<float *>(cupva::mem::Alloc(interBytes)));
    m_LinvImag.reset(static_cast<float *>(cupva::mem::Alloc(interBytes)));
    if (m_LinvReal == nullptr || m_LinvImag == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "Failed to allocate the Capon ROI L_inv scratch");
    }

    // Part 2 data flows
    int32_t *steeringCh0Buffer = arenaBPtr;
    int32_t *steeringCh1Buffer = arenaCPtr;
    float *peakPowerBuffer     = prog["peakPowerBuffer"].ptr<float>();
    float *azBinBuffer         = prog["azBinBuffer"].ptr<float>();
    float *elBinBuffer         = prog["elBinBuffer"].ptr<float>();

    SequenceDataFlow &interOutSQDF = prog.addDataFlowHead<SequenceDataFlow>(0).handler(prog["interOutSQDFHdl"]);
    interOutSQDF.addTransfer()
        .src(LinvRealBuffer, interWidthBytes)
        .dst(m_LinvRealOp, interGroupStride)
        .tile(interWidthBytes, 1)
        .mode(TransferModeType::CONTINUOUS);
    interOutSQDF.addTransfer()
        .src(LinvImagBuffer, interWidthBytes)
        .dst(m_LinvImagOp, interGroupStride)
        .tile(interWidthBytes, 1)
        .mode(TransferModeType::CONTINUOUS);

    SequenceDataFlow &interInSQDF = prog.addDataFlowHead<SequenceDataFlow>(1).handler(prog["interInSQDFHdl"]);
    interInSQDF.addTransfer()
        .src(m_LinvRealOp, interGroupStride)
        .dst(LinvRealBuffer, interWidthBytes)
        .tile(interWidthBytes, 1)
        .mode(TransferModeType::CONTINUOUS);
    interInSQDF.addTransfer()
        .src(m_LinvImagOp, interGroupStride)
        .dst(LinvImagBuffer, interWidthBytes)
        .tile(interWidthBytes, 1)
        .mode(TransferModeType::CONTINUOUS);

    // Steering vectors are read in 8-ROI tiles.
    int32_t const steeringRoiTileSize = 8;
    int32_t const numSteeringTiles    = (numROIs + steeringRoiTileSize - 1) / steeringRoiTileSize;
    int32_t const rfmWidth            = 16;
    int32_t const rfmLpBytes          = rfmWidth * sizeof(int32_t);
    int32_t const srcLpBytes          = 2 * rfmLpBytes;
    int32_t const steeringTileHeight  = steeringRoiTileSize * svDim * svNum / 16;
    int32_t const srcTileAdvBytes     = srcLpBytes * steeringTileHeight;
    int32_t const dstTileAdvBytes     = rfmLpBytes * steeringTileHeight;
    prog["numTilesROIsv"]             = numSteeringTiles;

    SequenceDataFlow &svCh0SQDF = prog.addDataFlowHead<SequenceDataFlow>(1).handler(prog["svCh0SQDFHdl"]);
    SequenceDataFlow &svCh1SQDF = prog.addDataFlowHead<SequenceDataFlow>(1).handler(prog["svCh1SQDFHdl"]);

    // A partial last tile reads only its remaining ROIs so the transfer stays inside the tensor.
    int32_t const numFullSteeringTiles = numROIs / steeringRoiTileSize;
    int32_t const lastSteeringTileRois = numROIs % steeringRoiTileSize;
    int32_t const numSteeringTilePairs = numFullSteeringTiles / 2;
    if (numSteeringTilePairs > 0)
    {
        svCh0SQDF.addTransfer()
            .src(m_svOp, srcLpBytes)
            .dst(steeringCh0Buffer, rfmLpBytes)
            .tile(rfmLpBytes, steeringTileHeight)
            .srcDim1(2, srcTileAdvBytes)
            .srcDim2(numSteeringTilePairs, 2 * srcTileAdvBytes)
            .dstDim1(2, dstTileAdvBytes)
            .dstDim2(numSteeringTilePairs, 0)
            .mode(TransferModeType::TILE);
        svCh1SQDF.addTransfer()
            .src(m_svOp + rfmWidth, srcLpBytes)
            .dst(steeringCh1Buffer, rfmLpBytes)
            .tile(rfmLpBytes, steeringTileHeight)
            .srcDim1(2, srcTileAdvBytes)
            .srcDim2(numSteeringTilePairs, 2 * srcTileAdvBytes)
            .dstDim1(2, dstTileAdvBytes)
            .dstDim2(numSteeringTilePairs, 0)
            .mode(TransferModeType::TILE);
    }

    // Tiles after the full pairs alternate between the ping and pong halves of each buffer.
    for (int32_t tile = 2 * numSteeringTilePairs; tile < numSteeringTiles; ++tile)
    {
        int32_t const tileHeight = tile < numFullSteeringTiles
                                       ? steeringTileHeight
                                       : lastSteeringTileRois * steeringTileHeight / steeringRoiTileSize;
        int32_t const srcOffset  = tile * srcTileAdvBytes / sizeof(int32_t);
        int32_t const dstOffset  = (tile & 1) * dstTileAdvBytes / sizeof(int32_t);
        svCh0SQDF.addTransfer()
            .src(m_svOp + srcOffset, srcLpBytes)
            .dst(steeringCh0Buffer + dstOffset, rfmLpBytes)
            .tile(rfmLpBytes, tileHeight)
            .mode(TransferModeType::TILE);
        svCh1SQDF.addTransfer()
            .src(m_svOp + srcOffset + rfmWidth, srcLpBytes)
            .dst(steeringCh1Buffer + dstOffset, rfmLpBytes)
            .tile(rfmLpBytes, tileHeight)
            .mode(TransferModeType::TILE);
    }

    int32_t const outWidthBytes      = params.numPeaksPerROI * sizeof(float);
    int32_t const outTileHeight      = std::min(BUF_MAX_ROI_NUM, numROIs);
    int32_t const outVmemPitchBytes  = PEAK_POWER_LINE_PITCH * sizeof(float);
    int32_t const peakPowerRowStride = static_cast<int32_t>(peakPowerReq->strides[0]);
    int32_t const azBinRowStride     = static_cast<int32_t>(azBinReq->strides[0]);
    int32_t const elBinRowStride     = static_cast<int32_t>(elBinReq->strides[0]);
    prog["peakPowerRowStride"]       = peakPowerRowStride;
    prog["azBinRowStride"]           = azBinRowStride;
    prog["elBinRowStride"]           = elBinRowStride;

    SequenceDataFlow &outSQDF = prog.addDataFlowHead<SequenceDataFlow>(1).handler(prog["outSQDFHdl"]);
    outSQDF.addTransfer()
        .src(peakPowerBuffer, outVmemPitchBytes)
        .dst(m_peakPowerOp, peakPowerRowStride)
        .tile(outWidthBytes, outTileHeight)
        .mode(TransferModeType::CONTINUOUS);
    outSQDF.addTransfer()
        .src(azBinBuffer, outVmemPitchBytes)
        .dst(m_azBinOp, azBinRowStride)
        .tile(outWidthBytes, outTileHeight)
        .mode(TransferModeType::CONTINUOUS);
    outSQDF.addTransfer()
        .src(elBinBuffer, outVmemPitchBytes)
        .dst(m_elBinOp, elBinRowStride)
        .tile(outWidthBytes, outTileHeight)
        .mode(TransferModeType::CONTINUOUS);

    prog.compileDataFlows();
    ConvertTo(std::move(prog), m_prog);
}

DoaCaponROI::~DoaCaponROI()
{
    CupvaCmdDestroy(&m_prog);
}

void DoaCaponROI::Bind(const NVCVTensorData &snapshots, const NVCVTensorData &steering, const NVCVTensorData &peakPower,
                       const NVCVTensorData &azBin, const NVCVTensorData &elBin)
{
    int32_t *snapshotsBase = static_cast<int32_t *>(static_cast<void *>(snapshots.buffer.strided.basePtr));
    float *interRealBase   = m_LinvReal.get();
    float *interImagBase   = m_LinvImag.get();
    int32_t *steeringBase  = static_cast<int32_t *>(static_cast<void *>(steering.buffer.strided.basePtr));
    float *peakPowerBase   = static_cast<float *>(static_cast<void *>(peakPower.buffer.strided.basePtr));
    float *azBinBase       = static_cast<float *>(static_cast<void *>(azBin.buffer.strided.basePtr));
    float *elBinBase       = static_cast<float *>(static_cast<void *>(elBin.buffer.strided.basePtr));

    m_snapshotsOp.setBase(snapshotsBase);
    m_LinvRealOp.setBase(interRealBase);
    m_LinvImagOp.setBase(interImagBase);
    m_svOp.setBase(steeringBase);
    m_peakPowerOp.setBase(peakPowerBase);
    m_azBinOp.setBase(azBinBase);
    m_elBinOp.setBase(elBinBase);

    auto setExternalPointer = [this](const char *name, void *base) {
        cupvaParameter_t parameter;
        if (CupvaCmdProgramGetParameter(&m_prog, &parameter, name) != CUPVA_ERROR_NONE ||
            CupvaParameterSetValuePointer(&parameter, base) != CUPVA_ERROR_NONE)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "Failed to update a Capon ROI external pointer");
        }
    };
    setExternalPointer("snapshotsExtPtr", snapshotsBase);
    setExternalPointer("interRealExtPtr", interRealBase);
    setExternalPointer("interImagExtPtr", interImagBase);
    setExternalPointer("peakPowerExtPtr", peakPowerBase);
    setExternalPointer("azBinExtPtr", azBinBase);
    setExternalPointer("elBinExtPtr", elBinBase);

    if (CupvaCmdProgramUpdateDataFlows(&m_prog) != CUPVA_ERROR_NONE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "Failed to update combined Capon ROI data flows");
    }
}

template<typename S>
void DoaCaponROI::operator()(S stream, NVCVTensorHandle snapshotsHandle, NVCVTensorHandle steeringHandle,
                             NVCVTensorHandle peakPowerHandle, NVCVTensorHandle azBinHandle,
                             NVCVTensorHandle elBinHandle)
{
    NVCVTensorData snapshotsData, steeringData, peakPowerData, azBinData, elBinData;
    auto exportTensor = [](NVCVTensorHandle handle, NVCVTensorData *data) {
        if (nvcvTensorExportData(handle, data) != NVCV_SUCCESS)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Failed to export combined Capon ROI tensor data");
        }
    };
    exportTensor(snapshotsHandle, &snapshotsData);
    exportTensor(steeringHandle, &steeringData);
    exportTensor(peakPowerHandle, &peakPowerData);
    exportTensor(azBinHandle, &azBinData);
    exportTensor(elBinHandle, &elBinData);

    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_snapshotsReq, snapshotsData) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_steeringReq, steeringData) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_peakPowerReq, peakPowerData) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_azBinReq, azBinData) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_elBinReq, elBinData))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Capon ROI tensors must match the requirements used to create the operator");
    }

    if (snapshotsData.buffer.strided.basePtr == nullptr || steeringData.buffer.strided.basePtr == nullptr ||
        peakPowerData.buffer.strided.basePtr == nullptr || azBinData.buffer.strided.basePtr == nullptr ||
        elBinData.buffer.strided.basePtr == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Combined Capon ROI tensor base pointers must not be NULL");
    }

    Bind(snapshotsData, steeringData, peakPowerData, azBinData, elBinData);
    if (SubmitCommand(stream, Command()) != CUPVA_ERROR_NONE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_DEVICE, "Combined Capon ROI command submission failed");
    }
}

cupvaCmd_t const *DoaCaponROI::Command() const
{
    return &m_prog;
}

} // namespace pva_solutions::priv

extern "C" NVCVStatus pvaDoaCaponROICreate(NVCVOperatorHandle *handle, NVCVTensorRequirements const *const snapshotsReq,
                                           NVCVTensorRequirements const *const steeringReq,
                                           NVCVTensorRequirements const *const peakPowerReq,
                                           NVCVTensorRequirements const *const azBinReq,
                                           NVCVTensorRequirements const *const elBinReq,
                                           PVADoaCaponROIParams const *const params)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr || snapshotsReq == nullptr || steeringReq == nullptr || peakPowerReq == nullptr ||
            azBinReq == nullptr || elBinReq == nullptr || params == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Combined Capon ROI create arguments must not be NULL");
        }
        PVADoaCaponROIParams normalized = pva_solutions::priv::ValidateCombinedRequirements(
            snapshotsReq, steeringReq, peakPowerReq, azBinReq, elBinReq, params);
        *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::DoaCaponROI(
            snapshotsReq, steeringReq, peakPowerReq, azBinReq, elBinReq, normalized));
    });
}

template<typename StreamType>
NVCVStatus DoaCaponROISubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle snapshotsROI,
                                 NVCVTensorHandle steeringVectorsROI, NVCVTensorHandle peakPower,
                                 NVCVTensorHandle azimuthBinIndices, NVCVTensorHandle elevationBinIndices)
{
    return nvcv::ProtectCall([&] {
        pva_solutions::priv::ToDynamicRef<pva_solutions::priv::DoaCaponROI>(handle)(
            stream, snapshotsROI, steeringVectorsROI, peakPower, azimuthBinIndices, elevationBinIndices);
    });
}

template NVCVStatus DoaCaponROISubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle,
                                                         NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle,
                                                         NVCVTensorHandle);
template NVCVStatus DoaCaponROISubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle,
                                                        NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle,
                                                        NVCVTensorHandle);

NVCVStatus pvaDoaCaponROISubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle const snapshotsROI,
                                NVCVTensorHandle const steeringVectorsROI, NVCVTensorHandle peakPower,
                                NVCVTensorHandle azimuthBinIndices, NVCVTensorHandle elevationBinIndices)
{
    return DoaCaponROISubmitImpl(handle, stream, snapshotsROI, steeringVectorsROI, peakPower, azimuthBinIndices,
                                 elevationBinIndices);
}

NVCVStatus pvaDoaCaponROISubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle const snapshotsROI,
                                NVCVTensorHandle const steeringVectorsROI, NVCVTensorHandle peakPower,
                                NVCVTensorHandle azimuthBinIndices, NVCVTensorHandle elevationBinIndices)
{
    return DoaCaponROISubmitImpl(handle, stream, snapshotsROI, steeringVectorsROI, peakPower, azimuthBinIndices,
                                 elevationBinIndices);
}
