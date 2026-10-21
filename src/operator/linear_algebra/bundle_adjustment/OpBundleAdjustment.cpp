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

#include "OpBundleAdjustment.hpp"

#include "device/bundle_adjustment_param.h"

#include <IOperator.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <cupva_host_utils.hpp>
#include <nvcv/Tensor.hpp>

#include <OpBundleAdjustment.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <cstdint>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(bundle_adjustment_dev);

namespace pva_solutions::priv {

std::mutex BundleAdjustment::s_execMapMutex;

std::map<BundleAdjustment::ExecKey, std::weak_ptr<cupva::Executable>> BundleAdjustment::s_bundleAdjustmentExecutables;

std::shared_ptr<cupva::Executable> BundleAdjustment::GetBundleAdjustmentExec()
{
    static IOperator::ExecutableData sBundleAdjustmentExecData = {CUPVA_EXECUTABLE_DATA(bundle_adjustment_dev),
                                                                  CUPVA_EXECUTABLE_SIZE(bundle_adjustment_dev)};

    ExecKey execKey = std::make_tuple(cupva::Context::GetCurrent());
    return IOperator::GetExecutable<ExecKey>(sBundleAdjustmentExecData, execKey, s_bundleAdjustmentExecutables,
                                             s_execMapMutex);
}

namespace {

void ValidateTensorRequirements(NVCVTensorRequirements const *req, int32_t rank, int64_t const *shape,
                                NVCVDataType dtype, char const *name)
{
    if (req == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Pointer to tensor requirements must not be NULL");
    }
    if (req->dtype != dtype)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unexpected data type");
    }
    if (req->rank != rank)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unexpected tensor rank");
    }
    for (int32_t i = 0; i < rank; ++i)
    {
        if (req->shape[i] != shape[i])
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unexpected tensor shape");
        }
    }
    (void)name;
}

} // namespace

BundleAdjustment::BundleAdjustment(NVCVTensorRequirements const *rigPoses, NVCVTensorRequirements const *points3d,
                                   NVCVTensorRequirements const *measurements,
                                   NVCVTensorRequirements const *cameraBitmask, NVCVTensorRequirements const *rig2cam,
                                   NVCVTensorRequirements const *updatedRigPoses,
                                   NVCVTensorRequirements const *updatedPoints3d,
                                   NVCVTensorRequirements const *finalCost,
                                   NVCVTensorRequirements const *iterationsUsed,
                                   PvaBundleAdjustmentConfig const *config)
    : m_rigPosesTensorRequirements(*rigPoses)
    , m_points3dTensorRequirements(*points3d)
    , m_measurementsTensorRequirements(*measurements)
    , m_cameraBitmaskTensorRequirements(*cameraBitmask)
    , m_rig2camTensorRequirements(*rig2cam)
    , m_updatedRigPosesTensorRequirements(*updatedRigPoses)
    , m_updatedPoints3dTensorRequirements(*updatedPoints3d)
    , m_finalCostTensorRequirements(*finalCost)
    , m_iterationsUsedTensorRequirements(*iterationsUsed)
    , m_activeCameraCount(config->activeCameraCount)
    , m_activeCameraBegin(config->activeCameraBegin)
    , m_rigPosesOp(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr))
    , m_points3dOp(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr))
    , m_measurementsOp(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr))
    , m_cameraBitmaskOp(cupva::OffsetPointer<int32_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_rig2camOp(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr))
    , m_updatedRigPosesOp(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr))
    , m_updatedPoints3dOp(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr))
    , m_finalCostOp(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr))
    , m_iterationsUsedOp(cupva::OffsetPointer<int32_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_tpcScratch(nullptr)
    , m_tpScratch(nullptr)
    , m_points3dScratch(nullptr)
    , m_points3dScratch2(nullptr)
    , m_exec{}
    , m_prog{}
{
    if (config == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Pointer to config must not be NULL");
    }

    int64_t const rigPosesShape[2]     = {kMaxCameraCount, 16};
    int64_t const points3dShape[2]     = {kMaxTrackCount, 4};
    int64_t const measurementsShape[3] = {kMaxTrackCount, kMaxCameraCount, 2};
    int64_t const bitmaskShape[1]      = {kMaxTrackCount};
    int64_t const rig2camShape[1]      = {16};
    int64_t const scalarShape[1]       = {1};

    ValidateTensorRequirements(rigPoses, 2, rigPosesShape, NVCV_DATA_TYPE_F32, "rigPoses");
    ValidateTensorRequirements(points3d, 2, points3dShape, NVCV_DATA_TYPE_F32, "points3d");
    ValidateTensorRequirements(measurements, 3, measurementsShape, NVCV_DATA_TYPE_F32, "measurements");
    ValidateTensorRequirements(cameraBitmask, 1, bitmaskShape, NVCV_DATA_TYPE_S32, "cameraBitmask");
    ValidateTensorRequirements(rig2cam, 1, rig2camShape, NVCV_DATA_TYPE_F32, "rig2cam");
    ValidateTensorRequirements(updatedRigPoses, 2, rigPosesShape, NVCV_DATA_TYPE_F32, "updatedRigPoses");
    ValidateTensorRequirements(updatedPoints3d, 2, points3dShape, NVCV_DATA_TYPE_F32, "updatedPoints3d");
    ValidateTensorRequirements(finalCost, 1, scalarShape, NVCV_DATA_TYPE_F32, "finalCost");
    ValidateTensorRequirements(iterationsUsed, 1, scalarShape, NVCV_DATA_TYPE_S32, "iterationsUsed");

    // cameraCount is only known at Submit; begin + count <= cameraCount is checked in operator().
    if (m_activeCameraCount < 1 || m_activeCameraCount > kActiveCameraCount)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "activeCameraCount must be in [1, kActiveCameraCount]");
    }
    if (m_activeCameraBegin < 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "activeCameraBegin must be >= 0");
    }

    // Worst-case sizing: trackCount is only known at Submit.
    int64_t const tpcScratchElems      = static_cast<int64_t>(kMaxTrackCount) * kActiveCameraCount * 18;
    int64_t const tpScratchElems       = static_cast<int64_t>(kMaxTrackCount) * 3;
    int64_t const points3dScratchElems = static_cast<int64_t>(kMaxTrackCount) * 4;

    m_tpcScratch = cupva::mem::Alloc(tpcScratchElems * static_cast<int64_t>(sizeof(float)));
    m_tpScratch  = cupva::mem::Alloc(tpScratchElems * static_cast<int64_t>(sizeof(float)));
    // The device swaps these on an accepted LM step instead of copying.
    m_points3dScratch  = cupva::mem::Alloc(points3dScratchElems * static_cast<int64_t>(sizeof(float)));
    m_points3dScratch2 = cupva::mem::Alloc(points3dScratchElems * static_cast<int64_t>(sizeof(float)));

    m_exec    = GetBundleAdjustmentExec();
    auto prog = CmdProgram::Create(*m_exec.get());

    prog["active_camera_count"] = m_activeCameraCount;
    prog["active_camera_begin"] = m_activeCameraBegin;

    auto oneShotInputHdl     = prog["oneShotInputHdl"];
    auto points3dHdl         = prog["points3dHdl"];
    auto measurementsHdl     = prog["measurementsHdl"];
    auto cameraBitmaskHdl    = prog["cameraBitmaskHdl"];
    auto tpcWriteHdl         = prog["tpcWriteHdl"];
    auto tpcReadHdl          = prog["tpcReadHdl"];
    auto tpWriteHdl          = prog["tpWriteHdl"];
    auto tpReadHdl           = prog["tpReadHdl"];
    auto pnewTentativeOutHdl = prog["pnewTentativeOutHdl"];
    auto pnewCommitLoadHdl   = prog["pnewCommitLoadHdl"];
    auto pnewCommitStoreHdl  = prog["pnewCommitStoreHdl"];
    auto oneShotOutputHdl    = prog["oneShotOutputHdl"];

    float *rig2cam_v    = prog["rig2cam"].ptr<float>();
    int32_t *bitmask_v  = prog["camera_bitmask"].ptr<int32_t>();
    float *tpc_v        = prog["Tpc"].ptr<float>();
    float *hpc_v        = prog["Hpc"].ptr<float>();
    float *tp_v         = prog["tp"].ptr<float>();
    float *pnew_v       = prog["pnew"].ptr<float>();
    float *totalCost_v  = prog["total_cost"].ptr<float>();
    int32_t *iterUsed_v = prog["iterationsUsed"].ptr<int32_t>();

    // Padded staging for transposing load/store (pitches in bundle_adjustment_param.h); DMA lands here,
    // not in the plain SoA symbols.
    float *points3dStage_v     = prog["points3d_stage"].ptr<float>();
    float *rigPoseStage_v      = prog["rig_pose_stage"].ptr<float>();
    float *measurementsStage_v = prog["measurements_stage"].ptr<float>();

    int32_t const rig2camBytes = static_cast<int32_t>(16 * sizeof(float));
    int32_t const bitmaskBytes = static_cast<int32_t>(kTrackCountTile * sizeof(int32_t));
    int32_t const tpcBytes     = static_cast<int32_t>(18 * kTrackCountTile * sizeof(float));
    int32_t const tpcReadBytes = static_cast<int32_t>(18 * m_activeCameraCount * kTrackCountTile * sizeof(float));
    int32_t const tpBytes      = static_cast<int32_t>(3 * kTrackCountTile * sizeof(float));
    int32_t const pnewBytes    = static_cast<int32_t>(4 * kTrackCountTile * sizeof(float));
    int32_t const scalarBytes  = static_cast<int32_t>(sizeof(float));

    // DRAM side stays tight; padding applies only to the VMEM-side pitch.
    int32_t const rigPoseRowBytes         = static_cast<int32_t>(16 * sizeof(float));
    int32_t const rigPosePaddedBytes      = static_cast<int32_t>(kRigPoseStagePitch * sizeof(float));
    int32_t const points3dRowBytes        = static_cast<int32_t>(4 * sizeof(float));
    int32_t const points3dPaddedBytes     = static_cast<int32_t>(kPoints3dStagePitch * sizeof(float));
    int32_t const measurementsRowBytes    = static_cast<int32_t>(2 * kMaxCameraCount * sizeof(float));
    int32_t const measurementsPaddedBytes = static_cast<int32_t>(kMeasurementsStagePitch * sizeof(float));

    // All transfers are CONTINUOUS; bundle_adjustment_top.c sets runtime addresses via cupvaSQDFUpdateAddr().
    int32_t phase = 0;

    // Shared handler (0 = rig poses, 1 = rig2cam) to save a DMA trigger-pool entry.
    SequenceDataFlow &oneShotInputFlow = prog.addDataFlowHead<SequenceDataFlow>(phase).handler(oneShotInputHdl);
    oneShotInputFlow.addTransfer()
        .src(m_rigPosesOp, rigPoseRowBytes)
        .dst(rigPoseStage_v, rigPosePaddedBytes)
        .tile(rigPoseRowBytes, kMaxCameraCount)
        .mode(TransferModeType::CONTINUOUS);
    oneShotInputFlow.addTransfer()
        .src(m_rig2camOp, rig2camBytes)
        .dst(rig2cam_v, rig2camBytes)
        .tile(rig2camBytes, 1)
        .mode(TransferModeType::CONTINUOUS);
    phase++;

    // Per-tile handlers feed the padded staging buffers consumed by the transposing load in bundle_adjustment_top.c.
    SequenceDataFlow &points3dFlow = prog.addDataFlowHead<SequenceDataFlow>(phase).handler(points3dHdl);
    points3dFlow.addTransfer()
        .src(m_points3dOp, points3dRowBytes)
        .dst(points3dStage_v, points3dPaddedBytes)
        .tile(points3dRowBytes, kTrackCountTile)
        .mode(TransferModeType::CONTINUOUS);
    phase++;

    SequenceDataFlow &measurementsFlow = prog.addDataFlowHead<SequenceDataFlow>(phase).handler(measurementsHdl);
    measurementsFlow.addTransfer()
        .src(m_measurementsOp, measurementsRowBytes)
        .dst(measurementsStage_v, measurementsPaddedBytes)
        .tile(measurementsRowBytes, kTrackCountTile)
        .mode(TransferModeType::CONTINUOUS);

    SequenceDataFlow &cameraBitmaskFlow = prog.addDataFlowHead<SequenceDataFlow>(phase).handler(cameraBitmaskHdl);
    cameraBitmaskFlow.addTransfer()
        .src(m_cameraBitmaskOp, bitmaskBytes)
        .dst(bitmask_v, bitmaskBytes)
        .tile(bitmaskBytes, 1)
        .mode(TransferModeType::CONTINUOUS);

    // A compiled transfer fixes one direction, so each scratch store and load needs its own handler.
    SequenceDataFlow &tpcReadFlow = prog.addDataFlowHead<SequenceDataFlow>(phase).handler(tpcReadHdl);
    tpcReadFlow.addTransfer()
        .src(static_cast<void *>(m_tpcScratch), tpcReadBytes)
        .dst(hpc_v, tpcReadBytes)
        .tile(tpcReadBytes, 1)
        .mode(TransferModeType::CONTINUOUS);

    SequenceDataFlow &tpReadFlow = prog.addDataFlowHead<SequenceDataFlow>(phase).handler(tpReadHdl);
    tpReadFlow.addTransfer()
        .src(static_cast<void *>(m_tpScratch), tpBytes)
        .dst(tp_v, tpBytes)
        .tile(tpBytes, 1)
        .mode(TransferModeType::CONTINUOUS);

    // SoA LM point state: TentativeOut stores the tentative tile, CommitLoad loads the current tile
    // (LM loop and final export).
    SequenceDataFlow &pnewTentativeOutFlow = prog.addDataFlowHead<SequenceDataFlow>(phase).handler(pnewTentativeOutHdl);
    pnewTentativeOutFlow.addTransfer()
        .src(pnew_v, pnewBytes)
        .dst(static_cast<void *>(m_points3dScratch), pnewBytes)
        .tile(pnewBytes, 1)
        .mode(TransferModeType::CONTINUOUS);

    SequenceDataFlow &pnewCommitLoadFlow = prog.addDataFlowHead<SequenceDataFlow>(phase).handler(pnewCommitLoadHdl);
    pnewCommitLoadFlow.addTransfer()
        .src(static_cast<void *>(m_points3dScratch), pnewBytes)
        .dst(pnew_v, pnewBytes)
        .tile(pnewBytes, 1)
        .mode(TransferModeType::CONTINUOUS);

    SequenceDataFlow &pnewCommitStoreFlow = prog.addDataFlowHead<SequenceDataFlow>(phase).handler(pnewCommitStoreHdl);
    pnewCommitStoreFlow.addTransfer()
        .src(points3dStage_v, points3dPaddedBytes)
        .dst(m_updatedPoints3dOp, points3dRowBytes)
        .tile(points3dRowBytes, kTrackCountTile)
        .mode(TransferModeType::CONTINUOUS);
    phase++;

    SequenceDataFlow &tpcWriteFlow = prog.addDataFlowHead<SequenceDataFlow>(phase).handler(tpcWriteHdl);
    tpcWriteFlow.addTransfer()
        .src(tpc_v, tpcBytes)
        .dst(static_cast<void *>(m_tpcScratch), tpcBytes)
        .tile(tpcBytes, 1)
        .mode(TransferModeType::CONTINUOUS);

    SequenceDataFlow &tpWriteFlow = prog.addDataFlowHead<SequenceDataFlow>(phase).handler(tpWriteHdl);
    tpWriteFlow.addTransfer()
        .src(tp_v, tpBytes)
        .dst(static_cast<void *>(m_tpScratch), tpBytes)
        .tile(tpBytes, 1)
        .mode(TransferModeType::CONTINUOUS);
    phase++;

    // Shared handler (0 = updated rig poses, 1 = final cost, 2 = iterations used).
    SequenceDataFlow &oneShotOutputFlow = prog.addDataFlowHead<SequenceDataFlow>(phase).handler(oneShotOutputHdl);
    oneShotOutputFlow.addTransfer()
        .src(rigPoseStage_v, rigPosePaddedBytes)
        .dst(m_updatedRigPosesOp, rigPoseRowBytes)
        .tile(rigPoseRowBytes, kMaxCameraCount)
        .mode(TransferModeType::CONTINUOUS);
    oneShotOutputFlow.addTransfer()
        .src(totalCost_v, scalarBytes)
        .dst(m_finalCostOp, scalarBytes)
        .tile(scalarBytes, 1)
        .mode(TransferModeType::CONTINUOUS);
    oneShotOutputFlow.addTransfer()
        .src(iterUsed_v, static_cast<int32_t>(sizeof(int32_t)))
        .dst(m_iterationsUsedOp, static_cast<int32_t>(sizeof(int32_t)))
        .tile(static_cast<int32_t>(sizeof(int32_t)), 1)
        .mode(TransferModeType::CONTINUOUS);

    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
}

BundleAdjustment::~BundleAdjustment()
{
    CupvaCmdDestroy(&m_prog);
    if (m_tpcScratch != nullptr)
    {
        cupva::mem::Free(m_tpcScratch);
    }
    if (m_tpScratch != nullptr)
    {
        cupva::mem::Free(m_tpScratch);
    }
    if (m_points3dScratch2 != nullptr)
    {
        cupva::mem::Free(m_points3dScratch2);
        m_points3dScratch2 = nullptr;
    }
    if (m_points3dScratch != nullptr)
    {
        cupva::mem::Free(m_points3dScratch);
    }
}

template<typename StreamType>
void BundleAdjustment::operator()(StreamType stream, const nvcv::Tensor &rigPoses, const nvcv::Tensor &points3d,
                                  const nvcv::Tensor &measurements, const nvcv::Tensor &cameraBitmask,
                                  const nvcv::Tensor &rig2cam, const nvcv::Tensor &updatedRigPoses,
                                  const nvcv::Tensor &updatedPoints3d, const nvcv::Tensor &finalCost,
                                  const nvcv::Tensor &iterationsUsed, PvaBundleAdjustmentParams const *params)
{
    if (params == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Pointer to params must not be NULL");
    }
    if (params->trackCount <= 0 || (params->trackCount % kTrackCountTile) != 0 || params->trackCount > kMaxTrackCount)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "trackCount must be a positive multiple of kTrackCountTile and <= kMaxTrackCount");
    }
    if (params->cameraCount <= 0 || params->cameraCount > kMaxCameraCount)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "cameraCount must be in [1, kMaxCameraCount]");
    }
    if ((m_activeCameraBegin + m_activeCameraCount) > params->cameraCount)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "activeCameraBegin + activeCameraCount must be <= cameraCount");
    }
    if (params->maxIterations <= 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "maxIterations must be > 0");
    }
    if (!(params->minDistance > 0.0f))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "minDistance must be > 0, got %g",
                              static_cast<double>(params->minDistance));
    }
    if (!(params->robustifierScaleMeas > 0.0f))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "robustifierScaleMeas must be > 0, got %g",
                              static_cast<double>(params->robustifierScaleMeas));
    }
    if (!(params->rinvMeas > 0.0f))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "rinvMeas must be > 0, got %g",
                              static_cast<double>(params->rinvMeas));
    }
    if (!(params->lambdaInit > 0.0f))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "lambdaInit must be > 0, got %g",
                              static_cast<double>(params->lambdaInit));
    }

    auto rigPosesData        = rigPoses.exportData<nvcv::TensorDataStrided>();
    auto points3dData        = points3d.exportData<nvcv::TensorDataStrided>();
    auto measurementsData    = measurements.exportData<nvcv::TensorDataStrided>();
    auto cameraBitmaskData   = cameraBitmask.exportData<nvcv::TensorDataStrided>();
    auto rig2camData         = rig2cam.exportData<nvcv::TensorDataStrided>();
    auto updatedRigPosesData = updatedRigPoses.exportData<nvcv::TensorDataStrided>();
    auto updatedPoints3dData = updatedPoints3d.exportData<nvcv::TensorDataStrided>();
    auto finalCostData       = finalCost.exportData<nvcv::TensorDataStrided>();
    auto iterationsUsedData  = iterationsUsed.exportData<nvcv::TensorDataStrided>();

    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_rigPosesTensorRequirements, rigPosesData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_points3dTensorRequirements, points3dData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_measurementsTensorRequirements,
                                                          measurementsData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_cameraBitmaskTensorRequirements,
                                                          cameraBitmaskData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_rig2camTensorRequirements, rig2camData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_updatedRigPosesTensorRequirements,
                                                          updatedRigPosesData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_updatedPoints3dTensorRequirements,
                                                          updatedPoints3dData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_finalCostTensorRequirements, finalCostData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_iterationsUsedTensorRequirements,
                                                          iterationsUsedData->cdata()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Tensors must have the same requirements that were used to create the operator");
    }

    void *rigPosesBase        = static_cast<void *>(rigPosesData->basePtr());
    void *points3dBase        = static_cast<void *>(points3dData->basePtr());
    void *measurementsBase    = static_cast<void *>(measurementsData->basePtr());
    void *cameraBitmaskBase   = static_cast<void *>(cameraBitmaskData->basePtr());
    void *rig2camBase         = static_cast<void *>(rig2camData->basePtr());
    void *updatedRigPosesBase = static_cast<void *>(updatedRigPosesData->basePtr());
    void *updatedPoints3dBase = static_cast<void *>(updatedPoints3dData->basePtr());
    void *finalCostBase       = static_cast<void *>(finalCostData->basePtr());
    void *iterationsUsedBase  = static_cast<void *>(iterationsUsedData->basePtr());

    if (rigPosesBase == nullptr || points3dBase == nullptr || measurementsBase == nullptr ||
        cameraBitmaskBase == nullptr || rig2camBase == nullptr || updatedRigPosesBase == nullptr ||
        updatedPoints3dBase == nullptr || finalCostBase == nullptr || iterationsUsedBase == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor base pointers must not be NULL");
    }

    m_rigPosesOp.setBase(static_cast<float *>(rigPosesBase));
    m_points3dOp.setBase(static_cast<float *>(points3dBase));
    m_measurementsOp.setBase(static_cast<float *>(measurementsBase));
    m_cameraBitmaskOp.setBase(static_cast<int32_t *>(cameraBitmaskBase));
    m_rig2camOp.setBase(static_cast<float *>(rig2camBase));
    m_updatedRigPosesOp.setBase(static_cast<float *>(updatedRigPosesBase));
    m_updatedPoints3dOp.setBase(static_cast<float *>(updatedPoints3dBase));
    m_finalCostOp.setBase(static_cast<float *>(finalCostBase));
    m_iterationsUsedOp.setBase(static_cast<int32_t *>(iterationsUsedBase));

    // bundle_adjustment_top.c computes every runtime DMA address from these base pointers.
    cupvaParameter_t param;

    CupvaCmdProgramGetParameter(&m_prog, &param, "rigPosesExtPtr");
    CupvaParameterSetValuePointer(&param, rigPosesBase);
    CupvaCmdProgramGetParameter(&m_prog, &param, "points3dExtPtr");
    CupvaParameterSetValuePointer(&param, points3dBase);
    CupvaCmdProgramGetParameter(&m_prog, &param, "measurementsExtPtr");
    CupvaParameterSetValuePointer(&param, measurementsBase);
    CupvaCmdProgramGetParameter(&m_prog, &param, "cameraBitmaskExtPtr");
    CupvaParameterSetValuePointer(&param, cameraBitmaskBase);
    CupvaCmdProgramGetParameter(&m_prog, &param, "rig2camExtPtr");
    CupvaParameterSetValuePointer(&param, rig2camBase);
    CupvaCmdProgramGetParameter(&m_prog, &param, "updatedRigPosesExtPtr");
    CupvaParameterSetValuePointer(&param, updatedRigPosesBase);
    CupvaCmdProgramGetParameter(&m_prog, &param, "updatedPoints3dExtPtr");
    CupvaParameterSetValuePointer(&param, updatedPoints3dBase);
    CupvaCmdProgramGetParameter(&m_prog, &param, "finalCostExtPtr");
    CupvaParameterSetValuePointer(&param, finalCostBase);
    CupvaCmdProgramGetParameter(&m_prog, &param, "iterationsUsedExtPtr");
    CupvaParameterSetValuePointer(&param, iterationsUsedBase);
    CupvaCmdProgramGetParameter(&m_prog, &param, "tpcScratchExtPtr");
    CupvaParameterSetValuePointer(&param, m_tpcScratch);
    CupvaCmdProgramGetParameter(&m_prog, &param, "tpScratchExtPtr");
    CupvaParameterSetValuePointer(&param, m_tpScratch);
    CupvaCmdProgramGetParameter(&m_prog, &param, "points3dScratchExtPtr");
    CupvaParameterSetValuePointer(&param, m_points3dScratch);
    CupvaCmdProgramGetParameter(&m_prog, &param, "points3dScratch2ExtPtr");
    CupvaParameterSetValuePointer(&param, m_points3dScratch2);

    // LM schedule constants live in bundle_adjustment_param.h.
    int32_t const trackIter = params->trackCount / kTrackCountTile;

    CupvaCmdProgramGetParameter(&m_prog, &param, "track_iter");
    CupvaParameterSetValueScalar(&param, &trackIter, sizeof(int32_t));
    CupvaCmdProgramGetParameter(&m_prog, &param, "camera_count");
    CupvaParameterSetValueScalar(&param, &params->cameraCount, sizeof(int32_t));
    CupvaCmdProgramGetParameter(&m_prog, &param, "min_distance");
    CupvaParameterSetValueScalar(&param, &params->minDistance, sizeof(float));
    CupvaCmdProgramGetParameter(&m_prog, &param, "scale_meas");
    CupvaParameterSetValueScalar(&param, &params->robustifierScaleMeas, sizeof(float));
    CupvaCmdProgramGetParameter(&m_prog, &param, "rinv_meas");
    CupvaParameterSetValueScalar(&param, &params->rinvMeas, sizeof(float));
    CupvaCmdProgramGetParameter(&m_prog, &param, "lambdaInit");
    CupvaParameterSetValueScalar(&param, &params->lambdaInit, sizeof(float));
    CupvaCmdProgramGetParameter(&m_prog, &param, "maxIterations");
    CupvaParameterSetValueScalar(&param, &params->maxIterations, sizeof(int32_t));

    if (CupvaCmdProgramUpdateDataFlows(&m_prog) != CUPVA_ERROR_NONE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Update DataFlow failed");
    }

    cupvaCmd_t const *submitCmds[1] = {&m_prog};
    cupvaCmdStatus_t cmdStatus[1]   = {NULL};

    if constexpr (std::is_same_v<StreamType, cupvaStream_t>)
    {
        if (CupvaStreamSubmit(stream, submitCmds, cmdStatus, 1, CUPVA_IN_ORDER, -1, -1) != CUPVA_ERROR_NONE)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Submit failed");
        }
    }
    else if constexpr (std::is_same_v<StreamType, cudaStream_t>)
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
        static_assert(!sizeof(StreamType), "Unsupported stream type for BundleAdjustment operator");
    }
}

} // namespace pva_solutions::priv

NVCVStatus pvaBundleAdjustmentCreate(
    NVCVOperatorHandle *handle, NVCVTensorRequirements const *const rigPoses,
    NVCVTensorRequirements const *const points3d, NVCVTensorRequirements const *const measurements,
    NVCVTensorRequirements const *const cameraBitmask, NVCVTensorRequirements const *const rig2cam,
    NVCVTensorRequirements const *const updatedRigPoses, NVCVTensorRequirements const *const updatedPoints3d,
    NVCVTensorRequirements const *const finalCost, NVCVTensorRequirements const *const iterationsUsed,
    PvaBundleAdjustmentConfig const *const config)
{
    return nvcv::ProtectCall([&] {
        if (GetHardwareInfo().pvaGen == PVA_GEN2)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "BundleAdjustment operator is not supported on PVA GEN2");
        }
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        if (rigPoses == nullptr || points3d == nullptr || measurements == nullptr || cameraBitmask == nullptr ||
            rig2cam == nullptr || updatedRigPoses == nullptr || updatedPoints3d == nullptr || finalCost == nullptr ||
            iterationsUsed == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to tensor requirements must not be NULL");
        }
        if (config == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Pointer to config must not be NULL");
        }
        *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::BundleAdjustment(
            rigPoses, points3d, measurements, cameraBitmask, rig2cam, updatedRigPoses, updatedPoints3d, finalCost,
            iterationsUsed, config));
    });
}

template<typename StreamType>
NVCVStatus BundleAdjustmentSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle rigPoses,
                                      NVCVTensorHandle points3d, NVCVTensorHandle measurements,
                                      NVCVTensorHandle cameraBitmask, NVCVTensorHandle rig2cam,
                                      NVCVTensorHandle updatedRigPoses, NVCVTensorHandle updatedPoints3d,
                                      NVCVTensorHandle finalCost, NVCVTensorHandle iterationsUsed,
                                      PvaBundleAdjustmentParams const *params)
{
    return nvcv::ProtectCall([&] {
        nvcv::TensorWrapHandle poses(rigPoses), pts3d(points3d), meas(measurements), bitmask(cameraBitmask),
            r2c(rig2cam), updPoses(updatedRigPoses), updPts3d(updatedPoints3d), cost(finalCost), iters(iterationsUsed);
        pva_solutions::priv::ToDynamicRef<pva_solutions::priv::BundleAdjustment>(handle)(
            stream, poses, pts3d, meas, bitmask, r2c, updPoses, updPts3d, cost, iters, params);
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus BundleAdjustmentSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle,
                                                              NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle,
                                                              NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle,
                                                              NVCVTensorHandle, NVCVTensorHandle,
                                                              PvaBundleAdjustmentParams const *);
template NVCVStatus BundleAdjustmentSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle,
                                                             NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle,
                                                             NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle,
                                                             NVCVTensorHandle, NVCVTensorHandle,
                                                             PvaBundleAdjustmentParams const *);

// C++ overload implementations
NVCVStatus pvaBundleAdjustmentSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle rigPoses,
                                     NVCVTensorHandle points3d, NVCVTensorHandle measurements,
                                     NVCVTensorHandle cameraBitmask, NVCVTensorHandle rig2cam,
                                     NVCVTensorHandle updatedRigPoses, NVCVTensorHandle updatedPoints3d,
                                     NVCVTensorHandle finalCost, NVCVTensorHandle iterationsUsed,
                                     PvaBundleAdjustmentParams const *params)
{
    return BundleAdjustmentSubmitImpl(handle, stream, rigPoses, points3d, measurements, cameraBitmask, rig2cam,
                                      updatedRigPoses, updatedPoints3d, finalCost, iterationsUsed, params);
}

NVCVStatus pvaBundleAdjustmentSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle rigPoses,
                                     NVCVTensorHandle points3d, NVCVTensorHandle measurements,
                                     NVCVTensorHandle cameraBitmask, NVCVTensorHandle rig2cam,
                                     NVCVTensorHandle updatedRigPoses, NVCVTensorHandle updatedPoints3d,
                                     NVCVTensorHandle finalCost, NVCVTensorHandle iterationsUsed,
                                     PvaBundleAdjustmentParams const *params)
{
    return BundleAdjustmentSubmitImpl(handle, stream, rigPoses, points3d, measurements, cameraBitmask, rig2cam,
                                      updatedRigPoses, updatedPoints3d, finalCost, iterationsUsed, params);
}
