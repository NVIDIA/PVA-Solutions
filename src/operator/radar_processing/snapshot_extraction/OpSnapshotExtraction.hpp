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
 * @file OpSnapshotExtraction.hpp
 *
 * @brief Defines the private C++ Class for the Snapshot Extraction operation.
 */

#ifndef PVA_SOLUTIONS_PRIV_SNAPSHOT_EXTRACTION_HPP
#define PVA_SOLUTIONS_PRIV_SNAPSHOT_EXTRACTION_HPP

#include "IOperator.hpp"

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Tensor.hpp>

#include <OpSnapshotExtraction.h>
#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>

namespace pva_solutions::priv {

template<typename TI>
class SnapshotExtraction final : public IOperator
{
public:
    static constexpr PVASnapshotExtractionParams kDefaultParams{4, PVA_FFT_LAYOUT_RANGE_RX_DOPPLER};

    using ExecDataKey = std::tuple<bool>; // bool: useGSDF
    using ExecKey     = std::tuple<bool, cupva::impl::Context *>;

    explicit SnapshotExtraction(NVCVTensorRequirements const *const foldedDetectionListTensorRequirements,
                                NVCVTensorRequirements const *const ddmDopplerOffsetsTensorRequirements,
                                NVCVTensorRequirements const *const nciRxTensorRequirements,
                                NVCVTensorRequirements const *const rangeDopplerMapTensorRequirements,
                                NVCVTensorRequirements const *const detectionListTensorRequirements,
                                NVCVTensorRequirements const *const snapshotsTensorRequirements,
                                NVCVTensorRequirements const *const calibrationWeightsTensorRequirements,
                                PVASnapshotExtractionParams const *const params);

    ~SnapshotExtraction();

    template<typename S>
    void operator()(S stream, const nvcv::Tensor &detectionCountTensor, const nvcv::Tensor &foldedDetectionListTensor,
                    const nvcv::Tensor &ddmDopplerOffsetsTensor, const nvcv::Tensor &nciRxTensor,
                    const nvcv::Tensor &rangeDopplerMapTensor, const nvcv::Tensor &calibrationWeightsTensor,
                    const nvcv::Tensor &detectionListTensor, const nvcv::Tensor &snapshotsTensor);

private:
    NVCVTensorRequirements m_foldedDetectionListTensorRequirements;
    NVCVTensorRequirements m_ddmDopplerOffsetsTensorRequirements;
    NVCVTensorRequirements m_nciRxTensorRequirements;
    NVCVTensorRequirements m_rangeDopplerMapTensorRequirements;
    NVCVTensorRequirements m_detectionListTensorRequirements;
    NVCVTensorRequirements m_snapshotsTensorRequirements;
    NVCVTensorRequirements m_calibrationWeightsTensorRequirements;
    bool m_useCalibrationWeights;
    PVASnapshotExtractionParams m_params;
    bool m_useGSDFForRDM;

    // PVA offset pointers for input/output tensors (used in dataflows)
    cupva::OffsetPointer<int32_t> m_foldedDetectionListOp;
    cupva::OffsetPointer<uint32_t> m_nciRxOp;
    cupva::OffsetPointer<TI> m_rangeDopplerMapOp;
    cupva::OffsetPointer<int32_t> m_detectionListOp;
    cupva::OffsetPointer<TI> m_snapshotsOp;
    cupva::OffsetPointer<int32_t> m_calibrationWeightsOp; // For user-provided calibration weights
    cupva::OffsetPointer<int32_t> m_DetectionCountOp;

    // Coefficient buffers allocated on PVA memory
    int32_t *m_ddmaWeights_d; // DDMA weights (used when calibration not provided)
    int32_t *m_ddmSpace_d;    // DDM space [numTx]

    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;

    // Mutexes to protect shared Executable maps
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_SnapshotExtractionExecutables;
    static std::shared_ptr<cupva::Executable> GetSnapshotExtractionExec(ExecDataKey execDataKey);
};

} // namespace pva_solutions::priv

// Internal implementation function for submit operations
// Used by macro-generated __cupva/__cuda wrappers
template<typename StreamType>
NVCVStatus SnapshotExtractionSubmitImpl(NVCVOperatorHandle handle, StreamType stream,
                                        NVCVTensorHandle const detectionCount,
                                        NVCVTensorHandle const foldedDetectionList,
                                        NVCVTensorHandle const ddmDopplerOffsets, NVCVTensorHandle const nciRx,
                                        NVCVTensorHandle const rangeDopplerMap,
                                        NVCVTensorHandle const calibrationWeights, NVCVTensorHandle detectionList,
                                        NVCVTensorHandle snapshots);

#endif // PVA_SOLUTIONS_PRIV_SNAPSHOT_EXTRACTION_HPP
