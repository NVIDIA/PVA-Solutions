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

#ifndef RADAR_PIPELINE_TENSORS_HPP
#define RADAR_PIPELINE_TENSORS_HPP

#include <PvaAllocator.h>
#include <PvaOperatorTypes.h>
#include <nvcv/Fwd.h>
#include <nvcv/alloc/Allocator.h>

#include <vector>

struct RadarPipelineContext; // full definition in radar_pipeline_context.hpp; .cpp includes both

// ============================================================================
// RadarPipelineTensorRequirements - Holds tensor requirements without allocation
// ============================================================================

class RadarPipelineTensorRequirements
{
public:
    RadarPipelineTensorRequirements(const int32_t Ns, const int32_t NofRx, const int32_t Nr, const int32_t NofRangeBins,
                                    const int32_t repeatFold, const int32_t NofTx)
    {
        createTensorRequirements(Ns, NofRx, Nr, NofRangeBins, repeatFold, NofTx);
    }

    void createTensorRequirements(const int32_t Ns, const int32_t NofRx, const int32_t Nr, int32_t NofRangeBins,
                                  const int32_t repeatFold, const int32_t NofTx);

    // Tensor requirements
    NVCVTensorRequirements inRangeFFTRequirements; // used by pvaRangeFFTCreate and validate(); no tensor allocated
    NVCVTensorRequirements outRangeFFTRequirements;
    NVCVTensorRequirements winRangeFFTRequirements;
    NVCVTensorRequirements inDopplerFFTRequirements;
    NVCVTensorRequirements winDopplerFFTRequirements;
    NVCVTensorRequirements outDopplerFFTRequirements;
    NVCVTensorRequirements outNciRxTensorRequirements;
    NVCVTensorRequirements outNciFinalTensorRequirements;
    NVCVTensorRequirements outNoiseEstimateTensorRequirements;
    NVCVTensorRequirements inNciTensorRequirements;
    NVCVTensorRequirements outPeakCountTensorRequirements;
    NVCVTensorRequirements outPeakIndexTensorRequirements;
    NVCVTensorRequirements outPeakSnapTensorRequirements;
    NVCVTensorRequirements inDOACalibVectorTensorRequirements;
    NVCVTensorRequirements outDOATensorRequirements;
    std::vector<const NVCVTensorRequirements *> outNciTensorReqs;
    std::vector<const NVCVTensorRequirements *> inDOATensorReqs;
};

// ============================================================================
// RadarPipelineTensors - Manages all tensor allocations
// ============================================================================

class RadarPipelineTensors
{
public:
    /** Owned by @ref RadarPipelineContext::tensorReqs; must outlive this object. */
    RadarPipelineTensorRequirements &m_tensorReqs;

    RadarPipelineContext *m_ctx;

    /** @a ctx->tensorReqs must be non-null. */
    RadarPipelineTensors(RadarPipelineContext *ctx);

    ~RadarPipelineTensors()
    {
        cleanupRadarPipelineTensors();
    }

    void initRadarPipelineTensors();
    void cleanupRadarPipelineTensors();
    void createRadarPipelineTensors();

    // Allocator handle
    NVCVAllocatorHandle m_allocatorHandle;

    // Range FFT tensors
    NVCVTensorHandle inRangeFFTTensorHandle; // set per-call in radar_pipeline_process(); not owned
    NVCVTensorHandle winRangeFFTTensorHandle;
    NVCVTensorHandle outRangeFFTTensorHandle;
    NVCVTensorHandle refRangeFFTTensorHandle;

    // Doppler FFT tensors
    NVCVTensorHandle inDopplerFFTTensorHandle;
    NVCVTensorHandle winDopplerFFTTensorHandle;
    NVCVTensorHandle outDopplerFFTTensorHandle;
    NVCVTensorHandle refDopplerFFTTensorHandle;

    // NCI tensors
    NVCVTensorHandle inNciTensorHandle;
    NVCVTensorHandle outNciRxTensorHandle;
    NVCVTensorHandle outNciFinalTensorHandle;
    NVCVTensorHandle outNoiseEstimateTensorHandle;
    NVCVTensorHandle refNciRxTensorHandle;
    NVCVTensorHandle refNciFinalTensorHandle;
    NVCVTensorHandle refNoiseEstimateTensorHandle;

    // Peak Detection tensors
    NVCVTensorHandle outPeakCountTensorHandle;
    NVCVTensorHandle outPeakIndexTensorHandle;
    NVCVTensorHandle outPeakSnapTensorHandle;
    NVCVTensorHandle refPeakCountTensorHandle;
    NVCVTensorHandle refPeakIndexTensorHandle;
    NVCVTensorHandle refPeakSnapTensorHandle;

    // DOA tensors
    NVCVTensorHandle inDOACalibVectorTensorHandle;
    NVCVTensorHandle outDOATensorHandle;
    NVCVTensorHandle refDOATensorHandle;

    // GP parameters (needed for reference implementation)
    PVARadarGP GP;
};

#endif // RADAR_PIPELINE_TENSORS_HPP
