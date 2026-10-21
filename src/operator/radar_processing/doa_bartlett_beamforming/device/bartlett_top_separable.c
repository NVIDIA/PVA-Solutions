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

// Separable Bartlett Beamforming VPU ELF

#include "bartlett_top_common.h"

// ============================================================================
// VMEM Declarations
// ============================================================================

VMEM(C, int16_t, steeringVectorsBuffer, RDF_DOUBLE(int16_t, STEERING_BATCH_WIDTH, STEERING_BATCH_HEIGHT));
VMEM(B, float, powerSpectrum, SEPARABLE_POWER_SPECTRUM_SIZE);
VMEM(A, int32_t, snapshotBuffer, RDF_DOUBLE(int32_t, SNAPSHOT_BUFFER_WIDTH, SNAPSHOT_BUFFER_HEIGHT));
VMEM(A, float, anglesOutputBuffer, RDF_DOUBLE(float, SNAPSHOT_BATCH_SIZE, ANGLES_NUM_ROWS_WITH_POWER));
VMEM(B, int32_t, peakAzIdx, SNAPSHOT_BATCH_SIZE);
VMEM(B, int32_t, peakElIdx, SNAPSHOT_BATCH_SIZE);
VMEM(B, float, azY0, SNAPSHOT_BATCH_SIZE);
VMEM(B, float, azY1, SNAPSHOT_BATCH_SIZE);
VMEM(B, float, azY2, SNAPSHOT_BATCH_SIZE);
VMEM(B, float, elY0, SNAPSHOT_BATCH_SIZE);
VMEM(B, float, elY1, SNAPSHOT_BATCH_SIZE);
VMEM(B, float, elY2, SNAPSHOT_BATCH_SIZE);
VMEM(B, float, peakMagnitude, SNAPSHOT_BATCH_SIZE);
VMEM(B, int32_t, targetMap, MAX_DETECTIONS);
VMEM(B, BartlettGlobalParams, GP);
VMEM(B, uint32_t, detectionCount);
VMEM(B, int32_t, maxTargetCount);
VMEM(B, int32_t, targetCount);
VMEM_SEQDF_HANDLER(B, detectionCntSqdfHdl);
VMEM_SEQDF_HANDLER(B, snapshotsSqdfHdl);
VMEM_SEQDF_HANDLER(B, anglesOutputSqdfHdl);
VMEM_SEQDF_HANDLER(B, steeringVectorsSqdfHdl);
VMEM_SEQDF_HANDLER(B, targetCountAndMapSqdfHdl);
VMEM_POINTER(B, snapshotsExtAddr);
VMEM_POINTER(B, anglesOutputExtAddr);
VMEM_POINTER(B, targetCountExtAddr);
VMEM(C, int16_t, step1SteeringBuffer, STEERING_BATCH_WIDTH *STEP1_AZ_BATCH_SIZE);
VMEM_SEQDF_HANDLER(B, step1SteeringSqdfHdl);

// DRAM steering vector pointer (bypass L2 for separable mode)
VMEM_POINTER(B, steeringDramExtAddr);

// ============================================================================
// VPU Main
// ============================================================================

CUPVA_VPU_MAIN()
{
    BartlettVmemPtrs vmem = {
        .GP                       = &GP,
        .detectionCount           = &detectionCount,
        .maxTargetCount           = &maxTargetCount,
        .targetCount              = &targetCount,
        .targetMap                = targetMap,
        .steeringVectorsBuffer    = steeringVectorsBuffer,
        .snapshotBuffer           = snapshotBuffer,
        .powerSpectrum            = powerSpectrum,
        .anglesOutputBuffer       = anglesOutputBuffer,
        .peakAzIdx                = peakAzIdx,
        .peakElIdx                = peakElIdx,
        .azY0                     = azY0,
        .azY1                     = azY1,
        .azY2                     = azY2,
        .elY0                     = elY0,
        .elY1                     = elY1,
        .elY2                     = elY2,
        .peakMagnitude            = peakMagnitude,
        .detectionCntSqdfHdl      = &detectionCntSqdfHdl,
        .snapshotsSqdfHdl         = &snapshotsSqdfHdl,
        .anglesOutputSqdfHdl      = &anglesOutputSqdfHdl,
        .steeringVectorsSqdfHdl   = &steeringVectorsSqdfHdl,
        .targetCountAndMapSqdfHdl = &targetCountAndMapSqdfHdl,
        .snapshotsExtAddr         = &snapshotsExtAddr,
        .anglesOutputExtAddr      = &anglesOutputExtAddr,
        .targetCountExtAddr       = &targetCountExtAddr,
        .step1SteeringBuffer      = step1SteeringBuffer,
        .step1SteeringSqdfHdl     = &step1SteeringSqdfHdl,
        .steeringDramExtAddr      = &steeringDramExtAddr,
    };

    return bartlettRun(&vmem, bartlettSeparableKnl, bartlettInitAgensSeparable);
}
