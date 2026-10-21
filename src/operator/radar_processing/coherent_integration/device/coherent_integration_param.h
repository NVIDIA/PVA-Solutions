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

#ifndef COHERENT_INTEGRATION_PARAM_H
#define COHERENT_INTEGRATION_PARAM_H

#include <stdint.h>

#define CI_KIB (1024U)

#define CI_RX4 (4)
#define CI_RX8 (8)
#define CI_RX12 (12)
#define CI_RX16 (16)

#define CI_STEERING_CAPACITY_BYTES (48U * CI_KIB)

#define CI_RAW_INPUT_BYTES (64U * CI_KIB)
#define CI_CELL_SHIFT_BYTES (2U * CI_KIB)
#define CI_PEAK_POWER_BYTES (16U * CI_KIB)
#define CI_FOLD_ACCUMULATOR_BYTES (8U * CI_KIB)
#define CI_LIMB_BYTES (48U * CI_KIB)
#define CI_OUTPUT_STAGE_BYTES (8U * CI_KIB)
#define CI_COHERENT_INTEGRATION_RX_PING_OFFSET (0U)
#define CI_COHERENT_INTEGRATION_RX_PONG_OFFSET (CI_COHERENT_INTEGRATION_RX_PING_OFFSET + CI_OUTPUT_STAGE_BYTES)
#define CI_COHERENT_INTEGRATION_FINAL_STAGE_OFFSET (CI_COHERENT_INTEGRATION_RX_PONG_OFFSET + CI_OUTPUT_STAGE_BYTES)
#define CI_WORK_BYTES (CI_COHERENT_INTEGRATION_FINAL_STAGE_OFFSET + CI_OUTPUT_STAGE_BYTES)

#define CI_CELL_VECTOR_LANES (16U)
#define CI_INPUT_COMPLEX_BYTES (8U)
#define CI_LIMB_COMPLEX_BYTES (4U)
#define CI_STEERING_COMPLEX_BYTES (4U)
#define CI_MAX_PROCESSING_RANGE_TILE (256U)
/* 64 KiB / (minimum 4 Rx * 8 bytes/complex * 16 cells/vector). */
#define CI_MAX_PROCESSING_CELL_BLOCKS (128U)

/* Values are fixed at operator creation, except the steering-grid fields are filled at submission. */
typedef struct CoherentIntegrationParamsRec
{
    uint32_t numRangeBins;
    uint32_t numRx;
    uint32_t repeatFold;
    uint32_t foldedWidth;
    uint32_t numBeams;
    uint32_t numAzimuthBins;
    uint32_t numElevationBins;

    uint32_t processingRangeTile;
    uint32_t rawCellCapacity;

    uint32_t inputRangeStrideBytes;
    uint32_t inputDopplerStrideBytes;
    uint32_t coherentIntegrationRxRangeStrideBytes;
    uint32_t coherentIntegrationFinalRangeStrideBytes;

    uint32_t termShift;
    uint32_t coherentIntegrationRxShift;
    uint32_t coherentIntegrationFinalShift;
} CoherentIntegrationParams;

#endif /* COHERENT_INTEGRATION_PARAM_H */
