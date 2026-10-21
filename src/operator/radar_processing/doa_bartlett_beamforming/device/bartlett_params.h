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

#ifndef PVA_SOLUTIONS_PRIV_BARTLETT_PARAMS_H
#define PVA_SOLUTIONS_PRIV_BARTLETT_PARAMS_H

#include <stdint.h>

// Fixed-point format definitions
#define STEERING_QBITS (15)
#define STEERING_SCALE (1 << STEERING_QBITS)
#define SNAPSHOT_QBITS (20)
#define SNAPSHOT_SCALE (1 << SNAPSHOT_QBITS)
#define BEAMFORMING_SHIFT (12)
#define BEAMFORMING_ROUND (1 << (BEAMFORMING_SHIFT - 1))
#define BEAMFORMING_QBITS (23)

// Configuration limits
#define MAX_DETECTIONS (8192)
#define MAX_VIRTUAL_CHANNELS (64)
// Full 2D power spectrum buffer size (also used for L2 steering vector allocation)
// Separable mode uses SEPARABLE_POWER_SPECTRUM_SIZE instead
#define FULL2D_POWER_SPECTRUM_SIZE (2048)
#define ELEMENT_NUM_OF_COMPLEX (2)

// Processing parameters
#define SNAPSHOT_BATCH_SIZE (8)

// Steering vector batch: 128 angles × 64 VCs × 2 × 2 bytes = 32 KB
#define STEERING_ANGLE_BATCH_SIZE (128)

// Step 1 pre-loaded steering vectors: max azimuths for separable mode
// 256 azimuths × 64 VCs × 2 × 2 bytes = 64 KB (single buffer, pre-loaded once)
// Constraint: numAzimuthBins must be <= STEP1_AZ_BATCH_SIZE (pre-loaded in one DMA)
#define STEP1_AZ_BATCH_SIZE (256)

// Maximum azimuth neighborhood for separable mode (±N bins around Step 1 peak)
// Smaller values = faster Step 2, larger values = better peak refinement
#define MAX_AZ_NEIGHBORHOOD (2)

// Separable: 2*numAz (azimuth half, double buffered for the step-1 lookahead) + (2N+1)*numEl.
// Constraint: 2*numAz + (2*N+1)*numEl <= SEPARABLE_POWER_SPECTRUM_SIZE
// Constraint: numElevationBins <= 255 for separable mode (DMA pitch limit)
#define SEPARABLE_POWER_SPECTRUM_SIZE (4096)

// Bartlett VMEM buffer dimensions used by bartlett_top.c
#define STEERING_BATCH_WIDTH (MAX_VIRTUAL_CHANNELS * ELEMENT_NUM_OF_COMPLEX)
#define STEERING_BATCH_HEIGHT (STEERING_ANGLE_BATCH_SIZE)
#define SNAPSHOT_BUFFER_WIDTH (MAX_VIRTUAL_CHANNELS * ELEMENT_NUM_OF_COMPLEX)
#define SNAPSHOT_BUFFER_HEIGHT (SNAPSHOT_BATCH_SIZE)

// Output: azimuth, elevation, [power] (2 or 3 floats per detection)
// SOA layout: [numRows][maxTargets] - numRows rows of maxTargets floats each
#define ANGLES_NUM_ROWS_WITH_POWER (3)
#define ANGLES_NUM_ROWS_WITHOUT_POWER (2)
#define ANGLES_ROW_AZ (0)
#define ANGLES_ROW_EL (1)
#define ANGLES_ROW_POWER (2)

// DMA stride helpers (bytes per detection)
#define SNAPSHOT_STRIDE_BYTES(numVC) ((numVC)*ELEMENT_NUM_OF_COMPLEX * sizeof(int32_t))

#ifndef PI_CONST_F
#    define PI_CONST_F 3.14159265358979323846f
#endif

typedef struct
{
    int32_t numTx;
    int32_t numRx;
    int32_t numAzimuthBins;
    int32_t numElevationBins;
    int32_t enableInterpolation;
    int32_t enablePowerOutput;
    float azimuthMin;
    float azimuthMax;
    float elevationMin;
    float elevationMax;

    int32_t enableSeparable;
    int32_t azNeighborhood;
    int32_t azFromEl0;
    int32_t elZeroIdx; // Pre-computed index of elevation bin closest to 0° (computed on host)

    uint32_t azChannelMaskLo; // bits 0-31 of azChannelMask (Step 1 VC bitmask)
    uint32_t azChannelMaskHi; // bits 32-63 of azChannelMask (Step 1 VC bitmask)
} BartlettGlobalParams;

#endif // PVA_SOLUTIONS_PRIV_BARTLETT_PARAMS_H
