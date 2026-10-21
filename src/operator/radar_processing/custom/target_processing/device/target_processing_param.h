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

#ifndef TARGET_PROCESSING_PARAM_H
#define TARGET_PROCESSING_PARAM_H

#include <stdint.h>

/**
 * @brief RV Decoupling parameters structure
 */

typedef struct
{
    int32_t enableRvDecoupling;           // Enable RV decoupling mode (0 = standard, 1 = RV decoupling)
    int32_t enableQuadraticInterpolation; // Enable quadratic interpolation for sub-bin accuracy
    float f_fast_delta;                   // Delta_f_fast = fs / numSamples
    float f_slow_delta;                   // Delta_f_slow = 1.0 / (PRI * numChirps)
    float contR;                          // Range coupling constant: c / (2 * slope)
    float contVFast;                      // Velocity fast constant for FMCW coupling
    float contVSlow;                      // Velocity slow constant for FMCW coupling
    float velocityAmbHigh;                // Velocity ambiguity high bound (negative, in m/s)
    float velocityAmbLow;                 // Velocity ambiguity low bound (positive, in m/s)
    float tempVCoeff[5];                  // 5 hypothesis coefficients for velocity disambiguation
} rv_decoupling_params_t;

// NCI tile parameters for quadratic interpolation (3x3 neighborhood around peak)
#define NCI_NODE_WIDTH (3)
#define NCI_NODE_HEIGHT (3)
// NCI_NODE_SIZE must be 17 (16+1) to align with transposed load lane pitch
// For 32-bit data with lane_ofst=1: pitch = 64*1 + 4 = 68 bytes = 17 elements
// This allows efficient vectorized loading across 16 tiles
#define NCI_NODE_SIZE (16 + 1)
#define NUM_NCI_NODES (32)

// Maximum number of targets that can be processed
#define MAX_SUPPORTED_TARGET_COUNT (8192)

// Processing batch size
#define BATCH_SIZE (64)

// Detection list tile parameters
#define DETECTION_LIST_TILE_WIDTH (2)
#define DETECTION_LIST_TILE_HEIGHT (MAX_SUPPORTED_TARGET_COUNT)
#define DETECTION_LIST_TILE_PITCH (2)

// Target index map tile parameters
#define TARGET_INDEX_MAP_TILE_WIDTH (BATCH_SIZE)
#define TARGET_INDEX_MAP_TILE_HEIGHT (1)
#define TARGET_INDEX_MAP_TILE_PITCH (BATCH_SIZE)

// Target angles tile parameters (azimuth, elevation, power)
#define TARGET_ANGLES_TILE_WIDTH (BATCH_SIZE)
#define TARGET_ANGLES_TILE_HEIGHT (3)
#define TARGET_ANGLES_TILE_PITCH (BATCH_SIZE)

// DDM Doppler offsets tile parameters
#define DDM_DOPPLER_OFFSETS_TILE_WIDTH (32)
#define DDM_DOPPLER_OFFSETS_TILE_HEIGHT (1)
#define DDM_DOPPLER_OFFSETS_TILE_PITCH (32)

// Target list output tile parameters (velocity, range, angles, XYZ, power)
#define TARGET_LIST_TILE_WIDTH (BATCH_SIZE)
#define TARGET_LIST_TILE_HEIGHT (8)
#define TARGET_LIST_TILE_PITCH (BATCH_SIZE)

// Constants
#ifndef PI_CONST_F
#    define PI_CONST_F 3.14159265358979323846f
#endif

#endif // TARGET_PROCESSING_PARAM_H
