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

#ifndef RADAR_TRANSPOSE_PARAM_H
#define RADAR_TRANSPOSE_PARAM_H

#include <RadarDefinitions.h>

// Output layout options (matches PVAFFTLayout in PvaOperatorTypes.h)
typedef enum
{
    RADAR_TRANSPOSE_OUTPUT_LAYOUT_DOPPLER_RX_RANGE = 0, // [nDoppler][nRx][nRange]
    RADAR_TRANSPOSE_OUTPUT_LAYOUT_RANGE_RX_DOPPLER = 1, // [nRange][nRx][nDoppler] (identity)
    RADAR_TRANSPOSE_OUTPUT_LAYOUT_RANGE_DOPPLER_RX = 2, // [nRange][nDoppler][nRx]
} RadarTransposeOutputLayout;

// Maximum tensor dimensions supported
#define MAX_NRX (256)
#define MAX_NDOPPLER (1024)
#define MAX_NRANGE (1024)
#define TILE_BATCH (8)

// Number of int32_t lanes in the vector type used by the transpose kernels.
#define RADAR_TRANSPOSE_VECW_INT32 (16)

// CHUNK_RANGE: maximum range bins per VPU iteration for DOPPLER_RX_RANGE layout.
// The host uses 4 when that gives an even number of range chunks; otherwise it uses 2.
// Doppler slicing, rather than the range chunk size, is adjusted to satisfy transfer-size limits.
#define CHUNK_RANGE (4)

// Tile buffer sizes (in int32_t units).
// Each VMEM superbank is 128 KB.  srcBuf and dstBuf each live in their own superbank.
// MAX_TILE_SIZE = TILE_BATCH * MAX_NDOPPLER * COMPLEX_ELEMENT_NUM = 8*1024*2 = 16384 int32 = 64 KB.
// This is the maximum padded tile that can be held in one VMEM superbank (half of 128 KB),
// leaving the other half for the sibling buffer.
#define MAX_TILE_SIZE (TILE_BATCH * MAX_NDOPPLER * COMPLEX_ELEMENT_NUM)

#endif // RADAR_TRANSPOSE_PARAM_H
