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

#ifndef DOA_ANGLE_FFT_PARAM_H
#define DOA_ANGLE_FFT_PARAM_H

typedef enum
{
    ANGLE_FFT_SIZE_8    = 8,
    ANGLE_FFT_SIZE_16   = 16,
    ANGLE_FFT_SIZE_32   = 32,
    ANGLE_FFT_SIZE_64   = 64,
    ANGLE_FFT_SIZE_128  = 128,
    ANGLE_FFT_SIZE_256  = 256,
    ANGLE_FFT_SIZE_512  = 512,
    ANGLE_FFT_SIZE_1024 = 1024,
} DoaAngleFFTSize;

#define COMPLEX_ELEMENT_NUM (2)
#define KERNEL_BATCH (8)
#define TILE_BATCH (ANGLE_FFT_SIZE_32)

#define MAX_SUPPORTED_TARGET_COUNT (8192)

#define MAX_AZIMUTH_FFT_SIZE (ANGLE_FFT_SIZE_128)
#define MAX_ELEVATION_FFT_SIZE (ANGLE_FFT_SIZE_32)

#define IN_TILE_WIDTH (MAX_ELEVATION_FFT_SIZE * COMPLEX_ELEMENT_NUM * 2)

/* Most detections packed into one FFT batch (see fft_groups_for); a power of two dividing OUTPUT_BATCH_SIZE. */
#define MAX_FFT_GROUPS (8)

/* Rows of real snapshot data per detection. The windowing read region is much taller than this, but
 * everything outside these rows meets a zero window coefficient, so slots may overlap. */
#define IN_SLOT_HEIGHT (8)

#define IN_TILE_WIDTH_PAD (IN_TILE_WIDTH + (1 * COMPLEX_ELEMENT_NUM))

/* find_max_val_loc writes a single peak. */
#define PEAK_INDEX_BUF_SIZE (1)

#define TILE_WIDTH (MAX_ELEVATION_FFT_SIZE * COMPLEX_ELEMENT_NUM)
#define TILE_HEIGHT (MAX_AZIMUTH_FFT_SIZE)

#define IN_SLOT_SIZE (IN_SLOT_HEIGHT * IN_TILE_WIDTH_PAD)

/* Room past the last slot's rows for the span its windowing reads, IN_SLOT_HEIGHT * pitch * az / 16.
 * The supported shapes are 32x8 (G = 8), 64x16 (G = 8) and 128x32 (G = 2); the largest
 * (G - 1) slots plus that span is at 64x16. The host checks the bound for the shape it configures. */
#define IN_TILE_READ_SPAN (IN_SLOT_HEIGHT * IN_TILE_WIDTH_PAD * (ANGLE_FFT_SIZE_64 / 16))
/* One ping-pong group. The two groups live in different banks: with the FFT working buffers at
 * FFT_BUF_GROUPS = 2 no single bank has room for both halves alongside a working buffer. */
#define IN_GROUP_BUF_SIZE ((MAX_FFT_GROUPS * IN_SLOT_SIZE) + IN_TILE_READ_SPAN - IN_SLOT_SIZE)

/* Agen configurations per FFT stage set. The largest shape needs 14 for azimuth (3 radix-4 stages
 * of 4 agens plus 2 for the digit reverse) and 10 for elevation (2 stages plus 2). */
#define FFT_CFGS_PER_SET (16)

/* The working buffers are sized for FFT_BUF_GROUPS grids of the maximum shape, which sets the limit on the packing factor. */
#define FFT_BUF_GROUPS (2)

/* The buffers hold both layouts the packed path uses: the azimuth grid (az rows of G*el complex)
 * and the transposed grid the digit reverse produces (G*el rows of az complex). The azimuth form is
 * the larger of the two at the maximum shape. */
#define FFT_BUF_SIZE (((COMPLEX_ELEMENT_NUM * FFT_BUF_GROUPS * MAX_ELEVATION_FFT_SIZE) + 2) * MAX_AZIMUTH_FFT_SIZE)

/* Shared by host and device; detections packed per azimuth FFT batch. */
static inline int32_t fft_groups_for(int32_t az, int32_t el)
{
    const int32_t groups = (FFT_BUF_GROUPS * MAX_AZIMUTH_FFT_SIZE * MAX_ELEVATION_FFT_SIZE) / (az * el);
    return (groups > MAX_FFT_GROUPS) ? MAX_FFT_GROUPS : ((groups < 1) ? 1 : groups);
}

#define TILE_WIDTH_PAD (TILE_WIDTH + (1 * COMPLEX_ELEMENT_NUM))
#define TILE_HEIGHT_PAD (TILE_HEIGHT + (1 * COMPLEX_ELEMENT_NUM))

#define OUTPUT_BATCH_SIZE (8)
#define OUT_TARGET_ANGLE_TILE_WIDTH (OUTPUT_BATCH_SIZE)
#define OUT_TARGET_ANGLE_TILE_PITCH (OUTPUT_BATCH_SIZE)
#define OUT_TARGET_ANGLE_TILE_HEIGHT (3)

#define OUT_TARGET_INDEX_MAP_TILE_WIDTH (OUTPUT_BATCH_SIZE)
#define OUT_TARGET_INDEX_MAP_TILE_PITCH (OUTPUT_BATCH_SIZE)
#define OUT_TARGET_INDEX_MAP_TILE_HEIGHT (1)

#define OUT_TARGET_COUNT_TILE_WIDTH (1)
#define OUT_TARGET_COUNT_TILE_PITCH (1)
#define OUT_TARGET_COUNT_TILE_HEIGHT (1)

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

#define RADIX_NUM (4)

#define DOA_TWIDDLE_SIZE_CALCULATE(SIZE) (((SIZE) / RADIX_NUM) * (RADIX_NUM - 1) * COMPLEX_ELEMENT_NUM)

#define TWIDDLE_SIZE \
    (DOA_TWIDDLE_SIZE_CALCULATE(MAX_AZIMUTH_FFT_SIZE) + DOA_TWIDDLE_SIZE_CALCULATE(MAX_ELEVATION_FFT_SIZE))

#define WINDOW_SIZE (MAX_AZIMUTH_FFT_SIZE * MAX_ELEVATION_FFT_SIZE)

#define ANGLE_BINS_SIZE (MAX_AZIMUTH_FFT_SIZE + MAX_ELEVATION_FFT_SIZE)

#define TRUNCATE_QBITS (16)
#define TWIDDLE_QBITS (30)
#define WINDOW_QBITS (30)

#define POWER_QBITS (20)

#define ANGLE_QBITS (16)
#define ANGLE_SCALE (1 << ANGLE_QBITS)

#define ADDITIONAL_SHIFT_QBITS (2)

#endif // DOA_ANGLE_FFT_PARAM_H