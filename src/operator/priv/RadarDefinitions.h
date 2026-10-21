/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/*
 * @file RadarDefinitions.h
 *
 * @brief Defines definitions for radar type's use.
 */

#ifndef RADAR_DEFINITIONS_H
#define RADAR_DEFINITIONS_H

#include <stdint.h>

enum RadixNum
{
    RADIX3_NUM = 3,
    RADIX4_NUM = 4,
    RADIX5_NUM = 5,
};

enum RadixGroupIdx
{
    RADIX3_GROUP_IDX     = 0,
    RADIX4_GROUP_IDX     = 1,
    RADIX5_GROUP_IDX     = 2,
    MAX_NUM_RADIX_GROUPS = 3,
};

enum FFTSize
{
    FFT_SIZE_512  = 512,
    FFT_SIZE_1024 = 1024,
};

#define COMPLEX_ELEMENT_NUM (2)

#define TWIDDLE_SIZE_COMPLEX_FFT_CALCULATE(SIZE, RADIX) ((SIZE / RADIX) * (RADIX - 1) * COMPLEX_ELEMENT_NUM)
#define ALIGNED_SIZE(SIZE) ((SIZE + 63) & ~63)

#define TWIDDLE4_INTRA_STAGE_SIZE (TWIDDLE_SIZE_COMPLEX_FFT_CALCULATE(FFT_SIZE_1024, RADIX4_NUM))
#define TWIDDLE5_INTRA_STAGE_SIZE (TWIDDLE_SIZE_COMPLEX_FFT_CALCULATE(FFT_SIZE_1024, RADIX5_NUM))
#define TWIDDLE4_SIZE ALIGNED_SIZE(TWIDDLE4_INTRA_STAGE_SIZE)
#define TWIDDLE3_SIZE \
    ALIGNED_SIZE(TWIDDLE_SIZE_COMPLEX_FFT_CALCULATE(FFT_SIZE_1024, RADIX3_NUM) + 2 * COMPLEX_ELEMENT_NUM)
#define TWIDDLE5_SIZE ALIGNED_SIZE(TWIDDLE5_INTRA_STAGE_SIZE + 4 * COMPLEX_ELEMENT_NUM)

#define TRUNCATE_QBITS (16)
#define TWIDDLE_QBITS (30)
// Most FFT stages a packed (fused) FFT can have: the digit-reverse agen spends one of its six dimensions per
// stage and needs a free one for the stride between packed groups.
#define FFT_MAX_FUSED_STAGES (5)
#define WINDOW_QBITS (30)

struct RadixGroup
{
    int32_t numStages;
    int32_t groupSize;
    int32_t isOdd;
};

// Helper function to factorize n into powers of 5, 3, and 2 (for mixed-radix)
typedef struct
{
    int32_t n; // FFT size
    RadixGroup groups[MAX_NUM_RADIX_GROUPS];
    int32_t iters[6];                             // Number of iterations for each radix
    int32_t strides[6];                           // Strides for each radix
    int32_t twiddleSize;                          // Total twiddle size
    int32_t twiddlePPSizePerCoef;                 // Twiddle post process size per coef
    int16_t twiddleOffsets[MAX_NUM_RADIX_GROUPS]; // Twiddle offsets for each radix
    int8_t isValid;                               // Whether factorization is valid
} FFTFactorization;

#endif // RADAR_DEFINITIONS_H