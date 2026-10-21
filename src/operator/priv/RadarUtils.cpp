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

#include "RadarUtils.hpp"

#include "RadarDefinitions.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace pva_solutions::priv {
// Helper function to determine the number of bits needed to represent n
inline int32_t bit_length(int32_t n)
{
    if (n <= 1)
    {
        return 0;
    }
    return (int32_t)ceil(log2(n));
}

inline bool is_power_of_2(int32_t n)
{
    return ((n > 0) && ((n & (n - 1)) == 0));
}

FFTFactorization FactorizeFFTSize(int32_t n, bool transposeOutput, bool hasPostProcessing, int32_t outGroupsPerTile,
                                  int32_t outPitchOverride)
{
    FFTFactorization result = {n, {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, {0}, {0}, 0, 0, {0}, 0};
    int32_t temp_n          = n;
    // The digit-reverse output strides are multiples of this pitch.
    int32_t oPitch =
        (outPitchOverride > 0) ? outPitchOverride : ((transposeOutput == true) ? 2 : (16 * outGroupsPerTile));
    int32_t numRadixs = 0;
    int32_t groupSize = 1;
    std::vector<int32_t> iters;

    if (n < 2 || (n > 1024) || ((n == 1024) && (transposeOutput || hasPostProcessing)))
    {
        result.isValid = 0;
        return result;
    }

    result.groups[RADIX5_GROUP_IDX].groupSize = groupSize;
    // Count factors of 5 first
    while ((temp_n > 0) && (temp_n % 5 == 0))
    {
        result.groups[RADIX5_GROUP_IDX].numStages++;
        temp_n /= 5;
        iters.push_back(5);
    }

    if (result.groups[RADIX5_GROUP_IDX].numStages > 0)
    {
        groupSize *= pow(5, result.groups[RADIX5_GROUP_IDX].numStages);
    }

    // Count factors of 3
    while ((temp_n > 0) && (temp_n % 3 == 0))
    {
        result.groups[RADIX3_GROUP_IDX].numStages++;
        temp_n /= 3;
        iters.push_back(3);
    }

    if (result.groups[RADIX3_GROUP_IDX].numStages > 0)
    {
        result.groups[RADIX3_GROUP_IDX].groupSize = groupSize;
        groupSize *= pow(3, result.groups[RADIX3_GROUP_IDX].numStages);
    }

    // Remaining part should be a power of 2
    result.isValid = (is_power_of_2(temp_n)) ? 1 : 0;
    if ((temp_n > 0) && (result.isValid == 1))
    {
        int32_t log2n           = bit_length(temp_n);
        RadixGroup &radix4Group = result.groups[RADIX4_GROUP_IDX];
        radix4Group.numStages   = log2n / 2;
        radix4Group.isOdd       = (log2n & 1) == 1;
        radix4Group.groupSize   = groupSize;
        for (int32_t i = 0; i < radix4Group.numStages; i++)
        {
            iters.push_back(4);
        }
        if (radix4Group.isOdd == 1)
        {
            iters.push_back(2);
        }
    }

    result.isValid &= ((iters.size() <= 6)) ? 1 : 0;

    // Validate total number of stages
    if (result.isValid)
    {
        temp_n                = n;
        int32_t iterArraySize = iters.size();
        for (int32_t i = 0; i < iterArraySize; i++)
        {
            int32_t iter = iters.back();
            temp_n /= iter;
            result.iters[i]   = iter;
            result.strides[i] = temp_n * oPitch;
            iters.pop_back();
        }

        const int32_t twiddle3Size =
            (result.groups[RADIX3_GROUP_IDX].numStages > 0)
                ? ALIGNED_SIZE(TWIDDLE_SIZE_COMPLEX_FFT_CALCULATE(result.n, RADIX3_NUM) + 2 * COMPLEX_ELEMENT_NUM)
                : 0;
        const int32_t twiddle4Size = (result.groups[RADIX4_GROUP_IDX].numStages > 0)
                                         ? ALIGNED_SIZE(TWIDDLE_SIZE_COMPLEX_FFT_CALCULATE(result.n, RADIX4_NUM))
                                         : 0;
        const int32_t twiddle5Size =
            (result.groups[RADIX5_GROUP_IDX].numStages > 0)
                ? ALIGNED_SIZE(TWIDDLE_SIZE_COMPLEX_FFT_CALCULATE(result.n, RADIX5_NUM) + 2 * COMPLEX_ELEMENT_NUM)
                : 0;
        const int32_t twiddlePPSizePerCoef = (hasPostProcessing) ? ALIGNED_SIZE(result.n * COMPLEX_ELEMENT_NUM) : 0;
        result.twiddleOffsets[0]           = twiddle3Size;
        result.twiddleOffsets[1]           = twiddle3Size + twiddle4Size;
        result.twiddleOffsets[2]           = twiddle3Size + twiddle4Size + twiddle5Size;
        result.twiddleSize = (twiddle3Size + twiddle4Size + twiddle5Size + twiddlePPSizePerCoef * 2 + 63) & ~63;
        result.twiddlePPSizePerCoef = twiddlePPSizePerCoef;
        result.twiddleSize          = (result.twiddleSize > 64) ? result.twiddleSize : 64;
    }

    return result;
}

RadixGroupIdx getRadixGroupIdx(int32_t radix)
{
    switch (radix)
    {
    case RADIX3_NUM:
        return RADIX3_GROUP_IDX;
    case RADIX4_NUM:
        return RADIX4_GROUP_IDX;
    case RADIX5_NUM:
        return RADIX5_GROUP_IDX;
    default:
        throw std::runtime_error("Unsupported radix " + std::to_string(radix));
    }
}

void generateComplexFFTTwiddleIntraStage(int32_t *twiddle, std::vector<int32_t> &twiddle_interleaved,
                                         FFTFactorization &factorization, int32_t radix)
{
    const int32_t n        = factorization.n;
    const int32_t groupIdx = getRadixGroupIdx(radix);
    const int32_t stride   = factorization.groups[groupIdx].groupSize;
    int32_t dist           = n / (radix * stride);

    for (int32_t i = 0; i < dist; i++)
    {
        for (int32_t j = 1; j < radix; j++)
        {
            const int32_t tidx = i * j * stride;

            // Read twiddle factors
            *twiddle++ = twiddle_interleaved[2 * tidx];
            *twiddle++ = twiddle_interleaved[2 * tidx + 1];
        }
    }
}

void generateComplexFFTTwiddleRadix3(int32_t *twiddle, std::vector<int32_t> &twiddle_interleaved,
                                     FFTFactorization &factorization)
{
    const int32_t n      = factorization.n;
    const int32_t stride = n / 3;

    for (int32_t i = 1; i < 3; i++)
    {
        const int32_t tidx = i * stride;

        // Read twiddle factors
        *twiddle++ = twiddle_interleaved[2 * tidx];
        *twiddle++ = twiddle_interleaved[2 * tidx + 1];
    }
}

void generateComplexFFTTwiddleRadix5(int32_t *twiddle, std::vector<int32_t> &twiddle_interleaved,
                                     FFTFactorization &factorization)
{
    const int32_t n      = factorization.n;
    const int32_t stride = n / 5;

    for (int32_t i = 1; i < 5; i++)
    {
        const int32_t tidx = i * stride;

        // Read twiddle factors
        *twiddle++ = twiddle_interleaved[2 * tidx];
        *twiddle++ = twiddle_interleaved[2 * tidx + 1];
    }
}

void generateComplexFFTTwiddlePostProcess(int32_t *twiddle, int32_t qbitsTwiddle, int32_t qbitsTrunc,
                                          FFTFactorization &factorization)
{
    const int32_t n    = factorization.n;
    int16_t *twiddle_b = (int16_t *)(twiddle + factorization.twiddlePPSizePerCoef);
    // coef A
    for (int32_t i = 0; i < n; i++)
    {
        double angle = 2.0f * M_PI * i / (2 * n);
        *twiddle++   = (int32_t)round(0.5f * (1.0f - sin(angle)) * (1UL << qbitsTwiddle));  // real
        *twiddle++   = (int32_t)round(0.5f * (-1.0f * cos(angle)) * (1UL << qbitsTwiddle)); // imag
    }

    // coef B
    for (int32_t i = 0; i < n; i++)
    {
        double angle = 2.0f * M_PI * i / (2 * n);
        *twiddle_b++ = (int16_t)round(0.5f * (1.0f * cos(angle)) *
                                      (1UL << std::min(qbitsTwiddle - qbitsTrunc, qbitsTwiddle))); // imag
        *twiddle_b++ = (int16_t)round(0.5f * (1.0f + sin(angle)) *
                                      (1UL << std::min(qbitsTwiddle - qbitsTrunc, qbitsTwiddle))); // real
    }
}

} // namespace pva_solutions::priv
