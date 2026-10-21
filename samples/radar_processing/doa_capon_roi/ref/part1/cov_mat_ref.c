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

#include "cov_mat_ref.h"

#include <math.h>

static int32_t realIndex(int32_t matrixSize, int32_t row, int32_t column)
{
    return column * matrixSize - column * (column - 1) / 2 + row - column;
}

static int32_t imagIndex(int32_t matrixSize, int32_t row, int32_t column)
{
    return column * matrixSize - (column + 1) * column / 2 + row - column - 1;
}

void computeFpCovarianceRef(const float *snapshots, int32_t matrixSize, int32_t numSamples,
                            float diagonalLoadingConstant, float *covarianceReal, float *covarianceImag)
{
    int32_t realSize = matrixSize * (matrixSize + 1) / 2;
    int32_t imagSize = matrixSize * (matrixSize - 1) / 2;
    for (int32_t index = 0; index < realSize; ++index)
    {
        covarianceReal[index] = 0.0F;
    }
    for (int32_t index = 0; index < imagSize; ++index)
    {
        covarianceImag[index] = 0.0F;
    }

    float inverseSampleCount = 1.0F / (float)numSamples;
    int32_t numTiles         = (numSamples + 31) / 32;
    for (int32_t tile = 0; tile < numTiles; ++tile)
    {
        int32_t sampleStart = tile * 32;
        int32_t sampleEnd   = sampleStart + 32;
        if (sampleEnd > numSamples)
        {
            sampleEnd = numSamples;
        }

        for (int32_t column = 0; column < matrixSize; ++column)
        {
            for (int32_t row = column; row < matrixSize; ++row)
            {
                float sumReal = 0.0F;
                float sumImag = 0.0F;
                for (int32_t sample = sampleStart; sample < sampleEnd; ++sample)
                {
                    int32_t rowOffset    = (sample * matrixSize + row) * 2;
                    int32_t columnOffset = (sample * matrixSize + column) * 2;
                    float rowReal        = snapshots[rowOffset];
                    float rowImag        = snapshots[rowOffset + 1];
                    float columnReal     = snapshots[columnOffset];
                    float columnImag     = snapshots[columnOffset + 1];
                    sumReal += rowReal * columnReal + rowImag * columnImag;
                    sumImag += rowImag * columnReal - rowReal * columnImag;
                }

                covarianceReal[realIndex(matrixSize, row, column)] += sumReal * inverseSampleCount;
                if (row > column)
                {
                    covarianceImag[imagIndex(matrixSize, row, column)] += sumImag * inverseSampleCount;
                }
            }
        }
    }

    float trace = 0.0F;
    for (int32_t diagonal = 0; diagonal < matrixSize; ++diagonal)
    {
        trace += covarianceReal[realIndex(matrixSize, diagonal, diagonal)];
    }

    float diagonalScale   = diagonalLoadingConstant * 0x1p-4F;
    float diagonalLoading = trace * diagonalScale;
    diagonalLoading       = diagonalLoading > 0.0F ? diagonalLoading : 1.0F;
    for (int32_t diagonal = 0; diagonal < matrixSize; ++diagonal)
    {
        covarianceReal[realIndex(matrixSize, diagonal, diagonal)] += diagonalLoading;
    }
}

void computeS32CovarianceRef(const int32_t *snapshots, int32_t matrixSize, int32_t numSamples, int32_t qBits,
                             float diagonalLoadingConstant, float *covarianceReal, float *covarianceImag)
{
    int32_t numSampleBits = 0;
    for (int32_t samples = numSamples; samples > 1; samples >>= 1) ++numSampleBits;

    int32_t covarianceQBits = qBits * 2 - 16 - numSampleBits;
    float outputScale       = ldexpf(1.0f, -covarianceQBits) / (float)numSamples;
    float diagonalSum       = 0.0f;

    /* The device resets the integer accumulator for each 32-sample tile. */
    for (int32_t column = 0; column < matrixSize; ++column)
    {
        float diagonal = 0.0f;
        for (int32_t tileStart = 0; tileStart < numSamples; tileStart += 32)
        {
            int32_t tileEnd = tileStart + 32;
            int64_t sum     = 0;
            if (tileEnd > numSamples)
                tileEnd = numSamples;

            for (int32_t sample = tileStart; sample < tileEnd; ++sample)
            {
                int32_t index  = (sample * matrixSize + column) * 2;
                int32_t real   = snapshots[index];
                int32_t imag   = snapshots[index + 1];
                int64_t realSq = ((int64_t)real * real) >> 16;
                int64_t imagSq = ((int64_t)imag * imag) >> 16;
                sum += (realSq + imagSq) >> numSampleBits;
            }
            diagonal += (float)sum * outputScale;
        }

        covarianceReal[realIndex(matrixSize, column, column)] = diagonal;
        diagonalSum += diagonal;
    }

    for (int32_t column = 0; column < matrixSize; ++column)
    {
        for (int32_t row = column + 1; row < matrixSize; ++row)
        {
            float covarianceElementReal = 0.0f;
            float covarianceElementImag = 0.0f;
            for (int32_t tileStart = 0; tileStart < numSamples; tileStart += 32)
            {
                int32_t tileEnd = tileStart + 32;
                int64_t sumReal = 0;
                int64_t sumImag = 0;
                if (tileEnd > numSamples)
                    tileEnd = numSamples;

                for (int32_t sample = tileStart; sample < tileEnd; ++sample)
                {
                    int32_t rowIndex    = (sample * matrixSize + row) * 2;
                    int32_t columnIndex = (sample * matrixSize + column) * 2;
                    int32_t a           = snapshots[rowIndex];
                    int32_t b           = snapshots[rowIndex + 1];
                    int32_t c           = snapshots[columnIndex];
                    int32_t d           = snapshots[columnIndex + 1];
                    int64_t ac          = ((int64_t)a * c) >> 16;
                    int64_t bd          = ((int64_t)b * d) >> 16;
                    int64_t ad          = ((int64_t)a * d) >> 16;
                    int64_t bc          = ((int64_t)b * c) >> 16;
                    sumReal += (ac + bd) >> numSampleBits;
                    sumImag += (bc - ad) >> numSampleBits;
                }
                covarianceElementReal += (float)sumReal * outputScale;
                covarianceElementImag += (float)sumImag * outputScale;
            }

            covarianceReal[realIndex(matrixSize, row, column)] = covarianceElementReal;
            covarianceImag[imagIndex(matrixSize, row, column)] = covarianceElementImag;
        }
    }

    float diagonalScale   = diagonalLoadingConstant * 0x1p-4f;
    float diagonalLoading = diagonalSum * diagonalScale;
    diagonalLoading       = diagonalLoading > 0.0f ? diagonalLoading : 1.0f;
    for (int32_t diagonal = 0; diagonal < matrixSize; ++diagonal)
    {
        covarianceReal[realIndex(matrixSize, diagonal, diagonal)] += diagonalLoading;
    }
}
