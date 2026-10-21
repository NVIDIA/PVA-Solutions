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

#include "complex_cholesky_ref.h"

#include <math.h>

/* Packed lower-triangular indexing shared by the Cholesky and Linv stages. */

static int32_t realIndex(int32_t matrixSize, int32_t row, int32_t column)
{
    return column * matrixSize - column * (column - 1) / 2 + row - column;
}

static int32_t imagIndex(int32_t matrixSize, int32_t row, int32_t column)
{
    return column * matrixSize - (column + 1) * column / 2 + row - column - 1;
}

void complexCholeskyRef(float *matrixReal, float *matrixImag, int32_t matrixSize)
{
    for (int32_t column = 0; column < matrixSize; ++column)
    {
        int32_t diagonalIndex     = realIndex(matrixSize, column, column);
        float inverseDiagonal     = 1.0F / sqrtf(matrixReal[diagonalIndex]);
        matrixReal[diagonalIndex] = inverseDiagonal;

        for (int32_t row = column + 1; row < matrixSize; ++row)
        {
            int32_t elementRealIndex     = realIndex(matrixSize, row, column);
            int32_t elementImagIndex     = imagIndex(matrixSize, row, column);
            float elementReal            = matrixReal[elementRealIndex] * inverseDiagonal;
            float elementImag            = matrixImag[elementImagIndex] * inverseDiagonal;
            matrixReal[elementRealIndex] = elementReal;
            matrixImag[elementImagIndex] = elementImag;

            for (int32_t trailingColumn = column + 1; trailingColumn <= row; ++trailingColumn)
            {
                float columnReal = matrixReal[realIndex(matrixSize, trailingColumn, column)];
                float columnImag = matrixImag[imagIndex(matrixSize, trailingColumn, column)];
                matrixReal[realIndex(matrixSize, row, trailingColumn)] -=
                    columnReal * elementReal + columnImag * elementImag;

                if (trailingColumn < row)
                {
                    matrixImag[imagIndex(matrixSize, row, trailingColumn)] -=
                        columnReal * elementImag - columnImag * elementReal;
                }
            }
        }
    }
}

void complexLinvRef(const float *matrixReal, const float *matrixImag, float *inverseReal, float *inverseImag,
                    int32_t matrixSize)
{
    for (int32_t column = 0; column < matrixSize; ++column)
    {
        int32_t diagonalIndex      = realIndex(matrixSize, column, column);
        inverseReal[diagonalIndex] = matrixReal[diagonalIndex];
        inverseImag[diagonalIndex] = 0.0F;

        for (int32_t row = column + 1; row < matrixSize; ++row)
        {
            float sumReal = 0.0F;
            float sumImag = 0.0F;
            for (int32_t inner = column; inner < row; ++inner)
            {
                int32_t matrixIndex      = realIndex(matrixSize, row, inner);
                int32_t inverseIndex     = realIndex(matrixSize, inner, column);
                float matrixElementReal  = matrixReal[matrixIndex];
                float matrixElementImag  = matrixImag[imagIndex(matrixSize, row, inner)];
                float inverseElementReal = inverseReal[inverseIndex];
                float inverseElementImag = inverseImag[inverseIndex];
                sumReal += matrixElementReal * inverseElementReal - matrixElementImag * inverseElementImag;
                sumImag += matrixElementReal * inverseElementImag + matrixElementImag * inverseElementReal;
            }

            float inverseDiagonal    = matrixReal[realIndex(matrixSize, row, row)];
            int32_t outputIndex      = realIndex(matrixSize, row, column);
            inverseReal[outputIndex] = -sumReal * inverseDiagonal;
            inverseImag[outputIndex] = -sumImag * inverseDiagonal;
        }
    }
}
