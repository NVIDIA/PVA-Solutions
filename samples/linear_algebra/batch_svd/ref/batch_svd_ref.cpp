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

#include "batch_svd_ref.h"

#include <OpBatchSVD.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

class RandomGenerator
{
private:
    static std::mt19937 gen;

public:
    static int32_t getRandomInt(int32_t min, int32_t max)
    {
        std::uniform_int_distribution<> dis(min, max);
        return dis(gen);
    }

    static float getRandomFloat(float min, float max)
    {
        std::uniform_real_distribution<float> dis(min, max);
        return dis(gen);
    }
};
std::mt19937 RandomGenerator::gen(std::random_device{}());

void generateRandomInput(int32_t nRows, int32_t nCols, int32_t nMats, int32_t pitch, float min, float max, float *A)
{
    for (int32_t k = 0; k < nMats; k++)
    {
        for (int32_t j = 0; j < nRows; ++j)
        {
            for (int32_t i = 0; i < nCols; ++i)
            {
                A[(j * nCols + i) * pitch + k] = RandomGenerator::getRandomFloat(min, max);
            }
        }
    }
}

void generateRankDeficientRandomInput(int32_t nRows, int32_t nCols, int32_t nMats, int32_t pitch, float min, float max,
                                      float *A)
{
    for (int32_t k = 0; k < nMats; k++)
    {
        int32_t nRanks = RandomGenerator::getRandomInt(0, nCols - 1);
        // Reduce rank by making some columns linearly dependent
        for (int32_t j = 0; j < nRanks; j++)
        {
            for (int32_t i = 0; i < nRows; i++)
            {
                A[(i * nCols + j) * pitch + k] = RandomGenerator::getRandomFloat(min, max);
            }
        }

        for (int32_t j = nRanks; j < nCols; j++)
        {
            for (int32_t i = 0; i < nRows; i++)
            {
                A[(i * nCols + j) * pitch + k] = 0;
                for (int32_t t = 0; t < nRanks; t++)
                {
                    A[(i * nCols + j) * pitch + k] +=
                        RandomGenerator::getRandomInt(0, 10) * A[(i * nCols + t) * pitch + k];
                }
            }
        }
    }
}

void batchSetEye(const int32_t n, const int32_t pitch, const int32_t batch, float *V)
{
    memset(V, 0, n * n * pitch * sizeof(float));
    for (int32_t i = 0; i < n; i++)
    {
        for (int32_t j = 0; j < batch; j++)
        {
            V[(i * n + i) * pitch + j] = 1.0f;
        }
    }
}

float dot_prod(const int32_t m, const int32_t n, const int32_t pitch, const float *A, const float *B)
{
    float sum = 0;
    for (int32_t i = 0; i < m; i++)
    {
        sum += A[i * n * pitch] * B[i * n * pitch];
    }
    return sum;
}

void rotate_mat(const int32_t m, const int32_t n, const int32_t pitch, const float c, const float s, float *A, float *B)
{
    for (int32_t i = 0; i < m; i++)
    {
        float a          = A[i * n * pitch];
        float b          = B[i * n * pitch];
        A[i * n * pitch] = c * a - s * b;
        B[i * n * pitch] = s * a + c * b;
    }
}

void batchSVDRefImpl(const int32_t m, const int32_t n, const int32_t pitch, const int32_t batch, float tol,
                     int32_t maxIters, const float *A, float *sorted_U, float *sorted_V, float *sorted_S)
{
    std::vector<float> U(m * n * pitch);
    std::vector<float> V(n * n * pitch);
    std::vector<float> S(n * pitch);

    int32_t converged, niter;

    memcpy(U.data(), A, m * n * pitch * sizeof(float));
    batchSetEye(n, pitch, batch, V.data());

    for (int32_t k = 0; k < batch; k++)
    {
        niter     = 0;
        converged = 0;
        while (!converged && niter < maxIters)
        {
            converged = 1;
            for (int32_t i = 0; i < n - 1; i++)
            {
                for (int32_t j = i + 1; j < n; j++)
                {
                    float alpha = dot_prod(m, n, pitch, &U[i * pitch + k], &U[i * pitch + k]);
                    float beta  = dot_prod(m, n, pitch, &U[j * pitch + k], &U[j * pitch + k]);
                    float gamma = dot_prod(m, n, pitch, &U[i * pitch + k], &U[j * pitch + k]);

                    float temp1 = std::abs(gamma);
                    float temp2 = sqrt(alpha * beta);

                    // If abs(gamma) is very close to 0, the two columns are nearly orthogonal
                    // and if abs(alpha) or abs(beta) is very close to 0, one of the columns has become negligible
                    if (temp1 <= temp2 * tol ||
                        (temp1 < FLT_EPSILON && (std::abs(alpha) < FLT_EPSILON || std::abs(beta) < FLT_EPSILON)))
                        continue; // skip (i, j) pair

                    converged = 0;

                    float zeta = (beta - alpha) / (2.0f * gamma);
                    float t    = (zeta >= 0 ? 1 : -1) / (std::abs(zeta) + sqrt(1.f + zeta * zeta));
                    float c    = 1.f / sqrt(1.f + t * t);
                    float s    = 0.f;

                    // c could be -nan if t is -nan
                    int32_t intC, intOne;
                    float floatOne = 1.f;
                    memcpy(&intC, &c, sizeof(float));
                    memcpy(&intOne, &floatOne, sizeof(float));
                    if ((intC & 0x7fffffffu) > intOne)
                    {
                        c = 1.f;
                    }
                    else
                    {
                        s = t * c;
                    }

                    rotate_mat(m, n, pitch, c, s, &U[i * pitch + k], &U[j * pitch + k]);
                    rotate_mat(n, n, pitch, c, s, &V[i * pitch + k], &V[j * pitch + k]);
                }
            }
            niter++;
        }
    }

    // Normalize columns of U
    for (int32_t k = 0; k < batch; k++)
    {
        for (int32_t i = 0; i < n; i++)
        {
            float norm       = dot_prod(m, n, pitch, &U[i * pitch + k], &U[i * pitch + k]);
            norm             = sqrt(norm);
            S[i * pitch + k] = norm;
        }
    }
    for (int32_t k = 0; k < batch; k++)
    {
        for (int32_t i = 0; i < n; i++)
        {
            for (int32_t j = 0; j < m; j++)
            {
                U[(j * n + i) * pitch + k] /= S[i * pitch + k];
            }
        }
    }

    // Sort via ranking T
    // Example [3 4 1 5] -> [2 1 3 0], rank = # greater data points = location to deposit
    // When there're ties [3 4 1 3] -> [1 0 3 1] -> [1 0 3 2], + # lower-index ties
    std::vector<int> T(n * batch, 0);

    for (int32_t k = 0; k < batch; k++)
    {
        for (int32_t i = 0; i < n; i++)
        {
            for (int32_t j = 0; j < n; j++)
            {
                T[i * batch + k] += (S[i * pitch + k] < S[j * pitch + k]);
                T[i * batch + k] += (j < i) && (S[i * pitch + k] == S[j * pitch + k]);
            }
        }
    }

    // sort U
    for (int32_t k = 0; k < batch; k++)
    {
        for (int32_t i = 0; i < n; i++)
        {
            int32_t idx = T[i * batch + k];
            for (int32_t j = 0; j < m; j++)
            {
                sorted_U[(j * n + idx) * batch + k] = U[(j * n + i) * pitch + k];
            }
        }
    }

    // sort V
    for (int32_t k = 0; k < batch; k++)
    {
        for (int32_t i = 0; i < n; i++)
        {
            int32_t idx = T[i * batch + k];
            for (int32_t j = 0; j < n; j++)
            {
                sorted_V[(j * n + idx) * batch + k] = V[(j * n + i) * pitch + k];
            }
        }
    }

    for (int32_t k = 0; k < batch; k++)
    {
        for (int32_t i = 0; i < n; i++)
        {
            int32_t idx               = T[i * batch + k];
            sorted_S[idx * batch + k] = S[i * pitch + k];
        }
    }
}

// limited precision when input range is large using float32
bool compareFloat(float a, float b, float tol, float epsilon = 1e-2f)
{
    // nan
    if (std::isnan(a) || std::isnan(b))
    {
        return std::isnan(a) && std::isnan(b);
    }

    // inf
    if (std::isinf(a) || std::isinf(b))
    {
        return a == b;
    }

    if (std::abs(a) <= tol && std::abs(b) <= tol)
    {
        return true;
    }

    float ratio = std::min(std::abs(a), std::abs(b)) / std::max(std::abs(a), std::abs(b)); // for large values
    float diff  = std::abs(a - b);                                                         // for small values

    return ((1 - ratio) <= epsilon) || (diff <= epsilon);
}

// limited precision when input range is large using float32
bool compareFloatVector(float *a, float *b, int32_t m, int32_t n, int32_t pitch, float epsilon = 1e-2f)
{
    bool allOpposite = true;
    bool allEqual    = true;
    // column
    for (int32_t i = 0; i < m; i++)
    {
        // nan
        if (std::isnan(a[i * n * pitch]) || std::isnan(b[i * n * pitch]))
        {
            if (std::isnan(a[i * n * pitch]) != std::isnan(b[i * n * pitch]))
                return false;
        }

        float num1 = std::abs(a[i * n * pitch]);
        float num2 = std::abs(b[i * n * pitch]);
        if (std::abs(num1 - num2) > epsilon)
            return false;

        if (std::signbit(num1) == std::signbit(num2))
        {
            allOpposite = false;
        }
        else
        {
            allEqual = false;
        }
    }
    return allOpposite || allEqual;
}

} // namespace

void generateBatchSVDRandomInput(NVCVTensorHandle tensorHandle, const float rangeMin, const float rangeMax,
                                 bool rankDeficient)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    float *tensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&tensorPtr, (void *)tensorData.buffer.strided.basePtr);

    int32_t height    = tensorData.shape[0];
    int32_t width     = tensorData.shape[1];
    int32_t channel   = tensorData.shape[2];
    int32_t linePitch = tensorData.buffer.strided.strides[1] / sizeof(float);

    if (rankDeficient)
        generateRankDeficientRandomInput(height, width, channel, linePitch, rangeMin, rangeMax, tensorPtr);
    else
        generateRandomInput(height, width, channel, linePitch, rangeMin, rangeMax, tensorPtr);
}

int32_t checkBatchSVDOutput(NVCVTensorHandle uPvaTensorHandle, NVCVTensorHandle sPvaTensorHandle,
                            NVCVTensorHandle vPvaTensorHandle, NVCVTensorHandle uRefTensorHandle,
                            NVCVTensorHandle sRefTensorHandle, NVCVTensorHandle vRefTensorHandle, const float tol)
{
    NVCVTensorData uPvaTensorData;
    nvcvTensorExportData(uPvaTensorHandle, &uPvaTensorData);
    float *uPva = NULL;
    CupvaMemGetHostPointer((void **)&uPva, (void *)uPvaTensorData.buffer.strided.basePtr);

    NVCVTensorData vPvaTensorData;
    nvcvTensorExportData(vPvaTensorHandle, &vPvaTensorData);
    float *vPva = NULL;
    CupvaMemGetHostPointer((void **)&vPva, (void *)vPvaTensorData.buffer.strided.basePtr);

    NVCVTensorData sPvaTensorData;
    nvcvTensorExportData(sPvaTensorHandle, &sPvaTensorData);
    float *sPva = NULL;
    CupvaMemGetHostPointer((void **)&sPva, (void *)sPvaTensorData.buffer.strided.basePtr);

    NVCVTensorData uRefTensorData;
    nvcvTensorExportData(uRefTensorHandle, &uRefTensorData);
    float *uRef = NULL;
    CupvaMemGetHostPointer((void **)&uRef, (void *)uRefTensorData.buffer.strided.basePtr);

    NVCVTensorData vRefTensorData;
    nvcvTensorExportData(vRefTensorHandle, &vRefTensorData);
    float *vRef = NULL;
    CupvaMemGetHostPointer((void **)&vRef, (void *)vRefTensorData.buffer.strided.basePtr);

    NVCVTensorData sRefTensorData;
    nvcvTensorExportData(sRefTensorHandle, &sRefTensorData);
    float *sRef = NULL;
    CupvaMemGetHostPointer((void **)&sRef, (void *)sRefTensorData.buffer.strided.basePtr);

    int32_t height    = uPvaTensorData.shape[0];
    int32_t width     = uPvaTensorData.shape[1];
    int32_t channel   = uPvaTensorData.shape[2];
    int32_t linePitch = uPvaTensorData.buffer.strided.strides[1] / sizeof(float);

    int32_t err = 0;
    for (int32_t k = 0; k < channel; k++)
    {
        for (int32_t i = 0; i < width; i++)
        {
            int32_t idx = i * linePitch + k;
            // Check if singular value is very close to 0, cols of U and V are not unique
            // normalizing U could divide a small value by a small value
            if ((!compareFloatVector(&uPva[idx], &uRef[idx], height, width, linePitch) && sRef[idx] > 1e-3f) ||
                (!compareFloatVector(&vPva[idx], &vRef[idx], width, width, linePitch) && sRef[idx] > 1e-3f))
            {
                // In case two singular values are close, two columns may be swapped
                bool swap = false;
                err       = 1;
                for (int32_t t = 0; t < width; t++)
                {
                    if (t == i)
                        continue;
                    if (compareFloatVector(&uPva[idx], &uRef[t * linePitch + k], height, width, linePitch) &&
                        compareFloatVector(&vPva[idx], &vRef[t * linePitch + k], width, width, linePitch))
                    {
                        swap = true;
                        err  = 0;
                        break;
                    }
                }
                if (err)
                {
                    if (!compareFloatVector(&uPva[idx], &uRef[idx], height, width, linePitch) && sRef[idx] > 1e-3f)
                    {
                        for (int32_t j = 0; j < height; j++)
                        {
                            printf("U mismatch[channel %d, col %d, row %d]: pva = %e, ref = %e\n", k, i, j,
                                   uPva[idx + j * width * linePitch], uRef[idx + j * width * linePitch]);
                        }
                    }
                    if (!compareFloatVector(&vPva[idx], &vRef[idx], width, width, linePitch) && sRef[idx] > 1e-3f)
                    {
                        for (int32_t j = 0; j < width; j++)
                        {
                            printf("V mismatch[channel %d, col %d, row %d]: pva = %e, ref = %e\n", k, i, j,
                                   vPva[idx + j * width * linePitch], vRef[idx + j * width * linePitch]);
                        }
                    }
                }
            }

            // If the input range is large, resulting in large S values, use ratio comparison
            if (!compareFloat(sPva[idx], sRef[idx], tol))
            {
                err = 1;
                printf("S mismatch[channel %d, col %d]: pva = %e, ref = %e\n", k, i, sPva[idx], sRef[idx]);
            }
        }
    }
    return err;
}

void batchSVDRef(NVCVTensorHandle srcTensorHandle, NVCVTensorHandle uTensorHandle, NVCVTensorHandle sTensorHandle,
                 NVCVTensorHandle vTensorHandle, const int32_t maxIters, const float tol)
{
    NVCVTensorData srcTensorData;
    nvcvTensorExportData(srcTensorHandle, &srcTensorData);

    float *srcPtr = NULL;
    CupvaMemGetHostPointer((void **)&srcPtr, (void *)srcTensorData.buffer.strided.basePtr);

    NVCVTensorData uTensorData;
    nvcvTensorExportData(uTensorHandle, &uTensorData);

    float *uPtr = NULL;
    CupvaMemGetHostPointer((void **)&uPtr, (void *)uTensorData.buffer.strided.basePtr);

    NVCVTensorData vTensorData;
    nvcvTensorExportData(vTensorHandle, &vTensorData);

    float *vPtr = NULL;
    CupvaMemGetHostPointer((void **)&vPtr, (void *)vTensorData.buffer.strided.basePtr);

    NVCVTensorData sTensorData;
    nvcvTensorExportData(sTensorHandle, &sTensorData);

    float *sPtr = NULL;
    CupvaMemGetHostPointer((void **)&sPtr, (void *)sTensorData.buffer.strided.basePtr);

    int32_t height    = uTensorData.shape[0];
    int32_t width     = uTensorData.shape[1];
    int32_t channel   = uTensorData.shape[2];
    int32_t linePitch = uTensorData.buffer.strided.strides[1] / sizeof(float);

    batchSVDRefImpl(height, width, linePitch, channel, tol, maxIters, srcPtr, uPtr, vPtr, sPtr);
}
