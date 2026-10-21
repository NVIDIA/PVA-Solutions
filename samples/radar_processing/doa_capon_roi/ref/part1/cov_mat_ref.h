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

#ifndef COV_MAT_REF_H
#define COV_MAT_REF_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Compute one loaded covariance matrix using the FP kernel's arithmetic order. */
void computeFpCovarianceRef(const float *snapshots, int32_t matrixSize, int32_t numSamples,
                            float diagonalLoadingConstant, float *covarianceReal, float *covarianceImag);

/** Compute one loaded covariance matrix using the S32 kernel's arithmetic order. */
void computeS32CovarianceRef(const int32_t *snapshots, int32_t matrixSize, int32_t numSamples, int32_t qBits,
                             float diagonalLoadingConstant, float *covarianceReal, float *covarianceImag);

#ifdef __cplusplus
}
#endif

#endif /* COV_MAT_REF_H */
