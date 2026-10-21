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

#ifndef COMPLEX_CHOLESKY_REF_H
#define COMPLEX_CHOLESKY_REF_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Part 1 scalar reference functions for Cholesky decomposition and triangular inversion. */

/** Compute an in-place complex Cholesky decomposition using packed lower-triangular storage. */
void complexCholeskyRef(float *matrixReal, float *matrixImag, int32_t matrixSize);

/** Compute the inverse of a packed complex lower-triangular matrix. */
void complexLinvRef(const float *matrixReal, const float *matrixImag, float *inverseReal, float *inverseImag,
                    int32_t matrixSize);

#ifdef __cplusplus
}
#endif

#endif /* COMPLEX_CHOLESKY_REF_H */
