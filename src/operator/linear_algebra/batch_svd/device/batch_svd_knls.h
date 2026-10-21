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

#ifndef BATCH_SVD_KNLS_H
#define BATCH_SVD_KNLS_H

#include <cupva_device.h>

void set_eye_matrix(float *v, const int32_t n, const int32_t pitch);

void calculate_alpha_beta_gamma(float *alpha, float *beta, float *gamma, const float *u, const int32_t m,
                                const int32_t n, const int32_t pitch, const int32_t i, const int32_t j);

void calculate_alpha_beta_gamma_2(float *alpha, float *beta, float *gamma, const float *u, const int32_t m,
                                  const int32_t n, const int32_t pitch, const int32_t i, const int32_t j);
void calculate_alpha_beta_gamma_3(float *alpha, float *beta, float *gamma, const float *u, const int32_t m,
                                  const int32_t n, const int32_t pitch, const int32_t i, const int32_t j);
void calculate_alpha_beta_gamma_4(float *alpha, float *beta, float *gamma, const float *u, const int32_t m,
                                  const int32_t n, const int32_t pitch, const int32_t i, const int32_t j);

void check_converge(float *e, int16_t *p, const float *alpha, const float *beta, const float *gamma, const float tol);

void calculate_sin_cos(float *s, float *c, const float *alpha, const float *beta, const float *gamma);

void update_matrix(float *u, const float *s, const float *c, const int16_t *p, const int32_t m, const int32_t n,
                   const int32_t pitch, const int32_t i, const int32_t j);
void update_matrix_2_3(float *u, const float *s, const float *c, const int16_t *p, const int32_t m, const int32_t n,
                       const int32_t pitch, const int32_t i, const int32_t j);

void calculate_singular_values(float *s, const float *u, const int32_t m, const int32_t n, const int32_t pitch);

void normalize_matrix(float *u, const float *s, const int32_t m, const int32_t n, const int32_t pitch);

void sort_singular_values(int32_t *t, float *sorted_s, float *sorted_u, float *sorted_v, const float *s, const float *u,
                          const float *v, const int32_t m, const int32_t n, const int32_t pitch);

#endif