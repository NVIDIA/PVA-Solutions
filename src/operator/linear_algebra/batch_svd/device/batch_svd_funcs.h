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

#ifndef BATCH_SVD_FUNCS_H
#define BATCH_SVD_FUNCS_H

#include "batch_svd_knls.h"

#include <cupva_device.h>

typedef struct
{
    float *sorted_s;
    float *sorted_u;
    float *sorted_v;
    float *s;
    float *u;
    float *v;
    float *e;
    int16_t *p;
    int32_t *t;
    float *alpha;
    float *beta;
    float *gamma;
    float *sin;
    float *cos;
    uint64_t addr_u;
    uint64_t addr_s;
    uint64_t addr_v;
    SequenceDataFlowHandler *src_handler;
    SequenceDataFlowHandler *dst_handler;
} buffers_batch_svd_t;

void svd_opt(buffers_batch_svd_t &buf, const float tol, const int32_t max_iters, const int32_t m, const int32_t n,
             const int32_t pitch, const int32_t batch_count, const int32_t num_channel);

#endif