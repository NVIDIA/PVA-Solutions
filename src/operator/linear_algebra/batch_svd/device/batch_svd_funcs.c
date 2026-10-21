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

#include "batch_svd_funcs.h"

#include "batch_svd_params.h"

/* volatile where the device toolchain accepts it: NATIVE cannot qualify vector types. */
#if CUPVA_BUILD_MODE == CUPVA_NATIVE
#    define PVA_VPU_VOLATILE
#else
#    define PVA_VPU_VOLATILE volatile
#endif

/**
 * m x n matrix, m >= n
 * 16 matrices are interleaved
 * pitch = 16 + 1
 * A0, B0, C0, ...
 * A1, B1, C1, ...
 * A2, B2, C2, ...
 * A3, B3, C3, ...
 * A4, B4, C4, ...
 * ...
 * ...
 * ...
 */
typedef void (*alpha_beta_gamma_ptr)(float *, float *, float *, const float *, const int32_t, const int32_t,
                                     const int32_t, const int32_t, const int32_t);
typedef void (*update_matrix_ptr)(float *, const float *, const float *, const int16_t *, const int32_t, const int32_t,
                                  const int32_t, const int32_t, const int32_t);

void svd_opt(buffers_batch_svd_t &buf, const float tol, const int32_t max_iters, const int32_t m, const int32_t n,
             const int32_t pitch, const int32_t batch_count, const int32_t num_channel)
{
    SequenceDataFlowHandler *src_handler = buf.src_handler;
    SequenceDataFlowHandler *dst_handler = buf.dst_handler;
    cupvaSQDFTrig(*src_handler);

    float *sorted_s = buf.sorted_s;
    float *sorted_u = buf.sorted_u;
    float *sorted_v = buf.sorted_v;
    float *s        = buf.s;
    float *u_db     = buf.u;
    float *v        = buf.v;
    float *e        = buf.e;
    int16_t *p      = buf.p;
    int32_t *t      = buf.t;
    float *alpha    = buf.alpha;
    float *beta     = buf.beta;
    float *gamma    = buf.gamma;
    float *sin      = buf.sin;
    float *cos      = buf.cos;
    uint64_t addr_u = buf.addr_u;
    uint64_t addr_s = buf.addr_s;
    uint64_t addr_v = buf.addr_v;

    int32_t last_tile     = num_channel - (batch_count - 1) * BATCH;
    bool update_tile_size = (last_tile != BATCH) && (num_channel > BATCH);

    int32_t pingpong = 0;
    alpha_beta_gamma_ptr func_alpha_beta_gamma =
        (m == 2) ? calculate_alpha_beta_gamma_2
                 : ((m == 3) ? calculate_alpha_beta_gamma_3
                             : ((m == 4) ? calculate_alpha_beta_gamma_4 : calculate_alpha_beta_gamma));
    update_matrix_ptr func_update_matrix = (m > 3) ? update_matrix : update_matrix_2_3;

    for (int32_t count = 0; count < batch_count; count++)
    {

        float *u = u_db + m * n * pitch * pingpong;
        set_eye_matrix(v, n, pitch);

        float max_converge = tol + 1;
        int32_t niter      = 0;
        dvfloatx vzero     = as_dvfloatx(chess_dont_care(dvintx) & 0);

        cupvaSQDFSync(*src_handler);
        if (count < batch_count - 1)
            cupvaSQDFTrig(*src_handler);

        while (max_converge > tol && niter < max_iters)
        {
            *(PVA_VPU_VOLATILE dvfloat *)e = extract(vzero);
            for (int32_t i = 0; i < n - 1; i++)
            {
                for (int32_t j = i + 1; j < n; j++)
                {
                    func_alpha_beta_gamma(alpha, beta, gamma, u, m, n, pitch, i, j);
                    check_converge(e, p, alpha, beta, gamma, tol);
                    calculate_sin_cos(sin, cos, alpha, beta, gamma);
                    func_update_matrix(u, sin, cos, p, m, n, pitch, i, j);
                    func_update_matrix(v, sin, cos, p, n, n, pitch, i, j);
                }
            }
            const dvfloatx v_converge = sign_extend(*(PVA_VPU_VOLATILE dvfloat *)e);

            max_converge = max(vmaxr_s(as_vintx(v_converge.lo)), vmaxr_s(as_vintx(v_converge.hi)));

            niter++;
        }
        calculate_singular_values(s, u, m, n, pitch);
        normalize_matrix(u, s, m, n, pitch);

        if (count > 0)
        {
            cupvaSQDFSync(*dst_handler);
            uint64_t dst_u = addr_u + count * BATCH * sizeof(float);
            uint64_t dst_s = addr_s + count * BATCH * sizeof(float);
            uint64_t dst_v = addr_v + count * BATCH * sizeof(float);
            cupvaSQDFUpdateAddr(*dst_handler, 0, cupvaGetVmemAddress(sorted_u), MEMTYPE_VMEM, dst_u, MEMTYPE_DRAM);
            cupvaSQDFUpdateAddr(*dst_handler, 1, cupvaGetVmemAddress(sorted_s), MEMTYPE_VMEM, dst_s, MEMTYPE_DRAM);
            cupvaSQDFUpdateAddr(*dst_handler, 2, cupvaGetVmemAddress(sorted_v), MEMTYPE_VMEM, dst_v, MEMTYPE_DRAM);
        }
        if ((count == batch_count - 1) && update_tile_size)
        {
            cupvaSQDFUpdateTileSize(*dst_handler, 0, last_tile * sizeof(float), m * n);
            cupvaSQDFUpdateTileSize(*dst_handler, 1, last_tile * sizeof(float), n);
            cupvaSQDFUpdateTileSize(*dst_handler, 2, last_tile * sizeof(float), n * n);
        }
        sort_singular_values(t, sorted_s, sorted_u, sorted_v, s, u, v, m, n, pitch);

        cupvaSQDFFlushAndTrig(*dst_handler);
        pingpong ^= 1;
    }
    cupvaSQDFSync(*dst_handler);
}