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

VMEM(C, float, U, M *N *PITCH * 2);
VMEM(B, float, V, N *N *PITCH);
VMEM(A, float, S, N *PITCH);

VMEM(A, float, sorted_U, M *N *BATCH);
VMEM(B, float, sorted_V, N *N *BATCH);
VMEM(C, float, sorted_S, N *BATCH);

VMEM(A, float, alpha_svd, BATCH);
VMEM(B, float, beta_svd, BATCH);
VMEM(C, float, gamma_svd, BATCH);
VMEM(A, float, rotate_s, BATCH);
VMEM(B, float, rotate_c, BATCH);

VMEM(A, int32_t, idx, N *BATCH);
VMEM(B, float, e, BATCH);
VMEM(B, int16_t, p, BATCH);

VMEM(B, float, tol);
VMEM(B, int32_t, max_iters);
VMEM(B, int32_t, m);           // #rows
VMEM(B, int32_t, n);           // #cols
VMEM(B, int32_t, num_channel); // total #matrices
VMEM(B, int32_t, batch_count);

VMEM_SEQDF_HANDLER(B, src_handler);
VMEM_SEQDF_HANDLER(B, dst_handler);
VMEM_POINTER(B, dram_U);
VMEM_POINTER(B, dram_S);
VMEM_POINTER(B, dram_V);

CUPVA_VPU_MAIN()
{
    cupvaFloatingPointNANErrorEnabled(false);
    cupvaSQDFOpen(src_handler);
    cupvaSQDFOpen(dst_handler);

    buffers_batch_svd_t buf_ptrs = {.sorted_s    = sorted_S,
                                    .sorted_u    = sorted_U,
                                    .sorted_v    = sorted_V,
                                    .s           = S,
                                    .u           = U,
                                    .v           = V,
                                    .e           = e,   // indicator of convergence
                                    .p           = p,   // predicate to update matrix
                                    .t           = idx, // sorted index
                                    .alpha       = alpha_svd,
                                    .beta        = beta_svd,
                                    .gamma       = gamma_svd,
                                    .sin         = rotate_s,
                                    .cos         = rotate_c,
                                    .addr_u      = dram_U.base + dram_U.offset,
                                    .addr_s      = dram_S.base + dram_S.offset,
                                    .addr_v      = dram_V.base + dram_V.offset,
                                    .src_handler = &src_handler,
                                    .dst_handler = &dst_handler};

    svd_opt(buf_ptrs, tol, max_iters, m, n, PITCH, batch_count, num_channel);

    cupvaSQDFClose(src_handler);
    cupvaSQDFClose(dst_handler);

    return 0;
}