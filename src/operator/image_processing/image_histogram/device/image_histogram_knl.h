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

#ifndef IMAGE_HISTOGRAM_KNL_H
#define IMAGE_HISTOGRAM_KNL_H

#include <cupva_device.h>

typedef struct ImageHistogramConfig
{
    AgenCFG validAgenCfgs[2];
    AgenCFG maskAgenCfgs[3];
    AgenCFG indexAgenCfgs[2];
    AgenCFG execAgenCfgs[2];
    AgenCFG reduceAgenCfgs[2];

    int niter_valid;
    int niter_valid_last;
    int niter_mask;
    int niter_mask_last;
    int niter_index;
    int niter_index_last;
    int niter_exec;
    int niter_exec_last;
    int validNiter;
    int maskNiter;
    int indexNiter;
    int execNiter;

    int start;
    int end;
    int a; /* bins per raw value, scaled by 2^qbits */
    int b; /* -start * a, U8 only; the U16 path subtracts start before the multiply */
    void *hist;
    uint32_t *dst;
    int32_t *bucketAcc;
    int nbins;

    int maskWidth;
} ImageHistogramConfig;

static inline void bzero_opt(void *mem, int size)
{
    dvcharx vzero;
    vzero.hi = vzero.lo = replicateb(0);

    int vecw  = chess_elementsof(dvchar);
    int niter = (size + vecw - 1) / vecw;

    agen out = init(mem);
    out.n1   = niter;
    out.mod1 = sizeof(dvchar);

    for (int i = 0; i < niter; i++)
    {
        vstore(vzero, out);
    }

    mem_fence();
}

void image_histogram_knl_fold_u8(ImageHistogramConfig *cfg);
void image_histogram_knl_emit_u8(ImageHistogramConfig *cfg);

#endif // IMAGE_HISTOGRAM_KNL_H
