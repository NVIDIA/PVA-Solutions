/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @file image_flip_vertical_knl.c
 * @brief PVA kernel for vertical image flip operation
 */

#include "image_flip_vertical_knl.h"

#include "image_flip_param.h"

#include <cupva_device.h>

/**
   * @brief Initialize vertical flip VPU operation
   * @param src Source image buffer
   * @param dst Destination image buffer
   * @param width Width of computation region
   * @param height Height of computation region
   * @param pitch Line pitch (bytes per row)
   * @param param Parameter structure to initialize
   * @return Number of iterations
   */

int vertical_flip_init(uint8_t *src, uint8_t *dst, int32_t width, int32_t height, int32_t linepitch,
                       param_vertical_flip_t *param)
{
    int vecw   = chess_elementsof(vcharx);
    int niter1 = width / vecw;
    int niter2 = height;

    AgenWrapper wrapper;
    agen_A in;
    agen_B out;

    in           = init_A((vchar *)src);
    wrapper.size = sizeof(char);
    wrapper.n1   = niter1;
    wrapper.n2   = niter2;
    wrapper.s1   = vecw;      // Move right by vector width
    wrapper.s2   = linepitch; // Jump to next row start
    INIT_AGEN2(in, wrapper);

    out          = init_B((vchar *)(dst + ((height - 1) * linepitch)));
    wrapper.size = sizeof(char);
    wrapper.n1   = niter1;
    wrapper.n2   = niter2;
    wrapper.s1   = vecw;       // Move right by vector width
    wrapper.s2   = -linepitch; // Jump to previous row start
    INIT_AGEN2(out, wrapper);

    // Store configurations in parameter structure
    param->cfgs[0]   = extract_agen_A_cfg(in);
    param->cfgs[1]   = extract_agen_B_cfg(out);
    param->niter1    = niter1;
    param->niter2    = niter2;
    param->width     = width;
    param->height    = height;
    param->linepitch = linepitch;

    return 1;
}

/**
   * @brief Execute vertical flip VPU operation using vectorized operations
   * @param args Parameter structure containing configuration
   */
int vertical_flip_exec(void *args)
{
    agen_A in;
    agen_B out;
    param_vertical_flip_t *param = (param_vertical_flip_t *)args;

    AgenCFG *cfgs     = param->cfgs;
    int32_t niter1    = param->niter1;
    int32_t niter2    = param->niter2;
    int32_t width     = param->width;
    int32_t height    = param->height;
    int32_t linepitch = param->linepitch;

    int vecw = chess_elementsof(vcharx);

    vcharx vA;

    in  = init_agen_A_from_cfg(cfgs[0]);
    out = init_agen_B_from_cfg(cfgs[1]);

    // Process row by row with vertical flipping

    for (int i = 0; i < niter1 * niter2; i++) chess_unroll_loop(16)
    chess_prepare_for_pipelining
    {
        vA = vchar_load(in);
        vstore(vA, out);
    }

    return 1;
}
