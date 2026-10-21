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
 * @file image_flip_combined_knl.c
 * @brief PVA kernel for combined image flip operation (both horizontal and vertical)
 */

#include "image_flip_combined_knl.h"

#include "image_flip_param.h"

#include <cupva_device.h>

/**
  * @brief Initialize combined flip VPU operation (180-degree rotation)
  * @param src Source image buffer
  * @param dst Destination image buffer
  * @param width Width of computation region
  * @param height Height of computation region
  * @param pitch Line pitch (bytes per row)
  * @param param Parameter structure to initialize
  * @return Number of iterations
  */

const char load_pat[] = {31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
                         15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0};

int flip_both_init(uint8_t *src, uint8_t *dst, int32_t width, int32_t height, int32_t linepitch,
                   param_flip_both_t *param)
{
    int vecw   = chess_elementsof(vcharx);
    int niter1 = width / vecw;
    int niter2 = height;
    int niter  = niter1 * niter2;

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

    out          = init_B((vchar *)(dst + ((height)*linepitch) - vecw));
    wrapper.size = sizeof(char);
    wrapper.n1   = niter1;
    wrapper.n2   = niter2;
    wrapper.s1   = -vecw;      // Move right by vector width
    wrapper.s2   = -linepitch; // Jump to previous row start
    INIT_AGEN2(out, wrapper);

    // Store configurations in parameter structure
    param->cfgs[0]   = extract_agen_A_cfg(in);
    param->cfgs[1]   = extract_agen_B_cfg(out);
    param->width     = width;
    param->height    = height;
    param->linepitch = linepitch;
    param->niter1    = niter1;
    param->niter2    = niter2;

    return 1;
}

/**
  * @brief Execute combined flip VPU operation (180-degree rotation) using vectorized operations
  * @param args Parameter structure containing configuration
  */
int flip_both_exec(void *args)
{
    int it, j;
    agen_A in;
    agen_B out;

    param_flip_both_t *param = (param_flip_both_t *)args;

    AgenCFG *cfgs = param->cfgs;
    int niter1    = param->niter1;
    int niter2    = param->niter2;
    int width     = param->width;
    int height    = param->height;
    int linepitch = param->linepitch;

    dvcharx vin_elem;
    int vecw = chess_elementsof(vcharx);

    vcharx vA, vC;
    vcharx vB = sign_extend(*(vchar *)load_pat);

    in  = init_agen_A_from_cfg(cfgs[0]);
    out = init_agen_B_from_cfg(cfgs[1]);

    for (int i = 0; i < niter1 * niter2; i++) chess_unroll_loop(16)
    chess_prepare_for_pipelining
    {
        vA = vchar_load(in);
        vC = vpermute(vA, vB);
        vstore(vC, out);
    }
    return 1;
}
