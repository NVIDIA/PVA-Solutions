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

#ifndef MEDIAN_FILTER_PARAM_H
#define MEDIAN_FILTER_PARAM_H

// kernel size in pixels
#if defined(KNL_3x3)
#    define KNL_H 3
#    define KNL_W 3
#elif defined(KNL_5x5)
#    define KNL_H 5
#    define KNL_W 5
#elif defined(KNL_5x7)
#    define KNL_H 5
#    define KNL_W 7
#elif defined(KNL_7x7)
#    define KNL_H 7
#    define KNL_W 7
#endif

#if defined(INTERLEAVED2C)
#    define CHANNEL_NUM 2
#else
#    define CHANNEL_NUM 1
#endif

// halo size
#define HALO_X (KNL_W >> 1) * CHANNEL_NUM
#define HALO_Y (KNL_H >> 1)

// tile size in bytes
#define TILE_W 128
#define TILE_H 128

// tile size in pixels
#if defined(U8_IO) || defined(S8_IO)
#    define TILE_H_PIX TILE_H
#elif defined(U16_IO) || defined(S16_IO) || defined(I2S16_IO)
#    define TILE_H_PIX (TILE_H / 2)
#elif defined(U32_IO) || defined(S32_IO)
#    define TILE_H_PIX (TILE_H / 4)
#endif

// Taller-tile factors for the channel-interleaved 2S16 separable programs:
// they run the per-channel pipeline at half tile width, so 5x5 and 5x7 use
// taller tiles to amortize per-tile overhead — the tallest heights that keep
// every VMEM superbank within budget (7x7 already fills VMEM at the default
// height).
#define MEDIAN_2S16_TALL_TILE_H_5X5(h) ((h)*3 / 2)
#define MEDIAN_2S16_TALL_TILE_H_5X7(h) ((h)*5 / 4)

// Dataflow (RDF) tile height in elements. The device supplies compile-time
// parameters, while the host supplies its template and runtime parameters.
#define MEDIAN_TILE_H_DF(tileHPix, channels, kernelH, kernelW)                                            \
    (((channels) == 2 && (kernelH) == 5 && (kernelW) == 5)                                                \
         ? MEDIAN_2S16_TALL_TILE_H_5X5(tileHPix)                                                          \
         : (((channels) == 2 && (kernelH) == 5 && (kernelW) == 7) ? MEDIAN_2S16_TALL_TILE_H_5X7(tileHPix) \
                                                                  : (tileHPix)))

#if defined(TILE_H_PIX)
#    define TILE_H_DF MEDIAN_TILE_H_DF(TILE_H_PIX, CHANNEL_NUM, KNL_H, KNL_W)
#endif

// intermediate tile size for sorted rows
#if defined(SEPA_2x2)
#    define INTERM_TILE_SIZE (TILE_W / 2 * (TILE_H_PIX + KNL_H - 1))
#else
#    define INTERM_TILE_SIZE (TILE_W * (TILE_H_PIX + KNL_H - 1))
#endif

// separable sorting network parameters
#if defined(SEPA_2x2)
#    define GROUP_NUM_PASS_2 (TILE_W / 2 * TILE_H_PIX / 2)

#    if defined(KNL_5x5)
#        define KNL_FUNC_EXEC serial_run_sepa_2x2_5x5_exec
// 5x5 uses the double-vector second pass with its own agen layouts
#        define KNL_INIT_FUNC median_filter_5x5_sepa_2x2_init
#        define KEEP_IDX_1_SIZE (10)
#        define KEEP_IDX_2_SIZE (6)
// 0, 3 use C; 1 uses B; 2 uses A.
#        define SEPA_2X2_NUM_ELEMENTS_A 1
#        define SEPA_2X2_NUM_ELEMENTS_B 1
#        define SEPA_2X2_NUM_ELEMENTS_C 2
#    elif defined(KNL_5x7)
#        define KNL_FUNC_EXEC serial_run_sepa_2x2_5x7_exec
#        define KNL_INIT_FUNC median_filter_nxm_sepa_2x2_init
#        define KEEP_IDX_1_SIZE (12)
#        define KEEP_IDX_2_SIZE (6)
// 0, 3 use A; 1, 4 use B; 2, 5 use C.
#        define SEPA_2X2_NUM_ELEMENTS_A 2
#        define SEPA_2X2_NUM_ELEMENTS_B 2
#        define SEPA_2X2_NUM_ELEMENTS_C 2
#    elif defined(KNL_7x7)
#        define KNL_FUNC_EXEC serial_run_sepa_2x2_7x7_exec
#        define KNL_INIT_FUNC median_filter_nxm_sepa_2x2_init
#        define KEEP_IDX_1_SIZE (14)
#        define KEEP_IDX_2_SIZE (8)
// 0, 3 use A; 1, 4 use B; 2, 5 use C.
#        define SEPA_2X2_NUM_ELEMENTS_A 2
#        define SEPA_2X2_NUM_ELEMENTS_B 2
#        define SEPA_2X2_NUM_ELEMENTS_C 2
#    endif

#    if defined(INTERLEAVED2C)
// Planar-transform parameters for channel-interleaved input: the tile is
// deinterleaved into two single-channel planes, the single-channel pipeline
// runs once per plane, and the planar outputs are re-interleaved. See
// MEDIAN_TILE_H_DF for the taller-tile geometry.
#        define CH_TILE_W (TILE_W / 2)
#        define CH_VECW pva_elementsof(VECX_TYPE)
#        define PLANE_ROWS (TILE_H_DF + KNL_H - 1)
#        define PLANE_COPY_W (CH_TILE_W + HALO_X)
#        define PLANE_ITERS_X ((PLANE_COPY_W + CH_VECW - 1) / CH_VECW)
#        define PLANE_LP (PLANE_ITERS_X * CH_VECW)
#    endif
#endif // SEPA_2x2

// vector type for kernel functions
// 1st pass
#if defined(KNL_3x3) || defined(INTERLEAVED2C)
#    define KNL_VEC_TYPE_1 DVECX_TYPE
#else
#    define KNL_VEC_TYPE_1 VECX_TYPE
#endif

// 2nd pass
#if defined(KNL_3x3)
#    define KNL_VEC_TYPE_2 DVECX_TYPE
#else
#    define KNL_VEC_TYPE_2 VECX_TYPE
#endif

// define data type and vector type
#if defined(U8_IO)
#    define BASE_TYPE unsigned char
#    define VECX_TYPE vcharx
#    define DVECX_TYPE dvcharx
#    define VEC_TYPE vuchar
#    define DVEC_TYPE dvuchar
#    define DVLOAD_DI dvuchar_load_di
#    define DVLOAD dvuchar_load
#    define VLOAD vuchar_load
#elif defined(S8_IO)
#    define BASE_TYPE char
#    define VECX_TYPE vcharx
#    define DVECX_TYPE dvcharx
#    define VEC_TYPE vchar
#    define DVEC_TYPE dvchar
#    define DVLOAD_DI dvchar_load_di
#    define DVLOAD dvchar_load
#    define VLOAD vchar_load
#elif defined(U16_IO)
#    define BASE_TYPE unsigned short
#    define VECX_TYPE vshortx
#    define DVECX_TYPE dvshortx
#    define VEC_TYPE vushort
#    define DVEC_TYPE dvushort
#    define DVLOAD_DI dvushort_load_di
#    define DVLOAD dvushort_load
#    define VLOAD vushort_load
#elif defined(S16_IO) || defined(I2S16_IO)
#    define BASE_TYPE short
#    define VECX_TYPE vshortx
#    define DVECX_TYPE dvshortx
#    define VEC_TYPE vshort
#    define DVEC_TYPE dvshort
#    define DVLOAD_DI dvshort_load_di
#    define DVLOAD dvshort_load
#    define VLOAD vshort_load
#elif defined(U32_IO)
#    define BASE_TYPE unsigned int
#    define VECX_TYPE vintx
#    define DVECX_TYPE dvintx
#    define VEC_TYPE vuint
#    define DVEC_TYPE dvuint
#    define DVLOAD_DI dvuint_load_di
#    define DVLOAD dvuint_load
#    define VLOAD vuint_load
#elif defined(S32_IO)
#    define BASE_TYPE int
#    define VECX_TYPE vintx
#    define DVECX_TYPE dvintx
#    define VEC_TYPE vint
#    define DVEC_TYPE dvint
#    define DVLOAD_DI dvint_load_di
#    define DVLOAD dvint_load
#    define VLOAD vint_load
#endif

#if defined(U8_IO) || defined(U16_IO) || defined(U32_IO)
#    define VEC_EXTEND zero_extend
#else
#    define VEC_EXTEND sign_extend
#endif

#endif // MEDIAN_FILTER_PARAM_H
