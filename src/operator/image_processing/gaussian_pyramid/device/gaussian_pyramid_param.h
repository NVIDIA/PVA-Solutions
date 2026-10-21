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

#ifndef GAUSSIAN_PYRAMID_PARAM_H
#define GAUSSIAN_PYRAMID_PARAM_H

#include <stdint.h>

#if defined(DATA_TYPE_U8) || defined(DATA_TYPE_S8)
#    define VDOTP4x2 vdotp4x2_bbh
#    define VSTORE_I vstore_i_hb
#    define REPLICATE replicateb
#elif defined(DATA_TYPE_U16) || defined(DATA_TYPE_S16)
#    define VDOTP4x2 vdotp4x2_hhw
#    define VSTORE_I vstore_i_wh
#    define REPLICATE replicateh
#endif

#if defined(DATA_TYPE_U8) || defined(DATA_TYPE_S8)
#    define DIN_VEC_TYPE dvcharx
#    define DACC_VEC_TYPE dvshortx
#    define VECW pva_elementsof(dvcharx)
#    define IN_MEM_TYPE dvchar
#    define OUT_MEM_TYPE dvchar
#    if defined(DATA_TYPE_S8)
#        define IN_LOAD_FUNC dvchar_load_di
#        define DIO_LOAD_FUNC dvchar_load
#        define BASE_TYPE int8_t
#    elif defined(DATA_TYPE_U8)
#        define IN_LOAD_FUNC dvuchar_load_di
#        define DIO_LOAD_FUNC dvuchar_load
#        define BASE_TYPE uint8_t
#    endif
#elif defined(DATA_TYPE_U16) || defined(DATA_TYPE_S16)
#    define DIN_VEC_TYPE dvshortx
#    define DACC_VEC_TYPE dvintx
#    define VECW pva_elementsof(dvshortx)
#    define IN_MEM_TYPE dvshort
#    define OUT_MEM_TYPE dvshort
#    if defined(DATA_TYPE_S16)
#        define IN_LOAD_FUNC dvshort_load_di
#        define DIO_LOAD_FUNC dvshort_load
#        define BASE_TYPE int16_t
#    elif defined(DATA_TYPE_U16)
#        define IN_LOAD_FUNC dvushort_load_di
#        define DIO_LOAD_FUNC dvushort_load
#        define BASE_TYPE uint16_t
#    endif
#endif

#if defined(DATA_TYPE_U8) || defined(DATA_TYPE_S8)
#    define DKNL_VEC_TYPE dvcharx
#    define KNL_MEM_TYPE dvchar
#    if defined(DATA_TYPE_S8)
#        define DKNL_LOAD_FUNC dvchar_load
#    elif defined(DATA_TYPE_U8)
#        define DKNL_LOAD_FUNC dvuchar_load
#    endif
#elif defined(DATA_TYPE_U16) || defined(DATA_TYPE_S16)
#    define DKNL_VEC_TYPE dvshortx
#    define KNL_MEM_TYPE dvshort
#    if defined(DATA_TYPE_S16)
#        define DKNL_LOAD_FUNC dvshort_load
#    elif defined(DATA_TYPE_U16)
#        define DKNL_LOAD_FUNC dvushort_load
#    endif
#endif

// Default VECW definition if no data type is specified
#ifndef VECW
#    define VECW 32 // Default vector width
#endif

#define QBITS 8

// kernel width
// kernel radius
// halo width of a tile for kernel
#define KW 5
#define KR (KW >> 1)
#define KP (KR << 1)

// level0 input
#define TW0 (244)
#define TH0 (64)
// level1 output
#define TW1 (TW0 >> 1)
#define TH1 (TH0 >> 1)
// level2 output
#define TW2 (TW0 >> 2)
#define TH2 (TH0 >> 2)

#define KERNEL_TH0 (8)
#define KERNEL_TH1 (KERNEL_TH0 >> 1)
#define KERNEL_TH2 (KERNEL_TH1 >> 1)

#define KERNEL_TH0_SPLIT_BUF (KERNEL_TH0 >> 1)
#define KERNEL_TH1_SPLIT_BUF (KERNEL_TH1 >> 1)

#define NITER_Y_TILE0 ((TH0 + 2 * HALO_L0 + 7) / KERNEL_TH0)
#define NITER_Y_TILE1 ((TH1 + 2 * HALO_L1 + 3) / KERNEL_TH1)

// ------------------------------------------- //
// Kernel dims
// ------------------------------------------- //
#define KERNEL_W (5)
#define KERNEL_HALO (KERNEL_W >> 1)

// ------------------------------------------- //
// Halo needed for each level
// ------------------------------------------- //
#define DOWNSAMPLE_RATE (2)
#define HALO_L1 (KERNEL_HALO)
#define HALO_L0 (HALO_L1 * DOWNSAMPLE_RATE + KERNEL_HALO)

// ----------------------------------------------------------- //
// Adjustments to halo and tile heights in the buffers split
// between even and odd in separate superbanks
// ----------------------------------------------------------- //
#define HALO_L1_Y_SPLIT_BUF (HALO_L1 >> 1)
#define HALO_L0_Y_SPLIT_BUF (HALO_L0 >> 1)
#define TH0_SPLIT_BUF (TH0 >> 1)
#define TH1_SPLIT_BUF (TH1 >> 1)

#define L0_LINE_PITCH_ALIGN_VECW (((TW0 + 2 * HALO_L0) + VECW - 1) / VECW * VECW) // In pixels
#define L1_LINE_PITCH_ALIGN_VECW (((TW1 + 2 * HALO_L1) + VECW - 1) / VECW * VECW) // In pixels
#define L2_LINE_PITCH_ALIGN_VECW (((TW2) + VECW - 1) / VECW * VECW)               // In pixels

// ----------------------------------------------------------- //
// Adjustments to tile arena
// ----------------------------------------------------------- //
// We will output more data at L1 than we need to write back
// to DRAM, since part of this is needed for L2 only.
// To account for this, we will set up tileArena for L1 output.
// The tileArena will always apply to the bottom/right of the tile, meaning
// we need to add a negative adjustment here.
#define L1_TILE_ARENA_X (L1_LINE_PITCH_ALIGN_VECW)                    // In pixels
#define L1_TILE_ARENA_Y (TH1_SPLIT_BUF + 2 * HALO_L1_Y_SPLIT_BUF + 2) // In pixels

#define L2_TILE_ARENA_X (L2_LINE_PITCH_ALIGN_VECW) // In pixels
#define L2_TILE_ARENA_Y (TH2 + 2)                  // In pixels

#define ALIGN_64(x) (((x) + 63) & ~63) // Adjustment needs to be 64B aligned
#define L1_TILE_ADJUSTMENT_MAX ALIGN_64(HALO_L1 + L1_TILE_ARENA_X * HALO_L1_Y_SPLIT_BUF) // In pixels

#endif // GAUSSIAN_PYRAMID_PARAM_H
