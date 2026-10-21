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

#ifndef ORB_DESC_EXTR_PARAMS_H
#define ORB_DESC_EXTR_PARAMS_H

#if defined(U8_IO)
#    define DATA_TYPE uint8_t
#    define VTYPE vuchar
#    define DVTYPE dvuchar
#    define DVTYPEX dvcharx
#    define VLOAD_TRANSP vuchar_load_transp
#    define DVLOAD_DI dvuchar_load_di
#elif defined(S8_IO)
#    define DATA_TYPE int8_t
#    define VTYPE vchar
#    define DVTYPE dvchar
#    define DVTYPEX dvcharx
#    define VLOAD_TRANSP vchar_load_transp
#    define DVLOAD_DI dvchar_load_di
#elif defined(U16_IO)
#    define DATA_TYPE uint16_t
#    define VTYPE vushort
#    define DVTYPE dvushort
#    define DVTYPEX dvshortx
#    define DVLOAD_TRANSP2 dvushort_load_transp2_di
#    define DVLOAD_DI dvushort_load_di
#elif defined(S16_IO)
#    define DATA_TYPE int16_t
#    define VTYPE vshort
#    define DVTYPE dvshort
#    define DVTYPEX dvshortx
#    define DVLOAD_TRANSP2 dvshort_load_transp2_di
#    define DVLOAD_DI dvshort_load_di
#endif

#if defined(U8_IO) || defined(S8_IO)
#    define X8_IO (1)
#elif defined(U16_IO) || defined(S16_IO)
#    define X16_IO (1)
#endif

#define BATCH_SIZE_8_BIT (32)
#define BATCH_SIZE_16_BIT (16)

#if defined(X8_IO)
#    define WTS_TYPE int8_t
#    define WTS_VTYPE dvchar
#    define WTS_PERM_TYPE int8_t
#    define BATCH_SIZE BATCH_SIZE_8_BIT
#    define TMP_DESC_BUF_TYPE uint32_t
#elif defined(X16_IO)
#    define WTS_TYPE int8_t
#    define WTS_VTYPE vchar
#    define WTS_PERM_TYPE int16_t
#    define BATCH_SIZE BATCH_SIZE_16_BIT
#    define TMP_DESC_BUF_TYPE uint16_t
#endif

// 2 * BATCH_SIZE / pva_elementsof(dvfloat)
#define TRUNC_FLOAT_TO_INT_XY_NITER BATCH_SIZE / 8
// BATCH_SIZE / pva_elementsof(dvint)
#define GET_ROTATION_ANGLE_NITER BATCH_SIZE / 16

// radius for moments: 15
// radius for orb patterns: 19
#define PATCH_WIDTH (39)
#define PATCH_HEIGHT (39)

// align batched image patches lp for transpose load
// for u8: lp ~ 64k + 2
// for u16: lp ~ 32k + 2
#define TABLE_BATCH_STRIDE ((PATCH_WIDTH * PATCH_HEIGHT - 2 + 63) / 64 * 64 + 2)
// 2D lookup table offset, must align to 64 bytes
#define LOOKUP_TABLE_OFFSET (TABLE_BATCH_STRIDE - 2)
// align rotated orb patterns lp to 32k+2 for hf T2 store
#define LOOKUP_IDX_LP ((DESCRIPTOR_PATTERN_SIZE * 2 - 2 + 31) / 32 * 32 + 2)

#define ANGLE_QBITS (6)
#define DESCRIPTOR_ARRAY_NUM_BITS (256)
#define DESCRIPTOR_PATTERN_SIZE (512)
#define DESCRIPTOR_ARRAY_NUM_BYTES (DESCRIPTOR_ARRAY_NUM_BITS / 8)
// 32x31x2
#define WEIGHTS_ARRAY_SIZE (1984)

#if defined(KNL_RBRIEF)
#    define OFFSET_TYPE uint16_t
#else
#    define OFFSET_TYPE uint8_t
#endif

#endif // ORB_DESC_EXTR_PARAMS_H
