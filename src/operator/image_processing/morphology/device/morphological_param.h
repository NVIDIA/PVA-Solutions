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

#ifndef MORPHOLOGICAL_PVA_PARAM_H
#define MORPHOLOGICAL_PVA_PARAM_H

#define TILE_W 128
#define TILE_H 128

#if defined(ERODE_OP)
#    define MORPH_FIL_OP vmin3
#elif defined(DILATE_OP)
#    define MORPH_FIL_OP vmax3
#endif

#ifdef KSIZE_3X3
#    define KWIDTH 3
#    define KHEIGHT 3
#endif

#ifdef KSIZE_5X5
#    define KWIDTH 5
#    define KHEIGHT 5
#endif

#define HALO_X (KWIDTH >> 1)
#define HALO_Y (KHEIGHT >> 1)

#ifdef U8_IO
#    define BASE_TYPE unsigned char

#    define VEC_TYPE vcharx
#    define DVEC_TYPE dvcharx
#    define DMEM_TYPE dvuchar
#    define DLOAD_DI_TYPE dvuchar_load_di
#    define DLOAD_TYPE dvuchar_load
#    define UNROLL_FACTOR 2
#endif

#ifdef S8_IO
#    define BASE_TYPE char

#    define VEC_TYPE vcharx
#    define DVEC_TYPE dvcharx
#    define DMEM_TYPE dvchar
#    define DLOAD_DI_TYPE dvchar_load_di
#    define DLOAD_TYPE dvchar_load
#    define UNROLL_FACTOR 2
#endif

#ifdef U16_IO
#    define BASE_TYPE unsigned short

#    define VEC_TYPE vshortx
#    define DVEC_TYPE dvshortx
#    define DMEM_TYPE dvushort
#    define DLOAD_DI_TYPE dvushort_load_di
#    define DLOAD_TYPE dvushort_load
#    define UNROLL_FACTOR 1
#endif

#ifdef S16_IO
#    define BASE_TYPE short

#    define VEC_TYPE vshortx
#    define DVEC_TYPE dvshortx
#    define DMEM_TYPE dvshort
#    define DLOAD_DI_TYPE dvshort_load_di
#    define DLOAD_TYPE dvshort_load
#    define UNROLL_FACTOR 1
#endif

#define DVECW pva_elementsof(DVEC_TYPE)
#define INNER_LOOPCOUNT ((TILE_W / DVECW) * TILE_H / 2)

#endif // MORPHOLOGICAL_PVA_PARAM_H
