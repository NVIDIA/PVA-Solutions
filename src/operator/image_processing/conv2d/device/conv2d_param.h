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

#ifndef _CONV_PARAM_H_
#define _CONV_PARAM_H_

#ifdef U8_IO
#    define DATA_TYPE uint8_t
#    define KERNEL_TYPE uint8_t
#endif

#ifdef S8_IO
#    define DATA_TYPE int8_t
#    define KERNEL_TYPE uint8_t
#endif

#ifdef U16_IO
#    define DATA_TYPE uint16_t
#    define KERNEL_TYPE uint16_t
#endif

#ifdef S16_IO
#    define DATA_TYPE int16_t
#    define KERNEL_TYPE uint16_t
#endif

#ifdef KSIZE_3X3
#    define KW 3
#    define KH 3
#endif
#ifdef KSIZE_5X5
#    define KW 5
#    define KH 5
#endif
#ifdef KSIZE_7X7
#    define KW 7
#    define KH 7
#endif

#ifndef TW
#    define TW 128
#endif
#ifndef TH
#    define TH 128
#endif

#define QBITS_8 8
#define QBITS_16 16

#define KWR (KW >> 1)
#define KHR (KH >> 1)
#define KWA (((KW - 1) / 4 + 1) * 4)
#define KHA (((KH - 1) / 2 + 1) * 2)

#define THA (((((TH / 2) - 1) / 3) + 1) * 3) // TH = 128 while 3x3 u8 outputs 6 rows in each iteration

#endif
