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

#ifndef BILATERAL_FILTER_PARAM_H
#define BILATERAL_FILTER_PARAM_H

#ifdef KSIZE_3X3
#    define KERNEL_SIZE 3
#    define RADIUS 1
#    define SPACE2_PARAMS_SIZE 2
#endif
#ifdef KSIZE_5X5
#    define KERNEL_SIZE 5
#    define RADIUS 2
#    define SPACE2_PARAMS_SIZE 4
#endif
#ifdef KSIZE_7X7
#    define KERNEL_SIZE 7
#    define RADIUS 3
#    define SPACE2_PARAMS_SIZE 7
#endif

#define TILE_WIDTH 64
#define TILE_HEIGHT 256

#define COLOR_8BIT 256
#define QBITS 7

#endif
