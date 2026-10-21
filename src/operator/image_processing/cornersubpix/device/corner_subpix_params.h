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

#ifndef _CORNER_SUBPIX_PARAMS_H_
#define _CORNER_SUBPIX_PARAMS_H_

#define WIN_PITCH(WIN_SIZE) (WIN_SIZE + 2)
#define TOL (3)                                                        // 2k + 1
#define TILE_SIZE(WIN_SIZE) (WIN_PITCH(WIN_SIZE) + WIN_SIZE + TOL * 2) // 4k + 2
#define TILE_PITCH(WIN_SIZE) ((TILE_SIZE(WIN_SIZE) * TILE_SIZE(WIN_SIZE) + 31) / 32 * 32)

#define BATCH 32 // fixed value
#define BATCH_PITCH (BATCH + 1)
#define QBITS 15
#define QBITS_IMG 8
#define PREFETCH_ITER 5 // to prefetch next pong batch, >= 5 from rtl

#define MAX_CORNERS 4096 // or value from application requirement
#define MAX_TILE_SIZE 26
#define MAX_TILE_PITCH ((MAX_TILE_SIZE * MAX_TILE_SIZE + 31) / 32 * 32)
#define MAX_WIN_PITCH 11
#define MAX_WIN_SIZE 9

#endif