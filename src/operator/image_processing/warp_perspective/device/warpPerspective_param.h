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

#ifndef WARP_PERSPECTIVE_PVA_PARAM_H
#define WARP_PERSPECTIVE_PVA_PARAM_H

#define TILE_W 64
#define TILE_H 54
#define QBITS 16
// input is variable, we assign the half superbank(64KB) for double buffer
#define HALF_SUPERBANK_BYTE_SIZE (64 * 1024)
#define MAX_TABLE_SIZE (HALF_SUPERBANK_BYTE_SIZE / 2)
#define MAX_TABLE_SP_SIZE (HALF_SUPERBANK_BYTE_SIZE / 4)

typedef enum
{
    /// Nearest neighbor
    NN = 0,
    /// Bilinear interpolation
    BILINEAR = 1,
} InterpTypeDev;

typedef enum
{
    /// Perspective transformation
    PERSPECTIVE = 0,
    /// Affine transformation
    AFFINE = 1,
} TransformTypeDev;

#endif // WARP_PERSPECTIVE_PVA_PARAM_H