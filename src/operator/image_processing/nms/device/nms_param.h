/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NMS_PARAM_H
#define NMS_PARAM_H

// Window size, selected by the per-variant build flag. Only used by the device compilation; the host
// side derives the radius from the run-time nmsSize argument.
#ifdef NMS_SIZE_3
#    define NMS_SIZE 3
#endif
#ifdef NMS_SIZE_5
#    define NMS_SIZE 5
#endif

// Valid-output tile. TILE_HEIGHT is kept a multiple of 16 because the primitive over-runs its inner
// loops to the unroll factor and writes the output in blocks of 16 rows.
#ifndef TILE_WIDTH
#    define TILE_WIDTH (64)
#endif
#ifndef TILE_HEIGHT
#    define TILE_HEIGHT (64)
#endif

#ifdef NMS_SIZE
#    define NMS_RADIUS (NMS_SIZE >> 1)

// Padded input tile handed to the primitive.
#    define SRC_TILE_WIDTH (TILE_WIDTH + (2 * NMS_RADIUS))
#    define SRC_TILE_HEIGHT (TILE_HEIGHT + (2 * NMS_RADIUS))

// Must match pvaApl::NmsVpu<NMS_SIZE>::ScratchElements(SRC_TILE_WIDTH, SRC_TILE_HEIGHT).
#    define NMS_SCRATCH_ELEMS ((((SRC_TILE_WIDTH * SRC_TILE_HEIGHT) + 15) / 16) * 16)
#endif

#endif // NMS_PARAM_H
