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

#ifndef IMAGE_STATS_PARAM_H
#define IMAGE_STATS_PARAM_H

#define TILE_WIDTH (128)
#define TILE_HEIGHT (128)
#define MAX_PLANE_COUNT (2)
#define MAX_CHANNEL_NUM (3)
// output size: 4 * (1 + 4 + 4 + 16) = 100
#define IMAGE_STAT_OUTPUT_SIZE (100)

// fixed-point qbits in post-processing
#define QBITS (24)

typedef enum
{
    IMAGE_STAT_EXEC_CNT = 0,
    IMAGE_STAT_EXEC_SUM,
    IMAGE_STAT_EXEC_CNT_SUM,
    IMAGE_STAT_EXEC_VAR,
    IMAGE_STAT_EXEC_COV
} PVAImageStatOpType;

#endif /* IMAGE_STATS_PARAM_H */