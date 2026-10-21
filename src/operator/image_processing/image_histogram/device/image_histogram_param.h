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

#ifndef IMAGE_HISTOGRAM_PARAM_H
#define IMAGE_HISTOGRAM_PARAM_H

#define TILE_W (128)
#define TILE_H (128)

/*
 * U8 counts raw sample values, so the table is 256 buckets whatever numBins is. At 32 ways that is
 * 16 KB of int16 counters, and a tile can add at most TILE_W * TILE_H / 32 to any one of them.
 */
#define HIST_U8_BUCKETS 256
#define HIST_U8_WAYS 32
#define HIST_U8_DRAIN_TILES (32767 / (TILE_W * TILE_H / HIST_U8_WAYS))

#define NUM_BINS_MAX_16WAY (2048)
#define NUM_BINS_MAX_8WAY (4096)
#define NUM_BINS_MAX_4WAY (8192)
#define NUM_BINS_MAX_2WAY (16384)
#define NUM_BINS_MAX (NUM_BINS_MAX_2WAY)

enum ImageHistogramImageFormat
{
    IMAGE_FORMAT_U8 = 1,
    IMAGE_FORMAT_U16
};

#endif // IMAGE_HISTOGRAM_PARAM_H
