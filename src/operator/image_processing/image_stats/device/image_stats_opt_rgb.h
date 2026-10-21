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

#ifndef IMAGE_STATS_OPT_RGB_H
#define IMAGE_STATS_OPT_RGB_H

#include "image_stats_common.h"
#include "image_stats_param.h"

void init_image_stats_rgb_cfg(IMAGE_STATS_CFG &cfg, uint8_t *srcPtrs[MAX_PLANE_COUNT], uint8_t *mask,
                              uint16_t *tileWidth, uint16_t *tileHeight, uint16_t *tileLp, uint16_t maskLp);

void image_stats_opt_rgb_exec(IMAGE_STATS_CFG &cfg, uint8_t *srcPtrs[MAX_PLANE_COUNT]);

void image_stats_opt_mask_rgb_exec(IMAGE_STATS_CFG &cfg, uint8_t *srcPtrs[MAX_PLANE_COUNT], uint8_t *mask);

#endif // IMAGE_STATS_OPT_RGB_H
