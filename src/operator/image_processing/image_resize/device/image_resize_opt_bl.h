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

#ifndef IMAGE_RESIZE_OPT_BL_H
#define IMAGE_RESIZE_OPT_BL_H

#include "image_resize_common.h"
#include "image_resize_param.h"

void init_bl_resize_cfg(IMAGE_RESIZE_CFG &cfg, void *srcBuf, void *dstBuf, void *preBuf, void *postBuf,
                        uint32_t *idxBuf, uint32_t *laneIds, int32_t *transformMat, uint16_t srcTileWidth,
                        uint16_t srcTileHeight, uint16_t srcTileLp, uint16_t dstTileWidth, uint16_t dstTileHeight,
                        uint16_t dstTileLp, uint8_t channelNum, bool u8Flag);

void init_bl_sampler_cfg(IMAGE_RESIZE_CFG &cfg, CupvaSampler &sampler, uint32_t stepCol, uint32_t stepRow);

void update_bl_sampler_addrs(IMAGE_RESIZE_CFG &cfg, CupvaSampler &sampler, void *src, void *dst);

void update_bl_sampler_offsets(IMAGE_RESIZE_CFG &cfg, CupvaSampler &sampler, int32_t ofstCol, int32_t ofstRow);

// pre/post processing interface
void pre_knl_bl_exec(IMAGE_RESIZE_CFG &cfg);
void post_knl_bl_exec(IMAGE_RESIZE_CFG &cfg);

#endif // IMAGE_RESIZE_OPT_BL_H
