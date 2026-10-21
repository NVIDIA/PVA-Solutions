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

#ifndef IMAGE_RESIZE_OPT_NN_H
#define IMAGE_RESIZE_OPT_NN_H

#include "image_resize_common.h"
#include "image_resize_param.h"

void init_nn_resize_cfg(IMAGE_RESIZE_CFG &cfg, void *srcBuf, void *dstBuf, void *preBuf, void *postBuf,
                        uint32_t *idxBuf, uint32_t *laneIds, int32_t *transformMat, uint16_t srcTileWidth,
                        uint16_t srcTileHeight, uint16_t srcTileLp, uint16_t dstTileWidth, uint16_t dstTileHeight,
                        uint16_t dstTileLp, uint8_t channelNum, bool u8Flag);

void init_nn_sampler_cfg(IMAGE_RESIZE_CFG &cfg, CupvaSampler &sampler, uint32_t stepCol, uint32_t stepRow);

void generate_nn_sampler_tile_indices(IMAGE_RESIZE_CFG &cfg, uint32_t stepX, uint32_t stepY, uint32_t offsetX,
                                      uint32_t offsetY);

void update_nn_sampler_addrs(IMAGE_RESIZE_CFG &cfg, CupvaSampler &sampler, void *src, void *dst);

void update_nn_sampler_offsets(IMAGE_RESIZE_CFG &cfg, CupvaSampler &sampler, int32_t ofstCol, int32_t ofstRow);

// multi-channel u8: split/merge channels
void split_channels_u8_nn_exec(IMAGE_RESIZE_CFG &cfg);
void merge_channels_u8_nn_exec(IMAGE_RESIZE_CFG &cfg);

#endif // IMAGE_RESIZE_OPT_NN_H
