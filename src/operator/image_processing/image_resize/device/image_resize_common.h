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

#ifndef IMAGE_RESIZE_COMMON_H
#define IMAGE_RESIZE_COMMON_H

#include "image_resize_param.h"

#include <cupva_device.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    // parameters for sampler
    uint16_t _inputW;
    uint16_t _inputH;
    uint16_t _outputW;
    uint16_t _outputH;
    uint16_t _tableLp;
    uint8_t _bpp;

    // buffer for NN indices
    uint32_t *_lane_ids;
    uint32_t *_idxBuf;
    // buffer for BL transformation matrix
    int32_t *_transformMatrix;

    // for pre/post processing
    void *_preBuf;
    void *_postBuf;
    bool _transpose_flag;
    bool _preprocess_flag;
    bool _u8_flag;
    int _niter_pre;
    int _niter_post;

    AgenCFG _src_agen_cfg;
    AgenCFG _dst_agen_cfg;
    AgenCFG _pre_agen_cfg;
    AgenCFG _post_agen_cfg;
} IMAGE_RESIZE_CFG;

static inline void init_sampler_tile_info(uint16_t *srcTileSizesBuf, int32_t *srcTileStartsBuf, uint32_t *srcOffsetsBuf,
                                          int32_t nTile, int32_t dstTileSize, int32_t dstImageSize, int64_t scaleQHigh)
{
    for (int i = 0; i < nTile; i++) chess_no_hw_loop
    {
        int dst_start_idx = i * dstTileSize;
        int dst_end_idx   = (i + 1) * dstTileSize - 1;

        dst_end_idx = min(dst_end_idx, dstImageSize - 1);

        int64_t src_start_q_high = dst_start_idx * scaleQHigh + scaleQHigh / 2 - (1llu << QBIT_HIGH) / 2;
        int32_t src_start_idx    = (int32_t)(src_start_q_high >> QBIT_HIGH);

        int64_t src_end_q_high = dst_end_idx * scaleQHigh + scaleQHigh / 2 - (1llu << QBIT_HIGH) / 2;
        int32_t src_end_idx    = (int32_t)((src_end_q_high + (1llu << QBIT_HIGH) - 1) >> QBIT_HIGH);

        srcTileStartsBuf[i] = src_start_idx;
        srcTileSizesBuf[i]  = src_end_idx - src_start_idx + 1;
        srcOffsetsBuf[i]    = (int32_t)(src_start_q_high >> (QBIT_HIGH - QBIT)) - (src_start_idx << QBIT);
    }
}

void fixMultiChannelPadding(uint8_t *src, int32_t fixColPixel, int32_t refColPixel, uint8_t channelNum,
                            uint16_t srcTileHeight, uint16_t srcTileLp, int32_t leftFlag);

#ifdef __cplusplus
}
#endif

#endif // IMAGE_RESIZE_COMMON_H
