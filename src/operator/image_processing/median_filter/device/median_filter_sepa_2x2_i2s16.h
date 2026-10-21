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

#ifndef MEDIAN_FILTER_SEPA_2X2_I2S16_H
#define MEDIAN_FILTER_SEPA_2X2_I2S16_H

#include "median_filter_common_sepa_2x2.h"

// State of the planar pre/post copy passes for channel-interleaved input.
// Kept separate from MedianFilterSepa2x2Config: the copy passes handle both
// channels in one pass, so this state is per tile, not per channel.
typedef struct MedianFilterSepa2x2PlanarConfig
{
    AgenCFG pre_src_agen_cfg;
    AgenCFG post_dst_agen_cfg;
    BASE_TYPE *inCh0;
    BASE_TYPE *inCh1;
    BASE_TYPE *outCh0;
    BASE_TYPE *outCh1;
    int pre_pass_niter;
    int post_pass_niter;
} MedianFilterSepa2x2PlanarConfig;

// Planar transform for channel-interleaved input: the deinterleave pass reads
// the source tile through a circular-buffer agen and writes the two channel
// planes through raw superbank pointers; the re-interleave pass reads the
// planar outputs through raw pointers and stores to the destination tile
// through an agen. srcLp/srcCbLen are in BASE_TYPE elements (already scaled
// by CHANNEL_NUM).
static inline void median_filter_sepa_2x2_planar_init(BASE_TYPE *srcBuf, int srcLp, int srcCbLenBytes,
                                                      BASE_TYPE *plane0, BASE_TYPE *plane1, BASE_TYPE *outPlane0,
                                                      BASE_TYPE *outPlane1, BASE_TYPE *dst, int dstLp,
                                                      MedianFilterSepa2x2PlanarConfig &cfg)
{
    // pre-pass: one deinterleaving double-vector load per vecw pixels, whole
    // halo'd footprint
    AgenWrapper w;
    w.size = sizeof(BASE_TYPE);
    w.n1   = PLANE_ITERS_X;
    w.s1   = 2 * CH_VECW;
    w.n2   = PLANE_ROWS;
    w.s2   = srcLp;

    agen a_src = init((DVECX_TYPE *)srcBuf);
    INIT_AGEN2(a_src, w);
    a_src                = update_agen_cb_start(a_src, (intptr_t)srcBuf);
    a_src                = update_agen_cb_size(a_src, srcCbLenBytes);
    cfg.pre_src_agen_cfg = extract_agen_cfg(a_src);

    // The copy loops are unrolled by 6; round the trip counts up.
    cfg.pre_pass_niter = max(18, (int)((PLANE_ITERS_X * PLANE_ROWS + 5) / 6 * 6));

    // post-pass: planar outputs -> interleaved destination tile
    AgenWrapper wd;
    wd.size = sizeof(BASE_TYPE);
    wd.n1   = CH_TILE_W / CH_VECW;
    wd.s1   = 2 * CH_VECW;
    wd.n2   = TILE_H_DF;
    wd.s2   = dstLp;

    agen a_dst = init((DVECX_TYPE *)dst);
    INIT_AGEN2(a_dst, wd);
    cfg.post_dst_agen_cfg = extract_agen_cfg(a_dst);

    cfg.post_pass_niter = max(18, (int)(((CH_TILE_W / CH_VECW) * TILE_H_DF + 5) / 6 * 6));

    // plane buffers accessed by raw pointer in the copy passes
    cfg.inCh0  = plane0;
    cfg.inCh1  = plane1;
    cfg.outCh0 = outPlane0;
    cfg.outCh1 = outPlane1;
}

// Copy-pass loops (defined in median_filter_sepa_2x2_i2s16.c so they remain
// separate functions in profiles).
void i2s16_deinterleave_src_exec(const MedianFilterSepa2x2PlanarConfig &cfg);
void i2s16_interleave_dst_exec(const MedianFilterSepa2x2PlanarConfig &cfg);

#endif /* MEDIAN_FILTER_SEPA_2X2_I2S16_H */
