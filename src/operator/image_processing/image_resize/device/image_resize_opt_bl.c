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

#include "image_resize_opt_bl.h"

void init_bl_resize_cfg(IMAGE_RESIZE_CFG &cfg, void *srcBuf, void *dstBuf, void *preBuf, void *postBuf,
                        uint32_t *idxBuf, uint32_t *laneIds, int32_t *transformMat, uint16_t srcTileWidth,
                        uint16_t srcTileHeight, uint16_t srcTileLp, uint16_t dstTileWidth, uint16_t dstTileHeight,
                        uint16_t dstTileLp, uint8_t channelNum, bool u8Flag)
{
    cfg._transpose_flag  = channelNum > 1;
    cfg._preprocess_flag = u8Flag;
    cfg._bpp             = 2;

    cfg._preBuf          = preBuf;
    cfg._postBuf         = postBuf;
    cfg._transformMatrix = transformMat;

    cfg._inputW  = cfg._transpose_flag ? srcTileHeight : srcTileWidth;
    cfg._inputH  = cfg._transpose_flag ? srcTileWidth : srcTileHeight;
    cfg._outputW = cfg._transpose_flag ? dstTileHeight : dstTileWidth;
    cfg._outputH = cfg._transpose_flag ? dstTileWidth : dstTileHeight;

    // align table line pitch to 32k + 2 to reduce bank conflict
    cfg._tableLp = cfg._preprocess_flag ? (cfg._inputW - 2 + 31) / 32 * 32 + 2 : cfg._inputW;

    if (!cfg._preprocess_flag)
    {
        return;
    }

    // init parameters for pre/post processing
    int vecw = pva_elementsof(vcharx);

    if (channelNum == 1)
    {
        // src
        AgenWrapper w0;
        w0.size      = sizeof(uint8_t);
        w0.n1        = (srcTileWidth + vecw - 1) / vecw;
        w0.s1        = vecw;
        w0.n2        = srcTileHeight;
        w0.s2        = srcTileLp;
        agen srcAgen = init((vuchar *)srcBuf);
        INIT_AGEN2(srcAgen, w0);

        cfg._niter_pre = w0.n1 * w0.n2;

        // preBuf: U16 src
        w0.size      = sizeof(uint16_t);
        w0.s2        = cfg._tableLp;
        agen preAgen = init((dvushort *)preBuf);
        INIT_AGEN2(preAgen, w0);

        // postBuf: U16 dst
        AgenWrapper w1;
        w1.size       = sizeof(uint16_t);
        w1.n1         = (dstTileWidth + vecw - 1) / vecw;
        w1.s1         = vecw;
        w1.n2         = dstTileHeight;
        w1.s2         = dstTileWidth;
        agen postAgen = init((dvushort *)postBuf);
        INIT_AGEN2(postAgen, w1);

        // dst
        w1.size      = sizeof(uint8_t);
        w1.s2        = dstTileLp;
        agen dstAgen = init((vuchar *)dstBuf);
        INIT_AGEN2(dstAgen, w1);

        cfg._niter_post = w1.n1 * w1.n2;

        cfg._src_agen_cfg  = extract_agen_cfg(srcAgen);
        cfg._dst_agen_cfg  = extract_agen_cfg(dstAgen);
        cfg._pre_agen_cfg  = extract_agen_cfg(preAgen);
        cfg._post_agen_cfg = extract_agen_cfg(postAgen);
    }
    else
    {
        // src: transpose load
        AgenWrapper w0;
        w0.size      = sizeof(uint8_t);
        w0.n1        = (srcTileHeight + vecw - 1) / vecw;
        w0.s1        = srcTileLp * vecw;
        w0.n2        = srcTileWidth;
        w0.s2        = channelNum;
        agen srcAgen = init((vuchar *)srcBuf);
        INIT_AGEN2(srcAgen, w0);

        int srcLaneOfst   = (srcTileLp - 2) / 64;
        srcAgen.lane_ofst = srcLaneOfst;

        cfg._niter_pre = w0.n1 * w0.n2;

        // preBuf: transposed single channel image, normal store
        w0.size        = sizeof(uint16_t);
        w0.s1          = vecw;
        w0.s2          = cfg._tableLp;
        agen srcCxAgen = init((dvushort *)preBuf);
        INIT_AGEN2(srcCxAgen, w0);

        // postBuf: transposed single channel output, normal load
        AgenWrapper w1;
        w1.size        = sizeof(uint16_t);
        w1.n1          = (dstTileHeight + vecw - 1) / vecw;
        w1.s1          = vecw;
        w1.n2          = dstTileWidth;
        w1.s2          = dstTileHeight;
        agen dstCxAgen = init((dvushort *)postBuf);
        INIT_AGEN2(dstCxAgen, w1);

        // dst: normal HWC output, transpose store
        w1.size      = sizeof(uint8_t);
        w1.s1        = dstTileLp * vecw;
        w1.s2        = channelNum;
        agen dstAgen = init((vuchar *)dstBuf);
        INIT_AGEN2(dstAgen, w1);

        int dstLaneOfst   = (dstTileLp - 2) / 64;
        dstAgen.lane_ofst = dstLaneOfst;

        cfg._niter_post = w1.n1 * w1.n2;

        cfg._src_agen_cfg  = extract_agen_cfg(srcAgen);
        cfg._pre_agen_cfg  = extract_agen_cfg(srcCxAgen);
        cfg._dst_agen_cfg  = extract_agen_cfg(dstAgen);
        cfg._post_agen_cfg = extract_agen_cfg(dstCxAgen);
    }
}

void init_bl_sampler_cfg(IMAGE_RESIZE_CFG &cfg, CupvaSampler &sampler, uint32_t stepCol, uint32_t stepRow)
{
    // init bilinear interpolation transformation matrix scaling factors
    cfg._transformMatrix[2] = stepCol;
    cfg._transformMatrix[3] = stepRow;

    // init CupvaSampler
    CupvaSamplerInput2D input = {
        .data           = NULL,
        .type           = SAMPLER_INPUT_TYPE_U16,
        .width          = (uint32_t)cfg._inputW - 1,
        .height         = (uint32_t)cfg._inputH - 1,
        .linePitch      = (uint32_t)cfg._tableLp,
        .outOfRangeMode = SAMPLER_OUT_OF_RANGE_PREDICATE_OFF,
    };
    CupvaSamplerOutput output = {
        .data      = NULL,
        .transMode = TRANS_MODE_NONE,
    };
    CupvaSamplerTiles tiles = {
        .data              = cfg._transformMatrix,
        .type              = SAMPLER_TILE_TRANS_SCALE,
        .width             = (uint8_t)cfg._outputW,
        .height            = (uint8_t)cfg._outputH,
        .count             = 1,
        .fractionalBits    = (uint8_t)QBIT,
        .offsetX           = 0,
        .offsetY           = 0,
        .tileTraversalMode = SAMPLER_TRAVERSAL_STANDARD,
    };
    cupvaSamplerSetup(&sampler, &input, &tiles, &output);

    if (cfg._preprocess_flag)
    {
        update_bl_sampler_addrs(cfg, sampler, cfg._preBuf, cfg._postBuf);
    }
}

void update_bl_sampler_addrs(IMAGE_RESIZE_CFG &cfg, CupvaSampler &sampler, void *src, void *dst)
{
    {
        cupvaSamplerUpdateAddrs(&sampler, src, cfg._transformMatrix, dst);

#if CUPVA_BUILD_MODE == CUPVA_NATIVE
        // manually fix address range check in native mode
        sampler.inputAddrStart  = src;
        sampler.inputAddrEnd    = (void *)((uintptr_t)src + cfg._tableLp * cfg._inputH * cfg._bpp);
        sampler.outputAddrStart = dst;
        sampler.outputAddrEnd   = (void *)((uintptr_t)dst + cfg._outputW * cfg._outputH * cfg._bpp);
#endif
    }
}

void update_bl_sampler_offsets(IMAGE_RESIZE_CFG &cfg, CupvaSampler &sampler, int32_t ofstCol, int32_t ofstRow)
{
    cfg._transformMatrix[0] = ofstCol;
    cfg._transformMatrix[1] = ofstRow;
}

void promote_u8_table_bl_exec(IMAGE_RESIZE_CFG &cfg)
{
    agen_B a_src = init_agen_B_from_cfg(cfg._src_agen_cfg);
    agen_C a_pre = init_agen_C_from_cfg(cfg._pre_agen_cfg);

    int niter = max(cfg._niter_pre, 24);
    niter     = (niter + 7) / 8 * 8;

    chess_separator_scheduler();

    for (int i = 0; i < niter; i++) chess_loop_range(24, )
    chess_unroll_loop(8)
    chess_prepare_for_pipelining
    {
        vcharx b_pre = vuchar_load(a_src);
        vstore_bh(b_pre, a_pre);
    }
}

void demote_u16_results_bl_exec(IMAGE_RESIZE_CFG &cfg)
{
    agen_B a_post = init_agen_B_from_cfg(cfg._post_agen_cfg);
    agen_C a_dst  = init_agen_C_from_cfg(cfg._dst_agen_cfg);

    int niter = max(cfg._niter_post, 24);
    niter     = (niter + 7) / 8 * 8;

    chess_separator_scheduler();

    for (int i = 0; i < niter; i++) chess_loop_range(24, )
    chess_unroll_loop(8)
    chess_prepare_for_pipelining
    {
        dvshortx dh_post = dvushort_load(a_post);
        vstore_hb(dh_post, a_dst);
    }
}

void split_channels_promote_u8_table_bl_exec(IMAGE_RESIZE_CFG &cfg)
{
    agen_B srcAgen = init_agen_B_from_cfg(cfg._src_agen_cfg);
    agen_C preAgen = init_agen_C_from_cfg(cfg._pre_agen_cfg);

    int niter = max(cfg._niter_pre, 24);
    niter     = (niter + 7) / 8 * 8;

    chess_separator_scheduler();

    for (int i = 0; i < niter; i++) chess_loop_range(24, )
    chess_unroll_loop(8)
    chess_prepare_for_pipelining
    {
        vcharx vdata = vuchar_load_transp(srcAgen);
        vstore_bh(vdata, preAgen);
    }
}

void merge_channels_demote_u16_results_bl_exec(IMAGE_RESIZE_CFG &cfg)
{
    agen_B dstAgen  = init_agen_B_from_cfg(cfg._dst_agen_cfg);
    agen_C postAgen = init_agen_C_from_cfg(cfg._post_agen_cfg);

    int niter = max(cfg._niter_post, 24);
    niter     = (niter + 7) / 8 * 8;

    chess_separator_scheduler();

    for (int i = 0; i < niter; i++) chess_loop_range(24, )
    chess_unroll_loop(8)
    chess_prepare_for_pipelining
    {
        dvshortx dvdata = dvushort_load(postAgen);
        vstore_transp_hb(dvdata, dstAgen);
    }
}

void pre_knl_bl_exec(IMAGE_RESIZE_CFG &cfg)
{
    if (cfg._transpose_flag)
    {
        split_channels_promote_u8_table_bl_exec(cfg);
    }
    else
    {
        promote_u8_table_bl_exec(cfg);
    }
}

void post_knl_bl_exec(IMAGE_RESIZE_CFG &cfg)
{
    if (cfg._transpose_flag)
    {
        merge_channels_demote_u16_results_bl_exec(cfg);
    }
    else
    {
        demote_u16_results_bl_exec(cfg);
    }
}