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

#include "image_resize_opt_nn.h"

void init_nn_resize_cfg(IMAGE_RESIZE_CFG &cfg, void *srcBuf, void *dstBuf, void *preBuf, void *postBuf,
                        uint32_t *idxBuf, uint32_t *laneIds, int32_t *transformMat, uint16_t srcTileWidth,
                        uint16_t srcTileHeight, uint16_t srcTileLp, uint16_t dstTileWidth, uint16_t dstTileHeight,
                        uint16_t dstTileLp, uint8_t channelNum, bool u8Flag)
{
    cfg._transpose_flag  = channelNum > 1;
    cfg._preprocess_flag = cfg._transpose_flag;
    cfg._u8_flag         = u8Flag;
    cfg._bpp             = u8Flag ? 1 : 2;

    cfg._preBuf   = preBuf;
    cfg._postBuf  = postBuf;
    cfg._idxBuf   = idxBuf;
    cfg._lane_ids = laneIds;

    cfg._inputW  = cfg._transpose_flag ? srcTileHeight : srcTileWidth;
    cfg._inputH  = cfg._transpose_flag ? srcTileWidth : srcTileHeight;
    cfg._outputW = cfg._transpose_flag ? dstTileHeight : dstTileWidth;
    cfg._outputH = cfg._transpose_flag ? dstTileWidth : dstTileHeight;

    cfg._tableLp = cfg._inputW;

    if (!cfg._preprocess_flag)
    {
        return;
    }

    // init pre/post processing agens for multi-channel inputs
    int vecw = pva_elementsof(vcharx);
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
    w0.s1          = vecw;
    w0.s2          = cfg._tableLp;
    agen srcCxAgen = init((vuchar *)preBuf);
    INIT_AGEN2(srcCxAgen, w0);

    // postBuf: transposed single channel output, normal load
    AgenWrapper w1;
    w1.size = sizeof(uint8_t);
    w1.n1   = (dstTileHeight + vecw - 1) / vecw;
    w1.s1   = vecw;
    w1.n2   = dstTileWidth;
    w1.s2   = dstTileHeight;

    agen dstCxAgen = init((vuchar *)postBuf);
    INIT_AGEN2(dstCxAgen, w1);

    // dst: normal HWC output, transpose store
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

void init_nn_sampler_cfg(IMAGE_RESIZE_CFG &cfg, CupvaSampler &sampler, uint32_t stepCol, uint32_t stepRow)
{
    // generate indices for one tile with offsets 0
    generate_nn_sampler_tile_indices(cfg, stepCol, stepRow, 0, 0);

    CupvaSamplerInputType inputType = cfg._u8_flag ? SAMPLER_INPUT_TYPE_U8 : SAMPLER_INPUT_TYPE_U16;

    // init CupvaSampler
    CupvaSamplerInput2D input = {
        .data           = NULL,
        .type           = inputType,
        .width          = (uint32_t)cfg._inputW - 1,
        .height         = (uint32_t)cfg._inputH - 1,
        .linePitch      = (uint32_t)cfg._tableLp,
        .outOfRangeMode = SAMPLER_OUT_OF_RANGE_PREDICATE_OFF,
    };
    CupvaSamplerOutput output = {
        .data      = NULL,
        .transMode = TRANS_MODE_NONE,
    };
    CupvaSamplerIndices2D indices = {
        .data               = cfg._idxBuf,
        .type               = SAMPLER_INDEX_TYPE_U32,
        .width              = (uint16_t)(cfg._outputW * cfg._outputH),
        .height             = 1u,
        .fractionalBits     = (uint8_t)QBIT,
        .fractionalHandling = SAMPLER_FRAC_HANDLING_ROUND,
        .interleaving       = SAMPLER_INTERLEAVING_ELEMENTS,
    };
    cupvaSamplerSetup(&sampler, &input, &indices, &output);

    if (cfg._preprocess_flag)
    {
        update_nn_sampler_addrs(cfg, sampler, cfg._preBuf, cfg._postBuf);
    }
}

void generate_nn_sampler_tile_indices(IMAGE_RESIZE_CFG &cfg, uint32_t stepX, uint32_t stepY, uint32_t offsetX,
                                      uint32_t offsetY)
{
    // init dvintx lane ids
    for (int i = 0; i < 16; i++)
    {
        cfg._lane_ids[i] = i;
    }

    // init index buffer agen
    AgenWrapper w0;
    w0.size     = sizeof(int32_t);
    int vecw    = pva_elementsof(dvintx);
    int niter_w = (cfg._outputW + vecw - 1) / vecw;
    int niter_h = cfg._outputH;
    w0.n1       = 2 * niter_w;
    w0.s1       = vecw;
    w0.n2       = niter_h;
    w0.s2       = cfg._outputW * 2;

    agen a_idx = init((dvuint *)cfg._idxBuf);
    INIT_AGEN2(a_idx, w0);

    dvintx dv_stepX;
    dv_stepX.lo               = replicatew(stepX);
    dv_stepX.hi               = replicatew(stepX);
    dvintx dv_idx_x_row_start = dvmulw(zero_extend(*(dvuint *)cfg._lane_ids), dv_stepX, 0) + offsetX;
    dvintx dv_idx_x           = dv_idx_x_row_start;
    int idx_y                 = offsetY;
    int col_cnt               = 0;
    dvintx dv_zero            = chess_dont_care(dvintx) & 0;
    int vec_step_x            = stepX * vecw;
    int step_y                = stepY;

    int niter = niter_w * niter_h;
    niter     = max(niter, 24);
    niter     = (niter + 7) / 8 * 8;

    chess_separator_scheduler();

    for (int i = 0; i < niter; i++) chess_loop_range(24, )
    chess_unroll_loop(8)
    {
        // write XY-interleaved indices out
        dvintx dv_idx_xy;
        dv_idx_xy.lo = dv_idx_x.lo;
        dv_idx_xy.hi = replicatew(idx_y);
        vstore_i(dv_idx_xy, a_idx);

        dv_idx_xy.lo = dv_idx_x.hi;
        vstore_i(dv_idx_xy, a_idx);

        // update X and Y indices
        // col_cnt == 0 means a new row
        col_cnt = mod_inc(col_cnt, niter_w - 1);
        idx_y   = mux(col_cnt, idx_y, idx_y + step_y);
        // dv_new_row_flag == 0 means a new row
        dvintx dv_new_row_flag = dv_zero + col_cnt;
        dv_idx_x               = dvmux(dv_new_row_flag, dv_idx_x + vec_step_x, dv_idx_x_row_start);
    }
}

void update_nn_sampler_addrs(IMAGE_RESIZE_CFG &cfg, CupvaSampler &sampler, void *src, void *dst)
{
    cupvaSamplerUpdateAddrs(&sampler, src, cfg._idxBuf, dst);

#if CUPVA_BUILD_MODE == CUPVA_NATIVE
    // manually fix address range check in native mode
    sampler.inputAddrStart  = src;
    sampler.inputAddrEnd    = (void *)((uintptr_t)src + cfg._tableLp * cfg._inputH * cfg._bpp);
    sampler.outputAddrStart = dst;
    sampler.outputAddrEnd   = (void *)((uintptr_t)dst + cfg._outputW * cfg._outputH * cfg._bpp);
#endif
}

void update_nn_sampler_offsets(IMAGE_RESIZE_CFG &cfg, CupvaSampler &sampler, int32_t ofstCol, int32_t ofstRow)
{
    sampler.x_offset = ofstCol;
    sampler.y_offset = ofstRow;
}

void split_channels_u8_nn_exec(IMAGE_RESIZE_CFG &cfg)
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
        vstore(vdata, preAgen);
    }
}

void merge_channels_u8_nn_exec(IMAGE_RESIZE_CFG &cfg)
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
        vcharx vdata = vuchar_load(postAgen);
        vstore_transp(vdata, dstAgen);
    }
}