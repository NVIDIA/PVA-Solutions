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

#include "split_rgba8_to_u8_knl.h"

void split_rgba8_to_u8_init(AgenCFG *cfgs, uint32_t srcTileLP, uint32_t dstTileLP)
{
    int vecW          = pva_elementsof(vcharx);
    uint32_t srcTileW = TILE_W * 4;
    uint32_t srcTileH = TILE_H;
    uint32_t dstTileW = TILE_W;
    uint32_t dstTileH = TILE_H;

    // for src
    AgenWrapper srcWrap;
    srcWrap.size = sizeof(uint8_t);
    srcWrap.n1   = srcTileW;
    srcWrap.s1   = 1;
    srcWrap.n2   = srcTileH / vecW;
    srcWrap.s2   = srcTileLP * vecW;

    // for dst
    AgenWrapper dstWrap;
    dstWrap.size = sizeof(uint8_t);
    dstWrap.n1   = dstTileW;
    dstWrap.s1   = 1;
    dstWrap.n2   = dstTileH / vecW;
    dstWrap.s2   = dstTileLP * vecW;

    agen srcAgen  = init(0);
    agen dst0Agen = init(0);
    agen dst1Agen = init(0);
    agen dst2Agen = init(0);
    agen dst3Agen = init(0);

    INIT_AGEN2(srcAgen, srcWrap);
    INIT_AGEN2(dst0Agen, dstWrap);
    INIT_AGEN2(dst1Agen, dstWrap);
    INIT_AGEN2(dst2Agen, dstWrap);
    INIT_AGEN2(dst3Agen, dstWrap);

    // for char type, transpose linepitch = (64 * lane_ofst + 2)
    srcAgen.lane_ofst  = (srcTileLP - 2) / 64;
    dst0Agen.lane_ofst = (dstTileLP - 2) / 64;
    dst1Agen.lane_ofst = (dstTileLP - 2) / 64;
    dst2Agen.lane_ofst = (dstTileLP - 2) / 64;
    dst3Agen.lane_ofst = (dstTileLP - 2) / 64;

    cfgs[SRC]   = extract_agen_cfg(srcAgen);
    cfgs[DST_0] = extract_agen_cfg(dst0Agen);
    cfgs[DST_1] = extract_agen_cfg(dst1Agen);
    cfgs[DST_2] = extract_agen_cfg(dst2Agen);
    cfgs[DST_3] = extract_agen_cfg(dst3Agen);
}

void split_rgba8_to_u8_exec(AgenCFG *cfgs)
{
    int niter       = TILE_H * TILE_W / pva_elementsof(vcharx);
    agen_B srcAgen  = init_agen_B_from_cfg(cfgs[SRC]);
    agen_C dst0Agen = init_agen_C_from_cfg(cfgs[DST_0]);
    agen_C dst1Agen = init_agen_C_from_cfg(cfgs[DST_1]);
    agen_C dst2Agen = init_agen_C_from_cfg(cfgs[DST_2]);
    agen_C dst3Agen = init_agen_C_from_cfg(cfgs[DST_3]);

    // 4 load, 4 store -> 4 VLIW
    for (int i = 0; i < niter; ++i) chess_loop_range(4, )
    chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        vcharx vdata0 = vuchar_load_transp(srcAgen);
        vcharx vdata1 = vuchar_load_transp(srcAgen);
        vcharx vdata2 = vuchar_load_transp(srcAgen);
        vcharx vdata3 = vuchar_load_transp(srcAgen);
        vstore_transp(vdata0, dst0Agen);
        vstore_transp(vdata1, dst1Agen);
        vstore_transp(vdata2, dst2Agen);
        vstore_transp(vdata3, dst3Agen);
    }
}
