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

#include "merge_u8_to_rgba8_knl.h"

void merge_u8_to_rgba8_init(AgenCFG *cfgs, uint32_t srcTileLP, uint32_t dstTileLP)
{
    int vecW     = pva_elementsof(vcharx);
    int srcTileW = TILE_W;
    int srcTileH = TILE_H;
    int dstTileW = TILE_W * 4;
    int dstTileH = TILE_H;

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

    agen src0Agen = init(0);
    agen src1Agen = init(0);
    agen src2Agen = init(0);
    agen src3Agen = init(0);
    agen dstAgen  = init(0);

    INIT_AGEN2(src0Agen, srcWrap);
    INIT_AGEN2(src1Agen, srcWrap);
    INIT_AGEN2(src2Agen, srcWrap);
    INIT_AGEN2(src3Agen, srcWrap);
    INIT_AGEN2(dstAgen, dstWrap);

    // for char type, transpose linepitch = (64 * lane_ofst + 2)
    src0Agen.lane_ofst = (srcTileLP - 2) / 64;
    src1Agen.lane_ofst = (srcTileLP - 2) / 64;
    src2Agen.lane_ofst = (srcTileLP - 2) / 64;
    src3Agen.lane_ofst = (srcTileLP - 2) / 64;
    dstAgen.lane_ofst  = (dstTileLP - 2) / 64;

    cfgs[SRC_0] = extract_agen_cfg(src0Agen);
    cfgs[SRC_1] = extract_agen_cfg(src1Agen);
    cfgs[SRC_2] = extract_agen_cfg(src2Agen);
    cfgs[SRC_3] = extract_agen_cfg(src3Agen);
    cfgs[DST]   = extract_agen_cfg(dstAgen);
}

void merge_u8_to_rgba8_exec(AgenCFG *cfgs)
{
    int niter       = TILE_H * TILE_W / pva_elementsof(vcharx);
    agen_B src0Agen = init_agen_B_from_cfg(cfgs[SRC_0]);
    agen_B src1Agen = init_agen_B_from_cfg(cfgs[SRC_1]);
    agen_B src2Agen = init_agen_B_from_cfg(cfgs[SRC_2]);
    agen_B src3Agen = init_agen_B_from_cfg(cfgs[SRC_3]);
    agen_C dstAgen  = init_agen_C_from_cfg(cfgs[DST]);

    // 4 load, 4 store -> 4 VLIW
    for (int i = 0; i < niter; ++i) chess_loop_range(4, )
    chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        vcharx vdata0 = vuchar_load_transp(src0Agen);
        vcharx vdata1 = vuchar_load_transp(src1Agen);
        vcharx vdata2 = vuchar_load_transp(src2Agen);
        vcharx vdata3 = vuchar_load_transp(src3Agen);
        vstore_transp(vdata0, dstAgen);
        vstore_transp(vdata1, dstAgen);
        vstore_transp(vdata2, dstAgen);
        vstore_transp(vdata3, dstAgen);
    }
}
