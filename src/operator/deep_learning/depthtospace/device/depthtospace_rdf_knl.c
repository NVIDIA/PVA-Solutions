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

/*
 * DepthToSpace inner kernel -- a BLK_SZ-way byte interleave with ZERO vector-ALU ops.
 *
 * No vector store writes a strided pattern and the ISA has no 3-way interleaving store, so
 * the interleave lives in the store agen's s1 = BLK_SZ instead -- the technique
 * convert_image_format uses to pack planar data into RGBA. Rounding and saturation stay OFF.
 */

#include "depthtospace_rdf_knl.h"

void depthtospace_agen_init(int32_t srcLinePitch, int32_t dstLinePitch, DepthToSpaceConfig *cfg)
{
    int32_t const vecRows = chess_elementsof(vchar); /* == TRANSP_ROWS */

    AgenWrapper wrapper;

    /* Source: walk the tile's columns (n1 = TILE_W, s1 = 1), stepping down TRANSP_ROWS
     * rows at a time. Each access is a transposed load: one column across 32 rows. */
    wrapper.size = sizeof(uint8_t);
    wrapper.n1   = TILE_W;
    wrapper.s1   = 1;
    wrapper.n2   = TILE_H / TRANSP_ROWS;
    wrapper.s2   = srcLinePitch * vecRows;

    /*
     * Byte-type transposed access requires a transpose line pitch of the form 64 * lane_ofst + 2,
     * which TRANS_MODE_1 guarantees. lane_ofst = 0 would read/write every OTHER byte instead of
     * consecutive rows, so it must be derived, never left at zero.
     */
    agen agenSrc0 = init(0);
    INIT_AGEN2(agenSrc0, wrapper);
    agenSrc0.lane_ofst = (srcLinePitch - 2) / 64;
    cfg->srcAgenCfg[0] = extract_agen_cfg(agenSrc0);

    agen agenSrc1 = init(0);
    INIT_AGEN2(agenSrc1, wrapper);
    agenSrc1.lane_ofst = (srcLinePitch - 2) / 64;
    cfg->srcAgenCfg[1] = extract_agen_cfg(agenSrc1);

#if BLK_SZ >= 3
    agen agenSrc2 = init(0);
    INIT_AGEN2(agenSrc2, wrapper);
    agenSrc2.lane_ofst = (srcLinePitch - 2) / 64;
    cfg->srcAgenCfg[2] = extract_agen_cfg(agenSrc2);
#endif
#if BLK_SZ >= 4
    agen agenSrc3 = init(0);
    INIT_AGEN2(agenSrc3, wrapper);
    agenSrc3.lane_ofst = (srcLinePitch - 2) / 64;
    cfg->srcAgenCfg[3] = extract_agen_cfg(agenSrc3);
#endif

    /* Destination: s1 = BLK_SZ IS the interleave. agen_update offsets the bases by bx
     * so stream bx lands on byte bx of every output column group. */
    wrapper.n1 = TILE_W;
    wrapper.s1 = BLK_SZ;
    wrapper.n2 = TILE_H / TRANSP_ROWS;
    wrapper.s2 = dstLinePitch * vecRows;

    agen agenDst0 = init(0);
    INIT_AGEN2(agenDst0, wrapper);
    agenDst0.lane_ofst = (dstLinePitch - 2) / 64;
    cfg->dstAgenCfg[0] = extract_agen_cfg(agenDst0);

    agen agenDst1 = init(0);
    INIT_AGEN2(agenDst1, wrapper);
    agenDst1.lane_ofst = (dstLinePitch - 2) / 64;
    cfg->dstAgenCfg[1] = extract_agen_cfg(agenDst1);

#if BLK_SZ >= 3
    agen agenDst2 = init(0);
    INIT_AGEN2(agenDst2, wrapper);
    agenDst2.lane_ofst = (dstLinePitch - 2) / 64;
    cfg->dstAgenCfg[2] = extract_agen_cfg(agenDst2);
#endif
#if BLK_SZ >= 4
    agen agenDst3 = init(0);
    INIT_AGEN2(agenDst3, wrapper);
    agenDst3.lane_ofst = (dstLinePitch - 2) / 64;
    cfg->dstAgenCfg[3] = extract_agen_cfg(agenDst3);
#endif
}

void depthtospace_agen_update(uint8_t *const *srcTiles, uint8_t *dstTile, DepthToSpaceConfig *cfg)
{
    /* BLK_SZ is a compile-time constant; enumerated so no scalar loop reaches the VPU. */
    cupvaModifyAgenCfgBase(&(cfg->srcAgenCfg[0]), srcTiles[0]);
    cupvaModifyAgenCfgBase(&(cfg->srcAgenCfg[1]), srcTiles[1]);
#if BLK_SZ >= 3
    cupvaModifyAgenCfgBase(&(cfg->srcAgenCfg[2]), srcTiles[2]);
#endif
#if BLK_SZ >= 4
    cupvaModifyAgenCfgBase(&(cfg->srcAgenCfg[3]), srcTiles[3]);
#endif

    /* Stream bx writes byte bx of each output column group. */
    cupvaModifyAgenCfgBase(&(cfg->dstAgenCfg[0]), dstTile + 0);
    cupvaModifyAgenCfgBase(&(cfg->dstAgenCfg[1]), dstTile + 1);
#if BLK_SZ >= 3
    cupvaModifyAgenCfgBase(&(cfg->dstAgenCfg[2]), dstTile + 2);
#endif
#if BLK_SZ >= 4
    cupvaModifyAgenCfgBase(&(cfg->dstAgenCfg[3]), dstTile + 3);
#endif
}

void depthtospace_exec(int32_t niter, DepthToSpaceConfig *cfg)
{
    /*
     * Inputs on A/B so the BLK_SZ loads issue in parallel; the destination is on C, which
     * therefore sees only writes.
     */
    agen_A agenSrc0 = init_agen_A_from_cfg(cfg->srcAgenCfg[0]);
    agen_B agenSrc1 = init_agen_B_from_cfg(cfg->srcAgenCfg[1]);
#if BLK_SZ >= 3
    agen_A agenSrc2 = init_agen_A_from_cfg(cfg->srcAgenCfg[2]);
#endif
#if BLK_SZ >= 4
    agen_B agenSrc3 = init_agen_B_from_cfg(cfg->srcAgenCfg[3]);
#endif

    agen_C agenDst0 = init_agen_C_from_cfg(cfg->dstAgenCfg[0]);
    agen_C agenDst1 = init_agen_C_from_cfg(cfg->dstAgenCfg[1]);
#if BLK_SZ >= 3
    agen_C agenDst2 = init_agen_C_from_cfg(cfg->dstAgenCfg[2]);
#endif
#if BLK_SZ >= 4
    agen_C agenDst3 = init_agen_C_from_cfg(cfg->dstAgenCfg[3]);
#endif

    /* BLK_SZ loads + BLK_SZ stores, no vector-ALU ops -> mem/3 is the slot floor. */
    for (int32_t i = 0; i < niter; i++) chess_loop_range(1, 8192)
    chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        vcharx s0 = vuchar_load_transp(agenSrc0);
        vcharx s1 = vuchar_load_transp(agenSrc1);
#if BLK_SZ >= 3
        vcharx s2 = vuchar_load_transp(agenSrc2);
#endif
#if BLK_SZ >= 4
        vcharx s3 = vuchar_load_transp(agenSrc3);
#endif

        vstore_transp(s0, agenDst0);
        vstore_transp(s1, agenDst1);
#if BLK_SZ >= 3
        vstore_transp(s2, agenDst2);
#endif
#if BLK_SZ >= 4
        vstore_transp(s3, agenDst3);
#endif
    }
}
