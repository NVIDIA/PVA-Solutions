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

#include "image_blend_knl_u8.h"

void image_blend_agen_init(int32_t src0LinePitch, int32_t src1LinePitch, int32_t dstLinePitch, ImageBlendConfig *cfg)
{
    AgenWrapper wrapper;

    // config for agen src0
    wrapper.size = sizeof(uint8_t);
    wrapper.n1   = TILE_WIDTH / chess_elementsof(dvchar);
    wrapper.s1   = chess_elementsof(dvchar);
    wrapper.n2   = TILE_HEIGHT;
    wrapper.s2   = src0LinePitch;

    agen agen_src0 = init(0);
    INIT_AGEN2(agen_src0, wrapper);
    cfg->src0AgenCfg = extract_agen_cfg(agen_src0);

    // config for agen src1
    wrapper.s2     = src1LinePitch;
    agen agen_src1 = init(0);
    INIT_AGEN2(agen_src1, wrapper);
    cfg->src1AgenCfg = extract_agen_cfg(agen_src1);

    // config for agen dst
    wrapper.s2    = dstLinePitch;
    agen agen_dst = init(0);
    INIT_AGEN2(agen_dst, wrapper);
    cfg->dstAgenCfg = extract_agen_cfg(agen_dst);
}

void image_blend_agen_update(uint8_t *src0Tile, uint8_t *src1Tile, uint8_t *dstTile, ImageBlendConfig *cfg)
{
    cupvaModifyAgenCfgBase(&(cfg->src0AgenCfg), src0Tile);
    cupvaModifyAgenCfgBase(&(cfg->src1AgenCfg), src1Tile);
    cupvaModifyAgenCfgBase(&(cfg->dstAgenCfg), dstTile);
}

void image_blend_exec(uint32_t alpha, int32_t niter, ImageBlendConfig *cfg)
{
    agen_A agen_load_src0 = init_agen_A_from_cfg(cfg->src0AgenCfg);
    agen_B agen_load_src1 = init_agen_B_from_cfg(cfg->src1AgenCfg);
    agen_C agen_store_dst = init_agen_C_from_cfg(cfg->dstAgenCfg);

    for (int i = 0; i < niter; i++) chess_loop_range(8, )
    chess_unroll_loop(8)
    chess_prepare_for_pipelining
    {
        dvcharx dv_src0 = dvuchar_load(agen_load_src0);
        dvcharx dv_src1 = dvuchar_load(agen_load_src1);

        dvcharx dv_dst = dvblend(dv_src0, dv_src1, alpha);

        vstore(dv_dst, agen_store_dst);
    }
}
