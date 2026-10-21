/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gaussian_pyramid_knl.h"

#include "gaussian_pyramid_param.h"

#if defined(DATA_TYPE_U8) || defined(DATA_TYPE_S8)
void nonSep6x6FIR_downsamp2x2_weight_init(const uint16_t *filter, uint16_t *weightPtr)
{
    dvchar *weight = (dvchar *)weightPtr;

    // filter 6x6 indices
    //  0   1   2   3   4   5
    //  6   7   8   9  10  11
    // 12  13  14  15  16  17
    // 18  19  20  21  22  23
    // 24  25  26  27  28  29
    // 30  31  32  33  34  35

    vshortx t = chess_dont_care(vshortx);
    t         = vcollate_idx(replicateh(0)); // 0, 1, 2, 3, ...
    t         = (t & 1) * 6;                 // 0, 6, 0, 6, 0, 6, 0, 6, ...

    dvcharx p = chess_dont_care(dvcharx);
    vdemote_i(t, t + 1, p.lo); // 0, 1, 6, 7, 0, 1, 6, 7, ...
    p.hi = p.lo - 2 * 6;       // last two rows: -12, -11, -6, -5, -12, -11, -6, -5, ...

    agen s1 = init((int16_t *)filter);

    s1.n1   = 3;
    s1.mod1 = 2 * sizeof(int16_t);
    s1.n2   = 4;
    s1.mod2 = -4 * (int)sizeof(int16_t);
    agen s2 = s1;

    for (int32_t y = 0; y < 4; ++y, p += 2 * 6 /*advance 2 rows*/) chess_unroll_loop(*)
    {
        for (int32_t x = 0; x < 3; ++x) chess_unroll_loop(*)
        {
            dvcharx r;
            r.lo              = vdemote(dvshort_load_perm(s1, p.lo));
            weight[x * 4 + y] = extract(update_hi(r, vdemote(dvshort_load_perm(s2, p.hi))));
        }
    }
}
#elif defined(DATA_TYPE_U16) || defined(DATA_TYPE_S16)
void nonSep6x6FIR_downsamp2x2_weight_init(const uint16_t *filter, uint16_t *weight)
{
    dvshort *weightPtr = (dvshort *)weight;

    dvshortx t = chess_dont_care(dvshortx);
    t.lo       = vcollate_idx(replicateh(0)); // 0, 1, 2, 3, ...
    t.lo       = (t.lo & 1) * 6 +
           vmux(t.lo >= 8, -2 * 6, replicateh(0)); // 0, 6, 0, 6, 0, 6, 0, 6, -12, -6, -12, -6, -12, -6, -12, -6,
    t.hi     = t.lo + 1;
    vcharx p = chess_dont_care(vcharx);
    vdemote_i(t.lo, t.hi, p);

    agen s = init((uint16_t *)filter);
    s.n1   = 3;
    s.mod1 = 2 * sizeof(uint16_t);
    s.n2   = 4;
    s.mod2 = -4 * (int32_t)sizeof(uint16_t);

    for (int32_t y = 0; y < 4; ++y, p += 12) chess_unroll_loop(*)
    for (int32_t x = 0; x < 3; ++x) chess_unroll_loop(*)
    weightPtr[x * 4 + y] = extract(dvshort_load_perm(s, p));
}
#endif

static inline void rb2_body(const DIN_VEC_TYPE v0, const DIN_VEC_TYPE v1, const DIN_VEC_TYPE v2, const DIN_VEC_TYPE v3,
                            const DIN_VEC_TYPE v4, const DIN_VEC_TYPE v5, const DIN_VEC_TYPE v6, const DIN_VEC_TYPE v7,
                            DACC_VEC_TYPE &a1, DACC_VEC_TYPE &a2, const DKNL_VEC_TYPE w0, const DKNL_VEC_TYPE w1,
                            const DKNL_VEC_TYPE w2, const DKNL_VEC_TYPE w3, const int32_t p, const int32_t pT)
{
    VDOTP4x2(v0, v1, w0, a1, a2, p, a1, a2);
    VDOTP4x2(v2, v3, w1, a1, a2, pT, a1, a2);
    VDOTP4x2(v4, v5, w2, a1, a2, pT, a1, a2);
    VDOTP4x2(v6, v7, w3, a1, a2, pT, a1, a2);
}

void nonSep6x6FIR_downsamp2x2_regblk4(AgenCFG *cfg_in_even, AgenCFG *cfg_in_odd, AgenCFG *cfg_out_even,
                                      AgenCFG *cfg_out_odd, const size_t rb_cnt, const uint16_t *weightPtr)
{
    agen_A in_even  = init_agen_A_from_cfg(*cfg_in_even);
    agen_B in_odd   = init_agen_B_from_cfg(*cfg_in_odd);
    agen_A out_even = init_agen_A_from_cfg(*cfg_out_even);
    agen_C out_odd  = init_agen_C_from_cfg(*cfg_out_odd);

    // use agen to allow DWdst
    agen t = init((KNL_MEM_TYPE *)weightPtr);
    t.n1   = 3 * 4;
    t.mod1 = sizeof(KNL_MEM_TYPE);

    const DKNL_VEC_TYPE chess_storage(DW) w00 = DKNL_LOAD_FUNC(t);
    const DKNL_VEC_TYPE chess_storage(DW) w10 = DKNL_LOAD_FUNC(t);
    const DKNL_VEC_TYPE chess_storage(DW) w20 = DKNL_LOAD_FUNC(t);
    const DKNL_VEC_TYPE chess_storage(DW) w30 = DKNL_LOAD_FUNC(t);

    const DKNL_VEC_TYPE chess_storage(DW) w01 = DKNL_LOAD_FUNC(t);
    const DKNL_VEC_TYPE chess_storage(DW) w11 = DKNL_LOAD_FUNC(t);
    const DKNL_VEC_TYPE chess_storage(DW) w21 = DKNL_LOAD_FUNC(t);
    const DKNL_VEC_TYPE chess_storage(DW) w31 = DKNL_LOAD_FUNC(t);

    const DKNL_VEC_TYPE chess_storage(DW) w02 = DKNL_LOAD_FUNC(t);
    const DKNL_VEC_TYPE chess_storage(DW) w12 = DKNL_LOAD_FUNC(t);
    const DKNL_VEC_TYPE chess_storage(DW) w22 = DKNL_LOAD_FUNC(t);
    const DKNL_VEC_TYPE chess_storage(DW) w32 = DKNL_LOAD_FUNC(t);
    chess_dont_warn_dead(t);

    DACC_VEC_TYPE a1 = chess_dont_care(DACC_VEC_TYPE);
    DACC_VEC_TYPE a2 = chess_dont_care(DACC_VEC_TYPE);
    DACC_VEC_TYPE a3 = chess_dont_care(DACC_VEC_TYPE);
    DACC_VEC_TYPE a4 = chess_dont_care(DACC_VEC_TYPE);

    const int32_t chess_storage(PA) pT = chess_copy(-1);
    const int32_t chess_storage(PB) pF = chess_copy(0);

    for (size_t it = 0; it < NITER_Y_TILE0; ++it) chess_prepare_for_pipelining
    {
        for (size_t n = 0; n < rb_cnt; ++n) chess_prepare_for_pipelining
        {
            for (size_t x = 0; x < 3; ++x) chess_unroll_loop(*)
            {
                const DKNL_VEC_TYPE w0 = (x == 0) ? w00 : (x == 1) ? w01 : w02;
                const DKNL_VEC_TYPE w1 = (x == 0) ? w10 : (x == 1) ? w11 : w12;
                const DKNL_VEC_TYPE w2 = (x == 0) ? w20 : (x == 1) ? w21 : w22;
                const DKNL_VEC_TYPE w3 = (x == 0) ? w30 : (x == 1) ? w31 : w32;

                const DIN_VEC_TYPE v0 = IN_LOAD_FUNC(in_even);
                const DIN_VEC_TYPE v1 = IN_LOAD_FUNC(in_odd);
                const DIN_VEC_TYPE v2 = IN_LOAD_FUNC(in_even);
                const DIN_VEC_TYPE v3 = IN_LOAD_FUNC(in_odd);
                const DIN_VEC_TYPE v4 = IN_LOAD_FUNC(in_even);
                const DIN_VEC_TYPE v5 = IN_LOAD_FUNC(in_odd);
                const DIN_VEC_TYPE v6 = IN_LOAD_FUNC(in_even);
                const DIN_VEC_TYPE v7 = IN_LOAD_FUNC(in_odd);
                const DIN_VEC_TYPE v8 = IN_LOAD_FUNC(in_even);
                const DIN_VEC_TYPE v9 = IN_LOAD_FUNC(in_odd);
                const DIN_VEC_TYPE vA = IN_LOAD_FUNC(in_even);
                const DIN_VEC_TYPE vB = IN_LOAD_FUNC(in_odd);

                rb2_body(v0, v1, v2, v3, v4, v5, v6, v7, a1, a2, w0, w1, w2, w3, x ? pT : pF, pT);
                rb2_body(v4, v5, v6, v7, v8, v9, vA, vB, a3, a4, w0, w1, w2, w3, x ? pT : pF, pT);

                if (x == 2)
                {
                    VSTORE_I(a1, out_even);
                    VSTORE_I(a2, out_odd);
                    VSTORE_I(a3, out_even);
                    VSTORE_I(a4, out_odd);
                }
                else
                {
                    vstore(a1, out_even, pF);
                    vstore(a2, out_odd, pF);
                    vstore(a3, out_even, pF);
                    vstore(a4, out_odd, pF);
                }
            }
        }
    }
}

void nonSep6x6FIR_downsamp2x2_regblk2(AgenCFG *cfg_in_even, AgenCFG *cfg_in_odd, AgenCFG *cfg_out_even,
                                      AgenCFG *cfg_out_odd, const size_t rb_cnt, const uint16_t *weightPtr)
{
    agen_A in_even  = init_agen_A_from_cfg(*cfg_in_even);
    agen_C in_odd   = init_agen_C_from_cfg(*cfg_in_odd);
    agen_B out_even = init_agen_B_from_cfg(*cfg_out_even);
    agen_B out_odd  = init_agen_B_from_cfg(*cfg_out_odd);

    // use agen to allow DWdst
    agen t = init((KNL_MEM_TYPE *)weightPtr);
    t.n1   = 3 * 4;
    t.mod1 = sizeof(KNL_MEM_TYPE);

    const DKNL_VEC_TYPE chess_storage(DW) w00 = DKNL_LOAD_FUNC(t);
    const DKNL_VEC_TYPE chess_storage(DW) w10 = DKNL_LOAD_FUNC(t);
    const DKNL_VEC_TYPE chess_storage(DW) w20 = DKNL_LOAD_FUNC(t);
    const DKNL_VEC_TYPE chess_storage(DW) w30 = DKNL_LOAD_FUNC(t);

    const DKNL_VEC_TYPE chess_storage(DW) w01 = DKNL_LOAD_FUNC(t);
    const DKNL_VEC_TYPE chess_storage(DW) w11 = DKNL_LOAD_FUNC(t);
    const DKNL_VEC_TYPE chess_storage(DW) w21 = DKNL_LOAD_FUNC(t);
    const DKNL_VEC_TYPE chess_storage(DW) w31 = DKNL_LOAD_FUNC(t);

    const DKNL_VEC_TYPE chess_storage(DW) w02 = DKNL_LOAD_FUNC(t);
    const DKNL_VEC_TYPE chess_storage(DW) w12 = DKNL_LOAD_FUNC(t);
    const DKNL_VEC_TYPE chess_storage(DW) w22 = DKNL_LOAD_FUNC(t);
    const DKNL_VEC_TYPE chess_storage(DW) w32 = DKNL_LOAD_FUNC(t);
    chess_dont_warn_dead(t);

    DACC_VEC_TYPE a1 = chess_dont_care(DACC_VEC_TYPE);
    DACC_VEC_TYPE a2 = chess_dont_care(DACC_VEC_TYPE);

    const int32_t chess_storage(PA) pT = chess_copy(-1);
    const int32_t chess_storage(PB) pF = chess_copy(0);

    for (size_t it = 0; it < NITER_Y_TILE1; ++it) chess_prepare_for_pipelining
    {
        for (size_t n = 0; n < rb_cnt; ++n) chess_prepare_for_pipelining
        {
            for (size_t x = 0; x < 3; ++x) chess_unroll_loop(*)
            {
                const DKNL_VEC_TYPE w0 = (x == 0) ? w00 : (x == 1) ? w01 : w02;
                const DKNL_VEC_TYPE w1 = (x == 0) ? w10 : (x == 1) ? w11 : w12;
                const DKNL_VEC_TYPE w2 = (x == 0) ? w20 : (x == 1) ? w21 : w22;
                const DKNL_VEC_TYPE w3 = (x == 0) ? w30 : (x == 1) ? w31 : w32;

                const DIN_VEC_TYPE v0 = IN_LOAD_FUNC(in_even);
                const DIN_VEC_TYPE v1 = IN_LOAD_FUNC(in_odd);
                const DIN_VEC_TYPE v2 = IN_LOAD_FUNC(in_even);
                const DIN_VEC_TYPE v3 = IN_LOAD_FUNC(in_odd);
                const DIN_VEC_TYPE v4 = IN_LOAD_FUNC(in_even);
                const DIN_VEC_TYPE v5 = IN_LOAD_FUNC(in_odd);
                const DIN_VEC_TYPE v6 = IN_LOAD_FUNC(in_even);
                const DIN_VEC_TYPE v7 = IN_LOAD_FUNC(in_odd);

                rb2_body(v0, v1, v2, v3, v4, v5, v6, v7, a1, a2, w0, w1, w2, w3, x ? pT : pF, pT);
            }
            VSTORE_I(a1, out_even);
            VSTORE_I(dvmov(a2), out_odd);
        }
    }
}