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

#include "radar_transpose_knl.h"

#include "radar_transpose_param.h"

/*
 * Set up agen configs for the vectorized and scalar transpose_tile_rdx paths.
 */
void transpose_tile_rdx_init(int32_t *src, int32_t *dst, param_transpose_tile_rdx_t *param, int32_t nRx,
                             int32_t nDoppler, int32_t useVectorizedPath)
{
    AgenWrapper wrapper;

    agen a_in_s  = init(src);
    wrapper.size = sizeof(int32_t);
    wrapper.n1   = nDoppler;
    wrapper.s1   = 2; /* 1 complex = 2 int32 per step              */
    wrapper.n2   = nRx;
    wrapper.s2   = nDoppler * 2; /* int32 offset to start of next rx row      */
    INIT_AGEN2(a_in_s, wrapper);
    param->cfg_in_scalar = extract_agen_cfg(a_in_s);

    agen a_out_s = init(dst);
    wrapper.size = sizeof(int32_t);
    wrapper.n1   = nDoppler;
    wrapper.s1   = nRx * 2; /* stride along doppler in output            */
    wrapper.n2   = nRx;
    wrapper.s2   = 2; /* 1 complex step (2 int32) to next rx col   */
    INIT_AGEN2(a_out_s, wrapper);
    param->cfg_out_scalar = extract_agen_cfg(a_out_s);

    param->niter_scalar = nDoppler * nRx;
    param->niter_min    = 0;

    if (useVectorizedPath)
    {
        const int32_t vecw           = RADAR_TRANSPOSE_VECW_INT32;
        const int32_t nDopplerGroups = nDoppler / 8;

        const int32_t outLinePitch = ((nRx * 2 + vecw - 1) & ~(vecw - 1)) + 2;
        /* Input: [nRx rows] x [nDopplerGroups groups of 8 complex] */
        agen a_in_cfg = init(src);
        wrapper.size  = sizeof(int32_t);
        wrapper.n1    = nDopplerGroups;
        wrapper.s1    = vecw; /* 16 int32 = 8 complex per step */
        wrapper.n2    = nRx;
        wrapper.s2    = nDoppler * 2; /* int32 per row */
        INIT_AGEN2(a_in_cfg, wrapper);
        param->cfg_in = extract_agen_cfg(a_in_cfg);

        /* Output: scatter 8 complex pairs to 8 doppler rows via lane_ofst. */
        agen a_out_cfg = init(dst);
        wrapper.size   = sizeof(int32_t);
        wrapper.n1     = nDopplerGroups;
        wrapper.s1     = outLinePitch * 8;
        wrapper.n2     = nRx;
        wrapper.s2     = 2; /* one complex step (2 int32) to next rx */
        INIT_AGEN2(a_out_cfg, wrapper);
        a_out_cfg.lane_ofst = outLinePitch / vecw;
        param->cfg_out      = extract_agen_cfg(a_out_cfg);

        const int32_t niter     = nDopplerGroups * nRx;
        const int32_t niter_adj = (niter + 7) / 8 * 8;
        param->niter_min        = niter_adj > RADAR_TRANSPOSE_VECW_INT32 ? niter_adj : RADAR_TRANSPOSE_VECW_INT32;
    }
}

/*
 * Transpose [nRx][nDoppler] complex -> [nDoppler][nRx] complex.
 *
 * src layout: src[(rx * nDoppler + d) * 2 + c]  — sequential along d
 * dst layout: dst[(d  * nRx     + rx) * 2 + c]  — stride nRx along d
 *
 * The host selects the vectorized or scalar path together with the destination DMA shape.
 * Both paths restore agen state from param: the vectorized path runs dvint_load_di +
 * vstore_transp2_i, while the scalar path runs int_load + int_store.
 */
void transpose_tile_rdx(int32_t *src, int32_t *restrict dst, int32_t nRx, int32_t nDoppler,
                        param_transpose_tile_rdx_t *param)
{
    if (param->niter_min > 0)
    {
        agen_A a_in  = init_agen_A_from_cfg(param->cfg_in);
        agen_C a_out = init_agen_C_from_cfg(param->cfg_out);

        chess_separator_scheduler();
        for (int32_t i = 0; i < param->niter_min; i++) chess_unroll_loop(16)
        chess_loop_range(16, )
        chess_prepare_for_pipelining
        {
            dvintx vec = dvint_load_di(a_in);
            vstore_transp2_i(vec, a_out);
        }
    }
    else
    {
        /* Scalar path: agen-based 2D transpose.
         * agen_A streams src sequentially (s1=2 per complex, wraps at s2=nDoppler*2).
         * agen_B writes dst with doppler stride (s1=nRx*2) and rx-col wrap (s2=2).
         * Dual load/store handles both re and im in one instruction each. */
        agen_A a_in  = init_agen_A_from_cfg(param->cfg_in_scalar);
        agen_B a_out = init_agen_B_from_cfg(param->cfg_out_scalar);

        chess_separator_scheduler();
        for (int32_t i = 0; i < param->niter_scalar; i++) chess_unroll_loop(4)
        chess_prepare_for_pipelining
        chess_loop_range(2, )
        {
            int32_t re, im;
            int_load(a_in, re, im);
            int_store(re, im, a_out);
        }
    }
}

/*
 * Set up agen configs for transpose_chunk_drx.
 *
 */
void transpose_chunk_drx_init(int32_t *src, int32_t *scatter, int32_t *dst, param_transpose_chunk_drx_t *param,
                              int32_t chunkRange, int32_t nRx, int32_t nDoppler)
{
    AgenWrapper wrapper;
    int32_t rb;
    wrapper.size        = sizeof(int32_t);
    param->niter_scalar = nRx * nDoppler;
    param->niter_min    = 0; /* default: scalar path */

    for (rb = 0; rb < chunkRange; rb++)
    {
        /* agen_A: src[rb][rx][d] — d fast (s1=2), rx slow (s2=nDoppler*2) */
        agen a_src = init(src + rb * nRx * nDoppler * 2);
        wrapper.n1 = nDoppler;
        wrapper.s1 = 2;
        wrapper.n2 = nRx;
        wrapper.s2 = nDoppler * 2;
        INIT_AGEN2(a_src, wrapper);
        param->cfg_in_scl[rb] = extract_agen_cfg(a_src);

        /* agen_B: dst[d][rx][rb] tight — d stride (s1=nRx*chunkRange*2), rx step (s2=chunkRange*2) */
        agen a_dst = init(dst + rb * 2);
        wrapper.n1 = nDoppler;
        wrapper.s1 = nRx * chunkRange * 2;
        wrapper.n2 = nRx;
        wrapper.s2 = chunkRange * 2;
        INIT_AGEN2(a_dst, wrapper);
        param->cfg_out_scl[rb] = extract_agen_cfg(a_dst);
    }

    if ((nDoppler & 7) == 0)
    {
        const int32_t vecw           = RADAR_TRANSPOSE_VECW_INT32;
        const int32_t nDopplerGroups = nDoppler / 8;
        const int32_t rowWidth       = nRx * chunkRange * 2;
        const int32_t outLinePitch   = ((rowWidth + vecw - 1) & ~(vecw - 1)) + 2;
        int32_t niter, niter_adj;
        agen a_compact;

        if (((rowWidth % vecw) == 0) && ((nDoppler * outLinePitch) <= MAX_TILE_SIZE))
        {
            for (rb = 0; rb < chunkRange; rb++)
            {
                agen a_in_vec, a_out_vec;

                /* agen_A: src[rb][rx][d] — groups of 8 complex (vecw int32) along d */
                a_in_vec   = init(src + rb * nRx * nDoppler * 2);
                wrapper.n1 = nDopplerGroups;
                wrapper.s1 = vecw; /* 16 int32 = 8 complex per step */
                wrapper.n2 = nRx;
                wrapper.s2 = nDoppler * 2;
                INIT_AGEN2(a_in_vec, wrapper);
                param->cfg_in_vec[rb] = extract_agen_cfg(a_in_vec);

                /* agen_C: scatter to padded scatterBuf[d][rx][rb].
                 * Base: scatter+rb*2; s1=outLinePitch*8 (stride 8 doppler rows); s2=chunkRange*2 (next rx). */
                a_out_vec  = init(scatter + rb * 2);
                wrapper.n1 = nDopplerGroups;
                wrapper.s1 = outLinePitch * 8; /* stride across 8 doppler rows */
                wrapper.n2 = nRx;
                wrapper.s2 = chunkRange * 2; /* one complex step to next rx */
                INIT_AGEN2(a_out_vec, wrapper);
                a_out_vec.lane_ofst    = outLinePitch / vecw;
                param->cfg_out_vec[rb] = extract_agen_cfg(a_out_vec);
            }

            niter            = nDopplerGroups * nRx;
            niter_adj        = (niter + 7) / 8 * 8;
            param->niter_min = niter_adj > 16 ? niter_adj : 16;

            a_compact  = init(scatter);
            wrapper.n1 = rowWidth / vecw;
            wrapper.s1 = vecw;
            wrapper.n2 = nDoppler;
            wrapper.s2 = outLinePitch;
            INIT_AGEN2(a_compact, wrapper);
            param->cfg_cmp_rd = extract_agen_cfg(a_compact);

            a_compact  = init(dst);
            wrapper.n1 = rowWidth / vecw;
            wrapper.s1 = vecw;
            wrapper.n2 = nDoppler;
            wrapper.s2 = rowWidth;
            INIT_AGEN2(a_compact, wrapper);
            param->cfg_cmp_wr = extract_agen_cfg(a_compact);

            param->niter_compact = nDoppler * (rowWidth / vecw);
        }
    }
}

/*
 * Rearrange [chunkRange][nRx][nDoppler] -> [nDoppler][nRx][chunkRange] complex.
 */
void transpose_chunk_drx(int32_t *src, int32_t *restrict scatter, int32_t *restrict dst, int32_t chunkRange,
                         int32_t nRx, int32_t nDoppler, param_transpose_chunk_drx_t *param)
{
    int32_t rb;
    if ((nDoppler & 7) == 0 && param->niter_min > 0)
    {
        for (rb = 0; rb < chunkRange; rb++)
        {
            agen_A a_in  = init_agen_A_from_cfg(param->cfg_in_vec[rb]);
            agen_C a_out = init_agen_C_from_cfg(param->cfg_out_vec[rb]);

            chess_separator_scheduler();
            for (int32_t i = 0; i < param->niter_min; i++) chess_unroll_loop(8)
            chess_loop_range(16, )
            chess_prepare_for_pipelining
            {
                dvintx vec = dvint_load_di(a_in);
                vstore_transp2_i(vec, a_out);
            }
        }

        /* Cross-superbank compaction: padded scatterBuf (bank A) → tight dstBuf (bank C).
         * dvint_load reads rowWidth/vecw chunks of 16 int32 per doppler row;
         * dvint_store writes them sequentially to dstBuf with tight row pitch. */
        {
            agen_A a_crd = init_agen_A_from_cfg(param->cfg_cmp_rd);
            agen_B a_cwr = init_agen_B_from_cfg(param->cfg_cmp_wr);
            chess_separator_scheduler();
            for (int32_t i = 0; i < param->niter_compact; i++) chess_prepare_for_pipelining
            chess_loop_range(2, )
            {
                dvintx val = dvint_load(a_crd);
                vstore(val, a_cwr);
            }
        }
    }
    else
    {
        for (rb = 0; rb < chunkRange; rb++)
        {
            agen_A a_in  = init_agen_A_from_cfg(param->cfg_in_scl[rb]);
            agen_B a_out = init_agen_B_from_cfg(param->cfg_out_scl[rb]);

            chess_separator_scheduler();
            for (int32_t i = 0; i < param->niter_scalar; i++) chess_unroll_loop(4)
            chess_prepare_for_pipelining
            chess_loop_range(2, )
            {
                int32_t re, im;
                int_load(a_in, re, im);
                int_store(re, im, a_out);
            }
        }
    }
}
