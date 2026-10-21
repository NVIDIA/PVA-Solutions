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

#ifndef RADAR_TRANSPOSE_KNL_H
#define RADAR_TRANSPOSE_KNL_H

#include "radar_transpose_param.h"

#include <cupva_device.h>
#include <stdint.h>

/*
 * Agen configs for the vectorized and scalar paths of transpose_tile_rdx.
 * Populated once by transpose_tile_rdx_init() before the tile loop.
 */
typedef struct
{
    AgenCFG cfg_in;         /* vec:    2D agen_A: [nRx][nDopplerGroups] sequential read      */
    AgenCFG cfg_out;        /* vec:    2D agen_C: scatter via lane_ofst to doppler rows       */
    int32_t niter_min;      /* vec:    loop trip count (rounded up for unroll/pipeline)       */
    AgenCFG cfg_in_scalar;  /* scalar: 2D agen_A: s1=2 per complex, s2=nDoppler*2 to next rx */
    AgenCFG cfg_out_scalar; /* scalar: 2D agen_B: s1=nRx*2 per doppler, s2=2 to next rx col  */
    int32_t niter_scalar;   /* scalar: = nRx * nDoppler                                      */
} param_transpose_tile_rdx_t;

/*
 * transpose_tile_rdx_init - Set up agen configs for both transpose paths.
 *
 * Call ONCE before the tile loop.  src and dst are the fixed VMEM buffer
 * addresses (srcBuf / dstBuf); nRx and nDoppler are the tile dimensions.
 *
 * Always initialises the scalar agen configs (cfg_in_scalar / cfg_out_scalar).
 * Also initialises the vectorized agen configs when useVectorizedPath is nonzero;
 * otherwise only the scalar path is armed.
 *
 * @param src      VMEM input buffer base (srcBuf)
 * @param dst      VMEM output buffer base (dstBuf)
 * @param param    Output: agen configs + loop count
 * @param nRx      Number of Rx channels
 * @param nDoppler Number of Doppler bins
 * @param useVectorizedPath Nonzero when the host configured vectorized destination DMA
 */
void transpose_tile_rdx_init(int32_t *src, int32_t *dst, param_transpose_tile_rdx_t *param, int32_t nRx,
                             int32_t nDoppler, int32_t useVectorizedPath);

/*
 * transpose_tile_rdx - Transpose [nRx][nDoppler] complex tile to [nDoppler][nRx].
 *
 * Vectorized path (supports any nRx): restores agen configs from param and runs
 * dvint_load_di + vstore_transp2_i. Scalar fallback restores its agen_A/agen_B
 * configs and runs int_load + int_store.
 *
 * @param src      Input buffer: [nRx][nDoppler][2] int32_t
 * @param dst      Output buffer: [nDoppler][nRx][2] int32_t
 * @param nRx      Number of Rx channels
 * @param nDoppler Number of Doppler bins
 * @param param    Agen configs from transpose_tile_rdx_init()
 */
void transpose_tile_rdx(int32_t *src, int32_t *restrict dst, int32_t nRx, int32_t nDoppler,
                        param_transpose_tile_rdx_t *param);

/*
 * Agen configs for both paths of transpose_chunk_drx.
 * Populated once by transpose_chunk_drx_init() before the tile loop.
 *
 * Vectorized path (nDoppler%8==0, rowWidth%16==0, padded tile fits in scatterBuf):
 *   dvint_load_di + vstore_transp2_i writes padded scatterBuf (bank A).
 *   Followed by cross-superbank compaction: dvint_load from scatterBuf (bank A)
 *   + dvint_store to dstBuf (bank C), producing tight output.
 *   DMA reads from dstBuf.
 *   outLinePitch_drx = ((nRx*chunkRange*2+15)&~15)+2; lane_ofst = outLinePitch_drx/16.
 *
 * Scalar path (all other cases):
 *   int_load/int_store with agen_A/agen_B; writes tight dstBuf directly.
 */
typedef struct
{
    AgenCFG cfg_in_vec[CHUNK_RANGE];  /* vec:    agen_A: n1=nDopplerGroups, s1=vecw, n2=nRx, s2=nDoppler*2 */
    AgenCFG cfg_out_vec[CHUNK_RANGE]; /* vec:    agen_C: scatter to padded scatterBuf rows via lane_ofst    */
    int32_t niter_min;                /* vec:    loop trip count (rounded up for unroll/pipeline, 0=skip)   */
    AgenCFG cfg_cmp_rd;               /* vec:    agen_A: read padded scatterBuf (bank A), n2=nDoppler       */
    AgenCFG cfg_cmp_wr;               /* vec:    agen_B: write tight dstBuf (bank C), n2=nDoppler           */
    int32_t niter_compact;            /* vec:    nDoppler * (rowWidth / vecw)                               */
    AgenCFG cfg_in_scl[CHUNK_RANGE];  /* scalar: agen_A: n1=nDoppler, s1=2, n2=nRx, s2=nDoppler*2          */
    AgenCFG cfg_out_scl[CHUNK_RANGE]; /* scalar: agen_B: n1=nDoppler, s1=nRx*chunkRange*2, s2=chunkRange*2 */
    int32_t niter_scalar;             /* scalar: nRx * nDoppler                                             */
} param_transpose_chunk_drx_t;

/*
 * transpose_chunk_drx_init - Set up agen configs for transpose_chunk_drx.
 *
 * Call ONCE before the tile loop. src/scatter/dst are fixed VMEM buffer addresses
 * (srcBuf / scatterBuf / dstBuf). Initialises scalar agen configs (cfg_in_scl / cfg_out_scl).
 * When nDoppler%8==0, rowWidth%16==0 and the padded tile fits in scatterBuf, also initialises
 * vectorized agen configs (cfg_in_vec / cfg_out_vec / cfg_cmp_rd / cfg_cmp_wr).
 *
 * @param src        VMEM input buffer base (srcBuf, bank B)
 * @param scatter    VMEM padded intermediate (scatterBuf, bank A)
 * @param dst        VMEM tight output base (dstBuf, bank C)
 * @param param      Output: agen configs + loop counts
 * @param chunkRange Number of range bins per chunk (≤ CHUNK_RANGE)
 * @param nRx        Number of Rx channels
 * @param nDoppler   Number of Doppler bins
 */
void transpose_chunk_drx_init(int32_t *src, int32_t *scatter, int32_t *dst, param_transpose_chunk_drx_t *param,
                              int32_t chunkRange, int32_t nRx, int32_t nDoppler);

/*
 * transpose_chunk_drx - Rearrange [CHUNK_RANGE][nRx][nDoppler] to [nDoppler][nRx][CHUNK_RANGE].
 *
 * Vectorized path: writes padded intermediate to scatterBuf (bank A), then compacts
 *   to tight dstBuf (bank C) via dvint_load + dvint_store. DMA reads from dstBuf.
 * Scalar path: int_load/int_store directly to tight dstBuf.
 *
 * @param src        Input buffer: srcBuf (bank B)
 * @param scatter    Padded intermediate: scatterBuf (bank A)
 * @param dst        Tight output: dstBuf (bank C)
 * @param chunkRange Number of range bins in this chunk
 * @param nRx        Number of Rx channels
 * @param nDoppler   Number of Doppler bins
 * @param param      Agen configs from transpose_chunk_drx_init()
 */
void transpose_chunk_drx(int32_t *src, int32_t *restrict scatter, int32_t *restrict dst, int32_t chunkRange,
                         int32_t nRx, int32_t nDoppler, param_transpose_chunk_drx_t *param);

#endif // RADAR_TRANSPOSE_KNL_H
