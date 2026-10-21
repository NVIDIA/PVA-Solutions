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

#include "median_filter_sepa_2x2_i2s16.h"

// Deinterleave the halo'd source tile into the two channel planes.
void i2s16_deinterleave_src_exec(const MedianFilterSepa2x2PlanarConfig &cfg)
{
    agen_B a_src                                     = init_agen_B_from_cfg(cfg.pre_src_agen_cfg);
    VEC_TYPE chess_storage(RAM_Ab) *restrict ch0_ptr = (VEC_TYPE chess_storage(RAM_Ab) *)cfg.inCh0;
    VEC_TYPE chess_storage(RAM_Cb) *restrict ch1_ptr = (VEC_TYPE chess_storage(RAM_Cb) *)cfg.inCh1;

    int niter = cfg.pre_pass_niter;

    for (int i = 0; i < niter; i++) chess_loop_range(18, )
    chess_unroll_loop(6)
    chess_prepare_for_pipelining
    {
        DVECX_TYPE dv = DVLOAD_DI(a_src);
        *(ch0_ptr++)  = extract(dv.lo);
        *(ch1_ptr++)  = extract(dv.hi);
    }
}

// Re-interleave the two planar outputs into the destination tile.
void i2s16_interleave_dst_exec(const MedianFilterSepa2x2PlanarConfig &cfg)
{
    agen_C a_dst                                     = init_agen_C_from_cfg(cfg.post_dst_agen_cfg);
    VEC_TYPE chess_storage(RAM_Bb) *restrict ch0_ptr = (VEC_TYPE chess_storage(RAM_Bb) *)cfg.outCh0;
    VEC_TYPE chess_storage(RAM_Ab) *restrict ch1_ptr = (VEC_TYPE chess_storage(RAM_Ab) *)cfg.outCh1;

    int niter = cfg.post_pass_niter;

    for (int i = 0; i < niter; i++) chess_loop_range(18, )
    chess_unroll_loop(6)
    chess_prepare_for_pipelining
    {
        DVECX_TYPE dv;
        dv.lo = VEC_EXTEND(*(ch0_ptr++));
        dv.hi = VEC_EXTEND(*(ch1_ptr++));
        vstore_i(dv, a_dst);
    }
}
