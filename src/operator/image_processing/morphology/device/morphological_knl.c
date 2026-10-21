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

#include "morphological_knl.h"

void morphological_filter_init(BASE_TYPE *src, int srcLp, BASE_TYPE *dst, int dstLp, int w, int h, int r, int s,
                               BASE_TYPE *cbStart, int cbSize, AgenCFG *cfg0, AgenCFG *cfg1)
{
    int niter1 = (r + 1) / 2; // number of single row loads
    int niter2 = s + 1;       // number of single column loads
    int niter3 = w / DVECW;   // number of blocks along x
    int niter4 = h / 2;       // number of blocks along y

    AgenWrapper wrapper1, wrapper3;
    agen in, out;

    in            = init(src);
    wrapper1.size = sizeof(BASE_TYPE);
    wrapper1.n1   = niter1;
    wrapper1.n2   = niter2;
    wrapper1.n3   = niter3;
    wrapper1.n4   = niter4;
    wrapper1.s1   = 2;         // pixel shift along row due to double vec
    wrapper1.s2   = srcLp;     // column wise shift based on exec logic
    wrapper1.s3   = DVECW;     // row wise block advance, DVECW cols processed per one iter of n2 loop
    wrapper1.s4   = srcLp * 2; // column wise block advance, 2 rows processed per one iter of n3 loop
    INIT_AGEN4(in, wrapper1);
    in    = update_agen_cb_start(in, (intptr_t)cbStart);
    in    = update_agen_cb_size(in, cbSize);
    *cfg0 = extract_agen_cfg(in);

    out           = init(dst);
    wrapper3.size = sizeof(BASE_TYPE);
    wrapper3.n1   = 2; // double vec
    wrapper3.n2   = niter3;
    wrapper3.n3   = niter4;
    wrapper3.s1   = dstLp;
    wrapper3.s2   = DVECW;
    wrapper3.s3   = dstLp * 2;
    INIT_AGEN3(out, wrapper3);
    *cfg1 = extract_agen_cfg(out);

    chess_separator_scheduler();
}

#ifdef KSIZE_3X3
void morphological_filter_exec(AgenCFG *cfg0, AgenCFG *cfg1)
{
    agen_A in  = init_agen_A_from_cfg(*cfg0);
    agen_B out = init_agen_B_from_cfg(*cfg1);

    for (uint32_t i = 0; i < INNER_LOOPCOUNT; i++) chess_loop_range(16, INNER_LOOPCOUNT)
    chess_unroll_loop(UNROLL_FACTOR)
    chess_prepare_for_pipelining
    {
        // uses double vector loads to load 2 vectors of DVECW horizontally x 4 DVECW vertically
        // computes min in appropriate neighbourhoods
        // writes 2 output rows of DVECW pixels per iter
        //           index:         0    |   1    |   2    |    3
        // First set of pixels:   v00.lo | v00.hi | v02.lo |  v02.hi
        // Second set of pixels : v10.lo | v10.hi | v12.lo |  v12.hi
        // Third set of pixels:   v20.lo | v20.hi | v22.lo |  v22.hi
        // Fourth set of pixels : v30.lo | v30.hi | v32.lo |  v32.hi
        // Row0
        DVEC_TYPE v00 = DLOAD_DI_TYPE(in); // Col[0->DVECW]
        DVEC_TYPE v02 = DLOAD_DI_TYPE(in); // Col[2->DVECW+2]
        // Row1
        DVEC_TYPE v10 = DLOAD_DI_TYPE(in);
        DVEC_TYPE v12 = DLOAD_DI_TYPE(in);
        // Row2
        DVEC_TYPE v20 = DLOAD_DI_TYPE(in);
        DVEC_TYPE v22 = DLOAD_DI_TYPE(in);
        // Row3
        DVEC_TYPE v30 = DLOAD_DI_TYPE(in);
        DVEC_TYPE v32 = DLOAD_DI_TYPE(in);

        DVEC_TYPE vout0;
        DVEC_TYPE vout1;

#    if defined(KSHAPE_RECT)
        // take min/max of kernel spanning pixels
        VEC_TYPE v0 = MORPH_FIL_OP(v00.lo, v00.hi, v02.lo);
        VEC_TYPE v1 = MORPH_FIL_OP(v10.lo, v10.hi, v12.lo);
        VEC_TYPE v2 = MORPH_FIL_OP(v20.lo, v20.hi, v22.lo);
        vout0.lo    = MORPH_FIL_OP(v0, v1, v2);

        VEC_TYPE v3 = MORPH_FIL_OP(v00.hi, v02.lo, v02.hi);
        VEC_TYPE v4 = MORPH_FIL_OP(v10.hi, v12.lo, v12.hi);
        VEC_TYPE v5 = MORPH_FIL_OP(v20.hi, v22.lo, v22.hi);
        vout0.hi    = MORPH_FIL_OP(v3, v4, v5);
        vstore_i(vout0, out);

        v0       = MORPH_FIL_OP(v30.lo, v30.hi, v32.lo);
        vout1.lo = MORPH_FIL_OP(v0, v1, v2);
        v3       = MORPH_FIL_OP(v30.hi, v32.lo, v32.hi);
        vout1.hi = MORPH_FIL_OP(v3, v4, v5);

#    elif defined(KSHAPE_CROSS)

        VEC_TYPE v0 = MORPH_FIL_OP(v00.hi, v10.lo, v10.hi);
        vout0.lo    = MORPH_FIL_OP(v0, v12.lo, v20.hi);
        VEC_TYPE v1 = MORPH_FIL_OP(v02.lo, v10.hi, v12.lo);
        vout0.hi    = MORPH_FIL_OP(v1, v12.hi, v22.lo);
        vstore_i(vout0, out);

        v0       = MORPH_FIL_OP(v10.hi, v20.lo, v20.hi);
        vout1.lo = MORPH_FIL_OP(v0, v22.lo, v30.hi);
        v1       = MORPH_FIL_OP(v12.lo, v20.hi, v22.lo);
        vout1.hi = MORPH_FIL_OP(v1, v22.hi, v32.lo);

#    endif
        vstore_i(vout1, out);
    }
}
#endif

#if defined(KSIZE_5X5) && defined(KSHAPE_RECT)
void morphological_filter_exec(AgenCFG *cfg0, AgenCFG *cfg1)
{
    agen_A in  = init_agen_A_from_cfg(*cfg0);
    agen_B out = init_agen_B_from_cfg(*cfg1);

    for (uint32_t i = 0; i < INNER_LOOPCOUNT; i++) chess_loop_range(16, INNER_LOOPCOUNT)
    chess_prepare_for_pipelining
    {
        // uses double vector loads to load 3 vectors of DVECW horizontally x 6 DVECW vertically
        // computes min in appropriate neighbourhoods
        // writes 2 output rows of DVECW pixels per iter
        //    index:         0    |   1    |   2    |    3    |   4    |   5
        // pixels set 1:   v00.lo | v00.hi | v02.lo |  v02.hi | v04.lo | v04.hi
        // pixels set 2:   v10.lo | v10.hi | v12.lo |  v12.hi | v14.lo | v14.hi
        // pixels set 3:   v20.lo | v20.hi | v22.lo |  v22.hi | v24.lo | v24.hi
        // pixels set 4:   v30.lo | v30.hi | v32.lo |  v32.hi | v34.lo | v34.hi
        // pixels set 5:   v00.lo | v00.hi | v02.lo |  v02.hi | v44.lo | v44.hi
        // pixels set 6:   v10.lo | v10.hi | v12.lo |  v12.hi | v54.lo | v54.hi
        // Row0
        DVEC_TYPE v00 = DLOAD_DI_TYPE(in); // Col[0->DVECW]
        DVEC_TYPE v02 = DLOAD_DI_TYPE(in); // Col[2->DVECW+2]
        DVEC_TYPE v04 = DLOAD_DI_TYPE(in); // Col[4->DVECW+4]

        // Row1
        DVEC_TYPE v10 = DLOAD_DI_TYPE(in);
        DVEC_TYPE v12 = DLOAD_DI_TYPE(in);
        DVEC_TYPE v14 = DLOAD_DI_TYPE(in);

        // Row2
        DVEC_TYPE v20 = DLOAD_DI_TYPE(in);
        DVEC_TYPE v22 = DLOAD_DI_TYPE(in);
        DVEC_TYPE v24 = DLOAD_DI_TYPE(in);

        // Row3
        DVEC_TYPE v30 = DLOAD_DI_TYPE(in);
        DVEC_TYPE v32 = DLOAD_DI_TYPE(in);
        DVEC_TYPE v34 = DLOAD_DI_TYPE(in);

        // Row4
        DVEC_TYPE v40 = DLOAD_DI_TYPE(in);
        DVEC_TYPE v42 = DLOAD_DI_TYPE(in);
        DVEC_TYPE v44 = DLOAD_DI_TYPE(in);

        // Row5
        DVEC_TYPE v50 = DLOAD_DI_TYPE(in);
        DVEC_TYPE v52 = DLOAD_DI_TYPE(in);
        DVEC_TYPE v54 = DLOAD_DI_TYPE(in);

        DVEC_TYPE vout0;
        DVEC_TYPE vout1;

        // take min/max of kernel spanning pixels
        VEC_TYPE v0   = MORPH_FIL_OP(v00.lo, v00.hi, v02.lo);
        v0            = MORPH_FIL_OP(v0, v02.hi, v04.lo);
        VEC_TYPE v1   = MORPH_FIL_OP(v10.lo, v10.hi, v12.lo);
        v1            = MORPH_FIL_OP(v1, v12.hi, v14.lo);
        VEC_TYPE v2   = MORPH_FIL_OP(v20.lo, v20.hi, v22.lo);
        v2            = MORPH_FIL_OP(v2, v22.hi, v24.lo);
        VEC_TYPE v3   = MORPH_FIL_OP(v30.lo, v30.hi, v32.lo);
        v3            = MORPH_FIL_OP(v3, v32.hi, v34.lo);
        VEC_TYPE v4   = MORPH_FIL_OP(v40.lo, v40.hi, v42.lo);
        v4            = MORPH_FIL_OP(v4, v42.hi, v44.lo);
        VEC_TYPE vtmp = MORPH_FIL_OP(v1, v2, v3);
        vout0.lo      = MORPH_FIL_OP(vtmp, v0, v4);

        VEC_TYPE v5    = MORPH_FIL_OP(v00.hi, v02.lo, v02.hi);
        v5             = MORPH_FIL_OP(v5, v04.lo, v04.hi);
        VEC_TYPE v6    = MORPH_FIL_OP(v10.hi, v12.lo, v12.hi);
        v6             = MORPH_FIL_OP(v6, v14.lo, v14.hi);
        VEC_TYPE v7    = MORPH_FIL_OP(v20.hi, v22.lo, v22.hi);
        v7             = MORPH_FIL_OP(v7, v24.lo, v24.hi);
        VEC_TYPE v8    = MORPH_FIL_OP(v30.hi, v32.lo, v32.hi);
        v8             = MORPH_FIL_OP(v8, v34.lo, v34.hi);
        VEC_TYPE v9    = MORPH_FIL_OP(v40.hi, v42.lo, v42.hi);
        v9             = MORPH_FIL_OP(v9, v44.lo, v44.hi);
        VEC_TYPE vtmp1 = MORPH_FIL_OP(v6, v7, v8);
        vout0.hi       = MORPH_FIL_OP(v5, vtmp1, v9);
        vstore_i(vout0, out);

        VEC_TYPE _v0 = MORPH_FIL_OP(v50.lo, v50.hi, v52.lo);
        _v0          = MORPH_FIL_OP(_v0, v52.hi, v54.lo);
        vout1.lo     = MORPH_FIL_OP(_v0, vtmp, v4);

        VEC_TYPE _v5 = MORPH_FIL_OP(v50.hi, v52.lo, v52.hi);
        _v5          = MORPH_FIL_OP(_v5, v54.lo, v54.hi);
        vout1.hi     = MORPH_FIL_OP(_v5, vtmp1, v9);
        vstore_i(vout1, out);
    }
}
#endif