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

#include "fft_radix5.h"

#include "fft_knl.h"

#include <cupva_device.h>

int32_t func(fft_batched_radix5_init)(STYPE *ping, STYPE *pong, STYPE *tmp, int32_t *twiddle, int32_t shift_qbits,
                                      AgenCFG *cfgs, int32_t &cfg_idx, FFTFactorization *factorization, int32_t it,
                                      int32_t pitch, int32_t groups, int32_t groupStride)
{
    const int32_t vecw                = chess_elementsof(DVTYPEX);
    int32_t n                         = factorization->n;
    const int32_t numR4Stages         = factorization->groups[RADIX4_GROUP_IDX].numStages;
    const int32_t numR5Stages         = factorization->groups[RADIX5_GROUP_IDX].numStages;
    const int32_t numR3Stages         = factorization->groups[RADIX3_GROUP_IDX].numStages;
    const int32_t isNumR4StagesOdd    = factorization->groups[RADIX4_GROUP_IDX].isOdd;
    const int32_t totalPowerOf2Stages = numR4Stages + isNumR4StagesOdd;
    const int32_t groupsz             = factorization->groups[RADIX5_GROUP_IDX].groupSize;
    int32_t ngroups                   = 1;
    bool hasTmpBuf                    = (tmp != NULL) && (numR3Stages == 0 && totalPowerOf2Stages <= 1);

    for (int32_t k = 0; k < numR5Stages; k++)
    {
        int32_t groupsz_radix5 = groupsz * ngroups;
        int32_t ngroups_radix5 = n / groupsz_radix5;
        int32_t dist           = ngroups_radix5 / 5;

        STYPE *src = (it & 1) ? pong : ping;
        STYPE *dst = (k == (numR5Stages - 1) && hasTmpBuf) ? tmp : ((it & 1) ? ping : pong);

        const int32_t niter1 = 5;              // radix-5
        const int32_t niter2 = dist;           // dist
        const int32_t niter3 = groupsz_radix5; // # of groups

        AgenWrapper wrapper;

        agen in      = init(src);
        wrapper.size = sizeof(STYPE);
        wrapper.n1   = niter1;
        wrapper.n2   = niter2;
        wrapper.n3   = niter3;
        wrapper.s1   = dist * pitch;
        wrapper.s2   = pitch;
        wrapper.s3   = ngroups_radix5 * pitch;
        wrapper.n4   = groups;
        wrapper.s4   = groupStride;

        INIT_AGEN4(in, wrapper);

        agen out1  = init(dst);
        out1.round = 0; // radix-5
        wrapper.n1 = 1; // first vstore
        // reuse configurations of src (including the group dimension)
        INIT_AGEN4(out1, wrapper);

        agen out2  = init(dst + dist * pitch);
        out2.round = shift_qbits;
        wrapper.n1 = 4; // remaining vstores
        // reuse configurations of src (including the group dimension)
        INIT_AGEN4(out2, wrapper);

        agen tw5     = init(twiddle);
        wrapper.size = sizeof(int32_t);
        wrapper.n1   = 4;
        wrapper.n2   = (n / 5) * 4;
        wrapper.s1   = 2;
        wrapper.s2   = 0;
        wrapper.n3   = groups;
        wrapper.s3   = 0; // every group replays the same twiddle table
        INIT_AGEN3(tw5, wrapper);

        agen intra_stage_tw = init(twiddle + 4 * COMPLEX_ELEMENT_NUM);
        wrapper.size        = sizeof(int32_t);
        wrapper.n1          = 4;
        wrapper.n2          = niter2;
        wrapper.n3          = niter3;
        wrapper.s1          = 2;
        wrapper.s2          = ngroups * 2 * 4;
        wrapper.s3          = 0;
        wrapper.n4          = groups;
        wrapper.s4          = 0; // every group replays the same twiddle table
        INIT_AGEN4(intra_stage_tw, wrapper);

        cfgs[cfg_idx++] = extract_agen_cfg(in);
        cfgs[cfg_idx++] = extract_agen_cfg(out1);
        cfgs[cfg_idx++] = extract_agen_cfg(out2);
        cfgs[cfg_idx++] = extract_agen_cfg(tw5);
        cfgs[cfg_idx++] = extract_agen_cfg(intra_stage_tw);

        it++;
        ngroups *= 5;
    }

    return it;
}

void func(fft_batched_radix5)(AgenCFG *cfgs, int32_t &cfg_idx, int32_t n, int32_t groups)
{
    DVTYPEX chess_storage(DV0) v11;
    DVTYPEX chess_storage(DV1) v12;
    DVTYPEX chess_storage(DV2) v13;
    DVTYPEX chess_storage(DV3) v14;
    DVTYPEX chess_storage(DV4) v15;

    DVTYPEX chess_storage(DV5) vbf1;
    DVTYPEX chess_storage(DV6) vbf2;
    DVTYPEX chess_storage(DV7) vbf3;
    DVTYPEX chess_storage(DV8) vbf4;
    DVTYPEX chess_storage(DV9) vbf5;

    DVTYPEX divCoeffs;
    divCoeffs.lo = replicatew(13107);
    divCoeffs.hi = replicatew(13107);

    // generate [0, ... ,0, 1, ... ,1]
    vshortx vpmt = vcollate_idx(replicateh(0)) >= chess_elementsof(vintx);

    agen_A in             = init_agen_A_from_cfg(cfgs[cfg_idx++]);
    agen_C out1           = init_agen_C_from_cfg(cfgs[cfg_idx++]);
    agen_C out2           = init_agen_C_from_cfg(cfgs[cfg_idx++]);
    agen_B tw5            = init_agen_B_from_cfg(cfgs[cfg_idx++]);
    agen_B intra_stage_tw = init_agen_B_from_cfg(cfgs[cfg_idx++]);

    const int32_t niter = max(((int32_t)n * groups) / 5, 10);

    for (int32_t i = 0; i < niter; i++) chess_loop_range(10, )
    chess_prepare_for_pipelining
    {
        v11 = DVLOAD_DI(in);
        v12 = DVLOAD_DI(in);
        v13 = DVLOAD_DI(in);
        v14 = DVLOAD_DI(in);
        v15 = DVLOAD_DI(in);

        DVTYPEX vtw12 = dvint_load_perm(tw5, vpmt);
        DVTYPEX vtw13 = dvint_load_perm(tw5, vpmt);
        DVTYPEX vtw14 = dvint_load_perm(tw5, vpmt);
        DVTYPEX vtw15 = dvint_load_perm(tw5, vpmt);

        DVTYPEX vitw12 = dvint_load_perm(intra_stage_tw, vpmt);
        DVTYPEX vitw13 = dvint_load_perm(intra_stage_tw, vpmt);
        DVTYPEX vitw14 = dvint_load_perm(intra_stage_tw, vpmt);
        DVTYPEX vitw15 = dvint_load_perm(intra_stage_tw, vpmt);

        DVTYPEX v11Shifted = v11 << (TWIDDLE_QBITS - TRUNCATE_QBITS);

        // radix-5 butterfly
        vbf1 = dvadd3(dvadd3(v11, v12, v13), v14, v15);

        chess_separator();

        // twiddle multiplication
        DVTYPEX vt2 = dvcmulw_t16(v12, vtw12);
        DVTYPEX vt3 = dvcmulw_t16(v13, vtw13);
        DVTYPEX vt4 = dvcmulw_t16(v14, vtw14);
        DVTYPEX vt5 = dvcmulw_t16(v15, vtw15);

        vbf2 = dvadd3(dvadd3(v11Shifted, vt2, vt3), vt4, vt5);

        chess_separator();

        vt2 = dvcmulw_t16(v12, vtw13);
        vt3 = dvcmulw_t16(v13, vtw15);
        vt4 = dvcmulw_t16(v14, vtw12);
        vt5 = dvcmulw_t16(v15, vtw14);

        vbf3 = dvadd3(dvadd3(v11Shifted, vt2, vt3), vt4, vt5);

        chess_separator();

        vt2 = dvcmulw_t16(v12, vtw14);
        vt3 = dvcmulw_t16(v13, vtw12);
        vt4 = dvcmulw_t16(v14, vtw15);
        vt5 = dvcmulw_t16(v15, vtw13);

        vbf4 = dvadd3(dvadd3(v11Shifted, vt2, vt3), vt4, vt5);

        chess_separator();

        vt2 = dvcmulw_t16(v12, vtw15);
        vt3 = dvcmulw_t16(v13, vtw14);
        vt4 = dvcmulw_t16(v14, vtw13);
        vt5 = dvcmulw_t16(v15, vtw12);

        vbf5 = dvadd3(dvadd3(v11Shifted, vt2, vt3), vt4, vt5);

        chess_separator();

        vbf2 = vbf2 >> (TWIDDLE_QBITS - TRUNCATE_QBITS);
        vbf3 = vbf3 >> (TWIDDLE_QBITS - TRUNCATE_QBITS);
        vbf4 = vbf4 >> (TWIDDLE_QBITS - TRUNCATE_QBITS);
        vbf5 = vbf5 >> (TWIDDLE_QBITS - TRUNCATE_QBITS);

        DVTYPEX vbfN1 = dvmulw_t16(vbf1, divCoeffs);
        DVTYPEX vbfN2 = dvmulw_t16(vbf2, divCoeffs);
        DVTYPEX vbfN3 = dvmulw_t16(vbf3, divCoeffs);
        DVTYPEX vbfN4 = dvmulw_t16(vbf4, divCoeffs);
        DVTYPEX vbfN5 = dvmulw_t16(vbf5, divCoeffs);

        chess_separator();

        /* intra-stage twiddle multiplication */
        vbf2 = dvcmulw_t16(vbfN2, vitw12);
        vbf3 = dvcmulw_t16(vbfN3, vitw13);
        vbf4 = dvcmulw_t16(vbfN4, vitw14);
        vbf5 = dvcmulw_t16(vbfN5, vitw15);

        vstore_i(vbfN1, out1);
        vstore_i(vbf2, out2);
        vstore_i(vbf3, out2);
        vstore_i(vbf4, out2);
        vstore_i(vbf5, out2);
    }
}

void func(fft_batched_radix5_digit_reverse_transpose)(AgenCFG *cfgs, int32_t &cfg_idx, FFTFactorization *factorization,
                                                      int32_t groups)
{
    int32_t n = factorization->n;

    agen_A in  = init_agen_A_from_cfg(cfgs[cfg_idx++]);
    agen_C out = init_agen_C_from_cfg(cfgs[cfg_idx++]);

    const int32_t niter = max(((((int32_t)n * groups) / 5) + 3) & ~3, 10);

    for (int32_t i = 0; i < niter; i++) chess_loop_range(10, )
    chess_unroll_loop(4)
    chess_prepare_for_pipelining
    {
        DVTYPEX v11 = DVLOAD_DI(in);
        DVTYPEX v12 = DVLOAD_DI(in);
        DVTYPEX v13 = DVLOAD_DI(in);
        DVTYPEX v14 = DVLOAD_DI(in);
        DVTYPEX v15 = DVLOAD_DI(in);

        vstore_transp2_i(v11, out);
        vstore_transp2_i(v12, out);
        vstore_transp2_i(v13, out);
        vstore_transp2_i(v14, out);
        vstore_transp2_i(v15, out);
    }
}