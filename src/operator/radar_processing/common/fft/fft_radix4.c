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
#include "fft_radix4.h"

#include "fft_knl.h"

#include <cupva_device.h>

int32_t func(fft_batched_radix4_init)(STYPE *ping, STYPE *pong, STYPE *tmp, int32_t *twiddle, int32_t shift_qbits,
                                      AgenCFG *cfgs, int32_t &cfg_idx, FFTFactorization *factorization, int32_t it,
                                      int32_t pitch, int32_t groups, int32_t groupStride, int32_t stage_round)
{
    const int32_t n                   = factorization->n;
    const int32_t isNumR4StagesOdd    = factorization->groups[RADIX4_GROUP_IDX].isOdd;
    const int32_t numR4Stages         = factorization->groups[RADIX4_GROUP_IDX].numStages;
    const int32_t totalPowerOf2Stages = numR4Stages + isNumR4StagesOdd - 1;
    const int32_t groupsz             = factorization->groups[RADIX4_GROUP_IDX].groupSize;
    const int32_t vecw                = chess_elementsof(DVTYPEX);
    int32_t ngroups                   = 1;
    bool hasTmpBuf                    = (tmp != NULL);

    for (int32_t k = 0; k < totalPowerOf2Stages; k++)
    {
        int32_t groupsz_radix4 = groupsz * ngroups;
        int32_t ngroups_radix4 = n / groupsz_radix4;
        int32_t dist           = ngroups_radix4 / 4;

        STYPE *src = (it & 1) ? pong : ping;
        STYPE *dst = ((k == totalPowerOf2Stages - 1) && hasTmpBuf) ? tmp : ((it & 1) ? ping : pong);

        const int32_t niter1 = 4;              // radix-4
        const int32_t niter2 = dist;           // dist
        const int32_t niter3 = groupsz_radix4; // # of groups

        AgenWrapper wrapper;

        agen in      = init(src);
        wrapper.size = sizeof(STYPE);
        wrapper.n1   = niter1;
        wrapper.n2   = niter2;
        wrapper.n3   = niter3;
        wrapper.s1   = dist * pitch;
        wrapper.s2   = pitch;
        wrapper.s3   = ngroups_radix4 * pitch;
        wrapper.n4   = groups;
        wrapper.s4   = groupStride;
        INIT_AGEN4(in, wrapper);

        agen out1  = init(dst);
        out1.round = stage_round; // radix-4 growth; 2 unless the caller keeps its own scaling
        wrapper.n1 = 1;           // first vstore
        // reuse configurations of src (including the group dimension)
        INIT_AGEN4(out1, wrapper);

        agen out2  = init(dst + dist * pitch);
        out2.round = stage_round + shift_qbits;
        wrapper.n1 = 3; // remaining vstores
        // reuse configurations of src (including the group dimension)
        INIT_AGEN4(out2, wrapper);

        agen tw      = init(twiddle);
        wrapper.size = sizeof(int32_t);
        wrapper.n1   = 3;
        wrapper.n2   = niter2;
        wrapper.n3   = niter3;
        wrapper.s1   = 2;
        wrapper.s2   = ngroups * 2 * 3;
        wrapper.s3   = 0;
        wrapper.n4   = groups;
        wrapper.s4   = 0; // every group replays the same twiddle table
        INIT_AGEN4(tw, wrapper);

        cfgs[cfg_idx++] = extract_agen_cfg(in);
        cfgs[cfg_idx++] = extract_agen_cfg(out1);
        cfgs[cfg_idx++] = extract_agen_cfg(out2);
        cfgs[cfg_idx++] = extract_agen_cfg(tw);

        it++;
        ngroups = ngroups << 2;
    }

    return it;
}

void func(fft_batched_radix4)(AgenCFG *cfgs, int32_t &cfg_idx, int32_t n, int32_t groups)
{
    DVTYPEX chess_storage(DV0) v11;
    DVTYPEX chess_storage(DV1) v12;
    DVTYPEX chess_storage(DV2) v13;
    DVTYPEX chess_storage(DV3) v14;
    DVTYPEX chess_storage(DV4) v21;
    DVTYPEX chess_storage(DV5) v22;
    DVTYPEX chess_storage(DV6) v23;
    DVTYPEX chess_storage(DV7) v24;
    DVTYPEX chess_storage(DV8) v31;
    DVTYPEX chess_storage(DV9) v32;
    DVTYPEX chess_storage(DV10) v33;
    DVTYPEX chess_storage(DV11) v34;

    DVTYPEX chess_storage(DV12) vbf1;
    DVTYPEX chess_storage(DV13) vbf2;
    DVTYPEX chess_storage(DV14) vbf3;
    DVTYPEX chess_storage(DV15) vbf4;

    DVTYPEX v5, v6, v7;

    // generate [0, ... ,0, 1, ... ,1]
    vshortx vpmt = vcollate_idx(replicateh(0)) >= chess_elementsof(vintx);

    agen_A in   = init_agen_A_from_cfg(cfgs[cfg_idx++]);
    agen_C out1 = init_agen_C_from_cfg(cfgs[cfg_idx++]);
    agen_C out2 = init_agen_C_from_cfg(cfgs[cfg_idx++]);
    agen_B tw   = init_agen_B_from_cfg(cfgs[cfg_idx++]);

    const int32_t niter = max((((int32_t)n * groups) / 4 + 2) / 3, 3);

    for (int32_t i = 0; i < niter; i++) chess_loop_range(3, )
    chess_prepare_for_pipelining
    {
        v11 = DVLOAD_DI(in);
        v12 = DVLOAD_DI(in);
        v13 = DVLOAD_DI(in);
        v14 = DVLOAD_DI(in);

        v21 = DVLOAD_DI(in);
        v22 = DVLOAD_DI(in);
        v23 = DVLOAD_DI(in);
        v24 = DVLOAD_DI(in);

        v31 = DVLOAD_DI(in);
        v32 = DVLOAD_DI(in);
        v33 = DVLOAD_DI(in);
        v34 = DVLOAD_DI(in);

        DVTYPEX vtw12 = dvint_load_perm(tw, vpmt);
        DVTYPEX vtw13 = dvint_load_perm(tw, vpmt);
        DVTYPEX vtw14 = dvint_load_perm(tw, vpmt);

        DVTYPEX vtw22 = dvint_load_perm(tw, vpmt);
        DVTYPEX vtw23 = dvint_load_perm(tw, vpmt);
        DVTYPEX vtw24 = dvint_load_perm(tw, vpmt);

        DVTYPEX vtw32 = dvint_load_perm(tw, vpmt);
        DVTYPEX vtw33 = dvint_load_perm(tw, vpmt);
        DVTYPEX vtw34 = dvint_load_perm(tw, vpmt);

        chess_separator();

        vaddsub4x2_0(v11.lo, v12.lo, v13.lo, v14.lo, vbf1.lo, vbf3.lo);
        vaddsub4x2_0(v11.hi, v12.hi, v13.hi, v14.hi, vbf1.hi, vbf3.hi);

        chess_separator();

        vaddsub4x2_1(v11.lo, v12.hi, v13.lo, v14.hi, vbf2.lo, vbf4.lo);
        vaddsub4x2_1(v11.hi, v12.lo, v13.hi, v14.lo, vbf4.hi, vbf2.hi);

        chess_separator();

        v5 = dvcmulw_t16(vbf2, vtw12);
        v6 = dvcmulw_t16(vbf3, vtw13);
        v7 = dvcmulw_t16(vbf4, vtw14);

        vstore_i(vbf1, out1);
        vstore_i(v5, out2);
        vstore_i(v6, out2);
        vstore_i(v7, out2);

        chess_separator();

        vaddsub4x2_0(v21.lo, v22.lo, v23.lo, v24.lo, vbf1.lo, vbf3.lo);
        vaddsub4x2_0(v21.hi, v22.hi, v23.hi, v24.hi, vbf1.hi, vbf3.hi);

        chess_separator();

        vaddsub4x2_1(v21.lo, v22.hi, v23.lo, v24.hi, vbf2.lo, vbf4.lo);
        vaddsub4x2_1(v21.hi, v22.lo, v23.hi, v24.lo, vbf4.hi, vbf2.hi);

        chess_separator();

        v5 = dvcmulw_t16(vbf2, vtw22);
        v6 = dvcmulw_t16(vbf3, vtw23);
        v7 = dvcmulw_t16(vbf4, vtw24);

        vstore_i(vbf1, out1);
        vstore_i(v5, out2);
        vstore_i(v6, out2);
        vstore_i(v7, out2);

        chess_separator();

        vaddsub4x2_0(v31.lo, v32.lo, v33.lo, v34.lo, vbf1.lo, vbf3.lo);
        vaddsub4x2_0(v31.hi, v32.hi, v33.hi, v34.hi, vbf1.hi, vbf3.hi);

        chess_separator();

        vaddsub4x2_1(v31.lo, v32.hi, v33.lo, v34.hi, vbf2.lo, vbf4.lo);
        vaddsub4x2_1(v31.hi, v32.lo, v33.hi, v34.lo, vbf4.hi, vbf2.hi);

        chess_separator();

        v5 = dvcmulw_t16(vbf2, vtw32);
        v6 = dvcmulw_t16(vbf3, vtw33);
        v7 = dvcmulw_t16(vbf4, vtw34);

        vstore_i(vbf1, out1);
        vstore_i(v5, out2);
        vstore_i(v6, out2);
        vstore_i(v7, out2);
    }
}

void func(fft_batched_radix4_digit_reverse_transpose)(AgenCFG *cfgs, int32_t &cfg_idx, FFTFactorization *factorization,
                                                      int32_t groups)
{
    int32_t n = factorization->n;
    DVTYPEX chess_storage(DV0) v11;
    DVTYPEX chess_storage(DV1) v12;
    DVTYPEX chess_storage(DV2) v13;
    DVTYPEX chess_storage(DV3) v14;
    DVTYPEX chess_storage(DV4) v21;
    DVTYPEX chess_storage(DV5) v22;
    DVTYPEX chess_storage(DV6) v23;
    DVTYPEX chess_storage(DV7) v24;
    DVTYPEX chess_storage(DV8) v31;
    DVTYPEX chess_storage(DV9) v32;
    DVTYPEX chess_storage(DV10) v33;
    DVTYPEX chess_storage(DV11) v34;

    DVTYPEX chess_storage(DV12) vbf1;
    DVTYPEX chess_storage(DV13) vbf2;
    DVTYPEX chess_storage(DV14) vbf3;
    DVTYPEX chess_storage(DV15) vbf4;

    agen_A in  = init_agen_A_from_cfg(cfgs[cfg_idx++]);
    agen_C out = init_agen_C_from_cfg(cfgs[cfg_idx++]);

    const int32_t niter = max((((int32_t)n * groups) / 4 + 2) / 3, 3);

    /* Always the transposed store: when the output is not transposed the init leaves
     * lane_ofst at 0, and the transposed store then writes the same contiguous vector as a
     * plain one. */
    for (int32_t i = 0; i < niter; i++) chess_loop_range(3, )
    chess_prepare_for_pipelining
    {
        v11 = DVLOAD_DI(in);
        v12 = DVLOAD_DI(in);
        v13 = DVLOAD_DI(in);
        v14 = DVLOAD_DI(in);

        v21 = DVLOAD_DI(in);
        v22 = DVLOAD_DI(in);
        v23 = DVLOAD_DI(in);
        v24 = DVLOAD_DI(in);

        v31 = DVLOAD_DI(in);
        v32 = DVLOAD_DI(in);
        v33 = DVLOAD_DI(in);
        v34 = DVLOAD_DI(in);

        chess_separator();

        vaddsub4x2_0(v11.lo, v12.lo, v13.lo, v14.lo, vbf1.lo, vbf3.lo);
        vaddsub4x2_0(v11.hi, v12.hi, v13.hi, v14.hi, vbf1.hi, vbf3.hi);

        chess_separator();

        vaddsub4x2_1(v11.lo, v12.hi, v13.lo, v14.hi, vbf2.lo, vbf4.lo);
        vaddsub4x2_1(v11.hi, v12.lo, v13.hi, v14.lo, vbf4.hi, vbf2.hi);

        chess_separator();

        vstore_transp2_i(vbf1, out);
        vstore_transp2_i(vbf2, out);
        vstore_transp2_i(vbf3, out);
        vstore_transp2_i(vbf4, out);

        chess_separator();

        vaddsub4x2_0(v21.lo, v22.lo, v23.lo, v24.lo, vbf1.lo, vbf3.lo);
        vaddsub4x2_0(v21.hi, v22.hi, v23.hi, v24.hi, vbf1.hi, vbf3.hi);

        chess_separator();

        vaddsub4x2_1(v21.lo, v22.hi, v23.lo, v24.hi, vbf2.lo, vbf4.lo);
        vaddsub4x2_1(v21.hi, v22.lo, v23.hi, v24.lo, vbf4.hi, vbf2.hi);

        chess_separator();

        vstore_transp2_i(vbf1, out);
        vstore_transp2_i(vbf2, out);
        vstore_transp2_i(vbf3, out);
        vstore_transp2_i(vbf4, out);

        chess_separator();

        vaddsub4x2_0(v31.lo, v32.lo, v33.lo, v34.lo, vbf1.lo, vbf3.lo);
        vaddsub4x2_0(v31.hi, v32.hi, v33.hi, v34.hi, vbf1.hi, vbf3.hi);

        chess_separator();

        vaddsub4x2_1(v31.lo, v32.hi, v33.lo, v34.hi, vbf2.lo, vbf4.lo);
        vaddsub4x2_1(v31.hi, v32.lo, v33.hi, v34.lo, vbf4.hi, vbf2.hi);

        chess_separator();

        vstore_transp2_i(vbf1, out);
        vstore_transp2_i(vbf2, out);
        vstore_transp2_i(vbf3, out);
        vstore_transp2_i(vbf4, out);
    }
}

int32_t func(fft_batched_radix4_digit_reverse_power_init)(STYPE *src, STYPE *dst, AgenCFG *cfgs, int32_t &cfg_idx,
                                                          int32_t log2N, bool need_transpose, int32_t it,
                                                          int32_t round_qbits, int32_t in_width, int32_t in_pad,
                                                          int32_t out_width, int32_t out_pad, int32_t iter,
                                                          int32_t kernel_batch)
{
    return func(fft_batched_digit_reverse_power_init)(src, dst, cfgs, cfg_idx, log2N, need_transpose, it, 4,
                                                      round_qbits, in_width, in_pad, out_width, out_pad, iter,
                                                      kernel_batch);
}

float func(fft_batched_radix4_digit_reverse_power)(AgenCFG *cfgs, int32_t &cfg_idx, int32_t n, int32_t iter,
                                                   int32_t round_qbits)
{
    DVTYPEX chess_storage(DV0) v11;
    DVTYPEX chess_storage(DV1) v12;
    DVTYPEX chess_storage(DV2) v13;
    DVTYPEX chess_storage(DV3) v14;
    DVTYPEX chess_storage(DV4) v21;
    DVTYPEX chess_storage(DV5) v22;
    DVTYPEX chess_storage(DV6) v23;
    DVTYPEX chess_storage(DV7) v24;
    DVTYPEX chess_storage(DV8) v31;
    DVTYPEX chess_storage(DV9) v32;
    DVTYPEX chess_storage(DV10) v33;
    DVTYPEX chess_storage(DV11) v34;

    DVTYPEX chess_storage(DV12) vbf1;
    DVTYPEX chess_storage(DV13) vbf2;
    DVTYPEX chess_storage(DV14) vbf3;
    DVTYPEX chess_storage(DV15) vbf4;

    vfloatx vFFTFinal1;
    vfloatx vFFTFinal2;
    vfloatx vFFTFinal3;
    vfloatx vFFTFinal4;

    agen_A in             = init_agen_A_from_cfg(cfgs[cfg_idx++]);
    agen_C power_spectrum = init_agen_C_from_cfg(cfgs[cfg_idx++]);

    const int32_t niter = max(((((n / 4) * iter) + 2) / 3), 11);

    for (int32_t i = 0; i < niter; i++) chess_loop_range(11, )
    chess_prepare_for_pipelining
    {
        v11 = dvint_load_di(in);
        v12 = dvint_load_di(in);
        v13 = dvint_load_di(in);
        v14 = dvint_load_di(in);

        v21 = dvint_load_di(in);
        v22 = dvint_load_di(in);
        v23 = dvint_load_di(in);
        v24 = dvint_load_di(in);

        v31 = dvint_load_di(in);
        v32 = dvint_load_di(in);
        v33 = dvint_load_di(in);
        v34 = dvint_load_di(in);

        chess_separator();

        vaddsub4x2_0(v11.lo, v12.lo, v13.lo, v14.lo, vbf1.lo, vbf3.lo);
        vaddsub4x2_0(v11.hi, v12.hi, v13.hi, v14.hi, vbf1.hi, vbf3.hi);

        chess_separator();

        vaddsub4x2_1(v11.lo, v12.hi, v13.lo, v14.hi, vbf2.lo, vbf4.lo);
        vaddsub4x2_1(v11.hi, v12.lo, v13.hi, v14.lo, vbf4.hi, vbf2.hi);

        chess_separator();

        vFFTFinal1 = func(round_calc_power)(vbf1, round_qbits);
        vFFTFinal2 = func(round_calc_power)(vbf2, round_qbits);
        vFFTFinal3 = func(round_calc_power)(vbf3, round_qbits);
        vFFTFinal4 = func(round_calc_power)(vbf4, round_qbits);

        vstore(vFFTFinal1, power_spectrum);
        vstore(vFFTFinal2, power_spectrum);
        vstore(vFFTFinal3, power_spectrum);
        vstore(vFFTFinal4, power_spectrum);

        chess_separator();

        vaddsub4x2_0(v21.lo, v22.lo, v23.lo, v24.lo, vbf1.lo, vbf3.lo);
        vaddsub4x2_0(v21.hi, v22.hi, v23.hi, v24.hi, vbf1.hi, vbf3.hi);

        chess_separator();

        vaddsub4x2_1(v21.lo, v22.hi, v23.lo, v24.hi, vbf2.lo, vbf4.lo);
        vaddsub4x2_1(v21.hi, v22.lo, v23.hi, v24.lo, vbf4.hi, vbf2.hi);

        chess_separator();
        vFFTFinal1 = func(round_calc_power)(vbf1, round_qbits);
        vFFTFinal2 = func(round_calc_power)(vbf2, round_qbits);
        vFFTFinal3 = func(round_calc_power)(vbf3, round_qbits);
        vFFTFinal4 = func(round_calc_power)(vbf4, round_qbits);

        vstore(vFFTFinal1, power_spectrum);
        vstore(vFFTFinal2, power_spectrum);
        vstore(vFFTFinal3, power_spectrum);
        vstore(vFFTFinal4, power_spectrum);

        chess_separator();

        vaddsub4x2_0(v31.lo, v32.lo, v33.lo, v34.lo, vbf1.lo, vbf3.lo);
        vaddsub4x2_0(v31.hi, v32.hi, v33.hi, v34.hi, vbf1.hi, vbf3.hi);

        chess_separator();

        vaddsub4x2_1(v31.lo, v32.hi, v33.lo, v34.hi, vbf2.lo, vbf4.lo);
        vaddsub4x2_1(v31.hi, v32.lo, v33.hi, v34.lo, vbf4.hi, vbf2.hi);

        chess_separator();

        vFFTFinal1 = func(round_calc_power)(vbf1, round_qbits);
        vFFTFinal2 = func(round_calc_power)(vbf2, round_qbits);
        vFFTFinal3 = func(round_calc_power)(vbf3, round_qbits);
        vFFTFinal4 = func(round_calc_power)(vbf4, round_qbits);
        vstore(vFFTFinal1, power_spectrum);
        vstore(vFFTFinal2, power_spectrum);
        vstore(vFFTFinal3, power_spectrum);
        vstore(vFFTFinal4, power_spectrum);
    }

    int power_max_val_int     = power_spectrum.max_val;
    float power_max_val_float = *((float *)(&power_max_val_int));
    return power_max_val_float;
}
