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

#include "estimate_noise_floor.h"

#include "capon_roi_part2_config.h"

#include <cupva_device.h>

static const int32_t b_scan_time      = 2;
static const int32_t est_bit_per_scan = 5;

void init_estimate_noise_floor(const uint16_t *const restrict fixed_pt_spectrum, void *const restrict hist_buf,
                               void *const restrict cumul_buf, uint16_t *const restrict noise_floor,
                               Para_capon_roi_part2_config &para)
{
    para.noise_n_roi_groups = CAPON_PART2_ROI_GROUPS;
    para.cumul_buf          = (vshort chess_storage(RAM_Cb) *)cumul_buf;
    {
        AgenWrapper w;
        w.size       = sizeof(dvushort);
        w.n1         = (1 << est_bit_per_scan) - 1;
        w.s1         = 1;
        w.n2         = b_scan_time;
        w.s2         = 0;
        agen a       = init(hist_buf);
        a.sat_opt    = 3; // unsigned comparison and always clip to 0
        a.sat_lim_lo = 0;
        a.sat_val_lo = 0;
        a.sat_lim_hi = 0;
        a.sat_val_hi = 0;
        INIT_AGEN2(a, w);
        para.cfg_hist = extract_agen_cfg(a);
    }
    {
        AgenWrapper w;
        w.size = sizeof(uint16_t);
        w.n1   = CAPON_PART2_NOISE_LOADS;
        w.s1   = chess_elementsof(dvushort);
        w.n2   = b_scan_time;
        w.s2   = 0;
        agen a = init((void *)fixed_pt_spectrum);
        INIT_AGEN2(a, w);
        para.cfg_spectrum = extract_agen_cfg(a);
    }
    {
        AgenWrapper w;
        w.size = sizeof(uint16_t);
        w.n1   = CAPON_PART2_ROI_GROUPS;
        w.s1   = chess_elementsof(vushort);
        agen a = init(noise_floor);
        INIT_AGEN1(a, w);
        para.noise_cfg_noise_floor = extract_agen_cfg(a);
    }
}

void estimate_noise_floor(const Para_capon_roi_part2_config &para)
{
    agen_A a_hist_clr    = init_agen_A_from_cfg(para.cfg_hist);
    agen_B a_spectrum    = init_agen_B_from_cfg(para.cfg_spectrum);
    agen_B a_noise_floor = init_agen_B_from_cfg(para.noise_cfg_noise_floor);

    const uint32_t n_loads_per_scan = CAPON_PART2_NOISE_LOADS;

    vshort chess_storage(RAM_Cb) *const restrict cumul_table = para.cumul_buf;
    const int32_t nth_smallest                               = CAPON_PART2_NOISE_ORDER;

    const dvshortx zero = chess_dont_care(dvshortx) & 0;
    {
        dvshort chess_storage(RAM_Ab) *p = (dvshort chess_storage(RAM_Ab) *)a_hist_clr.a;
        // last bin does not really matter
        for (int32_t i = 0; i < (1 << est_bit_per_scan); ++i) chess_no_unroll *p++ = extract(zero);
    }

    int32_t n_roi_groups = para.noise_n_roi_groups;
    do
    {
        vshortx passed_cnt = replicateh(0);
        vshortx estimated  = replicateh(0);

        for (int32_t b = 0; b < b_scan_time; ++b) chess_unroll_loop(*)
        {
            const bool is_last_iter     = b == b_scan_time - 1;
            const int32_t pre_shift_bit = 16 - est_bit_per_scan * b;
            const int32_t shift_bit     = pre_shift_bit - est_bit_per_scan;

            int16_t chess_storage(RAM_Ab) *const hist = (int16_t chess_storage(RAM_Ab) *)a_hist_clr.a;

            if (b == 0)
            {
                const dvshortx one = zero == zero;

                for (uint32_t i = 0; i < n_loads_per_scan; ++i) chess_prepare_for_pipelining
                chess_unroll_loop(8)
                chess_loop_range(24, )
                {
                    const dvshortx p = dvushort_load(a_spectrum);
                    vhist_simple_32h(hist, p >> shift_bit, one);
                }
            }
            else
            {
                const uint32_t bitmask = UINT32_MAX << pre_shift_bit;
                dvshortx d_estimated;
                d_estimated.hi = d_estimated.lo = estimated;
                const dvshortx shift            = zero - shift_bit;

                for (uint32_t i = 0; i < n_loads_per_scan; ++i) chess_prepare_for_pipelining
                chess_unroll_loop(4)
                chess_loop_range(12, )
                {
                    const dvshortx p        = dvushort_load(a_spectrum);
                    const dvshortx selected = (p & bitmask) == d_estimated;
                    const dvshortx bin      = dvextract_bits(p, shift, (est_bit_per_scan - 1) << 8);
                    vhist_simple_32h(hist, bin, selected);
                }
            }

            chess_separator_scheduler(); // VLD just needs to be after VHist

            vshortx bin = replicateh(0);
            vshortx acc = passed_cnt;

            if (!is_last_iter)
                cumul_table[0] = extract(passed_cnt);

            agen_A a_hist = a_hist_clr;

            /* Scan all 32 histogram bins so the loop count is a multiple of the unroll factor;
             * the last bin cannot change the selected bin, and cumul_table holds 33 entries. */
            {
                const int32_t orig_count = (1 << est_bit_per_scan) - 1;
                const int32_t niter      = (orig_count + 4 - 1) / 4 * 4; /* = 32 */
                for (int32_t i = 0; i < niter; ++i) chess_unroll_loop(4)
                {
                    const dvshortx h = dvushort_load(a_hist);

                    vstore_transp(h /*use loaded h to create in-place dependency*/, a_hist_clr);

                    acc = vadd3_ca(h.lo, h.hi, acc, -1);
                    bin = bin + (acc < nth_smallest);

                    if (!is_last_iter)
                        cumul_table[i + 1] = extract(acc);
                }
            }

            estimated = vshiftor(bin, shift_bit, estimated);

            mem_fence();

            if (!is_last_iter)
            {
                dvshortx t;
                t.hi = t.lo       = bin >> 1;
                const dvshortx dp = vlookup_32h((int16_t chess_storage(RAM_Cb) *)cumul_table, t);
                passed_cnt        = vmux(bin & 1, dp.hi, dp.lo);
            }
        }

        vstore(estimated, a_noise_floor);

        if (n_roi_groups <= 1)
            chess_frequent_then break;
    } while (--n_roi_groups) chess_keep_sw_loop;
}
