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

#include "distance_transform_knl.h"

// -------------------------------------------------------------------------------- //
// stage1 kernels                                                                   //
// -------------------------------------------------------------------------------- //
int distance_transform_stage1_init(VX_USHORT_TILE *tile, int niter, int max_dist, param_dt_stage1_t *param)
{
    int vecw                  = pva_elementsof(dvshortx);
    unsigned short *base_addr = tile->tp;

    agen in1, out1, in2, out2;

    in1           = init((vshort *)base_addr);
    in1.n1        = niter;
    in1.mod1      = sizeof(unsigned short);
    in1.lane_ofst = tile->lp / 32;

    out1           = init((vshort *)base_addr);
    out1.n1        = niter;
    out1.mod1      = sizeof(unsigned short);
    out1.lane_ofst = tile->lp / 32;

    in2           = init((vshort *)(base_addr + niter - 1));
    in2.n1        = niter;
    in2.mod1      = -(int32_t)(sizeof(unsigned short));
    in2.lane_ofst = tile->lp / 32;

    out2           = init((vshort *)(base_addr + niter - 1));
    out2.n1        = niter;
    out2.mod1      = -(int32_t)(sizeof(unsigned short));
    out2.lane_ofst = tile->lp / 32;

    param->cfgs[0]  = extract_agen_cfg(in1);
    param->cfgs[1]  = extract_agen_cfg(out1);
    param->cfgs[2]  = extract_agen_cfg(in2);
    param->cfgs[3]  = extract_agen_cfg(out2);
    param->niter    = niter;
    param->max_dist = max_dist;
    param->w        = tile->tw;

    return 0;
}

int distance_transform_stage1_l2r(param_dt_stage1_t *param)
{
    int i;
    int niter    = param->niter;
    int max_dist = param->max_dist;
    dvshortx map_data;
    dvshortx on_pix;
    dvshortx prev_dist;
    dvshortx prev_label;
    dvshortx hd;
    dvshortx label;
    dvshortx dist;
    dvshortx prop_shorter;
    vshortx const_zero;

    agen_B in1  = init_agen_B_from_cfg(param->cfgs[0]);
    agen_B out1 = init_agen_B_from_cfg(param->cfgs[1]);

    const_zero    = replicateh(0);
    prev_dist.lo  = replicateh(max_dist);
    prev_dist.hi  = replicateh(max_dist);
    prev_label.lo = replicateh(0);
    prev_label.hi = replicateh(0);

    for (i = 0; i < niter; i++) chess_prepare_for_pipelining
    chess_unroll_loop(4)
    chess_loop_range(12, )
    {
        map_data  = dvshort_load_transp(in1);
        on_pix    = (map_data != -1);
        prev_dist = dvmux(on_pix, 0, prev_dist + 1);

        chess_separator();

        prev_label = dvmux(on_pix, map_data, prev_label);
        hd.lo      = vshiftor(prev_label.lo, 12, prev_dist.lo);
        hd.hi      = vshiftor(prev_label.hi, 12, prev_dist.hi);
        vstore_transp(hd, out1);
    }

    return 0;
}

int distance_transform_stage1_r2l(param_dt_stage1_t *param)
{
    int i;
    int niter    = param->niter;
    int max_dist = param->max_dist;
    int w        = param->w;
    dvshortx map_data;
    dvshortx on_pix;
    dvshortx prev_dist;
    dvshortx prev_label;
    dvshortx hd;
    dvshortx label;
    dvshortx dist;
    dvshortx prop_shorter;
    vshortx const_zero;

    agen_B in2  = init_agen_B_from_cfg(param->cfgs[2]);
    agen_B out2 = init_agen_B_from_cfg(param->cfgs[3]);

    prev_dist.lo  = replicateh(max_dist);
    prev_dist.hi  = replicateh(max_dist);
    prev_label.lo = replicateh(0);
    prev_label.hi = replicateh(0);

    for (i = niter - 1; i >= 0; i--) chess_prepare_for_pipelining
    chess_unroll_loop(2)
    chess_loop_range(12, )
    {
        hd           = dvshort_load_transp(in2);
        dist.lo      = vextract_bits(hd.lo, replicateh(0), 0xB00);
        dist.hi      = vextract_bits(hd.hi, replicateh(0), 0xB00);
        label.lo     = vextract_bits(hd.lo, replicateh(-12), 0x300);
        label.hi     = vextract_bits(hd.hi, replicateh(-12), 0x300);
        prev_dist    = prev_dist + 1;
        prop_shorter = (prev_dist < dist);

        chess_separator();
        prev_dist = dvmin(prev_dist, w);

        prev_dist.lo  = vmux(prop_shorter.lo, prev_dist.lo, dist.lo);
        prev_dist.hi  = vmux(prop_shorter.hi, prev_dist.hi, dist.hi);
        prev_label.lo = vmux(prop_shorter.lo, prev_label.lo, label.lo);
        prev_label.hi = vmux(prop_shorter.hi, prev_label.hi, label.hi);
        hd.lo         = vshiftor(prev_label.lo, 12, prev_dist.lo);
        hd.hi         = vshiftor(prev_label.hi, 12, prev_dist.hi);
        vstore_transp(hd, out2);
    }

    return 0;
}

// -------------------------------------------------------------------------------- //
// stage2 kernels                                                                   //
// -------------------------------------------------------------------------------- //
// --------------------------------------------------------- //
// this macro calculates the dividend according to the eq:   //
// n = add0 * add0 - sub0 * sub0 + add1 * add1 - sub1 * sub1 //
// --------------------------------------------------------- //
#define CALC_N(n_lo, n_hi, add0, sub0, add1, sub1)        \
    (n_lo) = vmulhw((add0).lo, (add0).lo, 0);             \
    (n_hi) = vmulhw((add0).hi, (add0).hi, 0);             \
    chess_separator();                                    \
    (n_lo) = vmsubhw((sub0).lo, (sub0).lo, (n_lo), 0, 1); \
    (n_hi) = vmsubhw((sub0).hi, (sub0).hi, (n_hi), 0, 1); \
    chess_separator();                                    \
    (n_lo) = vmaddhw((add1).lo, (add1).lo, (n_lo), 0, 1); \
    (n_hi) = vmaddhw((add1).hi, (add1).hi, (n_hi), 0, 1); \
    chess_separator();                                    \
    (n_lo) = vmsubhw((sub1).lo, (sub1).lo, (n_lo), 0, 1); \
    (n_hi) = vmsubhw((sub1).hi, (sub1).hi, (n_hi), 0, 1);

// --------------------------------------------------------- //
// this macro calculates the sum of square according to eq:  //
// d = add0 * add0 + add1 * add1                             //
// --------------------------------------------------------- //
#define SUM_SQR(d_lo, d_hi, add0, add1)                   \
    (d_lo) = vmulhw((add0).lo, (add0).lo, 0);             \
    (d_hi) = vmulhw((add0).hi, (add0).hi, 0);             \
    chess_separator();                                    \
    (d_lo) = vmaddhw((add1).lo, (add1).lo, (d_lo), 0, 1); \
    (d_hi) = vmaddhw((add1).hi, (add1).hi, (d_hi), 0, 1);

#define SUM_SQR_H(d_lo, d_hi, add0, add1)                \
    (d_lo) = vmulh((add0).lo, (add0).lo, 0);             \
    (d_hi) = vmulh((add0).hi, (add0).hi, 0);             \
    chess_separator();                                   \
    (d_lo) = vmaddh((add1).lo, (add1).lo, (d_lo), 0, 1); \
    (d_hi) = vmaddh((add1).hi, (add1).hi, (d_hi), 0, 1);

// --------------------------------------------------------- //
// this inline functions calculates sqrt of a short with     //
// lookup table                                              //
// --------------------------------------------------------- //
inline dvshortx sqrt_dist(dvshortx din, unsigned short *sqrt_tbl)
{
    dvshortx exp, mant, idx, frac;
    dvshortx y0, y1, z, z2;
    dvshortx exp_odd, exp_dv2;

    exp = dvmsbd(din);

    chess_separator();

    mant.lo = vshift(din.lo, replicateh(23) - exp.lo); // move MSB to bit 23
    mant.hi = vshift(din.hi, replicateh(23) - exp.hi);

    chess_separator();

    idx.lo = vextract_bits(mant.lo, replicateh(-17), 0x500); // shift by tbl_bits-23
    idx.hi = vextract_bits(mant.hi, replicateh(-17), 0x500);

    chess_separator();

    frac.lo = vextract_bits(mant.lo, replicateh(-2), 0xE00); // shift by tbl_bits-23+15
    frac.hi = vextract_bits(mant.hi, replicateh(-2), 0xE00);

    chess_separator();

    y0 = vlookup_32hu(sqrt_tbl, idx);
    y1 = vlookup_32hu(sqrt_tbl + 32, idx);

    chess_separator();

    z  = dvblend(y0, y1, frac);
    z2 = dvmulh(z, replicateh(46341), 2); // RND_OPT=2: round 15 bits

    chess_separator();

    exp_odd.lo = exp.lo & replicateh(1);
    exp_odd.hi = exp.hi & replicateh(1);

    chess_separator();

    exp_dv2 = exp >> 1;
    z       = dvmux(exp_odd, z2, z);

    chess_separator();

    z.lo = vround(z.lo, replicateh(12) - exp_dv2.lo);
    z.hi = vround(z.hi, replicateh(12) - exp_dv2.hi);

    chess_separator();

    return z;
}

int distance_transform_stage2_sweep1(int tw, int th, int iw, int ih, int max_dist, unsigned short *horz_dist,
                                     unsigned short *v_arr, unsigned short *output_dist, unsigned short *output_label,
                                     unsigned short *sqrt_tbl, dvshortx *k_out)
{
    short *v_ptr  = (short *)v_arr;
    short *hd_ptr = (short *)horz_dist;
    int vecw      = pva_elementsof(dvshortx);

    int all_done = 0;
    dvshortx zero, one, thm1_v;
    dvshortx k, q, done, not_done, hd_q, valid_hd;
    dvshortx km1, km2, vkm1, vkm2, hd_vkm1, hd_vkm2;
    dvshortx n1, d1, n2, d2;
    dvintx n1_lo, n1_hi, n2_lo, n2_hi;
    dvshortx s_not_too_large, replacing_vk;
    dvshortx write_v_idx;
    dvshortx tmp0, tmp1, tmp2;
    dvshortx v_arr_update, q_done_update, kp1_update, km1_update;
    vcharx v_arr_update_v;
    int v_arr_update_s;
    dvintx d1_lo, d1_hi, d2_lo, d2_hi;
    dvshortx mask;

    zero.lo   = replicateh(0);
    zero.hi   = replicateh(0);
    one.lo    = replicateh(1);
    one.hi    = replicateh(1);
    thm1_v.lo = replicateh(th - 1);
    thm1_v.hi = replicateh(th - 1);
    mask.lo   = replicateh(0xFFF);
    mask.hi   = replicateh(0xFFF);

    chess_separator();

    k        = zero; // k = 0;
    q        = zero; // q = 0;
    done     = zero; // done = 0;
    not_done = one;
    vast_32h(v_ptr, zero, zero, -1); // v_arr[0] = 0;

    chess_separator();

    while (!all_done)
    {
        hd_q = vlookup_32h(hd_ptr, q);

        chess_separator();

        km1 = dvmux((k == 0), 0, (k - 1)); // km1 = (k == 0) ? 0 : k - 1;
        chess_separator();

        km2 = dvmux((k <= 1), 0, (k - 2)); // km2 = (k <= 1) ? 0 : k - 2;
        chess_separator();

        vkm1 = vlookup_32h(v_ptr, km1);
        vkm2 = vlookup_32h(v_ptr, km2);

        chess_separator();

        hd_q &= mask;

        valid_hd = hd_q < iw; // valid_hd = hd_q < iw;

        chess_separator();

        hd_vkm1 = vlookup_32h(hd_ptr, vkm1);
        hd_vkm2 = vlookup_32h(hd_ptr, vkm2);

        chess_separator();

        d1 = (q - vkm1) << 1; // d1 = 2 * (q - vkm1);

        chess_separator();

        d2 = (vkm1 - vkm2) << 1; // d2 = 2 * (vkm1 - vkm2);

        chess_separator();

        hd_vkm1 &= mask;
        hd_vkm2 &= mask;

        chess_separator();

        CALC_N(n1_lo, n1_hi, hd_q, hd_vkm1, q, vkm1); // n1 = hd_q * hd_q - hd_vkm1 * hd_vkm1 + q * q - vkm1 * vkm1;

        chess_separator();

        vdemote_i(n1_lo.lo, n1_lo.hi, n1.lo);
        vdemote_i(n1_hi.lo, n1_hi.hi, n1.hi); // (demote n1 from int to short)

        chess_separator();

        CALC_N(n2_lo, n2_hi, hd_vkm1, hd_vkm2, vkm1,
               vkm2); // n2 = hd_vkm1 * hd_vkm1 - hd_vkm2 * hd_vkm2 + vkm1 * vkm1 - vkm2 * vkm2;

        chess_separator();

        vdemote_i(n2_lo.lo, n2_lo.hi, n2.lo);
        vdemote_i(n2_hi.lo, n2_hi.hi, n2.hi); // (demote n2 from int to short)

        chess_separator();

        n2 = dvmux((k <= 1), 0, n2); // n2 = (k <= 1) ? 0 : n2;

        chess_separator();

        d2 = dvmux((k <= 1), 1, d2); // d2 = (k <= 1) ? 1 : d2;
        vpromote_di(d1.lo, d1_lo.lo, d1_lo.hi);
        vpromote_di(d1.hi, d1_hi.lo, d1_hi.hi);
        vpromote_di(d2.lo, d2_lo.lo, d2_lo.hi);
        vpromote_di(d2.hi, d2_hi.lo, d2_hi.hi);

        chess_separator();

        tmp0.lo = vmulh(d1.lo, thm1_v.lo, 0);
        tmp0.hi = vmulh(d1.hi, thm1_v.hi, 0);

        chess_separator();

        s_not_too_large = (n1 <= tmp0); // s_not_too_large = n1 <= d1 * (th - 1);

        chess_separator();

        dvintx tmp1_lo = dvmulwhw(n1_lo, d2_lo, 0);
        dvintx tmp1_hi = dvmulwhw(n1_hi, d2_hi, 0);
        // tmp1.lo = vmulh(n1.lo, d2.lo, 0);
        // tmp1.hi = vmulh(n1.hi, d2.hi, 0);

        chess_separator();

        dvintx tmp2_lo = dvmulwhw(n2_lo, d1_lo, 0);
        dvintx tmp2_hi = dvmulwhw(n2_hi, d1_hi, 0);
        // tmp2.lo = vmulh(n2.lo, d1.lo, 0);
        // tmp2.hi = vmulh(n2.hi, d1.hi, 0);

        chess_separator();

        dvintx replacing_vk_lo = (tmp1_lo <= tmp2_lo);
        dvintx replacing_vk_hi = (tmp1_hi <= tmp2_hi);

        vdemote_i(replacing_vk_lo.lo, replacing_vk_lo.hi, replacing_vk.lo);
        vdemote_i(replacing_vk_hi.lo, replacing_vk_hi.hi, replacing_vk.hi);
        // replacing_vk = (tmp1 <= tmp2);                         // replacing_vk = n1*d2 <= n2*d1;

        chess_separator();

        // state: v[?] = q
        dvshortx write_v_idx_and = dvandL((k == 1), replacing_vk);
        write_v_idx  = dvmux(write_v_idx_and, k - 1, k); // write_v_idx = (k == 1 && replacing_vk) ? (k - 1) : k;
        v_arr_update = dvorL((k == 0), s_not_too_large);

        chess_separator();

        v_arr_update.lo = vand3(valid_hd.lo, v_arr_update.lo, not_done.lo);
        v_arr_update.hi = vand3(valid_hd.hi, v_arr_update.hi, not_done.hi);

        chess_separator();

        vdemote(v_arr_update.lo, v_arr_update.hi, v_arr_update_v);

        chess_separator();

        // state: update q & done
        km1_update = (k > 1);

        chess_separator();

        v_arr_update_s = vbitcmp_s(v_arr_update_v, 1);

        km1_update.lo = vand3(valid_hd.lo, km1_update.lo, replacing_vk.lo);
        km1_update.hi = vand3(valid_hd.hi, km1_update.hi, replacing_vk.hi);

        chess_separator_scheduler();

        dvshortx q_done_update_not = dvnotL(km1_update);

        chess_separator_scheduler();

        q_done_update = dvandL(q_done_update_not, not_done); // if (!(valid_hd && k > 1 && replacing_vk))

        chess_separator_scheduler();

        done     = dvmux(q_done_update, (q >= thm1_v), done);
        all_done = vandr_s(done.lo & done.hi);
        vast_32h(v_ptr, write_v_idx, q,
                 v_arr_update_s); // if (valid_hd && (k == 0 || s_not_too_large)) v_arr[write_v_idx] = q;

        chess_separator();

        q += q_done_update;

        chess_separator_scheduler();

        // state: update k
        dvshortx not_replacing_vk = dvnotL(replacing_vk);
        dvshortx kp1_update_and   = dvandL(not_replacing_vk, s_not_too_large);
        kp1_update                = dvorL((k == 0), kp1_update_and);

        chess_separator();

        kp1_update.lo = vand3(valid_hd.lo, kp1_update.lo, not_done.lo);
        kp1_update.hi = vand3(valid_hd.hi, kp1_update.hi, not_done.hi);

        chess_separator();

        k += kp1_update;
        dvshortx km1_and = dvandL(km1_update, not_done);
        k -= km1_and;

        chess_separator();

        not_done = dvnotL(done);
    }
    *(k_out) = k;

    return 0;
}

int distance_transform_stage2_sweep2(int tw, int th, int iw, int ih, int max_dist, unsigned short *horz_dist,
                                     unsigned short *v_arr, unsigned short *output_dist, unsigned short *output_label,
                                     unsigned short *sqrt_tbl, dvshortx *k_in, dvshortx *k2_in, dvshortx *q_start,
                                     dvshortx *q_end)
{
    // second vertical scan, v[0..k-1] capture Voronoi regions
    short *v_ptr  = (short *)v_arr;
    short *hd_ptr = (short *)horz_dist;
    int vecw      = pva_elementsof(dvshortx);

    dvshortx k2;
    dvshortx vk2, hd_vk2, vk1, hd_vk1;
    dvshortx dist1, dist2, dist, label, label_vk;
    dvintx dist1_lo, dist1_hi, dist2_lo, dist2_hi;
    dvshortx update_k2, update_result;
    vcharx update_result_v;
    int update_result_s;
    int all_done = 0;
    dvshortx tmp0, tmp1, tmp2;
    dvshortx k, q, done, zero, thm1_v;
    dvshortx mask;

    zero.lo = replicateh(0);
    zero.hi = replicateh(0);
    mask.lo = replicateh(0xFFF);
    mask.hi = replicateh(0xFFF);

    k      = *(k_in);
    tmp0   = vlookup_32h(v_ptr, zero);
    thm1_v = *(q_end);
    vast_32h(v_ptr, k, tmp0, -1); // v_arr[k] = v_arr[0];
    k2 = *(k2_in);
    q  = *(q_start);
    chess_separator_scheduler();

    done     = zero;
    all_done = 0;
    vk2      = vlookup_32h(v_ptr, k2);      // vk2 = v_arr[k2];
    vk1      = vlookup_32h(v_ptr + tw, k2); // vk1 = v_arr[k2 + 1];
    chess_separator_scheduler();

    hd_vk2 = vlookup_32h(hd_ptr, vk2);
    hd_vk1 = vlookup_32h(hd_ptr, vk1);
    chess_separator_scheduler();

    while (!all_done)
    {
        hd_vk2 &= mask;
        hd_vk1 &= mask;

        chess_separator();

        dvshortx sum_lhs_2 = q - vk2;
        dvshortx sum_lhs_1 = q - vk1;

        dvintx dist1_lo = vsumsq(sum_lhs_2.lo, hd_vk2.lo);
        dvintx dist1_hi = vsumsq(sum_lhs_2.hi, hd_vk2.hi);
        dvintx dist2_lo = vsumsq(sum_lhs_1.lo, hd_vk1.lo);
        dvintx dist2_hi = vsumsq(sum_lhs_1.hi, hd_vk1.hi);

        dist1.lo = vdemote_i(dist1_lo);
        dist1.hi = vdemote_i(dist1_hi);
        dist2.lo = vdemote_i(dist2_lo);
        dist2.hi = vdemote_i(dist2_hi);

        chess_separator();

        tmp0          = (dist2 <= dist1);
        tmp1          = ((k2 + 1) < k);
        update_k2     = dvandL(tmp0, tmp1); // update_k2 = (d2 <= d1 && k2 + 1 < k);
        update_result = dvandL(dvnotL(update_k2), dvnotL(done));

        // update k2
        k2 += update_k2;
        vk2 = vlookup_32h(v_ptr, k2);      // vk2 = v_arr[k2];
        vk1 = vlookup_32h(v_ptr + tw, k2); // vk1 = v_arr[k2 + 1];
        chess_separator();

        // update all_done
        done = dvmux(update_result, (q >= thm1_v), done);
        vdemote(update_result.lo, update_result.hi, update_result_v);
        all_done = vandr_s(done.lo & done.hi);

        // update dist and label
        dist            = sqrt_dist(dist1, sqrt_tbl); // dist = sqrt_q3_sca(d1, sqrt_tbl);
        update_result_s = vbitcmp_s(update_result_v, 1);
        dist            = dvmux(dist1, dist, zero);

        hd_vk2 = vlookup_32hu((unsigned short *)hd_ptr, vk2);
        hd_vk1 = vlookup_32h(hd_ptr, vk1);
        chess_separator();

        dist = dvmux(dist >= ((iw + ih) << 3), max_dist, dist);
        chess_separator();

        dist = dvmin(dist, max_dist); // (dist >= 8 * (tw + th) || dist >= max_dist) ? max_dist : dist;
        chess_separator();

        label_vk = hd_vk2 >> 12;
        chess_separator();

        label = dvmux(dist >= max_dist, zero,
                      label_vk); // (dist >= max_dist) ? 0 : ((horz_dist[vk2 * tw + j] >> 12) & 0xF);

        chess_separator();

        vast_32h((short *)output_dist, q - *(q_start), dist, update_result_s);
        vast_32h((short *)output_label, q - *(q_start), label, update_result_s);

        // update q
        q += update_result;
    }
    *(k2_in) = k2;

    return 0;
}
