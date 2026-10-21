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

#include "hog_knl.h"

#include "hog_tdata.h"

#include <stdint.h>

/* WAR, repeat 32 times */
#undef _
#define _(V) V, V, V, V, V, V, V, V, V, V, V, V, V, V, V, V, V, V, V, V, V, V, V, V, V, V, V, V, V, V, V, V

/********************************************************************
 HOG split 360Deg into 18bins, idx calculated through 5 * oct + pre_bin
 In order to reflect bin number from quad to four quads:
 oct0~1: 9, 10, 11, 12, 13;
 oct2~3: 0, 17, 16, 15, 14;
 oct4~5: 0,  1,  2,  3,  4;
 oct6~7: 5,  6,  7,  8,  9.
 ********************************************************************/
alignas(64) const int16_t chess_storage(DMb % 64
                                        : chess_segment(C)) bin_reflect_tbl[40 * 32] = {
    _(9),  _(10), _(11), _(12), _(13), _(9),  _(10), _(11), _(12), _(13), _(0), _(17), _(16), _(15),
    _(14), _(0),  _(17), _(16), _(15), _(14), _(0),  _(1),  _(2),  _(3),  _(4), _(0),  _(1),  _(2),
    _(3),  _(4),  _(9),  _(8),  _(7),  _(6),  _(5),  _(9),  _(8),  _(7),  _(6), _(5),
};
#undef _

#define VPU_T4_WORD (0x5)

static uint8_t vperm[32] = {0, 4, 8, 12, 16, 20, 24, 28, 32, 32, 32, 32, 32, 32, 32, 32,
                            1, 5, 9, 13, 17, 21, 25, 29, 32, 32, 32, 32, 32, 32, 32, 32};

void memset_opt(void *mem, char data, int32_t size)
{
    dvcharx vdata;
    vdata.lo = replicateb(data);
    vdata.hi = replicateb(data);

    int32_t vecw  = chess_elementsof(dvcharx);
    int32_t niter = size / vecw;
    dvchar *ptr   = (dvchar *)mem;

    for (int32_t i = 0; i < niter; i++)
    {
        *ptr++ = extract(vdata);
    }
}

void toRgbAgenInitU8(uint8_t *in_y, uint8_t *in_uv, uint8_t *out_r, uint8_t *out_g, uint8_t *out_b, const int32_t tw,
                     const int32_t th, const int32_t lp, TORgbU8Param *param)
{
    //BWU_PDI load
    const int32_t vecw   = pva_elementsof(dvintx);
    const int32_t niter1 = (tw + vecw - 1) / vecw;
    const int32_t niter2 = th;
    const int32_t niter3 = (niter1 + 1) >> 1;
    const int32_t niter4 = (niter2 + 1) >> 1;

    AgenWrapper w0, w1, w2, w3, w4;

    agen agen_ld_y = init(0);
    w0.size        = sizeof(uint8_t);
    w0.n1          = 2;
    w0.s1          = vecw;
    w0.n2          = 2;
    w0.s2          = tw;
    w0.n3          = niter3;
    w0.s3          = vecw * 2;
    w0.n4          = niter4;
    w0.s4          = tw * 2;
    INIT_AGEN4(agen_ld_y, w0);
    agen_ld_y = update_agen_cb_start(agen_ld_y, (intptr_t)(&in_y[0]));
    agen_ld_y = update_agen_cb_size(agen_ld_y, tw * th * 2 * sizeof(uint8_t));

    // uv interleaved with x2 downsampled
    agen agen_ld_uv = init(0);
    w1.size         = sizeof(uint8_t);
    w1.n1           = 2;
    w1.s1           = vecw;
    w1.n2           = niter3;
    w1.s2           = vecw * 2;
    w1.n3           = niter4;
    w1.s3           = tw;
    INIT_AGEN3(agen_ld_uv, w1);
    agen_ld_uv = update_agen_cb_start(agen_ld_uv, (intptr_t)(&in_uv[0]));
    agen_ld_uv = update_agen_cb_size(agen_ld_uv, tw * th * sizeof(uint8_t));

    agen agen_st_r = init(0);
    w2.size        = sizeof(uint8_t);
    w2.n1          = 2;
    w2.s1          = lp;
    w2.n2          = niter3;
    w2.s2          = vecw * 2; //HB_P
    w2.n3          = niter4;
    w2.s3          = lp * 2;
    INIT_AGEN3(agen_st_r, w2);
    agen_st_r.sat_opt    = UNSIGNED_SATURATION;
    agen_st_r.sat_lim_lo = 0;
    agen_st_r.sat_val_lo = 0;
    agen_st_r.sat_lim_hi = 255;
    agen_st_r.sat_val_hi = 255;

    agen agen_st_g = init(0);
    w3.size        = sizeof(uint8_t);
    w3.n1          = 2;
    w3.s1          = lp;
    w3.n2          = niter3;
    w3.s2          = vecw * 2; //HB_P
    w3.n3          = niter4;
    w3.s3          = lp * 2;
    INIT_AGEN3(agen_st_g, w3);
    agen_st_g.sat_opt    = UNSIGNED_SATURATION;
    agen_st_g.sat_lim_lo = 0;
    agen_st_g.sat_val_lo = 0;
    agen_st_g.sat_lim_hi = 255;
    agen_st_g.sat_val_hi = 255;

    agen agen_st_b = init(0);
    w4.size        = sizeof(uint8_t);
    w4.n1          = 2;
    w4.s1          = lp;
    w4.n2          = niter3;
    w4.s2          = vecw * 2; //HB_P
    w4.n3          = niter4;
    w4.s3          = lp * 2;
    INIT_AGEN3(agen_st_b, w4);
    agen_st_b.sat_opt    = UNSIGNED_SATURATION;
    agen_st_b.sat_lim_lo = 0;
    agen_st_b.sat_val_lo = 0;
    agen_st_b.sat_lim_hi = 255;
    agen_st_b.sat_val_hi = 255;

    param->cfgs[0] = extract_agen_cfg(agen_ld_y);
    param->cfgs[1] = extract_agen_cfg(agen_ld_uv);
    param->cfgs[2] = extract_agen_cfg(agen_st_r);
    param->cfgs[3] = extract_agen_cfg(agen_st_g);
    param->cfgs[4] = extract_agen_cfg(agen_st_b);

    param->niter = niter3 * niter4;
}

// Round (x*255 + 0.5) and convert fp32 RGB values into halfword integers.
static inline void roundFpToIntHb(dvfloatx &v0, dvfloatx &v1, dvfloatx &v2, dvfloatx &v3, dvshortx &v4, dvshortx &v5,
                                  dvfloatx &dvHalf)
{
    v0 = dvmaddf(v0, 255.0f, dvHalf);
    v1 = dvmaddf(v1, 255.0f, dvHalf);
    v2 = dvmaddf(v2, 255.0f, dvHalf);
    v3 = dvmaddf(v3, 255.0f, dvHalf);

    dvintx t0 = dvfp_dvint_trunc(v0);
    dvintx t1 = dvfp_dvint_trunc(v1);
    dvintx t2 = dvfp_dvint_trunc(v2);
    dvintx t3 = dvfp_dvint_trunc(v3);

    v4.lo = vdemote_i(t0);
    v4.hi = vdemote_i(t1);
    v5.lo = vdemote_i(t2);
    v5.hi = vdemote_i(t3);
}

static inline void accumulateHist8Way(int16_t *histBase, const dvshortx &ofstIn, const dvshortx &wIn,
                                      const dvshortx &invalidX, const int32_t invalidY, const vshortx &vHistPat)
{
    // boundary check
    dvshortx w = dvmux(invalidX || invalidY, 0, wIn);

    // accumulate lower slice
    vhist_simple_8h(histBase, ofstIn.lo, w.lo);
    vhist_simple_8h(histBase, ofstIn.hi, w.hi);

    // accumulate higher slice
    vshortx w_lo    = vpermute(w.lo, vHistPat);
    vshortx ofst_lo = vpermute(ofstIn.lo, vHistPat);
    vshortx w_hi    = vpermute(w.hi, vHistPat);
    vshortx ofst_hi = vpermute(ofstIn.hi, vHistPat);
    vhist_simple_8h(histBase, ofst_lo, w_lo);
    vhist_simple_8h(histBase, ofst_hi, w_hi);
}

void toRgbExec(TORgbU8Param *param)
{
    const float Coffset = 128.0f;

    agen_A agen_ld_y  = init_agen_A_from_cfg(param->cfgs[0]);
    agen_B agen_ld_uv = init_agen_B_from_cfg(param->cfgs[1]);
    agen_A agen_st_r  = init_agen_A_from_cfg(param->cfgs[2]);
    agen_B agen_st_g  = init_agen_B_from_cfg(param->cfgs[3]);
    agen_C agen_st_b  = init_agen_C_from_cfg(param->cfgs[4]);

    const int32_t niter = param->niter;

    dvfloatx dvHalf;
    dvHalf.lo = replicatef(0.5f);
    dvHalf.hi = replicatef(0.5f);

    for (int32_t i = 0; i < niter; i++) chess_loop_range(12, )
    chess_prepare_for_pipelining
    {
        dvintx dvy00 = vuchar_dvintx_load_di(agen_ld_y);
        dvintx dvy01 = vuchar_dvintx_load_di(agen_ld_y);
        dvintx dvy10 = vuchar_dvintx_load_di(agen_ld_y);
        dvintx dvy11 = vuchar_dvintx_load_di(agen_ld_y);
        dvintx dvuv0 = vuchar_dvintx_load_di(agen_ld_uv);
        dvintx dvuv1 = vuchar_dvintx_load_di(agen_ld_uv);

        // y0, y2, ..., y14 | y1, y3, ..., y15
        dvfloatx dvfy00 = dvintx_dvfp(dvy00);
        // y16, y18, ..., y30 | y17, y19, ..., y31
        dvfloatx dvfy01 = dvintx_dvfp(dvy01);
        dvfloatx dvfy10 = dvintx_dvfp(dvy10);
        dvfloatx dvfy11 = dvintx_dvfp(dvy11);
        // u0, u2, ..., u14 | v0, v2, ..., v14
        dvfloatx dvfuv0 = dvintx_dvfp(dvuv0);
        // u16, u18, ..., u30 | v16, v18, ..., v31
        dvfloatx dvfuv1 = dvintx_dvfp(dvuv1);

        // compute Ey
        dvfloatx dvEy00 = dvfy00 * INV_SCALE;
        dvfloatx dvEy01 = dvfy01 * INV_SCALE;
        dvfloatx dvEy10 = dvfy10 * INV_SCALE;
        dvfloatx dvEy11 = dvfy11 * INV_SCALE;

        dvfuv0 = dvfuv0 - Coffset;
        dvfuv1 = dvfuv1 - Coffset;

        // compute Er, interleaved format
        dvfloatx dvEr00, dvEr01, dvEr10, dvEr11;
        dvEr00.lo = vmaddf(dvfuv0.hi, CRMAX_FACTOR, dvEy00.lo);
        dvEr00.hi = vmaddf(dvfuv0.hi, CRMAX_FACTOR, dvEy00.hi);
        dvEr01.lo = vmaddf(dvfuv1.hi, CRMAX_FACTOR, dvEy01.lo);
        dvEr01.hi = vmaddf(dvfuv1.hi, CRMAX_FACTOR, dvEy01.hi);
        dvEr10.lo = vmaddf(dvfuv0.hi, CRMAX_FACTOR, dvEy10.lo);
        dvEr10.hi = vmaddf(dvfuv0.hi, CRMAX_FACTOR, dvEy10.hi);
        dvEr11.lo = vmaddf(dvfuv1.hi, CRMAX_FACTOR, dvEy11.lo);
        dvEr11.hi = vmaddf(dvfuv1.hi, CRMAX_FACTOR, dvEy11.hi);

        // compute Eb, interleaved format
        dvfloatx dvEb00, dvEb01, dvEb10, dvEb11;
        dvEb00.lo = vmaddf(dvfuv0.lo, CBMAX_FACTOR, dvEy00.lo);
        dvEb00.hi = vmaddf(dvfuv0.lo, CBMAX_FACTOR, dvEy00.hi);
        dvEb01.lo = vmaddf(dvfuv1.lo, CBMAX_FACTOR, dvEy01.lo);
        dvEb01.hi = vmaddf(dvfuv1.lo, CBMAX_FACTOR, dvEy01.hi);
        dvEb10.lo = vmaddf(dvfuv0.lo, CBMAX_FACTOR, dvEy10.lo);
        dvEb10.hi = vmaddf(dvfuv0.lo, CBMAX_FACTOR, dvEy10.hi);
        dvEb11.lo = vmaddf(dvfuv1.lo, CBMAX_FACTOR, dvEy11.lo);
        dvEb11.hi = vmaddf(dvfuv1.lo, CBMAX_FACTOR, dvEy11.hi);

        // compute Eg: Ey * (1.0f / Wg) - (Wr / Wg) * Er - (Wb / Wg) * Eb
        dvfloatx dvEyScale00, dvEyScale01, dvEyScale10, dvEyScale11;
        dvEyScale00 = dvEy00 * INV_WG;
        dvEyScale01 = dvEy01 * INV_WG;
        dvEyScale10 = dvEy10 * INV_WG;
        dvEyScale11 = dvEy11 * INV_WG;
        dvfloatx dvEg00, dvEg01, dvEg10, dvEg11;
        dvEg00 = dvmsubf(dvEr00, WR_OVER_WG, dvEyScale00);
        dvEg01 = dvmsubf(dvEr01, WR_OVER_WG, dvEyScale01);
        dvEg10 = dvmsubf(dvEr10, WR_OVER_WG, dvEyScale10);
        dvEg11 = dvmsubf(dvEr11, WR_OVER_WG, dvEyScale11);
        dvEg00 = dvmsubf(dvEb00, WB_OVER_WG, dvEg00);
        dvEg01 = dvmsubf(dvEb01, WB_OVER_WG, dvEg01);
        dvEg10 = dvmsubf(dvEb10, WB_OVER_WG, dvEg10);
        dvEg11 = dvmsubf(dvEb11, WB_OVER_WG, dvEg11);

        // Round (E*255 + 0.5) and pack to halfword lanes; saturate to [0,255] by store agens.
        dvshortx dvR0, dvR1, dvG0, dvG1, dvB0, dvB1;
        roundFpToIntHb(dvEr00, dvEr01, dvEr10, dvEr11, dvR0, dvR1, dvHalf);
        roundFpToIntHb(dvEg00, dvEg01, dvEg10, dvEg11, dvG0, dvG1, dvHalf);
        roundFpToIntHb(dvEb00, dvEb01, dvEb10, dvEb11, dvB0, dvB1, dvHalf);

        // HB_P to store outputs to u8
        vstore_hb(dvR0, agen_st_r);
        vstore_hb(dvR1, agen_st_r);
        vstore_hb(dvG0, agen_st_g);
        vstore_hb(dvG1, agen_st_g);
        vstore_hb(dvB0, agen_st_b);
        vstore_hb(dvB1, agen_st_b);
    }
}

/*****************************************************************************************
  * We need to produce bpe paddings for each image patch when calculating the gradients
  * pad_l/pad_r with transp store, pad_t/pad_b with normal store
  * |-----|------------------------------|-----|
  * |     |                              |     |
  * |     |           pad_t              |     |
  * |     |------------------------------|     |
  * |     |                              |     |
  * |     |                              |     |
  * |pad_l|                              |pad_r|
  * |     |                              |     |
  * |     |                              |     |
  * |     |                              |     |
  * |     |------------------------------|     |
  * |     |           pad_b              |     |
  * |     |                              |     |
  * |-----|------------------------------|-----|
 *****************************************************************************************/
void produceBPEAgenInit(const int32_t padding_x, const int32_t padding_y, const int32_t tw, const int32_t th,
                        const int32_t lp, BpeParam *param)
{
    //BWU_PDI load
    const int32_t vecw   = pva_elementsof(vcharx);
    const int32_t niter1 = (tw + vecw - 1) / vecw;

    AgenWrapper w0, w1, w2, w3;

    // for top/bottom reading
    agen agen_ld_tb = init(0);
    w0.size         = sizeof(uint8_t);
    w0.n1           = niter1;
    w0.s1           = vecw;
    w0.n2           = 2;
    w0.s2           = (th - 1) * lp;
    INIT_AGEN2(agen_ld_tb, w0);

    // for top/bottom padding store
    agen agen_st_tb = init(0);
    w1.size         = sizeof(uint8_t);
    w1.n1           = padding_y;
    w1.s1           = -lp;
    w1.n2           = niter1;
    w1.s2           = vecw;
    w1.n3           = 2;
    w1.s3           = (th + padding_y) * lp;
    INIT_AGEN3(agen_st_tb, w1);

    // for left/right padding
    const int32_t lane_ofst = lp >> 6;
    const int32_t niter2    = (th + 2 * padding_y + vecw - 1) / vecw;

    // we would predicate the uncessary data at vpu kernels
    agen agen_ld_lr = init(0);
    w2.size         = sizeof(uint8_t);
    w2.n1           = niter2;
    w2.s1           = vecw * lp;
    w2.n2           = 2;
    w2.s2           = tw - 1;
    INIT_AGEN2(agen_ld_lr, w2);
    agen_ld_lr.lane_ofst = lane_ofst;

    agen agen_st_lr = init(0);
    w3.size         = sizeof(uint8_t);
    w3.n1           = padding_x;
    w3.s1           = 1;
    w3.n2           = niter2;
    w3.s2           = vecw * lp;
    w3.n3           = 2;
    w3.s3           = tw + padding_x;
    INIT_AGEN3(agen_st_lr, w3);
    agen_st_lr.lane_ofst = lane_ofst;

    const int32_t rem_x  = tw % vecw;
    const int32_t rem_y  = (th + 2 * padding_y) % vecw;
    const int32_t pred_x = (rem_x == 0) ? -1 : (1 << rem_x) - 1;
    const int32_t pred_y = (rem_y == 0) ? -1 : (1 << rem_y) - 1;

    param->cfgs[0] = extract_agen_cfg(agen_ld_tb);
    param->cfgs[1] = extract_agen_cfg(agen_st_tb);
    param->cfgs[2] = extract_agen_cfg(agen_ld_lr);
    param->cfgs[3] = extract_agen_cfg(agen_st_lr);

    param->niter1 = niter1;
    param->niter2 = niter2;
    param->pred_x = pred_x;
    param->pred_y = pred_y;
}

void produceBPEExec(BpeParam *param)
{
    // Now doing BPE in place to save VMEM footprint
    agen_A agen_ld_tb = init_agen_A_from_cfg(param->cfgs[0]);
    agen_A agen_st_tb = init_agen_A_from_cfg(param->cfgs[1]);

    // do the BPE padding at top/bottom
    int32_t niter_tb      = 2 * param->niter1; // fuse top/bottom in niter
    const int32_t niter_t = param->niter1;
    const int32_t pred_x  = param->pred_x;
    for (int32_t i = 0; i < niter_tb; i++) chess_loop_range(2, )
    chess_prepare_for_pipelining
    {
        const int32_t pred = ((i == niter_t - 1) || i == (niter_tb - 1)) ? pred_x : -1;
        vcharx v0          = vuchar_load(agen_ld_tb);
        for (int32_t j = 0; j < CELL_SIZE; j++) chess_unroll_loop(*)
        {
            vstore(v0, agen_st_tb, pred);
        }
    }

    // do the BPE padding at left/right
    agen_A agen_ld_lr     = init_agen_A_from_cfg(param->cfgs[2]);
    agen_A agen_st_lr     = init_agen_A_from_cfg(param->cfgs[3]);
    int32_t niter_lr      = 2 * param->niter2; // fuse top/bottom in niter
    const int32_t niter_l = param->niter2;
    const int32_t pred_y  = param->pred_y;

    for (int32_t i = 0; i < niter_lr; i++) chess_loop_range(2, )
    chess_prepare_for_pipelining
    {
        const int32_t pred = ((i == niter_l - 1) || i == (niter_lr - 1)) ? pred_y : -1;
        vcharx v0          = vuchar_load_transp(agen_ld_lr);
        for (int32_t j = 0; j < CELL_SIZE; j++) chess_unroll_loop(*)
        {
            vstore_transp(v0, agen_st_lr, pred);
        }
    }
}

// split hist to 2 parts and feed into different superbanks
void calcMagBinAgenInit(const int32_t twPad, const int32_t thPad, const int32_t lp, uint16_t *mag, uint16_t *bin,
                        const int32_t lp_out, CalcMagBinParam *param)
{
    const int32_t vecw = pva_elementsof(vcharx);
    // skip the first and last pixel of a row
    const int32_t niter1 = ((twPad - 2) + vecw - 1) / vecw;
    // skip the first and last row when calculating gradients
    const int32_t niter2 = thPad - 2;

    AgenWrapper w0, w1, w2, w3, w4, w5, w6, w7;

    // horizontal gradient calc, start from second line
    agen agen_ld_hor_r = init(0);
    w0.size            = sizeof(uint8_t);
    w0.n1              = 2;
    w0.s1              = 2;
    w0.n2              = niter1;
    w0.s2              = vecw;
    w0.n3              = niter2;
    w0.s3              = lp;
    INIT_AGEN3(agen_ld_hor_r, w0);
    agen agen_ld_hor_g = init(0);
    w1.size            = sizeof(uint8_t);
    w1.n1              = 2;
    w1.s1              = 2;
    w1.n2              = niter1;
    w1.s2              = vecw;
    w1.n3              = niter2;
    w1.s3              = lp;
    INIT_AGEN3(agen_ld_hor_g, w1);
    agen agen_ld_hor_b = init(0);
    w2.size            = sizeof(uint8_t);
    w2.n1              = 2;
    w2.s1              = 2;
    w2.n2              = niter1;
    w2.s2              = vecw;
    w2.n3              = niter2;
    w2.s3              = lp;
    INIT_AGEN3(agen_ld_hor_b, w2);

    // vertical gradient calc
    agen agen_ld_ver_r = init(0);
    w3.size            = sizeof(uint8_t);
    w3.n1              = 2;
    w3.s1              = 2 * lp;
    w3.n2              = niter1;
    w3.s2              = vecw;
    w3.n3              = niter2;
    w3.s3              = lp;
    INIT_AGEN3(agen_ld_ver_r, w3);
    agen agen_ld_ver_g = init(0);
    w4.size            = sizeof(uint8_t);
    w4.n1              = 2;
    w4.s1              = 2 * lp;
    w4.n2              = niter1;
    w4.s2              = vecw;
    w4.n3              = niter2;
    w4.s3              = lp;
    INIT_AGEN3(agen_ld_ver_g, w4);
    agen agen_ld_ver_b = init(0);
    w5.size            = sizeof(uint8_t);
    w5.n1              = 2;
    w5.s1              = 2 * lp;
    w5.n2              = niter1;
    w5.s2              = vecw;
    w5.n3              = niter2;
    w5.s3              = lp;
    INIT_AGEN3(agen_ld_ver_b, w5);

    agen agen_st_mag = init((dvshort *)&mag[0]);
    w6.size          = sizeof(uint16_t);
    w6.n1            = niter1;
    w6.s1            = vecw;
    w6.n2            = niter2;
    w6.s2            = lp_out;
    INIT_AGEN2(agen_st_mag, w6);
    agen agen_st_bin = init((dvshort *)&bin[0]);
    w7.size          = sizeof(uint16_t);
    w7.n1            = niter1;
    w7.s1            = vecw;
    w7.n2            = niter2;
    w7.s2            = lp_out;
    INIT_AGEN2(agen_st_bin, w7);

    param->cfgs[0] = extract_agen_cfg(agen_ld_hor_r);
    param->cfgs[1] = extract_agen_cfg(agen_ld_hor_g);
    param->cfgs[2] = extract_agen_cfg(agen_ld_hor_b);
    param->cfgs[3] = extract_agen_cfg(agen_ld_ver_r);
    param->cfgs[4] = extract_agen_cfg(agen_ld_ver_g);
    param->cfgs[5] = extract_agen_cfg(agen_ld_ver_b);
    param->cfgs[6] = extract_agen_cfg(agen_st_mag);
    param->cfgs[7] = extract_agen_cfg(agen_st_bin);
    param->niter   = niter1 * niter2;
}

void calcMagBinExec(CalcMagBinParam *param)
{
    // here it's VPU compute bound, due to VMEM shortage, we put two channels(r/b) into one superbank
    agen_A agen_ld_hor_r = init_agen_A_from_cfg(param->cfgs[0]);
    agen_B agen_ld_hor_g = init_agen_B_from_cfg(param->cfgs[1]);
    agen_A agen_ld_hor_b = init_agen_A_from_cfg(param->cfgs[2]);
    agen_A agen_ld_ver_r = init_agen_A_from_cfg(param->cfgs[3]);
    agen_B agen_ld_ver_g = init_agen_B_from_cfg(param->cfgs[4]);
    agen_A agen_ld_ver_b = init_agen_A_from_cfg(param->cfgs[5]);
    agen_C agen_st_mag   = init_agen_C_from_cfg(param->cfgs[6]);
    agen_C agen_st_bin   = init_agen_C_from_cfg(param->cfgs[7]);

    // have to splist to 2 parts into different superbanks
    const int32_t vecw  = pva_elementsof(dvshortx);
    const int32_t niter = param->niter;

    for (int32_t i = 0; i < niter; i++) chess_prepare_for_pipelining
    {
        // calc horizontal diff
        vcharx horR0 = vuchar_load(agen_ld_hor_r);
        vcharx horR1 = vuchar_load(agen_ld_hor_r);
        vcharx horG0 = vuchar_load(agen_ld_hor_g);
        vcharx horG1 = vuchar_load(agen_ld_hor_g);
        vcharx horB0 = vuchar_load(agen_ld_hor_b);
        vcharx horB1 = vuchar_load(agen_ld_hor_b);
        // 1, 2, 3, ..., 32
        vcharx horDiffR = horR1 - horR0;
        vcharx horDiffG = horG1 - horG0;
        vcharx horDiffB = horB1 - horB0;

        // calc vertical diff
        vcharx verR0 = vuchar_load(agen_ld_ver_r);
        vcharx verR1 = vuchar_load(agen_ld_ver_r);
        vcharx verG0 = vuchar_load(agen_ld_ver_g);
        vcharx verG1 = vuchar_load(agen_ld_ver_g);
        vcharx verB0 = vuchar_load(agen_ld_ver_b);
        vcharx verB1 = vuchar_load(agen_ld_ver_b);

        // 1, 2, 3, ..., 32
        vcharx verDiffR = verR1 - verR0;
        vcharx verDiffG = verG1 - verG0;
        vcharx verDiffB = verB1 - verB0;

        // compute dx*dx + dy*dy
        // 1, 3, ..., 31  | 2, 4, 6, ..., 32
        dvshortx sqR = vsumsq(horDiffR, verDiffR);
        dvshortx sqG = vsumsq(horDiffG, verDiffG);
        dvshortx sqB = vsumsq(horDiffB, verDiffB);

        // gradient promotion
        dvshortx dvHorDiffR, dvHorDiffG, dvHorDiffB;
        vpromote_di(horDiffR, dvHorDiffR.lo, dvHorDiffR.hi);
        vpromote_di(horDiffG, dvHorDiffG.lo, dvHorDiffG.hi);
        vpromote_di(horDiffB, dvHorDiffB.lo, dvHorDiffB.hi);
        dvshortx dvVerDiffR, dvVerDiffG, dvVerDiffB;
        vpromote_di(verDiffR, dvVerDiffR.lo, dvVerDiffR.hi);
        vpromote_di(verDiffG, dvVerDiffG.lo, dvVerDiffG.hi);
        vpromote_di(verDiffB, dvVerDiffB.lo, dvVerDiffB.hi);

        // select the maximum gx/gy/magSq
        dvshortx gx, gy, magSq;
        dvshortx cmp = sqG > sqR;
        gx           = dvmux(cmp, dvHorDiffG, dvHorDiffR);
        gy           = dvmux(cmp, dvVerDiffG, dvVerDiffR);
        magSq        = dvmax(sqR, sqG);
        cmp          = sqB > magSq;
        gx           = dvmux(cmp, dvHorDiffB, gx);
        gy           = dvmux(cmp, dvVerDiffB, gy);
        magSq        = dvmax(magSq, sqB);

        // directly calculate bin idx through cmp and oct_detect
        dvshortx oct;
        oct.lo         = voct_detect(gy.lo, gx.lo);
        oct.hi         = voct_detect(gy.hi, gx.hi);
        dvshortx gxAbs = dvabsdif(gx, 0);
        dvshortx gyAbs = dvabsdif(gy, 0);

        dvfloatx fGx0, fGx1, fGy0, fGy1;
        fGx0 = dvintx_dvfp(vpromote_di(gxAbs.lo));
        fGx1 = dvintx_dvfp(vpromote_di(gxAbs.hi));
        fGy0 = dvintx_dvfp(vpromote_di(gyAbs.lo));
        fGy1 = dvintx_dvfp(vpromote_di(gyAbs.hi));

        dvfloatx dvCmp0 = dvmsubf(fGx0, TAN_10DEG, fGy0);
        dvfloatx dvCmp1 = dvmsubf(fGx1, TAN_10DEG, fGy1);
        dvfloatx dvCmp2 = dvmsubf(fGx0, TAN_30DEG, fGy0);
        dvfloatx dvCmp3 = dvmsubf(fGx1, TAN_30DEG, fGy1);
        dvfloatx dvCmp4 = dvmsubf(fGx0, TAN_50DEG, fGy0);
        dvfloatx dvCmp5 = dvmsubf(fGx1, TAN_50DEG, fGy1);
        dvfloatx dvCmp6 = dvmsubf(fGx0, TAN_70DEG, fGy0);
        dvfloatx dvCmp7 = dvmsubf(fGx1, TAN_70DEG, fGy1);
        dvintx dvCheck0, dvCheck1, dvCheck2, dvCheck3, dvCheck4, dvCheck5, dvCheck6, dvCheck7;
        dvCheck0 = as_dvintx(dvCmp0) > 0;
        dvCheck1 = as_dvintx(dvCmp1) > 0;
        dvCheck2 = as_dvintx(dvCmp2) > 0;
        dvCheck3 = as_dvintx(dvCmp3) > 0;
        dvCheck4 = as_dvintx(dvCmp4) > 0;
        dvCheck5 = as_dvintx(dvCmp5) > 0;
        dvCheck6 = as_dvintx(dvCmp6) > 0;
        dvCheck7 = as_dvintx(dvCmp7) > 0;

        dvintx bin0, bin1;
        bin0 = dvCheck0 + dvCheck2 + dvCheck4 + dvCheck6;
        bin1 = dvCheck1 + dvCheck3 + dvCheck5 + dvCheck7;

        dvshortx bin;
        bin.lo = vdemote_i(bin0);
        bin.hi = vdemote_i(bin1);

        dvshortx dvBinIdx = dvmaddh(oct, 5, bin, VPU_ROUND_0, -1);
        bin               = vlookup_32h(&bin_reflect_tbl[0], dvBinIdx);
        // note: PVA oct_detect treat pi/2 as 1, while acccording to the algo, it should be 2
        // PVA oct_detect treat (3/2)*pi as 6, no need to further check
        bin = dvmux((oct == 1) && (gx == 0), 14, bin);

        // calc sqrt from gradient magnitude square
        dvfloatx fmagSq0 = dvintx_dvfp(vpromote_di(magSq.lo));
        dvfloatx fmagSq1 = dvintx_dvfp(vpromote_di(magSq.hi));
        dvfloatx fmag0   = dvfsqrt(fmagSq0);
        dvfloatx fmag1   = dvfsqrt(fmagSq1);
        // Q5 for magnitude
        fmag0 = fmag0 * (float)(1 << WEIGHT_QBITS);
        fmag1 = fmag1 * (float)(1 << WEIGHT_QBITS);
        dvintx mag0, mag1;
        mag0 = dvfp_dvintx_trunc(fmag0);
        mag1 = dvfp_dvintx_trunc(fmag1);

        vstore_i(bin, agen_st_bin);
        vstore_i(mag0, mag1, agen_st_mag);
    }
}

void calcHistAgenInit(uint16_t *mag, uint16_t *bin, uint16_t *hist0, uint16_t *hist1, const int32_t tw,
                      const int32_t th, const int32_t lp, const int32_t oriNum, CalcHistParam *param)
{
    const int32_t vecw   = pva_elementsof(dvshortx);
    const int32_t niter1 = tw / vecw;
    const int32_t niter2 = th;

    AgenWrapper w0, w1;

    agen agen_ld_mag = init((dvshort *)&mag[0]);
    w0.size          = sizeof(uint16_t);
    w0.n1            = niter1;
    w0.s1            = vecw;
    w0.n2            = niter2;
    w0.s2            = lp;
    INIT_AGEN2(agen_ld_mag, w0);
    agen agen_ld_bin = init((dvshort *)&bin[0]);
    w1.size          = sizeof(uint16_t);
    w1.n1            = niter1;
    w1.s1            = vecw;
    w1.n2            = niter2;
    w1.s2            = lp;
    INIT_AGEN2(agen_ld_bin, w1);

    param->cfgs[0] = extract_agen_cfg(agen_ld_mag);
    param->cfgs[1] = extract_agen_cfg(agen_ld_bin);
    param->niter   = niter1 * niter2;
    param->niter1  = niter1;
    param->hist0   = hist0;
    param->hist1   = hist1;
    param->oriNum  = oriNum;
    param->histW   = oriNum * HIST_WIDTH;
}

void calcHistExec(CalcHistParam *param)
{
    // here it's VPU compute bound, due to VMEM shortage, we put two channels(r/b) into one superbank
    agen_C agen_ld_mag = init_agen_C_from_cfg(param->cfgs[0]);
    agen_C agen_ld_bin = init_agen_C_from_cfg(param->cfgs[1]);

    // have to split to 2 parts into different superbanks
    const int32_t vecw   = pva_elementsof(dvshortx);
    const int32_t niter  = param->niter;
    const int32_t oriNum = param->oriNum;
    uint16_t *hist0      = param->hist0;
    uint16_t *hist1      = param->hist1;
    const int32_t histW  = param->histW;

    // assert(((vecw / 2) % CELL_SIZE) == 0);

    dvshortx dvBase;
    // make dvBase = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
    //                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
    vshortx vZero = replicateh(0);
    dvBase.lo     = vcollate_idx(vZero) + 1;
    dvBase.hi     = vcollate_idx(vZero) + 17;

    // tansform into Q7 base from (x + 0.5f)/CELL_SIZE - 0.5f
    dvBase                      = dvshift(dvBase, CORDS_QBITS) + HALF_Q;
    dvBase                      = (dvBase >> DIV_CELL_SIZE_BITS) - HALF_Q;
    const int32_t idXUpdateStep = vecw * (1 << (CORDS_QBITS - DIV_CELL_SIZE_BITS));
    dvshortx ixpBase            = dvBase & INTEG_MASK_Q;
    dvshortx vx0                = dvBase - ixpBase;
    dvshortx oneQ7;
    oneQ7.lo     = replicateh(0x80);
    oneQ7.hi     = replicateh(0x80);
    dvshortx vx1 = oneQ7 - vx0;

    const int32_t niter1 = param->niter1;
    int32_t idX          = niter1 - 1;
    int32_t modX         = niter1 - 1;
    // assert(modX > 0);
    int32_t idY   = 0;
    int32_t predY = 0;

    // need to accumulate the output in 8 way, split the vector into two parts
    static uint16_t higherSlice[16] = {8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0};
    vshortx vHistPat                = zero_extend(*(vushort *)&higherSlice[0]);

    // when assigning weights for right/bottom cells, we need them in the allocated area
    const int32_t guard_rb = (HIST_WIDTH - 1) << CORDS_QBITS;
    // filter out the garbage
    const int32_t guard_lt = ((((RGB_WIDTH - 1) << CORDS_QBITS) + HALF_Q) >> DIV_CELL_SIZE_BITS) - HALF_Q;

    // hist part 1 will have tile row [0, 12], part two will have tile row [13, 25]
    const int32_t halfHistHeight   = HIST_HEIGHT >> 1;
    const int32_t halfHistHeightQ7 = halfHistHeight << CORDS_QBITS;
    const int32_t part1End         = (int32_t)((halfHistHeight + 0.5f) * CELL_SIZE - 0.5f);

    for (int32_t i = 0; i < niter; i++) chess_prepare_for_pipelining
    {
        dvshortx mag = dvushort_load(agen_ld_mag);
        dvshortx bin = dvushort_load(agen_ld_bin);

        idX         = mod_inc_pred_z(idX, modX, predY);
        dvshortx xp = dvBase + idX * idXUpdateStep;
        // start from second line
        idY = mux(predY, idY + 1, idY);
        // check whether it is the second half
        const int32_t part2Top    = idY > part1End;
        int16_t *histBaseTop      = part2Top ? (int16_t *)hist1 : (int16_t *)hist0;
        const int32_t part2Bottom = (idY + CELL_SIZE) > part1End;
        int16_t *histBaseBottom   = part2Bottom ? (int16_t *)hist1 : (int16_t *)hist0;

        // cell size is 4, Q7 for idx
        int32_t yp         = (((idY << CORDS_QBITS) + HALF_Q) >> DIV_CELL_SIZE_BITS) - HALF_Q;
        int32_t iyp        = yp & INTEG_MASK_Q;
        const dvshortx ixp = xp & INTEG_MASK_Q;
        int32_t vy0        = yp - iyp;
        int32_t vy1        = ONE_Q - vy0;

        const dvshortx invalidX_lt = (xp <= 0) || (xp >= guard_lt);
        const int32_t invalidY_lt  = (yp <= 0) || (yp >= guard_lt);
        const dvshortx invalidX_rb = xp >= guard_rb;
        const int32_t invalidY_rb  = yp >= guard_rb;
        int32_t iypTop             = part2Top ? (iyp - halfHistHeightQ7) : iyp;
        // next cell row
        int32_t iypBottom         = iyp + ONE_Q;
        iypBottom                 = part2Bottom ? (iypBottom - halfHistHeightQ7) : iypBottom;
        const int32_t ofstYTop    = (iypTop >> CORDS_QBITS) * histW;
        const int32_t ofstYBottom = (iypBottom >> CORDS_QBITS) * histW;
        dvshortx ofstX            = dvmaddh((ixp >> CORDS_QBITS), oriNum, bin, VPU_ROUND_0, -1);

        // first bin(left-top)
        dvshortx ofst0  = ofstX + ofstYTop;
        dvshortx w0Temp = dvmulh(mag, vy1, VPU_ROUND_7);
        dvshortx w0     = dvmulh(w0Temp, vx1, VPU_ROUND_7);
        accumulateHist8Way(histBaseTop, ofst0, w0, invalidX_lt, invalidY_lt, vHistPat);

        // second bin(right-top)
        dvshortx ofst1 = ofst0 + oriNum;
        dvshortx w1    = dvmulh(vx0, w0Temp, VPU_ROUND_7);
        accumulateHist8Way(histBaseTop, ofst1, w1, invalidX_rb, invalidY_lt, vHistPat);

        // third bin(left-bottom)
        dvshortx ofst2  = ofstX + ofstYBottom;
        dvshortx w2Temp = dvmulh(mag, vy0, VPU_ROUND_7);
        dvshortx w2     = dvmulh(w2Temp, vx1, VPU_ROUND_7);
        accumulateHist8Way(histBaseBottom, ofst2, w2, invalidX_lt, invalidY_rb, vHistPat);

        // fourth bin(right-bottom)
        dvshortx ofst3 = ofst2 + oriNum;
        dvshortx w3    = dvmulh(w2Temp, vx0, VPU_ROUND_7);
        accumulateHist8Way(histBaseBottom, ofst3, w3, invalidX_rb, invalidY_rb, vHistPat);
    }
}

void consolidateHistAgenInit(uint16_t *histSrc, uint32_t *histDst, const int32_t oriNum, const int32_t tw,
                             const int32_t th, const int32_t histSizePadded, ConsHistParam *param)
{
    AgenWrapper w0, w1, w2, w3, w4, w5;
    const int32_t vecw              = pva_elementsof(dvshortx);
    const int32_t histCntPerSubBank = 4;

    const int32_t iter1      = (tw * th) >> 1;
    const int32_t step1      = (2 * oriNum) / histCntPerSubBank * vecw;
    const int32_t halfOriNum = oriNum >> 1;
    const int32_t iter2Even  = halfOriNum & 0x1 ? (halfOriNum >> 1) + 1 : (halfOriNum >> 1);
    const int32_t iter2Odd   = halfOriNum >> 1;

    // p1 loader base offset relative to p0 in the packed 8-way layout.
    const int32_t oddPixelStart = (oriNum / histCntPerSubBank) * vecw;

    // consolidateHistExec() processes 2 spatial points together:
    //   p0 = 2*i, p1 = 2*i+1  (linear cell index in a tw*th grid).
    // Within each 4-bin group (b0..b3), it reduces 2 bins at a time:
    //   pass A: bins(0,1) using vpermEven/vpermOdd
    //   pass B: bins(2,3) using vpermOdd /vpermEven
    // The 4 load agens below match exactly that: (p0,p1) × (bins01,bins23).

    // Loaders for pass A (bins(0,1) within each 4-bin group):
    // - even0: p0 bins(0,1)
    // - even1: p1 bins(0,1）
    agen agen_ld_hist_even0 = init(&histSrc[0]);
    w0.size                 = sizeof(uint16_t);
    w0.n1                   = iter1;
    w0.s1                   = step1;
    w0.n2                   = iter2Even;
    w0.s2                   = vecw;
    INIT_AGEN2(agen_ld_hist_even0, w0);
    agen agen_ld_hist_even1 = init(&histSrc[oddPixelStart]);
    w1.size                 = sizeof(uint16_t);
    w1.n1                   = iter1;
    w1.s1                   = step1;
    w1.n2                   = iter2Even;
    w1.s2                   = vecw;
    INIT_AGEN2(agen_ld_hist_even1, w1);

    // Loaders for pass B (bins(2,3) within each 4-bin group):
    // - odd0: p0 bins(2,3)
    // - odd1: p1 bins(2,3)
    agen agen_ld_hist_odd0 = init(&histSrc[0]);
    w2.size                = sizeof(uint16_t);
    w2.n1                  = iter1;
    w2.s1                  = step1;
    w2.n2                  = iter2Odd;
    w2.s2                  = vecw;
    INIT_AGEN2(agen_ld_hist_odd0, w2);
    agen agen_ld_hist_odd1 = init(&histSrc[oddPixelStart + vecw]);
    w3.size                = sizeof(uint16_t);
    w3.n1                  = iter1;
    w3.s1                  = step1;
    w3.n2                  = iter2Odd;
    w3.s2                  = vecw;
    INIT_AGEN2(agen_ld_hist_odd1, w3);

    // Stores for bins(0,1) pass:
    // - n1=2, s1=histSizePadded: output 2 planes at a time (bin k and bin k+1) for the current 4-bin group
    // - n2=iter1, s2=2: walk spatial points in pairs (p0 then p1)
    // - n3=iter2Even, s3=4*histSizePadded: jump to the next 4-bin group (bin 0/1 -> 4/5 -> 8/9 -> ...)
    agen agen_st_hist_even = init(&histDst[0]);
    w4.size                = sizeof(uint32_t);
    w4.n1                  = 2;
    w4.s1                  = histSizePadded;
    w4.n2                  = iter1;
    w4.s2                  = 2;
    w4.n3                  = iter2Even;
    w4.s3                  = 4 * histSizePadded;
    INIT_AGEN3(agen_st_hist_even, w4);

    // Stores for bins(2,3) pass: start from plane "bin=2", then also jump by 4 planes per group
    // (bin 2/3 -> 6/7 -> 10/11 -> ...).
    agen agen_st_hist_odd = init(&histDst[2 * histSizePadded]);
    w5.size               = sizeof(uint32_t);
    w5.n1                 = 2;
    w5.s1                 = histSizePadded;
    w5.n2                 = iter1;
    w5.s2                 = 2;
    w5.n3                 = iter2Odd;
    w5.s3                 = 4 * histSizePadded;
    INIT_AGEN3(agen_st_hist_odd, w5);

    param->cfgs[0] = extract_agen_cfg(agen_ld_hist_even0);
    param->cfgs[1] = extract_agen_cfg(agen_ld_hist_even1);
    param->cfgs[2] = extract_agen_cfg(agen_ld_hist_odd0);
    param->cfgs[3] = extract_agen_cfg(agen_ld_hist_odd1);
    param->cfgs[4] = extract_agen_cfg(agen_st_hist_even);
    param->cfgs[5] = extract_agen_cfg(agen_st_hist_odd);
    param->niter1  = iter1 * iter2Even;
    param->niter2  = iter1 * iter2Odd;
}

void consolidateHistExec(ConsHistParam *param)
{
    vcharx vpermEven = zero_extend(*(vuchar *)&vperm[0]);
    vcharx vpermOdd  = vpermEven + 2;

    agen_B agen_ld_hist_even0 = init_agen_B_from_cfg(param->cfgs[0]);
    agen_B agen_ld_hist_even1 = init_agen_B_from_cfg(param->cfgs[1]);
    agen_A agen_ld_hist_odd0  = init_agen_A_from_cfg(param->cfgs[2]);
    agen_A agen_ld_hist_odd1  = init_agen_A_from_cfg(param->cfgs[3]);
    agen_C agen_st_hist_even  = init_agen_C_from_cfg(param->cfgs[4]);
    agen_C agen_st_hist_odd   = init_agen_C_from_cfg(param->cfgs[5]);

    const int32_t niter1  = param->niter1;
    const int32_t niter1P = (niter1 + 3) / 4 * 4;
    for (int32_t i = 0; i < niter1P; i++) chess_loop_range(16, )
    chess_unroll_loop(4)
    chess_prepare_for_pipelining
    {
        // p0b0 and p0b1
        dvshortx in0 = dvushort_load_perm(agen_ld_hist_even0, vpermEven);
        // p1b0 and p1b1
        dvshortx in1 = dvushort_load_perm(agen_ld_hist_even1, vpermOdd);

        int32_t p0b0 = vsumr_s(in0.lo);
        int32_t p0b1 = vsumr_s(in0.hi);
        int32_t p1b0 = vsumr_s(in1.lo);
        int32_t p1b1 = vsumr_s(in1.hi);

        // produce 2 channels together
        int_store(p0b0, p1b0, agen_st_hist_even);
        int_store(p0b1, p1b1, agen_st_hist_even);
    }

    const int32_t niter2  = param->niter2;
    const int32_t niter2P = (niter2 + 3) / 4 * 4;
    // consolidation part-1 from bankB
    for (int32_t i = 0; i < niter2P; i++) chess_loop_range(16, )
    chess_unroll_loop(4)
    chess_prepare_for_pipelining
    {
        // p0b2 and p0b3
        dvshortx in0 = dvushort_load_perm(agen_ld_hist_odd0, vpermOdd);
        // p1b2 and p1b3
        dvshortx in1 = dvushort_load_perm(agen_ld_hist_odd1, vpermEven);

        int32_t p0b2 = vsumr_s(in0.lo);
        int32_t p0b3 = vsumr_s(in0.hi);
        int32_t p1b2 = vsumr_s(in1.lo);
        int32_t p1b3 = vsumr_s(in1.hi);

        // produce 2 channels together
        int_store(p0b2, p1b2, agen_st_hist_odd);
        int_store(p0b3, p1b3, agen_st_hist_odd);
    }
}

void normAgenInit(uint32_t *hist, float *norm, const int32_t oriNum, const int32_t histSizePadded, NormParam *param)
{
    const int32_t vecw   = pva_elementsof(dvintx);
    const int32_t niter1 = 2;
    const int32_t niter2 = oriNum >> 1;

    // each plane will be multiple of 64 byte, and tails will be 0
    // assert(histSizePadded % vecw == 0);
    const int32_t niter3 = histSizePadded / vecw;

    AgenWrapper w0, w1;

    agen agen_ld_hist = init(&hist[0]);
    w0.size           = sizeof(uint32_t);
    w0.n1             = niter1;
    w0.s1             = niter2 * histSizePadded;
    w0.n2             = niter2;
    w0.s2             = histSizePadded;
    w0.n3             = niter3;
    w0.s3             = vecw;
    INIT_AGEN3(agen_ld_hist, w0);

    agen agen_st_norm = init((dvfloat *)&norm[0]);
    w1.size           = sizeof(float);
    w1.n1             = niter2;
    w1.s1             = 0;
    w1.n2             = niter3;
    w1.s2             = vecw;
    INIT_AGEN2(agen_st_norm, w1);

    param->cfgs[0] = extract_agen_cfg(agen_ld_hist);
    param->cfgs[1] = extract_agen_cfg(agen_st_norm);
    param->niter   = niter2 * niter3;
    param->oriNum  = oriNum;
}

void normExec(NormParam *param)
{
    agen_C agen_ld_hist = init_agen_C_from_cfg(param->cfgs[0]);
    agen_B agen_st_norm = init_agen_B_from_cfg(param->cfgs[1]);

    const int32_t niter  = param->niter;
    const int32_t oriNum = param->oriNum;

    const int32_t mod = (oriNum >> 1) - 1;
    int32_t cntAcc    = mod;
    int32_t cntSt     = 0;

    dvintx acc;

    for (int32_t i = 0; i < niter; i++) chess_loop_range(18, )
    chess_unroll_loop(3)
    chess_prepare_for_pipelining
    {
        // hist value saved in Q5
        dvintx hist0 = dvuint_load(agen_ld_hist);
        dvintx hist1 = dvuint_load(agen_ld_hist);

        dvintx sumHist = hist0 + hist1;
        sumHist        = dvmulw(sumHist, sumHist, VPU_T4_WORD);

        int32_t predAcc;
        cntAcc = mod_inc_pred_nz(cntAcc, mod, predAcc);
        acc    = dvmaddwhw(sumHist, 1, acc, 0, predAcc);

        // convert to fp32 to maintain accuracy
        dvfloatx fpAcc = dvintx_dvfp(acc);
        fpAcc          = fpAcc * DEQ_NORM_FACTOR;

        int32_t predSt;
        cntSt = mod_inc_pred_z(cntSt, mod, predSt);
        vstore(fpAcc, agen_st_norm, predSt);
    }
}

void genCoeffAgenInit(float *norm, float *coeff, const int32_t featureW, const int32_t featureH, const int32_t histW,
                      const int32_t coefSizePerPlane, CoeffParam *param)
{
    const int32_t vecw   = pva_elementsof(dvfloatx);
    const int32_t niter1 = (featureW + vecw - 1) / vecw;
    const int32_t niter2 = featureH;

    // assert(coefSizePerPlane % vecw == 0);

    AgenWrapper w0, w1;

    agen agen_ld_norm = init((dvfloat *)&norm[0]);
    w0.size           = sizeof(float);
    w0.n1             = 3;
    w0.s1             = 1;
    w0.n2             = 3;
    w0.s2             = histW;
    w0.n3             = niter1;
    w0.s3             = vecw;
    w0.n4             = niter2;
    w0.s4             = histW;
    INIT_AGEN4(agen_ld_norm, w0);

    agen agen_st_coeff = init((dvfloat *)&coeff[0]);
    w1.size            = sizeof(float);
    w1.n1              = 4;
    w1.s1              = coefSizePerPlane;
    w1.n2              = niter1;
    w1.s2              = vecw;
    w1.n3              = niter2;
    w1.s3              = featureW;
    INIT_AGEN3(agen_st_coeff, w1);

    param->cfgs[0] = extract_agen_cfg(agen_ld_norm);
    param->cfgs[1] = extract_agen_cfg(agen_st_coeff);
    param->niter   = niter1 * niter2;
}

void genCoeffExec(CoeffParam *param)
{
    const int32_t niter = param->niter;

    agen_B agen_ld_norm  = init_agen_B_from_cfg(param->cfgs[0]);
    agen_A agen_st_coeff = init_agen_A_from_cfg(param->cfgs[1]);

    dvfloatx dvEps;
    dvEps.lo = replicatef(EPS);
    dvEps.hi = replicatef(EPS);

    for (int32_t i = 0; i < niter; i++) chess_loop_range(6, )
    chess_prepare_for_pipelining
    {
        dvfloatx in00 = dvfloat_load(agen_ld_norm);
        dvfloatx in01 = dvfloat_load(agen_ld_norm);
        dvfloatx in02 = dvfloat_load(agen_ld_norm);
        dvfloatx in10 = dvfloat_load(agen_ld_norm);
        dvfloatx in11 = dvfloat_load(agen_ld_norm);
        dvfloatx in12 = dvfloat_load(agen_ld_norm);
        dvfloatx in20 = dvfloat_load(agen_ld_norm);
        dvfloatx in21 = dvfloat_load(agen_ld_norm);
        dvfloatx in22 = dvfloat_load(agen_ld_norm);

        dvfloatx n1Temp = in11 + in12 + dvEps;
        dvfloatx n1     = n1Temp + in21 + in22;
        dvfloatx n2     = n1Temp + in01 + in02;

        dvfloatx n3Temp = in10 + in11 + dvEps;
        dvfloatx n3     = n3Temp + in20 + in21;
        dvfloatx n4     = n3Temp + in00 + in01;

        // Note: avoid dvided by 0 or sqrt on negative numbers
        n1 = dvmux(as_dvintx(n1) <= 0, EPS, n1);
        n2 = dvmux(as_dvintx(n2) <= 0, EPS, n2);
        n3 = dvmux(as_dvintx(n3) <= 0, EPS, n3);
        n4 = dvmux(as_dvintx(n4) <= 0, EPS, n4);

        n1 = dvfsqrt(n1);
        n2 = dvfsqrt(n2);
        n3 = dvfsqrt(n3);
        n4 = dvfsqrt(n4);

        n1 = dvfrcp(n1);
        n2 = dvfrcp(n2);
        n3 = dvfrcp(n3);
        n4 = dvfrcp(n4);

        vstore(n1, agen_st_coeff);
        vstore(n2, agen_st_coeff);
        vstore(n3, agen_st_coeff);
        vstore(n4, agen_st_coeff);
    }
}

void genHogFeaturesAgenInit(uint32_t *hist, float *coeff, uint8_t *output, const int32_t oriNum,
                            const int32_t histSizePadded, const int32_t histW, const int32_t featureW,
                            const int32_t featureH, const int32_t coefSizePerPlane, HogOutParam *param)
{
    const int32_t vecw = pva_elementsof(dvfloatx);

    int32_t niter1 = (featureW + vecw - 1) / vecw;
    // demotion store to u8 relies on it
    niter1               = ((niter1 + 1) >> 1) << 1;
    const int32_t niter2 = featureH;
    const int32_t niter3 = oriNum;

    AgenWrapper w0, w1, w2;

    agen agen_ld_hist = init((dvintx *)&hist[histW + CELL_PADDING]);
    w0.size           = sizeof(uint32_t);
    w0.n1             = niter1;
    w0.s1             = vecw;
    w0.n2             = niter2;
    w0.s2             = histW;
    w0.n3             = niter3;
    w0.s3             = histSizePadded;
    INIT_AGEN3(agen_ld_hist, w0);

    agen agen_ld_coeff = init((dvfloat *)&coeff[0]);
    w1.size            = sizeof(float);
    w1.n1              = 4;
    w1.s1              = coefSizePerPlane;
    w1.n2              = niter1;
    w1.s2              = vecw;
    w1.n3              = niter2;
    w1.s3              = featureW;
    w1.n4              = niter3;
    w1.s4              = 0;
    INIT_AGEN4(agen_ld_coeff, w1);

    agen agen_st_out = init((vchar *)&output[0]);
    w2.size          = sizeof(uint8_t);
    // demotion store
    w2.n1 = niter1 >> 1;
    w2.s1 = 2 * vecw;
    w2.n2 = niter2;
    w2.s2 = featureW;
    w2.n3 = niter3;
    w2.s3 = featureW * featureH;
    INIT_AGEN3(agen_st_out, w2);
    agen_st_out.sat_opt    = UNSIGNED_SATURATION;
    agen_st_out.sat_lim_lo = 0;
    agen_st_out.sat_val_lo = 0;
    agen_st_out.sat_lim_hi = 255;
    agen_st_out.sat_val_hi = 255;

    param->cfgs[0] = extract_agen_cfg(agen_ld_hist);
    param->cfgs[1] = extract_agen_cfg(agen_ld_coeff);
    param->cfgs[2] = extract_agen_cfg(agen_st_out);
    param->niter   = (niter1 >> 1) * niter2 * niter3;
}

void genHogFeaturesExec(HogOutParam *param)
{
    const int32_t niter  = param->niter;
    agen_C agen_ld_hist  = init_agen_C_from_cfg(param->cfgs[0]);
    agen_A agen_ld_coeff = init_agen_A_from_cfg(param->cfgs[1]);
    agen_C agen_st_out   = init_agen_C_from_cfg(param->cfgs[2]);

    dvfloatx half;
    half.lo = replicatef(0.5f);
    half.hi = replicatef(0.5f);

    for (int32_t i = 0; i < niter; i++) chess_loop_range(6, )
    chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        dvintx hist0 = dvint_load(agen_ld_hist);
        dvfloatx n10 = dvfloat_load(agen_ld_coeff);
        dvfloatx n20 = dvfloat_load(agen_ld_coeff);
        dvfloatx n30 = dvfloat_load(agen_ld_coeff);
        dvfloatx n40 = dvfloat_load(agen_ld_coeff);

        dvintx hist1 = dvint_load(agen_ld_hist);
        dvfloatx n11 = dvfloat_load(agen_ld_coeff);
        dvfloatx n21 = dvfloat_load(agen_ld_coeff);
        dvfloatx n31 = dvfloat_load(agen_ld_coeff);
        dvfloatx n41 = dvfloat_load(agen_ld_coeff);

        dvfloatx fhist0 = dvintx_dvfp(hist0);
        dvfloatx fhist1 = dvintx_dvfp(hist1);
        // dequantize the Q7 hist value
        fhist0 = fhist0 * DEQ_HIST_FACTOR;
        fhist1 = fhist1 * DEQ_HIST_FACTOR;

        dvfloatx h10 = fhist0 * n10;
        h10          = dvmux(h10 > 0.2f, 0.2f, h10);
        dvfloatx h20 = fhist0 * n20;
        h20          = dvmux(h20 > 0.2f, 0.2f, h20);
        dvfloatx h30 = fhist0 * n30;
        h30          = dvmux(h30 > 0.2f, 0.2f, h30);
        dvfloatx h40 = fhist0 * n40;
        h40          = dvmux(h40 > 0.2f, 0.2f, h40);
        dvfloatx h0  = (h10 + h20 + h30 + h40) * 0.5f;

        dvfloatx h11 = fhist1 * n11;
        h11          = dvmux(h11 > 0.2f, 0.2f, h11);
        dvfloatx h21 = fhist1 * n21;
        h21          = dvmux(h21 > 0.2f, 0.2f, h21);
        dvfloatx h31 = fhist1 * n31;
        h31          = dvmux(h31 > 0.2f, 0.2f, h31);
        dvfloatx h41 = fhist1 * n41;
        h41          = dvmux(h41 > 0.2f, 0.2f, h41);
        dvfloatx h1  = (h11 + h21 + h31 + h41) * 0.5f;

        h0           = dvmaddf(h0, OUT_MAX_F, half);
        h1           = dvmaddf(h1, OUT_MAX_F, half);
        dvintx outH0 = dvfp_dvint_trunc(h0);
        dvintx outH1 = dvfp_dvint_trunc(h1);
        dvshortx out;
        out.lo = vdemote(outH0);
        out.hi = vdemote(outH1);

        vstore_hb(out, agen_st_out);
    }
}