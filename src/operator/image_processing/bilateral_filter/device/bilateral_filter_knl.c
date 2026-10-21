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

#include "bilateral_filter_knl.h"

void bilateral_filter_init(int32_t *lookupOffsets, uint32_t *value_sum, uint32_t *weight_sum, int32_t cb_line_pitch,
                           int32_t cb_size, uint8_t *cb_start_addr, int32_t lastTileMaskValid,
                           BilateralFilterConfig *cfg)
{
    int32_t niter1 = TILE_WIDTH / chess_elementsof(vuchar);
    int32_t niter2 = lastTileMaskValid ? lastTileMaskValid : TILE_HEIGHT;

    cfg->lookupOffsets = lookupOffsets;
    cfg->niter         = niter1 * niter2;

    AgenWrapper w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10;

    agen agen_ld_img_corner = init(0);
    w0.size                 = sizeof(uint8_t);
    w0.n1                   = 2;
    w0.n2                   = 2;
    w0.n3                   = 2;
    w0.n4                   = niter1 / 2;
    w0.n5                   = niter2;
    w0.s1                   = chess_elementsof(vuchar);
    w0.s2                   = RADIUS * cb_line_pitch - RADIUS;
    w0.s3                   = RADIUS * cb_line_pitch + RADIUS;
    w0.s4                   = chess_elementsof(vuchar) * 2;
    w0.s5                   = cb_line_pitch;
    INIT_AGEN5(agen_ld_img_corner, w0);
    agen_ld_img_corner     = update_agen_cb_start(agen_ld_img_corner, (intptr_t)cb_start_addr);
    agen_ld_img_corner     = update_agen_cb_size(agen_ld_img_corner, cb_size);
    cfg->cfg_ld_img_corner = extract_agen_cfg(agen_ld_img_corner);

    agen agen_ld_img_square = init(0);
    w1.size                 = sizeof(uint8_t);
    w1.n1                   = 2;
    w1.n2                   = KERNEL_SIZE - 2;
    w1.n3                   = KERNEL_SIZE - 2;
    w1.n4                   = niter1 / 2;
    w1.n5                   = niter2;
    w1.s1                   = chess_elementsof(vuchar);
    w1.s2                   = 1;
    w1.s3                   = cb_line_pitch;
    w1.s4                   = chess_elementsof(vuchar) * 2;
    w1.s5                   = cb_line_pitch;
    INIT_AGEN5(agen_ld_img_square, w1);
    agen_ld_img_square     = update_agen_cb_start(agen_ld_img_square, (intptr_t)cb_start_addr);
    agen_ld_img_square     = update_agen_cb_size(agen_ld_img_square, cb_size);
    cfg->cfg_ld_img_square = extract_agen_cfg(agen_ld_img_square);

    agen agen_ld_img_central = init(0);
    w2.size                  = sizeof(uint8_t);
    w2.n1                    = niter1;
    w2.n2                    = niter2;
    w2.s1                    = chess_elementsof(vuchar);
    w2.s2                    = cb_line_pitch;
    INIT_AGEN2(agen_ld_img_central, w2);
    agen_ld_img_central     = update_agen_cb_start(agen_ld_img_central, (intptr_t)cb_start_addr);
    agen_ld_img_central     = update_agen_cb_size(agen_ld_img_central, cb_size);
    cfg->cfg_ld_img_central = extract_agen_cfg(agen_ld_img_central);

    agen agen_st_value = init((dvuint *)value_sum);
    w3.size            = sizeof(uint32_t);
    w3.n1              = niter1 * 2;
    w3.n2              = niter2;
    w3.s1              = chess_elementsof(dvuint);
    w3.s2              = TILE_WIDTH;
    INIT_AGEN2(agen_st_value, w3);
    cfg->cfg_st_value = extract_agen_cfg(agen_st_value);

    agen agen_st_weight = init((dvuint *)weight_sum);
    w4.size             = sizeof(uint32_t);
    w4.n1               = niter1 * 2;
    w4.n2               = niter2;
    w4.s1               = chess_elementsof(dvuint);
    w4.s2               = TILE_WIDTH;
    INIT_AGEN2(agen_st_weight, w4);
    cfg->cfg_st_weight = extract_agen_cfg(agen_st_weight);

    agen agen_ld_value = init((dvuint *)value_sum);
    w5.size            = sizeof(uint32_t);
    w5.n1              = niter1 * 2;
    w5.n2              = niter2;
    w5.s1              = chess_elementsof(dvuint);
    w5.s2              = TILE_WIDTH;
    INIT_AGEN2(agen_ld_value, w5);
    cfg->cfg_ld_value = extract_agen_cfg(agen_ld_value);

    agen agen_ld_weight = init((dvuint *)weight_sum);
    w6.size             = sizeof(uint32_t);
    w6.n1               = niter1 * 2;
    w6.n2               = niter2;
    w6.s1               = chess_elementsof(dvuint);
    w6.s2               = TILE_WIDTH;
    INIT_AGEN2(agen_ld_weight, w6);
    cfg->cfg_ld_weight = extract_agen_cfg(agen_ld_weight);

    agen agen_st_img = init(0);
    w7.size          = sizeof(uint8_t);
    w7.n1            = niter1;
    w7.n2            = niter2;
    w7.s1            = chess_elementsof(vuchar);
    w7.s2            = TILE_WIDTH;
    INIT_AGEN2(agen_st_img, w7);
    cfg->cfg_st_img = extract_agen_cfg(agen_st_img);
}

void replicate_table_char_32(uint8_t *tbl_out, uint8_t *tbl_in, int32_t size)
{
    vcharx vdata1;
    int32_t vecw  = chess_elementsof(vdata1);
    int32_t niter = size >> 1;

    chess_separator_scheduler();

    AgenWrapper w5, w3;

    agen_B in1 = init_B((vuchar *)tbl_in);
    w5.size    = sizeof(uint8_t);
    w5.n1      = niter;
    w5.s1      = 2;
    INIT_AGEN1(in1, w5);

    agen_C out1 = init_C((vuchar *)tbl_out);
    w3.size     = sizeof(uint8_t);
    w3.n1       = niter;
    w3.s1       = vecw * 2;
    INIT_AGEN1(out1, w3);

    dvcharx dvdata1;

    for (int32_t i = 0; i < niter; i++) chess_loop_range(24, )
    chess_unroll_loop(8)
    chess_prepare_for_pipelining
    {
        vdata1 = vuchar_load_c2(in1);

        dvdata1.lo = vdata1;
        dvdata1.hi = vdata1;

        vstore(dvdata1, out1);
    }
}

#ifdef KSIZE_3X3
void bilateral_filter_accumulate_stage(uint8_t *weight_tbl, BilateralFilterConfig *cfg)
{
    int32_t niter              = cfg->niter / 2;
    agen_A agen_ld_img_corner  = init_agen_A_from_cfg(cfg->cfg_ld_img_corner);
    agen_A agen_ld_img_square  = init_agen_A_from_cfg(cfg->cfg_ld_img_square);
    agen_A agen_ld_img_central = init_agen_A_from_cfg(cfg->cfg_ld_img_central);
    agen_B agen_st_value       = init_agen_B_from_cfg(cfg->cfg_st_value);
    agen_C agen_st_weight      = init_agen_C_from_cfg(cfg->cfg_st_weight);

    dvcharx dvdata_circle, dvdata_central;
    dvcharx dvweight, dvdiff;
    dvshortx dvvalue_sum_lo, dvvalue_sum_hi, dvweight_sum_lo, dvweight_sum_hi;

    const uint8_t chess_storage(RAM_Cb) *weight_ptr = (uint8_t chess_storage(RAM_Cb) *)(weight_tbl);
    dvdata_central.lo = vuchar_load(agen_ld_img_central); // Pre-load and move to behind the loop
    dvdata_central.hi = vuchar_load(agen_ld_img_central); // Pre-load and move to behind the loop

    dvcharx dvweight_central;
    vcharx vdiff_central = replicateb(COLOR_8BIT);
    dvweight_central.lo  = vlookup_32bu(weight_ptr, vdiff_central);
    dvweight_central.hi  = dvweight_central.lo;

    for (int32_t iter = 0; iter < niter; iter++) chess_loop_range(128, )
    chess_prepare_for_pipelining
    {
        dvmaddbh(dvdata_central, dvweight_central, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, 0, dvvalue_sum_lo,
                 dvvalue_sum_hi);
        dvmaddbh(dvweight_central, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, 0, dvweight_sum_lo,
                 dvweight_sum_hi);

        for (int32_t i = 0; i < 4; i++) chess_unroll_loop(*)
        {
            dvdata_circle.lo = vuchar_load(agen_ld_img_corner);
            dvdata_circle.hi = vuchar_load(agen_ld_img_corner);
            dvdiff           = dvabsdif(dvdata_circle, dvdata_central);

            dvweight.lo = vlookup_32bu(weight_ptr, dvdiff.lo);
            dvweight.hi = vlookup_32bu(weight_ptr, dvdiff.hi);

            dvmaddbh(dvdata_circle, dvweight, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
                     dvvalue_sum_hi);
            dvmaddbh(dvweight, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);
        }

        dvdata_central.lo = vuchar_load(agen_ld_img_central); // Rotated loads manually
        dvdata_central.hi = vuchar_load(agen_ld_img_central); // Rotated loads manually

        vstore_hw(dvvalue_sum_lo.lo, agen_st_value);
        vstore_hw(dvvalue_sum_lo.hi, agen_st_value);
        vstore_hw(dvvalue_sum_hi.lo, agen_st_value);
        vstore_hw(dvvalue_sum_hi.hi, agen_st_value);

        vstore_hw(dvweight_sum_lo.lo, agen_st_weight);
        vstore_hw(dvweight_sum_lo.hi, agen_st_weight);
        vstore_hw(dvweight_sum_hi.lo, agen_st_weight);
        vstore_hw(dvweight_sum_hi.hi, agen_st_weight);
    }
}
#endif
#ifdef KSIZE_5X5
void bilateral_filter_accumulate_stage(uint8_t *weight_tbl, BilateralFilterConfig *cfg)
{
    int32_t niter              = cfg->niter / 2;
    agen_A agen_ld_img_corner  = init_agen_A_from_cfg(cfg->cfg_ld_img_corner);
    agen_A agen_ld_img_square  = init_agen_A_from_cfg(cfg->cfg_ld_img_square);
    agen_A agen_ld_img_central = init_agen_A_from_cfg(cfg->cfg_ld_img_central);
    agen_B agen_st_value       = init_agen_B_from_cfg(cfg->cfg_st_value);
    agen_C agen_st_weight      = init_agen_C_from_cfg(cfg->cfg_st_weight);

    dvcharx dvdata_circle, dvdata_central;
    dvcharx dvweight, dvdiff;
    dvshortx dvvalue_sum_lo, dvvalue_sum_hi, dvweight_sum_lo, dvweight_sum_hi;

    dvcharx dvdata_circle1, dvdata_circle2, dvdata_circle3, dvdata_circle4, dvdata_circle5, dvdata_circle6,
        dvdata_circle7, dvdata_circle8, dvdata_circle9, dvdata_circle10, dvdata_circle11, dvdata_circle12,
        dvdata_circle13;
    dvcharx dvweight1, dvweight2, dvweight3, dvweight4, dvweight5, dvweight6, dvweight7, dvweight8, dvweight9,
        dvweight10, dvweight11, dvweight12, dvweight13;
    dvcharx dvdiff1, dvdiff2, dvdiff3, dvdiff4, dvdiff5, dvdiff6, dvdiff7, dvdiff8, dvdiff9, dvdiff10, dvdiff11,
        dvdiff12, dvdiff13;

    dvcharx dvweight_bypass1, dvweight_bypass2_0, dvweight_bypass2_1;

    const uint8_t chess_storage(RAM_Cb) *weight_ptr = (uint8_t chess_storage(RAM_Cb) *)(weight_tbl);
    dvdata_central.lo = vuchar_load(agen_ld_img_central); // Pre-load and move to behind the loop
    dvdata_central.hi = vuchar_load(agen_ld_img_central); // Pre-load and move to behind the loop

    // Prolog 0
    dvdata_circle1.lo = vuchar_load(agen_ld_img_corner);
    dvdata_circle1.hi = vuchar_load(agen_ld_img_corner);
    dvdiff1           = dvabsdif(dvdata_circle1, dvdata_central);
    dvweight1.lo      = vlookup_32bu(weight_ptr, dvdiff1.lo);
    dvweight1.hi      = vlookup_32bu(weight_ptr, dvdiff1.hi);
    dvmaddbh(dvdata_circle1, dvweight1, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, 0, dvvalue_sum_lo, dvvalue_sum_hi);
    dvmaddbh(dvweight1, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, 0, dvweight_sum_lo, dvweight_sum_hi);

    dvdata_circle2.lo = vuchar_load(agen_ld_img_corner);
    dvdata_circle2.hi = vuchar_load(agen_ld_img_corner);
    dvdiff2           = dvabsdif(dvdata_circle2, dvdata_central);
    dvweight2.lo      = vlookup_32bu(weight_ptr, dvdiff2.lo);
    dvweight2.hi      = vlookup_32bu(weight_ptr, dvdiff2.hi);
    dvmaddbh(dvdata_circle2, dvweight2, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
             dvvalue_sum_hi);
    dvmaddbh(dvweight2, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

    dvdata_circle3.lo = vuchar_load(agen_ld_img_corner);
    dvdata_circle3.hi = vuchar_load(agen_ld_img_corner);
    dvdiff3           = dvabsdif(dvdata_circle3, dvdata_central);
    dvweight3.lo      = vlookup_32bu(weight_ptr, dvdiff3.lo);
    dvweight3.hi      = vlookup_32bu(weight_ptr, dvdiff3.hi);
    dvmaddbh(dvdata_circle3, dvweight3, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
             dvvalue_sum_hi);
    dvmaddbh(dvweight3, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

    dvdata_circle4.lo  = vuchar_load(agen_ld_img_corner);
    dvdata_circle4.hi  = vuchar_load(agen_ld_img_corner);
    dvdiff4            = dvabsdif(dvdata_circle4, dvdata_central);
    dvweight4.lo       = vlookup_32bu(weight_ptr, dvdiff4.lo);
    dvweight4.hi       = vlookup_32bu(weight_ptr, dvdiff4.hi);
    dvweight_bypass2_0 = dvweight4;
    dvmaddbh(dvdata_circle4, dvweight4, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
             dvvalue_sum_hi);
    dvmaddbh(dvweight4, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

    dvdata_circle5.lo = vuchar_load(agen_ld_img_square);
    dvdata_circle5.hi = vuchar_load(agen_ld_img_square);
    dvdiff5           = dvabsdif(dvdata_circle5, dvdata_central);
    dvdiff5           = dvdiff5 + cfg->lookupOffsets[0];
    dvweight5.lo      = vlookup_32bu(weight_ptr, dvdiff5.lo);
    dvweight5.hi      = vlookup_32bu(weight_ptr, dvdiff5.hi);
    dvmaddbh(dvdata_circle5, dvweight5, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
             dvvalue_sum_hi);
    dvmaddbh(dvweight5, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

    dvdata_circle6.lo = vuchar_load(agen_ld_img_square);
    dvdata_circle6.hi = vuchar_load(agen_ld_img_square);
    dvdiff6           = dvabsdif(dvdata_circle6, dvdata_central);
    dvdiff6           = dvdiff6 + cfg->lookupOffsets[1];
    dvweight6.lo      = vlookup_32bu(weight_ptr, dvdiff6.lo);
    dvweight6.hi      = vlookup_32bu(weight_ptr, dvdiff6.hi);
    dvmaddbh(dvdata_circle6, dvweight6, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
             dvvalue_sum_hi);
    dvmaddbh(dvweight6, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

    dvdata_circle7.lo = vuchar_load(agen_ld_img_square);
    dvdata_circle7.hi = vuchar_load(agen_ld_img_square);
    dvdiff7           = dvabsdif(dvdata_circle7, dvdata_central);
    dvdiff7           = dvdiff7 + cfg->lookupOffsets[2];
    dvweight7.lo      = vlookup_32bu(weight_ptr, dvdiff7.lo);
    dvweight7.hi      = vlookup_32bu(weight_ptr, dvdiff7.hi);
    dvmaddbh(dvdata_circle7, dvweight7, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
             dvvalue_sum_hi);
    dvmaddbh(dvweight7, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

    dvdata_circle8.lo = vuchar_load(agen_ld_img_square);
    dvdata_circle8.hi = vuchar_load(agen_ld_img_square);
    dvdiff8           = dvabsdif(dvdata_circle8, dvdata_central);
    dvdiff8           = dvdiff8 + cfg->lookupOffsets[3];
    dvweight8.lo      = vlookup_32bu(weight_ptr, dvdiff8.lo);
    dvweight8.hi      = vlookup_32bu(weight_ptr, dvdiff8.hi);
    dvmaddbh(dvdata_circle8, dvweight8, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
             dvvalue_sum_hi);
    dvmaddbh(dvweight8, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

    dvdata_circle9.lo = vuchar_load(agen_ld_img_square);
    dvdata_circle9.hi = vuchar_load(agen_ld_img_square);
    dvdiff9.lo        = replicateb(cfg->lookupOffsets[4]);
    dvdiff9.hi        = replicateb(cfg->lookupOffsets[4]);
    dvweight9.lo      = vlookup_32bu(weight_ptr, dvdiff9.lo);
    dvweight9.hi      = vlookup_32bu(weight_ptr, dvdiff9.hi);
    dvmaddbh(dvdata_central, dvweight9, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
             dvvalue_sum_hi);
    dvmaddbh(dvweight9, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

    dvdata_circle10.lo = vuchar_load(agen_ld_img_square);
    dvdata_circle10.hi = vuchar_load(agen_ld_img_square);
    dvdiff10           = dvabsdif(dvdata_circle10, dvdata_central);
    dvdiff10           = dvdiff10 + cfg->lookupOffsets[5];
    dvweight10.lo      = vlookup_32bu(weight_ptr, dvdiff10.lo);
    dvweight10.hi      = vlookup_32bu(weight_ptr, dvdiff10.hi);
    dvmaddbh(dvdata_circle10, dvweight10, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
             dvvalue_sum_hi);
    dvmaddbh(dvweight10, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

    dvdata_circle11.lo = vuchar_load(agen_ld_img_square);
    dvdata_circle11.hi = vuchar_load(agen_ld_img_square);
    dvdiff11           = dvabsdif(dvdata_circle11, dvdata_central);
    dvdiff11           = dvdiff11 + cfg->lookupOffsets[6];
    dvweight11.lo      = vlookup_32bu(weight_ptr, dvdiff11.lo);
    dvweight11.hi      = vlookup_32bu(weight_ptr, dvdiff11.hi);
    dvmaddbh(dvdata_circle11, dvweight11, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
             dvvalue_sum_hi);
    dvmaddbh(dvweight11, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

    dvdata_circle12.lo = vuchar_load(agen_ld_img_square);
    dvdata_circle12.hi = vuchar_load(agen_ld_img_square);
    dvdiff12           = dvabsdif(dvdata_circle12, dvdata_central);
    dvdiff12           = dvdiff12 + cfg->lookupOffsets[7];
    dvweight12.lo      = vlookup_32bu(weight_ptr, dvdiff12.lo);
    dvweight12.hi      = vlookup_32bu(weight_ptr, dvdiff12.hi);
    dvweight_bypass1   = dvweight12;
    dvmaddbh(dvdata_circle12, dvweight12, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
             dvvalue_sum_hi);
    dvmaddbh(dvweight12, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

    dvdata_circle13.lo = vuchar_load(agen_ld_img_square);
    dvdata_circle13.hi = vuchar_load(agen_ld_img_square);
    dvdiff13           = dvabsdif(dvdata_circle13, dvdata_central);
    dvdiff13           = dvdiff13 + cfg->lookupOffsets[8];
    dvweight13.lo      = vlookup_32bu(weight_ptr, dvdiff13.lo);
    dvweight13.hi      = vlookup_32bu(weight_ptr, dvdiff13.hi);
    dvmaddbh(dvdata_circle13, dvweight13, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
             dvvalue_sum_hi);
    dvmaddbh(dvweight13, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

    dvdata_central.lo = vuchar_load(agen_ld_img_central); // Rotated loads manually
    dvdata_central.hi = vuchar_load(agen_ld_img_central); // Rotated loads manually

    vstore_hw(dvvalue_sum_lo.lo, agen_st_value);
    vstore_hw(dvvalue_sum_lo.hi, agen_st_value);
    vstore_hw(dvvalue_sum_hi.lo, agen_st_value);
    vstore_hw(dvvalue_sum_hi.hi, agen_st_value);

    vstore_hw(dvweight_sum_lo.lo, agen_st_weight);
    vstore_hw(dvweight_sum_lo.hi, agen_st_weight);
    vstore_hw(dvweight_sum_hi.lo, agen_st_weight);
    vstore_hw(dvweight_sum_hi.hi, agen_st_weight);

    // Prolog 1
    dvdata_circle1.lo = vuchar_load(agen_ld_img_corner);
    dvdata_circle1.hi = vuchar_load(agen_ld_img_corner);
    dvdiff1           = dvabsdif(dvdata_circle1, dvdata_central);
    dvweight1.lo      = vlookup_32bu(weight_ptr, dvdiff1.lo);
    dvweight1.hi      = vlookup_32bu(weight_ptr, dvdiff1.hi);
    dvmaddbh(dvdata_circle1, dvweight1, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, 0, dvvalue_sum_lo, dvvalue_sum_hi);
    dvmaddbh(dvweight1, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, 0, dvweight_sum_lo, dvweight_sum_hi);

    dvdata_circle2.lo = vuchar_load(agen_ld_img_corner);
    dvdata_circle2.hi = vuchar_load(agen_ld_img_corner);
    dvdiff2           = dvabsdif(dvdata_circle2, dvdata_central);
    dvweight2.lo      = vlookup_32bu(weight_ptr, dvdiff2.lo);
    dvweight2.hi      = vlookup_32bu(weight_ptr, dvdiff2.hi);
    dvmaddbh(dvdata_circle2, dvweight2, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
             dvvalue_sum_hi);
    dvmaddbh(dvweight2, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

    dvdata_circle3.lo = vuchar_load(agen_ld_img_corner);
    dvdata_circle3.hi = vuchar_load(agen_ld_img_corner);
    dvdiff3           = dvabsdif(dvdata_circle3, dvdata_central);
    dvweight3.lo      = vlookup_32bu(weight_ptr, dvdiff3.lo);
    dvweight3.hi      = vlookup_32bu(weight_ptr, dvdiff3.hi);
    dvmaddbh(dvdata_circle3, dvweight3, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
             dvvalue_sum_hi);
    dvmaddbh(dvweight3, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

    dvdata_circle4.lo  = vuchar_load(agen_ld_img_corner);
    dvdata_circle4.hi  = vuchar_load(agen_ld_img_corner);
    dvdiff4            = dvabsdif(dvdata_circle4, dvdata_central);
    dvweight4.lo       = vlookup_32bu(weight_ptr, dvdiff4.lo);
    dvweight4.hi       = vlookup_32bu(weight_ptr, dvdiff4.hi);
    dvweight_bypass2_1 = dvweight4;
    dvmaddbh(dvdata_circle4, dvweight4, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
             dvvalue_sum_hi);
    dvmaddbh(dvweight4, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

    dvdata_circle5.lo = vuchar_load(agen_ld_img_square);
    dvdata_circle5.hi = vuchar_load(agen_ld_img_square);
    dvdiff5           = dvabsdif(dvdata_circle5, dvdata_central);
    dvdiff5           = dvdiff5 + cfg->lookupOffsets[0];
    dvweight5.lo      = vlookup_32bu(weight_ptr, dvdiff5.lo);
    dvweight5.hi      = vlookup_32bu(weight_ptr, dvdiff5.hi);
    dvmaddbh(dvdata_circle5, dvweight5, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
             dvvalue_sum_hi);
    dvmaddbh(dvweight5, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

    dvdata_circle6.lo = vuchar_load(agen_ld_img_square);
    dvdata_circle6.hi = vuchar_load(agen_ld_img_square);
    dvmaddbh(dvdata_circle6, dvweight_bypass1, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
             dvvalue_sum_hi);
    dvmaddbh(dvweight_bypass1, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

    dvdata_circle7.lo = vuchar_load(agen_ld_img_square);
    dvdata_circle7.hi = vuchar_load(agen_ld_img_square);
    dvdiff7           = dvabsdif(dvdata_circle7, dvdata_central);
    dvdiff7           = dvdiff7 + cfg->lookupOffsets[2];
    dvweight7.lo      = vlookup_32bu(weight_ptr, dvdiff7.lo);
    dvweight7.hi      = vlookup_32bu(weight_ptr, dvdiff7.hi);
    dvmaddbh(dvdata_circle7, dvweight7, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
             dvvalue_sum_hi);
    dvmaddbh(dvweight7, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

    dvdata_circle8.lo = vuchar_load(agen_ld_img_square);
    dvdata_circle8.hi = vuchar_load(agen_ld_img_square);
    dvdiff8           = dvabsdif(dvdata_circle8, dvdata_central);
    dvdiff8           = dvdiff8 + cfg->lookupOffsets[3];
    dvweight8.lo      = vlookup_32bu(weight_ptr, dvdiff8.lo);
    dvweight8.hi      = vlookup_32bu(weight_ptr, dvdiff8.hi);
    dvmaddbh(dvdata_circle8, dvweight8, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
             dvvalue_sum_hi);
    dvmaddbh(dvweight8, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

    dvdata_circle9.lo = vuchar_load(agen_ld_img_square);
    dvdata_circle9.hi = vuchar_load(agen_ld_img_square);
    dvdiff9.lo        = replicateb(cfg->lookupOffsets[4]);
    dvdiff9.hi        = replicateb(cfg->lookupOffsets[4]);
    dvweight9.lo      = vlookup_32bu(weight_ptr, dvdiff9.lo);
    dvweight9.hi      = vlookup_32bu(weight_ptr, dvdiff9.hi);
    dvmaddbh(dvdata_central, dvweight9, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
             dvvalue_sum_hi);
    dvmaddbh(dvweight9, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

    dvdata_circle10.lo = vuchar_load(agen_ld_img_square);
    dvdata_circle10.hi = vuchar_load(agen_ld_img_square);
    dvdiff10           = dvabsdif(dvdata_circle10, dvdata_central);
    dvdiff10           = dvdiff10 + cfg->lookupOffsets[5];
    dvweight10.lo      = vlookup_32bu(weight_ptr, dvdiff10.lo);
    dvweight10.hi      = vlookup_32bu(weight_ptr, dvdiff10.hi);
    dvmaddbh(dvdata_circle10, dvweight10, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
             dvvalue_sum_hi);
    dvmaddbh(dvweight10, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

    dvdata_circle11.lo = vuchar_load(agen_ld_img_square);
    dvdata_circle11.hi = vuchar_load(agen_ld_img_square);
    dvdiff11           = dvabsdif(dvdata_circle11, dvdata_central);
    dvdiff11           = dvdiff11 + cfg->lookupOffsets[6];
    dvweight11.lo      = vlookup_32bu(weight_ptr, dvdiff11.lo);
    dvweight11.hi      = vlookup_32bu(weight_ptr, dvdiff11.hi);
    dvmaddbh(dvdata_circle11, dvweight11, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
             dvvalue_sum_hi);
    dvmaddbh(dvweight11, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

    dvdata_circle12.lo = vuchar_load(agen_ld_img_square);
    dvdata_circle12.hi = vuchar_load(agen_ld_img_square);
    dvdiff12           = dvabsdif(dvdata_circle12, dvdata_central);
    dvdiff12           = dvdiff12 + cfg->lookupOffsets[7];
    dvweight12.lo      = vlookup_32bu(weight_ptr, dvdiff12.lo);
    dvweight12.hi      = vlookup_32bu(weight_ptr, dvdiff12.hi);
    dvweight_bypass1   = dvweight12;
    dvmaddbh(dvdata_circle12, dvweight12, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
             dvvalue_sum_hi);
    dvmaddbh(dvweight12, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

    dvdata_circle13.lo = vuchar_load(agen_ld_img_square);
    dvdata_circle13.hi = vuchar_load(agen_ld_img_square);
    dvdiff13           = dvabsdif(dvdata_circle13, dvdata_central);
    dvdiff13           = dvdiff13 + cfg->lookupOffsets[8];
    dvweight13.lo      = vlookup_32bu(weight_ptr, dvdiff13.lo);
    dvweight13.hi      = vlookup_32bu(weight_ptr, dvdiff13.hi);
    dvmaddbh(dvdata_circle13, dvweight13, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
             dvvalue_sum_hi);
    dvmaddbh(dvweight13, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

    dvdata_central.lo = vuchar_load(agen_ld_img_central); // Rotated loads manually
    dvdata_central.hi = vuchar_load(agen_ld_img_central); // Rotated loads manually

    vstore_hw(dvvalue_sum_lo.lo, agen_st_value);
    vstore_hw(dvvalue_sum_lo.hi, agen_st_value);
    vstore_hw(dvvalue_sum_hi.lo, agen_st_value);
    vstore_hw(dvvalue_sum_hi.hi, agen_st_value);

    vstore_hw(dvweight_sum_lo.lo, agen_st_weight);
    vstore_hw(dvweight_sum_lo.hi, agen_st_weight);
    vstore_hw(dvweight_sum_hi.lo, agen_st_weight);
    vstore_hw(dvweight_sum_hi.hi, agen_st_weight);

    for (int32_t iter = 0; iter < niter - 2; iter++) chess_loop_range(128, )
    chess_prepare_for_pipelining
    {
        dvdata_circle1.lo = vuchar_load(agen_ld_img_corner);
        dvdata_circle1.hi = vuchar_load(agen_ld_img_corner);
        dvmaddbh(dvdata_circle1, dvweight_bypass2_0, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, 0, dvvalue_sum_lo,
                 dvvalue_sum_hi);
        dvmaddbh(dvweight_bypass2_0, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, 0, dvweight_sum_lo,
                 dvweight_sum_hi);
        dvweight_bypass2_0 = dvweight_bypass2_1;

        dvdata_circle2.lo = vuchar_load(agen_ld_img_corner);
        dvdata_circle2.hi = vuchar_load(agen_ld_img_corner);
        dvdiff2           = dvabsdif(dvdata_circle2, dvdata_central);
        dvweight2.lo      = vlookup_32bu(weight_ptr, dvdiff2.lo);
        dvweight2.hi      = vlookup_32bu(weight_ptr, dvdiff2.hi);
        dvmaddbh(dvdata_circle2, dvweight2, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
                 dvvalue_sum_hi);
        dvmaddbh(dvweight2, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

        dvdata_circle3.lo = vuchar_load(agen_ld_img_corner);
        dvdata_circle3.hi = vuchar_load(agen_ld_img_corner);
        dvdiff3           = dvabsdif(dvdata_circle3, dvdata_central);
        dvweight3.lo      = vlookup_32bu(weight_ptr, dvdiff3.lo);
        dvweight3.hi      = vlookup_32bu(weight_ptr, dvdiff3.hi);
        dvmaddbh(dvdata_circle3, dvweight3, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
                 dvvalue_sum_hi);
        dvmaddbh(dvweight3, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

        dvdata_circle4.lo  = vuchar_load(agen_ld_img_corner);
        dvdata_circle4.hi  = vuchar_load(agen_ld_img_corner);
        dvdiff4            = dvabsdif(dvdata_circle4, dvdata_central);
        dvweight4.lo       = vlookup_32bu(weight_ptr, dvdiff4.lo);
        dvweight4.hi       = vlookup_32bu(weight_ptr, dvdiff4.hi);
        dvweight_bypass2_1 = dvweight4;
        dvmaddbh(dvdata_circle4, dvweight4, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
                 dvvalue_sum_hi);
        dvmaddbh(dvweight4, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

        dvdata_circle5.lo = vuchar_load(agen_ld_img_square);
        dvdata_circle5.hi = vuchar_load(agen_ld_img_square);
        dvdiff5           = dvabsdif(dvdata_circle5, dvdata_central);
        dvdiff5           = dvdiff5 + cfg->lookupOffsets[0];
        dvweight5.lo      = vlookup_32bu(weight_ptr, dvdiff5.lo);
        dvweight5.hi      = vlookup_32bu(weight_ptr, dvdiff5.hi);
        dvmaddbh(dvdata_circle5, dvweight5, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
                 dvvalue_sum_hi);
        dvmaddbh(dvweight5, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

        dvdata_circle6.lo = vuchar_load(agen_ld_img_square);
        dvdata_circle6.hi = vuchar_load(agen_ld_img_square);
        dvmaddbh(dvdata_circle6, dvweight_bypass1, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
                 dvvalue_sum_hi);
        dvmaddbh(dvweight_bypass1, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo,
                 dvweight_sum_hi);

        dvdata_circle7.lo = vuchar_load(agen_ld_img_square);
        dvdata_circle7.hi = vuchar_load(agen_ld_img_square);
        dvdiff7           = dvabsdif(dvdata_circle7, dvdata_central);
        dvdiff7           = dvdiff7 + cfg->lookupOffsets[2];
        dvweight7.lo      = vlookup_32bu(weight_ptr, dvdiff7.lo);
        dvweight7.hi      = vlookup_32bu(weight_ptr, dvdiff7.hi);
        dvmaddbh(dvdata_circle7, dvweight7, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
                 dvvalue_sum_hi);
        dvmaddbh(dvweight7, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

        dvdata_circle8.lo = vuchar_load(agen_ld_img_square);
        dvdata_circle8.hi = vuchar_load(agen_ld_img_square);
        dvdiff8           = dvabsdif(dvdata_circle8, dvdata_central);
        dvdiff8           = dvdiff8 + cfg->lookupOffsets[3];
        dvweight8.lo      = vlookup_32bu(weight_ptr, dvdiff8.lo);
        dvweight8.hi      = vlookup_32bu(weight_ptr, dvdiff8.hi);
        dvmaddbh(dvdata_circle8, dvweight8, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
                 dvvalue_sum_hi);
        dvmaddbh(dvweight8, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

        dvdata_circle9.lo = vuchar_load(agen_ld_img_square);
        dvdata_circle9.hi = vuchar_load(agen_ld_img_square);
        dvdiff9.lo        = replicateb(cfg->lookupOffsets[4]);
        dvdiff9.hi        = replicateb(cfg->lookupOffsets[4]);
        dvweight9.lo      = vlookup_32bu(weight_ptr, dvdiff9.lo);
        dvweight9.hi      = vlookup_32bu(weight_ptr, dvdiff9.hi);
        dvmaddbh(dvdata_central, dvweight9, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
                 dvvalue_sum_hi);
        dvmaddbh(dvweight9, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

        dvdata_circle10.lo = vuchar_load(agen_ld_img_square);
        dvdata_circle10.hi = vuchar_load(agen_ld_img_square);
        dvdiff10           = dvabsdif(dvdata_circle10, dvdata_central);
        dvdiff10           = dvdiff10 + cfg->lookupOffsets[5];
        dvweight10.lo      = vlookup_32bu(weight_ptr, dvdiff10.lo);
        dvweight10.hi      = vlookup_32bu(weight_ptr, dvdiff10.hi);
        dvmaddbh(dvdata_circle10, dvweight10, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
                 dvvalue_sum_hi);
        dvmaddbh(dvweight10, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

        dvdata_circle11.lo = vuchar_load(agen_ld_img_square);
        dvdata_circle11.hi = vuchar_load(agen_ld_img_square);
        dvdiff11           = dvabsdif(dvdata_circle11, dvdata_central);
        dvdiff11           = dvdiff11 + cfg->lookupOffsets[6];
        dvweight11.lo      = vlookup_32bu(weight_ptr, dvdiff11.lo);
        dvweight11.hi      = vlookup_32bu(weight_ptr, dvdiff11.hi);
        dvmaddbh(dvdata_circle11, dvweight11, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
                 dvvalue_sum_hi);
        dvmaddbh(dvweight11, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

        dvdata_circle12.lo = vuchar_load(agen_ld_img_square);
        dvdata_circle12.hi = vuchar_load(agen_ld_img_square);
        dvdiff12           = dvabsdif(dvdata_circle12, dvdata_central);
        dvdiff12           = dvdiff12 + cfg->lookupOffsets[7];
        dvweight12.lo      = vlookup_32bu(weight_ptr, dvdiff12.lo);
        dvweight12.hi      = vlookup_32bu(weight_ptr, dvdiff12.hi);
        dvweight_bypass1   = dvweight12;
        dvmaddbh(dvdata_circle12, dvweight12, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
                 dvvalue_sum_hi);
        dvmaddbh(dvweight12, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

        dvdata_circle13.lo = vuchar_load(agen_ld_img_square);
        dvdata_circle13.hi = vuchar_load(agen_ld_img_square);
        dvdiff13           = dvabsdif(dvdata_circle13, dvdata_central);
        dvdiff13           = dvdiff13 + cfg->lookupOffsets[8];
        dvweight13.lo      = vlookup_32bu(weight_ptr, dvdiff13.lo);
        dvweight13.hi      = vlookup_32bu(weight_ptr, dvdiff13.hi);
        dvmaddbh(dvdata_circle13, dvweight13, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
                 dvvalue_sum_hi);
        dvmaddbh(dvweight13, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);

        dvdata_central.lo = vuchar_load(agen_ld_img_central); // Rotated loads manually
        dvdata_central.hi = vuchar_load(agen_ld_img_central); // Rotated loads manually

        vstore_hw(dvvalue_sum_lo.lo, agen_st_value);
        vstore_hw(dvvalue_sum_lo.hi, agen_st_value);
        vstore_hw(dvvalue_sum_hi.lo, agen_st_value);
        vstore_hw(dvvalue_sum_hi.hi, agen_st_value);

        vstore_hw(dvweight_sum_lo.lo, agen_st_weight);
        vstore_hw(dvweight_sum_lo.hi, agen_st_weight);
        vstore_hw(dvweight_sum_hi.lo, agen_st_weight);
        vstore_hw(dvweight_sum_hi.hi, agen_st_weight);
    }
}
#endif
#ifdef KSIZE_7X7
void bilateral_filter_accumulate_stage(uint8_t *weight_tbl, BilateralFilterConfig *cfg)
{
    int32_t niter              = cfg->niter / 2;
    agen_A agen_ld_img_corner  = init_agen_A_from_cfg(cfg->cfg_ld_img_corner);
    agen_A agen_ld_img_square  = init_agen_A_from_cfg(cfg->cfg_ld_img_square);
    agen_A agen_ld_img_central = init_agen_A_from_cfg(cfg->cfg_ld_img_central);
    agen_B agen_st_value       = init_agen_B_from_cfg(cfg->cfg_st_value);
    agen_C agen_st_weight      = init_agen_C_from_cfg(cfg->cfg_st_weight);

    dvcharx dvdata_circle, dvdata_central;
    dvcharx dvweight, dvdiff;
    dvshortx dvvalue_sum_lo, dvvalue_sum_hi, dvweight_sum_lo, dvweight_sum_hi;

    const uint8_t chess_storage(RAM_Cb) *weight_ptr = (uint8_t chess_storage(RAM_Cb) *)(weight_tbl);
    dvdata_central.lo = vuchar_load(agen_ld_img_central); // Pre-load and move to behind the loop
    dvdata_central.hi = vuchar_load(agen_ld_img_central); // Pre-load and move to behind the loop

    for (int32_t iter = 0; iter < niter; iter++) chess_loop_range(128, )
    chess_prepare_for_pipelining
    {
        for (int32_t i = 0; i < 4; i++) chess_unroll_loop(*)
        {
            dvdata_circle.lo = vuchar_load(agen_ld_img_corner);
            dvdata_circle.hi = vuchar_load(agen_ld_img_corner);
            dvdiff           = dvabsdif(dvdata_circle, dvdata_central);

            dvweight.lo = vlookup_32bu(weight_ptr, dvdiff.lo);
            dvweight.hi = vlookup_32bu(weight_ptr, dvdiff.hi);

            int32_t pred = (i == 0) - 1;
            dvmaddbh(dvdata_circle, dvweight, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, pred, dvvalue_sum_lo,
                     dvvalue_sum_hi);
            dvmaddbh(dvweight, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, pred, dvweight_sum_lo,
                     dvweight_sum_hi);
        }
        int32_t cnt = 0;
        for (int32_t i = 0; i < (7 - 2) * (7 - 2); i++) chess_unroll_loop(5)
        {
            dvdata_circle.lo = vuchar_load(agen_ld_img_square);
            dvdata_circle.hi = vuchar_load(agen_ld_img_square);
            dvdiff           = dvabsdif(dvdata_circle, dvdata_central);
            dvdiff           = dvdiff + cfg->lookupOffsets[cnt];
            cnt++;

            dvweight.lo = vlookup_32bu(weight_ptr, dvdiff.lo);
            dvweight.hi = vlookup_32bu(weight_ptr, dvdiff.hi);
            dvmaddbh(dvdata_circle, dvweight, dvvalue_sum_lo, dvvalue_sum_hi, VPU_ROUND_0, -1, dvvalue_sum_lo,
                     dvvalue_sum_hi);
            dvmaddbh(dvweight, 1, dvweight_sum_lo, dvweight_sum_hi, VPU_ROUND_0, -1, dvweight_sum_lo, dvweight_sum_hi);
        }
        dvdata_central.lo = vuchar_load(agen_ld_img_central); // Rotated loads manually
        dvdata_central.hi = vuchar_load(agen_ld_img_central); // Rotated loads manually

        vstore_hw(dvvalue_sum_lo.lo, agen_st_value);
        vstore_hw(dvvalue_sum_lo.hi, agen_st_value);
        vstore_hw(dvvalue_sum_hi.lo, agen_st_value);
        vstore_hw(dvvalue_sum_hi.hi, agen_st_value);

        vstore_hw(dvweight_sum_lo.lo, agen_st_weight);
        vstore_hw(dvweight_sum_lo.hi, agen_st_weight);
        vstore_hw(dvweight_sum_hi.lo, agen_st_weight);
        vstore_hw(dvweight_sum_hi.hi, agen_st_weight);
    }
}
#endif

void bilateral_filter_division_stage(BilateralFilterConfig *cfg)
{
    int32_t niter = cfg->niter;

    agen_B agen_ld_value  = init_agen_B_from_cfg(cfg->cfg_ld_value);
    agen_C agen_ld_weight = init_agen_C_from_cfg(cfg->cfg_ld_weight);
    agen_A agen_st_img    = init_agen_A_from_cfg(cfg->cfg_st_img);

    dvintx dvint_value_sum_lo, dvint_value_sum_hi;
    dvintx dvint_weight_sum_lo, dvint_weight_sum_hi;
    dvfloatx dvf_value_sum_lo, dvf_value_sum_hi;
    dvfloatx dvf_weight_sum_lo, dvf_weight_sum_hi;
    dvintx dvint_output_image_lo, dvint_output_image_hi;
    dvshortx dvshort_output_image;

    for (int32_t iter = 0; iter < niter; iter++) chess_loop_range(1024, )
    chess_prepare_for_pipelining
    {
        dvint_value_sum_lo = dvuint_load(agen_ld_value);
        dvf_value_sum_lo   = dvint_dvfp(dvint_value_sum_lo);

        dvint_value_sum_hi = dvuint_load(agen_ld_value);
        dvf_value_sum_hi   = dvint_dvfp(dvint_value_sum_hi);

        dvint_weight_sum_lo = dvuint_load(agen_ld_weight);
        dvf_weight_sum_lo   = dvfrcp(dvint_dvfp(dvint_weight_sum_lo));

        dvint_weight_sum_hi = dvuint_load(agen_ld_weight);
        dvf_weight_sum_hi   = dvfrcp(dvint_dvfp(dvint_weight_sum_hi));

        dvint_output_image_lo = (dvintx)(dvf_value_sum_lo * dvf_weight_sum_lo);
        dvint_output_image_hi = (dvintx)(dvf_value_sum_hi * dvf_weight_sum_hi);

        dvshort_output_image.lo = vdemote(dvint_output_image_lo);
        dvshort_output_image.hi = vdemote(dvint_output_image_hi);

        vstore_i_hb(dvshort_output_image, agen_st_img);
    }
}