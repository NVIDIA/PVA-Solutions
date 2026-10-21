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

#include "orb_desc_extr_common.h"

#include "orb_desc_extr_params.h"

void init_orb_desc_extr_common_cfg(ORBDescExtractorConfig *cfg, int32_t keypoint_size, int32_t has_last_batch,
                                   int32_t last_batch_size, int32_t num_batch, int32_t *keypoint_buf_int_x,
                                   int32_t *keypoint_buf_int_y, int32_t *moment_x, int32_t *moment_y, WTS_TYPE *weights,
                                   int8_t *buf_cos_sin, int8_t *buf_msin_cos, int8_t *orb_bit_pattern,
                                   uint16_t *rotated_orb_coords, DATA_TYPE *lookup_pixels, float *feature_coords_buf,
                                   DATA_TYPE *image_patch_buf, WTS_PERM_TYPE *weight_perm_array,
                                   OFFSET_TYPE *offset_vec, CupvaSampler *sampler)
{
    // init weight perm array for centroid moment calculation
#if defined(X8_IO)
    int8_t weight_perm_values[32] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                                     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3};
    int32_t perm_len              = 32;
#elif defined(X16_IO)
    int16_t weight_perm_values[16] = {0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3};
    int32_t perm_len               = 16;
#endif

    for (int32_t i = 0; i < perm_len; i++)
    {
        weight_perm_array[i] = weight_perm_values[i];
    }
    cfg->_weight_perm_array = weight_perm_array;

    cfg->gsdf_batch_size = BATCH_SIZE;
    cfg->gsdf_tile_lp    = PATCH_WIDTH;
    // align batch stride to 64k+2 for transpose load
    cfg->gsdf_tile_stride = TABLE_BATCH_STRIDE;

    cfg->has_last_batch       = has_last_batch;
    cfg->last_batch_size      = last_batch_size;
    cfg->num_batch            = num_batch;
    cfg->keypoint_size        = keypoint_size;
    cfg->feature_tile_size    = keypoint_size * BATCH_SIZE;
    cfg->descriptor_tile_size = DESCRIPTOR_ARRAY_NUM_BYTES * BATCH_SIZE;
    // patch center offset for 2D lookup
    cfg->_patch_center_x_offset = (PATCH_WIDTH - 1) >> 1;
    cfg->_patch_center_y_offset = (PATCH_HEIGHT - 1) >> 1;

    // init intermediate buffer addresses
    cfg->_keypoint_buf_int_x = keypoint_buf_int_x;
    cfg->_keypoint_buf_int_y = keypoint_buf_int_y;
    cfg->_moment_x           = moment_x;
    cfg->_moment_y           = moment_y;
    cfg->_weights            = weights;
    cfg->_buf_cos_sin        = buf_cos_sin;
    cfg->_buf_msin_cos       = buf_msin_cos;
    cfg->_orb_bit_pattern    = orb_bit_pattern;
    cfg->_rotated_orb_coords = rotated_orb_coords;
    cfg->_lookup_pixels      = lookup_pixels;
    cfg->_offset_vec         = offset_vec;
    cfg->_sampler            = sampler;

    // init for truncating float feature coordinates to int32_t
    init_trunc_float_to_int_xy_params(cfg, feature_coords_buf);

    // init for centroid calculation
    init_centroid_calc_params(cfg, image_patch_buf);

    // init for rotation angle calculation
    init_get_rotation_angle_params(cfg);

    // init for rotating the orb patterns
#if defined(KNL_RBRIEF)
    init_rotate_orb_pattern_params(cfg);
#else
    init_shift_orb_pattern_params(cfg);
#endif

    // init 2D lookup sampler
    init_2d_lookup_sampler_params(cfg);

    // init for comparing pixels
    init_compare_pixels_params(cfg);
}

void init_trunc_float_to_int_xy_params(ORBDescExtractorConfig *cfg, float *feature_coords_buf)
{
    int32_t vecw  = pva_elementsof(dvfloat);
    int32_t niter = 2 * cfg->gsdf_batch_size / vecw;

    AgenWrapper w0;
    w0.size = sizeof(float);
    w0.n1   = niter;
    w0.s1   = vecw;

    agen agen_in = init((dvfloat *)feature_coords_buf);
    INIT_AGEN1(agen_in, w0);
    cfg->_cvt_feature_coords_to_int_agen_in_cfg = extract_agen_cfg(agen_in);

    AgenWrapper w1;
    w1.size = sizeof(int32_t);
    w1.n1   = niter;
    w1.s1   = pva_elementsof(vint);

    agen agen_out_x = init((vint *)cfg->_keypoint_buf_int_x);
    INIT_AGEN1(agen_out_x, w1);
    cfg->_cvt_feature_coords_to_int_agen_out_x_cfg = extract_agen_cfg(agen_out_x);

    agen agen_out_y = init((vint *)cfg->_keypoint_buf_int_y);
    INIT_AGEN1(agen_out_y, w1);
    cfg->_cvt_feature_coords_to_int_agen_out_y_cfg = extract_agen_cfg(agen_out_y);
}

void trunc_float_to_int_xy(ORBDescExtractorConfig *cfg)
{
    agen_A a_in    = init_agen_A_from_cfg(cfg->_cvt_feature_coords_to_int_agen_in_cfg);
    agen_B a_out_x = init_agen_B_from_cfg(cfg->_cvt_feature_coords_to_int_agen_out_x_cfg);
    agen_B a_out_y = init_agen_B_from_cfg(cfg->_cvt_feature_coords_to_int_agen_out_y_cfg);

    int32_t x_offset = cfg->_patch_center_x_offset;
    int32_t y_offset = cfg->_patch_center_y_offset;

    chess_separator_scheduler();

    for (int32_t i = 0; i < TRUNC_FLOAT_TO_INT_XY_NITER; i++) chess_unroll_loop(*)
    {
        dvfloatx df_xy = dvfloat_load_di(a_in);
        vintx w_x      = vfp_vint_trunc(df_xy.lo) - x_offset;
        vintx w_y      = vfp_vint_trunc(df_xy.hi) - y_offset;
        vstore(w_x, a_out_x);
        vstore(w_y, a_out_y);
    }
}

void update_gsdf_params(float *feature_coords_buf, DATA_TYPE *image_patch_buf, GSDFHandler *gsdf_handler,
                        ORBDescExtractorConfig *cfg, int32_t iter_idx)
{
    cupvaModifyAgenCfgBase(&cfg->_cvt_feature_coords_to_int_agen_in_cfg, feature_coords_buf);
    trunc_float_to_int_xy(cfg);

    // update batch size for the last batch
    int32_t batch_size =
        (iter_idx == cfg->num_batch - 1 && cfg->has_last_batch) ? cfg->last_batch_size : cfg->gsdf_batch_size;

    cupvaGSDFUpdateVmemAddr(gsdf_handler, (void *)image_patch_buf, cfg->gsdf_tile_lp, cfg->gsdf_tile_stride);
    cupvaGSDFUpdateTiles(gsdf_handler, cfg->_keypoint_buf_int_x, cfg->_keypoint_buf_int_y, batch_size);
}

void init_centroid_calc_params(ORBDescExtractorConfig *cfg, DATA_TYPE *image_patch_buf)
{
    // the neighborhood to compute the moments is a smaller patch inside the image patch
    int32_t patch_radius_x = (PATCH_WIDTH - 1) >> 1;
    int32_t patch_radius_y = (PATCH_HEIGHT - 1) >> 1;
    int32_t row_ofst       = patch_radius_x - 15;
    int32_t col_ofst       = patch_radius_y - 15;
    int32_t nbh_width      = 32;
    int32_t nbh_height     = 31;

#if defined(X8_IO)
    int32_t transp_stride = 1;
#elif defined(X16_IO)
    int32_t transp_stride = 2;
#endif

    int32_t vecw = pva_elementsof(VTYPE);

    AgenWrapper w0;
    w0.size = sizeof(DATA_TYPE);
    w0.n1   = nbh_width / transp_stride;
    w0.s1   = transp_stride;
    w0.n2   = nbh_height;
    w0.s2   = cfg->gsdf_tile_lp;
    w0.n3   = cfg->gsdf_batch_size / vecw;
    w0.s3   = cfg->gsdf_tile_stride * vecw;

    cfg->_centroid_calc_start_addr_ofst = col_ofst * cfg->gsdf_tile_lp + row_ofst;
    agen agen_in                        = init((VTYPE *)(image_patch_buf + cfg->_centroid_calc_start_addr_ofst));
    INIT_AGEN3(agen_in, w0);
    // set lane ofst for transpose load
#if defined(X8_IO)
    // byte transp load, lp ~ 64k+2
    int32_t lane_ofst = (cfg->gsdf_tile_stride - 2 + 63) / 64;
#elif defined(X16_IO)
    // halfword T2 transp load, lp ~ 32k+2
    int32_t lane_ofst               = (cfg->gsdf_tile_stride - 2 + 31) / 32;
#endif
    agen_in.lane_ofst          = lane_ofst;
    cfg->_centroid_calc_in_cfg = extract_agen_cfg(agen_in);

    // operator on 4 pixels at a time
    int32_t niter             = nbh_width * nbh_height * (cfg->gsdf_batch_size / vecw) / 4;
    cfg->_centroid_calc_niter = niter;

    // agen for loading the weights: interleaved by a group of 4
    AgenWrapper w1;
    w1.size = sizeof(WTS_TYPE);
    w1.n1   = nbh_width * nbh_height / 4;
    w1.s1   = 8;
    w1.n2   = cfg->gsdf_batch_size / vecw;
    w1.s2   = 0;

    agen agen_w = init((WTS_VTYPE *)cfg->_weights);
    INIT_AGEN2(agen_w, w1);
    cfg->_centroid_calc_weights_cfg = extract_agen_cfg(agen_w);

    // output moments: deinterleaved lo and hi of the double vector
    int32_t vecw2  = pva_elementsof(dvint);
    int32_t niter2 = cfg->gsdf_batch_size / vecw2;

    AgenWrapper w2;
    w2.size = sizeof(int32_t);
    w2.n1   = niter2;
    w2.s1   = vecw2;

    agen agen_mx = init((int32_t *)cfg->_moment_x);
    INIT_AGEN1(agen_mx, w2);
    cfg->_centroid_calc_out_mx_cfg = extract_agen_cfg(agen_mx);

    agen agen_my = init((int32_t *)cfg->_moment_y);
    INIT_AGEN1(agen_my, w2);
    cfg->_centroid_calc_out_my_cfg = extract_agen_cfg(agen_my);
}

void init_get_rotation_angle_params(ORBDescExtractorConfig *cfg)
{
    int32_t vecw  = pva_elementsof(dvint);
    int32_t niter = cfg->gsdf_batch_size / vecw;

    AgenWrapper w0;
    w0.size = sizeof(int32_t);
    w0.n1   = niter;
    w0.s1   = vecw;

    agen a_mx = init((dvint *)cfg->_moment_x);
    INIT_AGEN1(a_mx, w0);

    agen a_my = init((dvint *)cfg->_moment_y);
    INIT_AGEN1(a_my, w0);

    cfg->_rotation_angle_in_mx_cfg = extract_agen_cfg(a_mx);
    cfg->_rotation_angle_in_my_cfg = extract_agen_cfg(a_my);

    // output char vectors for [cos, sin] and [-sin, cos]
    AgenWrapper w1;
    w1.size = sizeof(int8_t);
    w1.n1   = niter;
    w1.s1   = pva_elementsof(vchar);

    agen a_cos_sin_char = init((vchar *)cfg->_buf_cos_sin);
    INIT_AGEN1(a_cos_sin_char, w1);

    agen a_msin_cos_char = init((vchar *)cfg->_buf_msin_cos);
    INIT_AGEN1(a_msin_cos_char, w1);

    cfg->_rotation_angle_out_cs_cfg  = extract_agen_cfg(a_cos_sin_char);
    cfg->_rotation_angle_out_msc_cfg = extract_agen_cfg(a_msin_cos_char);
}

void get_rotation_angle(ORBDescExtractorConfig *cfg)
{
    agen_A a_mx = init_agen_A_from_cfg(cfg->_rotation_angle_in_mx_cfg);
    agen_A a_my = init_agen_A_from_cfg(cfg->_rotation_angle_in_my_cfg);

    agen_A a_cs_char  = init_agen_A_from_cfg(cfg->_rotation_angle_out_cs_cfg);
    agen_A a_msc_char = init_agen_A_from_cfg(cfg->_rotation_angle_out_msc_cfg);

    int32_t one_in_qbits = (1 << ANGLE_QBITS);
    float one_in_qbits_f = int_fp(one_in_qbits);

    chess_separator_scheduler();

    for (int32_t i = 0; i < GET_ROTATION_ANGLE_NITER; i++) chess_unroll_loop(*)
    {
        // moment range: 16 + 4 + 8 = 28 bits
        dvintx dw_mx = dvint_load(a_mx);
        dvintx dw_my = dvint_load(a_my);

        // add 1 to avoid zero norm
        dvintx status = (dw_mx && dw_my);
        dw_mx         = dw_mx + !status;

        // multiply the y moment by -1 to rectify the weight sign
        dw_my = -dw_my;

        // double vector pseudo intrinsics
        dvfloatx df_mx       = dvint_dvfp(dw_mx);
        dvfloatx df_my       = dvint_dvfp(dw_my);
        dvfloatx df_mx2      = dvmulf(df_mx, df_mx);
        dvfloatx df_norm2    = dvmaddf(df_my, df_my, df_mx2);
        dvfloatx df_norm_div = dvfrsq(df_norm2);
        dvfloatx df_cos      = dvmulf(df_mx, df_norm_div);
        dvfloatx df_sin      = dvmulf(df_my, df_norm_div);

        // double vector pseudo intrinsics
        df_cos = dvmulf(df_cos, one_in_qbits_f);
        df_sin = dvmulf(df_sin, one_in_qbits_f);
        // round to get fixed-point representations
        dvintx dw_cosf_qbits = dvfp_dvint_round(df_cos);
        dvintx dw_sinf_qbits = dvfp_dvint_round(df_sin);

        // demote to char
        vshortx h_cosf_qbits = vdemote(dw_cosf_qbits);
        vshortx h_sinf_qbits = vdemote(dw_sinf_qbits);

        vcharx b_cos_sin, b_msin_cos;
        vdemote_i(h_cosf_qbits, h_sinf_qbits, b_cos_sin);
        vdemote_i(-h_sinf_qbits, h_cosf_qbits, b_msin_cos);
        vstore(b_cos_sin, a_cs_char);
        vstore(b_msin_cos, a_msc_char);
    }
}

void init_shift_orb_pattern_params(ORBDescExtractorConfig *cfg)
{
    int32_t vecw   = pva_elementsof(vchar);
    int32_t niter1 = 2 * DESCRIPTOR_PATTERN_SIZE / vecw;
    int32_t niter2 = cfg->gsdf_batch_size;

    AgenWrapper w1;
    w1.size = sizeof(int8_t);
    w1.n1   = niter1;
    w1.s1   = vecw;
    w1.n2   = niter2;
    w1.s2   = 0;

    agen a_orb_pattern_in = init((vchar *)cfg->_orb_bit_pattern);
    INIT_AGEN2(a_orb_pattern_in, w1);
    cfg->_rotate_orb_patterns_in_xy_cfg = extract_agen_cfg(a_orb_pattern_in);
    cfg->_rotate_orb_patterns_niter     = niter1 * niter2;

    cfg->_lookup_idx_lp = 2 * DESCRIPTOR_PATTERN_SIZE;

    AgenWrapper w2;
    w2.size                = sizeof(uint16_t);
    w2.n1                  = niter1;
    w2.s1                  = vecw;
    w2.n2                  = niter2;
    w2.s2                  = cfg->_lookup_idx_lp;
    agen a_orb_pattern_out = init((dvshort *)cfg->_rotated_orb_coords);
    INIT_AGEN2(a_orb_pattern_out, w2);
    cfg->_rotate_orb_patterns_out_cfg = extract_agen_cfg(a_orb_pattern_out);

    // orb pattern coordinates offsets for sampler
    for (int32_t i = 0; i < cfg->gsdf_batch_size; i++)
    {
        // xy-interleaved format
        // additional offset in x due to sampler_table_lp = true_lp - 2
        cfg->_offset_vec[2 * i]     = cfg->_patch_center_x_offset + 2 * i;
        cfg->_offset_vec[2 * i + 1] = cfg->_patch_center_y_offset;
    }

    AgenWrapper w3;
    w3.size       = sizeof(uint8_t);
    w3.n1         = niter1;
    w3.s1         = 0;
    w3.n2         = niter2;
    w3.s2         = 2;
    agen a_offset = init((vuchar *)cfg->_offset_vec);
    INIT_AGEN2(a_offset, w3);
    cfg->_rotate_orb_patterns_offset_cfg = extract_agen_cfg(a_offset);
}

void shift_orb_pattern_by_offsets(ORBDescExtractorConfig *cfg)
{
    agen_C a_orb_pattern_in  = init_agen_C_from_cfg(cfg->_rotate_orb_patterns_in_xy_cfg);
    agen_B a_offset          = init_agen_B_from_cfg(cfg->_rotate_orb_patterns_offset_cfg);
    agen_A a_orb_pattern_out = init_agen_A_from_cfg(cfg->_rotate_orb_patterns_out_cfg);

    vshortx pattern_perm = replicateh(0);

    int32_t niter = max(cfg->_rotate_orb_patterns_niter, 24);
    niter         = (niter + 7) / 8 * 8;

    chess_separator_scheduler();

    for (int32_t i = 0; i < niter; i++) chess_loop_range(24, )
    chess_unroll_loop(8)
    chess_prepare_for_pipelining
    {
        // xy-interleaved format
        vcharx orb_pattern    = vchar_load(a_orb_pattern_in);
        vcharx offset         = vuchar_load_perm(a_offset, pattern_perm);
        vcharx orb_coords_out = orb_pattern + offset;

        vstore_bh(orb_coords_out, a_orb_pattern_out);
    }
}

void init_rotate_orb_pattern_params(ORBDescExtractorConfig *cfg)
{
    int32_t vecw   = pva_elementsof(dvshort);
    int32_t niter1 = 2 * cfg->gsdf_batch_size / vecw;
    int32_t niter2 = DESCRIPTOR_PATTERN_SIZE;

    // agen to load orb pattern
    AgenWrapper w1;
    w1.size = sizeof(int8_t);
    w1.n1   = niter1;
    w1.s1   = 0;
    w1.n2   = niter2;
    w1.s2   = 2;

    agen a_orb_pattern = init((vchar *)cfg->_orb_bit_pattern);
    INIT_AGEN2(a_orb_pattern, w1);
    cfg->_rotate_orb_patterns_in_xy_cfg = extract_agen_cfg(a_orb_pattern);

    // double halfword T2 store, lp ~ 32k+2
    int32_t lookup_idx_lp = LOOKUP_IDX_LP;
    cfg->_lookup_idx_lp   = lookup_idx_lp;

    AgenWrapper w2;
    w2.size = sizeof(uint16_t);
    w2.n1   = niter1;
    w2.s1   = lookup_idx_lp * (vecw / 2);
    w2.n2   = niter2;
    w2.s2   = 2;

    agen a_orb_pattern_out = init((dvshort *)cfg->_rotated_orb_coords);
    INIT_AGEN2(a_orb_pattern_out, w2);
    int32_t lane_ofst           = (lookup_idx_lp - 2) / 32;
    a_orb_pattern_out.lane_ofst = lane_ofst;

    cfg->_rotate_orb_patterns_out_cfg = extract_agen_cfg(a_orb_pattern_out);
    cfg->_rotate_orb_patterns_niter   = niter1 * niter2;

    // init offset vector
    a_orb_pattern_out.round = ANGLE_QBITS;

    // orb pattern coordinates offsets for sampler
    for (int32_t i = 0; i < cfg->gsdf_batch_size; i++)
    {
        // xy-interleaved format
        // additional offset in x due to sampler_table_lp = true_lp - 2
        cfg->_offset_vec[2 * i]     = (cfg->_patch_center_x_offset + 2 * i) << ANGLE_QBITS;
        cfg->_offset_vec[2 * i + 1] = cfg->_patch_center_y_offset << ANGLE_QBITS;
    }

    AgenWrapper w3;
    w3.size       = sizeof(uint16_t);
    w3.n1         = niter1;
    w3.s1         = vecw;
    w3.n2         = niter2;
    w3.s2         = 0;
    agen a_offset = init((dvushort *)cfg->_offset_vec);
    INIT_AGEN2(a_offset, w3);
    cfg->_rotate_orb_patterns_offset_cfg = extract_agen_cfg(a_offset);

    // load rotation matrix
    AgenWrapper w4;
    w4.size        = sizeof(int8_t);
    w4.n1          = niter1;
    w4.s1          = vecw;
    w4.n2          = niter2;
    w4.s2          = 0;
    agen a_cos_sin = init((vchar *)cfg->_buf_cos_sin);
    INIT_AGEN2(a_cos_sin, w4);
    cfg->_rotate_orb_patterns_cos_sin_cfg = extract_agen_cfg(a_cos_sin);

    agen a_msin_cos = init((vchar *)cfg->_buf_msin_cos);
    INIT_AGEN2(a_msin_cos, w4);
    cfg->_rotate_orb_patterns_msin_cos_cfg = extract_agen_cfg(a_msin_cos);
}

void rotate_orb_patterns(ORBDescExtractorConfig *cfg)
{
    vshortx pattern_perm = replicateh(0);

    agen_A a_cs_char  = init_agen_A_from_cfg(cfg->_rotate_orb_patterns_cos_sin_cfg);
    agen_A a_msc_char = init_agen_A_from_cfg(cfg->_rotate_orb_patterns_msin_cos_cfg);

    agen_C a_orb_pattern_in  = init_agen_C_from_cfg(cfg->_rotate_orb_patterns_in_xy_cfg);
    agen_B a_offset          = init_agen_B_from_cfg(cfg->_rotate_orb_patterns_offset_cfg);
    agen_A a_orb_pattern_out = init_agen_A_from_cfg(cfg->_rotate_orb_patterns_out_cfg);

    int32_t niter = max(cfg->_rotate_orb_patterns_niter, 12);
    niter         = (niter + 3) / 4 * 4;

    chess_separator_scheduler();

    for (int32_t i = 0; i < niter; i++) chess_loop_range(12, )
    chess_unroll_loop(4)
    chess_prepare_for_pipelining
    {
        vcharx cs_char          = vchar_load(a_cs_char);
        vcharx msc_char         = vchar_load(a_msc_char);
        vcharx orb_coords       = vchar_load_perm(a_orb_pattern_in, pattern_perm);
        dvshortx orb_coords_out = dvushort_load_di(a_offset);

        orb_coords_out = vdotp2_bbh(cs_char, msc_char, orb_coords, orb_coords_out, -1);

        vstore_transp2_i(orb_coords_out, a_orb_pattern_out);
    }
}

void init_2d_lookup_sampler_params(ORBDescExtractorConfig *cfg)
{
#if defined(X8_IO)
    CupvaSamplerInputType inputType = SAMPLER_INPUT_TYPE_U8;
#else
    CupvaSamplerInputType inputType = SAMPLER_INPUT_TYPE_U16;
#endif

    // init CupvaSampler for 2D lookup
    CupvaSamplerInput2D input = {
        .data = NULL,
        .type = inputType,
        // x range: [x_ofst, x_ofst + PATCH_WIDTH - 1], x_ofst <= 2 * BATCH_SIZE
        .width     = (uint32_t)PATCH_WIDTH + 2 * cfg->gsdf_batch_size - 1,
        .height    = (uint32_t)PATCH_HEIGHT - 1,
        .linePitch = (uint32_t)cfg->gsdf_tile_lp,
        // table offset in bytes
        .inputAdv       = (uint32_t)LOOKUP_TABLE_OFFSET * sizeof(DATA_TYPE),
        .outOfRangeMode = SAMPLER_OUT_OF_RANGE_PREDICATE_OFF,
    };
    CupvaSamplerOutput output = {
        .data      = NULL,
        .pitch     = DESCRIPTOR_PATTERN_SIZE,
        .transMode = TRANS_MODE_NONE,
    };

    CupvaSamplerIndices2D indices = {
        .data   = cfg->_rotated_orb_coords,
        .type   = SAMPLER_INDEX_TYPE_U16,
        .width  = (uint16_t)DESCRIPTOR_PATTERN_SIZE,
        .height = (uint16_t)cfg->gsdf_batch_size,
        // indices line pitch in pixels
        .linePitch = (uint16_t)(cfg->_lookup_idx_lp / 2),
#if defined(KNL_RBRIEF)
        .fractionalBits = (uint8_t)ANGLE_QBITS,
#endif
        .fractionalHandling = SAMPLER_FRAC_HANDLING_ROUND,
        .interleaving       = SAMPLER_INTERLEAVING_ELEMENTS,
    };
    cupvaSamplerSetup(cfg->_sampler, &input, &indices, &output);
}

void init_compare_pixels_params(ORBDescExtractorConfig *cfg)
{
    int32_t vecw = pva_elementsof(DVTYPE);

    AgenWrapper w0;
    w0.size = sizeof(DATA_TYPE);
    w0.n1   = DESCRIPTOR_PATTERN_SIZE / vecw;
    w0.s1   = vecw;
    w0.n2   = cfg->gsdf_batch_size;
    w0.s2   = DESCRIPTOR_PATTERN_SIZE;

    agen a_pixels = init((DVTYPE *)cfg->_lookup_pixels);
    INIT_AGEN2(a_pixels, w0);
    cfg->_compare_pixels_in_cfg = extract_agen_cfg(a_pixels);
    cfg->_compare_pixels_niter  = w0.n1 * w0.n2;
}

void compare_pixels(ORBDescExtractorConfig *cfg, uint8_t *descriptor_buffer)
{
    agen_B a_pixels = init_agen_B_from_cfg(cfg->_compare_pixels_in_cfg);

    TMP_DESC_BUF_TYPE *out_buf = (TMP_DESC_BUF_TYPE *)descriptor_buffer;

    chess_separator_scheduler();

    // niter is larger than 24, multiple of 8 already
    int niter = cfg->_compare_pixels_niter;

    for (int i = 0; i < niter; i++) chess_loop_range(24, )
    chess_unroll_loop(8)
    chess_prepare_for_pipelining
    {
        DVTYPEX dv_pix = DVLOAD_DI(a_pixels);
        // descriptor bit value: val = (lo < hi) ? 1 : 0
        // i.e. val = (lo >= hi) ? 0 : 1
        // val1 = (lo >= hi) ? 1 : 0 = ^val
        int val1 = vbitcmp_s(dv_pix.lo, dv_pix.hi);
        // flip the bits
        out_buf[i] = ~(TMP_DESC_BUF_TYPE)val1;
    }
}
