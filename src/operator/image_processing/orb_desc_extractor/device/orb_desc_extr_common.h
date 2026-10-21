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

#ifndef ORB_DESC_EXTR_COMMON_H
#define ORB_DESC_EXTR_COMMON_H

#include "orb_desc_extr_params.h"

#include <cupva_device.h>

typedef struct ORBDescExtractorConfig
{
    int32_t gsdf_tile_lp;
    int32_t gsdf_tile_stride;
    int32_t gsdf_batch_size;
    int32_t has_last_batch;
    int32_t last_batch_size;
    int32_t num_batch;
    int32_t feature_tile_size;
    int32_t keypoint_size;
    int32_t descriptor_tile_size;
    int32_t _patch_center_x_offset;
    int32_t _patch_center_y_offset;
    int32_t _lookup_idx_lp;

    // intermediate buffers
    int32_t *_keypoint_buf_int_x;
    int32_t *_keypoint_buf_int_y;
    int32_t *_moment_x;
    int32_t *_moment_y;
    WTS_TYPE *_weights;
    int8_t *_buf_cos_sin;
    int8_t *_buf_msin_cos;
    int8_t *_orb_bit_pattern;
    uint16_t *_rotated_orb_coords;
    DATA_TYPE *_lookup_pixels;
    WTS_PERM_TYPE *_weight_perm_array;
    OFFSET_TYPE *_offset_vec;

    // convert float feature coords to int
    AgenCFG _cvt_feature_coords_to_int_agen_in_cfg;
    AgenCFG _cvt_feature_coords_to_int_agen_out_x_cfg;
    AgenCFG _cvt_feature_coords_to_int_agen_out_y_cfg;

    // centroid moment calculation
    AgenCFG _centroid_calc_in_cfg;
    AgenCFG _centroid_calc_weights_cfg;
    AgenCFG _centroid_calc_out_mx_cfg;
    AgenCFG _centroid_calc_out_my_cfg;
    int32_t _centroid_calc_start_addr_ofst;
    int32_t _centroid_calc_niter;

    // get rotation angle
    AgenCFG _rotation_angle_in_mx_cfg;
    AgenCFG _rotation_angle_in_my_cfg;
    AgenCFG _rotation_angle_out_cs_cfg;
    AgenCFG _rotation_angle_out_msc_cfg;

    // rotate orb patterns
    AgenCFG _rotate_orb_patterns_in_xy_cfg;
    AgenCFG _rotate_orb_patterns_offset_cfg;
    AgenCFG _rotate_orb_patterns_cos_sin_cfg;
    AgenCFG _rotate_orb_patterns_msin_cos_cfg;
    AgenCFG _rotate_orb_patterns_out_cfg;
    int32_t _rotate_orb_patterns_niter;

    // compare pixels
    AgenCFG _compare_pixels_in_cfg;
    int32_t _compare_pixels_niter;

    CupvaSampler *_sampler;
} ORBDescExtractorConfig;

void init_orb_desc_extr_common_cfg(ORBDescExtractorConfig *cfg, int32_t keypoint_size, int32_t has_last_batch,
                                   int32_t last_batch_size, int32_t num_batch, int32_t *keypoint_buf_int_x,
                                   int32_t *keypoint_buf_int_y, int32_t *moment_x, int32_t *moment_y, WTS_TYPE *weights,
                                   int8_t *buf_cos_sin, int8_t *buf_msin_cos, int8_t *orb_bit_pattern,
                                   uint16_t *rotated_orb_coords, DATA_TYPE *lookup_pixels, float *feature_coords_buf,
                                   DATA_TYPE *image_patch_buf, WTS_PERM_TYPE *weight_perm_array,
                                   OFFSET_TYPE *offset_vec, CupvaSampler *sampler);

void init_trunc_float_to_int_xy_params(ORBDescExtractorConfig *cfg, float *feature_coords_buf);

void trunc_float_to_int_xy(ORBDescExtractorConfig *cfg);

void update_gsdf_params(float *feature_coords_buf, DATA_TYPE *image_patch_buf, GSDFHandler *gsdf_handler,
                        ORBDescExtractorConfig *cfg, int32_t iter_idx);

void init_centroid_calc_params(ORBDescExtractorConfig *cfg, DATA_TYPE *image_patch_buf);

void init_get_rotation_angle_params(ORBDescExtractorConfig *cfg);

void get_rotation_angle(ORBDescExtractorConfig *cfg);

void init_shift_orb_pattern_params(ORBDescExtractorConfig *cfg);

void shift_orb_pattern_by_offsets(ORBDescExtractorConfig *cfg);

void init_rotate_orb_pattern_params(ORBDescExtractorConfig *cfg);

void rotate_orb_patterns(ORBDescExtractorConfig *cfg);

void init_2d_lookup_sampler_params(ORBDescExtractorConfig *cfg);

void init_compare_pixels_params(ORBDescExtractorConfig *cfg);

void compare_pixels(ORBDescExtractorConfig *cfg, uint8_t *descriptor_buffer);

static inline void update_2d_lookup_sampler_addrs(ORBDescExtractorConfig *cfg, void *image_patch_buffer)
{
    cupvaSamplerUpdateAddrs(cfg->_sampler, image_patch_buffer, (void *)cfg->_rotated_orb_coords,
                            (void *)cfg->_lookup_pixels);
}

#endif // ORB_DESC_EXTR_COMMON_H
