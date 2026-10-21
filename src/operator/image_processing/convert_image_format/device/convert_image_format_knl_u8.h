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

#ifndef CONVERT_IMAGE_FORMAT_KNL_U8_H
#define CONVERT_IMAGE_FORMAT_KNL_U8_H

#include "convert_image_format_knl.h"

static inline void generate_alpha(uint8_t *buf)
{
    agen out = init(buf);
    out.n1   = 1;
    out.mod1 = chess_elementsof(dvchar);

    dvcharx value;
    value.lo = replicateb(ALPHA_VALUE);
    value.hi = replicateb(ALPHA_VALUE);

    chess_separator_scheduler();

    vstore(value, out);

    mem_fence();
}

void convert_image_format_yuv2nv12_agen_init(int tw, int th, int lp_in, int lp_out, int channel_in, int channel_out,
                                             uint8_t *tmp_uv, ConvertImageFormatConfig *cfg);
void convert_image_format_yuv2yuv_agen_init(int tw, int th, int lp_in, int lp_out, int channel_in, int channel_out,
                                            ConvertImageFormatConfig *cfg);
void convert_image_format_nv122rgb_agen_init(int tw, int th, int lp_in, int lp_out, int channel_in, int channel_out,
                                             uint8_t *tmp_uv, ConvertImageFormatConfig *cfg);
void convert_image_format_yuv2rgb_agen_init(int tw, int th, int lp_in, int lp_out, int channel_in, int channel_out,
                                            ConvertImageFormatConfig *cfg);
void convert_image_format_rgba2rgba_agen_init(int tw, int th, int lp_in, int lp_out, int channel_in, int channel_out,
                                              uint8_t *src_a, uint8_t *dst_a, ConvertImageFormatConfig *cfg);
void convert_image_format_rgb2nv12_agen_init(int tw, int th, int lp_in, int lp_out, int channel_in, int channel_out,
                                             uint8_t *tmp_uv, ConvertImageFormatConfig *cfg);
void convert_image_format_rgb2yuv_agen_init(int tw, int th, int lp_in, int lp_out, int channel_in, int channel_out,
                                            ConvertImageFormatConfig *cfg);

void convert_image_format_yuv2nv12_agen_update(uint8_t *src_y, uint8_t *src_u, uint8_t *src_v, uint8_t *dst_y,
                                               uint8_t *dst_uv, ConvertImageFormatConfig *cfg);
void convert_image_format_yuv2yuv_agen_update(uint8_t *src_y, uint8_t *src_u, uint8_t *src_v, uint8_t *dst_y,
                                              uint8_t *dst_u, uint8_t *dst_v, ConvertImageFormatConfig *cfg);
void convert_image_format_nv122rgb_agen_update(uint8_t *src_y, uint8_t *src_uv, uint8_t *dst_r, uint8_t *dst_g,
                                               uint8_t *dst_b, ConvertImageFormatConfig *cfg);
void convert_image_format_nv122rgba_agen_update(uint8_t *src_y, uint8_t *src_uv, uint8_t *dst_r, uint8_t *dst_g,
                                                uint8_t *dst_b, uint8_t *dst_a, ConvertImageFormatConfig *cfg);
void convert_image_format_yuv2rgb_agen_update(uint8_t *src_y, uint8_t *src_u, uint8_t *src_v, uint8_t *dst_r,
                                              uint8_t *dst_g, uint8_t *dst_b, ConvertImageFormatConfig *cfg);
void convert_image_format_rgba2rgba_agen_update(uint8_t *src_r, uint8_t *src_g, uint8_t *src_b, uint8_t *src_a,
                                                uint8_t *dst_r, uint8_t *dst_g, uint8_t *dst_b, uint8_t *dst_a,
                                                ConvertImageFormatConfig *cfg);
void convert_image_format_rgb2nv12_agen_update(uint8_t *src_r, uint8_t *src_g, uint8_t *src_b, uint8_t *dst_y,
                                               uint8_t *dst_uv, ConvertImageFormatConfig *cfg);
void convert_image_format_rgb2yuv_agen_update(uint8_t *src_r, uint8_t *src_g, uint8_t *src_b, uint8_t *dst_y,
                                              uint8_t *dst_u, uint8_t *dst_v, ConvertImageFormatConfig *cfg);

void convert_image_format_sampling_knl_exec(ConvertImageFormatConfig *cfg);

void convert_image_format_yuv2yuv_knl_exec(ConvertImageFormatConfig *cfg);
void convert_image_format_rgba2rgba_knl_exec(ConvertImageFormatConfig *cfg);
void convert_image_format_rgb2rgb_knl_exec(ConvertImageFormatConfig *cfg);
void convert_image_format_alpha_fill_knl_exec(ConvertImageFormatConfig *cfg);
void convert_image_format_yuv2rgb_knl_exec(ConvertImageFormatConfig *cfg, uint8_t *coeff_r, int8_t *coeff_g,
                                           uint8_t *coeff_b);
void convert_image_format_rgb2yuv_knl_exec(ConvertImageFormatConfig *cfg, uint8_t *coeff_y, int8_t *coeff_u,
                                           int8_t *coeff_v);

#endif // CONVERT_IMAGE_FORMAT_KNL_U8_H