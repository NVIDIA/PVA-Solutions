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

#include "convert_image_format_knl_u8.h"

static void convert_image_format_upsampling_agen_init(int niter1, int niter2, int lp, ConvertImageFormatConfig *cfg)
{
    AgenWrapper wrapper;

    // config for agen_src
    wrapper.size = sizeof(uint8_t);
    wrapper.n1   = niter1;
    wrapper.n2   = 2;
    wrapper.n3   = niter2 / 2;
    wrapper.s1   = chess_elementsof(dvchar);
    wrapper.s2   = 0;
    wrapper.s3   = lp;

    agen agen_src = init(0);
    INIT_AGEN3(agen_src, wrapper);

    cfg->inSamplingAgenCfg = extract_agen_cfg(agen_src);

    // config for agen_dst
    wrapper.size = sizeof(uint8_t);
    wrapper.n1   = niter1;
    wrapper.n2   = niter2;
    wrapper.s1   = chess_elementsof(dvchar);
    wrapper.s2   = lp;

    agen agen_dst = init(0);
    INIT_AGEN2(agen_dst, wrapper);

    cfg->outSamplingAgenCfg = extract_agen_cfg(agen_dst);
}

static void convert_image_format_downsampling_agen_init(int niter1, int niter2, int lp, ConvertImageFormatConfig *cfg)
{
    AgenWrapper wrapper;

    // config for agen_src
    wrapper.size = sizeof(uint8_t);
    wrapper.n1   = niter1;
    wrapper.n2   = niter2;
    wrapper.s1   = chess_elementsof(dvchar);
    wrapper.s2   = lp * 2;

    agen agen_src = init(0);
    INIT_AGEN2(agen_src, wrapper);

    cfg->inSamplingAgenCfg = extract_agen_cfg(agen_src);

    // config for agen_dst
    wrapper.size = sizeof(uint8_t);
    wrapper.n1   = niter1;
    wrapper.n2   = niter2;
    wrapper.s1   = chess_elementsof(dvchar);
    wrapper.s2   = lp;

    agen agen_dst = init(0);
    INIT_AGEN2(agen_dst, wrapper);

    cfg->outSamplingAgenCfg = extract_agen_cfg(agen_dst);
}

static void convert_image_format_yuv_agen_init(int niter1, int niter2, int lp, int channel, AgenCFG *agen_cfg_y,
                                               AgenCFG *agen_cfg_u, AgenCFG *agen_cfg_v)
{
    AgenWrapper wrapper;

    // config for agen_y
    wrapper.size = sizeof(uint8_t);
    wrapper.n1   = niter1;
    wrapper.n2   = niter2;
    wrapper.s1   = channel;
    wrapper.s2   = chess_elementsof(vchar) * lp;

    agen agen_y      = init(0);
    agen_y.lane_ofst = lp / chess_elementsof(dvchar);
    INIT_AGEN2(agen_y, wrapper);

    *agen_cfg_y = extract_agen_cfg(agen_y);

    // config for agen_uv
    wrapper.size = sizeof(uint8_t);
    wrapper.n1   = niter1 / 2;
    wrapper.n2   = niter2;
    wrapper.s1   = 2 * channel;
    wrapper.s2   = chess_elementsof(vchar) * lp;

    agen agen_uv      = init(0);
    agen_uv.lane_ofst = lp / chess_elementsof(dvchar);
    INIT_AGEN2(agen_uv, wrapper);

    *agen_cfg_u = extract_agen_cfg(agen_uv);
    *agen_cfg_v = extract_agen_cfg(agen_uv);
}

static void convert_image_format_rgba_agen_init(int niter1, int niter2, int lp, int channel, AgenCFG *agen_cfg_r,
                                                AgenCFG *agen_cfg_g, AgenCFG *agen_cfg_b, AgenCFG *agen_cfg_a)
{
    AgenWrapper wrapper;

    // config for agen_rgb
    wrapper.size = sizeof(uint8_t);
    wrapper.n1   = niter1;
    wrapper.n2   = niter2;
    wrapper.s1   = channel;
    wrapper.s2   = chess_elementsof(vchar) * lp;

    agen agen_rgb      = init(0);
    agen_rgb.lane_ofst = lp / chess_elementsof(dvchar);
    INIT_AGEN2(agen_rgb, wrapper);

    *agen_cfg_r = extract_agen_cfg(agen_rgb);
    *agen_cfg_g = extract_agen_cfg(agen_rgb);
    *agen_cfg_b = extract_agen_cfg(agen_rgb);

    if (channel == 4)
    {
        *agen_cfg_a = extract_agen_cfg(chess_copy(agen_rgb));
    }
    else
    {
        // config for dummy agen_a
        wrapper.size = sizeof(uint8_t);
        wrapper.n1   = niter1;
        wrapper.n2   = niter2;
        wrapper.s1   = 0;
        wrapper.s2   = 0;

        agen agen_a = init(0);
        INIT_AGEN2(agen_a, wrapper);

        *agen_cfg_a = extract_agen_cfg(chess_copy(agen_a));
    }
}

static void convert_image_format_src_rgb_agen_init(int niter1, int niter2, int lp, int channel,
                                                   ConvertImageFormatConfig *cfg)
{
    AgenWrapper wrapper;

    // config for agen_src
    wrapper.size = sizeof(uint8_t);
    wrapper.n1   = niter1;
    wrapper.n2   = niter2;
    wrapper.s1   = channel;
    wrapper.s2   = chess_elementsof(vchar) * lp;

    agen agen_src      = init(0);
    agen_src.lane_ofst = lp / chess_elementsof(dvchar);
    INIT_AGEN2(agen_src, wrapper);

    for (int i = 0; i < 3; i++)
    {
        cfg->inAgenCfgs[CHANNEL_R + i] = extract_agen_cfg(agen_src);
    }
}

static void convert_image_format_src_yuv_agen_init(int niter1, int niter2, int lp, int channel,
                                                   ConvertImageFormatConfig *cfg)
{
    AgenWrapper wrapper;

    // config for agen_src_y
    wrapper.size = sizeof(uint8_t);
    wrapper.n1   = niter1;
    wrapper.n2   = niter2;
    wrapper.s1   = channel;
    wrapper.s2   = chess_elementsof(vchar) * lp;

    agen agen_src_y      = init(0);
    agen_src_y.lane_ofst = lp / chess_elementsof(dvchar);
    INIT_AGEN2(agen_src_y, wrapper);

    cfg->inAgenCfgs[CHANNEL_Y] = extract_agen_cfg(agen_src_y);

    // config for agen_src_uv
    wrapper.size = sizeof(uint8_t);
    wrapper.n1   = 2;
    wrapper.n2   = niter1 / 2;
    wrapper.n3   = niter2;
    wrapper.s1   = 0;
    wrapper.s2   = 2 * channel;
    wrapper.s3   = chess_elementsof(vchar) * lp;

    agen agen_src_uv      = init(0);
    agen_src_uv.lane_ofst = lp / chess_elementsof(dvchar);
    INIT_AGEN3(agen_src_uv, wrapper);

    cfg->inAgenCfgs[CHANNEL_U] = extract_agen_cfg(agen_src_uv);
    cfg->inAgenCfgs[CHANNEL_V] = extract_agen_cfg(agen_src_uv);
}

static void convert_image_format_dst_rgb_agen_init(int niter1, int niter2, int lp, int channel,
                                                   ConvertImageFormatConfig *cfg)
{
    AgenWrapper wrapper;

    // config for agen_dst
    wrapper.size = sizeof(uint8_t);
    wrapper.n1   = niter1;
    wrapper.n2   = niter2;
    wrapper.s1   = channel;
    wrapper.s2   = chess_elementsof(vchar) * lp;

    agen agen_dst      = init(0);
    agen_dst.lane_ofst = lp / chess_elementsof(dvchar);
    agen_dst.round     = 0x80 | QBITS; // truncate
    // saturation in [0, 255]
    agen_dst.sat_opt    = 2;
    agen_dst.sat_lim_lo = 0;
    agen_dst.sat_val_lo = 0;
    agen_dst.sat_lim_hi = 255;
    agen_dst.sat_val_hi = 255;
    INIT_AGEN2(agen_dst, wrapper);

    for (int i = 0; i < 3; i++)
    {
        cfg->outAgenCfgs[CHANNEL_R + i] = extract_agen_cfg(agen_dst);
    }
}

// Plain byte-stride dst agen for the alpha lane of a packed 4-channel output
// (no round/saturate — alpha is a constant byte written directly).
static void convert_image_format_dst_alpha_agen_init(int niter1, int niter2, int lp, int channel,
                                                     ConvertImageFormatConfig *cfg)
{
    AgenWrapper wrapper;

    wrapper.size = sizeof(uint8_t);
    wrapper.n1   = niter1;
    wrapper.n2   = niter2;
    wrapper.s1   = channel;
    wrapper.s2   = chess_elementsof(vchar) * lp;

    agen agen_a      = init(0);
    agen_a.lane_ofst = lp / chess_elementsof(dvchar);
    INIT_AGEN2(agen_a, wrapper);

    cfg->outAgenCfgs[CHANNEL_A] = extract_agen_cfg(agen_a);
}

static void convert_image_format_dst_yuv_agen_init(int niter1, int niter2, int lp, int channel,
                                                   ConvertImageFormatConfig *cfg)
{
    AgenWrapper wrapper;

    // config for agen_dst_y
    wrapper.size = sizeof(uint8_t);
    wrapper.n1   = niter1;
    wrapper.n2   = niter2;
    wrapper.s1   = channel;
    wrapper.s2   = chess_elementsof(vchar) * lp;

    agen agen_dst_y      = init(0);
    agen_dst_y.lane_ofst = lp / chess_elementsof(dvchar);
    agen_dst_y.round     = 0x80 | QBITS; // truncate
    // saturation in [0, 255]
    agen_dst_y.sat_opt    = 2;
    agen_dst_y.sat_lim_lo = 0;
    agen_dst_y.sat_val_lo = 0;
    agen_dst_y.sat_lim_hi = 255;
    agen_dst_y.sat_val_hi = 255;
    INIT_AGEN2(agen_dst_y, wrapper);

    cfg->outAgenCfgs[CHANNEL_Y] = extract_agen_cfg(agen_dst_y);

    // config for agen_dst_uv
    wrapper.size = sizeof(uint8_t);
    wrapper.n1   = 2;
    wrapper.n2   = niter1 / 2;
    wrapper.n3   = niter2;
    wrapper.s1   = 0;
    wrapper.s2   = 2 * channel;
    wrapper.s3   = chess_elementsof(vchar) * lp;

    agen agen_dst_uv      = init(0);
    agen_dst_uv.lane_ofst = lp / chess_elementsof(dvchar);
    agen_dst_uv.round     = 0x80 | QBITS; // truncate
    // saturation in [0, 255]
    agen_dst_uv.sat_opt    = 2;
    agen_dst_uv.sat_lim_lo = 0;
    agen_dst_uv.sat_val_lo = 0;
    agen_dst_uv.sat_lim_hi = 255;
    agen_dst_uv.sat_val_hi = 255;
    INIT_AGEN3(agen_dst_uv, wrapper);

    cfg->outAgenCfgs[CHANNEL_U] = extract_agen_cfg(agen_dst_uv);
    cfg->outAgenCfgs[CHANNEL_V] = extract_agen_cfg(agen_dst_uv);
}

static inline void convert_image_format_src_rgb_agen_update(uint8_t *src_r, uint8_t *src_g, uint8_t *src_b,
                                                            ConvertImageFormatConfig *cfg)
{
    cupvaModifyAgenCfgBase(&(cfg->inAgenCfgs[CHANNEL_R]), src_r);
    cupvaModifyAgenCfgBase(&(cfg->inAgenCfgs[CHANNEL_G]), src_g);
    cupvaModifyAgenCfgBase(&(cfg->inAgenCfgs[CHANNEL_B]), src_b);
}

static inline void convert_image_format_src_yuv_agen_update(uint8_t *src_y, uint8_t *src_u, uint8_t *src_v,
                                                            ConvertImageFormatConfig *cfg)
{
    cupvaModifyAgenCfgBase(&(cfg->inAgenCfgs[CHANNEL_Y]), src_y);
    cupvaModifyAgenCfgBase(&(cfg->inAgenCfgs[CHANNEL_U]), src_u);
    cupvaModifyAgenCfgBase(&(cfg->inAgenCfgs[CHANNEL_V]), src_v);
}

static inline void convert_image_format_dst_rgb_agen_update(uint8_t *dst_r, uint8_t *dst_g, uint8_t *dst_b,
                                                            ConvertImageFormatConfig *cfg)
{
    cupvaModifyAgenCfgBase(&(cfg->outAgenCfgs[CHANNEL_R]), dst_r);
    cupvaModifyAgenCfgBase(&(cfg->outAgenCfgs[CHANNEL_G]), dst_g);
    cupvaModifyAgenCfgBase(&(cfg->outAgenCfgs[CHANNEL_B]), dst_b);
}

static inline void convert_image_format_dst_yuv_agen_update(uint8_t *dst_y, uint8_t *dst_u, uint8_t *dst_v,
                                                            ConvertImageFormatConfig *cfg)
{
    cupvaModifyAgenCfgBase(&(cfg->outAgenCfgs[CHANNEL_Y]), dst_y);
    cupvaModifyAgenCfgBase(&(cfg->outAgenCfgs[CHANNEL_U]), dst_u);
    cupvaModifyAgenCfgBase(&(cfg->outAgenCfgs[CHANNEL_V]), dst_v);
}

void convert_image_format_yuv2nv12_agen_init(int tw, int th, int lp_in, int lp_out, int channel_in, int channel_out,
                                             uint8_t *tmp_uv, ConvertImageFormatConfig *cfg)
{
    // config agen for YUV swizzling kernel

    const int vecw   = chess_elementsof(vchar);
    const int niter1 = tw / channel_in;
    const int niter2 = th / vecw;

    // src agens
    convert_image_format_yuv_agen_init(niter1, niter2, lp_in, channel_in, &cfg->inAgenCfgs[CHANNEL_Y],
                                       &cfg->inAgenCfgs[CHANNEL_U], &cfg->inAgenCfgs[CHANNEL_V]);

    // dst agens
    convert_image_format_yuv_agen_init(niter1, niter2, lp_out, channel_out, &cfg->outAgenCfgs[CHANNEL_Y],
                                       &cfg->outAgenCfgs[CHANNEL_U], &cfg->outAgenCfgs[CHANNEL_V]);
    cupvaModifyAgenCfgBase(&(cfg->outAgenCfgs[CHANNEL_U]), tmp_uv);
    cupvaModifyAgenCfgBase(&(cfg->outAgenCfgs[CHANNEL_V]), tmp_uv + 1);

    cfg->niter = niter1 * niter2;

    // config agen for UV plane 2x downsampling kernel in vertical direction

    const int vecw_uv   = chess_elementsof(dvchar);
    const int niter1_uv = tw / vecw_uv;
    const int niter2_uv = th / 2;

    convert_image_format_downsampling_agen_init(niter1_uv, niter2_uv, lp_out, cfg);
    cupvaModifyAgenCfgBase(&(cfg->inSamplingAgenCfg), tmp_uv);

    cfg->samplingNiter = niter1_uv * niter2_uv;
}

void convert_image_format_yuv2yuv_agen_init(int tw, int th, int lp_in, int lp_out, int channel_in, int channel_out,
                                            ConvertImageFormatConfig *cfg)
{
    // config agen for YUV swizzling kernel

    const int vecw   = chess_elementsof(vchar);
    const int niter1 = tw / channel_in;
    const int niter2 = th / vecw;

    // src agens
    convert_image_format_yuv_agen_init(niter1, niter2, lp_in, channel_in, &cfg->inAgenCfgs[CHANNEL_Y],
                                       &cfg->inAgenCfgs[CHANNEL_U], &cfg->inAgenCfgs[CHANNEL_V]);

    // dst agens
    convert_image_format_yuv_agen_init(niter1, niter2, lp_out, channel_out, &cfg->outAgenCfgs[CHANNEL_Y],
                                       &cfg->outAgenCfgs[CHANNEL_U], &cfg->outAgenCfgs[CHANNEL_V]);

    cfg->niter = niter1 * niter2;
}

void convert_image_format_rgba2rgba_agen_init(int tw, int th, int lp_in, int lp_out, int channel_in, int channel_out,
                                              uint8_t *src_a, uint8_t *dst_a, ConvertImageFormatConfig *cfg)
{
    // config agen for RGBA swizzling kernel

    const int vecw   = chess_elementsof(vchar);
    const int niter1 = tw / channel_in;
    const int niter2 = th / vecw;

    // src agens
    convert_image_format_rgba_agen_init(niter1, niter2, lp_in, channel_in, &cfg->inAgenCfgs[CHANNEL_R],
                                        &cfg->inAgenCfgs[CHANNEL_G], &cfg->inAgenCfgs[CHANNEL_B],
                                        &cfg->inAgenCfgs[CHANNEL_A]);
    cupvaModifyAgenCfgBase(&(cfg->inAgenCfgs[CHANNEL_A]), src_a);

    // dst agens
    convert_image_format_rgba_agen_init(niter1, niter2, lp_out, channel_out, &cfg->outAgenCfgs[CHANNEL_R],
                                        &cfg->outAgenCfgs[CHANNEL_G], &cfg->outAgenCfgs[CHANNEL_B],
                                        &cfg->outAgenCfgs[CHANNEL_A]);
    cupvaModifyAgenCfgBase(&(cfg->outAgenCfgs[CHANNEL_A]), dst_a);

    cfg->niter = niter1 * niter2;
}

void convert_image_format_nv122rgb_agen_init(int tw, int th, int lp_in, int lp_out, int channel_in, int channel_out,
                                             uint8_t *tmp_uv, ConvertImageFormatConfig *cfg)
{
    // config agen for YUV to RGBA conversion kernel

    const int vecw   = chess_elementsof(vchar);
    const int niter1 = tw / channel_in;
    const int niter2 = th / vecw;

    // src agens
    convert_image_format_src_yuv_agen_init(niter1, niter2, lp_in, channel_in, cfg);
    cupvaModifyAgenCfgBase(&(cfg->inAgenCfgs[CHANNEL_U]), tmp_uv);
    cupvaModifyAgenCfgBase(&(cfg->inAgenCfgs[CHANNEL_V]), tmp_uv + 1);

    // dst agens
    convert_image_format_dst_rgb_agen_init(niter1, niter2, lp_out, channel_out, cfg);

    // For packed 4-channel output (RGBA8/BGRA8) also configure the alpha lane
    // agen; alpha is filled with a constant 255 by alpha_fill_knl_exec.
    if (channel_out == 4)
    {
        convert_image_format_dst_alpha_agen_init(niter1, niter2, lp_out, channel_out, cfg);
    }

    cfg->niter = niter1 * niter2;

    // config agen for UV plane 2x upsampling kernel in vertical direction

    const int vecw_uv   = chess_elementsof(dvchar);
    const int niter1_uv = tw / vecw_uv;
    const int niter2_uv = th;

    convert_image_format_upsampling_agen_init(niter1_uv, niter2_uv, lp_in, cfg);
    cupvaModifyAgenCfgBase(&(cfg->outSamplingAgenCfg), tmp_uv);

    cfg->samplingNiter = niter1_uv * niter2_uv;
}

void convert_image_format_yuv2rgb_agen_init(int tw, int th, int lp_in, int lp_out, int channel_in, int channel_out,
                                            ConvertImageFormatConfig *cfg)
{
    // config agen for YUV to RGB conversion kernel

    const int vecw   = chess_elementsof(vchar);
    const int niter1 = tw / channel_in;
    const int niter2 = th / vecw;

    // src agens
    convert_image_format_src_yuv_agen_init(niter1, niter2, lp_in, channel_in, cfg);

    // dst agens
    convert_image_format_dst_rgb_agen_init(niter1, niter2, lp_out, channel_out, cfg);

    cfg->niter = niter1 * niter2;
}

void convert_image_format_rgb2nv12_agen_init(int tw, int th, int lp_in, int lp_out, int channel_in, int channel_out,
                                             uint8_t *tmp_uv, ConvertImageFormatConfig *cfg)
{
    // config agen for RGB to YUV conversion kernel

    const int vecw   = chess_elementsof(vchar);
    const int niter1 = tw / channel_in;
    const int niter2 = th / vecw;

    // src agens
    convert_image_format_src_rgb_agen_init(niter1, niter2, lp_in, channel_in, cfg);

    // dst agens
    convert_image_format_dst_yuv_agen_init(niter1, niter2, lp_out, channel_out, cfg);
    cupvaModifyAgenCfgBase(&(cfg->outAgenCfgs[CHANNEL_U]), tmp_uv);
    cupvaModifyAgenCfgBase(&(cfg->outAgenCfgs[CHANNEL_V]), tmp_uv + 1);

    cfg->niter = niter1 * niter2;

    // config agen for UV plane 2x downsampling kernel in vertical direction

    const int vecw_uv   = chess_elementsof(dvchar);
    const int niter1_uv = tw / vecw_uv;
    const int niter2_uv = th / 2;

    convert_image_format_downsampling_agen_init(niter1_uv, niter2_uv, lp_out, cfg);
    cupvaModifyAgenCfgBase(&(cfg->inSamplingAgenCfg), tmp_uv);

    cfg->samplingNiter = niter1_uv * niter2_uv;
}

void convert_image_format_rgb2yuv_agen_init(int tw, int th, int lp_in, int lp_out, int channel_in, int channel_out,
                                            ConvertImageFormatConfig *cfg)
{
    // config agen for RGB to YUV conversion kernel

    const int vecw   = chess_elementsof(vchar);
    const int niter1 = tw / channel_in;
    const int niter2 = th / vecw;

    // src agens
    convert_image_format_src_rgb_agen_init(niter1, niter2, lp_in, channel_in, cfg);

    // dst agens
    convert_image_format_dst_yuv_agen_init(niter1, niter2, lp_out, channel_out, cfg);

    cfg->niter = niter1 * niter2;
}

void convert_image_format_yuv2nv12_agen_update(uint8_t *src_y, uint8_t *src_u, uint8_t *src_v, uint8_t *dst_y,
                                               uint8_t *dst_uv, ConvertImageFormatConfig *cfg)
{
    // src agens
    convert_image_format_src_yuv_agen_update(src_y, src_u, src_v, cfg);

    // dst agens
    cupvaModifyAgenCfgBase(&(cfg->outAgenCfgs[CHANNEL_Y]), dst_y);

    // update agen base of output UV plane for downsampling kernel
    cupvaModifyAgenCfgBase(&(cfg->outSamplingAgenCfg), dst_uv);
}

void convert_image_format_yuv2yuv_agen_update(uint8_t *src_y, uint8_t *src_u, uint8_t *src_v, uint8_t *dst_y,
                                              uint8_t *dst_u, uint8_t *dst_v, ConvertImageFormatConfig *cfg)
{
    // src agens
    convert_image_format_src_yuv_agen_update(src_y, src_u, src_v, cfg);

    // dst agens
    convert_image_format_dst_yuv_agen_update(dst_y, dst_u, dst_v, cfg);
}

void convert_image_format_rgba2rgba_agen_update(uint8_t *src_r, uint8_t *src_g, uint8_t *src_b, uint8_t *src_a,
                                                uint8_t *dst_r, uint8_t *dst_g, uint8_t *dst_b, uint8_t *dst_a,
                                                ConvertImageFormatConfig *cfg)
{
    // src agens
    convert_image_format_src_rgb_agen_update(src_r, src_g, src_b, cfg);
    if (src_a)
    {
        cupvaModifyAgenCfgBase(&(cfg->inAgenCfgs[CHANNEL_A]), src_a);
    }

    // dst agens
    convert_image_format_dst_rgb_agen_update(dst_r, dst_g, dst_b, cfg);
    if (dst_a)
    {
        cupvaModifyAgenCfgBase(&(cfg->outAgenCfgs[CHANNEL_A]), dst_a);
    }
}

void convert_image_format_nv122rgb_agen_update(uint8_t *src_y, uint8_t *src_uv, uint8_t *dst_r, uint8_t *dst_g,
                                               uint8_t *dst_b, ConvertImageFormatConfig *cfg)
{
    // src agens
    cupvaModifyAgenCfgBase(&(cfg->inAgenCfgs[CHANNEL_Y]), src_y);

    // dst agens
    convert_image_format_dst_rgb_agen_update(dst_r, dst_g, dst_b, cfg);

    // update agen base of input UV plane for upsampling kernel
    cupvaModifyAgenCfgBase(&(cfg->inSamplingAgenCfg), src_uv);
}

void convert_image_format_nv122rgba_agen_update(uint8_t *src_y, uint8_t *src_uv, uint8_t *dst_r, uint8_t *dst_g,
                                                uint8_t *dst_b, uint8_t *dst_a, ConvertImageFormatConfig *cfg)
{
    // RGB channels + UV upsampling are identical to the 3-channel path.
    convert_image_format_nv122rgb_agen_update(src_y, src_uv, dst_r, dst_g, dst_b, cfg);

    // alpha lane (packed 4-channel output)
    cupvaModifyAgenCfgBase(&(cfg->outAgenCfgs[CHANNEL_A]), dst_a);
}

void convert_image_format_yuv2rgb_agen_update(uint8_t *src_y, uint8_t *src_u, uint8_t *src_v, uint8_t *dst_r,
                                              uint8_t *dst_g, uint8_t *dst_b, ConvertImageFormatConfig *cfg)
{
    // src agens
    convert_image_format_src_yuv_agen_update(src_y, src_u, src_v, cfg);

    // dst agens
    convert_image_format_dst_rgb_agen_update(dst_r, dst_g, dst_b, cfg);
}

void convert_image_format_rgb2nv12_agen_update(uint8_t *src_r, uint8_t *src_g, uint8_t *src_b, uint8_t *dst_y,
                                               uint8_t *dst_uv, ConvertImageFormatConfig *cfg)
{
    // src agens
    convert_image_format_src_rgb_agen_update(src_r, src_g, src_b, cfg);

    // dst agens
    cupvaModifyAgenCfgBase(&(cfg->outAgenCfgs[CHANNEL_Y]), dst_y);

    // update agen base of output UV plane for downsampling kernel
    cupvaModifyAgenCfgBase(&(cfg->outSamplingAgenCfg), dst_uv);
}

void convert_image_format_rgb2yuv_agen_update(uint8_t *src_r, uint8_t *src_g, uint8_t *src_b, uint8_t *dst_y,
                                              uint8_t *dst_u, uint8_t *dst_v, ConvertImageFormatConfig *cfg)
{
    // src agens
    convert_image_format_src_rgb_agen_update(src_r, src_g, src_b, cfg);

    // dst agens
    convert_image_format_dst_yuv_agen_update(dst_y, dst_u, dst_v, cfg);
}

void convert_image_format_sampling_knl_exec(ConvertImageFormatConfig *cfg)
{
    int niter = cfg->samplingNiter;

    agen in  = init_agen_from_cfg(cfg->inSamplingAgenCfg);
    agen out = init_agen_from_cfg(cfg->outSamplingAgenCfg);

    for (int i = 0; i < niter; i++) chess_loop_range(3, )
    chess_unroll_loop(8)
    chess_prepare_for_pipelining
    {
        dvcharx dv0 = dvuchar_load(in);

        vstore(dv0, out);
    }
}

void convert_image_format_yuv2yuv_knl_exec(ConvertImageFormatConfig *cfg)
{
    int niter = cfg->niter / 2;

    agen_B in0  = init_agen_B_from_cfg(cfg->inAgenCfgs[CHANNEL_Y]);
    agen_B in1  = init_agen_B_from_cfg(cfg->inAgenCfgs[CHANNEL_U]);
    agen_B in2  = init_agen_B_from_cfg(cfg->inAgenCfgs[CHANNEL_V]);
    agen_C out0 = init_agen_C_from_cfg(cfg->outAgenCfgs[CHANNEL_Y]);
    agen_C out1 = init_agen_C_from_cfg(cfg->outAgenCfgs[CHANNEL_U]);
    agen_C out2 = init_agen_C_from_cfg(cfg->outAgenCfgs[CHANNEL_V]);

    for (int i = 0; i < niter; i++) chess_loop_range(3, )
    chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        vcharx y0 = vuchar_load_transp(in0);
        vcharx y1 = vuchar_load_transp(in0);
        vcharx u  = vuchar_load_transp(in1);
        vcharx v  = vuchar_load_transp(in2);

        vstore_transp(y0, out0);
        vstore_transp(y1, out0);
        vstore_transp(u, out1);
        vstore_transp(v, out2);
    }
}

void convert_image_format_rgba2rgba_knl_exec(ConvertImageFormatConfig *cfg)
{
    int niter = cfg->niter;

    agen_B in0  = init_agen_B_from_cfg(cfg->inAgenCfgs[CHANNEL_R]);
    agen_B in1  = init_agen_B_from_cfg(cfg->inAgenCfgs[CHANNEL_G]);
    agen_B in2  = init_agen_B_from_cfg(cfg->inAgenCfgs[CHANNEL_B]);
    agen_B in3  = init_agen_B_from_cfg(cfg->inAgenCfgs[CHANNEL_A]);
    agen_C out0 = init_agen_C_from_cfg(cfg->outAgenCfgs[CHANNEL_R]);
    agen_C out1 = init_agen_C_from_cfg(cfg->outAgenCfgs[CHANNEL_G]);
    agen_C out2 = init_agen_C_from_cfg(cfg->outAgenCfgs[CHANNEL_B]);
    agen_C out3 = init_agen_C_from_cfg(cfg->outAgenCfgs[CHANNEL_A]);

    for (int i = 0; i < niter; i++) chess_loop_range(3, )
    chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        vcharx r = vuchar_load_transp(in0);
        vcharx g = vuchar_load_transp(in1);
        vcharx b = vuchar_load_transp(in2);
        vcharx a = vuchar_load_transp(in3);

        vstore_transp(r, out0);
        vstore_transp(g, out1);
        vstore_transp(b, out2);
        vstore_transp(a, out3);
    }
}

// Packed 4-channel (RGBA8/BGRA8) → packed 3-channel (RGB8/BGR8): copy R,G,B and
// drop the alpha lane. Same as rgba2rgba_knl_exec without the alpha load/store.
void convert_image_format_rgb2rgb_knl_exec(ConvertImageFormatConfig *cfg)
{
    int niter = cfg->niter;

    agen_B in0  = init_agen_B_from_cfg(cfg->inAgenCfgs[CHANNEL_R]);
    agen_B in1  = init_agen_B_from_cfg(cfg->inAgenCfgs[CHANNEL_G]);
    agen_B in2  = init_agen_B_from_cfg(cfg->inAgenCfgs[CHANNEL_B]);
    agen_C out0 = init_agen_C_from_cfg(cfg->outAgenCfgs[CHANNEL_R]);
    agen_C out1 = init_agen_C_from_cfg(cfg->outAgenCfgs[CHANNEL_G]);
    agen_C out2 = init_agen_C_from_cfg(cfg->outAgenCfgs[CHANNEL_B]);

    for (int i = 0; i < niter; i++) chess_loop_range(3, )
    chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        vcharx r = vuchar_load_transp(in0);
        vcharx g = vuchar_load_transp(in1);
        vcharx b = vuchar_load_transp(in2);

        vstore_transp(r, out0);
        vstore_transp(g, out1);
        vstore_transp(b, out2);
    }
}

// Fill the alpha lane of a packed 4-channel output with the constant 255.
// Run after yuv2rgb_knl_exec on the NV12(_ER) → RGBA8 path.
void convert_image_format_alpha_fill_knl_exec(ConvertImageFormatConfig *cfg)
{
    int niter = cfg->niter;

    agen_C out3 = init_agen_C_from_cfg(cfg->outAgenCfgs[CHANNEL_A]);

    vcharx a = replicateb(ALPHA_VALUE);

    for (int i = 0; i < niter; i++) chess_loop_range(3, )
    chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        vstore_transp(a, out3);
    }
}

void convert_image_format_yuv2rgb_knl_exec(ConvertImageFormatConfig *cfg, uint8_t *coeff_r, int8_t *coeff_g,
                                           uint8_t *coeff_b)
{
    int niter = cfg->niter;

    agen_B in0  = init_agen_B_from_cfg(cfg->inAgenCfgs[CHANNEL_Y]);
    agen_B in1  = init_agen_B_from_cfg(cfg->inAgenCfgs[CHANNEL_U]);
    agen_B in2  = init_agen_B_from_cfg(cfg->inAgenCfgs[CHANNEL_V]);
    agen_C out0 = init_agen_C_from_cfg(cfg->outAgenCfgs[CHANNEL_R]);
    agen_C out1 = init_agen_C_from_cfg(cfg->outAgenCfgs[CHANNEL_G]);
    agen_C out2 = init_agen_C_from_cfg(cfg->outAgenCfgs[CHANNEL_B]);

    dvshortx r_out, b_out, g_out;
    dvcharx wt_rb, yu, vx;
    vcharx wt_g;

    wt_rb.lo = zero_extend(*(vuchar *)coeff_r);
    wt_rb.hi = zero_extend(*(vuchar *)coeff_b);
    wt_g     = sign_extend(*(vchar *)coeff_g);

    vx.lo = replicateb(0);
    vx.hi = replicateb(1);

    /* Studio-range BT.601 subtracts 16 from Y; NV12_ER (full range) subtracts
     * 0. Set by the device top-level per conversion code. */
    const int lumaOffset = cfg->lumaOffset;

    chess_separator_scheduler();

    for (int i = 0; i < niter; i++) chess_loop_range(3, )
    chess_unroll_loop(4)
    chess_prepare_for_pipelining
    {
        vcharx y = vuchar_load_transp(in0);
        vcharx u = vuchar_load_transp(in1);
        vcharx v = vuchar_load_transp(in2);

        yu.lo = y - lumaOffset;
        yu.hi = u - COLOR_OFFSET_C;
        vx.lo = v - COLOR_OFFSET_C;

        vdotp4x2_bbh(yu, vx, wt_rb, r_out, b_out, 0, r_out, b_out);
        g_out = vdotp4_bbh(yu, vx, wt_g, g_out, 0);

        vstore_transp_i_hb(r_out, out0);
        vstore_transp_i_hb(g_out, out1);
        vstore_transp_i_hb(b_out, out2);
    }
}

void convert_image_format_rgb2yuv_knl_exec(ConvertImageFormatConfig *cfg, uint8_t *coeff_y, int8_t *coeff_u,
                                           int8_t *coeff_v)
{
    int niter = cfg->niter;

    agen_B in0  = init_agen_B_from_cfg(cfg->inAgenCfgs[CHANNEL_R]);
    agen_B in1  = init_agen_B_from_cfg(cfg->inAgenCfgs[CHANNEL_G]);
    agen_B in2  = init_agen_B_from_cfg(cfg->inAgenCfgs[CHANNEL_B]);
    agen_C out0 = init_agen_C_from_cfg(cfg->outAgenCfgs[CHANNEL_Y]);
    agen_C out1 = init_agen_C_from_cfg(cfg->outAgenCfgs[CHANNEL_U]);
    agen_C out2 = init_agen_C_from_cfg(cfg->outAgenCfgs[CHANNEL_V]);

    dvshortx y_out, u_out, v_out;
    dvcharx wt_uv, rg, bx;
    vcharx wt_y;
    int pred = -1;

    wt_y     = zero_extend(*(vuchar *)coeff_y);
    wt_uv.lo = sign_extend(*(vchar *)coeff_u);
    wt_uv.hi = sign_extend(*(vchar *)coeff_v);

    bx.lo = replicateb(0);
    bx.hi = replicateb(1 + cfg->lumaOffset * (QBITS_SCALE / QBITS_SCALE_HALF));

    /* Loop-invariant chroma bias. bx.hi already injects lumaOffset*QBITS_SCALE
     * into U/V, so the net offset is lumaOffset + (COLOR_OFFSET_C - lumaOffset)
     * = COLOR_OFFSET_C for both studio (lumaOffset=16) and NV12_ER (lumaOffset=0).
     * Hoisted out of the loop so cfg->lumaOffset is not reloaded each iteration. */
    const int chromaBias = (COLOR_OFFSET_C - cfg->lumaOffset) * QBITS_SCALE;

    chess_separator_scheduler();

    for (int i = 0; i < niter; i++) chess_loop_range(3, )
    chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        rg.lo = vuchar_load_transp(in0);
        rg.hi = vuchar_load_transp(in1);
        bx.lo = vuchar_load_transp(in2);

        y_out = vdotp4_bbh(rg, bx, wt_y, y_out, 0);
        vdotp4x2_bbh(rg, bx, wt_uv, u_out, v_out, 0, u_out, v_out);

        u_out = u_out + chromaBias;
        v_out = v_out + chromaBias;

        pred = mux(i & 1, 0, -1);
        vstore_transp_i_hb(y_out, out0);
        vstore_transp_i_hb(u_out, out1, pred);
        vstore_transp_i_hb(v_out, out2, pred);
    }
}
