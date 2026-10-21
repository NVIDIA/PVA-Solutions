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

VMEM_RDF_UNIFIED(A, srcHdl0);
VMEM_RDF_UNIFIED(A, srcHdl1);
VMEM_RDF_UNIFIED(A, srcHdl2);
VMEM_RDF_UNIFIED(A, srcHdl3);
VMEM_RDF_UNIFIED(A, dstHdl0);
VMEM_RDF_UNIFIED(A, dstHdl1);
VMEM_RDF_UNIFIED(A, dstHdl2);
VMEM_RDF_UNIFIED(A, dstHdl3);

VMEM(A, int32_t, niterX);
VMEM(A, int32_t, niterY);
VMEM(A, ImageFormatConversionCode, code);

VMEM(A, int32_t, inPlaneCount);
VMEM(A, int32_t, inChannelCount);
VMEM(A, int32_t, inTileWidth);
VMEM(A, int32_t, outPlaneCount);
VMEM(A, int32_t, outChannelCount);
VMEM(A, int32_t, outTileWidth);
VMEM(A, int32_t, tileHeight);

VMEM(A, ConvertImageFormatConfig, cfg);

VMEM(A, uint8_t, coeffR, chess_elementsof(vuchar));
VMEM(A, int8_t, coeffG, chess_elementsof(vchar));
VMEM(A, uint8_t, coeffB, chess_elementsof(vuchar));
VMEM(A, uint8_t, coeffY, chess_elementsof(vuchar));
VMEM(A, int8_t, coeffU, chess_elementsof(vchar));
VMEM(A, int8_t, coeffV, chess_elementsof(vchar));

// buffer allocation for UV plane up/down sampling
VMEM(A, uint8_t, tmpBuf, RDF_SINGLE(uint8_t, TILE_W *(MAX_CHANNEL_COUNT / 2), TILE_H, 0, 0, TRANS_MODE_1));
// buffer allocation for 1st plane of src image
VMEM(B, uint8_t, srcBuf0, RDF_DOUBLE(uint8_t, TILE_W *MAX_CHANNEL_COUNT, TILE_H, 0, 0, TRANS_MODE_1));
// buffer allocation for 2nd plane of src image
VMEM(B, uint8_t, srcBuf1, RDF_DOUBLE(uint8_t, TILE_W *(MAX_CHANNEL_COUNT / 2), TILE_H, 0, 0, TRANS_MODE_1));
// buffer allocation for 3rd plane of src image
VMEM(B, uint8_t, srcBuf2, RDF_DOUBLE(uint8_t, TILE_W *(MAX_CHANNEL_COUNT / 3), TILE_H, 0, 0, TRANS_MODE_1));
// buffer allocation for 4th plane of src image
VMEM(A, uint8_t, srcBuf3, RDF_DOUBLE(uint8_t, TILE_W *(MAX_CHANNEL_COUNT / 4), TILE_H, 0, 0, TRANS_MODE_1));
// buffer allocation for 1st plane of dst image
VMEM(C, uint8_t, dstBuf0, RDF_DOUBLE(uint8_t, TILE_W *MAX_CHANNEL_COUNT, TILE_H, 0, 0, TRANS_MODE_1));
// buffer allocation for 2nd plane of dst image
VMEM(C, uint8_t, dstBuf1, RDF_DOUBLE(uint8_t, TILE_W *(MAX_CHANNEL_COUNT / 2), TILE_H, 0, 0, TRANS_MODE_1));
// buffer allocation for 3rd plane of dst image
VMEM(C, uint8_t, dstBuf2, RDF_DOUBLE(uint8_t, TILE_W *(MAX_CHANNEL_COUNT / 3), TILE_H, 0, 0, TRANS_MODE_1));
// buffer allocation for 4th plane of dst image
VMEM(A, uint8_t, dstBuf3, RDF_DOUBLE(uint8_t, TILE_W *(MAX_CHANNEL_COUNT / 4), TILE_H, 0, 0, TRANS_MODE_1));

static void yuv2nv12ExecFunc(uint8_t *inPtrs[MAX_CHANNEL_COUNT], uint8_t *outPtrs[MAX_CHANNEL_COUNT],
                             ConvertImageFormatConfig *cfg)
{
    convert_image_format_yuv2nv12_agen_update(inPtrs[0], inPtrs[1], inPtrs[2], outPtrs[0], outPtrs[1], cfg);
    convert_image_format_yuv2yuv_knl_exec(cfg);
    convert_image_format_sampling_knl_exec(cfg);
}

static void yuv2yuvExecFunc(uint8_t *inPtrs[MAX_CHANNEL_COUNT], uint8_t *outPtrs[MAX_CHANNEL_COUNT],
                            ConvertImageFormatConfig *cfg)
{
    convert_image_format_yuv2yuv_agen_update(inPtrs[0], inPtrs[1], inPtrs[2], outPtrs[0], outPtrs[1], outPtrs[2], cfg);
    convert_image_format_yuv2yuv_knl_exec(cfg);
}

static void nv122rgbExecFunc(uint8_t *inPtrs[MAX_CHANNEL_COUNT], uint8_t *outPtrs[MAX_CHANNEL_COUNT],
                             ConvertImageFormatConfig *cfg)
{
    convert_image_format_nv122rgb_agen_update(inPtrs[0], inPtrs[1], outPtrs[0], outPtrs[1], outPtrs[2], cfg);
    convert_image_format_sampling_knl_exec(cfg);
    convert_image_format_yuv2rgb_knl_exec(cfg, coeffR, coeffG, coeffB);
}

static void yuv2rgbExecFunc(uint8_t *inPtrs[MAX_CHANNEL_COUNT], uint8_t *outPtrs[MAX_CHANNEL_COUNT],
                            ConvertImageFormatConfig *cfg)
{
    convert_image_format_yuv2rgb_agen_update(inPtrs[0], inPtrs[1], inPtrs[2], outPtrs[0], outPtrs[1], outPtrs[2], cfg);
    convert_image_format_yuv2rgb_knl_exec(cfg, coeffR, coeffG, coeffB);
}

static void rgba2rgbaExecFunc(uint8_t *inPtrs[MAX_CHANNEL_COUNT], uint8_t *outPtrs[MAX_CHANNEL_COUNT],
                              ConvertImageFormatConfig *cfg)
{
    convert_image_format_rgba2rgba_agen_update(inPtrs[0], inPtrs[1], inPtrs[2], inPtrs[3], outPtrs[0], outPtrs[1],
                                               outPtrs[2], outPtrs[3], cfg);
    convert_image_format_rgba2rgba_knl_exec(cfg);
}

static void nv122rgbaExecFunc(uint8_t *inPtrs[MAX_CHANNEL_COUNT], uint8_t *outPtrs[MAX_CHANNEL_COUNT],
                              ConvertImageFormatConfig *cfg)
{
    convert_image_format_nv122rgba_agen_update(inPtrs[0], inPtrs[1], outPtrs[0], outPtrs[1], outPtrs[2], outPtrs[3],
                                               cfg);
    convert_image_format_sampling_knl_exec(cfg);
    convert_image_format_yuv2rgb_knl_exec(cfg, coeffR, coeffG, coeffB);
    convert_image_format_alpha_fill_knl_exec(cfg);
}

static void rgb2rgbExecFunc(uint8_t *inPtrs[MAX_CHANNEL_COUNT], uint8_t *outPtrs[MAX_CHANNEL_COUNT],
                            ConvertImageFormatConfig *cfg)
{
    // Packed 4-channel → packed 3-channel (alpha dropped). inPtrs[3]/outPtrs[3]
    // are NULL so the shared update helper skips the alpha lane.
    convert_image_format_rgba2rgba_agen_update(inPtrs[0], inPtrs[1], inPtrs[2], inPtrs[3], outPtrs[0], outPtrs[1],
                                               outPtrs[2], outPtrs[3], cfg);
    convert_image_format_rgb2rgb_knl_exec(cfg);
}

static void rgb2nv12ExecFunc(uint8_t *inPtrs[MAX_CHANNEL_COUNT], uint8_t *outPtrs[MAX_CHANNEL_COUNT],
                             ConvertImageFormatConfig *cfg)
{
    convert_image_format_rgb2nv12_agen_update(inPtrs[0], inPtrs[1], inPtrs[2], outPtrs[0], outPtrs[1], cfg);
    convert_image_format_rgb2yuv_knl_exec(cfg, coeffY, coeffU, coeffV);
    convert_image_format_sampling_knl_exec(cfg);
}

static void rgb2yuvExecFunc(uint8_t *inPtrs[MAX_CHANNEL_COUNT], uint8_t *outPtrs[MAX_CHANNEL_COUNT],
                            ConvertImageFormatConfig *cfg)
{
    convert_image_format_rgb2yuv_agen_update(inPtrs[0], inPtrs[1], inPtrs[2], outPtrs[0], outPtrs[1], outPtrs[2], cfg);
    convert_image_format_rgb2yuv_knl_exec(cfg, coeffY, coeffU, coeffV);
}

/* Checks whether a conversion code is an NV12_ER (full-range
 * YUV) variant. */
static int isErCode(ImageFormatConversionCode code)
{
    switch (code)
    {
    case CONVERT_IMAGE_FORMAT_NV12_ER2BGR8:
    case CONVERT_IMAGE_FORMAT_NV12_ER2RGB8:
    case CONVERT_IMAGE_FORMAT_NV12_ER2BGR8p:
    case CONVERT_IMAGE_FORMAT_NV12_ER2RGB8p:
    case CONVERT_IMAGE_FORMAT_NV12_ER2RGBA8:
    case CONVERT_IMAGE_FORMAT_BGR82NV12_ER:
    case CONVERT_IMAGE_FORMAT_RGB82NV12_ER:
    case CONVERT_IMAGE_FORMAT_BGR8p2NV12_ER:
    case CONVERT_IMAGE_FORMAT_RGB8p2NV12_ER:
    case CONVERT_IMAGE_FORMAT_YUYV2NV12_ER:
    case CONVERT_IMAGE_FORMAT_UYVY2NV12_ER:
    case CONVERT_IMAGE_FORMAT_VYUY2NV12_ER:
    case CONVERT_IMAGE_FORMAT_YUV8p2NV12_ER:
        return true;
    default:
        return false;
    }
}

CUPVA_VPU_MAIN()
{
    UnifiedRDFHandler *srcHdls[] = {&srcHdl0, &srcHdl1, &srcHdl2, &srcHdl3};
    UnifiedRDFHandler *dstHdls[] = {&dstHdl0, &dstHdl1, &dstHdl2, &dstHdl3};

    uint8_t *srcBufs[] = {srcBuf0, srcBuf1, srcBuf2, srcBuf3};
    uint8_t *dstBufs[] = {dstBuf0, dstBuf1, dstBuf2, dstBuf3};

    for (int32_t p = 0; p < inPlaneCount; p++)
    {
        cupvaRasterDataFlowOpen(*srcHdls[p], &srcBufs[p][0]);
    }
    for (int32_t p = 0; p < outPlaneCount; p++)
    {
        cupvaRasterDataFlowOpen(*dstHdls[p], &dstBufs[p][0]);
    }

    int32_t srcLinePitch = cupvaRasterDataFlowGetLinePitch(srcHdl0);
    int32_t dstLinePitch = cupvaRasterDataFlowGetLinePitch(dstHdl0);

    void (*execFunc)(uint8_t *inPtrs[MAX_CHANNEL_COUNT], uint8_t *outPtrs[MAX_CHANNEL_COUNT],
                     ConvertImageFormatConfig *cfg);
    switch (code)
    {
    case CONVERT_IMAGE_FORMAT_YUYV2NV12:
    case CONVERT_IMAGE_FORMAT_UYVY2NV12:
    case CONVERT_IMAGE_FORMAT_VYUY2NV12:
    case CONVERT_IMAGE_FORMAT_YUYV2NV12_ER:
    case CONVERT_IMAGE_FORMAT_UYVY2NV12_ER:
    case CONVERT_IMAGE_FORMAT_VYUY2NV12_ER:
        /* YUV → NV12 chroma resampling; no YUV→RGB conversion happens here,
         * so the studio/ER distinction is irrelevant at the pixel-data
         * level. The output is range-tagged in the host's image format
         * descriptor. */
        convert_image_format_yuv2nv12_agen_init(inTileWidth, tileHeight, srcLinePitch, dstLinePitch, 2, 1, tmpBuf,
                                                &cfg);
        execFunc = yuv2nv12ExecFunc;
        break;
    case CONVERT_IMAGE_FORMAT_YUV8p2NV12:
    case CONVERT_IMAGE_FORMAT_YUV8p2NV12_ER:
        convert_image_format_yuv2nv12_agen_init(inTileWidth, tileHeight, srcLinePitch, dstLinePitch, 1, 1, tmpBuf,
                                                &cfg);
        execFunc = yuv2nv12ExecFunc;
        break;
    case CONVERT_IMAGE_FORMAT_YUV8p2YUYV:
    case CONVERT_IMAGE_FORMAT_YUV8p2UYVY:
    case CONVERT_IMAGE_FORMAT_YUV8p2VYUY:
        convert_image_format_yuv2yuv_agen_init(inTileWidth, tileHeight, srcLinePitch, dstLinePitch, 1, 2, &cfg);
        execFunc = yuv2yuvExecFunc;
        break;
    case CONVERT_IMAGE_FORMAT_NV122BGR8:
    case CONVERT_IMAGE_FORMAT_NV122RGB8:
    case CONVERT_IMAGE_FORMAT_NV12_ER2BGR8:
    case CONVERT_IMAGE_FORMAT_NV12_ER2RGB8:
        convert_image_format_nv122rgb_agen_init(inTileWidth, tileHeight, srcLinePitch, dstLinePitch, 1, 3, tmpBuf,
                                                &cfg);
        execFunc = nv122rgbExecFunc;
        break;
    case CONVERT_IMAGE_FORMAT_NV12_ER2RGBA8:
        /* NV12_ER → packed 4-channel RGBA8: same YUV→RGB path with channel_out
         * = 4; the alpha lane is filled with 255 by alpha_fill_knl_exec. */
        convert_image_format_nv122rgb_agen_init(inTileWidth, tileHeight, srcLinePitch, dstLinePitch, 1, 4, tmpBuf,
                                                &cfg);
        execFunc = nv122rgbaExecFunc;
        break;
    case CONVERT_IMAGE_FORMAT_NV122BGR8p:
    case CONVERT_IMAGE_FORMAT_NV122RGB8p:
    case CONVERT_IMAGE_FORMAT_NV12_ER2BGR8p:
    case CONVERT_IMAGE_FORMAT_NV12_ER2RGB8p:
        convert_image_format_nv122rgb_agen_init(inTileWidth, tileHeight, srcLinePitch, dstLinePitch, 1, 1, tmpBuf,
                                                &cfg);
        execFunc = nv122rgbExecFunc;
        break;
    case CONVERT_IMAGE_FORMAT_YUYV2BGR8:
    case CONVERT_IMAGE_FORMAT_YUYV2RGB8:
    case CONVERT_IMAGE_FORMAT_UYVY2BGR8:
    case CONVERT_IMAGE_FORMAT_UYVY2RGB8:
    case CONVERT_IMAGE_FORMAT_VYUY2BGR8:
    case CONVERT_IMAGE_FORMAT_VYUY2RGB8:
        convert_image_format_yuv2rgb_agen_init(inTileWidth, tileHeight, srcLinePitch, dstLinePitch, 2, 3, &cfg);
        execFunc = yuv2rgbExecFunc;
        break;
    case CONVERT_IMAGE_FORMAT_YUYV2BGR8p:
    case CONVERT_IMAGE_FORMAT_YUYV2RGB8p:
    case CONVERT_IMAGE_FORMAT_UYVY2BGR8p:
    case CONVERT_IMAGE_FORMAT_UYVY2RGB8p:
    case CONVERT_IMAGE_FORMAT_VYUY2BGR8p:
    case CONVERT_IMAGE_FORMAT_VYUY2RGB8p:
        convert_image_format_yuv2rgb_agen_init(inTileWidth, tileHeight, srcLinePitch, dstLinePitch, 2, 1, &cfg);
        execFunc = yuv2rgbExecFunc;
        break;
    case CONVERT_IMAGE_FORMAT_RGB82BGRA8:
    case CONVERT_IMAGE_FORMAT_RGB82RGBA8:
    case CONVERT_IMAGE_FORMAT_BGR82BGRA8:
        /* 3-channel packed → 4-channel packed with constant alpha = 255.
         * Same kernel as RGB→BGRA; the inPtrs/outPtrs setup below picks
         * the right per-channel byte ordering for each variant. */
        generate_alpha(tmpBuf);
        convert_image_format_rgba2rgba_agen_init(inTileWidth, tileHeight, srcLinePitch, dstLinePitch, 3, 4, tmpBuf,
                                                 dstBuf0 + 3, &cfg);
        execFunc = rgba2rgbaExecFunc;
        break;
    case CONVERT_IMAGE_FORMAT_RGBA82RGB8:
    case CONVERT_IMAGE_FORMAT_BGRA82BGR8:
        /* 4-channel packed → 3-channel packed; the alpha lane is dropped.
         * channel_in = 4, channel_out = 3; src_a/dst_a bases are unused by
         * rgb2rgb_knl_exec but kept non-NULL for the shared agen_init. */
        convert_image_format_rgba2rgba_agen_init(inTileWidth, tileHeight, srcLinePitch, dstLinePitch, 4, 3, tmpBuf,
                                                 dstBuf0 + 3, &cfg);
        execFunc = rgb2rgbExecFunc;
        break;
    case CONVERT_IMAGE_FORMAT_BGR82NV12:
    case CONVERT_IMAGE_FORMAT_RGB82NV12:
    case CONVERT_IMAGE_FORMAT_BGR82NV12_ER:
    case CONVERT_IMAGE_FORMAT_RGB82NV12_ER:
        convert_image_format_rgb2nv12_agen_init(inTileWidth, tileHeight, srcLinePitch, dstLinePitch, 3, 1, tmpBuf,
                                                &cfg);
        execFunc = rgb2nv12ExecFunc;
        break;
    case CONVERT_IMAGE_FORMAT_BGR8p2NV12:
    case CONVERT_IMAGE_FORMAT_RGB8p2NV12:
    case CONVERT_IMAGE_FORMAT_BGR8p2NV12_ER:
    case CONVERT_IMAGE_FORMAT_RGB8p2NV12_ER:
        convert_image_format_rgb2nv12_agen_init(inTileWidth, tileHeight, srcLinePitch, dstLinePitch, 1, 1, tmpBuf,
                                                &cfg);
        execFunc = rgb2nv12ExecFunc;
        break;
    case CONVERT_IMAGE_FORMAT_BGR82YUYV:
    case CONVERT_IMAGE_FORMAT_RGB82YUYV:
    case CONVERT_IMAGE_FORMAT_BGR82UYVY:
    case CONVERT_IMAGE_FORMAT_RGB82UYVY:
    case CONVERT_IMAGE_FORMAT_BGR82VYUY:
    case CONVERT_IMAGE_FORMAT_RGB82VYUY:
        convert_image_format_rgb2yuv_agen_init(inTileWidth, tileHeight, srcLinePitch, dstLinePitch, 3, 2, &cfg);
        execFunc = rgb2yuvExecFunc;
        break;
    case CONVERT_IMAGE_FORMAT_BGR8p2YUYV:
    case CONVERT_IMAGE_FORMAT_RGB8p2YUYV:
    case CONVERT_IMAGE_FORMAT_BGR8p2UYVY:
    case CONVERT_IMAGE_FORMAT_RGB8p2UYVY:
    case CONVERT_IMAGE_FORMAT_BGR8p2VYUY:
    case CONVERT_IMAGE_FORMAT_RGB8p2VYUY:
        convert_image_format_rgb2yuv_agen_init(inTileWidth, tileHeight, srcLinePitch, dstLinePitch, 1, 2, &cfg);
        execFunc = rgb2yuvExecFunc;
        break;
    default:
        break;
    }

    /* Default: studio-range BT.601 coefficients. */
    uint8_t coeff_r[] = {REPLICATE_X8(COEFF_R_FIXED)};
    int8_t coeff_g[]  = {REPLICATE_X8(COEFF_G_FIXED)};
    uint8_t coeff_b[] = {REPLICATE_X8(COEFF_B_FIXED)};
    uint8_t coeff_y[] = {REPLICATE_X8(COEFF_Y_FIXED)};
    int8_t coeff_u[]  = {REPLICATE_X8(COEFF_U_FIXED)};
    int8_t coeff_v[]  = {REPLICATE_X8(COEFF_V_FIXED)};

    /* Luma offset: studio-range BT.601 uses +16/-16 on Y; NV12_ER full-range
     * uses 0. */
    cfg.lumaOffset = isErCode(code) ? 0 : COLOR_OFFSET_L;

    /* For NV12_ER conversions, override with the ER color matrix (chroma
     * coefficients differ slightly). The coefficients are direction-specific:
     * decode overrides R/G/B (yuv2rgb), encode overrides Y/U/V (rgb2yuv). */
    switch (code)
    {
    case CONVERT_IMAGE_FORMAT_NV12_ER2BGR8:
    case CONVERT_IMAGE_FORMAT_NV12_ER2RGB8:
    case CONVERT_IMAGE_FORMAT_NV12_ER2BGR8p:
    case CONVERT_IMAGE_FORMAT_NV12_ER2RGB8p:
    case CONVERT_IMAGE_FORMAT_NV12_ER2RGBA8:
    {
        uint8_t er_r[] = {REPLICATE_X8(COEFF_R_ER_FIXED)};
        int8_t er_g[]  = {REPLICATE_X8(COEFF_G_ER_FIXED)};
        uint8_t er_b[] = {REPLICATE_X8(COEFF_B_ER_FIXED)};
        for (int i = 0; i < (int)sizeof(coeff_r); ++i)
        {
            coeff_r[i] = er_r[i];
        }
        for (int i = 0; i < (int)sizeof(coeff_g); ++i)
        {
            coeff_g[i] = er_g[i];
        }
        for (int i = 0; i < (int)sizeof(coeff_b); ++i)
        {
            coeff_b[i] = er_b[i];
        }
        break;
    }
    case CONVERT_IMAGE_FORMAT_BGR82NV12_ER:
    case CONVERT_IMAGE_FORMAT_RGB82NV12_ER:
    case CONVERT_IMAGE_FORMAT_BGR8p2NV12_ER:
    case CONVERT_IMAGE_FORMAT_RGB8p2NV12_ER:
    {
        uint8_t er_y[] = {REPLICATE_X8(COEFF_Y_ER_FIXED)};
        int8_t er_u[]  = {REPLICATE_X8(COEFF_U_ER_FIXED)};
        int8_t er_v[]  = {REPLICATE_X8(COEFF_V_ER_FIXED)};
        for (int i = 0; i < (int)sizeof(coeff_y); ++i)
        {
            coeff_y[i] = er_y[i];
        }
        for (int i = 0; i < (int)sizeof(coeff_u); ++i)
        {
            coeff_u[i] = er_u[i];
        }
        for (int i = 0; i < (int)sizeof(coeff_v); ++i)
        {
            coeff_v[i] = er_v[i];
        }
        break;
    }
    default:
        break;
    }

    *(vuchar *)coeffR = *(vuchar *)coeff_r;
    *(vchar *)coeffG  = *(vchar *)coeff_g;
    *(vuchar *)coeffB = *(vuchar *)coeff_b;
    *(vuchar *)coeffY = *(vuchar *)coeff_y;
    *(vchar *)coeffU  = *(vchar *)coeff_u;
    *(vchar *)coeffV  = *(vchar *)coeff_v;

    uint8_t *inTiles[MAX_CHANNEL_COUNT], *outTiles[MAX_CHANNEL_COUNT], *inPtrs[MAX_CHANNEL_COUNT],
        *outPtrs[MAX_CHANNEL_COUNT];
    for (int32_t y = 0; y < niterY; y++)
    {
        for (int32_t x = 0; x < niterX; x++)
        {
            for (int32_t p = 0; p < inPlaneCount; p++)
            {
                inTiles[p] = (uint8_t *)cupvaRasterDataFlowAcquire(*srcHdls[p]);
            }
            for (int32_t p = 0; p < outPlaneCount; p++)
            {
                outTiles[p] = (uint8_t *)cupvaRasterDataFlowAcquire(*dstHdls[p]);
            }

            if (code == CONVERT_IMAGE_FORMAT_YUYV2NV12 || code == CONVERT_IMAGE_FORMAT_YUYV2NV12_ER)
            {
                inPtrs[0]  = inTiles[0];     // y
                inPtrs[1]  = inTiles[0] + 1; // u
                inPtrs[2]  = inTiles[0] + 3; // v
                outPtrs[0] = outTiles[0];    // y
                outPtrs[1] = outTiles[1];    // uv
            }
            else if (code == CONVERT_IMAGE_FORMAT_UYVY2NV12 || code == CONVERT_IMAGE_FORMAT_UYVY2NV12_ER)
            {
                inPtrs[0]  = inTiles[0] + 1; // y
                inPtrs[1]  = inTiles[0];     // u
                inPtrs[2]  = inTiles[0] + 2; // v
                outPtrs[0] = outTiles[0];    // y
                outPtrs[1] = outTiles[1];    // uv
            }
            else if (code == CONVERT_IMAGE_FORMAT_VYUY2NV12 || code == CONVERT_IMAGE_FORMAT_VYUY2NV12_ER)
            {
                inPtrs[0]  = inTiles[0] + 1; // y
                inPtrs[1]  = inTiles[0] + 2; // u
                inPtrs[2]  = inTiles[0];     // v
                outPtrs[0] = outTiles[0];    // y
                outPtrs[1] = outTiles[1];    // uv
            }
            else if (code == CONVERT_IMAGE_FORMAT_YUV8p2NV12 || code == CONVERT_IMAGE_FORMAT_YUV8p2NV12_ER)
            {
                inPtrs[0]  = inTiles[0];  // y
                inPtrs[1]  = inTiles[1];  // u
                inPtrs[2]  = inTiles[2];  // v
                outPtrs[0] = outTiles[0]; // y
                outPtrs[1] = outTiles[1]; // uv
            }
            else if (code == CONVERT_IMAGE_FORMAT_YUV8p2YUYV)
            {
                inPtrs[0]  = inTiles[0];      // y
                inPtrs[1]  = inTiles[1];      // u
                inPtrs[2]  = inTiles[2];      // v
                outPtrs[0] = outTiles[0];     // y
                outPtrs[1] = outTiles[0] + 1; // u
                outPtrs[2] = outTiles[0] + 3; // v
            }
            else if (code == CONVERT_IMAGE_FORMAT_YUV8p2UYVY)
            {
                inPtrs[0]  = inTiles[0];      // y
                inPtrs[1]  = inTiles[1];      // u
                inPtrs[2]  = inTiles[2];      // v
                outPtrs[0] = outTiles[0] + 1; // y
                outPtrs[1] = outTiles[0];     // u
                outPtrs[2] = outTiles[0] + 2; // v
            }
            else if (code == CONVERT_IMAGE_FORMAT_YUV8p2VYUY)
            {
                inPtrs[0]  = inTiles[0];      // y
                inPtrs[1]  = inTiles[1];      // u
                inPtrs[2]  = inTiles[2];      // v
                outPtrs[0] = outTiles[0] + 1; // y
                outPtrs[1] = outTiles[0] + 2; // u
                outPtrs[2] = outTiles[0];     // v
            }
            else if (code == CONVERT_IMAGE_FORMAT_NV122BGR8 || code == CONVERT_IMAGE_FORMAT_NV12_ER2BGR8)
            {
                inPtrs[0]  = inTiles[0];      // y
                inPtrs[1]  = inTiles[1];      // uv
                outPtrs[0] = outTiles[0] + 2; // r
                outPtrs[1] = outTiles[0] + 1; // g
                outPtrs[2] = outTiles[0];     // b
            }
            else if (code == CONVERT_IMAGE_FORMAT_NV122RGB8 || code == CONVERT_IMAGE_FORMAT_NV12_ER2RGB8)
            {
                inPtrs[0]  = inTiles[0];      // y
                inPtrs[1]  = inTiles[1];      // uv
                outPtrs[0] = outTiles[0];     // r
                outPtrs[1] = outTiles[0] + 1; // g
                outPtrs[2] = outTiles[0] + 2; // b
            }
            else if (code == CONVERT_IMAGE_FORMAT_NV122BGR8p || code == CONVERT_IMAGE_FORMAT_NV12_ER2BGR8p)
            {
                inPtrs[0]  = inTiles[0];  // y
                inPtrs[1]  = inTiles[1];  // uv
                outPtrs[0] = outTiles[2]; // r
                outPtrs[1] = outTiles[1]; // g
                outPtrs[2] = outTiles[0]; // b
            }
            else if (code == CONVERT_IMAGE_FORMAT_NV122RGB8p || code == CONVERT_IMAGE_FORMAT_NV12_ER2RGB8p)
            {
                inPtrs[0]  = inTiles[0];  // y
                inPtrs[1]  = inTiles[1];  // uv
                outPtrs[0] = outTiles[0]; // r
                outPtrs[1] = outTiles[1]; // g
                outPtrs[2] = outTiles[2]; // b
            }
            else if (code == CONVERT_IMAGE_FORMAT_NV12_ER2RGBA8)
            {
                inPtrs[0]  = inTiles[0];      // y
                inPtrs[1]  = inTiles[1];      // uv
                outPtrs[0] = outTiles[0];     // r
                outPtrs[1] = outTiles[0] + 1; // g
                outPtrs[2] = outTiles[0] + 2; // b
                outPtrs[3] = outTiles[0] + 3; // a (filled with 255)
            }
            else if (code == CONVERT_IMAGE_FORMAT_YUYV2BGR8)
            {
                inPtrs[0]  = inTiles[0];      // y
                inPtrs[1]  = inTiles[0] + 1;  // u
                inPtrs[2]  = inTiles[0] + 3;  // v
                outPtrs[0] = outTiles[0] + 2; // r
                outPtrs[1] = outTiles[0] + 1; // g
                outPtrs[2] = outTiles[0];     // b
            }
            else if (code == CONVERT_IMAGE_FORMAT_YUYV2RGB8)
            {
                inPtrs[0]  = inTiles[0];      // y
                inPtrs[1]  = inTiles[0] + 1;  // u
                inPtrs[2]  = inTiles[0] + 3;  // v
                outPtrs[0] = outTiles[0];     // r
                outPtrs[1] = outTiles[0] + 1; // g
                outPtrs[2] = outTiles[0] + 2; // b
            }
            else if (code == CONVERT_IMAGE_FORMAT_UYVY2BGR8)
            {
                inPtrs[0]  = inTiles[0] + 1;  // y
                inPtrs[1]  = inTiles[0];      // u
                inPtrs[2]  = inTiles[0] + 2;  // v
                outPtrs[0] = outTiles[0] + 2; // r
                outPtrs[1] = outTiles[0] + 1; // g
                outPtrs[2] = outTiles[0];     // b
            }
            else if (code == CONVERT_IMAGE_FORMAT_UYVY2RGB8)
            {
                inPtrs[0]  = inTiles[0] + 1;  // y
                inPtrs[1]  = inTiles[0];      // u
                inPtrs[2]  = inTiles[0] + 2;  // v
                outPtrs[0] = outTiles[0];     // r
                outPtrs[1] = outTiles[0] + 1; // g
                outPtrs[2] = outTiles[0] + 2; // b
            }
            else if (code == CONVERT_IMAGE_FORMAT_VYUY2BGR8)
            {
                inPtrs[0]  = inTiles[0] + 1;  // y
                inPtrs[1]  = inTiles[0] + 2;  // u
                inPtrs[2]  = inTiles[0];      // v
                outPtrs[0] = outTiles[0] + 2; // r
                outPtrs[1] = outTiles[0] + 1; // g
                outPtrs[2] = outTiles[0];     // b
            }
            else if (code == CONVERT_IMAGE_FORMAT_VYUY2RGB8)
            {
                inPtrs[0]  = inTiles[0] + 1;  // y
                inPtrs[1]  = inTiles[0] + 2;  // u
                inPtrs[2]  = inTiles[0];      // v
                outPtrs[0] = outTiles[0];     // r
                outPtrs[1] = outTiles[0] + 1; // g
                outPtrs[2] = outTiles[0] + 2; // b
            }
            else if (code == CONVERT_IMAGE_FORMAT_YUYV2BGR8p)
            {
                inPtrs[0]  = inTiles[0];     // y
                inPtrs[1]  = inTiles[0] + 1; // u
                inPtrs[2]  = inTiles[0] + 3; // v
                outPtrs[0] = outTiles[2];    // r
                outPtrs[1] = outTiles[1];    // g
                outPtrs[2] = outTiles[0];    // b
            }
            else if (code == CONVERT_IMAGE_FORMAT_YUYV2RGB8p)
            {
                inPtrs[0]  = inTiles[0];     // y
                inPtrs[1]  = inTiles[0] + 1; // u
                inPtrs[2]  = inTiles[0] + 3; // v
                outPtrs[0] = outTiles[0];    // r
                outPtrs[1] = outTiles[1];    // g
                outPtrs[2] = outTiles[2];    // b
            }
            else if (code == CONVERT_IMAGE_FORMAT_UYVY2BGR8p)
            {
                inPtrs[0]  = inTiles[0] + 1; // y
                inPtrs[1]  = inTiles[0];     // u
                inPtrs[2]  = inTiles[0] + 2; // v
                outPtrs[0] = outTiles[2];    // r
                outPtrs[1] = outTiles[1];    // g
                outPtrs[2] = outTiles[0];    // b
            }
            else if (code == CONVERT_IMAGE_FORMAT_UYVY2RGB8p)
            {
                inPtrs[0]  = inTiles[0] + 1; // y
                inPtrs[1]  = inTiles[0];     // u
                inPtrs[2]  = inTiles[0] + 2; // v
                outPtrs[0] = outTiles[0];    // r
                outPtrs[1] = outTiles[1];    // g
                outPtrs[2] = outTiles[2];    // b
            }
            else if (code == CONVERT_IMAGE_FORMAT_VYUY2BGR8p)
            {
                inPtrs[0]  = inTiles[0] + 1; // y
                inPtrs[1]  = inTiles[0] + 2; // u
                inPtrs[2]  = inTiles[0];     // v
                outPtrs[0] = outTiles[2];    // r
                outPtrs[1] = outTiles[1];    // g
                outPtrs[2] = outTiles[0];    // b
            }
            else if (code == CONVERT_IMAGE_FORMAT_VYUY2RGB8p)
            {
                inPtrs[0]  = inTiles[0] + 1; // y
                inPtrs[1]  = inTiles[0] + 2; // u
                inPtrs[2]  = inTiles[0];     // v
                outPtrs[0] = outTiles[0];    // r
                outPtrs[1] = outTiles[1];    // g
                outPtrs[2] = outTiles[2];    // b
            }
            else if (code == CONVERT_IMAGE_FORMAT_RGB82BGRA8)
            {
                inPtrs[0]  = inTiles[0];      // r
                inPtrs[1]  = inTiles[0] + 1;  // g
                inPtrs[2]  = inTiles[0] + 2;  // b
                inPtrs[3]  = NULL;            // a
                outPtrs[0] = outTiles[0] + 2; // r
                outPtrs[1] = outTiles[0] + 1; // g
                outPtrs[2] = outTiles[0];     // b
                outPtrs[3] = outTiles[0] + 3; // a
            }
            else if (code == CONVERT_IMAGE_FORMAT_RGB82RGBA8)
            {
                /* RGB → RGBA: no R/B swap; alpha at byte 3 of each 4-byte pixel. */
                inPtrs[0]  = inTiles[0];      // r
                inPtrs[1]  = inTiles[0] + 1;  // g
                inPtrs[2]  = inTiles[0] + 2;  // b
                inPtrs[3]  = NULL;            // a (from generated const)
                outPtrs[0] = outTiles[0];     // r
                outPtrs[1] = outTiles[0] + 1; // g
                outPtrs[2] = outTiles[0] + 2; // b
                outPtrs[3] = outTiles[0] + 3; // a
            }
            else if (code == CONVERT_IMAGE_FORMAT_BGR82BGRA8)
            {
                /* BGR → BGRA: same channel order, alpha at byte 3. */
                inPtrs[0]  = inTiles[0];     // b
                inPtrs[1]  = inTiles[0] + 1; // g
                inPtrs[2]  = inTiles[0] + 2; // r
                inPtrs[3]  = NULL;
                outPtrs[0] = outTiles[0];     // b
                outPtrs[1] = outTiles[0] + 1; // g
                outPtrs[2] = outTiles[0] + 2; // r
                outPtrs[3] = outTiles[0] + 3; // a
            }
            else if (code == CONVERT_IMAGE_FORMAT_RGBA82RGB8 || code == CONVERT_IMAGE_FORMAT_BGRA82BGR8)
            {
                /* Packed 4→3: copy the first 3 bytes of every 4-byte pixel and
                 * drop the alpha lane. Identical byte handling for RGBA→RGB and
                 * BGRA→BGR (channel naming differs, operation does not). */
                inPtrs[0]  = inTiles[0];      // ch0 (r / b)
                inPtrs[1]  = inTiles[0] + 1;  // ch1 (g)
                inPtrs[2]  = inTiles[0] + 2;  // ch2 (b / r)
                inPtrs[3]  = NULL;            // a (skipped)
                outPtrs[0] = outTiles[0];     // ch0
                outPtrs[1] = outTiles[0] + 1; // ch1
                outPtrs[2] = outTiles[0] + 2; // ch2
                outPtrs[3] = NULL;            // a (skipped)
            }
            else if (code == CONVERT_IMAGE_FORMAT_BGR82NV12 || code == CONVERT_IMAGE_FORMAT_BGR82NV12_ER)
            {
                inPtrs[0]  = inTiles[0] + 2; // r
                inPtrs[1]  = inTiles[0] + 1; // g
                inPtrs[2]  = inTiles[0];     // b
                outPtrs[0] = outTiles[0];    // y
                outPtrs[1] = outTiles[1];    // uv
            }
            else if (code == CONVERT_IMAGE_FORMAT_RGB82NV12 || code == CONVERT_IMAGE_FORMAT_RGB82NV12_ER)
            {
                inPtrs[0]  = inTiles[0];     // r
                inPtrs[1]  = inTiles[0] + 1; // g
                inPtrs[2]  = inTiles[0] + 2; // b
                outPtrs[0] = outTiles[0];    // y
                outPtrs[1] = outTiles[1];    // uv
            }
            else if (code == CONVERT_IMAGE_FORMAT_BGR8p2NV12 || code == CONVERT_IMAGE_FORMAT_BGR8p2NV12_ER)
            {
                inPtrs[0]  = inTiles[2];  // r
                inPtrs[1]  = inTiles[1];  // g
                inPtrs[2]  = inTiles[0];  // b
                outPtrs[0] = outTiles[0]; // y
                outPtrs[1] = outTiles[1]; // uv
            }
            else if (code == CONVERT_IMAGE_FORMAT_RGB8p2NV12 || code == CONVERT_IMAGE_FORMAT_RGB8p2NV12_ER)
            {
                inPtrs[0]  = inTiles[0];  // r
                inPtrs[1]  = inTiles[1];  // g
                inPtrs[2]  = inTiles[2];  // b
                outPtrs[0] = outTiles[0]; // y
                outPtrs[1] = outTiles[1]; // uv
            }
            else if (code == CONVERT_IMAGE_FORMAT_BGR82YUYV)
            {
                inPtrs[0]  = inTiles[0] + 2;  // r
                inPtrs[1]  = inTiles[0] + 1;  // g
                inPtrs[2]  = inTiles[0];      // b
                outPtrs[0] = outTiles[0];     // y
                outPtrs[1] = outTiles[0] + 1; // u
                outPtrs[2] = outTiles[0] + 3; // v
            }
            else if (code == CONVERT_IMAGE_FORMAT_RGB82YUYV)
            {
                inPtrs[0]  = inTiles[0];      // r
                inPtrs[1]  = inTiles[0] + 1;  // g
                inPtrs[2]  = inTiles[0] + 2;  // b
                outPtrs[0] = outTiles[0];     // y
                outPtrs[1] = outTiles[0] + 1; // u
                outPtrs[2] = outTiles[0] + 3; // v
            }
            else if (code == CONVERT_IMAGE_FORMAT_BGR82UYVY)
            {
                inPtrs[0]  = inTiles[0] + 2;  // r
                inPtrs[1]  = inTiles[0] + 1;  // g
                inPtrs[2]  = inTiles[0];      // b
                outPtrs[0] = outTiles[0] + 1; // y
                outPtrs[1] = outTiles[0];     // u
                outPtrs[2] = outTiles[0] + 2; // v
            }
            else if (code == CONVERT_IMAGE_FORMAT_RGB82UYVY)
            {
                inPtrs[0]  = inTiles[0];      // r
                inPtrs[1]  = inTiles[0] + 1;  // g
                inPtrs[2]  = inTiles[0] + 2;  // b
                outPtrs[0] = outTiles[0] + 1; // y
                outPtrs[1] = outTiles[0];     // u
                outPtrs[2] = outTiles[0] + 2; // v
            }
            else if (code == CONVERT_IMAGE_FORMAT_BGR82VYUY)
            {
                inPtrs[0]  = inTiles[0] + 2;  // r
                inPtrs[1]  = inTiles[0] + 1;  // g
                inPtrs[2]  = inTiles[0];      // b
                outPtrs[0] = outTiles[0] + 1; // y
                outPtrs[1] = outTiles[0] + 2; // u
                outPtrs[2] = outTiles[0];     // v
            }
            else if (code == CONVERT_IMAGE_FORMAT_RGB82VYUY)
            {
                inPtrs[0]  = inTiles[0];      // r
                inPtrs[1]  = inTiles[0] + 1;  // g
                inPtrs[2]  = inTiles[0] + 2;  // b
                outPtrs[0] = outTiles[0] + 1; // y
                outPtrs[1] = outTiles[0] + 2; // u
                outPtrs[2] = outTiles[0];     // v
            }
            else if (code == CONVERT_IMAGE_FORMAT_BGR8p2YUYV)
            {
                inPtrs[0]  = inTiles[2];      // r
                inPtrs[1]  = inTiles[1];      // g
                inPtrs[2]  = inTiles[0];      // b
                outPtrs[0] = outTiles[0];     // y
                outPtrs[1] = outTiles[0] + 1; // u
                outPtrs[2] = outTiles[0] + 3; // v
            }
            else if (code == CONVERT_IMAGE_FORMAT_RGB8p2YUYV)
            {
                inPtrs[0]  = inTiles[0];      // r
                inPtrs[1]  = inTiles[1];      // g
                inPtrs[2]  = inTiles[2];      // b
                outPtrs[0] = outTiles[0];     // y
                outPtrs[1] = outTiles[0] + 1; // u
                outPtrs[2] = outTiles[0] + 3; // v
            }
            else if (code == CONVERT_IMAGE_FORMAT_BGR8p2UYVY)
            {
                inPtrs[0]  = inTiles[2];      // r
                inPtrs[1]  = inTiles[1];      // g
                inPtrs[2]  = inTiles[0];      // b
                outPtrs[0] = outTiles[0] + 1; // y
                outPtrs[1] = outTiles[0];     // u
                outPtrs[2] = outTiles[0] + 2; // v
            }
            else if (code == CONVERT_IMAGE_FORMAT_RGB8p2UYVY)
            {
                inPtrs[0]  = inTiles[0];      // r
                inPtrs[1]  = inTiles[1];      // g
                inPtrs[2]  = inTiles[2];      // b
                outPtrs[0] = outTiles[0] + 1; // y
                outPtrs[1] = outTiles[0];     // u
                outPtrs[2] = outTiles[0] + 2; // v
            }
            else if (code == CONVERT_IMAGE_FORMAT_BGR8p2VYUY)
            {
                inPtrs[0]  = inTiles[2];      // r
                inPtrs[1]  = inTiles[1];      // g
                inPtrs[2]  = inTiles[0];      // b
                outPtrs[0] = outTiles[0] + 1; // y
                outPtrs[1] = outTiles[0] + 2; // u
                outPtrs[2] = outTiles[0];     // v
            }
            else if (code == CONVERT_IMAGE_FORMAT_RGB8p2VYUY)
            {
                inPtrs[0]  = inTiles[0];      // r
                inPtrs[1]  = inTiles[1];      // b
                inPtrs[2]  = inTiles[2];      // g
                outPtrs[0] = outTiles[0] + 1; // y
                outPtrs[1] = outTiles[0] + 2; // u
                outPtrs[2] = outTiles[0];     // v
            }

            execFunc(inPtrs, outPtrs, &cfg);

            for (int32_t p = 0; p < inPlaneCount; p++)
            {
                cupvaRasterDataFlowRelease(*srcHdls[p]);
            }
            for (int32_t p = 0; p < outPlaneCount; p++)
            {
                cupvaRasterDataFlowRelease(*dstHdls[p]);
            }
        }
    }

    for (int32_t p = 0; p < inPlaneCount; p++)
    {
        cupvaRasterDataFlowClose(*srcHdls[p]);
    }
    for (int32_t p = 0; p < outPlaneCount; p++)
    {
        cupvaRasterDataFlowClose(*dstHdls[p]);
    }

    return 0;
}