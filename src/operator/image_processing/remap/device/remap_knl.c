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

#include "remap_knl.h"

#include <cupva_device.h>

void get_tile_info_init(RemapParam *param, int32_t *coords)
{
    const int32_t dvecw = 16;
    const int32_t niter = 2 * TILE_W * TILE_H / dvecw;

    AgenWrapper w0;
    agen in = init(NULL);
    w0.size = sizeof(float);
    w0.n1   = niter;
    w0.s1   = dvecw;
    INIT_AGEN1(in, w0);

    AgenWrapper w1;
    agen out = init(coords);
    w1.size  = sizeof(int32_t);
    w1.n1    = niter;
    w1.s1    = dvecw;
    INIT_AGEN1(out, w1);

    param->niter[TILE_INFO] = niter;
    param->in[TILE_INFO]    = extract_agen_cfg(in);
    param->out[TILE_INFO]   = extract_agen_cfg(out);
}

/*
 * Get tile info in Y plane
 * 1. warpMap_v = [x, y] (interleaved) -> coords_v = [8 x', 8 y'] (32Byte interleaved, quantized)
 * 2. coords_v = [8 x', 8 y'] -> tileInfo = {xBegin, yBegin, tx, ty}
 */
void get_tile_info_exec(RemapParam *param, TileInfo *tileInfo, int32_t imageWidth, int32_t imageHeight)
{
    int32_t niter = param->niter[TILE_INFO];
    agen in       = init_agen_from_cfg(param->in[TILE_INFO]);
    agen out      = init_agen_from_cfg(param->out[TILE_INFO]);

    float qbits_f = (float)(1 << QBITS);

    dvfloatx half;
    half.lo = replicatef(0.5f);
    half.hi = half.lo;

    dvintx vmin, vmax; // lo = x, hi = y
    vmin.lo = replicatew(INT32_MAX);
    vmin.hi = vmin.lo;
    vmax.lo = replicatew(INT32_MIN);
    vmax.hi = vmax.lo;

    // 1 load            -> 1 VLIW
    // 6 vector computes -> 3 VLIW
    // 1 store           -> 1 VLIW
    for (int32_t i = 0; i < niter; i++) chess_loop_range(6, )
    chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        // reformat and quantize indices
        dvfloatx idxFp                  = dvfloat_load_di(in);
        idxFp                           = dvmaddf(idxFp, qbits_f, half); // pseudo
        dvintx chess_storage(DV) idxInt = dvfp_dvint_trunc(idxFp);       // pseudo

        // update min/max
        vmin = dvmin(vmin, idxInt);
        vmax = dvmax(vmax, idxInt);

        // store quantized indices
        vstore(idxInt, out);
    }

    int32_t x_min = vminr_s(vmin.lo) >> QBITS;
    int32_t y_min = vminr_s(vmin.hi) >> QBITS;
    int32_t x_max = vmaxr_s(vmax.lo) >> QBITS;
    int32_t y_max = vmaxr_s(vmax.hi) >> QBITS;

    // Due to quantization error, few interpolated indices may spread quite close to the integer number, -1/+1 for safe guard
    x_min = ((x_min - 1) < 0) ? 0 : (((x_min - 1) > (imageWidth - 1)) ? (imageWidth - 1) : (x_min - 1));
    y_min = ((y_min - 1) < 0) ? 0 : (((y_min - 1) > (imageHeight - 1)) ? (imageHeight - 1) : (y_min - 1));
    x_max = ((x_max + 1) < 0) ? 0 : (((x_max + 1) > (imageWidth - 1)) ? (imageWidth - 1) : (x_max + 1));
    y_max = ((y_max + 1) < 0) ? 0 : (((y_max + 1) > (imageHeight - 1)) ? (imageHeight - 1) : (y_max + 1));

    tileInfo->xBegin = x_min;
    tileInfo->yBegin = y_min;
    tileInfo->tx     = x_max - x_min + 1;
    tileInfo->ty     = y_max - y_min + 1;
}

void clamp_coords_init(RemapParam *param, int32_t *coords)
{
    // 32byte interleaved format for DLUT index
    const int32_t dvecw = 16;
    const int32_t niter = 2 * TILE_W * TILE_H / dvecw;

    AgenWrapper w0;
    agen in = init(coords);
    w0.size = sizeof(int32_t);
    w0.n1   = niter;
    w0.s1   = dvecw;
    INIT_AGEN1(in, w0);

    AgenWrapper w1;
    agen out = init(NULL);
    w1.size  = sizeof(int32_t);
    w1.n1    = niter;
    w1.s1    = dvecw;
    INIT_AGEN1(out, w1);

    param->niter[CLAMP_COORDS] = niter;
    param->in[CLAMP_COORDS]    = extract_agen_cfg(in);
    param->out[CLAMP_COORDS]   = extract_agen_cfg(out);
}

/*
 * Clamp in bound coordinates to valid range in DLUT
 * coords_v = [8 x', 8 y'] -> clamped_coords = [8 clamped_x, 8 clamped_y]
*/
void clamp_coords_exec(RemapParam *param, int32_t imageWidth, int32_t imageHeight, BorderModeDev borderMode)
{
    int32_t niter = param->niter[CLAMP_COORDS];
    agen in       = init_agen_from_cfg(param->in[CLAMP_COORDS]);
    agen out      = init_agen_from_cfg(param->out[CLAMP_COORDS]);

    dvintx vLimit;
    vLimit.lo = replicatew((imageWidth - 1) << QBITS);
    vLimit.hi = replicatew((imageHeight - 1) << QBITS);

    dvintx vBound;
    vBound.lo = replicatew(imageWidth << QBITS);
    vBound.hi = replicatew(imageHeight << QBITS);

    if (borderMode == CONSTANT)
    {
        // 1 load     -> 1 VLIW
        // 3 computes -> 1.5 VLIW
        // 1 store    -> 1 VLIW
        for (int32_t i = 0; i < niter; i++) chess_loop_range(24, )
        chess_unroll_loop(8)
        chess_prepare_for_pipelining
        {
            dvintx chess_storage(DV) vCoords = dvint_load(in);

            // if vCoords < vBound, clamp vCoords to vLimit
            dvintx chess_storage(DV) vCoordsOut = dvmux(vCoords < vBound, dvmin(vCoords, vLimit), vCoords);

            vstore(vCoordsOut, out);
        }
    }
    else
    {
        // 1 load     -> 1 VLIW
        // 2 computes -> 1 VLIW
        // 1 store    -> 1 VLIW
        for (int32_t i = 0; i < niter; i++) chess_loop_range(18, )
        chess_unroll_loop(6)
        chess_prepare_for_pipelining
        {
            dvintx vCoords = dvint_load(in);

            // clamp all coordinates to [0, vLimit]
            vCoords = dvmin(vCoords, vLimit);
            vCoords = dvmax(vCoords, 0);

            vstore(vCoords, out);
        }
    }
}

void calc_coords_UV_init(RemapParam *param, int32_t *coords, int32_t *coordsUV_mask)
{
    const int32_t dvecw  = 16;
    const int32_t niter1 = (2 * TILE_W + dvecw - 1) / dvecw;
    const int32_t niter2 = TILE_H / 2; // process 2 lines each time
    const int32_t niter  = niter1 * niter2;

    AgenWrapper w0;
    agen in = init(coords);
    w0.size = sizeof(int32_t);
    w0.n1   = 2; // for top and bottom lines
    w0.s1   = TILE_W * 2;
    w0.n2   = 2; // for normal load to check boundary
    w0.s2   = 0;
    w0.n3   = niter1;
    w0.s3   = dvecw;
    w0.n4   = niter2;
    w0.s4   = TILE_W * 4;
    INIT_AGEN4(in, w0);

    AgenWrapper w1;
    agen out = init(NULL);
    w1.size  = sizeof(int32_t);
    w1.n1    = niter1;
    w1.s1    = dvecw / 2;
    w1.n2    = niter2;
    w1.s2    = TILE_W;
    INIT_AGEN2(out, w1);
    out.round = 0x80 | 1; // truncate

    agen mask = init(coordsUV_mask);
    INIT_AGEN2(mask, w1);

    param->niter[CALC_COORDS_UV] = niter;
    param->in[CALC_COORDS_UV]    = extract_agen_cfg(in);
    param->out[CALC_COORDS_UV]   = extract_agen_cfg(out);
    param->mask[CALC_COORDS_UV]  = extract_agen_cfg(mask);
}

/*
 * Calculate the chroma coordinates for UV plane
 * 1. coords_v = [8 x', 8 y'] -> ((tl + tr + bl + br) * 0.25 - 0.5) * 0.5 -> coordsUV_v = [ch_x, ch_y]
 * 2. coords_v = [8 x', 8 y'] -> coordsUV_mask = out_of_range[left, right]
 */
void calc_coords_UV_exec(RemapParam *param, int32_t imageWidth, int32_t imageHeight)
{
    int32_t niter = param->niter[CALC_COORDS_UV];

    agen in   = init_agen_from_cfg(param->in[CALC_COORDS_UV]);
    agen out  = init_agen_from_cfg(param->out[CALC_COORDS_UV]);
    agen mask = init_agen_from_cfg(param->mask[CALC_COORDS_UV]);

    dvintx vBound;
    vBound.lo = replicatew(imageWidth << QBITS);
    vBound.hi = replicatew(imageHeight << QBITS);

    dvintx vZero = chess_dont_care(dvintx) & 0;

    // 1/2 in Q format
    const int32_t half_q = 1 << (QBITS - 1);
    vintx vNegHalfQ      = replicatew(-half_q);
    vintx vTwo           = replicatew(2);

    // chroma_idx = ((x_tl + x_tr + x_bl + x_br) * 0.25f - 0.5f) * 0.5f according to VPI implementation

    // 4 loads     -> 4 VLIW (bank conflict)
    // 13 computes -> 6.5 VLIW
    // 2 stores    -> 1 or 2 VLIW (depends on whether out and mask are from different banks)
    for (int32_t i = 0; i < niter; i++) chess_loop_range(6, )
    chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        // in:  [lo: x0, x1, x2, x3, x4, x5, x6, x7, hi: y0, y1, y2, y3, y4, y5, y6, y7]
        // out: [lo: x0, x2, x4, x6, y0, y2, y4, y6, hi: x1, x3, x5, x7, y1, y3, y5, y7]
        dvintx vTop    = dvint_load_di(in); // lo = [left], hi = [right]
        dvintx vBottom = dvint_load_di(in);

        // need normal load for boundary check
        dvintx vTop_p    = dvint_load(in); // lo = [x], hi = [y]
        dvintx vBottom_p = dvint_load(in);

        // vFlag = vTop_p >= vBound || vBottom_p >= vBound || vTop_p < 0 || vBottom_p < 0;
        dvintx vFlag0 = dvCmpLE_orL(vBound, vTop_p, (vBottom_p >= vBound)); // pseudo
        dvintx vFlag1 = dvCmpLT_orL(vTop_p, vZero, vFlag0);                 // pseudo
        dvintx vFlag  = dvCmpLT_orL(vBottom_p, vZero, vFlag1);              // pseudo

        // combine x and y direction together
        vintx vmask = vFlag.lo || vFlag.hi; // [left, right, ...]

        // 1/4 with rounding: vRes = (tl + tr + bl + br + 0x2) >> 2 - half_q
        vintx vRes = vadd3(vTop.lo, vTop.hi, vBottom.lo);
        vRes       = vadd3(vRes, vBottom.hi, vTwo);
        vRes       = vshiftadd(vRes, -2, vNegHalfQ);
        // 1/2 with rounding
        vRes = (vRes + 0x1); // >> 1 is handled by out agen
        // turn the results into x/y elements interleaved format
        vRes = vshuffle(vRes); // [ch_x, ch_y]

        vstore(vRes, out);

        // 2x2 region is out of range when either vFlag[2n] or vFlag[2n+1] are true
        vstore(vmask, mask);
    }
}

void get_tile_info_UV_init(RemapParam *param)
{
    const int32_t dvecw = 16;
    const int32_t niter = TILE_W * (TILE_H / 2) / dvecw;

    AgenWrapper w0;
    agen in = init(NULL);
    w0.size = sizeof(int32_t);
    w0.n1   = niter;
    w0.s1   = dvecw;
    INIT_AGEN1(in, w0);

    param->niter[TILE_INFO_UV] = niter;
    param->in[TILE_INFO_UV]    = extract_agen_cfg(in);
}

/*
 * Get tile info in UV plane
 * coordsUV_v = [ch_x, ch_y] -> tileInfoUV = {xBegin, yBegin, tx, ty}
*/
void get_tile_info_UV_exec(RemapParam *param, TileInfo *tileInfoUV, int32_t imageWidth, int32_t imageHeight)
{
    int32_t niter = param->niter[TILE_INFO_UV];
    agen in       = init_agen_from_cfg(param->in[TILE_INFO_UV]);

    dvintx vmin, vmax; // lo = x, hi = y
    vmin.lo = replicatew(INT32_MAX);
    vmin.hi = vmin.lo;
    vmax.lo = replicatew(INT32_MIN);
    vmax.hi = vmax.lo;

    // 1 load     -> 1 VLIW
    // 2 computes -> 1 VLIW
    for (int32_t i = 0; i < niter; i++) chess_loop_range(12, )
    chess_unroll_loop(6)
    chess_prepare_for_pipelining
    {
        // reformat chroma coordinates
        dvintx coords = dvint_load_di(in);

        // update min/max
        vmin = dvmin(vmin, coords);
        vmax = dvmax(vmax, coords);
    }

    int32_t x_min = vminr_s(vmin.lo) >> QBITS;
    int32_t y_min = vminr_s(vmin.hi) >> QBITS;
    int32_t x_max = vmaxr_s(vmax.lo) >> QBITS;
    int32_t y_max = vmaxr_s(vmax.hi) >> QBITS;

    // Due to quantization error, few interpolated indices may spread quite close to the integer number, -1/+1 for safe guard
    x_min = ((x_min - 1) < 0) ? 0 : (((x_min - 1) > (imageWidth - 1)) ? (imageWidth - 1) : (x_min - 1));
    y_min = ((y_min - 1) < 0) ? 0 : (((y_min - 1) > (imageHeight - 1)) ? (imageHeight - 1) : (y_min - 1));
    x_max = ((x_max + 1) < 0) ? 0 : (((x_max + 1) > (imageWidth - 1)) ? (imageWidth - 1) : (x_max + 1));
    y_max = ((y_max + 1) < 0) ? 0 : (((y_max + 1) > (imageHeight - 1)) ? (imageHeight - 1) : (y_max + 1));

    tileInfoUV->xBegin = x_min;
    tileInfoUV->yBegin = y_min;
    tileInfoUV->tx     = x_max - x_min + 1;
    tileInfoUV->ty     = y_max - y_min + 1;
}

void clamp_coords_UV_init(RemapParam *param, int32_t *coordsUV_mask)
{
    const int32_t dvecw = 16;
    const int32_t niter = ((TILE_W / 2) * (TILE_H / 2) * 2 + dvecw - 1) / dvecw;

    AgenWrapper w0;
    agen in = init(NULL);
    w0.size = sizeof(int32_t);
    w0.n1   = niter;
    w0.s1   = dvecw;
    INIT_AGEN1(in, w0);

    agen out = init(NULL);
    INIT_AGEN1(out, w0);

    agen mask = init(coordsUV_mask);
    INIT_AGEN1(mask, w0);

    param->niter[CLAMP_COORDS_UV] = niter;
    param->in[CLAMP_COORDS_UV]    = extract_agen_cfg(in);
    param->out[CLAMP_COORDS_UV]   = extract_agen_cfg(out);
    param->mask[CLAMP_COORDS_UV]  = extract_agen_cfg(mask);
}

/*
 * Clamp in bound coordinates to valid range in DLUT
 * coordsUV_v = [ch_x, ch_y] -> clamped_coordsUV = [clamped_chx, clamped_chy]
 */
void clamp_coords_UV_exec(RemapParam *param, int32_t imageWidth, int32_t imageHeight, BorderModeDev borderMode)
{
    int32_t niter = param->niter[CLAMP_COORDS_UV];

    agen in   = init_agen_from_cfg(param->in[CLAMP_COORDS_UV]);
    agen out  = init_agen_from_cfg(param->out[CLAMP_COORDS_UV]);
    agen mask = init_agen_from_cfg(param->mask[CLAMP_COORDS_UV]);

    dvintx vLimit;
    vLimit.lo = replicatew((imageWidth - 1) << QBITS);
    vLimit.hi = replicatew((imageHeight - 1) << QBITS);

    // filter out the out of range pixels
    const int32_t flag_out = -(1 << QBITS);
    dvintx vFlagOut;
    vFlagOut.lo = replicatew(flag_out);
    vFlagOut.hi = vFlagOut.lo;

    if (borderMode == CONSTANT)
    {
        // 2 loads    -> 1 VLIW (if in and mask are from different banks)
        // 3 computes -> 1.5 VLIW
        // 1 store    -> 1 VLIW
        for (int32_t i = 0; i < niter; i++) chess_loop_range(18, )
        chess_unroll_loop(6)
        chess_prepare_for_pipelining
        {
            dvintx vCoords = dvint_load_di(in);   // [ch_x, ch_y] -> [8 ch_x, 8 ch_y]
            dvintx vFlag   = dvint_load_di(mask); // [left, right] -> [8 left, 8 right]

            // clamp result for all coords
            vCoords = dvmax(vCoords, 0);
            vCoords = dvmin(vCoords, vLimit);

            // set out-of-range coords to -1Q
            vCoords = dvmux(vFlag, vFlagOut, vCoords);

            vstore_i(vCoords, out);
        }
    }
    else
    {
        // 2 loads    -> 1 VLIW (if in and mask are from different banks)
        // 2 computes -> 1 VLIW
        // 1 store    -> 1 VLIW
        for (int32_t i = 0; i < niter; i++) chess_loop_range(18, )
        chess_unroll_loop(6)
        chess_prepare_for_pipelining
        {
            dvintx vCoords = dvint_load_di(in);   // [ch_x, ch_y] -> [8 ch_x, 8 ch_y]
            dvintx vFlag   = dvint_load_di(mask); // [left, right] -> [8 left, 8 right]

            // clamp result for all coords
            vCoords = dvmax(vCoords, 0);
            vCoords = dvmin(vCoords, vLimit);

            vstore_i(vCoords, out);
        }
    }
}

/*
 * Split the UV plane into U and V buffers
 * srcUV_v = [U, V] -> srcU = [U], srcV = [V]
 * No init function because the tile size of srcUV_v is variable.
 */
void split_UV_exec(uint8_t *src, uint8_t *dstU, uint8_t *dstV, int32_t tx, int32_t ty)
{
    const int32_t dvecw  = 64;
    const int32_t niter1 = (tx + dvecw - 1) / dvecw;
    const int32_t niter2 = ty;

    AgenWrapper w0;
    agen in = init(src);
    w0.size = sizeof(uint8_t);
    w0.n1   = niter1;
    w0.s1   = dvecw;
    w0.n2   = niter2;
    w0.s2   = tx;
    INIT_AGEN2(in, w0);

    agen outU = init(dstU);
    w0.s1     = dvecw / 2;
    w0.s2     = tx / 2;
    INIT_AGEN2(outU, w0);

    agen outV = init(dstV);
    INIT_AGEN2(outV, w0);

    int32_t niter = ((niter1 * niter2 + 3) / 4) * 4;
    niter         = niter < 12 ? 12 : niter;

    // 1 load   -> 1 VLIW
    // 2 stores -> 1 VLIW (if outU and outV are from different banks)
    for (int32_t i = 0; i < niter; i++) chess_loop_range(8, )
    chess_unroll_loop(4)
    chess_prepare_for_pipelining
    {
        dvcharx vIn = dvuchar_load_di(in);
        vstore(vIn.lo, outU);
        vstore(vIn.hi, outV);
    }
}

void merge_UV_init(MergeUVParam *param, uint8_t *dstUV)
{
    const int32_t vecw = 32;
    // downsample 2x2 for both U and V planes
    const int32_t size  = TILE_W * TILE_H / 4;
    const int32_t niter = (size + vecw - 1) / vecw;

    AgenWrapper w0;
    agen out = init(dstUV);
    w0.size  = sizeof(uint8_t);
    w0.n1    = niter;
    w0.s1    = vecw * 2;
    INIT_AGEN1(out, w0);

    param->niter = (niter + 3) / 4 * 4;
    param->out   = extract_agen_cfg(out);
}

/*
 * Merge the U and V results into the UV plane
 * dstU = [U], dstV = [V] -> dstUV_v = [U, V]
 */
void merge_UV_exec(MergeUVParam *param, uint8_t *srcU, uint8_t *srcV)
{
    int32_t niter = param->niter;
    agen out      = init_agen_from_cfg(param->out);

    // 2 loads, 1 store -> 1 VLIW (if inU and inV are from different banks)
    for (int32_t i = 0; i < niter; i++) chess_loop_range(12, )
    chess_unroll_loop(4)
    chess_prepare_for_pipelining
    {
        dvcharx vDst;
        vDst.lo = zero_extend(*((vuchar *)srcU));
        vDst.hi = zero_extend(*((vuchar *)srcV));
        vstore_i(vDst, out);
        srcU += 32;
        srcV += 32;
    }
}

void update_sampler(CupvaSampler *samplerCfg, uint8_t *src, int32_t *coords, uint8_t *dst, TileInfo *tileInfo)
{
    const int32_t tx = tileInfo->tx;
    const int32_t ty = tileInfo->ty;
    cupvaSamplerUpdateAddrs(samplerCfg, src, coords, dst);
    cupvaSamplerUpdateInputAttributes(samplerCfg, tx, ty, tx);
    samplerCfg->x_int_limit = tx - 1;
    samplerCfg->y_int_limit = ty - 1;
    samplerCfg->x_offset    = -tileInfo->xBegin * (1 << QBITS);
    samplerCfg->y_offset    = -tileInfo->yBegin * (1 << QBITS);
}