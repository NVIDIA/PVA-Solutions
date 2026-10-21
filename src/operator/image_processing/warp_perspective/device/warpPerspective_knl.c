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

#include "warpPerspective_knl.h"

#include <cupva_device.h>

static float vec_base[16] = {0,    1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,
                             8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f};

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/*
 *
 * Coordinates calculations
 * |a0, a1, a2| * / |baseX| + |tile_idx*dX| \   |x'|
 * |a3, a4, a5| * | |baseY| + |tile_idx*dY| | = |y'|
 * |a6, a7, a8| * \ |baseZ| + |      0    | /   |z'|
 * dX and dY are TILE_W and TILE_H respectively
 * Perform coordinates calculations for the tile in following steps:
 * 1. init_cords_base computes the base coordinates for the tile
 * 2. calc_delta_x and calc_delta_y compute the delta coordinates based on delta vector[TILE_W, TILE_H, 0]
 * 3. calc_cords combines the base coordinates, delta coordinates and transform coordinates back to Euclidean space (per tile processing)
 * 4. get_tile_info computes the bounding box for the tile (per tile processing)
 * 5. add_ofst_cords adds the offset to the coordinates as DLUT only accepts unsigned coordinates (per tile processing)
 *
*/
void init_cords_base(float *cordsBaseNum, float *cordsBaseDen, float *warpMat)
{
    const int32_t vecw   = 16;
    const int32_t niter1 = (TILE_W + vecw - 1) / vecw;
    const int32_t niter2 = TILE_H;

    dvfloatx vBaseX = sign_extend(*((dvfloat *)&vec_base[0]));
    float BaseY     = 0.0f;

    dvfloat *resNumX = (dvfloat *)&cordsBaseNum[0];
    dvfloat *resNumY = (dvfloat *)&cordsBaseNum[TILE_W * TILE_H];
    dvfloat *resDenZ = (dvfloat *)&cordsBaseDen[0];
    for (int32_t i = 0; i < niter2; i++)
    {
        dvfloatx vX = vBaseX;
        for (int32_t j = 0; j < niter1; j++) chess_loop_range(4, )
        chess_prepare_for_pipelining
        {
            dvfloatx vCordsX = vX * warpMat[0];
            vCordsX += BaseY * warpMat[1];
            vCordsX += warpMat[2];
            dvfloatx vCordsY = vX * warpMat[3];
            vCordsY += BaseY * warpMat[4];
            vCordsY += warpMat[5];
            dvfloatx vCordsZ = vX * warpMat[6];
            vCordsZ += BaseY * warpMat[7];
            vCordsZ += warpMat[8];

            *(resNumX++) = extract(vCordsX);
            *(resNumY++) = extract(vCordsY);
            *(resDenZ++) = extract(vCordsZ);

            vX += (float)vecw;
        }
        // update to beginning of the next line
        BaseY += 1.0f;
    }
}

void calc_delta_x(float *dMatX, float *warpMat)
{
    dMatX[0] = warpMat[0] * (float)TILE_W;
    dMatX[1] = warpMat[3] * (float)TILE_W;
    dMatX[2] = warpMat[6] * (float)TILE_W;
}

void calc_delta_y(float *dMatY, float *warpMat)
{
    dMatY[0] = warpMat[1] * (float)TILE_H;
    dMatY[1] = warpMat[4] * (float)TILE_H;
    dMatY[2] = warpMat[7] * (float)TILE_H;
}

void calc_cords(int32_t *cords, float *cordsNum, float *cordsDen, float *dMatX, float *dMatY, const int32_t tile_id_x,
                const int32_t tile_id_y, const int32_t qbits)
{
    float dx = tile_id_x * dMatX[0] + tile_id_y * dMatY[0];
    float dy = tile_id_x * dMatX[1] + tile_id_y * dMatY[1];
    float dz = tile_id_x * dMatX[2] + tile_id_y * dMatY[2];

    const int32_t vecw   = 16;
    const int32_t niter1 = (TILE_W + vecw - 1) / vecw;
    const int32_t niter2 = TILE_H;
    const int32_t niter  = niter1 * niter2;

    vint *res           = (vint *)&cords[0];
    dvfloat *cordsNumX  = (dvfloat *)&cordsNum[0];
    dvfloat *cordsNumY  = (dvfloat *)&cordsNum[TILE_W * TILE_H];
    dvfloat *cordsDenZ  = (dvfloat *)&cordsDen[0];
    const float qbits_f = (float)(1 << qbits);
    for (int32_t i = 0; i < niter; i++) chess_loop_range(6, )
    chess_prepare_for_pipelining
    {
        dvfloatx vX = sign_extend(*(cordsNumX++));
        dvfloatx vY = sign_extend(*(cordsNumY++));
        dvfloatx vZ = sign_extend(*(cordsDenZ++));

        vX += dx;
        vY += dy;
        vZ += dz;

        vZ = dvfrcp(vZ);
        vX = vX * vZ;
        vY = vY * vZ;

        vX          = vX * qbits_f;
        vY          = vY * qbits_f;
        dvintx vX_i = dvfp_dvint_round(vX);
        dvintx vY_i = dvfp_dvint_round(vY);

        // 32-byte interleaved format(32B X then 32B Y) for dlut index
        *(res++) = extract(vX_i.lo);
        *(res++) = extract(vY_i.lo);
        *(res++) = extract(vX_i.hi);
        *(res++) = extract(vY_i.hi);
    }
}

void calc_cords_affine(int32_t *cords, float *cordsNum, float *dMatX, float *dMatY, const int32_t tile_id_x,
                       const int32_t tile_id_y, const int32_t qbits)
{
    float dx = tile_id_x * dMatX[0] + tile_id_y * dMatY[0];
    float dy = tile_id_x * dMatX[1] + tile_id_y * dMatY[1];

    const int32_t vecw   = 16;
    const int32_t niter1 = (TILE_W + vecw - 1) / vecw;
    const int32_t niter2 = TILE_H;
    const int32_t niter  = niter1 * niter2;

    vint *res           = (vint *)&cords[0];
    dvfloat *cordsNumX  = (dvfloat *)&cordsNum[0];
    dvfloat *cordsNumY  = (dvfloat *)&cordsNum[TILE_W * TILE_H];
    const float qbits_f = (float)(1 << qbits);
    for (int32_t i = 0; i < niter; i++) chess_loop_range(6, )
    chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        dvfloatx vX = sign_extend(*(cordsNumX++));
        dvfloatx vY = sign_extend(*(cordsNumY++));

        vX += dx;
        vY += dy;

        vX          = vX * qbits_f;
        vY          = vY * qbits_f;
        dvintx vX_i = dvfp_dvint_round(vX);
        dvintx vY_i = dvfp_dvint_round(vY);

        // 32-byte interleaved format(32B X then 32B Y) for dlut index
        *(res++) = extract(vX_i.lo);
        *(res++) = extract(vY_i.lo);
        *(res++) = extract(vX_i.hi);
        *(res++) = extract(vY_i.hi);
    }
}

void get_tile_info(TileInfo *tileInfo, int32_t *cords, const int32_t imgWidth, const int32_t imgHeight,
                   const int32_t qbits)
{
    // 8 x coordinates followed by 8 y coordinates
    const int32_t vecw = 8;
    // for warp perspective, straight lines still kepp straight, deduce the bounding box from four corners
    int32_t x_tl = cords[0];
    int32_t y_tl = cords[vecw];
    int32_t x_tr = cords[TILE_W * 2 - (vecw + 1)];
    int32_t y_tr = cords[TILE_W * 2 - 1];
    int32_t x_bl = cords[(TILE_H - 1) * TILE_W * 2];
    int32_t y_bl = cords[(TILE_H - 1) * TILE_W * 2 + vecw];
    int32_t x_br = cords[TILE_H * TILE_W * 2 - (vecw + 1)];
    int32_t y_br = cords[TILE_H * TILE_W * 2 - 1];

    // find the minimum and maximum x and y coordinates
    int32_t x_min = MIN(MIN(x_tl, x_tr), MIN(x_bl, x_br));
    int32_t x_max = MAX(MAX(x_tl, x_tr), MAX(x_bl, x_br));
    int32_t y_min = MIN(MIN(y_tl, y_tr), MIN(y_bl, y_br));
    int32_t y_max = MAX(MAX(y_tl, y_tr), MAX(y_bl, y_br));

    if ((x_max < 0) || (y_max < 0) || x_min >= (imgWidth << qbits) || (y_min >= (imgHeight << qbits)))
    {
        // all the pixels are outside the image, return full black tile
        tileInfo->fullBlack = 1;
        return;
    }

    x_min = x_min >> qbits;
    x_max = x_max >> qbits;
    y_min = y_min >> qbits;
    y_max = y_max >> qbits;

    // Due to quantization error, few interpolated indices may spread quite close to the integer number, -1 for safe guard
    x_min = x_min - 1;
    y_min = y_min - 1;
    x_max = x_max >= imgWidth ? (imgWidth - 1) : x_max;
    y_max = y_max >= imgHeight ? (imgHeight - 1) : y_max;

    // all negative coordinates are treated as out of range
    tileInfo->xBegin = x_min < 0 ? 0 : x_min;
    tileInfo->yBegin = y_min < 0 ? 0 : y_min;
    tileInfo->tx     = (x_max - x_min) + 2;
    tileInfo->ty     = (y_max - y_min) + 2;

    // input tile should not exceed MAX_TABLE_SIZE
    if (tileInfo->tx * tileInfo->ty > MAX_TABLE_SIZE)
    {
        tileInfo->fullBlack = 1;
        return;
    }

    tileInfo->fullBlack = 0;

    return;
}

void get_tile_info_SP(TileInfo *tileInfoSP, int32_t *cords, const int32_t imgWidth, const int32_t imgHeight,
                      const int32_t qbits)
{
    // x/y interleaved format for UV coordinates
    const int32_t vecw = 8;
    // for warp perspective, straight lines still kepp straight, deduce the bounding box from four corners
    int32_t x_tl = cords[0];
    int32_t y_tl = cords[1];
    int32_t x_tr = cords[TILE_W - 2];
    int32_t y_tr = cords[TILE_W - 1];
    int32_t x_bl = cords[(TILE_H / 2 - 1) * TILE_W];
    int32_t y_bl = cords[(TILE_H / 2 - 1) * TILE_W + 1];
    int32_t x_br = cords[(TILE_H / 2) * TILE_W - 2];
    int32_t y_br = cords[(TILE_H / 2) * TILE_W - 1];

    // find the minimum and maximum x and y coordinates
    int32_t x_min = MIN(MIN(x_tl, x_tr), MIN(x_bl, x_br));
    int32_t x_max = MAX(MAX(x_tl, x_tr), MAX(x_bl, x_br));
    int32_t y_min = MIN(MIN(y_tl, y_tr), MIN(y_bl, y_br));
    int32_t y_max = MAX(MAX(y_tl, y_tr), MAX(y_bl, y_br));

    if ((x_max < 0) || (y_max < 0) || x_min >= (imgWidth << qbits) || y_min >= (imgHeight << qbits))
    {
        // all the pixels are outside the image, return full black tile
        tileInfoSP->fullBlack = 1;
        return;
    }

    x_min = x_min >> qbits;
    x_max = x_max >> qbits;
    y_min = y_min >> qbits;
    y_max = y_max >> qbits;

    // Due to quantization error, few interpolated indices may spread quite close to the integer number, -1 for safe guard
    x_min = x_min - 1;
    y_min = y_min - 1;
    x_max = x_max >= imgWidth ? (imgWidth - 1) : x_max;
    y_max = y_max >= imgHeight ? (imgHeight - 1) : y_max;

    // all negative coordinates are treated as out of range
    tileInfoSP->xBegin = x_min < 0 ? 0 : x_min;
    tileInfoSP->yBegin = y_min < 0 ? 0 : y_min;
    tileInfoSP->tx     = (x_max - x_min) + 2;
    tileInfoSP->ty     = (y_max - y_min) + 2;

    // input tile should not exceed MAX_TABLE_SIZE_SP / 2, since we have two planes
    if (tileInfoSP->tx * tileInfoSP->ty > (MAX_TABLE_SP_SIZE / 2))
    {
        tileInfoSP->fullBlack = 1;
        return;
    }

    tileInfoSP->fullBlack = 0;

    return;
}

void add_ofst_cords_init(AddOfstCordsParam *param, int32_t *cords)
{
    // 32byte interleaved format for dlut index
    const int32_t vecw  = 16;
    const int32_t niter = 2 * TILE_W * TILE_H / vecw;

    AgenWrapper wrapper0;
    agen a_in     = init(cords);
    wrapper0.size = sizeof(int32_t);
    wrapper0.n1   = niter;
    wrapper0.s1   = vecw;
    INIT_AGEN1(a_in, wrapper0);

    AgenWrapper wrapper1;
    agen a_out    = init(NULL);
    wrapper1.size = sizeof(int32_t);
    wrapper1.n1   = niter;
    wrapper1.s1   = vecw;
    INIT_AGEN1(a_out, wrapper1);

    param->niter = niter;
    param->a_in  = extract_agen_cfg(a_in);
    param->a_out = extract_agen_cfg(a_out);
}

void add_ofst_cords_exec(AddOfstCordsParam *param, int32_t *cords_rebased, const int32_t ofst_x, const int32_t ofst_y)
{
    const int32_t niter = param->niter;

    cupvaModifyAgenCfgBase(&param->a_out, cords_rebased);
    agen a_in  = init_agen_from_cfg(param->a_in);
    agen a_out = init_agen_from_cfg(param->a_out);

    dvintx vOfst;
    vOfst.lo = replicatew(ofst_x);
    vOfst.hi = replicatew(ofst_y);

    for (int32_t i = 0; i < niter; i++) chess_loop_range(24, )
    chess_unroll_loop(8)
    chess_prepare_for_pipelining
    {
        dvintx vCords = dvint_load(a_in);
        vCords -= vOfst;
        vstore(vCords, a_out);
    }
}

void add_ofst_cords_SP_init(AddOfstCordsParamSP *param, int32_t *cords, int32_t *cordsSP_mask)
{
    const int32_t vecw  = 16;
    const int32_t niter = ((TILE_W / 2) * (TILE_H / 2) * 2 + vecw - 1) / vecw;

    AgenWrapper wrapper0;
    agen a_in     = init(cords);
    wrapper0.size = sizeof(int32_t);
    wrapper0.n1   = niter;
    wrapper0.s1   = vecw;
    INIT_AGEN1(a_in, wrapper0);

    AgenWrapper wrapper1;
    agen a_out    = init(NULL);
    wrapper1.size = sizeof(int32_t);
    wrapper1.n1   = niter;
    wrapper1.s1   = vecw;
    INIT_AGEN1(a_out, wrapper1);

    AgenWrapper wrapper2;
    agen a_mask   = init(cordsSP_mask);
    wrapper2.size = sizeof(int32_t);
    wrapper2.n1   = niter;
    wrapper2.s1   = vecw;
    INIT_AGEN1(a_mask, wrapper2);

    param->niter  = niter;
    param->a_in   = extract_agen_cfg(a_in);
    param->a_out  = extract_agen_cfg(a_out);
    param->a_mask = extract_agen_cfg(a_mask);
}

void add_ofst_cords_SP_exec(AddOfstCordsParamSP *param, int32_t *cords, const int32_t ofst_x, const int32_t ofst_y)
{
    const int32_t niter = param->niter;
    cupvaModifyAgenCfgBase(&param->a_in, cords);
    cupvaModifyAgenCfgBase(&param->a_out, cords);

    agen a_in   = init_agen_from_cfg(param->a_in);
    agen a_out  = init_agen_from_cfg(param->a_out);
    agen a_mask = init_agen_from_cfg(param->a_mask);

    dvintx vOfst;
    vOfst.lo = replicatew(ofst_x);
    vOfst.hi = replicatew(ofst_y);

    // filter out the out of range pixels
    const int32_t flag_out = -(1 << QBITS);
    dvintx vFlagOut;
    vFlagOut.lo = replicatew(flag_out);
    vFlagOut.hi = replicatew(flag_out);

    for (int32_t i = 0; i < niter; i++) chess_loop_range(18, )
    chess_unroll_loop(6)
    chess_prepare_for_pipelining
    {
        dvintx vCords = dvint_load_di(a_in);
        dvintx vFlag  = dvint_load_di(a_mask);
        vCords -= vOfst;
        vCords = dvmux(vFlag, vFlagOut, vCords);
        vstore_i(vCords, a_out);
    }
}

void update_for_next_tile(TileInfo *tileInfoCur, TileInfo *tileInfoNext, int32_t &nextTileIdX, int32_t &nextTileIdY,
                          const int32_t ntileX)
{
    // calculate next tile id info
    nextTileIdY  = nextTileIdX == ntileX - 1 ? nextTileIdY + 1 : nextTileIdY;
    nextTileIdX  = nextTileIdX == ntileX - 1 ? 0 : nextTileIdX + 1;
    *tileInfoCur = *tileInfoNext;
}

void calc_cords_for_next_tile(TileInfo *tileInfoNext, int32_t *cords, float *cordsBaseNum, float *cordsBaseDen,
                              float *dMatX, float *dMatY, const int32_t nextTileIdX, const int32_t nextTileIdY,
                              const int32_t imageWidth, const int32_t imageHeight, const int32_t qbits,
                              const TransformTypeDev transformType)
{
    if (transformType == AFFINE)
    {
        calc_cords_affine(&cords[0], &cordsBaseNum[0], &dMatX[0], &dMatY[0], nextTileIdX, nextTileIdY, qbits);
    }
    else
    {
        calc_cords(&cords[0], &cordsBaseNum[0], &cordsBaseDen[0], &dMatX[0], &dMatY[0], nextTileIdX, nextTileIdY,
                   qbits);
    }
    get_tile_info(tileInfoNext, &cords[0], imageWidth, imageHeight, qbits);
}

void split_UV_exec(uint8_t *src, uint8_t *dstU, uint8_t *dstV, const int32_t tx, const int32_t ty)
{
    // dv for uint8_t type
    const int32_t vecw   = 64;
    const int32_t niter1 = (tx + vecw - 1) / vecw;
    const int32_t niter2 = ty;

    AgenWrapper wrapper;
    agen a_in    = init(src);
    wrapper.size = sizeof(uint8_t);
    wrapper.n1   = niter1;
    wrapper.s1   = vecw;
    wrapper.n2   = niter2;
    wrapper.s2   = tx;
    INIT_AGEN2(a_in, wrapper);

    agen a_outU = init(dstU);
    wrapper.s1  = vecw / 2;
    wrapper.s2  = tx / 2;
    INIT_AGEN2(a_outU, wrapper);

    agen a_outV = init(dstV);
    INIT_AGEN2(a_outV, wrapper);

    int32_t niter = ((niter1 * niter2 + 3) / 4) * 4;
    niter         = niter < 12 ? 12 : niter;

    for (int32_t i = 0; i < niter; i++) chess_loop_range(12, )
    chess_unroll_loop(4)
    chess_prepare_for_pipelining
    {
        dvcharx vIn = dvuchar_load_di(a_in);
        vstore(vIn.lo, a_outU);
        vstore(vIn.hi, a_outV);
    }
}

void calc_cords_SP_init(CalcCordsParamSP *param, int32_t *cordsSP_mask, const int32_t imageWidth,
                        const int32_t imageHeight)
{
    const int32_t vecw   = 16;
    const int32_t niter1 = (2 * TILE_W + vecw - 1) / vecw;
    const int32_t niter2 = TILE_H / 2;
    const int32_t niter  = niter1 * niter2;

    AgenWrapper w0;
    // will update base ptr while executing
    agen a_in = init(NULL);
    w0.size   = sizeof(int32_t);
    // fetch top and bottom lines
    w0.n1 = 2;
    // line pitch is 2 * TILE_W since x/y in 32Byte interleaved format
    w0.s1 = TILE_W * 2;
    w0.n2 = 2;
    w0.s2 = 0;
    w0.n3 = niter1;
    w0.s3 = vecw;
    w0.n4 = niter2;
    w0.s4 = TILE_W * 4;
    INIT_AGEN4(a_in, w0);

    AgenWrapper w1;
    // will update base ptr while executing
    agen a_out = init(NULL);
    w1.size    = sizeof(int32_t);
    w1.n1      = niter1;
    w1.s1      = vecw / 2;
    w1.n2      = niter2;
    w1.s2      = TILE_W;
    INIT_AGEN2(a_out, w1);

    // one of 2x2 is out of range means the subsample pixel is out of range
    AgenWrapper w2;
    agen a_mask = init(cordsSP_mask);
    w2.size     = sizeof(int32_t);
    w2.n1       = niter1;
    w2.s1       = vecw / 2;
    w2.n2       = niter2;
    w2.s2       = TILE_W;
    INIT_AGEN2(a_mask, w2);

    param->niter       = niter;
    param->a_in        = extract_agen_cfg(a_in);
    param->a_out       = extract_agen_cfg(a_out);
    param->a_mask      = extract_agen_cfg(a_mask);
    param->imageWidth  = imageWidth;
    param->imageHeight = imageHeight;
}

void calc_cords_SP_exec(CalcCordsParamSP *param, int32_t *cords_SP, int32_t *cords)
{
    const int32_t niter = param->niter;
    cupvaModifyAgenCfgBase(&param->a_in, cords);
    cupvaModifyAgenCfgBase(&param->a_out, cords_SP);

    agen a_in   = init_agen_from_cfg(param->a_in);
    agen a_out  = init_agen_from_cfg(param->a_out);
    agen a_mask = init_agen_from_cfg(param->a_mask);

    const int32_t imageWidth_q  = param->imageWidth << QBITS;
    const int32_t imageHeight_q = param->imageHeight << QBITS;

    dvintx vBound;
    vBound.lo = replicatew(imageWidth_q);
    vBound.hi = replicatew(imageHeight_q);

    dvintx vZero = chess_dont_care(dvintx) & 0;

    // 1/2 in Q format
    const int32_t half_q = 1 << (QBITS - 1);
    vintx vNegHalfQ      = replicatew(-half_q);
    vintx vTwo           = replicatew(2);

    // negative cords are treated as out of range
    vintx vMaskOut = replicatew(-1);

    // SP_idx = ((x_tl + x_tr + x_bl + x_br) * 0.25f - 0.5f) * 0.5f according to VPI implementation
    for (int32_t i = 0; i < niter; i++) chess_loop_range(6, )
    chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        // x/y in 32Byte interleaved format (lo: x0, x2, x4, x6, y0, y2, y4, y6, hi: x1, x3, x5, x7, y1, y3, y5, y7)
        dvintx vTop = dvint_load_di(a_in);
        // next line (lo: x0, x2, x4, x6, y0, y2, y4, y6, hi: x1, x3, x5, x7, y1, y3, y5, y7)
        dvintx vBottom = dvint_load_di(a_in);

        // need normal load for boundary check
        dvintx vTop_p    = dvint_load(a_in);
        dvintx vBottom_p = dvint_load(a_in);

        //dvintx vFlag = vTop_p >= vBound || vBottom_p >= vBound;
        //vFlag = vFlag || ((vTop_p < vZero) || (vBottom_p < vZero));
        dvintx vFlag0 = dvCmpLE_orL(vBound, vTop_p, (vBottom_p >= vBound));
        dvintx vFlag1 = dvCmpLT_orL(vBottom_p, vZero, vFlag0);
        dvintx vFlag  = dvCmpLT_orL(vTop_p, vZero, vFlag1);

        // combine x and y direction together
        vintx vmask = vFlag.lo || vFlag.hi;

        vintx vRes = vadd3(vTop.lo, vTop.hi, vBottom.lo);
        // 1/4 with rounding
        //vRes = (vRes + 0x2) >> 2;
        //vRes = vRes - half_q;
        vRes = vadd3(vRes, vBottom.hi, vTwo);
        vRes = vshiftadd(vRes, -2, vNegHalfQ);
        // 1/2 with rounding
        vRes = (vRes + 0x1) >> 1;
        // turn the results into x/y elements interleaved format
        vRes = vshuffle(vRes);

        vstore(vRes, a_out);

        // 2x2 region is out of range when either vFlag[2n] or vFlag[2n+1] are true
        vstore(vmask, a_mask);
    }
}

void merge_UV_init(MergeUVParam *param, uint8_t *srcU, uint8_t *srcV, uint8_t *dst)
{
    const int32_t vecw = 32;
    // downsample 2x2 for both U and V planes
    const int32_t size  = TILE_W * TILE_H / 4;
    const int32_t niter = (size + vecw - 1) / vecw;

    AgenWrapper w0;
    agen a_inU = init(srcU);
    w0.size    = sizeof(uint8_t);
    w0.n1      = niter;
    w0.s1      = vecw;
    INIT_AGEN1(a_inU, w0);

    agen a_inV = init(srcV);
    INIT_AGEN1(a_inV, w0);

    agen a_out = init(dst);
    w0.size    = sizeof(uint8_t);
    w0.n1      = niter;
    w0.s1      = vecw * 2;
    INIT_AGEN1(a_out, w0);

    param->niter = ((niter + 3) / 4) * 4;
    param->a_inU = extract_agen_cfg(a_inU);
    param->a_inV = extract_agen_cfg(a_inV);
    param->a_out = extract_agen_cfg(a_out);
}

void merge_UV_exec(MergeUVParam *param, uint8_t *dst)
{
    const int32_t niter = param->niter;
    cupvaModifyAgenCfgBase(&param->a_out, dst);
    agen a_inU = init_agen_from_cfg(param->a_inU);
    agen a_inV = init_agen_from_cfg(param->a_inV);
    agen a_out = init_agen_from_cfg(param->a_out);

    for (int32_t i = 0; i < niter; i++) chess_loop_range(12, )
    chess_unroll_loop(4)
    chess_prepare_for_pipelining
    {
        dvcharx vDst;
        vDst.lo = vuchar_load(a_inU);
        vDst.hi = vuchar_load(a_inV);
        vstore_i(vDst, a_out);
    }
}

void update_sampler(CupvaSampler *samplerCfg, uint8_t *src, int32_t *cords, uint8_t *dst, const int32_t tx,
                    const int32_t ty)
{
    cupvaSamplerUpdateAddrs(samplerCfg, src, cords, dst);
    cupvaSamplerUpdateInputAttributes(samplerCfg, tx, ty, tx);
    samplerCfg->x_int_limit = tx - 1;
    samplerCfg->y_int_limit = ty - 1;
}

void generate_cords_SP(CalcCordsParamSP *calcCordsParamSP, AddOfstCordsParamSP *addOfstCordsParamSP,
                       AddOfstCordsParam *addOfstCordsParam, TileInfo *tileInfoNext, TileInfo *tileInfoSP,
                       int32_t *cordsSP, int32_t *cords, int32_t *cords_rebased, const int32_t inImageWidth,
                       const int32_t inImageHeight)
{
    calc_cords_SP_exec(calcCordsParamSP, cordsSP, cords);
    add_ofst_cords_exec(addOfstCordsParam, cords_rebased, (tileInfoNext->xBegin) << QBITS,
                        (tileInfoNext->yBegin) << QBITS);
    get_tile_info_SP(tileInfoSP, cordsSP, inImageWidth / 2, inImageHeight / 2, QBITS);
    add_ofst_cords_SP_exec(addOfstCordsParamSP, cordsSP, (tileInfoSP->xBegin) << QBITS, (tileInfoSP->yBegin) << QBITS);
}