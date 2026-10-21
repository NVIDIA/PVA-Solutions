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

#include "corner_subpix_funcs.h"
#include "corner_subpix_knls.h"

#include <cupva_device.h>

VMEM(A, int32_t, num_corners);
VMEM(A, int32_t, img_w);
VMEM(A, int32_t, img_h);
VMEM(A, int32_t, win_size);
VMEM(A, int32_t, max_iters);
VMEM(A, float, eps);

VMEM_GSDF_HANDLER(A, gsdf_handler, BATCH);
VMEM_SEQDF_HANDLER(A, in_handler);
VMEM_SEQDF_HANDLER(A, out_handler);

VMEM(A, float, corners, MAX_CORNERS * 2);
VMEM(B, int32_t, corner_x, MAX_CORNERS);
VMEM(C, int32_t, corner_y, MAX_CORNERS);

VMEM(A, uint8_t, img_u8, MAX_TILE_PITCH *BATCH * 2);
VMEM(B, uint16_t, img_u16, MAX_TILE_PITCH *BATCH * 2);
VMEM(C, uint16_t, subpix_img, MAX_WIN_PITCH *MAX_WIN_PITCH *BATCH_PITCH * 2);
VMEM(B, int32_t, mask, MAX_WIN_SIZE *MAX_WIN_SIZE);

VMEM(B, int32_t, x_new, BATCH * 2);
VMEM(C, int32_t, y_new, BATCH * 2);
VMEM(B, int32_t, x_sat_lo, BATCH * 2);
VMEM(C, int32_t, y_sat_lo, BATCH * 2);
VMEM(B, int32_t, x_sat_hi, BATCH * 2);
VMEM(C, int32_t, y_sat_hi, BATCH * 2);
VMEM(B, int32_t, x_start, BATCH * 2);
VMEM(C, int32_t, y_start, BATCH * 2);
VMEM(B, int32_t, x_ofst, BATCH * 2);
VMEM(C, int32_t, y_ofst, BATCH * 2);
VMEM(A, int32_t, dlut_matrix, 2 * BATCH * 2);

VMEM(A, int32_t, gxx, MAX_WIN_SIZE *MAX_WIN_SIZE *BATCH);
VMEM(B, int32_t, gxy, MAX_WIN_SIZE *MAX_WIN_SIZE *BATCH);
VMEM(C, int32_t, gyy, MAX_WIN_SIZE *MAX_WIN_SIZE *BATCH);

VMEM(A, float, matrix, 3 * BATCH);
VMEM(C, float, bb, 2 * BATCH);

#define VMEM_A chess_storage(DMb % 64 : chess_segment(A))
param_corner_subpix_t VMEM_A param;
buf_corner_subpix_t VMEM_A buf[2];
CupvaSampler VMEM_A dlut_cfg[2];

CUPVA_VPU_MAIN()
{
    CupvaSamplerInput2D input_ping = {
        .data           = img_u16,
        .type           = SAMPLER_INPUT_TYPE_U16,
        .width          = (uint32_t)TILE_SIZE(win_size) - 1,
        .height         = (uint32_t)TILE_SIZE(win_size) - 1,
        .linePitch      = (uint32_t)TILE_SIZE(win_size),
        .inputAdv       = TILE_PITCH(win_size) * 2,
        .outOfRangeMode = SAMPLER_OUT_OF_RANGE_CONSTANT,
    };

    CupvaSamplerTiles tiles_ping = {
        .data              = dlut_matrix,
        .type              = SAMPLER_TILE_TRANS_ONLY,
        .width             = (uint8_t)WIN_PITCH(win_size),
        .height            = (uint8_t)WIN_PITCH(win_size),
        .count             = (uint16_t)BATCH,
        .fractionalBits    = (uint8_t)QBITS,
        .tileTraversalMode = SAMPLER_TRAVERSAL_TRANSPOSED,
    };

    CupvaSamplerOutput output_ping = {
        .data      = subpix_img,
        .pitch     = (uint16_t)BATCH_PITCH,
        .transMode = TRANS_MODE_1,
    };

    CupvaSamplerInput2D input_pong = {
        .data           = &img_u16[TILE_PITCH(win_size) * BATCH],
        .type           = SAMPLER_INPUT_TYPE_U16,
        .width          = (uint32_t)TILE_SIZE(win_size) - 1,
        .height         = (uint32_t)TILE_SIZE(win_size) - 1,
        .linePitch      = (uint32_t)TILE_SIZE(win_size),
        .inputAdv       = TILE_PITCH(win_size) * 2,
        .outOfRangeMode = SAMPLER_OUT_OF_RANGE_CONSTANT,
    };

    CupvaSamplerTiles tiles_pong = {
        .data              = &dlut_matrix[2 * BATCH],
        .type              = SAMPLER_TILE_TRANS_ONLY,
        .width             = (uint8_t)WIN_PITCH(win_size),
        .height            = (uint8_t)WIN_PITCH(win_size),
        .count             = (uint16_t)BATCH,
        .fractionalBits    = (uint8_t)QBITS,
        .tileTraversalMode = SAMPLER_TRAVERSAL_TRANSPOSED,
    };

    CupvaSamplerOutput output_pong = {
        .data      = &subpix_img[WIN_PITCH(win_size) * WIN_PITCH(win_size) * BATCH_PITCH],
        .pitch     = (uint16_t)BATCH_PITCH,
        .transMode = TRANS_MODE_1,
    };
    cupvaSamplerSetup(dlut_cfg, &input_ping, &tiles_ping, &output_ping);
    cupvaSamplerSetup(&dlut_cfg[1], &input_pong, &tiles_pong, &output_pong);

    param.corners      = corners;
    param.corner_x     = corner_x;
    param.corner_y     = corner_y;
    param.gxx          = gxx;
    param.gxy          = gxy;
    param.gyy          = gyy;
    param.mask         = mask;
    param.matrix       = matrix;
    param.bb           = bb;
    param.win_size     = win_size;
    param.tile_size    = TILE_SIZE(win_size);
    param.tile_pitch   = TILE_PITCH(win_size);
    param.tol          = TOL;
    param.gsdf_handler = gsdf_handler;
    param.in_handler   = &in_handler;
    param.out_handler  = &out_handler;

    buf[0].img_u8      = img_u8;
    buf[0].img_u16     = img_u16;
    buf[0].subpix_img  = subpix_img;
    buf[0].dlut_cfg    = dlut_cfg;
    buf[0].dlut_matrix = dlut_matrix;
    buf[0].x_new       = x_new;
    buf[0].y_new       = y_new;
    buf[0].x_sat_lo    = x_sat_lo;
    buf[0].y_sat_lo    = y_sat_lo;
    buf[0].x_sat_hi    = x_sat_hi;
    buf[0].y_sat_hi    = y_sat_hi;
    buf[0].x_start     = x_start;
    buf[0].y_start     = y_start;
    buf[0].x_ofst      = x_ofst;
    buf[0].y_ofst      = y_ofst;

    buf[1].img_u8      = img_u8 + TILE_PITCH(win_size) * BATCH;
    buf[1].img_u16     = img_u16 + TILE_PITCH(win_size) * BATCH;
    buf[1].subpix_img  = subpix_img + WIN_PITCH(win_size) * WIN_PITCH(win_size) * BATCH_PITCH;
    buf[1].dlut_cfg    = &dlut_cfg[1];
    buf[1].dlut_matrix = dlut_matrix + 2 * BATCH;
    buf[1].x_new       = x_new + BATCH;
    buf[1].y_new       = y_new + BATCH;
    buf[1].x_sat_lo    = x_sat_lo + BATCH;
    buf[1].y_sat_lo    = y_sat_lo + BATCH;
    buf[1].x_sat_hi    = x_sat_hi + BATCH;
    buf[1].y_sat_hi    = y_sat_hi + BATCH;
    buf[1].x_start     = x_start + BATCH;
    buf[1].y_start     = y_start + BATCH;
    buf[1].x_ofst      = x_ofst + BATCH;
    buf[1].y_ofst      = y_ofst + BATCH;

    cupvaSQDFOpen(in_handler);
    cupvaSQDFOpen(out_handler);
    cupvaGSDFOpen(gsdf_handler, img_u8, TILE_SIZE(win_size), TILE_PITCH(win_size));
    corner_subpix(&param, buf, num_corners, img_w, img_h, max_iters, eps);
    cupvaGSDFClose(gsdf_handler);
    cupvaSQDFClose(in_handler);
    cupvaSQDFClose(out_handler);

    return 0;
}
