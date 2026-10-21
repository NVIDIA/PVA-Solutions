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

static inline void init_gsdf_ofst(int32_t *corner_x, int32_t *corner_y, int32_t *x_ofst_ping, int32_t *y_ofst_ping,
                                  int32_t *x_ofst_pong, int32_t *y_ofst_pong, const int32_t ofst)
{
    int32_t vecw = pva_elementsof(dvintx);

    dvintx vx0 = sign_extend(*((dvint *)(corner_x)));
    dvintx vx1 = sign_extend(*((dvint *)(corner_x + vecw)));
    dvintx vx2 = sign_extend(*((dvint *)(corner_x + vecw * 2)));
    dvintx vx3 = sign_extend(*((dvint *)(corner_x + vecw * 3)));
    dvintx vy0 = sign_extend(*((dvint *)(corner_y)));
    dvintx vy1 = sign_extend(*((dvint *)(corner_y + vecw)));
    dvintx vy2 = sign_extend(*((dvint *)(corner_y + vecw * 2)));
    dvintx vy3 = sign_extend(*((dvint *)(corner_y + vecw * 3)));

    vx0 = vx0 >> QBITS;
    vx1 = vx1 >> QBITS;
    vx2 = vx2 >> QBITS;
    vx3 = vx3 >> QBITS;
    vy0 = vy0 >> QBITS;
    vy1 = vy1 >> QBITS;
    vy2 = vy2 >> QBITS;
    vy3 = vy3 >> QBITS;

    *((dvint *)(x_ofst_ping))        = extract(vx0 - ofst);
    *((dvint *)(x_ofst_ping + vecw)) = extract(vx1 - ofst);
    *((dvint *)(y_ofst_ping))        = extract(vy0 - ofst);
    *((dvint *)(y_ofst_ping + vecw)) = extract(vy1 - ofst);
    *((dvint *)(x_ofst_pong))        = extract(vx2 - ofst);
    *((dvint *)(x_ofst_pong + vecw)) = extract(vx3 - ofst);
    *((dvint *)(y_ofst_pong))        = extract(vy2 - ofst);
    *((dvint *)(y_ofst_pong + vecw)) = extract(vy3 - ofst);
}

int32_t corner_subpix(param_corner_subpix_t *param, buf_corner_subpix_t *buf, const int32_t num_corners_input,
                      const int32_t img_w, const int32_t img_h, const int32_t max_iters, const float eps)
{
    cupvaSQDFUpdateTileSize(*(param->in_handler), 0, 2 * sizeof(float), num_corners_input);
    cupvaSQDFFlushAndTrig(*(param->in_handler));
    cupvaSQDFUpdateTileSize(*(param->out_handler), 0, 2 * sizeof(float), num_corners_input);

    int32_t num_corners = (num_corners_input + 2 * BATCH - 1) / (2 * BATCH) * (2 * BATCH);

    buf_corner_subpix_t *buf_ping = buf;
    buf_corner_subpix_t *buf_pong = buf + 1;

    int32_t *x_ofst_ping = buf_ping->x_ofst;
    int32_t *y_ofst_ping = buf_ping->y_ofst;

    int32_t *x_ofst_pong = buf_pong->x_ofst;
    int32_t *y_ofst_pong = buf_pong->y_ofst;

    int32_t *status_ping   = buf_ping->status;
    int32_t *converge_ping = buf_ping->converge;
    int32_t *x_new_ping    = buf_ping->x_new;
    int32_t *y_new_ping    = buf_ping->y_new;
    int32_t *x_sat_lo_ping = buf_ping->x_sat_lo;
    int32_t *y_sat_lo_ping = buf_ping->y_sat_lo;
    int32_t *x_sat_hi_ping = buf_ping->x_sat_hi;
    int32_t *y_sat_hi_ping = buf_ping->y_sat_hi;
    int32_t *x_start_ping  = buf_ping->x_start;
    int32_t *y_start_ping  = buf_ping->y_start;

    int32_t *status_pong   = buf_pong->status;
    int32_t *converge_pong = buf_pong->converge;
    int32_t *x_new_pong    = buf_pong->x_new;
    int32_t *y_new_pong    = buf_pong->y_new;
    int32_t *x_sat_lo_pong = buf_pong->x_sat_lo;
    int32_t *y_sat_lo_pong = buf_pong->y_sat_lo;
    int32_t *x_sat_hi_pong = buf_pong->x_sat_hi;
    int32_t *y_sat_hi_pong = buf_pong->y_sat_hi;
    int32_t *x_start_pong  = buf_pong->x_start;
    int32_t *y_start_pong  = buf_pong->y_start;

    int32_t tile_pitch = param->tile_pitch;
    int32_t tile_size  = param->tile_size;
    int32_t win_size   = param->win_size;
    int32_t tol        = param->tol;
    int32_t *corner_x  = param->corner_x;
    int32_t *corner_y  = param->corner_y;
    int32_t ofst       = (tile_size >> 1) - 1;

    cupvaSQDFSync(*(param->in_handler));

    corner_conversion_preprocess(param, num_corners, num_corners_input);

    init_gsdf_ofst(corner_x, corner_y, x_ofst_ping, y_ofst_ping, x_ofst_pong, y_ofst_pong, ofst);
    cupvaGSDFUpdateTiles(param->gsdf_handler, x_ofst_ping, y_ofst_ping, BATCH);
    cupvaGSDFTrig(param->gsdf_handler);

    bilinear_interpolation_index_init(buf_ping);
    bilinear_interpolation_index_init(buf_pong);
    image_u8_u16_init(param, buf_ping);
    image_u8_u16_init(param, buf_pong);
    hessian_matrix_init(param, buf_ping);
    hessian_matrix_init(param, buf_pong);

    cupvaGSDFUpdateVmemAddr(param->gsdf_handler, buf_pong->img_u8, tile_size, tile_pitch);
    cupvaGSDFUpdateTiles(param->gsdf_handler, x_ofst_pong, y_ofst_pong, BATCH);

    cupvaGSDFSync(param->gsdf_handler);
    cupvaGSDFTrig(param->gsdf_handler);

    int32_t vecw   = pva_elementsof(dvintx); // 16
    int32_t win_w  = (win_size >> 1) << QBITS;
    int32_t win_h  = (win_size >> 1) << QBITS;
    int32_t ofst_Q = (ofst - (win_size >> 1) - 1) << QBITS; // - half_win_pitch in advance
    int32_t tol_Q  = tol << QBITS;
    int32_t one_Q  = 1 << QBITS;

    // num_corners = multiples of BATCH * 2
    for (int32_t corner_idx = 0; corner_idx < num_corners; corner_idx += BATCH * 2)
    {
        image_u8_u16_exec(param, buf_ping);

        int32_t *corner_x_cur = corner_x + corner_idx;
        int32_t *corner_y_cur = corner_y + corner_idx;
        // Initialize status, converge, corners
        {
            dvintx v0  = chess_dont_care(dvintx) & 0;
            dvintx vx0 = sign_extend(*((dvint *)(corner_x_cur)));
            dvintx vx1 = sign_extend(*((dvint *)(corner_x_cur + vecw)));
            dvintx vx2 = sign_extend(*((dvint *)(corner_x_cur + vecw * 2)));
            dvintx vx3 = sign_extend(*((dvint *)(corner_x_cur + vecw * 3)));
            dvintx vy0 = sign_extend(*((dvint *)(corner_y_cur)));
            dvintx vy1 = sign_extend(*((dvint *)(corner_y_cur + vecw)));
            dvintx vy2 = sign_extend(*((dvint *)(corner_y_cur + vecw * 2)));
            dvintx vy3 = sign_extend(*((dvint *)(corner_y_cur + vecw * 3)));

            buf_ping->all_converged[0] = 0;
            buf_ping->all_converged[1] = 0;
            buf_pong->all_converged[0] = 0;
            buf_pong->all_converged[1] = 0;

            *((dvint *)(status_ping))          = extract(v0);
            *((dvint *)(status_ping + vecw))   = extract(v0);
            *((dvint *)(converge_ping))        = extract(v0);
            *((dvint *)(converge_ping + vecw)) = extract(v0);
            *((dvint *)(x_new_ping))           = extract(vx0);
            *((dvint *)(x_new_ping + vecw))    = extract(vx1);
            *((dvint *)(y_new_ping))           = extract(vy0);
            *((dvint *)(y_new_ping + vecw))    = extract(vy1);
            *((dvint *)(x_sat_lo_ping))        = extract(vx0 - win_w - tol_Q);
            *((dvint *)(x_sat_lo_ping + vecw)) = extract(vx1 - win_w - tol_Q);
            *((dvint *)(x_sat_hi_ping))        = extract(vx0 + win_w + tol_Q + one_Q);
            *((dvint *)(x_sat_hi_ping + vecw)) = extract(vx1 + win_w + tol_Q + one_Q);
            *((dvint *)(y_sat_lo_ping))        = extract(vy0 - win_h - tol_Q);
            *((dvint *)(y_sat_lo_ping + vecw)) = extract(vy1 - win_h - tol_Q);
            *((dvint *)(y_sat_hi_ping))        = extract(vy0 + win_h + tol_Q + one_Q);
            *((dvint *)(y_sat_hi_ping + vecw)) = extract(vy1 + win_h + tol_Q + one_Q);
            *((dvint *)(x_start_ping))         = extract(vx0 - ofst_Q);
            *((dvint *)(x_start_ping + vecw))  = extract(vx1 - ofst_Q);
            *((dvint *)(y_start_ping))         = extract(vy0 - ofst_Q);
            *((dvint *)(y_start_ping + vecw))  = extract(vy1 - ofst_Q);

            *((dvint *)(status_pong))          = extract(v0);
            *((dvint *)(status_pong + vecw))   = extract(v0);
            *((dvint *)(converge_pong))        = extract(v0);
            *((dvint *)(converge_pong + vecw)) = extract(v0);
            *((dvint *)(x_new_pong))           = extract(vx2);
            *((dvint *)(x_new_pong + vecw))    = extract(vx3);
            *((dvint *)(y_new_pong))           = extract(vy2);
            *((dvint *)(y_new_pong + vecw))    = extract(vy3);
            *((dvint *)(x_sat_lo_pong))        = extract(vx2 - win_w - tol_Q);
            *((dvint *)(x_sat_lo_pong + vecw)) = extract(vx3 - win_w - tol_Q);
            *((dvint *)(x_sat_hi_pong))        = extract(vx2 + win_w + tol_Q + one_Q);
            *((dvint *)(x_sat_hi_pong + vecw)) = extract(vx3 + win_w + tol_Q + one_Q);
            *((dvint *)(y_sat_lo_pong))        = extract(vy2 - win_h - tol_Q);
            *((dvint *)(y_sat_lo_pong + vecw)) = extract(vy3 - win_h - tol_Q);
            *((dvint *)(y_sat_hi_pong))        = extract(vy2 + win_h + tol_Q + one_Q);
            *((dvint *)(y_sat_hi_pong + vecw)) = extract(vy3 + win_h + tol_Q + one_Q);
            *((dvint *)(x_start_pong))         = extract(vx2 - ofst_Q);
            *((dvint *)(x_start_pong + vecw))  = extract(vx3 - ofst_Q);
            *((dvint *)(y_start_pong))         = extract(vy2 - ofst_Q);
            *((dvint *)(y_start_pong + vecw))  = extract(vy3 - ofst_Q);
        }

        bilinear_interpolation_index_exec(buf_ping);
        cupvaSamplerStart(buf_ping->dlut_cfg);

        cupvaGSDFSync(param->gsdf_handler);

        if (corner_idx < num_corners - 2 * BATCH)
        {
            init_gsdf_ofst(corner_x_cur + 2 * BATCH, corner_y_cur + 2 * BATCH, x_ofst_ping, y_ofst_ping, x_ofst_pong,
                           y_ofst_pong, ofst);
            cupvaGSDFUpdateVmemAddr(param->gsdf_handler, buf_ping->img_u8, tile_size, tile_pitch);
            cupvaGSDFUpdateTiles(param->gsdf_handler, x_ofst_ping, y_ofst_ping, BATCH);
            cupvaGSDFTrig(param->gsdf_handler);
        }

        image_u8_u16_exec(param, buf_pong);

        for (int32_t i = 0; i < max_iters; i++)
        {
            int32_t condition_ping = buf_ping->all_converged[0] & buf_ping->all_converged[1];
            int32_t condition_pong = buf_pong->all_converged[0] & buf_pong->all_converged[1];
            if (condition_ping == 0)
            {
                cupvaSamplerWait();
            }
            if (condition_pong == 0)
            {
                bilinear_interpolation_index_exec(buf_pong);
                cupvaSamplerStart(buf_pong->dlut_cfg);
            }

            if (i == PREFETCH_ITER)
            {
                if (corner_idx < num_corners - 2 * BATCH)
                {
                    cupvaGSDFSync(param->gsdf_handler);
                    cupvaGSDFUpdateVmemAddr(param->gsdf_handler, buf_pong->img_u8, tile_size, tile_pitch);
                    cupvaGSDFUpdateTiles(param->gsdf_handler, x_ofst_pong, y_ofst_pong, BATCH);
                    cupvaGSDFTrig(param->gsdf_handler);
                }
            }

            if (condition_ping == 0)
            {
                hessian_matrix_exec(param, buf_ping);

                compute_motion_exec(param, buf_ping, img_w, img_h, eps);
            }

            if (condition_pong == 0)
            {
                cupvaSamplerWait();
            }

            if (((buf_ping->all_converged[0] & buf_ping->all_converged[1]) == 0) && (i < max_iters - 1))
            {
                bilinear_interpolation_index_exec(buf_ping);
                cupvaSamplerStart(buf_ping->dlut_cfg);
            }

            if (condition_pong == 0)
            {
                hessian_matrix_exec(param, buf_pong);

                compute_motion_exec(param, buf_pong, img_w, img_h, eps);
            }
        }

        // If new point is too far from initial, it means poor convergence.
        // Leave initial point as the result
        {
            dvintx v_win_h;
            v_win_h.lo = replicatew(win_h);
            v_win_h.hi = replicatew(win_h);

            dvintx vx0     = sign_extend(*((dvint *)(corner_x_cur)));
            dvintx vx1     = sign_extend(*((dvint *)(corner_x_cur + vecw)));
            dvintx vx2     = sign_extend(*((dvint *)(corner_x_cur + vecw * 2)));
            dvintx vx3     = sign_extend(*((dvint *)(corner_x_cur + vecw * 3)));
            dvintx vy0     = sign_extend(*((dvint *)(corner_y_cur)));
            dvintx vy1     = sign_extend(*((dvint *)(corner_y_cur + vecw)));
            dvintx vy2     = sign_extend(*((dvint *)(corner_y_cur + vecw * 2)));
            dvintx vy3     = sign_extend(*((dvint *)(corner_y_cur + vecw * 3)));
            dvintx vx0_new = sign_extend(*((dvint *)(x_new_ping)));
            dvintx vx1_new = sign_extend(*((dvint *)(x_new_ping + vecw)));
            dvintx vx2_new = sign_extend(*((dvint *)(x_new_pong)));
            dvintx vx3_new = sign_extend(*((dvint *)(x_new_pong + vecw)));
            dvintx vy0_new = sign_extend(*((dvint *)(y_new_ping)));
            dvintx vy1_new = sign_extend(*((dvint *)(y_new_ping + vecw)));
            dvintx vy2_new = sign_extend(*((dvint *)(y_new_pong)));
            dvintx vy3_new = sign_extend(*((dvint *)(y_new_pong + vecw)));

            dvintx tmp0  = dvabsdif(vx0_new, vx0) > win_w;
            dvintx pred0 = dvCmpLT_orL(v_win_h, dvabsdif(vy0_new, vy0), tmp0);
            dvintx tmp1  = dvabsdif(vx1_new, vx1) > win_w;
            dvintx pred1 = dvCmpLT_orL(v_win_h, dvabsdif(vy1_new, vy1), tmp1);
            dvintx tmp2  = dvabsdif(vx2_new, vx2) > win_w;
            dvintx pred2 = dvCmpLT_orL(v_win_h, dvabsdif(vy2_new, vy2), tmp2);
            dvintx tmp3  = dvabsdif(vx3_new, vx3) > win_w;
            dvintx pred3 = dvCmpLT_orL(v_win_h, dvabsdif(vy3_new, vy3), tmp3);

            vx0_new = dvmux(pred0, vx0, vx0_new);
            vy0_new = dvmux(pred0, vy0, vy0_new);
            vx1_new = dvmux(pred1, vx1, vx1_new);
            vy1_new = dvmux(pred1, vy1, vy1_new);
            vx2_new = dvmux(pred2, vx2, vx2_new);
            vy2_new = dvmux(pred2, vy2, vy2_new);
            vx3_new = dvmux(pred3, vx3, vx3_new);
            vy3_new = dvmux(pred3, vy3, vy3_new);

            *((dvint *)(corner_x_cur))            = extract(vx0_new);
            *((dvint *)(corner_x_cur + vecw))     = extract(vx1_new);
            *((dvint *)(corner_x_cur + vecw * 2)) = extract(vx2_new);
            *((dvint *)(corner_x_cur + vecw * 3)) = extract(vx3_new);
            *((dvint *)(corner_y_cur))            = extract(vy0_new);
            *((dvint *)(corner_y_cur + vecw))     = extract(vy1_new);
            *((dvint *)(corner_y_cur + vecw * 2)) = extract(vy2_new);
            *((dvint *)(corner_y_cur + vecw * 3)) = extract(vy3_new);
        }
    }
    corner_conversion_postprocess(param, num_corners);

    cupvaSQDFFlushAndTrig(*(param->out_handler));
    cupvaSQDFSync(*(param->out_handler));
    return 0;
}