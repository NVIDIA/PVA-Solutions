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

#include "corner_subpix_knls.h"

#include <float.h>

#define THRESHOLD (FLT_EPSILON * FLT_EPSILON)
#define OPT 0 // Performance optimization option: 1 -- set status to invalid when new point exceeds tile boundary

int32_t corner_conversion_preprocess(param_corner_subpix_t *param, const int32_t num_corners,
                                     const int32_t num_corners_input)
{
    int32_t vecw      = pva_elementsof(vintx); // 8
    int32_t *corner_x = param->corner_x;
    int32_t *corner_y = param->corner_y;
    float *corners    = param->corners;
    int32_t niter     = num_corners / vecw;

    for (int32_t i = num_corners_input; i < num_corners; i++)
    {
        corners[2 * i]     = corners[num_corners_input * 2 - 2];
        corners[2 * i + 1] = corners[num_corners_input * 2 - 1];
    }

    AgenWrapper wrapper;
    agen a_in    = init((dvfloat *)corners);
    wrapper.size = sizeof(float);
    wrapper.n1   = niter;
    wrapper.s1   = vecw * 2;
    INIT_AGEN1(a_in, wrapper);

    agen a_x   = init((dvint *)corner_x);
    wrapper.n1 = niter;
    wrapper.s1 = vecw;
    INIT_AGEN1(a_x, wrapper);

    agen a_y   = init((dvint *)corner_y);
    wrapper.n1 = niter;
    wrapper.s1 = vecw;
    INIT_AGEN1(a_y, wrapper);

    chess_separator_scheduler();

    niter       = (niter + 3) / 4 * 4;
    niter       = niter > 12 ? niter : 12;
    float scale = 1 << QBITS;

    for (int32_t i = 0; i < niter; i++) chess_loop_range(12, )
    chess_unroll_loop(4)
    chess_prepare_for_pipelining
    {
        dvfloatx corner = dvfloat_load_di(a_in);

        corner *= scale;
        dvintx out = dvfp_dvint_trunc(corner);

        vstore(out.lo, a_x);
        vstore(out.hi, a_y);
    }
    return 0;
}

int32_t corner_conversion_postprocess(param_corner_subpix_t *param, const int32_t num_corners)
{
    int32_t vecw      = pva_elementsof(vintx); // 8
    int32_t *corner_x = param->corner_x;
    int32_t *corner_y = param->corner_y;
    float *corners    = param->corners;
    int32_t niter     = num_corners / vecw;

    AgenWrapper wrapper;
    agen a_out   = init((dvfloat *)corners);
    wrapper.size = sizeof(float);
    wrapper.n1   = niter;
    wrapper.s1   = vecw * 2;
    INIT_AGEN1(a_out, wrapper);

    agen a_x     = init((dvint *)corner_x);
    wrapper.size = sizeof(int32_t);
    wrapper.n1   = niter;
    wrapper.s1   = vecw;
    INIT_AGEN1(a_x, wrapper);

    agen a_y   = init((dvint *)corner_y);
    wrapper.n1 = niter;
    wrapper.s1 = vecw;
    INIT_AGEN1(a_y, wrapper);

    chess_separator_scheduler();

    niter       = (niter + 3) / 4 * 4;
    niter       = niter > 12 ? niter : 12;
    float scale = 1.f / (1 << QBITS);

    for (int32_t i = 0; i < niter; i++) chess_loop_range(12, )
    chess_unroll_loop(4)
    chess_prepare_for_pipelining
    {
        vintx vx = vint_load(a_x);
        vintx vy = vint_load(a_y);

        dvfloatx vxy;
        vxy.lo = vint_vfp(vx);
        vxy.hi = vint_vfp(vy);

        vxy *= scale;

        vstore_i(vxy, a_out);
    }
    return 0;
}

void image_u8_u16_init(param_corner_subpix_t *param, buf_corner_subpix_t *buf)
{
    int32_t vecw       = pva_elementsof(dvshortx); // 32
    uint8_t *img_u8    = buf->img_u8;
    uint16_t *img_u16  = buf->img_u16;
    int32_t tile_pitch = param->tile_pitch;
    int32_t niter      = tile_pitch / vecw;

    AgenWrapper wrapper;
    agen a_in    = init((vuchar *)img_u8);
    wrapper.size = sizeof(uint8_t);
    wrapper.n1   = niter;
    wrapper.s1   = vecw;
    wrapper.n2   = BATCH;
    wrapper.s2   = tile_pitch;
    INIT_AGEN2(a_in, wrapper);

    agen a_out   = init((dvushort *)img_u16);
    wrapper.size = sizeof(uint16_t);
    wrapper.n1   = niter;
    wrapper.s1   = vecw;
    wrapper.n2   = BATCH;
    wrapper.s2   = tile_pitch;
    INIT_AGEN2(a_out, wrapper);

    buf->image_cfgs[0] = extract_agen_cfg(a_in);
    buf->image_cfgs[1] = extract_agen_cfg(a_out);
    param->image_niter = niter * BATCH;
}

int32_t image_u8_u16_exec(param_corner_subpix_t *param, buf_corner_subpix_t *buf)
{
    agen a_in     = init_agen_from_cfg(buf->image_cfgs[0]);
    agen a_out    = init_agen_from_cfg(buf->image_cfgs[1]);
    int32_t niter = param->image_niter;

    for (int32_t i = 0; i < niter; i++) chess_loop_range(24, )
    chess_unroll_loop(8)
    chess_prepare_for_pipelining
    {
        dvshortx vin  = vuchar_dvshortx_load(a_in);
        dvshortx vout = vin << QBITS_IMG;
        vstore(vout, a_out);
    }
    return 0;
}

void bilinear_interpolation_index_init(buf_corner_subpix_t *buf)
{
    int32_t vecw         = pva_elementsof(dvintx); // 16
    int32_t *dlut_matrix = buf->dlut_matrix;

    AgenWrapper wrapper;
    agen a_ofst  = init((dvint *)dlut_matrix);
    wrapper.size = sizeof(int32_t);
    wrapper.n1   = 2 * 2;
    wrapper.s1   = vecw;
    INIT_AGEN1(a_ofst, wrapper);
    buf->image_cfgs[2] = extract_agen_cfg(a_ofst);
}

int32_t bilinear_interpolation_index_exec(buf_corner_subpix_t *buf)
{
    agen a_ofst      = init_agen_from_cfg(buf->image_cfgs[2]);
    int32_t *x_new   = buf->x_new;
    int32_t *y_new   = buf->y_new;
    int32_t *x_start = buf->x_start;
    int32_t *y_start = buf->y_start;
    int32_t vecw     = pva_elementsof(dvintx); // 16

    for (int32_t i = 0; i < 2; i++) chess_flatten_loop
    {
        dvintx vx       = sign_extend(*((dvint *)(x_new + i * vecw)));
        dvintx vy       = sign_extend(*((dvint *)(y_new + i * vecw)));
        dvintx vx_start = sign_extend(*((dvint *)(x_start + i * vecw)));
        dvintx vy_start = sign_extend(*((dvint *)(y_start + i * vecw)));

        dvintx x_ofst = vx - vx_start;
        dvintx y_ofst = vy - vy_start;

        dvintx vout0, vout1;
        vout0.lo = x_ofst.lo;
        vout0.hi = y_ofst.lo;
        vout1.lo = x_ofst.hi;
        vout1.hi = y_ofst.hi;

        vstore_i(vout0, a_ofst);
        vstore_i(vout1, a_ofst);
    }
    return 0;
}

void hessian_matrix_init(param_corner_subpix_t *param, buf_corner_subpix_t *buf)
{
    int32_t vecw      = pva_elementsof(dvintx); // 16
    int32_t win_size  = param->win_size;
    int32_t win_pitch = win_size + 2;

    uint16_t *in  = buf->subpix_img;
    int32_t *gxx  = param->gxx;
    int32_t *gxy  = param->gxy;
    int32_t *gyy  = param->gyy;
    float *matrix = param->matrix;
    float *bb     = param->bb;

    AgenWrapper wrapper;

    agen a_x     = init((dvushort *)(in + BATCH_PITCH));
    wrapper.size = sizeof(uint16_t);
    wrapper.n1   = 2;
    wrapper.s1   = 2 * win_pitch * BATCH_PITCH;
    wrapper.n2   = win_size;
    wrapper.s2   = win_pitch * BATCH_PITCH;
    wrapper.n3   = win_size;
    wrapper.s3   = BATCH_PITCH;
    INIT_AGEN3(a_x, wrapper);

    agen a_y   = init((dvushort *)(in + win_pitch * BATCH_PITCH));
    wrapper.n1 = 2;
    wrapper.s1 = 2 * BATCH_PITCH;
    wrapper.n2 = win_size;
    wrapper.s2 = win_pitch * BATCH_PITCH;
    wrapper.n3 = win_size;
    wrapper.s3 = BATCH_PITCH;
    INIT_AGEN3(a_y, wrapper);

    agen a_gxx   = init((dvint *)(gxx));
    wrapper.size = sizeof(int32_t);
    wrapper.n1   = 2 * win_size * win_size;
    wrapper.s1   = vecw;
    INIT_AGEN1(a_gxx, wrapper);
    a_gxx.round = 16;

    agen a_gxy = init((dvint *)(gxy));
    INIT_AGEN1(a_gxy, wrapper);
    a_gxy.round = 16;

    agen a_gyy = init((dvint *)(gyy));
    INIT_AGEN1(a_gyy, wrapper);
    a_gyy.round = 16;

    agen a_matrix = init((dvfloat *)(matrix));
    wrapper.size  = sizeof(float);
    wrapper.n1    = 3 * 2;
    wrapper.s1    = vecw;
    INIT_AGEN1(a_matrix, wrapper);

    agen a_bb  = init((dvfloat *)(bb));
    wrapper.n1 = 2 * 2;
    wrapper.s1 = vecw;
    INIT_AGEN1(a_bb, wrapper);

    buf->image_cfgs[3]     = extract_agen_cfg(a_x);
    buf->image_cfgs[4]     = extract_agen_cfg(a_y);
    param->hessian_cfgs[0] = extract_agen_cfg(a_gxx);
    param->hessian_cfgs[1] = extract_agen_cfg(a_gxy);
    param->hessian_cfgs[2] = extract_agen_cfg(a_gyy);
    param->hessian_cfgs[3] = extract_agen_cfg(a_matrix);
    param->hessian_cfgs[4] = extract_agen_cfg(a_bb);
}

int32_t hessian_matrix_exec(param_corner_subpix_t *param, buf_corner_subpix_t *buf)
{
    agen a_x   = init_agen_from_cfg(buf->image_cfgs[3]);
    agen a_y   = init_agen_from_cfg(buf->image_cfgs[4]);
    agen a_gxx = init_agen_from_cfg(param->hessian_cfgs[0]);
    agen a_gxy = init_agen_from_cfg(param->hessian_cfgs[1]);
    agen a_gyy = init_agen_from_cfg(param->hessian_cfgs[2]);

    int32_t *mask    = param->mask;
    int32_t win_size = param->win_size;
    int32_t niter    = win_size * win_size;

    int32_t niter1 = (niter + 1) / 2 * 2;
    niter1         = niter1 > 6 ? niter1 : 6;

    for (int32_t i = 0; i < niter1; i++) chess_loop_range(6, )
    chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        dvshortx in0 = dvushort_load(a_x);
        dvshortx in1 = dvushort_load(a_x);
        dvshortx in2 = dvushort_load(a_y);
        dvshortx in3 = dvushort_load(a_y);
        int32_t m    = *mask++;

        dvshortx gx = in1 - in0; // 16 bit + 1 sign
        dvshortx gy = in3 - in2; // 16 bit + 1 sign

        // deinterleaving
        dvintx gxx0 = vmulhw(gx.lo, gx.lo, VPU_ROUND_0); // 32 bit + 1 sign
        dvintx gxx1 = vmulhw(gx.hi, gx.hi, VPU_ROUND_0);

        dvintx gyy0 = vmulhw(gy.lo, gy.lo, VPU_ROUND_0);
        dvintx gyy1 = vmulhw(gy.hi, gy.hi, VPU_ROUND_0);

        dvintx gxy0 = vmulhw(gx.lo, gy.lo, VPU_ROUND_0);
        dvintx gxy1 = vmulhw(gx.hi, gy.hi, VPU_ROUND_0);

        gxx0 = dvmulwhw(gxx0, m); // agen round 16 bits
        gxx1 = dvmulwhw(gxx1, m);

        gxy0 = dvmulwhw(gxy0, m);
        gxy1 = dvmulwhw(gxy1, m);

        gyy0 = dvmulwhw(gyy0, m);
        gyy1 = dvmulwhw(gyy1, m);

        vstore_i(gxx0, a_gxx);
        vstore_i(gxx1, a_gxx);

        vstore_i(gxy0, a_gxy);
        vstore_i(gxy1, a_gxy);

        vstore_i(gyy0, a_gyy);
        vstore_i(gyy1, a_gyy);
    }
    chess_separator_scheduler();

    a_gxx = init_agen_from_cfg(param->hessian_cfgs[0]);
    a_gxy = init_agen_from_cfg(param->hessian_cfgs[1]);
    a_gyy = init_agen_from_cfg(param->hessian_cfgs[2]);

    int32_t half_win_size = win_size >> 1;
    int32_t x             = 0;
    int32_t y             = 0;
    int32_t pred_y        = 0;

    dvintx a_lo, a_hi;
    dvintx b_lo, b_hi;
    dvintx c_lo, c_hi;
    dvintx vzero = chess_dont_care(dvintx) & 0;
    a_lo         = vzero;
    a_hi         = vzero;
    b_lo         = vzero;
    b_hi         = vzero;
    c_lo         = vzero;
    c_hi         = vzero;

    dvintx bb1_lo = chess_dont_care(dvintx);
    dvintx bb1_hi = chess_dont_care(dvintx);
    dvintx bb2_lo = chess_dont_care(dvintx);
    dvintx bb2_hi = chess_dont_care(dvintx);

    for (int32_t i = 0; i < niter; i++) chess_loop_range(3, )
    chess_unroll_loop(1)
    chess_prepare_for_pipelining
    {
        dvintx gxx0 = dvint_load(a_gxx);
        dvintx gxy0 = dvint_load(a_gxy);
        dvintx gyy0 = dvint_load(a_gyy);

        dvintx gxx1 = dvint_load(a_gxx);
        dvintx gxy1 = dvint_load(a_gxy);
        dvintx gyy1 = dvint_load(a_gyy);

        bb1_lo = dvmaddwhw(gxx0, (x - half_win_size), bb1_lo, VPU_ROUND_0, i);
        bb2_lo = dvmaddwhw(gxy0, (x - half_win_size), bb2_lo, VPU_ROUND_0, i);

        bb1_lo = dvmaddwhw(gxy0, (y - half_win_size), bb1_lo, VPU_ROUND_0, 1);
        bb2_lo = dvmaddwhw(gyy0, (y - half_win_size), bb2_lo, VPU_ROUND_0, 1);

        bb1_hi = dvmaddwhw(gxx1, (x - half_win_size), bb1_hi, VPU_ROUND_0, i);
        bb2_hi = dvmaddwhw(gxy1, (x - half_win_size), bb2_hi, VPU_ROUND_0, i);

        bb1_hi = dvmaddwhw(gxy1, (y - half_win_size), bb1_hi, VPU_ROUND_0, 1);
        bb2_hi = dvmaddwhw(gyy1, (y - half_win_size), bb2_hi, VPU_ROUND_0, 1);

        a_lo += gxx0;
        b_lo += gxy0;
        c_lo += gyy0;

        a_hi += gxx1;
        b_hi += gxy1;
        c_hi += gyy1;

        x = mod_inc_pred_z(x, win_size - 1, pred_y);
        y = mux(pred_y, y + 1, y);
    }

    chess_separator_scheduler();
    agen a_matrix = init_agen_from_cfg(param->hessian_cfgs[3]);
    agen a_bb     = init_agen_from_cfg(param->hessian_cfgs[4]);

    dvfloatx a_f_lo   = dvintx_dvfp(a_lo);
    dvfloatx b_f_lo   = dvintx_dvfp(b_lo);
    dvfloatx c_f_lo   = dvintx_dvfp(c_lo);
    dvfloatx a_f_hi   = dvintx_dvfp(a_hi);
    dvfloatx b_f_hi   = dvintx_dvfp(b_hi);
    dvfloatx c_f_hi   = dvintx_dvfp(c_hi);
    dvfloatx bb1_f_lo = dvintx_dvfp(bb1_lo);
    dvfloatx bb2_f_lo = dvintx_dvfp(bb2_lo);
    dvfloatx bb1_f_hi = dvintx_dvfp(bb1_hi);
    dvfloatx bb2_f_hi = dvintx_dvfp(bb2_hi);

    vstore(a_f_lo, a_matrix);
    vstore(b_f_lo, a_matrix);
    vstore(c_f_lo, a_matrix);
    vstore(a_f_hi, a_matrix);
    vstore(b_f_hi, a_matrix);
    vstore(c_f_hi, a_matrix);
    vstore(bb1_f_lo, a_bb);
    vstore(bb2_f_lo, a_bb);
    vstore(bb1_f_hi, a_bb);
    vstore(bb2_f_hi, a_bb);

    return 0;
}

// follow opencv .contains()
static inline dvintx check_image_boundary(const int32_t img_w, const int32_t img_h, dvintx x, dvintx y, dvintx status)
{
    dvintx pred = (x < 0) | (x >= img_w) | (y < 0) | (y >= img_h);
    status      = dvmux(pred, 1, status); // 1 is invalid
    return status;
}

static inline dvintx check_tile_boundary(dvintx &x, dvintx &y, dvintx x_sat_lo, dvintx y_sat_lo, dvintx x_sat_hi,
                                         dvintx y_sat_hi)
{
    dvintx status;
#if OPT
    dvintx pred0 = x < x_sat_lo;
    dvintx pred1 = x > x_sat_hi;
    dvintx pred2 = y < y_sat_lo;
    dvintx pred3 = y > y_sat_hi;

    status = pred0 | pred1 | pred2 | pred3;
#endif
    x = dvmax(x_sat_lo, x);
    x = dvmin(x_sat_hi, x);
    y = dvmax(y_sat_lo, y);
    y = dvmin(y_sat_hi, y);

    return status;
}

int32_t compute_motion_exec(param_corner_subpix_t *param, buf_corner_subpix_t *buf, const int32_t img_w,
                            const int32_t img_h, const float eps)
{
    agen a_matrix = init_agen_from_cfg(param->hessian_cfgs[3]);
    agen a_bb     = init_agen_from_cfg(param->hessian_cfgs[4]);
    int32_t vecw  = pva_elementsof(dvintx); // 16
    float eps_2   = eps * eps;

    int32_t *status        = buf->status;
    int32_t *converge      = buf->converge;
    int32_t *all_converged = buf->all_converged;
    int32_t *x_new         = buf->x_new;
    int32_t *y_new         = buf->y_new;
    int32_t *x_sat_lo      = buf->x_sat_lo;
    int32_t *y_sat_lo      = buf->y_sat_lo;
    int32_t *x_sat_hi      = buf->x_sat_hi;
    int32_t *y_sat_hi      = buf->y_sat_hi;
    const int32_t img_w_Q  = img_w << QBITS;
    const int32_t img_h_Q  = img_h << QBITS;

    for (int32_t i = 0; i < 2; i++) chess_unroll_loop(*)
    {
        dvfloatx a       = dvfloat_load(a_matrix);
        dvfloatx b       = dvfloat_load(a_matrix);
        dvfloatx c       = dvfloat_load(a_matrix);
        dvfloatx bb1     = dvfloat_load(a_bb);
        dvfloatx bb2     = dvfloat_load(a_bb);
        dvintx vstatus   = sign_extend(*((dvint *)(status + i * vecw)));
        dvintx vconverge = sign_extend(*((dvint *)(converge + i * vecw)));
        dvintx vx        = sign_extend(*((dvint *)(x_new + i * vecw))); // x and y in 2 super banks
        dvintx vy        = sign_extend(*((dvint *)(y_new + i * vecw)));
        dvintx vx_sat_lo = sign_extend(*((dvint *)(x_sat_lo + i * vecw)));
        dvintx vy_sat_lo = sign_extend(*((dvint *)(y_sat_lo + i * vecw)));
        dvintx vx_sat_hi = sign_extend(*((dvint *)(x_sat_hi + i * vecw)));
        dvintx vy_sat_hi = sign_extend(*((dvint *)(y_sat_hi + i * vecw)));

        dvfloatx det = dvmsubf(b, b, a * c);
        dvintx pred  = (det < THRESHOLD) & (det > -THRESHOLD);
        vstatus      = dvmux(pred, 1, vstatus); // 1 is invalid
        det          = dvmux(pred, THRESHOLD, det);

        dvfloatx scale = dvfrcp(det);

        dvfloatx tmp1 = scale * bb1;
        dvfloatx tmp2 = scale * bb2;

        // c*scale*bb1 - b*scale*bb2
        // - b*scale*bb1 + a*scale*bb2
        dvfloatx dx = dvmsubf(b, tmp2, c * tmp1);
        dvfloatx dy = dvmsubf(b, tmp1, a * tmp2);

        dvintx dx_i = dvfp_dvint_trunc(dx * (1 << QBITS));
        dvintx dy_i = dvfp_dvint_trunc(dy * (1 << QBITS));

        dvintx vx_new = vx + dx_i;
        dvintx vy_new = vy + dy_i;

        vstatus = check_image_boundary(img_w_Q, img_h_Q, vx_new, vy_new, vstatus);

        // Clamp to [center - win - tol, center + win + tol + 1]
        dvintx vtmp = check_tile_boundary(vx_new, vy_new, vx_sat_lo, vy_sat_lo, vx_sat_hi, vy_sat_hi);

        vconverge = vconverge | vstatus; // invalid belongs to converged

        vx_new = dvmux(vconverge, vx, vx_new);
        vy_new = dvmux(vconverge, vy, vy_new);

#if OPT
        vstatus |= vtmp;
#endif

        // Update convergence flag
        dvfloatx err = dvmaddf(dy, dy, dx * dx);
        vconverge    = dvmux(err <= eps_2, 1, vconverge); // 1 is converged

        *((dvint *)(status + i * vecw))   = extract(vstatus);
        *((dvint *)(converge + i * vecw)) = extract(vconverge);
        *((dvint *)(x_new + i * vecw))    = extract(vx_new);
        *((dvint *)(y_new + i * vecw))    = extract(vy_new);

        all_converged[i] = smovw(vandr(vconverge.lo & vconverge.hi));
    }
    return 0;
}