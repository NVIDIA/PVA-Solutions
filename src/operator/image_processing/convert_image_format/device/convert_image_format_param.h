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

#ifndef CONVERT_IMAGE_FORMAT_PARAM_H
#define CONVERT_IMAGE_FORMAT_PARAM_H

#define TILE_W (64)
#define TILE_H (128)

#define MAX_CHANNEL_COUNT (4)

typedef enum
{
    /// convert YUYV to NV12
    CONVERT_IMAGE_FORMAT_YUYV2NV12,
    /// convert UYVY to NV12
    CONVERT_IMAGE_FORMAT_UYVY2NV12,
    /// convert VYUY to NV12
    CONVERT_IMAGE_FORMAT_VYUY2NV12,
    /// convert YUV8p to NV12
    CONVERT_IMAGE_FORMAT_YUV8p2NV12,
    /// convert YUV8p to YUYV
    CONVERT_IMAGE_FORMAT_YUV8p2YUYV,
    /// convert YUV8p to UYVY
    CONVERT_IMAGE_FORMAT_YUV8p2UYVY,
    /// convert YUV8p to VYUY
    CONVERT_IMAGE_FORMAT_YUV8p2VYUY,
    /// convert NV12 to BGR8
    CONVERT_IMAGE_FORMAT_NV122BGR8,
    /// convert NV12 to RGB8
    CONVERT_IMAGE_FORMAT_NV122RGB8,
    /// convert NV12 to BGR8p
    CONVERT_IMAGE_FORMAT_NV122BGR8p,
    /// convert NV12 to RGB8p
    CONVERT_IMAGE_FORMAT_NV122RGB8p,
    /// convert YUYV to BGR8
    CONVERT_IMAGE_FORMAT_YUYV2BGR8,
    /// convert YUYV to RGB8
    CONVERT_IMAGE_FORMAT_YUYV2RGB8,
    /// convert UYVY to BGR8
    CONVERT_IMAGE_FORMAT_UYVY2BGR8,
    /// convert UYVY to RGB8
    CONVERT_IMAGE_FORMAT_UYVY2RGB8,
    /// convert VYUY to BGR8
    CONVERT_IMAGE_FORMAT_VYUY2BGR8,
    /// convert VYUY to RGB8
    CONVERT_IMAGE_FORMAT_VYUY2RGB8,
    /// convert YUYV to BGR8p
    CONVERT_IMAGE_FORMAT_YUYV2BGR8p,
    /// convert YUYV to RGB8p
    CONVERT_IMAGE_FORMAT_YUYV2RGB8p,
    /// convert UYVY to BGR8p
    CONVERT_IMAGE_FORMAT_UYVY2BGR8p,
    /// convert UYVY to RGB8p
    CONVERT_IMAGE_FORMAT_UYVY2RGB8p,
    /// convert VYUY to BGR8p
    CONVERT_IMAGE_FORMAT_VYUY2BGR8p,
    /// convert VYUY to RGB8p
    CONVERT_IMAGE_FORMAT_VYUY2RGB8p,
    /// convert RGB8 to BGR8 and add aphla channel
    CONVERT_IMAGE_FORMAT_RGB82BGRA8,
    /// convert BGR8 to NV12
    CONVERT_IMAGE_FORMAT_BGR82NV12,
    /// convert RGB8 to NV12
    CONVERT_IMAGE_FORMAT_RGB82NV12,
    /// convert BGR8p to NV12
    CONVERT_IMAGE_FORMAT_BGR8p2NV12,
    /// convert RGB8p to NV12
    CONVERT_IMAGE_FORMAT_RGB8p2NV12,
    /// convert BGR8 to YUYV
    CONVERT_IMAGE_FORMAT_BGR82YUYV,
    /// convert RGB8 to YUYV
    CONVERT_IMAGE_FORMAT_RGB82YUYV,
    /// convert BGR8 to UYVY
    CONVERT_IMAGE_FORMAT_BGR82UYVY,
    /// convert RGB8 to UYVY
    CONVERT_IMAGE_FORMAT_RGB82UYVY,
    /// convert BGR8 to VYUY
    CONVERT_IMAGE_FORMAT_BGR82VYUY,
    /// convert RGB8 to VYUY
    CONVERT_IMAGE_FORMAT_RGB82VYUY,
    /// convert BGR8p to YUYV
    CONVERT_IMAGE_FORMAT_BGR8p2YUYV,
    /// convert RGB8p to YUYV
    CONVERT_IMAGE_FORMAT_RGB8p2YUYV,
    /// convert BGR8p to UYVY
    CONVERT_IMAGE_FORMAT_BGR8p2UYVY,
    /// convert RGB8p to UYVY
    CONVERT_IMAGE_FORMAT_RGB8p2UYVY,
    /// convert BGR8p to VYUY
    CONVERT_IMAGE_FORMAT_BGR8p2VYUY,
    /// convert RGB8p to VYUY
    CONVERT_IMAGE_FORMAT_RGB8p2VYUY,
    /* ITU-R BT.601 EXTENDED RANGE (NV12_ER) variants. The kernel reuses the
     * existing yuv2rgb / rgb2yuv paths; only the color-matrix coefficients
     * differ (no Y-16 offset, no 1/(235-16) luma scale). The host loads ER
     * coefficient tables (see COEFF_*_ER_FIXED macros below) when one of
     * these codes is set. */
    /// convert NV12_ER to BGR8 (full-range YUV → RGB)
    CONVERT_IMAGE_FORMAT_NV12_ER2BGR8,
    /// convert NV12_ER to RGB8
    CONVERT_IMAGE_FORMAT_NV12_ER2RGB8,
    /// convert NV12_ER to BGR8p
    CONVERT_IMAGE_FORMAT_NV12_ER2BGR8p,
    /// convert NV12_ER to RGB8p
    CONVERT_IMAGE_FORMAT_NV12_ER2RGB8p,
    /// convert BGR8 to NV12_ER
    CONVERT_IMAGE_FORMAT_BGR82NV12_ER,
    /// convert RGB8 to NV12_ER
    CONVERT_IMAGE_FORMAT_RGB82NV12_ER,
    /// convert BGR8p to NV12_ER
    CONVERT_IMAGE_FORMAT_BGR8p2NV12_ER,
    /// convert RGB8p to NV12_ER
    CONVERT_IMAGE_FORMAT_RGB8p2NV12_ER,
    /// convert YUYV to NV12_ER (chroma resampling only; luma copied as-is)
    CONVERT_IMAGE_FORMAT_YUYV2NV12_ER,
    /// convert UYVY to NV12_ER
    CONVERT_IMAGE_FORMAT_UYVY2NV12_ER,
    /// convert VYUY to NV12_ER
    CONVERT_IMAGE_FORMAT_VYUY2NV12_ER,
    /// convert YUV8p to NV12_ER
    CONVERT_IMAGE_FORMAT_YUV8p2NV12_ER,
    /// convert RGB8 to RGBA8 (3→4 channel, no R/B swap; appends alpha = 255).
    /// Reuses the same kernel as RGB8→BGRA8 with R/B in-order output pointers.
    CONVERT_IMAGE_FORMAT_RGB82RGBA8,
    /// convert BGR8 to BGRA8 (3→4 channel, no swap; appends alpha = 255).
    CONVERT_IMAGE_FORMAT_BGR82BGRA8,
    /// convert NV12_ER (full range) to packed RGBA8 (YUV→RGB with no Y-16
    /// offset; alpha lane filled with 255 by alpha_fill_knl_exec).
    CONVERT_IMAGE_FORMAT_NV12_ER2RGBA8,
    /// convert RGBA8 to packed RGB8 (4→3 channel; alpha lane dropped).
    CONVERT_IMAGE_FORMAT_RGBA82RGB8,
    /// convert BGRA8 to packed BGR8 (4→3 channel; alpha lane dropped).
    CONVERT_IMAGE_FORMAT_BGRA82BGR8,
} ImageFormatConversionCode;

/** color spec according to ITU-R BT.601 */

#define COEFF_YR_FLOAT (0.299f)
#define COEFF_YG_FLOAT (0.587f)
#define COEFF_YB_FLOAT (0.114f)
#define COEFF_CB_FLOAT (1.772f)
#define COEFF_CR_FLOAT (1.402f)

#define COLOR_BITS (8)
#define COLOR_RANGE ((1 << COLOR_BITS) - 1)
#define ALPHA_VALUE (COLOR_RANGE)

#define COLOR_RANGE_L (235 - 16)
#define COLOR_RANGE_C (240 - 16)

#define COLOR_OFFSET_L (16)
#define COLOR_OFFSET_C (128)

// coefficients for converting from 8-bit YUV with studio range to 8-bit RGB

#define COEFF_RY_FLOAT ((float)COLOR_RANGE / COLOR_RANGE_L)           // 1.164f
#define COEFF_RU_FLOAT (0.000f)                                       // 0.000f
#define COEFF_RV_FLOAT (COEFF_CR_FLOAT * COLOR_RANGE / COLOR_RANGE_C) // 1.596f

#define COEFF_BY_FLOAT ((float)COLOR_RANGE / COLOR_RANGE_L)           // 1.164f
#define COEFF_BU_FLOAT (COEFF_CB_FLOAT * COLOR_RANGE / COLOR_RANGE_C) // 2.017f
#define COEFF_BV_FLOAT (0.000f)                                       // 0.000f

#define COEFF_GY_FLOAT ((float)COLOR_RANGE / COLOR_RANGE_L)                // 1.164f
#define COEFF_GU_FLOAT (-COEFF_BU_FLOAT * COEFF_YB_FLOAT / COEFF_YG_FLOAT) // -0.392f
#define COEFF_GV_FLOAT (-COEFF_RV_FLOAT * COEFF_YR_FLOAT / COEFF_YG_FLOAT) // -0.813f

// coefficients for converting from 8-bit RGB to 8-bit YUV with studio range

#define COEFF_YR_STUDIO_FLOAT (COEFF_YR_FLOAT * COLOR_RANGE_L / COLOR_RANGE) // 0.257f
#define COEFF_YG_STUDIO_FLOAT (COEFF_YG_FLOAT * COLOR_RANGE_L / COLOR_RANGE) // 0.504f
#define COEFF_YB_STUDIO_FLOAT (COEFF_YB_FLOAT * COLOR_RANGE_L / COLOR_RANGE) // 0.098f

#define COEFF_UR_STUDIO_FLOAT (-COEFF_YR_FLOAT / COEFF_CB_FLOAT * COLOR_RANGE_C / COLOR_RANGE)         // -0.148f
#define COEFF_UG_STUDIO_FLOAT (-COEFF_YG_FLOAT / COEFF_CB_FLOAT * COLOR_RANGE_C / COLOR_RANGE)         // -0.291f
#define COEFF_UB_STUDIO_FLOAT ((1.0f - COEFF_YB_FLOAT) / COEFF_CB_FLOAT * COLOR_RANGE_C / COLOR_RANGE) // 0.439f

#define COEFF_VR_STUDIO_FLOAT ((1.0f - COEFF_YR_FLOAT) / COEFF_CR_FLOAT * COLOR_RANGE_C / COLOR_RANGE) // 0.439f
#define COEFF_VG_STUDIO_FLOAT (-COEFF_YG_FLOAT / COEFF_CR_FLOAT * COLOR_RANGE_C / COLOR_RANGE)         // -0.368f
#define COEFF_VB_STUDIO_FLOAT (-COEFF_YB_FLOAT / COEFF_CR_FLOAT * COLOR_RANGE_C / COLOR_RANGE)         // -0.071f

#define QBITS (6)
#define QBITS_SCALE (1 << QBITS)
#define QBITS_SCALE_HALF (QBITS_SCALE >> 1)

/* Round a float coefficient to the nearest integer before casting.
 * Plain C casts truncate toward zero; these macros give round-to-nearest,
 * which is more accurate and matches the comment values in the _FIXED macros. */
#define COEFF_ROUND_U8(x) ((uint8_t)((x) + 0.5f))
#define COEFF_ROUND_S8(x) ((int8_t)((x) < 0.0f ? (x)-0.5f : (x) + 0.5f))

#define COEFF_RY_FIXED COEFF_ROUND_U8(COEFF_RY_FLOAT *QBITS_SCALE) // 74
#define COEFF_RU_FIXED COEFF_ROUND_U8(COEFF_RU_FLOAT *QBITS_SCALE) // 0
#define COEFF_RV_FIXED COEFF_ROUND_U8(COEFF_RV_FLOAT *QBITS_SCALE) // 102
#define COEFF_GY_FIXED COEFF_ROUND_S8(COEFF_GY_FLOAT *QBITS_SCALE) // 74
#define COEFF_GU_FIXED COEFF_ROUND_S8(COEFF_GU_FLOAT *QBITS_SCALE) // -25
#define COEFF_GV_FIXED COEFF_ROUND_S8(COEFF_GV_FLOAT *QBITS_SCALE) // -52
#define COEFF_BY_FIXED COEFF_ROUND_U8(COEFF_BY_FLOAT *QBITS_SCALE) // 74
#define COEFF_BU_FIXED COEFF_ROUND_U8(COEFF_BU_FLOAT *QBITS_SCALE) // 129
#define COEFF_BV_FIXED COEFF_ROUND_U8(COEFF_BV_FLOAT *QBITS_SCALE) // 0

#define COEFF_YR_FIXED COEFF_ROUND_U8(COEFF_YR_STUDIO_FLOAT *QBITS_SCALE) // 16
#define COEFF_YG_FIXED COEFF_ROUND_U8(COEFF_YG_STUDIO_FLOAT *QBITS_SCALE) // 32
#define COEFF_YB_FIXED COEFF_ROUND_U8(COEFF_YB_STUDIO_FLOAT *QBITS_SCALE) // 6
#define COEFF_UR_FIXED COEFF_ROUND_S8(COEFF_UR_STUDIO_FLOAT *QBITS_SCALE) // -9
#define COEFF_UG_FIXED COEFF_ROUND_S8(COEFF_UG_STUDIO_FLOAT *QBITS_SCALE) // -18
#define COEFF_UB_FIXED COEFF_ROUND_S8(COEFF_UB_STUDIO_FLOAT *QBITS_SCALE) // 28
#define COEFF_VR_FIXED COEFF_ROUND_S8(COEFF_VR_STUDIO_FLOAT *QBITS_SCALE) // 28
#define COEFF_VG_FIXED COEFF_ROUND_S8(COEFF_VG_STUDIO_FLOAT *QBITS_SCALE) // -23
#define COEFF_VB_FIXED COEFF_ROUND_S8(COEFF_VB_STUDIO_FLOAT *QBITS_SCALE) // -4

#define COEFF_R_FIXED COEFF_RY_FIXED, COEFF_RU_FIXED, COEFF_RV_FIXED, QBITS_SCALE_HALF
#define COEFF_G_FIXED COEFF_GY_FIXED, COEFF_GU_FIXED, COEFF_GV_FIXED, QBITS_SCALE_HALF
#define COEFF_B_FIXED COEFF_BY_FIXED, COEFF_BU_FIXED, COEFF_BV_FIXED, QBITS_SCALE_HALF

#define COEFF_Y_FIXED COEFF_YR_FIXED, COEFF_YG_FIXED, COEFF_YB_FIXED, QBITS_SCALE_HALF
#define COEFF_U_FIXED COEFF_UR_FIXED, COEFF_UG_FIXED, COEFF_UB_FIXED, QBITS_SCALE_HALF
#define COEFF_V_FIXED COEFF_VR_FIXED, COEFF_VG_FIXED, COEFF_VB_FIXED, QBITS_SCALE_HALF

/* ============================================================================
 * ITU-R BT.601 EXTENDED RANGE (full 0–255 luma, 0–255 chroma) coefficients.
 *
 * Used for NV12_ER. Differences vs studio range:
 *   - YUV→RGB: no `(Y - 16)` offset, no `1/(235-16)` luma scale, so the
 *     diagonal is 1.0 instead of 1.164. Chroma scaling stays at 1.402/1.772.
 *   - RGB→YUV: no `+16` luma offset; coefficients are the raw BT.601 weights
 *     (0.299/0.587/0.114) with U,V centered at 128.
 *
 * All matrices keep the same fixed-point Q6 (QBITS = 6) format as the
 * studio-range macros, so the kernel multiplier paths are unchanged.
 * ============================================================================ */

/* YUV_ER → RGB: full-range path */
#define COEFF_RY_ER_FLOAT (1.000f)
#define COEFF_RU_ER_FLOAT (0.000f)
#define COEFF_RV_ER_FLOAT (COEFF_CR_FLOAT) /* 1.402 */

#define COEFF_GY_ER_FLOAT (1.000f)
#define COEFF_GU_ER_FLOAT (-COEFF_CB_FLOAT * COEFF_YB_FLOAT / COEFF_YG_FLOAT) /* -0.344 */
#define COEFF_GV_ER_FLOAT (-COEFF_CR_FLOAT * COEFF_YR_FLOAT / COEFF_YG_FLOAT) /* -0.714 */

#define COEFF_BY_ER_FLOAT (1.000f)
#define COEFF_BU_ER_FLOAT (COEFF_CB_FLOAT) /* 1.772 */
#define COEFF_BV_ER_FLOAT (0.000f)

/* RGB → YUV_ER: full-range path */
#define COEFF_YR_ER_FLOAT (COEFF_YR_FLOAT) /* 0.299 */
#define COEFF_YG_ER_FLOAT (COEFF_YG_FLOAT) /* 0.587 */
#define COEFF_YB_ER_FLOAT (COEFF_YB_FLOAT) /* 0.114 */

#define COEFF_UR_ER_FLOAT (-COEFF_YR_FLOAT / COEFF_CB_FLOAT)         /* -0.169 */
#define COEFF_UG_ER_FLOAT (-COEFF_YG_FLOAT / COEFF_CB_FLOAT)         /* -0.331 */
#define COEFF_UB_ER_FLOAT ((1.0f - COEFF_YB_FLOAT) / COEFF_CB_FLOAT) /*  0.500 */

#define COEFF_VR_ER_FLOAT ((1.0f - COEFF_YR_FLOAT) / COEFF_CR_FLOAT) /*  0.500 */
#define COEFF_VG_ER_FLOAT (-COEFF_YG_FLOAT / COEFF_CR_FLOAT)         /* -0.419 */
#define COEFF_VB_ER_FLOAT (-COEFF_YB_FLOAT / COEFF_CR_FLOAT)         /* -0.081 */

/* Fixed-point Q6 (QBITS_SCALE = 64) — same scale as studio macros above. */
#define COEFF_RY_ER_FIXED COEFF_ROUND_U8(COEFF_RY_ER_FLOAT *QBITS_SCALE) /* 64 */
#define COEFF_RU_ER_FIXED COEFF_ROUND_U8(COEFF_RU_ER_FLOAT *QBITS_SCALE) /* 0  */
#define COEFF_RV_ER_FIXED COEFF_ROUND_U8(COEFF_RV_ER_FLOAT *QBITS_SCALE) /* 90 */
#define COEFF_GY_ER_FIXED COEFF_ROUND_S8(COEFF_GY_ER_FLOAT *QBITS_SCALE) /* 64 */
#define COEFF_GU_ER_FIXED COEFF_ROUND_S8(COEFF_GU_ER_FLOAT *QBITS_SCALE) /* -22 */
#define COEFF_GV_ER_FIXED COEFF_ROUND_S8(COEFF_GV_ER_FLOAT *QBITS_SCALE) /* -46 */
#define COEFF_BY_ER_FIXED COEFF_ROUND_U8(COEFF_BY_ER_FLOAT *QBITS_SCALE) /* 64 */
#define COEFF_BU_ER_FIXED COEFF_ROUND_U8(COEFF_BU_ER_FLOAT *QBITS_SCALE) /* 113 */
#define COEFF_BV_ER_FIXED COEFF_ROUND_U8(COEFF_BV_ER_FLOAT *QBITS_SCALE) /* 0  */

#define COEFF_YR_ER_FIXED COEFF_ROUND_U8(COEFF_YR_ER_FLOAT *QBITS_SCALE) /* 19 */
#define COEFF_YG_ER_FIXED COEFF_ROUND_U8(COEFF_YG_ER_FLOAT *QBITS_SCALE) /* 38 */
#define COEFF_YB_ER_FIXED COEFF_ROUND_U8(COEFF_YB_ER_FLOAT *QBITS_SCALE) /* 7  */
#define COEFF_UR_ER_FIXED COEFF_ROUND_S8(COEFF_UR_ER_FLOAT *QBITS_SCALE) /* -11 */
#define COEFF_UG_ER_FIXED COEFF_ROUND_S8(COEFF_UG_ER_FLOAT *QBITS_SCALE) /* -21 */
#define COEFF_UB_ER_FIXED COEFF_ROUND_S8(COEFF_UB_ER_FLOAT *QBITS_SCALE) /*  32 */
#define COEFF_VR_ER_FIXED COEFF_ROUND_S8(COEFF_VR_ER_FLOAT *QBITS_SCALE) /*  32 */
#define COEFF_VG_ER_FIXED COEFF_ROUND_S8(COEFF_VG_ER_FLOAT *QBITS_SCALE) /* -27 */
#define COEFF_VB_ER_FIXED COEFF_ROUND_S8(COEFF_VB_ER_FLOAT *QBITS_SCALE) /* -5  */

#define COEFF_R_ER_FIXED COEFF_RY_ER_FIXED, COEFF_RU_ER_FIXED, COEFF_RV_ER_FIXED, QBITS_SCALE_HALF
#define COEFF_G_ER_FIXED COEFF_GY_ER_FIXED, COEFF_GU_ER_FIXED, COEFF_GV_ER_FIXED, QBITS_SCALE_HALF
#define COEFF_B_ER_FIXED COEFF_BY_ER_FIXED, COEFF_BU_ER_FIXED, COEFF_BV_ER_FIXED, QBITS_SCALE_HALF

#define COEFF_Y_ER_FIXED COEFF_YR_ER_FIXED, COEFF_YG_ER_FIXED, COEFF_YB_ER_FIXED, QBITS_SCALE_HALF
#define COEFF_U_ER_FIXED COEFF_UR_ER_FIXED, COEFF_UG_ER_FIXED, COEFF_UB_ER_FIXED, QBITS_SCALE_HALF
#define COEFF_V_ER_FIXED COEFF_VR_ER_FIXED, COEFF_VG_ER_FIXED, COEFF_VB_ER_FIXED, QBITS_SCALE_HALF

#endif // CONVERT_IMAGE_FORMAT_PARAM_H