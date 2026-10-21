/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NHWC_TO_NCHW_LETTERBOX_NORMALIZE_PARAM_H
#define NHWC_TO_NCHW_LETTERBOX_NORMALIZE_PARAM_H

#ifdef __cplusplus
extern "C" {
#endif

/** Output tile dimensions (pixels). Must be multiples of 16 for 16-row blocking. */
#define NCHW_LN_TILE_W 64
#define NCHW_LN_TILE_H 64

/**
 * Host-to-device parameter block for the fused NHWC->NCHW + letterbox + normalize kernel.
 *
 * This kernel takes uint8 HWC input and produces float32 CHW output in one pass.
 * Input is the aspect-ratio-resized image (imgW x imgH). Output is the full
 * model input size (outW x outH); the kernel writes only the image region.
 *
 * Memory layout:
 *   Input  (src): uint8,   HWC, TRANS_MODE_1 in VMEM
 *   Output (dst): float32, CHW (3 planar), TRANS_MODE_1 in VMEM
 */
typedef struct
{
    /* Input (letterboxed-resized) image dimensions */
    int img_width;  ///< Resized image width
    int img_height; ///< Resized image height

    /* Normalization */
    float scale; ///< Multiplier for image pixels (e.g. 1.0/255.0)

    /* Tiling parameters (over the image region, NOT the full output) */
    int tile_width;  ///< Tile width  in pixels (default NCHW_LN_TILE_W)
    int tile_height; ///< Tile height in pixels (default NCHW_LN_TILE_H)
    int niter_x;     ///< Number of tile columns: ceil(img_width  / tile_width)
    int niter_y;     ///< Number of tile rows:    ceil(img_height / tile_height)

    /* Reserved for ABI stability */
    int reserved[57];
} nhwc_to_nchw_letterbox_normalize_host2dev_param_t;

#ifdef __cplusplus
}
#endif

#endif /* NHWC_TO_NCHW_LETTERBOX_NORMALIZE_PARAM_H */
