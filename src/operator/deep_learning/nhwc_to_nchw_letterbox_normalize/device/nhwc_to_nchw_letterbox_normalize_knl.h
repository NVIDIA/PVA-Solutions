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

#ifndef NHWC_TO_NCHW_LETTERBOX_NORMALIZE_KNL_H
#define NHWC_TO_NCHW_LETTERBOX_NORMALIZE_KNL_H

#include "nhwc_to_nchw_letterbox_normalize_param.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Fused NHWC->NCHW transpose + normalize: uint8 HWC -> float32 CHW in one pass.
 *
 * Uses vuchar_dvintx_load_transp to read 16 uint8 values from 16 rows of the
 * TRANS_MODE_1 input, zero-extends to int32, converts to float32, multiplies
 * by scale, and writes to 3 planar output buffers via vstore_transp.
 *
 * Processes 16 rows at a time. Channel deinterleave is implicit: stepping
 * through byte offsets x*3+0, x*3+1, x*3+2 in HWC data naturally separates
 * R, G, B channels.
 *
 * @param[in]  src_nhwc      Input tile: uint8 HWC, TRANS_MODE_1 layout in VMEM
 * @param[out] dst_c0        Output tile channel 0 (R): float32, TRANS_MODE_1 in VMEM
 * @param[out] dst_c1        Output tile channel 1 (G): float32, TRANS_MODE_1 in VMEM
 * @param[out] dst_c2        Output tile channel 2 (B): float32, TRANS_MODE_1 in VMEM
 * @param[in]  tile_width    Actual tile width in pixels (<= NCHW_LN_TILE_W at right edge)
 * @param[in]  tile_height   Actual tile height in pixels (<= NCHW_LN_TILE_H at bottom edge)
 * @param[in]  src_pitch     Source line pitch in uint8 elements (T1 pitch)
 * @param[in]  dst_pitch     Destination line pitch in float elements (T1 pitch)
 * @param[in]  src_lane_ofst Source lane offset for transposed access (from cupvaTransposeLaneOfst)
 * @param[in]  dst_lane_ofst Destination lane offset for transposed access
 * @param[in]  scale         Normalization multiplier (e.g. 1.0f/255.0f)
 */
void nhwc_to_nchw_letterbox_normalize_tile_kernel(const unsigned char *src_nhwc, float *dst_c0, float *dst_c1,
                                                  float *dst_c2, int tile_width, int tile_height, int src_pitch,
                                                  int dst_pitch, int src_lane_ofst, int dst_lane_ofst, float scale);

#ifdef __cplusplus
}
#endif

#endif /* NHWC_TO_NCHW_LETTERBOX_NORMALIZE_KNL_H */
