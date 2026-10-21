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

#ifndef PVA_SOLUTIONS_PRIV_CAPON_ROI_PREPROCESS_FFT_H
#define PVA_SOLUTIONS_PRIV_CAPON_ROI_PREPROCESS_FFT_H

#include "capon_roi_preprocess_knl.h"
#include "capon_roi_preprocess_params.h"

#include <fft_knl.h>
#include <stdint.h>

#define CRP_FFT_N 64
#define CRP_FFT_LANES 8                                /* chess_elementsof(vintx); == the engine's lanes-per-chunk */
#define CRP_FFT_DETS (CRP_FFT_LANES * CRP_FFT_GROUPS)  /* detections per engine run */
#define CRP_FFT_CHUNK (CRP_FFT_LANES * 2)              /* int32 per bin of one chunk: 8 lanes x (re, im) */
#define CRP_FFT_PITCH (CRP_FFT_CHUNK * CRP_FFT_GROUPS) /* one bin: the chunks side by side */
#define CRP_FFT_BUF_ELEMS (CRP_FFT_N * CRP_FFT_PITCH)  /* 2048 int32 = 8KB per working buffer at G = 2 */

#define CRP_FFT_ELEM_STRIDE (2 * CRP_FFT_N + 1) /* 129 floats */
#define CRP_FFT_ELEM_LANE_OFST 8                /* K in P = 16K+1 -> P = 129 */

#define CRP_FFT_SPEC_STRIDE (CRP_FFT_N + 1) /* 65 int32 */
#define CRP_FFT_SPEC_LANE_OFST 4            /* K in P = 16K+1 -> P = 65 */

/* Engine agen configs: N = 64 is 3 radix-4 stages -- 2 standalone (4 configs each) plus the last,
 * run by the digit reverse (2) -- so 10 are used; 16 leaves headroom. */
#define CRP_FFT_MAX_CFGS 16

void caponInitFft64Batch(const int32_t *twiddleQ30);

#endif /* PVA_SOLUTIONS_PRIV_CAPON_ROI_PREPROCESS_FFT_H */
