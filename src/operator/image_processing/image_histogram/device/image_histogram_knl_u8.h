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

#ifndef IMAGE_HISTOGRAM_KNL_U8_H
#define IMAGE_HISTOGRAM_KNL_U8_H

#include "image_histogram_knl.h"

#define QBITS (8)

void generate_mask_u8(uint8_t *mask, int tw, int mw);

void image_histogram_agen_init_u8(uint8_t *src, int tw, int th, int th_last, int lp, int start, int end, int nbins,
                                  int16_t *hist, uint32_t *dst, int32_t *bucketAcc, uint8_t *mask,
                                  ImageHistogramConfig *cfg);

void image_histogram_knl_exec_u8(ImageHistogramConfig *cfg);

void image_histogram_knl_exec_masked_u8(ImageHistogramConfig *cfg);

#endif // IMAGE_HISTOGRAM_KNL_U8_H
