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

#ifndef IMAGE_BLEND_KNL_U8_H
#define IMAGE_BLEND_KNL_U8_H

#include "image_blend_knl.h"

void image_blend_agen_init(int32_t src0LinePitch, int32_t src1LinePitch, int32_t dstLinePitch, ImageBlendConfig *cfg);
void image_blend_agen_update(uint8_t *src0Tile, uint8_t *src1Tile, uint8_t *dstTile, ImageBlendConfig *cfg);
void image_blend_exec(uint32_t alpha, int32_t niter, ImageBlendConfig *cfg);

#endif // IMAGE_BLEND_KNL_U8_H