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

#ifndef BLUR_FILTER_ROI_KNL_U8_H
#define BLUR_FILTER_ROI_KNL_U8_H

#include "blur_filter_roi_knl.h"

void blur_filter_roi_knl_exec(int32_t w, int32_t h, int32_t lp, uint8_t data, uint8_t *dst);

#endif // BLUR_FILTER_ROI_KNL_U8_H