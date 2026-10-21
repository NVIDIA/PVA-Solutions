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

#ifndef _CORNER_SUBPIX_FUNCS_H_
#define _CORNER_SUBPIX_FUNCS_H_

#include "corner_subpix_knls.h"

#include <cupva_device.h>

int32_t corner_subpix(param_corner_subpix_t *param, buf_corner_subpix_t *buf, const int32_t num_corners_input,
                      const int32_t img_w, const int32_t img_h, const int32_t max_iters, const float eps);

#endif