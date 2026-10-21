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

#ifndef GAUSSIAN_PYRAMID_KNL_H
#define GAUSSIAN_PYRAMID_KNL_H

#include <cupva_device.h>
#include <stdint.h>

void nonSep6x6FIR_downsamp2x2_weight_init(const uint16_t *filter, uint16_t *weight);

void nonSep6x6FIR_downsamp2x2_regblk4(AgenCFG *cfg_in_even, AgenCFG *cfg_in_odd, AgenCFG *cfg_out_even,
                                      AgenCFG *cfg_out_odd, const size_t rb_cnt, const uint16_t *weightPtr);

void nonSep6x6FIR_downsamp2x2_regblk2(AgenCFG *cfg_in_even, AgenCFG *cfg_in_odd, AgenCFG *cfg_out_even,
                                      AgenCFG *cfg_out_odd, const size_t rb_cnt, const uint16_t *weightPtr);

#endif // GAUSSIAN_PYRAMID_KNL_H