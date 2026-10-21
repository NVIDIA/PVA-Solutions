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

#ifndef MERGE_U8_TO_RGBA8_KNL_H
#define MERGE_U8_TO_RGBA8_KNL_H

#include "../mix_channels_param.h"

#include <cupva_device.h>

#define SRC_0 (0)
#define SRC_1 (1)
#define SRC_2 (2)
#define SRC_3 (3)
#define DST (4)
#define CFG_NUM (5)

void merge_u8_to_rgba8_init(AgenCFG *cfgs, uint32_t srcTileLP, uint32_t dstTileLP);

void merge_u8_to_rgba8_exec(AgenCFG *cfgs);

#endif
