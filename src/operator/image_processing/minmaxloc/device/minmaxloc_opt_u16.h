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

#ifndef MINMAXLOC_OPT_U16_H
#define MINMAXLOC_OPT_U16_H

#include "minmaxloc_common.h"

// unsigned short shares location functions with short, since they are of the data size.

void find_minmax_values_ushort(MINMAX_VAL_CFG &cfg, unsigned short *p_min_val, unsigned short *p_max_val);

#endif // MINMAXLOC_OPT_U16_H