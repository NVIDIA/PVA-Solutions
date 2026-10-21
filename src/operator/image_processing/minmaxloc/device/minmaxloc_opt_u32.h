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

#ifndef MINMAXLOC_OPT_U32_H
#define MINMAXLOC_OPT_U32_H

#include "minmaxloc_common.h"

// unsigned int shares location functions with int, since they are of the data size.

void find_minmax_values_uint(MINMAX_VAL_CFG &cfg, unsigned int *p_min_val, unsigned int *p_max_val);

#endif // MINMAXLOC_OPT_U32_H