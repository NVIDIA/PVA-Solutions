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

#ifndef MINMAXLOC_OPT_F32_H
#define MINMAXLOC_OPT_F32_H

#include "minmaxloc_common.h"

// F32 shares location functions with S32, since they are of the data size

void find_minmax_values_float(MINMAX_VAL_CFG &cfg, float *p_min_val, float *p_max_val, float *p_tmp_dvmin,
                              float *p_tmp_dvmax);

#endif // MINMAXLOC_OPT_F32_H