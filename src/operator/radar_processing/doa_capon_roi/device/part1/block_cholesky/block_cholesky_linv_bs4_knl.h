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

#ifndef BLOCK_CHOLESKY_LINV_BS4_KNL_H
#define BLOCK_CHOLESKY_LINV_BS4_KNL_H

#include "capon_roi_part1_config_types.h"

void block_cholesky_opt_full_driver_bs4(Para_capon_roi_part1_config *cfg);
void linv_opt_full_driver_bs4(Para_capon_roi_part1_config *cfg);

#endif /* BLOCK_CHOLESKY_LINV_BS4_KNL_H */
