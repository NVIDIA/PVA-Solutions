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

#ifndef MEDIAN_FILTER_KNL_SEPA_2X2_7X7_H
#define MEDIAN_FILTER_KNL_SEPA_2X2_7X7_H

#include "median_filter_common_sepa_2x2.h"
#include "median_filter_param.h"

#include <cupva_device.h>

void serial_run_sepa_2x2_7x7_exec(const MedianFilterSepa2x2Config &cfg);

#endif /* MEDIAN_FILTER_KNL_SEPA_2X2_7X7_H */
