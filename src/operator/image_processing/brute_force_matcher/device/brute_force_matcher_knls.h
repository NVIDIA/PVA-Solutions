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

#include "brute_force_matcher_param.h"

#include <cupva_device.h>

void brute_force_matcher(uint16_t *source, uint16_t *reference, uint32_t *matches_source, uint32_t *matches_reference,
                         const uint32_t query_count, const uint32_t reference_count, const int matching_algo,
                         const int multiplier_1, const int multiplier_2, const uint32_t real_reference_count,
                         const uint32_t real_query_count);
