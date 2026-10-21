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

#ifndef BACKGROUND_SUBTRACTOR_TOP_H
#define BACKGROUND_SUBTRACTOR_TOP_H

#include <background_subtractor_params.h>
#include <stdint.h>

#ifdef FP16
typedef hfloat gmm_t;
typedef short gmm_int_t;
#else
typedef float gmm_t;
typedef int gmm_int_t;
#endif
typedef uint8_t pixel_t;

#endif
