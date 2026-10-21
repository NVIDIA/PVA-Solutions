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
#ifndef IMAGE_FLIP_VERTICAL_KNL_H
#define IMAGE_FLIP_VERTICAL_KNL_H

#include <cupva_device.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PARAM_ID_FLIP_BOTH 1002

// Vertical flip parameter structure
typedef struct
{
    AgenCFG cfgs[2]; // More configs needed for intermediate buffer handling
    int32_t width, height;
    int32_t linepitch;
    int32_t niter1;
    int32_t niter2;
} param_vertical_flip_t;

// Function declarations for vertical flip VPU implementation
int vertical_flip_init(uint8_t *src, uint8_t *dst, int32_t width, int32_t height, int32_t linepitch,
                       param_vertical_flip_t *param);

int vertical_flip_exec(void *args);

#ifdef __cplusplus
}
#endif

#endif // IMAGE_FLIP_VERTICAL_KNL_H