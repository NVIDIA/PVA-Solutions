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

#ifndef DATA_TYPE_DEF_H
#define DATA_TYPE_DEF_H

#include <cupva_device.h>

#define DATA_TYPE uint8_t

#if defined(U8_IO)
#    include "image_stats_opt_u8.h"
#    define INIT_IMAGE_STATS_CFG init_image_stats_u8_cfg
#    if defined(MASK_IO)
#        define IMAGE_STATS_EXEC image_stats_opt_mask_u8_exec
#    else
#        define IMAGE_STATS_EXEC image_stats_opt_u8_exec
#    endif
#elif defined(RGB_IO)
#    include "image_stats_opt_rgb.h"
#    define INIT_IMAGE_STATS_CFG init_image_stats_rgb_cfg
#    if defined(MASK_IO)
#        define IMAGE_STATS_EXEC image_stats_opt_mask_rgb_exec
#    else
#        define IMAGE_STATS_EXEC image_stats_opt_rgb_exec
#    endif
#elif defined(NV12_IO)
#    include "image_stats_opt_nv12.h"
#    define INIT_IMAGE_STATS_CFG init_image_stats_nv12_cfg
#    if defined(MASK_IO)
#        define IMAGE_STATS_EXEC image_stats_opt_mask_nv12_exec
#    else
#        define IMAGE_STATS_EXEC image_stats_opt_nv12_exec
#    endif
#endif

#endif /* DATA_TYPE_DEF_H */
