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

/*
 * Shared parameters for YOLOv5 transpose-decode kernel.
 * Included by both VPU device code and host code.
 */

#ifndef YOLOV5_DECODE_PARAM_H
#define YOLOV5_DECODE_PARAM_H

#include <stdint.h>

typedef struct
{
    int tile_size;
    int num_tiles;
    int anchor_item_size;
    int num_classes;
    float conf_threshold;
    float deconf_threshold;
    int image_height;
    int image_width;
    int max_objects;
    int spatial_tile_size;
    int last_spatial_tile;
    int hwc_input;
} yolov5_decode_host2dev_param_t;

#define VPU_MAX_OBJECTS 5120 // 30*1024/6 = 5120

#endif // YOLOV5_DECODE_PARAM_H
