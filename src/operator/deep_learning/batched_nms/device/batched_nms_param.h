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

#ifndef BATCHED_NMS_PARAM_H
#define BATCHED_NMS_PARAM_H

#include <stdint.h>

typedef struct
{
    int max_input_boxes;
    float threshold;
    int topk;
} batched_nms_vpu_param_t;

#define MAX_NUM_CLASSES 300
#define MAX_BOX_NUM_PER_CLASS 4096 // limited by int32 sort
#define NMS_COMPACTION_PAD 64      // room for one full compact vector store at capacity
#define NMS_SORT_TAIL_PAD 32       // room for the final full-vector sort scratch clear
#define MAX_BOX_NUM_AFTER_NMS 1000
#define MAX_BOX_NUM_BEFORE_NMS_F32 8192 // limited by VMEM

#endif // BATCHED_NMS_PARAM_H
