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

#ifndef CONTOUR_TRACING_PARAM_H
#define CONTOUR_TRACING_PARAM_H

#define NUM_NEIGHBOR 8U
#define NUM_NEIGHBOR_PERM (1U << NUM_NEIGHBOR)
#define DIR_NONE 8U // Sentinel: no valid direction

// Maximum number of bounding rectangles per input image
#define MAX_RECT_NUM 32
// Maximum width and height of a single bounding rectangle
#define MAX_RECT_WIDTH 128
#define MAX_RECT_HEIGHT 128
#define MAX_RECT_AREA (MAX_RECT_WIDTH * MAX_RECT_HEIGHT)

#endif // CONTOUR_TRACING_PARAM_H
