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

#ifndef BLUR_FILTER_ROI_PARAM_H
#define BLUR_FILTER_ROI_PARAM_H

#define TILE_W (128)
#define TILE_H (128)

#define MAX_PLANE_COUNT (4)
#define MAX_BUF_SIZE (32768)

#define RECT_W (256)
#define RECT_H (256)

#define MAX_RECT_COUNT (256)

typedef struct
{
    int x;      //!< x coordinate of the top-left corner
    int y;      //!< y coordinate of the top-left corner
    int width;  //!< width of the rectangle
    int height; //!< height of the rectangle
} BlurFilterROIRect;

#endif // BLUR_FILTER_ROI_PARAM_H