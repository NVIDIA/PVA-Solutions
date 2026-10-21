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
#ifndef CANNY_EDGE_DETECTOR_FILTERS_H
#define CANNY_EDGE_DETECTOR_FILTERS_H

#include <cupva_device.h>
#include <stdint.h>

const int8_t chess_storage(DMb % 64 : chess_segment(A)) coef_Sobel3_h[3] = {-1, 0, 1};
const int8_t chess_storage(DMb % 64 : chess_segment(A)) coef_Sobel3_v[3] = {1, 2, 1};
const int8_t chess_storage(DMb % 64 : chess_segment(A)) coef_Sobel5_h[5] = {-1, -2, 0, 2, 1};
const int8_t chess_storage(DMb % 64 : chess_segment(A)) coef_Sobel5_v[5] = {1, 4, 6, 4, 1};
const int8_t chess_storage(DMb % 64 : chess_segment(A)) coef_Sobel7_h[7] = {-1, -4, -5, 0, 5, 4, 1};
const int8_t chess_storage(DMb % 64 : chess_segment(A)) coef_Sobel7_v[7] = {1, 6, 15, 20, 15, 6, 1};

#endif