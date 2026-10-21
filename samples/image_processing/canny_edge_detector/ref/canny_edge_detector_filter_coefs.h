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

#ifndef CANNY_EDGE_DETECTOR_FILTER_COEFS_H
#define CANNY_EDGE_DETECTOR_FILTER_COEFS_H

// clang-format off

// Sobel 3x3
const int16_t kSobel3Coef[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

const int16_t kSobel3CoefTransp[3][3] = {
    {-1, -2, -1},
    {0, 0, 0},
    {1, 2, 1}
};

// Sobel 5x5
const int16_t kSobel5Coef[5][5] = {
    {-1, -2, 0, 2, 1},
    {-4, -8, 0, 8, 4},
    {-6, -12, 0, 12, 6},
    {-4, -8, 0, 8, 4},
    {-1, -2, 0, 2, 1}
};

const int16_t kSobel5CoefTransp[5][5] = {
    {-1, -4, -6, -4, -1},
    {-2, -8, -12, -8, -2},
    {0, 0, 0, 0, 0},
    {2, 8, 12, 8, 2},
    {1, 4, 6, 4, 1}
};

// Sobel 7x7
const int16_t kSobel7Coef[7][7] = {
    {-1, -4, -5, 0, 5, 4, 1},
    {-6, -24, -30, 0, 30, 24, 6},
    {-15, -60, -75, 0, 75, 60, 15},
    {-20, -80, -100, 0, 100, 80, 20},
    {-15, -60, -75, 0, 75, 60, 15},
    {-6, -24, -30, 0, 30, 24, 6},
    {-1, -4, -5, 0, 5, 4, 1}
};

const int16_t kSobel7CoefTransp[7][7] = {
    {-1, -6, -15, -20, -15, -6, -1},
    {-4, -24, -60, -80, -60, -24, -4},
    {-5, -30, -75, -100, -75, -30, -5},
    {0, 0, 0, 0, 0, 0, 0},
    {5, 30, 75, 100, 75, 30, 5},
    {4, 24, 60, 80, 60, 24, 4},
    {1, 6, 15, 20, 15, 6, 1}
};

// clang-format on

#endif // CANNY_EDGE_DETECTOR_FILTER_COEFS_H
