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

#ifndef IMAGE_FLIP_PARAM_H
#define IMAGE_FLIP_PARAM_H
// Flip mode definitions - accessible to both C and C++ code
#define FLIP_HORIZONTAL 0
#define FLIP_VERTICAL 1
#define FLIP_COMB 2

// image width and height
#define IMW 1920
#define IMH 1080

// tile width and height for horizontal flip and combined flip
#define TW 256
#define TH 128

#endif // IMAGE_FLIP_PARAM_H
