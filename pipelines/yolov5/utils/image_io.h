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

#ifndef YOLOV5_COMMON_IMAGE_IO_H
#define YOLOV5_COMMON_IMAGE_IO_H

#include <cstdint>

namespace yolov5_common {

/** Decode a PNG file to a tightly packed, three-channel RGB buffer. */
uint8_t *loadPngRgb(const char *path, int *width, int *height);

/** Release a buffer returned by loadPngRgb. */
void freeImage(uint8_t *pixels);

/** Read PNG dimensions. */
bool getPngDimensions(const char *path, int *width, int *height);

} // namespace yolov5_common

#endif // YOLOV5_COMMON_IMAGE_IO_H
