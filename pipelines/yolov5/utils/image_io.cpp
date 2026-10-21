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

#include "image_io.h"

#include <lodepng.h>

#include <cstdio>
#include <cstdlib>
#include <limits>

namespace yolov5_common {

uint8_t *loadPngRgb(const char *path, int *width, int *height)
{
    if (!path || !width || !height)
        return nullptr;

    unsigned char *pixels = nullptr;
    unsigned pngWidth = 0, pngHeight = 0;
    unsigned error = lodepng_decode24_file(&pixels, &pngWidth, &pngHeight, path);
    if (error != 0)
    {
        fprintf(stderr, "Failed to decode PNG %s: %s\n", path, lodepng_error_text(error));
        free(pixels);
        return nullptr;
    }
    if (pngWidth > static_cast<unsigned>(std::numeric_limits<int>::max()) ||
        pngHeight > static_cast<unsigned>(std::numeric_limits<int>::max()))
    {
        fprintf(stderr, "PNG dimensions are too large: %s\n", path);
        free(pixels);
        return nullptr;
    }

    *width  = static_cast<int>(pngWidth);
    *height = static_cast<int>(pngHeight);
    return pixels;
}

void freeImage(uint8_t *pixels)
{
    free(pixels);
}

bool getPngDimensions(const char *path, int *width, int *height)
{
    uint8_t *pixels = loadPngRgb(path, width, height);
    bool success    = pixels != nullptr;
    freeImage(pixels);
    return success;
}

} // namespace yolov5_common
