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

#include "int8_preprocess.h"

#include <algorithm>

namespace yolov5_common {

int resizeToNCHWFloat(const unsigned char *imgRGB, int imgW, int imgH, int modelW, int modelH, float *outNCHW)
{
    if (!imgRGB || !outNCHW || imgW <= 0 || imgH <= 0 || modelW <= 0 || modelH <= 0)
        return -1;

    const int channelStride = modelH * modelW;
    for (int c = 0; c < 3; c++)
    {
        for (int y = 0; y < modelH; y++)
        {
            for (int x = 0; x < modelW; x++)
            {
                const int srcX = std::min(x * imgW / modelW, imgW - 1);
                const int srcY = std::min(y * imgH / modelH, imgH - 1);
                const int src  = (srcY * imgW + srcX) * 3 + c;
                const int dst  = c * channelStride + y * modelW + x;
                outNCHW[dst]   = static_cast<float>(imgRGB[src]) / 255.0f;
            }
        }
    }
    return 0;
}

} // namespace yolov5_common
