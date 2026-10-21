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

#include "yolov5_decode_ref.h"

#include <algorithm>
#include <cmath>

namespace {
float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}
} // namespace

std::vector<YoloV5RefDetection> yolov5DecodeRef(const float *predictions, const float *anchors, int numAnchors,
                                                int numClasses, float confThreshold, float imageHeight,
                                                float imageWidth, int maxObjects)
{
    std::vector<YoloV5RefDetection> result;
    const float deconf = -std::log(1.0f / confThreshold - 1.0f);
    for (int i = 0; i < numAnchors && static_cast<int>(result.size()) < maxObjects; ++i)
    {
        const float *prediction = predictions + i * (numClasses + 5);
        if (prediction[4] <= deconf)
            continue;
        const float objectness = sigmoid(prediction[4]);
        for (int cls = 0; cls < numClasses && static_cast<int>(result.size()) < maxObjects; ++cls)
        {
            const float confidence = sigmoid(prediction[5 + cls]) * objectness;
            if (confidence <= confThreshold)
                continue;
            const float *anchor = anchors + i * 5;
            const float stride  = anchor[4];
            const float cx      = (sigmoid(prediction[0]) * 2.0f - 0.5f + anchor[0]) * stride;
            const float cy      = (sigmoid(prediction[1]) * 2.0f - 0.5f + anchor[1]) * stride;
            const float dw      = sigmoid(prediction[2]) * 2.0f;
            const float dh      = sigmoid(prediction[3]) * 2.0f;
            const float width   = dw * dw * anchor[2];
            const float height  = dh * dh * anchor[3];
            result.push_back({std::max(0.0f, cx - width * 0.5f), std::max(0.0f, cy - height * 0.5f),
                              std::min(imageWidth, cx + width * 0.5f), std::min(imageHeight, cy + height * 0.5f),
                              confidence, cls});
        }
    }
    return result;
}

void yolov5ChwToFlatRef(const float *chw, float *flat, int spatialSize, int numClasses)
{
    const int attributes = numClasses + 5;
    for (int spatial = 0; spatial < spatialSize; ++spatial)
        for (int anchor = 0; anchor < 3; ++anchor)
            for (int attr = 0; attr < attributes; ++attr)
                flat[(spatial * 3 + anchor) * attributes + attr] =
                    chw[(anchor * attributes + attr) * spatialSize + spatial];
}
