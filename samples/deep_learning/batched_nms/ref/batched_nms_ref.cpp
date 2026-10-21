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

#include "batched_nms_ref.h"

#include <algorithm>

namespace {
float computeIoU(const BatchedNMSRefBox &a, const BatchedNMSRefBox &b)
{
    const float interLeft   = std::max(a.left, b.left);
    const float interTop    = std::max(a.top, b.top);
    const float interRight  = std::min(a.right, b.right);
    const float interBottom = std::min(a.bottom, b.bottom);
    const float interW      = std::max(0.0f, interRight - interLeft);
    const float interH      = std::max(0.0f, interBottom - interTop);
    const float interArea   = interW * interH;
    const float areaA       = (a.right - a.left) * (a.bottom - a.top);
    const float areaB       = (b.right - b.left) * (b.bottom - b.top);
    const float unionArea   = areaA + areaB - interArea;
    return unionArea > 0.0f ? interArea / unionArea : 0.0f;
}
} // namespace

std::vector<BatchedNMSRefBox> batchedNMSRef(const float *left, const float *top, const float *right,
                                            const float *bottom, const float *scores, const int16_t *classes,
                                            int numBoxes, float iouThreshold, int maxOutput)
{
    std::vector<int> uniqueClasses;
    std::vector<bool> seen(300, false);
    for (int i = 0; i < numBoxes; ++i)
    {
        const int cls = classes[i];
        if (cls >= 0 && cls < 300 && !seen[cls])
        {
            seen[cls] = true;
            uniqueClasses.push_back(cls);
        }
    }

    std::vector<BatchedNMSRefBox> result;
    for (int cls : uniqueClasses)
    {
        std::vector<int> indices;
        for (int i = 0; i < numBoxes; ++i)
            if (classes[i] == cls)
                indices.push_back(i);
        std::stable_sort(indices.begin(), indices.end(), [&](int a, int b) { return scores[a] > scores[b]; });

        std::vector<bool> suppressed(indices.size(), false);
        for (size_t ii = 0; ii < indices.size() && static_cast<int>(result.size()) < maxOutput; ++ii)
        {
            if (suppressed[ii])
                continue;
            const int i = indices[ii];
            BatchedNMSRefBox kept{left[i], top[i], right[i], bottom[i], scores[i], cls};
            result.push_back(kept);
            for (size_t jj = ii + 1; jj < indices.size(); ++jj)
            {
                const int j = indices[jj];
                BatchedNMSRefBox candidate{left[j], top[j], right[j], bottom[j], scores[j], cls};
                if (!suppressed[jj] && computeIoU(kept, candidate) > iouThreshold)
                    suppressed[jj] = true;
            }
        }
    }
    return result;
}
