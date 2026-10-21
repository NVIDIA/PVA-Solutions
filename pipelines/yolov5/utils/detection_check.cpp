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

#include "detection_check.h"

#include <algorithm>
#include <cstdio>
#include <vector>

namespace yolov5_common {

const RefBox TRAFFIC_REFERENCE[] = {
    {474.5F, 10.1F, 1280.0F, 706.1F, 5, "bus"},    {0.1F, 266.4F, 137.4F, 681.0F, 0, "person"},
    {196.3F, 296.7F, 314.4F, 570.6F, 0, "person"}, {404.3F, 325.7F, 475.5F, 380.9F, 2, "car"},
    {361.5F, 312.5F, 399.6F, 348.6F, 2, "car"},    {70.9F, 300.6F, 138.7F, 513.5F, 0, "person"},
    {0.0F, 327.7F, 55.6F, 455.0F, 24, "backpack"}, {392.0F, 315.7F, 435.4F, 348.3F, 2, "car"},
};

const int TRAFFIC_REFERENCE_COUNT = static_cast<int>(sizeof(TRAFFIC_REFERENCE) / sizeof(TRAFFIC_REFERENCE[0]));

float boxIoU(const DetBox &a, const RefBox &b)
{
    const float x1 = std::max(a.left, b.left);
    const float y1 = std::max(a.top, b.top);
    const float x2 = std::min(a.right, b.right);
    const float y2 = std::min(a.bottom, b.bottom);
    if (x2 <= x1 || y2 <= y1)
        return 0.0F;

    const float intersection = (x2 - x1) * (y2 - y1);
    const float areaA        = (a.right - a.left) * (a.bottom - a.top);
    const float areaB        = (b.right - b.left) * (b.bottom - b.top);
    const float unionArea    = areaA + areaB - intersection;
    return unionArea > 0.0F ? intersection / unionArea : 0.0F;
}

bool checkTopKAgainstReference(const std::vector<DetBox> &detections, int topK, float iouThreshold,
                               const RefBox *reference, int referenceCount)
{
    if (topK <= 0)
        return true;

    if (static_cast<int>(detections.size()) < topK)
    {
        fprintf(stderr, "ERROR: top-%d check needs %d detections, got %zu\n", topK, topK, detections.size());
        return false;
    }

    std::vector<const DetBox *> ranked;
    ranked.reserve(detections.size());
    for (const DetBox &d : detections) ranked.push_back(&d);
    std::stable_sort(ranked.begin(), ranked.end(),
                     [](const DetBox *a, const DetBox *b) { return a->score > b->score; });

    printf("\n  Top-%d check (IoU >= %.2f against the reference detections):\n", topK, iouThreshold);

    // Greedy one-to-one matching in confidence order, as in COCO evaluation:
    // each reference box can be claimed by at most one detection.
    std::vector<bool> claimed(static_cast<size_t>(referenceCount), false);
    bool passed = true;

    for (int i = 0; i < topK; i++)
    {
        const DetBox &d = *ranked[static_cast<size_t>(i)];
        float bestIoU   = 0.0F;
        int bestRef     = -1;

        for (int r = 0; r < referenceCount; r++)
        {
            if (claimed[static_cast<size_t>(r)] || reference[r].classId != d.classId)
                continue;
            const float iou = boxIoU(d, reference[r]);
            if (iou > bestIoU)
            {
                bestIoU = iou;
                bestRef = r;
            }
        }

        const bool matched = (bestRef >= 0 && bestIoU >= iouThreshold);
        if (matched)
            claimed[static_cast<size_t>(bestRef)] = true;
        else
            passed = false;

        printf("    [%d] score=%.3f class=%-3d -> %-9s IoU=%.3f %s\n", i, static_cast<double>(d.score), d.classId,
               bestRef >= 0 ? reference[bestRef].label : "no match", static_cast<double>(bestIoU),
               matched ? "OK" : "FAILED");
    }

    if (!passed)
        fprintf(stderr, "ERROR: top-%d detections do not match the reference\n", topK);

    return passed;
}

} // namespace yolov5_common
