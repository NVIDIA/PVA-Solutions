/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <vector>

struct Box
{
    float left, top, right, bottom, confidence;
    float class_label;

    Box() = default;

    Box(float left, float top, float right, float bottom, float confidence, int class_label)
        : left(left)
        , top(top)
        , right(right)
        , bottom(bottom)
        , confidence(confidence)
        , class_label(class_label)
    {
    }

    Box(float left, float top, float right, float bottom, int class_label)
        : left(left)
        , top(top)
        , right(right)
        , bottom(bottom)
        , confidence(0.0f)
        , class_label(class_label)
    {
    }
};

/**
 * Compute Intersection over Union (IoU) between two bounding boxes
 *
 * Reference: Standard IoU formula used in object detection evaluation
 *   IoU = intersection_area / union_area
 *   where intersection = area of overlapping region
 *   and union = area(box1) + area(box2) - intersection
 *
 * @param a First bounding box
 * @param b Second bounding box
 * @return IoU value in range [0.0, 1.0]
 */
float computeIoU(const Box &a, const Box &b);

/**
 * Perform COCO-style evaluation using IoU-based matching with validation and printing
 *
 * Reference: COCO Evaluation API (pycocotools)
 *   - https://github.com/cocodataset/cocoapi
 *
 * COCO Evaluation Methodology:
 *   1. Sort predictions by confidence (descending)
 *   2. For each prediction, find best matching ground truth (same class, IoU >= threshold)
 *   3. One-to-one matching: each ground truth can match at most one prediction
 *   4. Compute precision = TP / (TP + FP) and recall = TP / (TP + FN)
 *   5. Average Precision (AP) is area under precision-recall curve
 *
 * This implementation uses IoU threshold of 0.5 (AP@0.5 metric)
 *
 * The function evaluates, prints results, and optionally validates against thresholds.
 *
 * @param predictions Vector of detected boxes (will be scaled by invScale)
 * @param groundTruth Array of ground truth boxes (Box objects with confidence=0.0)
 * @param numGroundTruth Number of ground truth boxes
 * @param invScale Scale factor to convert predictions from model space to original image space
 * @param iouThreshold IoU threshold for matching (default 0.5 for AP@0.5)
 * @param minPrecision Optional minimum precision threshold for validation (0.0 = no validation)
 * @param minRecall Optional minimum recall threshold for validation (0.0 = no validation)
 * @return true if validation passes (or if validation is disabled), false otherwise
 */
bool evaluateCocoStyle(const std::vector<Box> &predictions, const Box *groundTruth, int numGroundTruth, float invScale,
                       float iouThreshold = 0.5f, float minPrecision = 0.0f, float minRecall = 0.0f);
