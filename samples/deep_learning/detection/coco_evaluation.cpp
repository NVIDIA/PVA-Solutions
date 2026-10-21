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

#include "coco_evaluation.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

float computeIoU(const Box &a, const Box &b)
{
    float cleft   = std::max(a.left, b.left);
    float ctop    = std::max(a.top, b.top);
    float cright  = std::min(a.right, b.right);
    float cbottom = std::min(a.bottom, b.bottom);

    float c_area = std::max(cright - cleft, 0.0f) * std::max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f)
        return 0.0f;

    float a_area = std::max(0.0f, a.right - a.left) * std::max(0.0f, a.bottom - a.top);
    float b_area = std::max(0.0f, b.right - b.left) * std::max(0.0f, b.bottom - b.top);
    return c_area / (a_area + b_area - c_area);
}

bool evaluateCocoStyle(const std::vector<Box> &predictions, const Box *groundTruth, int numGroundTruth, float invScale,
                       float iouThreshold, float minPrecision, float minRecall)
{
    int truePositives     = 0;
    int falsePositives    = 0;
    int falseNegatives    = 0;
    bool validationPassed = true;

    if (predictions.empty() || groundTruth == nullptr || numGroundTruth == 0)
    {
        printf("\nCOCO-style Evaluation (IoU threshold: %.2f):\n", iouThreshold);
        printf("  Ground Truth Objects: %d\n", numGroundTruth);
        printf("  Predictions:          0\n");
        printf("  True Positives:      0\n");
        printf("  False Positives:     0\n");
        printf("  False Negatives:     %d\n", numGroundTruth);
        printf("  Precision:           0.000\n");
        printf("  Recall:              0.000\n");
        return validationPassed;
    }

    // Sort predictions by confidence (descending) - required for COCO evaluation
    // Reference: COCO evaluation processes detections in confidence order
    //   This ensures high-confidence detections are matched first
    std::vector<std::pair<float, Box>> sortedBoxes;
    for (const Box &b : predictions)
    {
        float left   = b.left * invScale;
        float top    = b.top * invScale;
        float right  = b.right * invScale;
        float bottom = b.bottom * invScale;
        printf("%.1f, %.1f, %.1f, %.1f, %.3f, %d\n", left, top, right, bottom, b.confidence, (int)b.class_label);
        sortedBoxes.push_back({b.confidence, {left, top, right, bottom, b.confidence, (int)b.class_label}});
    }
    std::sort(sortedBoxes.begin(), sortedBoxes.end(), [](const auto &a, const auto &b) { return a.first > b.first; });

    // COCO-style one-to-one matching
    // Reference: COCO evaluation uses greedy matching - each ground truth can match at most one prediction
    //   Algorithm: For each prediction (in confidence order), find best matching unmatched ground truth
    std::vector<bool> gtMatched(numGroundTruth, false);

    for (const auto &[conf, pred] : sortedBoxes)
    {
        float bestIoU = 0.0f;
        int bestGtIdx = -1;

        // Find best matching ground truth (same class, IoU >= threshold, not already matched)
        for (int i = 0; i < numGroundTruth; i++)
        {
            if (gtMatched[i])
                continue; // Already matched to another prediction
            if (pred.class_label != groundTruth[i].class_label)
                continue; // Different class

            float iou = computeIoU(pred, groundTruth[i]);

            if (iou > bestIoU && iou >= iouThreshold)
            {
                bestIoU   = iou;
                bestGtIdx = i;
            }
        }

        if (bestGtIdx >= 0)
        {
            // True positive: prediction matched to ground truth
            gtMatched[bestGtIdx] = true;
            truePositives++;
        }
        else
        {
            // False positive: prediction has no matching ground truth
            falsePositives++;
        }
    }

    // Count unmatched ground truth (false negatives)
    for (int i = 0; i < numGroundTruth; i++)
    {
        if (!gtMatched[i])
            falseNegatives++;
    }

    // Compute precision and recall
    // Reference: Standard detection metrics
    //   Precision = TP / (TP + FP) = fraction of detections that are correct
    //   Recall = TP / (TP + FN) = fraction of ground truth objects detected
    int totalPredictions = truePositives + falsePositives;
    float precision      = (totalPredictions > 0) ? (float)truePositives / totalPredictions : 0.0f;
    float recall         = (numGroundTruth > 0) ? (float)truePositives / numGroundTruth : 0.0f;

    // Print evaluation results
    printf("\nCOCO-style Evaluation (IoU threshold: %.2f):\n", iouThreshold);
    printf("  Ground Truth Objects: %d\n", numGroundTruth);
    printf("  Predictions:          %d\n", totalPredictions);
    printf("  True Positives:      %d\n", truePositives);
    printf("  False Positives:     %d\n", falsePositives);
    printf("  False Negatives:     %d\n", falseNegatives);
    printf("  Precision:           %.3f\n", precision);
    printf("  Recall:              %.3f\n", recall);

    // Perform validation if thresholds are provided (non-zero)
    if (minPrecision > 0.0f || minRecall > 0.0f)
    {
        validationPassed = precision >= minPrecision && recall >= minRecall;
        if (!validationPassed)
        {
            printf("ERROR: Precision (%.3f) or Recall (%.3f) below threshold (%.2f)\n", precision, recall,
                   minPrecision);
        }
        else
        {
            printf("Validation PASSED\n");
        }
    }

    return validationPassed;
}
