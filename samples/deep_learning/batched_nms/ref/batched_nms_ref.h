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

/*
 * @file batched_nms_ref.h
 *
 * @brief Defines the CPU reference model of the BatchedNMS operator.
 *
 * The reference follows the operator's algorithm: classes are processed in order of first
 * appearance, boxes of a class are visited in descending score order, a box is kept unless a
 * previously kept box of its class overlaps it with IoU > iouThreshold, and processing stops
 * after maxOutput boxes have been kept. It differs from the operator in the following ways:
 *
 * - It does not apply the operator's limit of 4096 boxes per class. For inputs with more than
 *   4096 boxes of one class the operator ignores the boxes after the first 4096 of that class
 *   in input order, and the results can differ.
 * - It compares scores as the float values passed to it. The operator orders FP16 scores, so
 *   pass scores that are exactly representable in FP16 to reproduce the operator's order.
 * - It keeps boxes with equal scores in input order. The operator's order of boxes with equal
 *   scores is unspecified, so the results can differ in which of two tied boxes is kept.
 * - It computes IoU as a division, the operator compares the intersection against the
 *   threshold multiplied by the union. Boxes whose IoU is within rounding of the threshold can
 *   be suppressed by one and kept by the other.
 */

#ifndef BATCHED_NMS_REF_H
#define BATCHED_NMS_REF_H

#include <cstdint>
#include <vector>

/**
 * @brief One box kept by the reference model.
 */
struct BatchedNMSRefBox
{
    float left, top, right, bottom;
    float score;
    int classId;
};

/**
 * @brief Runs the Batched NMS reference model.
 *
 * @param left, top, right, bottom Box coordinates of numBoxes input boxes.
 * @param scores Scores of the input boxes.
 * @param classes Class identifiers of the input boxes. Boxes with identifiers outside [0, 299]
 *                are ignored.
 * @param numBoxes Number of input boxes.
 * @param iouThreshold IoU threshold. A box is suppressed when its IoU with a kept box of the
 *                     same class is greater than this value.
 * @param maxOutput Maximum number of boxes to keep across all classes.
 *
 * @return The kept boxes in output order.
 */
std::vector<BatchedNMSRefBox> batchedNMSRef(const float *left, const float *top, const float *right,
                                            const float *bottom, const float *scores, const int16_t *classes,
                                            int numBoxes, float iouThreshold, int maxOutput);

#endif
