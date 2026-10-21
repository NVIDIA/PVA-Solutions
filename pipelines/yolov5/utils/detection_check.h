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

#ifndef YOLOV5_COMMON_DETECTION_CHECK_H
#define YOLOV5_COMMON_DETECTION_CHECK_H

#include <vector>

namespace yolov5_common {

/// One detection in original-image coordinates.
struct DetBox
{
    float left, top, right, bottom;
    float score;
    int classId;
};

/// One reference detection. Confidence is not compared, only geometry and class.
struct RefBox
{
    float left, top, right, bottom;
    int classId;
    const char *label;
};

/// Reference detections for the packaged traffic image, in original-image
/// (1280x720) coordinates. Captured from the YOLOv5s ONNX model rather than
/// hand annotated, so they are a reference output and not ground truth. The
/// deep_learning/detection sample validates the same image with the same
/// coordinates.
extern const RefBox TRAFFIC_REFERENCE[];
extern const int TRAFFIC_REFERENCE_COUNT;

/// Intersection over union of two boxes.
float boxIoU(const DetBox &a, const RefBox &b);

/// Check that the topK highest-scoring detections each match a distinct
/// reference box of the same class with IoU >= iouThreshold.
///
/// Gating the top few detections rather than the total count keeps the check
/// on the part of the output that is numerically stable. Detections near the
/// confidence threshold move across it between TensorRT versions and GPUs, so
/// an exact count is not portable, while the leading detections agree to
/// better than 1px and 0.002 confidence across platforms.
///
/// Detections must be in original-image coordinates; callers un-letterbox
/// first. Ordering is not assumed. Intended for the packaged traffic image.
bool checkTopKAgainstReference(const std::vector<DetBox> &detections, int topK, float iouThreshold,
                               const RefBox *reference = TRAFFIC_REFERENCE,
                               int referenceCount      = TRAFFIC_REFERENCE_COUNT);

} // namespace yolov5_common

#endif // YOLOV5_COMMON_DETECTION_CHECK_H
