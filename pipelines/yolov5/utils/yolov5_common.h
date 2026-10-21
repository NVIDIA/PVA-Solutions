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
 * Shared types and helpers for YOLOv5 pipeline samples (eval, COCO output, timing, file listing).
 */

#ifndef YOLOV5_COMMON_H
#define YOLOV5_COMMON_H

#include <chrono>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace yolov5_common {

// COCO 80 class names (0-indexed)
extern const char *COCO_CLASSES[80];

// COCO category IDs: map model class index 0..79 to COCO 1-indexed category_id
extern const int COCO_CATEGORY_IDS[80];

/** Letterbox parameters for coordinate remapping (model coords -> original) */
struct LetterboxInfo
{
    float scaleX = 1.0f;
    float scaleY = 1.0f;
    float padX   = 0.0f;
    float padY   = 0.0f;
    int origW    = 0;
    int origH    = 0;
};

/** One detection for COCO JSON output */
struct CocoDetection
{
    int64_t imageId;
    int categoryId;
    float x, y, width, height;
    float score;
};

/** Timing statistics (min, max, avg, stddev, percentiles) */
struct TimingStats
{
    float min, max, avg, stddev;
    float p50, p95, p99;
    static TimingStats compute(std::vector<float> &samples);
};

void writeTimingJson(std::ostream &output, const char *name, const TimingStats &stats);

/** Simple CPU timer in milliseconds */
class Timer
{
public:
    void start();
    double stopMs();

private:
    std::chrono::high_resolution_clock::time_point m_start;
};

bool isImageFile(const std::string &filename);
std::vector<std::string> getImageFiles(const std::string &directory);
int64_t extractImageId(const std::string &filepath);
bool writeCocoJson(const std::vector<CocoDetection> &detections, const std::string &outputPath);

} // namespace yolov5_common

#endif // YOLOV5_COMMON_H
