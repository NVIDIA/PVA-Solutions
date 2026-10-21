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

#include "yolov5_common.h"

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>

namespace yolov5_common {

void writeTimingJson(std::ostream &output, const char *name, const TimingStats &stats)
{
    output << "    \"" << name << "\": {\n";
    output << "      \"min\": " << stats.min << ",\n";
    output << "      \"max\": " << stats.max << ",\n";
    output << "      \"avg\": " << stats.avg << ",\n";
    output << "      \"stddev\": " << stats.stddev << ",\n";
    output << "      \"p50\": " << stats.p50 << ",\n";
    output << "      \"p95\": " << stats.p95 << ",\n";
    output << "      \"p99\": " << stats.p99 << "\n";
    output << "    }";
}

const char *COCO_CLASSES[80] = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"};

const int COCO_CATEGORY_IDS[80] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                   22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                                   46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
                                   67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90};

TimingStats TimingStats::compute(std::vector<float> &samples)
{
    TimingStats stats = {};
    if (samples.empty())
        return stats;
    std::sort(samples.begin(), samples.end());
    stats.min      = samples.front();
    stats.max      = samples.back();
    stats.avg      = std::accumulate(samples.begin(), samples.end(), 0.0f) / samples.size();
    float variance = 0.0f;
    for (float s : samples) variance += (s - stats.avg) * (s - stats.avg);
    stats.stddev = std::sqrt(variance / samples.size());
    stats.p50    = samples[samples.size() * 50 / 100];
    stats.p95    = samples[samples.size() * 95 / 100];
    stats.p99    = samples[samples.size() * 99 / 100];
    return stats;
}

void Timer::start()
{
    m_start = std::chrono::high_resolution_clock::now();
}

double Timer::stopMs()
{
    return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - m_start).count();
}

bool isImageFile(const std::string &filename)
{
    size_t dot = filename.find_last_of(".");
    if (dot == std::string::npos)
        return false;
    std::string ext = filename.substr(dot + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == "png" || ext == "bin");
}

std::vector<std::string> getImageFiles(const std::string &directory)
{
    std::vector<std::string> images;
    DIR *dir = opendir(directory.c_str());
    if (!dir)
    {
        fprintf(stderr, "Error: Cannot open directory %s\n", directory.c_str());
        return images;
    }
    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr)
    {
        std::string filename = entry->d_name;
        if (isImageFile(filename))
            images.push_back(directory + "/" + filename);
    }
    closedir(dir);
    std::sort(images.begin(), images.end());
    return images;
}

int64_t extractImageId(const std::string &filepath)
{
    std::string filename = filepath.substr(filepath.find_last_of("/\\") + 1);
    std::string basename = filename.substr(0, filename.find_last_of("."));
    std::string numStr;
    for (char c : basename)
    {
        if (isdigit(c))
            numStr += c;
    }
    if (!numStr.empty())
    {
        size_t firstNonZero = numStr.find_first_not_of('0');
        if (firstNonZero != std::string::npos)
            numStr = numStr.substr(firstNonZero);
        else
            numStr = "0";
        return std::stoll(numStr);
    }
    std::hash<std::string> hasher;
    return static_cast<int64_t>(hasher(basename) & 0x7FFFFFFF);
}

bool writeCocoJson(const std::vector<CocoDetection> &detections, const std::string &outputPath)
{
    std::ofstream ofs(outputPath);
    if (!ofs.is_open())
    {
        fprintf(stderr, "Error: Cannot open output file %s\n", outputPath.c_str());
        return false;
    }
    ofs << "[\n";
    for (size_t i = 0; i < detections.size(); i++)
    {
        const auto &d = detections[i];
        ofs << "  {\n";
        ofs << "    \"image_id\": " << d.imageId << ",\n";
        ofs << "    \"category_id\": " << d.categoryId << ",\n";
        ofs << "    \"bbox\": [" << d.x << ", " << d.y << ", " << d.width << ", " << d.height << "],\n";
        ofs << "    \"score\": " << d.score << "\n";
        ofs << "  }";
        if (i < detections.size() - 1)
            ofs << ",";
        ofs << "\n";
    }
    ofs << "]\n";
    ofs.close();
    return true;
}

} // namespace yolov5_common
