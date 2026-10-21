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

// DFL (Distribution Focal Loss) postprocessing for YOLO11/YOLO12 and similar NMS-free architectures.
//
// Reference: Ultralytics YOLO11/YOLO12 postprocessing pipeline
//   https://github.com/ultralytics/ultralytics
//
// Accepts three separate feature level outputs (P3, P4, P5) as produced by the model, each in
// [H, W, channels] layout where channels = (4 * REG_MAX) + num_classes = 64 + 80 = 144.
//
// Strides and grid sizes are derived from image_height / image_width:
//   P3: stride  8, grid = (image_height/8)  x (image_width/8)
//   P4: stride 16, grid = (image_height/16) x (image_width/16)
//   P5: stride 32, grid = (image_height/32) x (image_width/32)

#include "yolo_postprocess.h"

#include <algorithm>
#include <vector>

static constexpr int REG_MAX = 16; // DFL bins: 16 for YOLO11/YOLO12

// Softmax over 'n' elements in-place
static void softmax(float *x, int n)
{
    float max_val = x[0];
    for (int i = 1; i < n; ++i) max_val = std::max(max_val, x[i]);

    float sum = 0.0f;
    for (int i = 0; i < n; ++i)
    {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < n; ++i) x[i] /= sum;
}

// DFL: decode one distance value from REG_MAX logits via softmax + dot([0..REG_MAX-1])
static float dflDecode(const float *logits)
{
    float bins[REG_MAX];
    for (int i = 0; i < REG_MAX; ++i) bins[i] = logits[i];

    softmax(bins, REG_MAX);

    float dist = 0.0f;
    for (int i = 0; i < REG_MAX; ++i) dist += bins[i] * i;

    return dist;
}

// Process one feature level.
// data layout: [grid_h * grid_w, (4 * REG_MAX) + num_classes]  (HWC, C-last)
static void processLevel(float *data, int grid_h, int grid_w, int stride, int num_classes, float de_conf_threshold,
                         int32_t image_height, int32_t image_width, std::vector<Box> &out)
{
    const int channels = 4 * REG_MAX + num_classes; // 64 + num_classes

    for (int ih = 0; ih < grid_h; ++ih)
    {
        for (int iw = 0; iw < grid_w; ++iw)
        {
            float *anchor_data = data + (ih * grid_w + iw) * channels;

            // --- Class confidence (fast pre-filter before DFL decode) ---
            const float *cls_logits = anchor_data + 4 * REG_MAX;
            float max_logit         = cls_logits[0];
            int best_class          = 0;
            for (int c = 1; c < num_classes; ++c)
            {
                if (cls_logits[c] > max_logit)
                {
                    max_logit  = cls_logits[c];
                    best_class = c;
                }
            }

            // Pre-filter in logit space to avoid sigmoid (expf) for rejected anchors
            if (max_logit < de_conf_threshold)
                continue;

            float confidence = sigmoid(max_logit);

            // --- DFL decode: 4 distances from [lt_x, lt_y, rb_x, rb_y] ---
            // anchor_data layout: [lt_x(REG_MAX), lt_y(REG_MAX), rb_x(REG_MAX), rb_y(REG_MAX), cls...]
            float lt_x = dflDecode(anchor_data + 0 * REG_MAX);
            float lt_y = dflDecode(anchor_data + 1 * REG_MAX);
            float rb_x = dflDecode(anchor_data + 2 * REG_MAX);
            float rb_y = dflDecode(anchor_data + 3 * REG_MAX);

            // Anchor center in grid units (0.5 offset = center of cell)
            float anchor_cx = iw + 0.5f;
            float anchor_cy = ih + 0.5f;

            // Decode to pixel coordinates in model-input space
            float x1 = (anchor_cx - lt_x) * stride;
            float y1 = (anchor_cy - lt_y) * stride;
            float x2 = (anchor_cx + rb_x) * stride;
            float y2 = (anchor_cy + rb_y) * stride;

            x1 = std::clamp(x1, 0.0f, (float)image_width);
            y1 = std::clamp(y1, 0.0f, (float)image_height);
            x2 = std::clamp(x2, 0.0f, (float)image_width);
            y2 = std::clamp(y2, 0.0f, (float)image_height);

            out.emplace_back(x1, y1, x2, y2, confidence, best_class);
        }
    }
}

std::vector<Box> yoloDflPostProcess(float *p3, float *p4, float *p5, int32_t num_classes, int32_t image_height,
                                    int32_t image_width, float conf_threshold)
{
    // Derive grid sizes from image dimensions and fixed strides
    struct Level
    {
        float *data;
        int grid_h, grid_w, stride;
    };

    Level levels[] = {
        {p3, image_height / 8, image_width / 8, 8},
        {p4, image_height / 16, image_width / 16, 16},
        {p5, image_height / 32, image_width / 32, 32},
    };

    // Precompute desigmoid once so processLevel can compare raw logits without expf
    float de_conf_threshold = desigmoid(conf_threshold);

    std::vector<Box> boxes;
    for (auto &lv : levels)
        processLevel(lv.data, lv.grid_h, lv.grid_w, lv.stride, num_classes, de_conf_threshold, image_height,
                     image_width, boxes);

    return boxes;
}
