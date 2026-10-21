/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * This file is based on cuDLA-samples/src/yolov5.cpp (https://github.com/NVIDIA-AI-IOT/cuDLA-samples/).
 * It has been modified to be used as a part of NVIDIA PVA-Solutions. This is the original license text.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "yolo_postprocess.h"

#include <algorithm>
#include <cmath>
#include <memory>

// YOLOv5 specific constants
static constexpr float CONFIDENCE_THRESHOLD     = 0.25f;
static constexpr float NMS_THRESHOLD            = 0.45f;
static constexpr int32_t MAX_NUM_BOX_BEFORE_NMS = 10000;
static constexpr int32_t MAX_NUM_BOX_AFTER_NMS  = 1000;
static constexpr int32_t NUM_BOX_ELEMENT        = 6; // left, top, right, bottom, confidence, class

static void initAnchors(float *prior_box, int32_t imgHeight, int32_t imgWidth)
{
    float w[16], h[16];
    int strides[]   = {8, 16, 32};
    int num_anchor  = 3;
    int num_level   = 3;
    float anchors[] = {10.000000,  13.000000, 16.000000,  30.000000,  33.000000,  23.000000,
                       30.000000,  61.000000, 62.000000,  45.000000,  59.000000,  119.000000,
                       116.000000, 90.000000, 156.000000, 198.000000, 373.000000, 326.000000};

    int abs_index = 0;
    for (int i = 0; i < num_level; ++i)
    {
        for (int j = 0; j < num_anchor; ++j)
        {
            int aidx = i * num_anchor + j;
            w[aidx]  = anchors[abs_index++];
            h[aidx]  = anchors[abs_index++];
        }
    }

    float *prior_ptr_start = prior_box;
    for (int ilevel = 0; ilevel < num_level; ++ilevel)
    {
        int stride           = strides[ilevel];
        int fm_width         = imgWidth / stride;
        int fm_height        = imgHeight / stride;
        int anchor_abs_index = ilevel * num_anchor;
        for (int ih = 0; ih < fm_height; ++ih)
        {
            for (int iw = 0; iw < fm_width; ++iw)
            {
                for (int ianchor = 0; ianchor < num_anchor; ++ianchor)
                {
                    *prior_ptr_start++ = iw;
                    *prior_ptr_start++ = ih;
                    *prior_ptr_start++ = w[anchor_abs_index + ianchor];
                    *prior_ptr_start++ = h[anchor_abs_index + ianchor];
                    *prior_ptr_start++ = stride;
                }
            }
        }
    }
}

static void decodeValidate(float *predict, float *parray, float *prior_box, int num_bboxes, int num_classes,
                           float confidence_threshold, float deconfidence_threshold, int max_objects)
{
    int output_idx = 0;
    for (int i = 0; i < num_bboxes && output_idx < MAX_NUM_BOX_AFTER_NMS; i++)
    {
        float *pitem     = predict + i * (num_classes + 5);
        float objectness = pitem[4];
        if (objectness <= deconfidence_threshold)
            continue;

        objectness = sigmoid(objectness);

        float predict_cx = sigmoid(pitem[0]);
        float predict_cy = sigmoid(pitem[1]);
        float predict_w  = sigmoid(pitem[2]);
        float predict_h  = sigmoid(pitem[3]);

        const float *prior_ptr = prior_box + i * 5;
        float stride           = prior_ptr[4];
        float cx               = (predict_cx * 2 - 0.5f + prior_ptr[0]) * stride;
        float cy               = (predict_cy * 2 - 0.5f + prior_ptr[1]) * stride;
        float width            = pow(predict_w * 2, 2.0f) * prior_ptr[2];
        float height           = pow(predict_h * 2, 2.0f) * prior_ptr[3];
        float left             = cx - width * 0.5f;
        float top              = cy - height * 0.5f;
        float right            = cx + width * 0.5f;
        float bottom           = cy + height * 0.5f;

        for (int j = 0; j < num_classes; ++j)
        {
            float class_confidence = pitem[j + 5];
            if (class_confidence <= deconfidence_threshold)
                continue;

            class_confidence = objectness * sigmoid(class_confidence);
            if (class_confidence <= confidence_threshold)
                continue;

            if (output_idx >= max_objects)
                return;

            float *pout_item = parray + 1 + output_idx * NUM_BOX_ELEMENT;
            *pout_item++     = left;
            *pout_item++     = top;
            *pout_item++     = right;
            *pout_item++     = bottom;
            *pout_item++     = class_confidence;
            *pout_item++     = j;

            output_idx++;
        }
    }
    parray[0] = output_idx;
}

static std::vector<Box> cpuNms(std::vector<Box> &boxes, float threshold)
{
    std::sort(boxes.begin(), boxes.end(), [](std::vector<Box>::const_reference a, std::vector<Box>::const_reference b) {
        return a.confidence > b.confidence;
    });

    std::vector<Box> output;
    output.reserve(boxes.size());

    std::vector<bool> remove_flags(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i)
    {
        if (remove_flags[i])
            continue;

        auto &a = boxes[i];
        output.emplace_back(a);

        for (size_t j = i + 1; j < boxes.size(); ++j)
        {
            if (remove_flags[j])
                continue;

            auto &b = boxes[j];
            if (b.class_label == a.class_label)
            {
                if (computeIoU(a, b) >= threshold)
                    remove_flags[j] = true;
            }
        }
    }
    return output;
}

// Public interface
std::vector<Box> yolov5PostProcess(float *predict, int num_bboxes, int32_t num_classes, int32_t image_height,
                                   int32_t image_width)
{
    // Allocate anchors
    std::unique_ptr<float[]> anchors = std::make_unique<float[]>(num_bboxes * 5);
    initAnchors(anchors.get(), image_height, image_width);

    // Allocate parray for intermediate results
    std::unique_ptr<float[]> parray = std::make_unique<float[]>(MAX_NUM_BOX_BEFORE_NMS * NUM_BOX_ELEMENT + 1);

    // Decode and validate
    decodeValidate(predict, parray.get(), anchors.get(), num_bboxes, num_classes, CONFIDENCE_THRESHOLD,
                   desigmoid(CONFIDENCE_THRESHOLD), MAX_NUM_BOX_BEFORE_NMS);

    // Convert to boxes
    std::vector<Box> boxes;
    int count = std::min(MAX_NUM_BOX_BEFORE_NMS, (int)parray[0]);

    for (int i = 0; i < count; i++)
    {
        float *pbox = parray.get() + 1 + i * NUM_BOX_ELEMENT;

        float left      = std::clamp(pbox[0], 0.0f, (float)image_width);
        float top       = std::clamp(pbox[1], 0.0f, (float)image_height);
        float right     = std::clamp(pbox[2], 0.0f, (float)image_width);
        float bottom    = std::clamp(pbox[3], 0.0f, (float)image_height);
        float confident = pbox[4];
        float label     = pbox[5];

        boxes.emplace_back(left, top, right, bottom, confident, label);
    }

    boxes = cpuNms(boxes, NMS_THRESHOLD);
    return boxes;
}