/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ccl_ref.h"

#include "../../../../src/operator/image_processing/ccl/device/ccl_params.h"

#include <ImageIO.h>
#include <OpCCL.h>
#include <sys/types.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

void CCLRef(NVCVTensorHandle imageHdl, NVCVTensorHandle labelHdl, NVCVTensorHandle numLabelsHdl,
            PVAConnectivityType connectivity, const uint8_t threshold)
{
    NVCVTensorData imageTensorData;
    nvcvTensorExportData(imageHdl, &imageTensorData);

    uint8_t *image = nullptr;
    CupvaMemGetHostPointer((void **)&image, (void *)imageTensorData.buffer.strided.basePtr);

    NVCVTensorData labelTensorData;
    nvcvTensorExportData(labelHdl, &labelTensorData);

    uint16_t *labels = nullptr;
    CupvaMemGetHostPointer((void **)&labels, (void *)labelTensorData.buffer.strided.basePtr);

    NVCVTensorData numLabelsTensorData;
    nvcvTensorExportData(numLabelsHdl, &numLabelsTensorData);

    int32_t *numLabels = nullptr;
    CupvaMemGetHostPointer((void **)&numLabels, (void *)numLabelsTensorData.buffer.strided.basePtr);

    int32_t width, height, lp;
    std::string imageLayout = std::string(nvcvTensorLayoutGetName(&imageTensorData.layout));

    if (imageLayout == "HW")
    {
        width  = imageTensorData.shape[1];
        height = imageTensorData.shape[0];
        lp     = imageTensorData.buffer.strided.strides[0];
    }
    else if (imageLayout == "CHW")
    {
        width  = imageTensorData.shape[2];
        height = imageTensorData.shape[1];
        lp     = imageTensorData.buffer.strided.strides[1];
    }
    else if (imageLayout == "NCHW")
    {
        width  = imageTensorData.shape[3];
        height = imageTensorData.shape[2];
        lp     = imageTensorData.buffer.strided.strides[2];
    }
    else
    {
        assert(false && "Invalid image layout");
        return;
    }

    std::memset(labels, 0, width * height * sizeof(uint16_t));

    int32_t max_labels = width * height;
    std::vector<uint16_t> parent(max_labels + 1, 0);
    uint16_t next_label = 1;

    // Define neighbor offsets for 4- and 8-connectivity
    std::vector<std::pair<int32_t, int32_t>> offsets4 = {{-1, 0}, {0, -1}}; // up, left
    std::vector<std::pair<int32_t, int32_t>> offsets8 = {{-1, -1},
                                                         {-1, 0},
                                                         {-1, 1},
                                                         {0, -1}}; // up-left, up, up-right, left

    const std::vector<std::pair<int32_t, int32_t>> *offsets = nullptr;
    if (connectivity == PVA_8_CONNECTIVITY)
        offsets = &offsets8;
    else // default to 4-connectivity
        offsets = &offsets4;

    auto find = [&](uint16_t x) {
        while (parent[x] != x) x = parent[x];
        return x;
    };
    auto unite = [&](uint16_t x, uint16_t y) {
        x = find(x);
        y = find(y);
        if (x != y)
            parent[std::max(x, y)] = std::min(x, y);
    };

    // 1st pass
    // Divide image into tiles and process by tiles,
    for (int32_t tile_y = 0; tile_y < height; tile_y += CCL_TILEH)
    {
        int32_t tile_h = std::min(CCL_TILEH, height - tile_y);

        for (int32_t tile_x = 0; tile_x < width + CCL_HALOX; tile_x += CCL_TILEW)
        {
            int32_t tile_w = std::min(CCL_TILEW, width + CCL_HALOX - tile_x);

            for (int32_t x = 0; x < tile_w; ++x)
            {
                for (int32_t y = 0; y < tile_h; ++y)
                {
                    int32_t img_y = tile_y + y;
                    int32_t img_x = tile_x + x - 2 * y;
                    if (img_x < 0 || img_x >= width)
                        continue;

                    int32_t idx     = img_y * width + img_x;
                    int32_t img_idx = img_y * lp + img_x;
                    if (image[img_idx] <= threshold)
                    {
                        labels[idx] = 0;
                        continue;
                    }

                    std::vector<uint16_t> neighbor_labels;
                    for (const auto &off : *offsets)
                    {
                        int32_t ny = img_y + off.first;
                        int32_t nx = img_x + off.second;
                        if (ny >= 0 && ny < height && nx >= 0 && nx < width)
                        {

                            uint16_t lbl = labels[ny * width + nx];
                            if (lbl > 0)
                                neighbor_labels.push_back(lbl);
                        }
                    }

                    if (neighbor_labels.empty())
                    {
                        labels[idx]        = next_label;
                        parent[next_label] = next_label;
                        ++next_label;
                    }
                    else
                    {
                        uint16_t min_label = *std::min_element(neighbor_labels.begin(), neighbor_labels.end());
                        labels[idx]        = min_label;
                        for (uint16_t l : neighbor_labels)
                        {
                            if (l != min_label)
                                unite(l, min_label);
                        }
                    }
                }
            }
        }
    }

    // 2nd pass: flatten equivalence and relabel in order
    std::vector<uint16_t> new_label(next_label, 0);
    uint16_t label_count = 0;

    for (int32_t i = 1; i < next_label; ++i)
    {
        uint16_t root = find(i);
        parent[i]     = root;

        if (new_label[root] == 0)
        {
            ++label_count;
            new_label[root] = label_count;
        }
        new_label[i] = new_label[root];
    }

    for (int32_t i = 0; i < width * height; ++i)
    {
        if (labels[i] > 0)
            labels[i] = parent[labels[i]];
    }

    *numLabels = label_count;
}

int32_t checkCCLOutput(NVCVTensorHandle refLabelHdl, NVCVTensorHandle pvaLabelHdl, NVCVTensorHandle refNumLabelsHdl,
                       NVCVTensorHandle pvaNumLabelsHdl)
{
    int32_t err = 0;

    NVCVTensorData refLabelTensorData;
    nvcvTensorExportData(refLabelHdl, &refLabelTensorData);

    uint16_t *ref_h = nullptr;
    CupvaMemGetHostPointer((void **)&ref_h, (void *)refLabelTensorData.buffer.strided.basePtr);

    NVCVTensorData pvaLabelTensorData;
    nvcvTensorExportData(pvaLabelHdl, &pvaLabelTensorData);

    uint16_t *pva_h = nullptr;
    CupvaMemGetHostPointer((void **)&pva_h, (void *)pvaLabelTensorData.buffer.strided.basePtr);

    NVCVTensorData refCountTensorData;
    nvcvTensorExportData(refNumLabelsHdl, &refCountTensorData);

    int32_t *ref_count = nullptr;
    CupvaMemGetHostPointer((void **)&ref_count, (void *)refCountTensorData.buffer.strided.basePtr);

    NVCVTensorData pvaCountTensorData;
    nvcvTensorExportData(pvaNumLabelsHdl, &pvaCountTensorData);

    int32_t *pva_count = nullptr;
    CupvaMemGetHostPointer((void **)&pva_count, (void *)pvaCountTensorData.buffer.strided.basePtr);

    if (*pva_count == -1)
    {
        printf("ERROR: PVA CCL operation failed - intermediate label count exceeded maximum supported limit\n");
        return -1;
    }
    if (*ref_count != *pva_count)
    {
        printf("Mismatch in number of labels: ref = %d, pva = %d\n", *ref_count, *pva_count);
        err = -1;
    }

    int32_t width, height, lp;
    std::string pvaLabelTensorLayout = std::string(nvcvTensorLayoutGetName(&pvaLabelTensorData.layout));
    if (pvaLabelTensorLayout == "HW")
    {
        height = pvaLabelTensorData.shape[0];
        width  = pvaLabelTensorData.shape[1];
        lp     = pvaLabelTensorData.buffer.strided.strides[0] / sizeof(uint16_t);
    }
    else if (pvaLabelTensorLayout == "CHW")
    {
        height = pvaLabelTensorData.shape[1];
        width  = pvaLabelTensorData.shape[2];
        lp     = pvaLabelTensorData.buffer.strided.strides[1] / sizeof(uint16_t);
    }
    else if (pvaLabelTensorLayout == "NCHW")
    {
        height = pvaLabelTensorData.shape[2];
        width  = pvaLabelTensorData.shape[3];
        lp     = pvaLabelTensorData.buffer.strided.strides[2] / sizeof(uint16_t);
    }
    else
    {
        assert(false && "Invalid label tensor layout");
        return -1;
    }

    for (int32_t y = 0; y < height; y++)
    {
        for (int32_t x = 0; x < width; x++)
        {
            uint16_t refData = ref_h[y * width + x];
            uint16_t pvaData = pva_h[y * lp + x];
            if (refData != pvaData)
            {
                printf("Mismatch[%d, %d]: ref = %u, pva = %u\n", y, x, refData, pvaData);
                err = -1;
            }
        }
    }

    return err;
}

void generateRandomCCLBinaryInput(NVCVTensorHandle imageHdl, const int32_t width, const int32_t height,
                                  const int32_t lp)
{
    std::random_device rd;
    std::mt19937 gen{rd()};
    std::uniform_int_distribution<uint8_t> dis{0, 1};

    NVCVTensorData tensorData;
    nvcvTensorExportData(imageHdl, &tensorData);

    uint8_t *imgData = NULL;
    CupvaMemGetHostPointer((void **)&imgData, (void *)tensorData.buffer.strided.basePtr);

    for (int32_t y = 0; y < height; y++)
    {
        for (int32_t x = 0; x < width; x++)
        {
            imgData[y * lp + x] = dis(gen);
        }
    }
}

void generateRandomCCLInput(NVCVTensorHandle imageHdl, const int32_t width, const int32_t height, const int32_t lp)
{
    std::random_device rd;
    std::mt19937 gen{rd()};
    std::uniform_int_distribution<uint8_t> dis{0, 255};

    NVCVTensorData tensorData;
    nvcvTensorExportData(imageHdl, &tensorData);

    uint8_t *imgData = NULL;
    CupvaMemGetHostPointer((void **)&imgData, (void *)tensorData.buffer.strided.basePtr);

    for (int32_t y = 0; y < height; y++)
    {
        for (int32_t x = 0; x < width; x++)
        {
            imgData[y * lp + x] = dis(gen);
        }
    }
}
