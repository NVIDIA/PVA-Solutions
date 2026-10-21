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

#include "canny_edge_detector_ref.h"

#include "../../../utils/Geometry.hpp"

#include "../../../../src/operator/image_processing/canny_edge_detector/device/canny_edge_detector_params.h"
#include "canny_edge_detector_filter_coefs.h"

#include <ImageIO.h>
#include <OpCannyEdgeDetector.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int32_t loadCannyImageInput(NVCVTensorHandle imageTensorHandle, const char *directory, const char *filename,
                            const int64_t size)
{
    int32_t err = 0;
    NVCVTensorData imageTensorData;
    nvcvTensorExportData(imageTensorHandle, &imageTensorData);

    uint8_t *image_h = nullptr;
    CupvaMemGetHostPointer((void **)&image_h, (void *)imageTensorData.buffer.strided.basePtr);

    err = ReadImageBuffer(filename, directory, image_h, sizeof(uint8_t) * size);
    return err;
}

int32_t checkCannyEdgeDetectorOutput(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle)
{
    int32_t err = 0;

    NVCVTensorData refTensorData;
    nvcvTensorExportData(refTensorHandle, &refTensorData);

    uint8_t *ref_h = nullptr;
    CupvaMemGetHostPointer((void **)&ref_h, (void *)refTensorData.buffer.strided.basePtr);

    NVCVTensorData pvaTensorData;
    nvcvTensorExportData(pvaTensorHandle, &pvaTensorData);

    uint8_t *pva_h = nullptr;
    CupvaMemGetHostPointer((void **)&pva_h, (void *)pvaTensorData.buffer.strided.basePtr);

    int32_t width, height, linePitch;
    std::string refTensorLayout = std::string(nvcvTensorLayoutGetName(&refTensorData.layout));
    if (refTensorLayout == "HW")
    {
        height    = refTensorData.shape[0];
        width     = refTensorData.shape[1];
        linePitch = refTensorData.buffer.strided.strides[0];
    }
    else if (refTensorLayout == "CHW")
    {
        height    = refTensorData.shape[1];
        width     = refTensorData.shape[2];
        linePitch = refTensorData.buffer.strided.strides[1];
    }
    else if (refTensorLayout == "NCHW")
    {
        height    = refTensorData.shape[2];
        width     = refTensorData.shape[3];
        linePitch = refTensorData.buffer.strided.strides[2];
    }

    for (int32_t y = 0; y < height; y++)
    {
        for (int32_t x = 0; x < width; x++)
        {
            uint8_t refData = ref_h[y * linePitch + x];
            uint8_t pvaData = pva_h[y * linePitch + x];
            if (refData != pvaData)
            {
                printf("Mismatch[%d, %d]: ref = %u, pva = %u\n", y, x, refData, pvaData);
                err = -1;
            }
        }
    }

    return err;
}

void Convolve2D(const uint8_t *input, const int16_t *kernel, int16_t *output, int32_t iw, int32_t ih, int32_t lp,
                int32_t kernelSize, int32_t qbits)
{
    const int32_t kernelRadius = kernelSize / 2;

    for (int32_t y = 0; y < ih; ++y)
    {
        for (int32_t x = 0; x < iw; ++x)
        {
            int32_t sum = 0;

            for (int32_t ky = 0; ky < kernelSize; ++ky)
            {
                for (int32_t kx = 0; kx < kernelSize; ++kx)
                {
                    int32_t imgY = y + ky - kernelRadius;
                    int32_t imgX = x + kx - kernelRadius;

                    if (imgX >= 0 && imgX < iw && imgY >= 0 && imgY < ih)
                    {
                        int32_t pixelVal = static_cast<int32_t>(input[imgY * lp + imgX]);
                        sum += kernel[ky * kernelSize + kx] * pixelVal;
                    }
                    // Out of bounds pixels contribute 0 to the sum
                }
            }

            if (qbits > 0)
            {
                sum = (sum + (1 << (qbits - 1))) >> qbits;
            }
            output[y * iw + x] = static_cast<int16_t>(sum);
        }
    }
}

void CannyEdgeDetectorFixedPointRef(NVCVTensorHandle imageTensorHandle, NVCVTensorHandle edgeMapTensorHandle,
                                    const int32_t gradientSize, const int32_t thresholdStrong,
                                    const int32_t thresholdWeak)
{
    NVCVTensorData imageTensorData;
    nvcvTensorExportData(imageTensorHandle, &imageTensorData);

    uint8_t *image = nullptr;
    CupvaMemGetHostPointer((void **)&image, (void *)imageTensorData.buffer.strided.basePtr);

    NVCVTensorData edgeMapTensorData;
    nvcvTensorExportData(edgeMapTensorHandle, &edgeMapTensorData);

    uint8_t *edgeMap = nullptr;
    CupvaMemGetHostPointer((void **)&edgeMap, (void *)edgeMapTensorData.buffer.strided.basePtr);

    int32_t iw, ih, lp;
    std::string imageLayout = std::string(nvcvTensorLayoutGetName(&imageTensorData.layout));
    if (imageLayout == "HW")
    {
        iw = imageTensorData.shape[1];
        ih = imageTensorData.shape[0];
        lp = imageTensorData.buffer.strided.strides[0];
    }
    else if (imageLayout == "CHW")
    {
        iw = imageTensorData.shape[2];
        ih = imageTensorData.shape[1];
        lp = imageTensorData.buffer.strided.strides[1];
    }
    else if (imageLayout == "NCHW")
    {
        iw = imageTensorData.shape[3];
        ih = imageTensorData.shape[2];
        lp = imageTensorData.buffer.strided.strides[2];
    }
    else
    {
        assert(false && "Invalid image layout");
        return;
    }

    std::vector<int16_t> gx(iw * ih);
    std::vector<int16_t> gy(iw * ih);

    const int16_t *kernel       = nullptr;
    const int16_t *kernelTransp = nullptr;
    int32_t qbits               = 0;
    switch (gradientSize)
    {
    case 3:
        kernel       = &kSobel3Coef[0][0];
        kernelTransp = &kSobel3CoefTransp[0][0];
        break;
    case 5:
        kernel       = &kSobel5Coef[0][0];
        kernelTransp = &kSobel5CoefTransp[0][0];
        break;
    case 7:
        kernel       = &kSobel7Coef[0][0];
        kernelTransp = &kSobel7CoefTransp[0][0];
        qbits        = QBITS_7;
        break;
    default:
        assert(false && "Invalid gradient size");
        return;
    }
    Convolve2D(image, kernel, gx.data(), iw, ih, lp, gradientSize, qbits);
    Convolve2D(image, kernelTransp, gy.data(), iw, ih, lp, gradientSize, qbits);

    std::vector<int32_t> intensity(iw * ih);
    for (int32_t j = 0; j < ih; j++)
    {
        for (int32_t i = 0; i < iw; i++)
        {
            int32_t gradX = static_cast<int32_t>(gx[j * iw + i]);
            int32_t gradY = static_cast<int32_t>(gy[j * iw + i]);

            int32_t L2Norm        = static_cast<int32_t>(std::sqrt(static_cast<float>(gradX * gradX + gradY * gradY)));
            intensity[j * iw + i] = L2Norm;
        }
    }

    std::vector<int32_t> edgeDir(iw * ih);
    for (int32_t j = 0; j < ih; j++)
    {
        for (int32_t i = 0; i < iw; i++)
        {
            int16_t gradX    = gx[j * iw + i];
            int16_t gradY    = gy[j * iw + i];
            int16_t absGradX = std::abs(gradX);
            int16_t absGradY = std::abs(gradY);
            bool isGradXPos  = gradX >= 0;
            bool isGradYPos  = gradY >= 0;

            int32_t sign = (isGradXPos ^ isGradYPos) ? 2 : 0;

            const int32_t tg22x = static_cast<int32_t>(absGradX) * TG22;
            const int32_t tg67x = static_cast<int32_t>(absGradX) * TG67;
            int32_t dyVal_int   = static_cast<int32_t>(absGradY) * (1 << CANNY_SHIFT);

            int dir = NUM_DIR;

            if (dyVal_int < tg22x)
            {
                // gradient direction horizontal
                dir = DIR_HORI;
            }
            else if (dyVal_int > tg67x)
            {
                // gradient direction is vertical
                dir = DIR_VERT;
            }
            else
            {
                // both gradX and gradY are same sign, then dir is 45 degree, otherwise
                // it is -45 degree
                dir = DIR_DIAG0 + sign;
            }
            edgeDir[j * iw + i] = dir;
        }
    }

    int32_t numStrongEdges = 0;
    int32_t numWeakEdges   = 0;

    std::vector<int32_t> edgeType(iw * ih);

    const int32_t dx[NUM_DIR] = {1, 1, 0, 1};  // Horizontal, Diagonal0, Vertical, Diagonal1
    const int32_t dy[NUM_DIR] = {0, 1, 1, -1}; // Horizontal, Diagonal0, Vertical, Diagonal1

    for (int32_t j = 0; j < ih; j++)
    {
        for (int32_t i = 0; i < iw; i++)
        {
            int32_t dir = edgeDir[j * iw + i];

            int32_t prevX = std::max(0, std::min(iw - 1, i - dx[dir]));
            int32_t prevY = std::max(0, std::min(ih - 1, j - dy[dir]));
            int32_t nextX = std::max(0, std::min(iw - 1, i + dx[dir]));
            int32_t nextY = std::max(0, std::min(ih - 1, j + dy[dir]));

            int32_t prevScore = intensity[prevY * iw + prevX];
            int32_t nextScore = intensity[nextY * iw + nextX];
            int32_t curScore  = intensity[j * iw + i];

            // Non-maximum suppression
            if (curScore > prevScore && curScore >= nextScore)
            {
                if (curScore > thresholdStrong)
                {
                    edgeType[j * iw + i] = EDGE_STRONG;
                    numStrongEdges++;
                }
                else if (curScore > thresholdWeak)
                {
                    edgeType[j * iw + i] = EDGE_WEAK;
                    numWeakEdges++;
                }
                else
                {
                    edgeType[j * iw + i] = EDGE_NO;
                }
            }
            else
            {
                edgeType[j * iw + i] = EDGE_NO;
            }
        }
    }

    // Connected Component Labeling (CCL)
    std::vector<int32_t> labels(iw * ih);

    // Define the 4 neighbors to check (top-left, top, top-right, left)
    const Point2i ccl_neighbors[4] = {
        {-1, -1}, // top-left
        {0, -1},  // top
        {1, -1},  // top-right
        {-1, 0}   // left
    };

    // Initialize labels: 0 for non-edges and weak edges, 1 for strong edges
    for (int32_t j = 0; j < ih; j++)
    {
        for (int32_t i = 0; i < iw; i++)
        {
            labels[j * iw + i] = (edgeType[j * iw + i] == EDGE_STRONG) ? 1 : 0;
        }
    }

    // Plabel for weak edges
    int32_t nextLabel                      = 2;      // Start weak edge labels from 2
    std::vector<int32_t> labelEquivalences = {0, 1}; // Reserve 0 and 1

    for (int32_t j = 0; j < ih; j++)
    {
        for (int32_t i = 0; i < iw; i++)
        {
            if (edgeType[j * iw + i] == EDGE_STRONG)
            {
                // For strong edges, set their neighbors' labels to 1 if they have valid
                // labels
                for (int32_t n = 0; n < 4; n++)
                {
                    int32_t ny = j + ccl_neighbors[n].y;
                    int32_t nx = i + ccl_neighbors[n].x;

                    if (nx >= 0 && nx < iw && ny >= 0 && ny < ih)
                    {
                        if (edgeType[ny * iw + nx] == EDGE_WEAK)
                        {

                            int32_t neighborLabel = labels[ny * iw + nx];

                            // If the neighbor has a label in labelEquivalences, chase its
                            // root
                            if (neighborLabel < labelEquivalences.size())
                            {
                                int32_t current = neighborLabel;
                                while (current != labelEquivalences[current])
                                {
                                    int32_t next = labelEquivalences[current];
                                    current      = next;
                                }
                                labelEquivalences[current] = 1;
                            }

                            // Set the neighbor's label to 1
                            labels[ny * iw + nx] = 1;
                        }
                    }
                }
            }
            else if (edgeType[j * iw + i] == EDGE_WEAK)
            {

                std::vector<int32_t> neighborLabels;

                for (int32_t n = 0; n < 4; n++)
                {
                    int32_t ny = j + ccl_neighbors[n].y;
                    int32_t nx = i + ccl_neighbors[n].x;

                    if (nx >= 0 && nx < iw && ny >= 0 && ny < ih)
                    {
                        if (edgeType[ny * iw + nx] == EDGE_WEAK || edgeType[ny * iw + nx] == EDGE_STRONG)
                        {
                            neighborLabels.push_back(labels[ny * iw + nx]);
                        }
                    }
                }

                if (!neighborLabels.empty())
                {
                    // Find the smallest label among neighbors
                    int minLabel       = *std::min_element(neighborLabels.begin(), neighborLabels.end());
                    labels[j * iw + i] = minLabel;

                    // Record equivalences for all neighbor labels
                    for (int32_t neighborLabel : neighborLabels)
                    {
                        if (neighborLabel != minLabel)
                        {
                            // Find the root of the neighbor label
                            int32_t neighborRoot = neighborLabel;
                            while (neighborRoot != labelEquivalences[neighborRoot])
                            {
                                neighborRoot = labelEquivalences[neighborRoot];
                            }

                            // Find the root of the minimum label
                            int32_t minLabelRoot = minLabel;
                            while (minLabelRoot != labelEquivalences[minLabelRoot])
                            {
                                minLabelRoot = labelEquivalences[minLabelRoot];
                            }

                            // Always point to the smaller root label
                            int32_t targetRoot = std::min(minLabelRoot, neighborRoot);

                            // Update both roots to point to the target root
                            labelEquivalences[minLabelRoot] = targetRoot;
                            labelEquivalences[neighborRoot] = targetRoot;
                        }
                    }
                }
                else
                {
                    // No labeled neighbors, create a new label
                    labels[j * iw + i] = nextLabel;

                    // Ensure the equivalence vector is large enough
                    while (labelEquivalences.size() <= nextLabel)
                    {
                        labelEquivalences.push_back(labelEquivalences.size());
                    }
                    nextLabel++;
                }
            }
        }
    }

    // Merge equivalent labels
    for (int32_t i = 2; i < labelEquivalences.size(); i++)
    {
        int32_t root = i;
        while (root != labelEquivalences[root])
        {
            root = labelEquivalences[root];
        }

        // Path compression
        int32_t current = i;
        while (current != labelEquivalences[current])
        {
            int32_t next               = labelEquivalences[current];
            labelEquivalences[current] = root;
            current                    = next;
        }
    }

    // Apply the flattened equivalences to the labels and generate edge map
    for (int32_t j = 0; j < ih; j++)
    {
        for (int32_t i = 0; i < iw; i++)
        {
            uint8_t currentEdgeType = edgeType[j * iw + i];
            int32_t currentLabel    = labels[j * iw + i];

            if (currentEdgeType == EDGE_STRONG)
            {
                edgeMap[j * lp + i] = 255;
            }
            else if (currentEdgeType == EDGE_WEAK)
            {
                if (labelEquivalences[currentLabel] == 1)
                {
                    edgeMap[j * lp + i] = 255;
                }
                else
                {
                    edgeMap[j * lp + i] = 0;
                }
            }
            else // EDGE_NO
            {
                edgeMap[j * lp + i] = 0;
            }
        }
    }
}