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

#include "contour_tracing_ref.h"

#include <nvcv/Tensor.hpp>

#include <OpContourTracing.h>
#include <cupva_host.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace {

constexpr uint8_t DIR_NONE{8U};
constexpr uint32_t NUM_NEIGHBOR{8U};
constexpr uint32_t NUM_NEIGHBOR_PERM{1U << NUM_NEIGHBOR};

typedef struct
{
    int32_t x;
    int32_t y;
    uint8_t prevDir; // R = 0, DR = 1, D = 2, DL = 3, L = 4, UL = 5, U = 6, UR = 7
} BorderPixel;

const int32_t X_OFFSETS[] = {1, 1, 0, -1, -1, -1, 0, 1};
const int32_t Y_OFFSETS[] = {0, 1, 1, 1, 0, -1, -1, -1};
static uint16_t NEXT_PIX_COND_DIR_TBL[NUM_NEIGHBOR_PERM];

/* For each pixel within the given ROI, collect the presence/absence of its 8 neighbors into an 8-bit bitmap. */
template<typename T>
static void collectNeighbors(T *input, std::vector<uint8_t> &roiBitmap, const NVCVRectI &roi, int32_t imgWidth,
                             int32_t imgHeight, int32_t linePitch)
{
    for (int32_t h = 0; h < roi.height; h++)
    {
        for (int32_t w = 0; w < roi.width; w++)
        {
            uint8_t bitmap = 0;
            for (int32_t i = 0; i < NUM_NEIGHBOR; i++)
            {
                int32_t hY   = h + Y_OFFSETS[i];
                int32_t hX   = w + X_OFFSETS[i];
                int32_t y    = roi.y + hY;
                int32_t x    = roi.x + hX;
                bool inRoi   = hY >= 0 && hY < roi.height && hX >= 0 && hX < roi.width;
                bool inImage = y >= 0 && y < imgHeight && x >= 0 && x < imgWidth;
                if (inRoi && inImage && input[y * linePitch + x] != 0)
                {
                    bitmap |= (1U << i);
                }
            }
            roiBitmap[h * roi.width + w] = bitmap;
        }
    }
}

/* Find the first border pixel within an ROI. */
template<typename T>
static bool findStartPixel(T *input, std::vector<uint8_t> &roiBitmap, const NVCVRectI &roi, int32_t linePitch,
                           BorderPixel &startPixel)
{
    for (int32_t h = 0; h < roi.height; h++)
    {
        for (int32_t w = 0; w < roi.width; w++)
        {
            int32_t y = roi.y + h;
            int32_t x = roi.x + w;
            if (input[y * linePitch + x] && ((roiBitmap[h * roi.width + w] & 0x55) != 0x55))
            {
                startPixel = {x, y, 0U};
                return true;
            }
        }
    }
    return false;
}

// Adjustable parameters for random shape generation
constexpr int32_t NUM_VERTICES = 24;  // Number of vertices
constexpr double MIN_RADIUS    = 0.5; // Minimum radius as fraction of half-ROI size
constexpr double MAX_RADIUS    = 1.1; // Maximum radius as fraction of half-ROI size
constexpr double ANGLE_JITTER  = 0.3; // Max angular perturbation as fraction of base step

/* Draw a line from (x0, y0) to (x1, y1) by interpolating N steps. */
template<typename T>
static void drawLine(T *dst, int32_t linePitch, int32_t offsetX, int32_t offsetY, int32_t x0, int32_t y0, int32_t x1,
                     int32_t y1)
{
    int32_t steps = std::max(std::abs(x1 - x0), std::abs(y1 - y0));
    for (int32_t s = 0; s <= steps; s++)
    {
        double t  = (steps == 0) ? 0.0 : static_cast<double>(s) / steps;
        int32_t x = static_cast<int32_t>(std::round(x0 + t * (x1 - x0)));
        int32_t y = static_cast<int32_t>(std::round(y0 + t * (y1 - y0)));

        dst[(offsetY + y) * linePitch + (offsetX + x)] = 1;
    }
}

/* Fill the interior of the polygon using scanline even-odd rule. */
template<typename T>
static void scanlineFill(T *dst, int32_t linePitch, int32_t offsetX, int32_t offsetY, int32_t height,
                         const std::vector<std::pair<int32_t, int32_t>> &verts)
{
    int32_t num = static_cast<int32_t>(verts.size());

    for (int32_t y = 0; y < height; y++)
    {
        std::vector<int32_t> crossings;

        // Traverse each edge of the polygon
        for (int32_t i = 0; i < num; i++)
        {
            int32_t j  = (i + 1) % num;
            int32_t x0 = verts[i].first, y0 = verts[i].second;
            int32_t x1 = verts[j].first, y1 = verts[j].second;
            if ((y0 <= y && y < y1) || (y1 <= y && y < y0))
            {
                int32_t xCross = x0 + static_cast<int32_t>(static_cast<double>(y - y0) / (y1 - y0) * (x1 - x0));
                crossings.push_back(xCross);
            }
        }

        // Sort the crossings according to X coordinate
        std::sort(crossings.begin(), crossings.end());

        for (size_t k = 0; k + 1 < crossings.size(); k += 2)
        {
            for (int32_t x = crossings[k]; x <= crossings[k + 1]; x++)
            {
                dst[(offsetY + y) * linePitch + (offsetX + x)] = 1;
            }
        }
    }
}

/* Print the input tensor within the given ROI. */
template<typename T>
static void printInput(T *inTensorPtr, const NVCVRectI &roi, int32_t linePitch)
{
    printf("ROI (x=%d, y=%d, %dx%d)\n", roi.x, roi.y, roi.width, roi.height);
    for (int32_t h = 0; h < roi.height; h++)
    {
        for (int32_t w = 0; w < roi.width; w++)
        {
            printf("%c ", inTensorPtr[(roi.y + h) * linePitch + (roi.x + w)] ? 'X' : '.');
        }
        printf("\n");
    }
}

/* Generate a random closed polygon inside each ROI using polar coordinate perturbation. */
template<typename T>
void generateInputImpl(NVCVTensorHandle inTensorHandle, const NVCVRectI *rectArray, int32_t numRects)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(inTensorHandle, &tensorData);
    T *inTensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&inTensorPtr, (void *)tensorData.buffer.strided.basePtr);

    const int32_t height    = tensorData.shape[0];
    const int32_t linePitch = tensorData.buffer.strided.strides[0] / sizeof(T);
    memset(inTensorPtr, 0, height * linePitch * sizeof(T));

    std::mt19937 gen(0);
    std::uniform_real_distribution<double> jitGen(-ANGLE_JITTER, ANGLE_JITTER);
    std::uniform_real_distribution<double> radGen(MIN_RADIUS, MAX_RADIUS);

    for (int32_t i = 0; i < numRects; i++)
    {
        const NVCVRectI &roi = rectArray[i];

        if (roi.width < 2 || roi.height < 2)
        {
            printf("ROI %d is too small to generate input. Skip.\n", i);
            continue;
        }

        // Center of the ROI space
        const double cx       = roi.width / 2.0;
        const double cy       = roi.height / 2.0;
        const double halfSize = std::min(cx, cy);

        // Generate polygon vertices in polar coordinates
        const double baseStep = 2.0 * M_PI / NUM_VERTICES;
        std::vector<std::pair<int32_t, int32_t>> verts(NUM_VERTICES);
        for (int32_t v = 0; v < NUM_VERTICES; v++)
        {
            double angle  = v * baseStep + jitGen(gen) * baseStep;
            double radius = radGen(gen) * halfSize;
            int32_t x     = static_cast<int32_t>(std::round(cx + radius * std::cos(angle)));
            int32_t y     = static_cast<int32_t>(std::round(cy + radius * std::sin(angle)));
            x             = std::max(0, std::min(roi.width - 1, x));
            y             = std::max(0, std::min(roi.height - 1, y));
            verts[v]      = {x, y};
        }

        // Rasterize polygon outline and fill interior cells
        const int32_t offsetX = roi.x;
        const int32_t offsetY = roi.y;
        for (int32_t v = 0; v < NUM_VERTICES; v++)
        {
            int32_t nv = (v + 1) % NUM_VERTICES;
            drawLine(inTensorPtr, linePitch, offsetX, offsetY, verts[v].first, verts[v].second, verts[nv].first,
                     verts[nv].second);
        }
        scanlineFill(inTensorPtr, linePitch, offsetX, offsetY, roi.height, verts);
    }
}

/* Find all possible start offsets of the PVA contour in the REF contour. */
static std::vector<int32_t> findStartOffsets(const uint16_t *refContour, const uint16_t *pvaContour, int32_t len)
{
    uint16_t pvaStartX = pvaContour[0];
    uint16_t pvaStartY = pvaContour[1];
    std::vector<int32_t> startOffsets;

    for (int32_t k = 0; k < len; k++)
    {
        if (refContour[2 * k] == pvaStartX && refContour[2 * k + 1] == pvaStartY)
        {
            startOffsets.push_back(k);
        }
    }
    return startOffsets;
}

/* Compare output and length tensors. Contour sequences may start from different start pixels. */
int32_t checkOutputImpl(NVCVTensorHandle refOutTensor, NVCVTensorHandle pvaOutTensor, NVCVTensorHandle refLenTensor,
                        NVCVTensorHandle pvaLenTensor, int32_t numRects)
{
    NVCVTensorData refOutData;
    nvcvTensorExportData(refOutTensor, &refOutData);
    uint16_t *refOutPtr = NULL;
    CupvaMemGetHostPointer((void **)&refOutPtr, (void *)refOutData.buffer.strided.basePtr);

    NVCVTensorData pvaOutData;
    nvcvTensorExportData(pvaOutTensor, &pvaOutData);
    uint16_t *pvaOutPtr = NULL;
    CupvaMemGetHostPointer((void **)&pvaOutPtr, (void *)pvaOutData.buffer.strided.basePtr);

    NVCVTensorData refLenData;
    nvcvTensorExportData(refLenTensor, &refLenData);
    uint16_t *refLenPtr = NULL;
    CupvaMemGetHostPointer((void **)&refLenPtr, (void *)refLenData.buffer.strided.basePtr);

    NVCVTensorData pvaLenData;
    nvcvTensorExportData(pvaLenTensor, &pvaLenData);
    uint16_t *pvaLenPtr = NULL;
    CupvaMemGetHostPointer((void **)&pvaLenPtr, (void *)pvaLenData.buffer.strided.basePtr);

    const int32_t maxContourPixels = refOutData.shape[1];

    int32_t errCnt = 0;
    for (int32_t r = 0; r < numRects; r++)
    {
        uint16_t refLen = refLenPtr[r];
        uint16_t pvaLen = pvaLenPtr[r];

        if (refLen != pvaLen)
        {
            printf("Contour length mismatch at ROI %d: refLen = %u, pvaLen = %u. Skip checking contour pixels.\n", r,
                   refLen, pvaLen);
            errCnt++;
        }
        else
        {
            const uint16_t *refContour        = refOutPtr + r * maxContourPixels * 2;
            const uint16_t *pvaContour        = pvaOutPtr + r * maxContourPixels * 2;
            std::vector<int32_t> startOffsets = findStartOffsets(refContour, pvaContour, refLen);

            if (startOffsets.empty())
            {
                printf("Contour mismatch at ROI %d: pva start pixel (%u, %u) is not found in ref contour.\n", r,
                       pvaContour[0], pvaContour[1]);
                errCnt++;
            }
            else
            {
                bool matchFound = false;
                for (int32_t startOffset : startOffsets)
                {
                    int32_t k = 0;
                    for (; k < refLen; k++)
                    {
                        int32_t refIdx = (k + startOffset) % refLen;
                        uint16_t refX  = refContour[2 * refIdx];
                        uint16_t refY  = refContour[2 * refIdx + 1];
                        uint16_t pvaX  = pvaContour[2 * k];
                        uint16_t pvaY  = pvaContour[2 * k + 1];

                        if (refX != pvaX || refY != pvaY)
                        {
                            break;
                        }
                    }

                    if (k == refLen)
                    {
                        matchFound = true;
                        break;
                    }
                }

                if (!matchFound)
                {
                    printf("Contour mismatch at ROI %d: no matching start offset found.\n", r);
                    errCnt++;
                }
            }
        }
    }

    if (errCnt > 0)
    {
        printf("Found %d mismatches in contour tracing output.\n", errCnt);
    }
    return errCnt;
}

/* For each ROI, traces the boundary of the foreground component and writes ordered (x, y)
 * coordinate pairs to the output tensor, and the contour length to the length tensor. */
template<typename T>
void contourTracingRefImpl(NVCVTensorHandle inTensorHandle, const NVCVRectI *rectArray, int32_t numRects,
                           NVCVTensorHandle outTensorHandle, NVCVTensorHandle lenTensorHandle)
{
    NVCVTensorData inTensorData;
    nvcvTensorExportData(inTensorHandle, &inTensorData);
    T *inTensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&inTensorPtr, (void *)inTensorData.buffer.strided.basePtr);

    NVCVTensorData outTensorData;
    nvcvTensorExportData(outTensorHandle, &outTensorData);
    uint16_t *outTensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&outTensorPtr, (void *)outTensorData.buffer.strided.basePtr);

    NVCVTensorData lenTensorData;
    nvcvTensorExportData(lenTensorHandle, &lenTensorData);
    uint16_t *lenTensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&lenTensorPtr, (void *)lenTensorData.buffer.strided.basePtr);

    const int32_t height    = inTensorData.shape[0];
    const int32_t width     = inTensorData.shape[1];
    const int32_t linePitch = inTensorData.buffer.strided.strides[0] / sizeof(T);

    // Output tensor shape: HWC {maxRects, maxContourPixels, 2}
    const int32_t maxContourPixels = outTensorData.shape[1];

    // Process each ROI
    for (int32_t r = 0; r < numRects; r++)
    {
        const NVCVRectI &roi = rectArray[r];
        uint16_t *roiOutput  = outTensorPtr + r * maxContourPixels * 2;
        std::vector<uint8_t> roiBitmap(roi.width * roi.height, 0);

        // Collect 8-neighbor bitmap for pixels inside this ROI
        collectNeighbors(inTensorPtr, roiBitmap, roi, width, height, linePitch);

        // Find starting border pixel within this ROI
        BorderPixel startPixel;
        if (findStartPixel(inTensorPtr, roiBitmap, roi, linePitch, startPixel))
        {
            // Traverse boundary pixels using bitmap table lookup
            BorderPixel currPixel = startPixel;
            int32_t k             = 0;
            do
            {
                roiOutput[2 * k]     = static_cast<uint16_t>(currPixel.x);
                roiOutput[2 * k + 1] = static_cast<uint16_t>(currPixel.y);
                k++;

                // Lookup next direction from precomputed bitmap table
                int32_t localY  = currPixel.y - roi.y;
                int32_t localX  = currPixel.x - roi.x;
                uint16_t entry  = NEXT_PIX_COND_DIR_TBL[roiBitmap[localY * roi.width + localX]];
                uint8_t nextDir = (entry >> (2 * currPixel.prevDir)) & 0xF;

                if (nextDir != DIR_NONE)
                {
                    currPixel.x += X_OFFSETS[nextDir];
                    currPixel.y += Y_OFFSETS[nextDir];
                    currPixel.prevDir = (nextDir + 2) & 0x6;
                }
                else
                {
                    printf("Next direction is NONE at ROI %d, pixel (%d, %d). Stop tracing.\n", r, currPixel.x,
                           currPixel.y);
                    break;
                }
            } while (k < maxContourPixels && !(currPixel.x == startPixel.x && currPixel.y == startPixel.y));

            lenTensorPtr[r] = k;
        }
        else
        {
            lenTensorPtr[r] = 0;
        }
    }
}

} // namespace

void generateContourTracingInput(NVCVTensorHandle inTensorHandle, const NVCVRectI *rectArray, int32_t numRects)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(inTensorHandle, &tensorData);

    switch (tensorData.dtype)
    {
    case NVCV_DATA_TYPE_U8:
        generateInputImpl<uint8_t>(inTensorHandle, rectArray, numRects);
        break;
    default:
        break;
    }
}

int32_t checkContourTracingOutput(NVCVTensorHandle refOutTensor, NVCVTensorHandle pvaOutTensor,
                                  NVCVTensorHandle refLenTensor, NVCVTensorHandle pvaLenTensor, int32_t numRects)
{
    return checkOutputImpl(refOutTensor, pvaOutTensor, refLenTensor, pvaLenTensor, numRects);
}

void contourTracingRef(NVCVTensorHandle inTensorHandle, const NVCVRectI *rectArray, int32_t numRects,
                       NVCVTensorHandle outTensorHandle, NVCVTensorHandle lenTensorHandle)
{
    NVCVTensorData inTensorData;
    nvcvTensorExportData(inTensorHandle, &inTensorData);

    pvaContourTracingBuildDirectionTable(NEXT_PIX_COND_DIR_TBL);

    switch (inTensorData.dtype)
    {
    case NVCV_DATA_TYPE_U8:
        contourTracingRefImpl<uint8_t>(inTensorHandle, rectArray, numRects, outTensorHandle, lenTensorHandle);
        break;
    default:
        break;
    }
}

void contourTracingMemcpyToTensor(NVCVTensorHandle tensorHandle, const void *src, int32_t numBytes)
{
    NVCVDataType dataType;
    nvcvTensorGetDataType(tensorHandle, &dataType);
    int32_t bitsPerPixel;
    nvcvDataTypeGetBitsPerPixel(dataType, &bitsPerPixel);

    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);
    void *tensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&tensorPtr, (void *)tensorData.buffer.strided.basePtr);

    int32_t sizeInBytes = tensorData.shape[0] * (bitsPerPixel / CHAR_BIT);
    if (numBytes > sizeInBytes)
    {
        printf("memcpyToTensor exceeds maximum tensor size in bytes.\n");
        return;
    }

    memcpy(tensorPtr, src, numBytes);
}
