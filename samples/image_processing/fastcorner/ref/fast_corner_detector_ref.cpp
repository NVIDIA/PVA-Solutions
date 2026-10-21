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

#include "fast_corner_detector_ref.h"

#include "../../../../src/operator/image_processing/fastcorner/device/fast_corner_detector_param.hpp"
#include "../../../utils/Geometry.hpp"

#include "ImageIO.h"

#include <cupva_host_utils.hpp>

#include <OpFastCorner.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ios>
#include <iostream>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

namespace {

template<typename T>
struct TileInfo
{
    T *tp;         // pointer to start of tile
    int32_t tileX; // tile width
    int32_t tileY; // tile height
    int32_t lp;    // line pitch
};

template<typename T>
void fastCornerDetectorTileRef(TileInfo<T> const *src, TileInfo<std::make_unsigned_t<T>> const *dst, int32_t threshold)
{
    T *pSrc                       = src->tp;
    std::make_unsigned_t<T> *pDst = dst->tp;
    int32_t tileX                 = dst->tileX;
    int32_t tileY                 = dst->tileY;

    // Points' relative axis on the circle, the top left point's axis is {0, 0}.
    // The sequence is arranged in a clockwise order.
    const std::array<Point2<int32_t>, 16> offset{
        // clang-format off
        {{2, 0}, {3, 0}, {4, 0}, {5, 1}, {6, 2}, {6, 3}, {6, 4}, {5, 5},
        {4, 6}, {3, 6}, {2, 6}, {1, 5}, {0, 4}, {0, 3}, {0, 2}, {1, 1}}
        // clang-format on
    };

    for (int32_t i = 0; i < tileY; i++)
    {
        for (int32_t j = 0; j < tileX; j++)
        {
            std::vector<T> circle;
            for (size_t k = 0; k < offset.size(); k++)
            {
                int32_t index = (i + offset[k].y) * src->lp + (j + offset[k].x);
                circle.push_back(pSrc[index]);
            }

            std::vector<T> min9Vec;
            std::vector<T> max9Vec;
            for (size_t k = 0; k < circle.size(); k++)
            {
                T min9 = circle[(k + 0) % circle.size()];
                T max9 = circle[(k + 0) % circle.size()];
                for (int32_t l = 1; l < 9; l++)
                {
                    min9 = std::min(min9, circle[(k + l) % circle.size()]);
                    max9 = std::max(max9, circle[(k + l) % circle.size()]);
                }
                min9Vec.push_back(min9);
                max9Vec.push_back(max9);
            }
            auto it     = std::max_element(min9Vec.begin(), min9Vec.end());
            T maxOfMin9 = *it;

            it          = std::min_element(max9Vec.begin(), max9Vec.end());
            T minOfMax9 = *it;

            T center      = pSrc[(i + 3) * src->lp + (j + 3)];
            int32_t index = i * dst->lp + j;

            if (maxOfMin9 > threshold + center)
            {
                assert(maxOfMin9 > center);
                pDst[index] = maxOfMin9 - center;
            }
            else if (minOfMax9 + threshold < center)
            {
                assert(minOfMax9 < center);
                pDst[index] = center - minOfMax9;
            }
            else
            {
                pDst[index] = 0;
            }
        }
    }
}

template<typename T>
void nmsPackTileRef(bool isNmsOn, TileInfo<T> const *src, Point2<int32_t> tilePos, int32_t maxLocNum, Point2f *loc,
                    T *score, int32_t &numLoc)
{
    T *pSrc       = src->tp;
    int32_t pitch = src->lp;
    int32_t tileX = src->tileX;
    int32_t tileY = src->tileY;

    for (int32_t j = 0; j < tileY; j++)
    {
        for (int32_t i = 0; i < tileX; i++)
        {
            T center = pSrc[(j + 1) * pitch + (i + 1)];

            if ((isNmsOn == false && center > 0) ||
                (center > pSrc[j * pitch + i] && center > pSrc[j * pitch + i + 1] && center > pSrc[j * pitch + i + 2] &&
                 center > pSrc[(j + 1) * pitch + i] && center > pSrc[(j + 1) * pitch + i + 2] &&
                 center > pSrc[(j + 2) * pitch + i] && center > pSrc[(j + 2) * pitch + i + 1] &&
                 center > pSrc[(j + 2) * pitch + i + 2]))
            {
                if (numLoc > maxLocNum - 1)
                {
                    continue;
                }
                int32_t x = i + tilePos.x;
                int32_t y = j + tilePos.y;
                *score++  = center;
                Point2f pt{static_cast<float>(x), static_cast<float>(y)};
                *loc++ = pt;
                numLoc++;
            }
        }
    }
}

template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value || std::is_same<T, uint16_t>::value ||
                                                 std::is_same<T, int8_t>::value || std::is_same<T, int16_t>::value>>
int32_t generateFastCornerDetectorInputImpl(NVCVTensorHandle tensorHandle, char const *fileAbsPathName)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    T *tensorPtr_h = nullptr;
    CupvaMemGetHostPointer((void **)&tensorPtr_h, (void *)tensorData.buffer.strided.basePtr);

    const int32_t width  = tensorData.shape[1];
    const int32_t height = tensorData.shape[0];
    int32_t linePitch    = tensorData.buffer.strided.strides[0] / sizeof(T);

    std::mt19937 gen;
    gen.seed(1);
    std::uniform_int_distribution<> distrib(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

    const bool isRandom = fileAbsPathName == nullptr;
    std::vector<T> rawImageData(width * height);
    if (!isRandom)
    {
        const std::string path(fileAbsPathName);
        auto const pos       = path.find_last_of('/');
        const auto directory = path.substr(0, pos);
        const auto filename  = path.substr(pos + 1);

        int err = ReadCvtImageBuffer(filename.c_str(), directory.c_str(), sizeof(uint16_t), sizeof(T),
                                     rawImageData.size(), (void *)rawImageData.data());
        if (err)
        {
            return err;
        }
    }

    int32_t cnt{0};
    auto generator = [&]() {
        return static_cast<T>(isRandom ? (distrib(gen)) : rawImageData[cnt++]);
    };

    for (int32_t i = 0; i < height; i++)
    {
        for (int32_t j = 0; j < width; j++)
        {
            tensorPtr_h[i * linePitch + j] = generator();
        }
    }
    return 0;
}

template<typename T>
void fastCornerDetectorRefImpl(NVCVTensorHandle inTensorHandle, PvaFastCornerDetectorParams const *params,
                               int32_t intensityThreshold, const NVCVBorderType borderMode, int32_t borderValue,
                               NVCVTensorHandle locHandle, NVCVTensorHandle numLocHandle)
{
    NVCVTensorData inTensorData;
    nvcvTensorExportData(inTensorHandle, &inTensorData);

    NVCVTensorData locData;
    nvcvTensorExportData(locHandle, &locData);

    T *img_h = nullptr;
    CupvaMemGetHostPointer((void **)&img_h, (void *)inTensorData.buffer.strided.basePtr);

    int32_t *numLoc_h = nullptr;
    NVCVTensorData numLocData;
    nvcvTensorExportData(numLocHandle, &numLocData);
    CupvaMemGetHostPointer((void **)&numLoc_h, (void *)numLocData.buffer.strided.basePtr);
    *numLoc_h = 0;

    const int32_t width        = inTensorData.shape[1];
    const int32_t height       = inTensorData.shape[0];
    const int32_t linePitch    = inTensorData.buffer.strided.strides[0] / sizeof(T);
    const int32_t circleRadius = params->circleRadius;

    // The reference codes use the same tile size as in the implementation, otherwise
    // the output needs to be sorted before comparing.
    const auto [tileX, tileY] = getTileSz<T>();

    const int32_t haloX{circleRadius};
    const int32_t haloY{circleRadius};
    const int32_t niterX = (width + tileX - 1) / tileX;
    const int32_t niterY = (height + tileY - 1) / tileY;

    std::vector<std::make_unsigned_t<T>> score(width * height);
    std::vector<T> imgPad((width + 2 * haloX) * (height + 2 * haloY), borderValue);
    const int32_t imgPadLinePitch = width + 2 * haloX;
    for (int32_t y = 0; y < height; y++)
    {
        for (int32_t x = 0; x < width; x++)
        {
            const int32_t offset                     = circleRadius * imgPadLinePitch + circleRadius;
            imgPad[offset + y * imgPadLinePitch + x] = img_h[y * linePitch + x];
        }
    }

    // If true, then rewrite the border value.
    if (borderMode == NVCV_BORDER_REPLICATE)
    {
        // Replicate left and right borders.
        for (int32_t y = 0; y < height; y++)
        {
            for (int32_t x = 0; x < circleRadius; x++)
            {
                // Replicate left border.
                imgPad[(y + circleRadius) * imgPadLinePitch + x] =
                    imgPad[(y + circleRadius) * imgPadLinePitch + circleRadius];
                // Replicate right border.
                imgPad[(y + circleRadius) * imgPadLinePitch + imgPadLinePitch - 1 - x] =
                    imgPad[(y + circleRadius) * imgPadLinePitch + imgPadLinePitch - 1 - circleRadius];
            }
        }

        // Replicate top and bottom borders.
        for (int32_t y = 0; y < circleRadius; y++)
        {
            // Replicate top border.
            memcpy(&imgPad[y * imgPadLinePitch], &imgPad[circleRadius * imgPadLinePitch], imgPadLinePitch * sizeof(T));
            // Replicate bottom border.
            memcpy(&imgPad[(height + 2 * circleRadius - 1 - y) * imgPadLinePitch],
                   &imgPad[(height + 2 * circleRadius - 1 - circleRadius) * imgPadLinePitch],
                   imgPadLinePitch * sizeof(T));
        }
    }

    const int32_t lastTileX = width % tileX ? width % tileX : tileX;
    const int32_t lastTileY = height % tileY ? height % tileY : tileY;
    for (int32_t y = 0; y < niterY; y++)
    {
        for (int32_t x = 0; x < niterX; x++)
        {
            TileInfo<T> in;
            TileInfo<std::make_unsigned_t<T>> out;
            in.tp    = &imgPad[y * imgPadLinePitch * tileY + x * tileX];
            in.tileX = (x == niterX - 1) ? (lastTileX + 2 * haloX) : (tileX + 2 * haloX);
            in.tileY = (y == niterY - 1) ? (lastTileY + 2 * haloY) : (tileY + 2 * haloY);
            in.lp    = imgPadLinePitch;

            out.tp    = &score[y * width * tileY + x * tileX];
            out.tileX = (x == niterX - 1) ? lastTileX : tileX;
            out.tileY = (y == niterY - 1) ? lastTileY : tileY;
            out.lp    = width;

            fastCornerDetectorTileRef(&in, &out, intensityThreshold);
        }
    }

    // NMS and pack
    const bool isNmsOn    = params->nonMaxSuppression != 0;
    const bool isTopKOn   = params->numTopKFeatures > 0;
    const int32_t maxFeat = params->numTopKFeatures;

    // Apply NMS to get filtered scores (if NMS enabled, zero out non-local-maxima)
    std::vector<std::make_unsigned_t<T>> scoreFiltered(width * height);
    if (isNmsOn)
    {
        // Create padded buffer for NMS boundary handling
        std::vector<std::make_unsigned_t<T>> scorePadNms((width + 2) * (height + 2), 0);
        int32_t padPitch = width + 2;
        for (int32_t j = 0; j < height; j++)
            std::memcpy(&scorePadNms[(j + 1) * padPitch + 1], &score[j * width],
                        width * sizeof(std::make_unsigned_t<T>));

        for (int32_t y = 0; y < height; y++)
        {
            for (int32_t x = 0; x < width; x++)
            {
                auto center = scorePadNms[(y + 1) * padPitch + (x + 1)];
                bool isLocMax =
                    (center > scorePadNms[y * padPitch + x] && center > scorePadNms[y * padPitch + x + 1] &&
                     center > scorePadNms[y * padPitch + x + 2] && center > scorePadNms[(y + 1) * padPitch + x] &&
                     center > scorePadNms[(y + 1) * padPitch + x + 2] && center > scorePadNms[(y + 2) * padPitch + x] &&
                     center > scorePadNms[(y + 2) * padPitch + x + 1] &&
                     center > scorePadNms[(y + 2) * padPitch + x + 2]);
                scoreFiltered[y * width + x] = isLocMax ? center : 0;
            }
        }
    }
    else
    {
        scoreFiltered = score;
    }

    Point2f *loc_h = nullptr;
    CupvaMemGetHostPointer((void **)&loc_h, (void *)locData.buffer.strided.basePtr);
    const int32_t maxLocNum = locData.shape[0];

    if (isTopKOn)
    {
        // Pad filtered scores for TopK tile-based packing
        std::vector<std::make_unsigned_t<T>> scorePad((width + 2) * (height + 2), 0);
        int32_t scorePadLinePitch = width + 2;
        for (int32_t j = 0; j < height; j++)
            std::memcpy(&scorePad[(j + 1) * scorePadLinePitch + 1], &scoreFiltered[j * width],
                        width * sizeof(std::make_unsigned_t<T>));
        // TopK: Pack all corners (up to 250k), build histogram, compute threshold, filter
        constexpr int32_t topKCapacity = PVA_FAST_CORNER_MAX_TOPK_FEATURES;

        // Phase 1: Pack all corners with scores
        std::vector<Point2f> tempLocs;
        std::vector<std::make_unsigned_t<T>> tempScores;
        tempLocs.reserve(topKCapacity);
        tempScores.reserve(topKCapacity);
        int32_t packedCount = 0;

        for (int32_t ty = 0; ty < niterY && packedCount < topKCapacity; ty++)
        {
            for (int32_t tx = 0; tx < niterX && packedCount < topKCapacity; tx++)
            {
                int32_t curTileX = (tx == niterX - 1) ? lastTileX : tileX;
                int32_t curTileY = (ty == niterY - 1) ? lastTileY : tileY;

                int32_t tilePackedCount = 0;
                for (int32_t j = 0; j < curTileY; j++)
                {
                    for (int32_t i = 0; i < curTileX; i++)
                    {
                        auto s = scorePad[(ty * tileY + j + 1) * scorePadLinePitch + (tx * tileX + i + 1)];
                        if (s > 0)
                        {
                            float fx = static_cast<float>(tx * tileX + i);
                            float fy = static_cast<float>(ty * tileY + j);
                            tempLocs.push_back({fx, fy});
                            tempScores.push_back(s);
                            tilePackedCount++;
                        }
                    }
                }

                int32_t writeCount = tilePackedCount;
                if (packedCount + writeCount > topKCapacity)
                    writeCount = topKCapacity - packedCount;
                packedCount += writeCount;
            }
        }

        // Phase 2: Build histogram and compute threshold
        int32_t totalPacked        = static_cast<int32_t>(tempScores.size());
        constexpr int32_t histBins = 256;
        uint32_t hist[histBins]    = {};
        for (int32_t i = 0; i < totalPacked; i++)
        {
            uint32_t bin;
            if constexpr (sizeof(T) == 1)
                bin = static_cast<uint32_t>(tempScores[i]);
            else
                bin = static_cast<uint32_t>(tempScores[i] >> 8);
            hist[bin]++;
        }

        // Scan from highest bin downward with lookahead
        std::make_unsigned_t<T> scoreThreshold = 1;
        int32_t cumulative                     = 0;
        for (int32_t bin = histBins - 1; bin > 0; bin--)
        {
            cumulative += hist[bin];
            int32_t nextVal = (bin > 0) ? hist[bin - 1] : 0;
            if (cumulative + nextVal >= maxFeat)
            {
                if constexpr (sizeof(T) == 1)
                    scoreThreshold = static_cast<std::make_unsigned_t<T>>(bin);
                else
                {
                    // For 16-bit, bin 0 contains valid corners with scores 1-255.
                    // If we don't yet have enough features without bin 0, include it.
                    if (bin == 1 && cumulative < maxFeat)
                        scoreThreshold = 1;
                    else
                        scoreThreshold = static_cast<std::make_unsigned_t<T>>(bin << 8);
                }
                break;
            }
        }

        // Phase 3: Filter by threshold
        int32_t numLoc = 0;
        for (int32_t i = 0; i < packedCount && numLoc < maxFeat; i++)
        {
            if (tempScores[i] >= scoreThreshold)
            {
                loc_h[numLoc] = tempLocs[i];
                numLoc++;
            }
        }
        *numLoc_h = numLoc;
    }
    else
    {
        // Non-TopK: Original tile-based packing with inline NMS
        std::vector<std::make_unsigned_t<T>> scorePad((width + 2) * (height + 2), 0);
        int32_t scorePadLinePitch = width + 2;
        for (int32_t j = 0; j < height; j++)
        {
            std::memcpy(&scorePad[(j + 1) * scorePadLinePitch + 1], &score[j * width],
                        width * sizeof(std::make_unsigned_t<T>));
        }

        std::vector<std::make_unsigned_t<T>> scoreNms(tileX * tileY);
        int32_t numLoc{0};
        for (int32_t y = 0; y < niterY; y++)
        {
            for (int32_t x = 0; x < niterX; x++)
            {
                TileInfo<std::make_unsigned_t<T>> nmsIn;
                nmsIn.lp    = scorePadLinePitch;
                nmsIn.tp    = &scorePad[y * tileY * nmsIn.lp + x * tileX];
                nmsIn.tileX = (x == niterX - 1) ? lastTileX : tileX;
                nmsIn.tileY = (y == niterY - 1) ? lastTileY : tileY;

                nmsPackTileRef(isNmsOn, &nmsIn, {x * tileX, y * tileY}, maxLocNum, &loc_h[numLoc], scoreNms.data(),
                               numLoc);
            }
        }
        *numLoc_h = numLoc;
    }
}

} // namespace

int32_t generateFastCornerDetectorInput(NVCVTensorHandle tensorHandle, char const *fileAbsPathName)
{
    NVCVDataType dataType;
    nvcvTensorGetDataType(tensorHandle, &dataType);

    int32_t (*funcPtr)(NVCVTensorHandle, char const *) = nullptr;

    switch (dataType)
    {
    case NVCV_DATA_TYPE_U8:
        funcPtr = generateFastCornerDetectorInputImpl<uint8_t>;
        break;
    case NVCV_DATA_TYPE_S8:
        funcPtr = generateFastCornerDetectorInputImpl<int8_t>;
        break;
    case NVCV_DATA_TYPE_U16:
        funcPtr = generateFastCornerDetectorInputImpl<uint16_t>;
        break;
    case NVCV_DATA_TYPE_S16:
        funcPtr = generateFastCornerDetectorInputImpl<int16_t>;
        break;
    }
    return funcPtr(tensorHandle, fileAbsPathName);
}

void fastCornerDetectorRef(NVCVTensorHandle inTensorHandle, PvaFastCornerDetectorParams const *params,
                           int32_t intensityThreshold, const NVCVBorderType borderMode, int32_t borderValue,
                           NVCVTensorHandle locHandle, NVCVTensorHandle numLocHandle)
{
    NVCVDataType dataType;
    nvcvTensorGetDataType(inTensorHandle, &dataType);

    void (*funcPtr)(NVCVTensorHandle, PvaFastCornerDetectorParams const *, int32_t, const NVCVBorderType, int32_t,
                    NVCVTensorHandle, NVCVTensorHandle) = nullptr;

    switch (dataType)
    {
    case NVCV_DATA_TYPE_U16:
        funcPtr = fastCornerDetectorRefImpl<uint16_t>;
        break;
    case NVCV_DATA_TYPE_S16:
        funcPtr = fastCornerDetectorRefImpl<int16_t>;
        break;
    case NVCV_DATA_TYPE_U8:
        funcPtr = fastCornerDetectorRefImpl<uint8_t>;
        break;
    case NVCV_DATA_TYPE_S8:
        funcPtr = fastCornerDetectorRefImpl<int8_t>;
        break;
    }

    funcPtr(inTensorHandle, params, intensityThreshold, borderMode, borderValue, locHandle, numLocHandle);
}

/* Since the location output is always 2F32, which is irrespective to the image's data type, convert them
 * into the int32_t for comparison.
 */
int32_t checkFastCornerDetectorOutput(NVCVTensorHandle refHandle, NVCVTensorHandle refNumLocHandle,
                                      NVCVTensorHandle pvaLocHandle, NVCVTensorHandle pvaNumLocHandle)
{
    NVCVTensorData refNumLocData;
    nvcvTensorExportData(refNumLocHandle, &refNumLocData);
    int32_t *refNumLoc_h = nullptr;
    CupvaMemGetHostPointer((void **)&refNumLoc_h, (void *)refNumLocData.buffer.strided.basePtr);
    const int32_t refNumLoc = *refNumLoc_h;

    NVCVTensorData pvaNumLocData;
    nvcvTensorExportData(pvaNumLocHandle, &pvaNumLocData);
    int32_t *pvaNumLoc_h = nullptr;
    CupvaMemGetHostPointer((void **)&pvaNumLoc_h, (void *)pvaNumLocData.buffer.strided.basePtr);
    const int32_t pvaNumLoc = *pvaNumLoc_h;

    if (pvaNumLoc != refNumLoc)
    {
        printf("Mismatch in feature count pva = %d, ref = %d\n", pvaNumLoc, refNumLoc);

        return 1;
    }

    NVCVTensorData refTensorData;
    nvcvTensorExportData(refHandle, &refTensorData);
    Point2f *refLoc_h = nullptr;
    CupvaMemGetHostPointer((void **)&refLoc_h, (void *)refTensorData.buffer.strided.basePtr);

    NVCVTensorData pvaLocData;
    nvcvTensorExportData(pvaLocHandle, &pvaLocData);
    Point2f *pvaLoc_h = nullptr;
    CupvaMemGetHostPointer((void **)&pvaLoc_h, (void *)pvaLocData.buffer.strided.basePtr);

    int32_t errCnt = 0;
    for (int32_t i = 0; i < refNumLoc; i++)
    {
        int32_t refLocX = static_cast<int32_t>(refLoc_h[i].x);
        int32_t refLocY = static_cast<int32_t>(refLoc_h[i].y);
        int32_t pvaLocX = static_cast<int32_t>(pvaLoc_h[i].x);
        int32_t pvaLocY = static_cast<int32_t>(pvaLoc_h[i].y);
        if (refLocX != pvaLocX || refLocY != pvaLocY)
        {
            printf("Mismatch in feature location for pos %d PVA = (%d,%d), ref = (%d,%d)\n", i, pvaLocX, pvaLocY,
                   refLocX, refLocY);
            errCnt++;
        }
        if (errCnt >= 10)
        {
            break;
        }
    }
    //printf("refNumLoc: %d, pvaNumLoc: %d\n", refNumLoc, pvaNumLoc);

    if (errCnt == 0)
    {
        return 0;
    }
    else
    {
        std::cout << "compare fails!\n";
        return 1;
    }
}