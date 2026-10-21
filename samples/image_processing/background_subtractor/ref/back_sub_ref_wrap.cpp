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

#include <back_sub_ref.hpp>

#include <ImageIO.h>
#include <nvcv/Image.h>

#include <cassert>
#include <cstdlib>

constexpr bool DEBUG_MISMATCHES = false;

struct BGSubRefHandle
{
    int gmmModelBitWidth;
    void *ptr;
};

extern "C" {

void BGSubRefCreate(BGSubRefHandle **handle, int w, int h)
{
    int gmmModelBitWidth = 32;
    *handle              = new BGSubRefHandle;
    (*handle)->ptr       = new BGSubRef<uint8_t, float>(w, h);
    // fp16 version of ref code can be added here, using an external fp16 library like below
    //    (*handle)->ptr = new BGSubRef <uint8_t, float16_t> (w, h);
    (*handle)->gmmModelBitWidth = gmmModelBitWidth;
}

void BGSubRefSubmit(BGSubRefHandle *handle, uint8_t *imgIn, uint8_t *maskOut, uint8_t *bgImgOut, float varThreshold,
                    float learningRate, int detectShadow, uint8_t shadowPixelValue)
{
    BackgroundSubtractorRefParams params;
    params.detectShadow     = detectShadow;
    params.learningRate     = learningRate;
    params.shadowPixelValue = shadowPixelValue;
    params.varThreshold     = varThreshold;

    if (handle->gmmModelBitWidth == 32)
    {
        auto *bgSubRef = static_cast<BGSubRef<uint8_t, float> *>(handle->ptr);
        bgSubRef->runGMMS(params, imgIn, maskOut);
        if (bgImgOut != nullptr)
        {
            bgSubRef->getBackgroundImage(bgImgOut);
        }
    }
}

void BGSubRefDestroy(BGSubRefHandle *handle)
{
    if (handle == nullptr)
    {
        return;
    }

    delete static_cast<BGSubRef<uint8_t, float> *>(handle->ptr);
    delete handle;
}

int compareArrays(uint8_t *ref, uint8_t *pva, int w, int h, int tol = 0)
{
    int errCount = 0;
    for (int r = 0; r < h; r++)
    {
        for (int c = 0; c < w; c++)
        {
            int ind        = r * w + c;
            uint8_t refVal = ref[ind];
            uint8_t pvaVal = pva[ind];
            if (std::abs(refVal - pvaVal) > tol)
            {
                if (DEBUG_MISMATCHES)
                {
                    printf("Mismatch at [%d, %d]. Ref = %u, PVA = %u\n", r, c, refVal, pvaVal);
                }
                errCount++;
            }
        }
    }
    return errCount;
}

void generateBackgroundSubtractorInput(uint8_t *imgPtr, const char *assetsDir, int frameNum, int imgWidth,
                                       int imgHeight)
{
    int testImgWidth = 960, testImgHeight = 540;
    assert(imgWidth <= testImgWidth && "requested image width should be <= width of assets images");
    assert(imgHeight <= testImgHeight && "requested image height should be <= height of assets images");

    char filename[32];
    uint8_t *fullImg = new uint8_t[testImgWidth * testImgHeight];
    snprintf(filename, sizeof(filename), "frames_%d.bin", frameNum);
    int32_t err = ReadImageBuffer(filename, assetsDir, fullImg, sizeof(uint8_t) * testImgHeight * testImgWidth);
    assert(err == 0 && "coud not read image");

    // Crop the full image to provided input width and height
    for (int row = 0; row < imgHeight; row++)
    {
        for (int col = 0; col < imgWidth; col++)
        {
            int srcInd     = row * testImgWidth + col;
            int dstInd     = row * imgWidth + col;
            imgPtr[dstInd] = fullImg[srcInd];
        }
    }

    delete[] fullImg;
}
}