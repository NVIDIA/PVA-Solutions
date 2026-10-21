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

#include "nhwc_to_nchw_letterbox_normalize_ref.h"

#include <cupva_host.h>
#include <nvcv/Tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

void generateInputImpl(NVCVTensorHandle imgIn)
{
    NVCVTensorData data;
    nvcvTensorExportData(imgIn, &data);

    uint8_t *ptr = nullptr;
    CupvaMemGetHostPointer((void **)&ptr, (void *)data.buffer.strided.basePtr);

    const int h        = static_cast<int>(data.shape[0]);
    const int w        = static_cast<int>(data.shape[1]);
    const int c        = static_cast<int>(data.shape[2]);
    const int rowPitch = static_cast<int>(data.buffer.strided.strides[0]);

    for (int row = 0; row < h; row++)
        for (int col = 0; col < w; col++)
            for (int ch = 0; ch < c; ch++)
                ptr[row * rowPitch + col * c + ch] = static_cast<uint8_t>((row + col + ch * 37) % 256);
}

/*
 * Reference: HWC U8 -> CHW F32 with letterbox padding.
 *
 * Input shape:  [imgH, imgW, 3]  (HWC)
 * Output shape: [3, outH, outW]  (CHW)
 *
 * For each output position [ch, outRow, outCol]:
 *   srcRow = outRow - padTop
 *   srcCol = outCol - padLeft
 *   if in image region: out = (float)src[srcRow][srcCol][ch] * scale
 *   else:               out = padValue
 */
void refImpl(NVCVTensorHandle imgIn, NVCVTensorHandle floatOut, float scale, float padValue)
{
    NVCVTensorData inData;
    nvcvTensorExportData(imgIn, &inData);

    uint8_t *src = nullptr;
    CupvaMemGetHostPointer((void **)&src, (void *)inData.buffer.strided.basePtr);

    NVCVTensorData outData;
    nvcvTensorExportData(floatOut, &outData);

    float *dst = nullptr;
    CupvaMemGetHostPointer((void **)&dst, (void *)outData.buffer.strided.basePtr);

    const int imgH       = static_cast<int>(inData.shape[0]);
    const int imgW       = static_cast<int>(inData.shape[1]);
    const int channels   = static_cast<int>(inData.shape[2]);
    const int outC       = static_cast<int>(outData.shape[0]);
    const int outH       = static_cast<int>(outData.shape[1]);
    const int outW       = static_cast<int>(outData.shape[2]);
    const int padTop     = (outH - imgH) / 2;
    const int padLeft    = (outW - imgW) / 2;
    const int inRowPitch = static_cast<int>(inData.buffer.strided.strides[0]);
    const int planeSize  = outH * outW;

    (void)outC; /* always 3 */

    for (int ch = 0; ch < channels; ch++)
    {
        for (int outRow = 0; outRow < outH; outRow++)
        {
            const int srcRow = outRow - padTop;
            for (int outCol = 0; outCol < outW; outCol++)
            {
                const int srcCol = outCol - padLeft;
                float val;

                if (srcRow >= 0 && srcRow < imgH && srcCol >= 0 && srcCol < imgW)
                {
                    val = static_cast<float>(src[srcRow * inRowPitch + srcCol * channels + ch]) * scale;
                }
                else
                {
                    val = padValue;
                }

                dst[ch * planeSize + outRow * outW + outCol] = val;
            }
        }
    }
}

int32_t checkOutputImpl(NVCVTensorHandle refOut, NVCVTensorHandle testOut)
{
    NVCVTensorData refData;
    nvcvTensorExportData(refOut, &refData);

    float *ref = nullptr;
    CupvaMemGetHostPointer((void **)&ref, (void *)refData.buffer.strided.basePtr);

    NVCVTensorData testData;
    nvcvTensorExportData(testOut, &testData);

    float *test = nullptr;
    CupvaMemGetHostPointer((void **)&test, (void *)testData.buffer.strided.basePtr);

    /* Output is CHW: shape[0]=C, shape[1]=H, shape[2]=W */
    const int C = static_cast<int>(refData.shape[0]);
    const int H = static_cast<int>(refData.shape[1]);
    const int W = static_cast<int>(refData.shape[2]);

    const float tol = 1e-5f;
    int32_t errors  = 0;

    for (int c = 0; c < C; c++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
            {
                int idx = c * H * W + h * W + w;
                if (std::fabs(ref[idx] - test[idx]) > tol)
                {
                    printf("Mismatch [c=%d h=%d w=%d]: ref=%.6f test=%.6f\n", c, h, w, ref[idx], test[idx]);
                    errors++;
                    if (errors >= 20)
                    {
                        printf("  (too many mismatches, stopping)\n");
                        return errors;
                    }
                }
            }

    return errors;
}

} // namespace

extern "C" {

void generateNhwcToNchwLetterboxNormalizeInput(NVCVTensorHandle imgIn)
{
    generateInputImpl(imgIn);
}

void nhwcToNchwLbNormRef(NVCVTensorHandle imgIn, NVCVTensorHandle floatOut, float scale, float padValue)
{
    refImpl(imgIn, floatOut, scale, padValue);
}

int32_t checkNhwcToNchwLetterboxNormalizeOutput(NVCVTensorHandle refOut, NVCVTensorHandle testOut)
{
    return checkOutputImpl(refOut, testOut);
}

} // extern "C"
