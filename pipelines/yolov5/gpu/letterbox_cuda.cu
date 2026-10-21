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

#include "letterbox_cuda.cuh"

#include <cmath>

__global__ static void letterboxRgbU8Kernel(const uint8_t *__restrict__ src, int64_t srcRowStride, int srcW, int srcH,
                                            uint8_t *__restrict__ dst, int64_t dstRowStride, int dstW, int dstH,
                                            int scaledW, int scaledH, int padX, int padY, uint8_t padValue)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstW || y >= dstH)
        return;

    uint8_t *out = dst + static_cast<int64_t>(y) * dstRowStride + x * 3;
    const int rx = x - padX;
    const int ry = y - padY;
    if (rx < 0 || ry < 0 || rx >= scaledW || ry >= scaledH)
    {
        out[0] = padValue;
        out[1] = padValue;
        out[2] = padValue;
        return;
    }

    // Half-pixel bilinear mapping, matching the conventional resize contract.
    const float sx = (static_cast<float>(rx) + 0.5f) * srcW / scaledW - 0.5f;
    const float sy = (static_cast<float>(ry) + 0.5f) * srcH / scaledH - 0.5f;
    const int x0   = max(0, min(srcW - 1, static_cast<int>(floorf(sx))));
    const int y0   = max(0, min(srcH - 1, static_cast<int>(floorf(sy))));
    const int x1   = min(srcW - 1, x0 + 1);
    const int y1   = min(srcH - 1, y0 + 1);
    const float wx = fminf(1.0f, fmaxf(0.0f, sx - floorf(sx)));
    const float wy = fminf(1.0f, fmaxf(0.0f, sy - floorf(sy)));

    const uint8_t *p00 = src + static_cast<int64_t>(y0) * srcRowStride + x0 * 3;
    const uint8_t *p01 = src + static_cast<int64_t>(y0) * srcRowStride + x1 * 3;
    const uint8_t *p10 = src + static_cast<int64_t>(y1) * srcRowStride + x0 * 3;
    const uint8_t *p11 = src + static_cast<int64_t>(y1) * srcRowStride + x1 * 3;
    for (int c = 0; c < 3; ++c)
    {
        const float top = p00[c] + wx * (p01[c] - p00[c]);
        const float bot = p10[c] + wx * (p11[c] - p10[c]);
        out[c]          = static_cast<uint8_t>(fminf(255.0f, fmaxf(0.0f, roundf(top + wy * (bot - top)))));
    }
}

cudaError_t launchLetterboxRgbU8(const uint8_t *src, int64_t srcRowStride, int srcW, int srcH, uint8_t *dst,
                                 int64_t dstRowStride, int dstW, int dstH, int scaledW, int scaledH, int padX, int padY,
                                 uint8_t padValue, cudaStream_t stream)
{
    const dim3 block(32, 8);
    const dim3 grid((dstW + block.x - 1) / block.x, (dstH + block.y - 1) / block.y);
    letterboxRgbU8Kernel<<<grid, block, 0, stream>>>(src, srcRowStride, srcW, srcH, dst, dstRowStride, dstW, dstH,
                                                     scaledW, scaledH, padX, padY, padValue);
    return cudaGetLastError();
}
