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

#include "preprocess_cuda.cuh"

/**
 * Convert RGB uint8 NHWC to NCHW float32 with 1/255 normalization.
 * Each thread handles one pixel (all 3 channels).
 */
__global__ void rgbNhwcToNchwKernel(const uint8_t *__restrict__ src, float *__restrict__ dst, int height, int width)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int hwSize = height * width;
    int srcIdx = (y * width + x) * 3;
    int pixIdx = y * width + x;

    constexpr float scale    = 1.0f / 255.0f;
    dst[0 * hwSize + pixIdx] = static_cast<float>(src[srcIdx + 0]) * scale; // R
    dst[1 * hwSize + pixIdx] = static_cast<float>(src[srcIdx + 1]) * scale; // G
    dst[2 * hwSize + pixIdx] = static_cast<float>(src[srcIdx + 2]) * scale; // B
}

extern "C" cudaError_t launchRgbNhwcToNchw(const uint8_t *src, float *dst, int height, int width, cudaStream_t stream)
{
    dim3 block(32, 8);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    rgbNhwcToNchwKernel<<<grid, block, 0, stream>>>(src, dst, height, width);
    return cudaGetLastError();
}
