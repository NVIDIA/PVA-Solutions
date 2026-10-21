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

#ifndef PREPROCESS_CUDA_CUH
#define PREPROCESS_CUDA_CUH

#include <cuda_runtime.h>

#include <cstdint>

extern "C" {

/**
 * @brief Convert RGB NHWC uint8 [H,W,3] to NCHW float32 [3,H,W] with 1/255 normalization.
 *
 * @param src     Device pointer to RGB uint8 image (H*W*3 bytes, compact)
 * @param dst     Device pointer to NCHW float32 output (3*H*W floats)
 * @param height  Image height
 * @param width   Image width
 * @param stream  CUDA stream
 */
cudaError_t launchRgbNhwcToNchw(const uint8_t *src, float *dst, int height, int width, cudaStream_t stream);
}

#endif // PREPROCESS_CUDA_CUH
