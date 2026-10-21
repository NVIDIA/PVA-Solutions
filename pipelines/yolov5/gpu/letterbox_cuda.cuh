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

#ifndef YOLOV5_COMMON_LETTERBOX_CUDA_CUH
#define YOLOV5_COMMON_LETTERBOX_CUDA_CUH

#include <cuda_runtime.h>

#include <cstdint>

cudaError_t launchLetterboxRgbU8(const uint8_t *src, int64_t srcRowStride, int srcW, int srcH, uint8_t *dst,
                                 int64_t dstRowStride, int dstW, int dstH, int scaledW, int scaledH, int padX, int padY,
                                 uint8_t padValue, cudaStream_t stream);

#endif
