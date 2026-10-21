/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DISTANCE_TRANSFORM_REF_H
#define DISTANCE_TRANSFORM_REF_H

#include <nvcv/Size.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

int32_t loadDistanceTransformData(NVCVTensorHandle tensorHandle, const char *directory, const char *filename,
                                  const int32_t tensorSizeInBytes);

int32_t checkDistanceTransformResult(NVCVTensorHandle devTensorHandle, NVCVTensorHandle refTensorHandle,
                                     const int32_t tensorSizeInBytes);

int32_t randomizeDistanceTransformInputData(NVCVTensorHandle inputTensorHandle, int32_t width, int32_t height,
                                            int32_t numDot, uint16_t offPixel, uint16_t labelRange);

int32_t getDistanceTransformRefData(NVCVTensorHandle inputTensorHandle, NVCVTensorHandle distTensorHandle,
                                    NVCVTensorHandle labelTensorHandle, int32_t width, int32_t height,
                                    uint16_t offPixel, uint16_t maxDist);

#ifdef __cplusplus
}
#endif

#endif // DISTANCE_TRANSFORM_REF_H
