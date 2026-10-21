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

/*
 * @file image_stats_ref.h
 *
 * @brief Defines functions for image statistics reference model verification
 */

#ifndef IMAGE_STATS_REF_H
#define IMAGE_STATS_REF_H

#include <ImageIO.h>
#include <OpImageStats.h>
#include <nvcv/Image.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

void generateImageStatsRandomImageInput(NVCVImageHandle imageHandle);
void generateImageStatsFixedImageInput(NVCVImageHandle imageHandle, const char *assetDir);
void generateImageStatsRandomBinaryMask(NVCVImageHandle maskHandle, float maskRatio);

int32_t checkImageStatsOutput(NVCVImageHandle imageHandle, NVCVTensorHandle outTensorHandle,
                              NVCVTensorHandle refTensorHandle);

void imageStatsRef(NVCVImageHandle imageHandle, NVCVImageHandle maskHandle, NVCVTensorHandle refTensorHandle,
                   uint32_t flags);

#ifdef __cplusplus
}
#endif

#endif /* IMAGE_STATS_REF_H */
