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

/*
 * @file image_resize_ref.h
 *
 * @brief Defines functions for reference model verification
 */

#ifndef IMAGE_RESIZE_REF_H
#define IMAGE_RESIZE_REF_H

#include <OpImageResize.h>
#include <PvaOperatorTypes.h>
#include <nvcv/Image.h>
#include <nvcv/Status.h>

#ifdef __cplusplus
extern "C" {
#endif

void generateRandomImageInput(NVCVImageHandle imageHandle);

int32_t checkImageOutput(NVCVImageHandle refImageHandle, NVCVImageHandle pvaImageHandle, uint8_t threshold);

void imageResizeFixedPointRef(NVCVImageHandle inImageHandle, NVCVImageHandle outImageHandle,
                              PVAInterpolationType resizeType);

#ifdef __cplusplus
}
#endif

#endif /* IMAGE_RESIZE_REF_H */
