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
 * @file convert_image_format_ref.h
 *
 * @brief Defines functions for reference model verification
 */

#ifndef CONVERT_IMAGE_FORMAT_REF_H
#define CONVERT_IMAGE_FORMAT_REF_H

#include <OpConvertImageFormat.h>
#include <nvcv/Image.h>

#ifdef __cplusplus
extern "C" {
#endif

void generateConvertImageFormatInput(NVCVImageHandle imageHandle);

int32_t checkConvertImageFormatOutput(NVCVImageHandle refImageHandle, NVCVImageHandle pvaImageHandle);

void convertImageFormatRef(NVCVImageHandle inImageHandle, NVCVImageHandle outImageHandle);

#ifdef __cplusplus
}
#endif

#endif /* CONVERT_IMAGE_FORMAT_REF_H */
