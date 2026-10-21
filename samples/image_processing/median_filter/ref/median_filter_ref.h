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
 * @file median_filter_ref.h
 *
 * @brief Defines functions for reference model verification
 */

#ifndef MEDIAN_FILTER_REF_H
#define MEDIAN_FILTER_REF_H

#include <OpMedianFilter.h>
#include <nvcv/Image.h>
#include <nvcv/Status.h>

#ifdef __cplusplus
extern "C" {
#endif

void generateMedianFilterRandomImageInput(NVCVImageHandle imageHandle);

void generateMedianFilterBoxFilterKernel(NVCVTensorHandle kernelHandle, int32_t kernelHeight, int32_t kernelWidth);

int32_t checkMedianFilterImageOutput(NVCVImageHandle refImageHandle, NVCVImageHandle pvaImageHandle);

void medianFilterRef(NVCVImageHandle inImageHandle, NVCVImageHandle outImageHandle, int32_t kernelHeight,
                     int32_t kernelWidth, NVCVTensorHandle kernel, NVCVBorderType borderType, int32_t borderValue);

#ifdef __cplusplus
}
#endif

#endif /* MEDIAN_FILTER_REF_H */
