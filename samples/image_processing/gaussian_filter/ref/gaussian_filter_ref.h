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
 * @file gaussian_filter_ref.h
 *
 * @brief Defines functions for reference model verification
 */

#ifndef GAUSSIAN_FILTER_REF_H
#define GAUSSIAN_FILTER_REF_H

#include <nvcv/BorderType.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

void generateRandomGaussianFilterInput(NVCVTensorHandle tensorHandle);

int32_t checkGaussianFilterOutput(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle);

void gaussianFilterRef(NVCVTensorHandle inTensorHandle, NVCVTensorHandle outTensorHandle, float sigmaX, float sigmaY,
                       int32_t kernelSize, const NVCVBorderType borderMode, int32_t borderValue);

// Functions for file I/O
int32_t loadGaussianFilterInput(NVCVTensorHandle tensorHandle, const char *directory, const char *filename,
                                const int64_t size);
int32_t saveGaussianFilterOutput(NVCVTensorHandle tensorHandle, const char *directory, const char *filename,
                                 const int64_t size);

#ifdef __cplusplus
}
#endif

#endif /* GAUSSIAN_FILTER_REF_H */
