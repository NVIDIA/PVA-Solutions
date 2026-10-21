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
 * @file bilateral_filter_ref.h
 *
 * @brief Defines functions for reference model verification
 */

#ifndef BILATERAL_FILTER_REF_H
#define BILATERAL_FILTER_REF_H

#include <nvcv/BorderType.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

int32_t loadBilateralFilterInput(NVCVTensorHandle tensorHandle, const char *directory, const char *filename,
                                 const int32_t size);
int32_t saveBilateralFilterOutput(NVCVTensorHandle tensorHandle, const char *directory, const char *filename,
                                  const int32_t size);
void generateRandomBilateralFilterInput(NVCVTensorHandle tensorHandle);
void bilateralFilterRef(NVCVTensorHandle inTensorHandle, NVCVTensorHandle fixedPointRefTensorHandle,
                        NVCVTensorHandle floatRefTensorHandle, const float sigmaRange, const float sigmaSpace,
                        const int32_t kernelSize, const NVCVBorderType borderMode, const int32_t borderValue);
int32_t checkBilateralFilterOutput(NVCVTensorHandle fixedPointRefTensorHandle, NVCVTensorHandle floatRefTensorHandle,
                                   NVCVTensorHandle outTensorHandle);

#ifdef __cplusplus
}
#endif

#endif /* BILATERAL_FILTER_REF_H */
