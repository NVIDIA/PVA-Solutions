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
 * @file gaussian_filter_ref.h
 *
 * @brief Defines functions for reference model verification
 */

#ifndef CONV2D_REF_H
#define CONV2D_REF_H

#include <nvcv/BorderType.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

void generateRandomConv2dInput(NVCVTensorHandle tensorHandle);

int32_t checkConv2dOutput(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle);

void conv2dRefWithKernel(NVCVTensorHandle inTensorHandle, NVCVTensorHandle outTensorHandle,
                         NVCVTensorHandle kernelTensorHandle, const NVCVBorderType borderMode, int32_t borderValue);

NVCVTensorHandle createConv2dKernel(NVCVAllocatorHandle allocatorHandle, int32_t kernelSize, NVCVDataType dataType);

// Functions for file I/O
int32_t loadConv2dInput(NVCVTensorHandle tensorHandle, const char *directory, const char *filename, const int64_t size);
int32_t saveConv2dOutput(NVCVTensorHandle tensorHandle, const char *directory, const char *filename,
                         const int64_t size);

#ifdef __cplusplus
}
#endif

#endif /* CONV2D_REF_H */
