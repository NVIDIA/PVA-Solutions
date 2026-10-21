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
 * @file box_filter_ref.h
 *
 * @brief Defines functions for reference model verification
 */

#ifndef BOX_FILTER_REF_H
#define BOX_FILTER_REF_H

#include <nvcv/BorderType.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

void generateRandomBoxFilterInput(NVCVTensorHandle tensorHandle);

int32_t checkBoxFilterOutput(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle);

void boxFilterRef(NVCVTensorHandle inTensorHandle, NVCVTensorHandle outTensorHandle, int32_t kernelSize,
                  const NVCVBorderType borderMode, int32_t borderValue);

// Functions for file I/O
int32_t loadBoxFilterInput(NVCVTensorHandle tensorHandle, const char *directory, const char *filename,
                           const int64_t size);
int32_t saveBoxFilterOutput(NVCVTensorHandle tensorHandle, const char *directory, const char *filename,
                            const int64_t size);

#ifdef __cplusplus
}
#endif

#endif /* BOX_FILTER_REF_H */