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

/*
 * @file contour_tracing_ref.h
 *
 * @brief Defines functions for reference model verification
 */

#ifndef CONTOUR_TRACING_REF_H
#define CONTOUR_TRACING_REF_H

// clang-format off
// <stdint.h> must precede <nvcv/Rect.h>: Rect.h uses int32_t without including stdint itself.
#include <stdint.h>
#include <nvcv/Rect.h>
#include <nvcv/Tensor.h>
// clang-format on

#ifdef __cplusplus
extern "C" {
#endif

void generateContourTracingInput(NVCVTensorHandle inTensorHandle, const NVCVRectI *rectArray, int32_t numRects);

int32_t checkContourTracingOutput(NVCVTensorHandle refOutTensor, NVCVTensorHandle pvaOutTensor,
                                  NVCVTensorHandle refLenTensor, NVCVTensorHandle pvaLenTensor, int32_t numRects);

void contourTracingRef(NVCVTensorHandle inTensorHandle, const NVCVRectI *rectArray, int32_t numRects,
                       NVCVTensorHandle outTensorHandle, NVCVTensorHandle lenTensorHandle);

void contourTracingMemcpyToTensor(NVCVTensorHandle tensorHandle, const void *src, int32_t numBytes);

#ifdef __cplusplus
}
#endif

#endif /* CONTOUR_TRACING_REF_H */