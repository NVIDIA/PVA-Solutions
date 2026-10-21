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
 * @file cornersubpix_ref.h
 *
 * @brief Defines functions for reference model verification
 */

#ifndef CORNERSUBPIX_REF_H
#define CORNERSUBPIX_REF_H

#include <nvcv/Size.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

int32_t loadCornerSubPixCornerInput(NVCVTensorHandle cornerTensorHandle, const char *directory, const char *filename,
                                    const int32_t count);

int32_t loadCornerSubPixImageInput(NVCVTensorHandle imageTensorHandle, const char *directory, const char *filename,
                                   const int64_t size);

int32_t checkCornerSubPixOutput(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle,
                                const int32_t count);

void cornerSubPixFixedPointRef(NVCVTensorHandle cornerTensorHandle, NVCVTensorHandle imageTensorHandle,
                               const NVCVSize2D src, const NVCVSize2D win, const NVCVSize2D zeroZone,
                               const int32_t maxIters, const float epsilon, const int32_t count);

#ifdef __cplusplus
}
#endif

#endif /* CORNERSUBPIX_REF_H */
