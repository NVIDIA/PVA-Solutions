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
 * @file remap_ref.h
 *
 * @brief Defines functions for reference model verification
 */

#ifndef REMAP_REF_H
#define REMAP_REF_H

#include <PvaOperatorTypes.h>
#include <nvcv/BorderType.h>
#include <nvcv/Image.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

void generateRandomRemapInput(NVCVImageHandle imageHandle, bool isRandom);

void generateRandomRemapWarpMap(NVCVTensorHandle tensorHandle, int32_t inWidth, int32_t inHeight, bool isRandom);

int32_t checkRemapOutput(NVCVImageHandle refImageHandle, NVCVImageHandle pvaImageHandle);

void remapRef(NVCVImageHandle inImageHandle, NVCVImageHandle outImageHandle, NVCVTensorHandle warpMapHandle,
              PVAInterpolationType interpType, NVCVBorderType borderMode);

#ifdef __cplusplus
}
#endif

#endif /* REMAP_REF_H */
