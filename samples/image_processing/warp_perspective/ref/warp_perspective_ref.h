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
 * @file warp_perspective_ref.h
 *
 * @brief Defines functions for reference model verification
 */

#ifndef WARP_PERSPECTIVE_REF_H
#define WARP_PERSPECTIVE_REF_H

#include <OpWarpPerspective.h>
#include <PvaOperatorTypes.h>
#include <nvcv/BorderType.h>
#include <nvcv/Image.h>
#include <nvcv/Status.h>

#ifdef __cplusplus
extern "C" {
#endif

void generateRandomWarpPerspectiveInput(NVCVImageHandle imageHandle, bool isRandom);

int32_t checkWarpPerspectiveOutput(NVCVImageHandle refImageHandle, NVCVImageHandle pvaImageHandle);

void warpPerspectiveRef(NVCVImageHandle inImageHandle, NVCVImageHandle outImageHandle, PVAInterpolationType interpType,
                        PVAWarpTransformationType transformType);

void setupWarpMatrix(WarpMatrixType invWarpMatrix, const int32_t inWidth, const int32_t inHeight,
                     const int32_t outWidth, const int32_t outHeight);

NVCVStatus pvaWarpPerspectiveValidate(WarpMatrixType invWarpMatrix, const int32_t inWidth, const int32_t inHeight,
                                      const int32_t inLinePitch, const int32_t outWidth, const int32_t outHeight,
                                      const PVAWarpTransformationType transformType);

#ifdef __cplusplus
}
#endif

#endif /* WARP_PERSPECTIVE_REF_H */
