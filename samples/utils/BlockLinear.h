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

#ifndef BLOCKLINEAR_H
#define BLOCKLINEAR_H

#include <cupva_host_scheduling.h>
#include <nvcv/Image.h>

#ifdef __cplusplus
extern "C" {
#endif

// Convert a NV12 image in pitch linear layout into block linear layout.
void plToBl(NVCVImageHandle inImageHandle, NVCVImageHandle outImageHandle);

// Convert a NV12 image in block linear layout into pitch linear layout.
void blToPl(NVCVImageHandle inImageHandle, NVCVImageHandle outImageHandle);

// Allocate a surface to hold a NV12 image in block linear layout.
NVCVStatus allocBlSurfaceNv12(void **in, const int32_t imageWidth, const int32_t imageHeight);

// Construct a NVCVImageData of a NV12 image or single plane with U8 data type in block linear layout.
NVCVImageData nvcvImageDataConstructBL(void *devPtr, int32_t imageWidth, int32_t imageHeight, NVCVImageFormat format);
/*
    Construct a NV12 image or single plane with U8 data type in BL layout. It only reads the width, height and format fields
    of NVCVImageRequirements due to the lack of support of underlying cupva_utils::AllocSurface. The starting address of the BL
    surface is output in devPtr,if it does not point to NULL.
*/
NVCVStatus nvcvImageConstructBL(const NVCVImageRequirements *reqs, NVCVImageHandle *handle, void **devPtr);

// Free the surface memory within the NVCVImageData.
void nvcvImageConstructBLCleanup(void *ctx, const NVCVImageData *data);

#ifdef __cplusplus
}
#endif

#endif // BLOCKLINEAR_H