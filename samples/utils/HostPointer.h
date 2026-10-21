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

#ifndef HOST_POINTER_H
#define HOST_POINTER_H

#include <nvcv/Image.h>
#include <nvcv/Tensor.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Returns a CPU-accessible pointer to the start of a tensor's data.
 *
 * Uses the host pointer attached to the handle with nvcvTensorSetUserPointer when there is
 * one, and otherwise falls back to CupvaMemGetHostPointer.
 *
 * A host pointer must be attached for memory that cuPVA cannot map on its own: cuPVA 2.10
 * onwards does not map CUDA buffers imported with CupvaCudaMemImport, and directs callers to
 * the host pointer returned by the CUDA allocation instead. nvcvTensorWrapCudaBuf attaches it.
 * Tensors backed by the PVA allocator or by NvSciBuf need nothing attached; the fallback
 * covers them.
 *
 * @param handle The tensor to map.
 * @return A CPU-accessible pointer to the tensor's base address, or NULL if it cannot be mapped.
 */
void *PvaGetTensorHostPointer(NVCVTensorHandle handle);

/**
 * @brief Returns a CPU-accessible pointer to the start of an image plane.
 *
 * Behaves like PvaGetTensorHostPointer. An attached host pointer refers to the start of the
 * whole allocation, so the plane offset is applied to it.
 *
 * @param handle The image to map.
 * @param plane The plane to map, in the range [0, numPlanes).
 * @return A CPU-accessible pointer to the plane's base address, or NULL if it cannot be mapped.
 */
void *PvaGetImageHostPointer(NVCVImageHandle handle, int32_t plane);

#ifdef __cplusplus
}
#endif

#endif // HOST_POINTER_H
