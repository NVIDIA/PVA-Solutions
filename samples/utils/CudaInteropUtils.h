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

#ifndef CUDA_INTEROP_UTILS_H
#define CUDA_INTEROP_UTILS_H

#include <cupva_host.h>
#include <cupva_host_cuda.h>
#include <nvcv/Image.h>
#include <nvcv/Tensor.h>
#include <stdio.h>

/**
 * @brief Wraps a CUDA buffer into an NVCVTensorHandle, managing CUDA import and cleanup.
 *
 * Imports a CUDA buffer to PVA memory, populates an NVCVTensorData structure,
 * and then wraps this data using nvcvTensorWrapDataConstruct. A custom cleanup
 * function is registered to free the imported cuPVA device pointer and associated structures when
 * the NVCVTensorHandle's reference count drops to zero.
 *
 * @note This function is only required for PVA SDK versions earlier than 2.6.0 and PVA Driver versions earlier than 2006.
 * @note For PVA SDK 2.6.0 and PVA Driver 2006 or newer, tensors allocated with the default allocator can be passed directly to PVA Operator APIs without wrapping.
 * @note The host pointer @a cudaBuf is attached to the tensor with nvcvTensorSetUserPointer so
 * that host-side code can recover it; see PvaGetTensorHostPointer in HostPointer.h.
 *
 * @param tensorRequirements Pointer to the NVCVTensorRequirements for the tensor.
 * @param cudaBuf The CUDA buffer to be imported and wrapped.
 * @param outTensorHandle Pointer to an NVCVTensorHandle where the handle of the
 *                        created tensor will be stored.
 * @return NVCV_SUCCESS on success, or an NVCV error code on failure.
 */
NVCVStatus nvcvTensorWrapCudaBuf(const NVCVTensorRequirements *tensorRequirements, void *cudaBuf,
                                 NVCVTensorHandle *outTensorHandle);

/**
 * @brief Wraps a CUDA buffer into an NVCVImageHandle, managing CUDA import and cleanup.
 *
 * Imports a CUDA buffer to PVA memory, populates an NVCVImageData structure,
 * and then wraps this data using nvcvImageWrapDataConstruct. A custom cleanup
 * function is registered to free the imported cuPVA device pointer and associated structures when
 * the NVCVImageHandle's reference count drops to zero.
 *
 * @note This function is only required for PVA SDK versions earlier than 2.6.0 and PVA Driver versions earlier than 2006.
 * @note For PVA SDK 2.6.0 and PVA Driver 2006 or newer, images allocated with the default allocator can be passed directly to PVA Operator APIs without wrapping.
 * @note The host pointer @a cudaBuf is attached to the image with nvcvImageSetUserPointer so
 * that host-side code can recover it; see PvaGetImageHostPointer in HostPointer.h.
 *
 * @param imageRequirements Pointer to the NVCVImageRequirements for the image.
 * @param cudaBuf The CUDA buffer to be imported and wrapped.
 * @param outImageHandle Pointer to an NVCVImageHandle where the handle of the
 *                        created image will be stored.
 * @return NVCV_SUCCESS on success, or an NVCV error code on failure.
 */
NVCVStatus nvcvImageWrapCudaBuf(const NVCVImageRequirements *imageRequirements, void *cudaBuf,
                                NVCVImageHandle *outImageHandle);

#endif // CUDA_INTEROP_UTILS_H