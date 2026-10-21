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

#ifndef NVSCI_INTEROP_UTILS_H
#define NVSCI_INTEROP_UTILS_H

#include <nvscibuf.h>
#include <nvscisync.h>
#include <stdio.h>

// NVCV Includes
#include <nvcv/Image.h>
#include <nvcv/Tensor.h>

// CUPVA Includes
#include <cupva_host.h>
#include <cupva_host_nvsci.h>

// Macro to check NvSci API call errors
#define NVSCI_CHECK_ERROR_GOTO(__v, __e, __l)      \
    __e = __v;                                     \
    if (__e != NvSciError_Success)                 \
    {                                              \
        printf("NvSci return error: %d\n", (__e)); \
        goto __l;                                  \
    }

// Function Declarations

/**
 * @brief Wraps an NvSciBufObj into an NVCVTensorHandle, managing PVA import and cleanup.
 *
 * Imports an NvSciBufObj to PVA memory, populates an NVCVTensorData structure,
 * and then wraps this data using nvcvTensorWrapDataConstruct. A custom cleanup
 * function is registered to free the imported cuPVA device pointer and associated structures when
 * the NVCVTensorHandle's reference count drops to zero.
 *
 * @param tensorRequirements Pointer to the NVCVTensorRequirements for the tensor.
 * @param nvsciBufObj The NvSciBufObj to be imported and wrapped.
 * @param outTensorHandle Pointer to an NVCVTensorHandle where the handle of the
 *                        created tensor will be stored.
 * @return NVCV_SUCCESS on success, or an NVCV error code on failure.
 */
NVCVStatus nvcvTensorWrapNvSciBuf(const NVCVTensorRequirements *tensorRequirements, NvSciBufObj nvsciBufObj,
                                  NVCVTensorHandle *outTensorHandle);

/**
 * @brief Wraps an NvSciBufObj into an NVCVImageHandle, managing PVA import and cleanup.
 *
 * Imports an NvSciBufObj to PVA memory, populates an NVCVImageData structure,
 * and then wraps this data using nvcvImageWrapDataConstruct. A custom cleanup
 * function is registered to free the imported cuPVA device pointer and associated structures when
 * the NVCVImageHandle's reference count drops to zero.
 *
 * @param imageRequirements Pointer to the NVCVImageRequirements for the image.
 * @param nvsciBufObj The NvSciBufObj to be imported and wrapped.
 * @param outImageHandle Pointer to an NVCVImageHandle where the handle of the
 *                        created image will be stored.
 * @return NVCV_SUCCESS on success, or an NVCV error code on failure.
 */
NVCVStatus nvcvImageWrapNvSciBuf(const NVCVImageRequirements *imageRequirements, NvSciBufObj nvsciBufObj,
                                 NVCVImageHandle *outImageHandle);
/**
 * @brief Creates an NvSciBufObj with RawBuffer type.
 *
 * Utility function to create and reconcile an NvSciBufObj of type RawBuffer
 * for use with both CPU and PVA.
 *
 * @param bufobj Pointer to NvSciBufObj to be created.
 * @param sciBufModule NvSciBufModule handle.
 * @param size Size of the buffer to allocate.
 * @return NvSciError_Success on success, or an NvSci error code on failure.
 */
NvSciError pvaCreateNvSciBufRawBuffer(NvSciBufObj *bufobj, NvSciBufModule sciBufModule, int64_t size);

/**
 * @brief Creates an NvSciBufObj with Tensor type.
 *
 * Utility function to create and reconcile an NvSciBufObj of type Tensor
 * for use with both CPU and PVA.
 *
 * Mandatory attributes:
 *   - numDims: Number of tensor dimensions (1 to NV_SCI_BUF_TENSOR_MAX_DIMS)
 *   - dims: Array of dimension sizes (SizePerDim)
 *   - dataType: Data type (NvSciBufAttrValDataType)
 *   - alignmentPerDim: Alignment per dimension
 *
 * Optional attributes (pass 0 or NULL to skip):
 *   - baseAddrAlign: Base address alignment (must be power of 2, 0 = use default)
 *   - pixelFormat: Pixel format (0 = skip)
 *
 * @param bufobj Pointer to NvSciBufObj to be created.
 * @param sciBufModule NvSciBufModule handle.
 * @param numDims Number of tensor dimensions (mandatory).
 * @param dims Array of dimension sizes (mandatory).
 * @param dataType NvSciBuf data type (mandatory, e.g., NvSciDataType_Uint8).
 * @param alignmentPerDim Array of alignment values per dimension (must be power of 2).
 *                        Pass NULL to use default alignment of 1 for all dimensions.
 * @param baseAddrAlign Base address alignment in bytes (optional, 0 = use default).
 *                      Must be power of 2 if specified.
 * @param pixelFormat Pixel format for Tensor reconciliation (optional, 0 = skip).
 *                    Only needed when reconciling with NvSciBufType_Image.
 * @return NvSciError_Success on success, or an NvSci error code on failure.
 */
NvSciError pvaCreateNvSciBufTensor(NvSciBufObj *bufobj, NvSciBufModule sciBufModule, uint32_t numDims,
                                   const uint64_t *dims, NvSciBufAttrValDataType dataType,
                                   const uint32_t *alignmentPerDim, uint64_t baseAddrAlign,
                                   NvSciBufAttrValColorFmt pixelFormat);

/**
 * @brief Imports tensor requirements from an NvSciBufType_Tensor buffer.
 *
 * Queries tensor attributes (dimensions, strides, data type) from an NvSciBufObj
 * of type NvSciBufType_Tensor and populates an NVCVTensorRequirements structure.
 * This allows deriving NVCV tensor requirements directly from an allocated NvSciBuf
 * tensor, eliminating the need to specify tensor parameters twice.
 *
 * @param nvsciBufObj The NvSciBufObj to query (must be NvSciBufType_Tensor).
 * @param tensorLayout The tensor layout to use (e.g., "HWC").
 * @param tensorRequirements Pointer to NVCVTensorRequirements to be populated.
 * @return NVCV_SUCCESS on success, or an NVCV error code on failure.
 */
NVCVStatus nvcvTensorImportRequirementsFromNvSci(NvSciBufObj nvsciBufObj, NVCVTensorLayout tensorLayout,
                                                 NVCVTensorRequirements *tensorRequirements);

/**
 * @brief Creates an NvSciSyncObj.
 *
 * Utility function to create and reconcile an NvSciSyncObj.
 *
 * @param syncObj Pointer to NvSciSyncObj to be created.
 * @param sciSyncModule NvSciSyncModule handle.
 * @param pvaPerm PVA client permissions for the sync object.
 * @param cpuPerm CPU access permissions for the sync object.
 * @return NvSciError_Success on success, or an NvSci error code on failure.
 */
NvSciError pvaCreateNvSciSyncObj(NvSciSyncObj *syncObj, NvSciSyncModule sciSyncModule, cupvaSyncClientType_t pvaPerm,
                                 NvSciSyncAccessPerm cpuPerm);

#endif // NVSCI_INTEROP_UTILS_H