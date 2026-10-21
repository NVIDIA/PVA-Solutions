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

#ifndef INTEROP_UTILS_COMMON_H
#define INTEROP_UTILS_COMMON_H

#include <cupva_host.h>
#include <nvcv/Image.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>
#include <stddef.h>

// Common context for resource cleanup
typedef struct
{
    void *pvaMemHandle;
    NVCVTensorData *tensorData; // Points to the allocated NVCVTensorData structure
    NVCVImageData *imageData;   // Points to the allocated NVCVImageData structure
} InteropCleanupCtx;

// Enum to distinguish between Tensor and Image data for internal logic
typedef enum
{
    INTEROP_DATA_TENSOR,
    INTEROP_DATA_IMAGE
} InteropInternalDataType;

/**
 * @brief Unified cleanup function for resources wrapped by NVCV data construct functions.
 *
 * This function is intended to be passed to nvcvTensorWrapDataConstruct or nvcvImageWrapDataConstruct.
 * It is responsible for freeing the InteropCleanupCtx AND the NVCVTensorData/NVCVImageData
 * structure that was pointed to by ctx->tensorData or ctx->imageData.
 *
 * @param ctxCleanup Pointer to the InteropCleanupCtx to be cleaned up.
 * @param data Associated data (typically the NVCVTensorData or NVCVImageData pointer), ignored by this implementation.
 */
void InteropUnifiedResourceCleanup(void *ctxCleanup, const void *data);

/**
 * @brief Allocates and initializes an InteropCleanupCtx.
 *
 * The pvaMemHandle, tensorData, and imageData members are initialized to NULL.
 * This function does not allocate the structures for tensorData or imageData themselves.
 *
 * @param outCtx Pointer to a InteropCleanupCtx pointer, which will be set to the newly allocated context.
 * @return NVCV_SUCCESS on success, or NVCV_ERROR_OUT_OF_MEMORY if allocation fails,
 *         or NVCV_ERROR_INVALID_ARGUMENT if outCtx is NULL.
 */
NVCVStatus InteropCreateCleanupCtx(InteropCleanupCtx **outCtx);

/**
 * @brief Performs cleanup if intermediate steps fail before NVCV takes ownership of resources.
 *
 * This function frees the InteropCleanupCtx (including its pvaMemHandle if set) AND the
 * separately allocated internalData (which would be an NVCVTensorData or NVCVImageData structure).
 * It is intended to be called in error paths of wrapping functions before nvcv*WrapDataConstruct
 * successfully completes and takes ownership via InteropUnifiedResourceCleanup.
 *
 * @param statusToReturn The NVCVStatus code to be returned by the calling function.
 * @param ctxToClean Pointer to the InteropCleanupCtx to clean. Can be NULL.
 * @param internalDataToClean Pointer to the separately allocated NVCVTensorData or NVCVImageData structure. Can be NULL.
 * @return The value of statusToReturn.
 */
NVCVStatus InteropPerformCleanupOnFailure(NVCVStatus statusToReturn, InteropCleanupCtx *ctxToClean,
                                          void *internalDataToClean);

/**
 * @brief Populates an NVCVTensorData structure from NVCVTensorRequirements.
 *
 * @param internalTensorData Pointer to the NVCVTensorData structure to populate.
 * @param tensorRequirements Pointer to the NVCVTensorRequirements providing the data.
 * @param pvaMemHandle The PVA memory handle to be used as the basePtr for the tensor data.
 * @return NVCV_SUCCESS on success, or an error code if inputs are invalid.
 */
NVCVStatus InteropPopulateTensorData(NVCVTensorData *internalTensorData,
                                     const NVCVTensorRequirements *tensorRequirements, void *pvaMemHandle);

/**
 * @brief Populates an NVCVImageData structure from NVCVImageRequirements.
 *
 * @param internalImageData Pointer to the NVCVImageData structure to populate.
 * @param imageRequirements Pointer to the NVCVImageRequirements providing the data.
 * @param pvaMemHandle The PVA memory handle to be used as the basePtr for the image planes.
 * @return NVCV_SUCCESS on success, or an error code (e.g., from nvcvImageFormatGetNumPlanes or nvcvImageFormatGetPlaneSize).
 */
NVCVStatus InteropPopulateImageData(NVCVImageData *internalImageData, const NVCVImageRequirements *imageRequirements,
                                    void *pvaMemHandle);

#endif // INTEROP_UTILS_COMMON_H