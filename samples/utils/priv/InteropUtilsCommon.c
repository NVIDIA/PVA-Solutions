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

#include "InteropUtilsCommon.h"

#include <cupva_host.h>
#include <nvcv/ImageData.h>
#include <nvcv/Status.h>
#include <nvcv/TensorData.h>
#include <stdio.h>
#include <stdlib.h>

NVCVStatus InteropCreateCleanupCtx(InteropCleanupCtx **outCtx)
{
    if (outCtx == NULL)
    {
        return NVCV_ERROR_INVALID_ARGUMENT;
    }

    *outCtx = (InteropCleanupCtx *)malloc(sizeof(InteropCleanupCtx));
    if (*outCtx == NULL)
    {
        printf("Failed to allocate InteropCleanupCtx\\n");
        return NVCV_ERROR_OUT_OF_MEMORY;
    }

    (*outCtx)->pvaMemHandle = NULL;
    (*outCtx)->tensorData   = NULL;
    (*outCtx)->imageData    = NULL;

    return NVCV_SUCCESS;
}

void InteropUnifiedResourceCleanup(void *ctxCleanup, const void *data)
{
    (void)data; // data param is ignored, present for API compatibility

    if (ctxCleanup == NULL)
    {
        printf("Error: InteropCleanupCtx is NULL in InteropUnifiedResourceCleanup\\n");
        return;
    }
    InteropCleanupCtx *ctx = (InteropCleanupCtx *)ctxCleanup;

    if (ctx->pvaMemHandle != NULL)
    {
        CupvaMemFree(ctx->pvaMemHandle);
        ctx->pvaMemHandle = NULL;
    }

    // In InteropCleanupCtx, either tensorData or imageData (or neither) would point to
    // a structure that was allocated and whose pointer was stored here.
    // That structure itself is freed here if the pointer is non-NULL.
    // This cleanup function is called by NVCV when the Tensor/Image is destroyed,
    // and it's responsible for freeing *both* the NVCVTensorData/NVCVImageData
    // AND the InteropCleanupCtx.
    if (ctx->tensorData != NULL)
    {
        free(ctx->tensorData);
        ctx->tensorData = NULL;
    }

    if (ctx->imageData != NULL)
    {
        free(ctx->imageData);
        ctx->imageData = NULL;
    }

    free(ctx); // Free the InteropCleanupCtx structure itself
}

NVCVStatus InteropPerformCleanupOnFailure(NVCVStatus statusToReturn, InteropCleanupCtx *ctxToClean,
                                          void *internalDataToClean)
{
    if (ctxToClean != NULL)
    {
        if (ctxToClean->pvaMemHandle != NULL)
        {
            CupvaMemFree(ctxToClean->pvaMemHandle);
            // pvaMemHandle is part of ctxToClean, so no need to set to NULL before free(ctxToClean)
        }
        free(ctxToClean); // Frees the context structure
    }

    // internalDataToClean is the NVCVTensorData or NVCVImageData structure that was
    // allocated separately. If the process failed before this data was linked to
    // ctxToClean (or before ctxToClean was linked to NVCV), it needs to be freed here.
    // If it *was* linked to ctxToClean, and ctxToClean was non-NULL,
    // InteropUnifiedResourceCleanup would be responsible if NVCV took ownership.
    // But this function is for *before* NVCV takes ownership.
    if (internalDataToClean != NULL)
    {
        free(internalDataToClean);
    }

    return statusToReturn;
}

NVCVStatus InteropPopulateTensorData(NVCVTensorData *internalTensorData,
                                     const NVCVTensorRequirements *tensorRequirements, void *pvaMemHandle)
{
    if (internalTensorData == NULL || tensorRequirements == NULL || pvaMemHandle == NULL)
    {
        return NVCV_ERROR_INVALID_ARGUMENT;
    }

    // Copy common fields between NVCVTensorRequirements and NVCVTensorData
    internalTensorData->dtype  = tensorRequirements->dtype;
    internalTensorData->layout = tensorRequirements->layout;
    internalTensorData->rank   = tensorRequirements->rank;
    memcpy(internalTensorData->shape, tensorRequirements->shape, sizeof(internalTensorData->shape));

    internalTensorData->bufferType = NVCV_TENSOR_BUFFER_STRIDED_CUDA; // Assuming CUDA accessible via PVA
    memcpy(&internalTensorData->buffer.strided.strides, &tensorRequirements->strides,
           sizeof(tensorRequirements->strides));
    internalTensorData->buffer.strided.basePtr = (NVCVByte *)pvaMemHandle;

    return NVCV_SUCCESS;
}

NVCVStatus InteropPopulateImageData(NVCVImageData *internalImageData, const NVCVImageRequirements *imageRequirements,
                                    void *pvaMemHandle)
{
    if (internalImageData == NULL || imageRequirements == NULL || pvaMemHandle == NULL)
    {
        return NVCV_ERROR_INVALID_ARGUMENT;
    }

    NVCVStatus local_nvcv_status = NVCV_SUCCESS;

    internalImageData->format     = imageRequirements->format;
    internalImageData->bufferType = NVCV_IMAGE_BUFFER_STRIDED_CUDA; // Assuming CUDA accessible via PVA

    int32_t numPlanes = 0;
    local_nvcv_status = nvcvImageFormatGetNumPlanes(imageRequirements->format, &numPlanes);
    if (local_nvcv_status != NVCV_SUCCESS)
    {
        printf("InteropPopulateImageData: nvcvImageFormatGetNumPlanes failed (%d)\n", local_nvcv_status);
        return local_nvcv_status;
    }
    internalImageData->buffer.strided.numPlanes = numPlanes;

    NVCVByte *currentPlaneBasePtr = (NVCVByte *)pvaMemHandle;
    for (int32_t i = 0; i < numPlanes; i++)
    {
        int32_t planeWidth  = 0;
        int32_t planeHeight = 0;
        local_nvcv_status   = nvcvImageFormatGetPlaneSize(imageRequirements->format, i, imageRequirements->width,
                                                          imageRequirements->height, &planeWidth, &planeHeight);
        if (local_nvcv_status != NVCV_SUCCESS)
        {
            printf("InteropPopulateImageData: nvcvImageFormatGetPlaneSize for plane %d failed (%d)\n", i,
                   local_nvcv_status);
            return local_nvcv_status;
        }
        internalImageData->buffer.strided.planes[i].width     = planeWidth;
        internalImageData->buffer.strided.planes[i].height    = planeHeight;
        internalImageData->buffer.strided.planes[i].rowStride = imageRequirements->planeRowStride[i];
        internalImageData->buffer.strided.planes[i].basePtr   = currentPlaneBasePtr;
        currentPlaneBasePtr += imageRequirements->planeRowStride[i] * planeHeight;
    }

    return NVCV_SUCCESS;
}