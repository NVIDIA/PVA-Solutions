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

#include "CudaInteropUtils.h"

#include "ErrorCheckMacros.h"
#include "InteropUtilsCommon.h" // Added for common types and functions

#include <cupva_host.h>
#include <cupva_host_cuda.h>
#include <nvcv/Image.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Requirements.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

// Helper function for common resource allocation and CUDA memory import
static NVCVStatus setup_cuda_import_resources(const NVCVMemRequirements *memReqForSizeCalc, void *cudaBufToImport,
                                              size_t internalDataStructSize,
                                              InteropInternalDataType dataType, // Changed type
                                              InteropCleanupCtx **outCtxForCleanup,
                                              void **outInternalData, // Changed type
                                              const char *callerFunctionName)
{
    NVCVStatus local_nvcv_status = NVCV_SUCCESS;
    cupvaError_t local_cupva_status;

    *outCtxForCleanup = NULL;
    *outInternalData  = NULL;

    // Use common context creation
    local_nvcv_status = InteropCreateCleanupCtx(outCtxForCleanup);
    if (local_nvcv_status != NVCV_SUCCESS)
    {
        // InteropCreateCleanupCtx already prints a message
        return local_nvcv_status;
    }

    *outInternalData = malloc(internalDataStructSize);
    if (*outInternalData == NULL)
    {
        printf("%s: Failed to allocate internal data structure\n", callerFunctionName);
        // *outCtxForCleanup is already allocated, InteropPerformCleanupOnFailure will handle it
        return NVCV_ERROR_OUT_OF_MEMORY;
    }

    // Link the allocated internal data to the appropriate pointer in the cleanup context
    if (dataType == INTEROP_DATA_TENSOR)
    {
        (*outCtxForCleanup)->tensorData = (NVCVTensorData *)*outInternalData;
    }
    else // INTEROP_DATA_IMAGE
    {
        (*outCtxForCleanup)->imageData = (NVCVImageData *)*outInternalData;
    }

    int64_t sizeBytes = 0;
    local_nvcv_status = nvcvMemRequirementsCalcTotalSizeBytes(memReqForSizeCalc, &sizeBytes);
    if (local_nvcv_status != NVCV_SUCCESS)
    {
        printf("%s: nvcvMemRequirementsCalcTotalSizeBytes failed (%d)\n", callerFunctionName, local_nvcv_status);
        return local_nvcv_status;
    }

    local_cupva_status =
        CupvaCudaMemImport(&(*outCtxForCleanup)->pvaMemHandle, cudaBufToImport, sizeBytes, CUPVA_READ_WRITE);
    if (local_cupva_status != CUPVA_ERROR_NONE)
    {
        printf("%s: CupvaCudaMemImport failed (%d)\n", callerFunctionName, local_cupva_status);
        return NVCV_ERROR_DEVICE;
    }

    return NVCV_SUCCESS;
}

NVCVStatus nvcvTensorWrapCudaBuf(const NVCVTensorRequirements *tensorRequirements, void *cudaBuf,
                                 NVCVTensorHandle *outTensorHandle)
{
    NVCVStatus nvcv_status = NVCV_SUCCESS;

    InteropCleanupCtx *ctxForCleanup   = NULL;
    NVCVTensorData *internalTensorData = NULL;

    nvcv_status = setup_cuda_import_resources(&tensorRequirements->mem.cudaMem, cudaBuf, sizeof(NVCVTensorData),
                                              INTEROP_DATA_TENSOR, &ctxForCleanup, (void **)&internalTensorData,
                                              "nvcvTensorWrapCudaBuf");

    if (nvcv_status != NVCV_SUCCESS)
    {
        goto nvcv_wrap_failed;
    }

    // Populate NVCVTensorData (Tensor-specific part)
    nvcv_status = InteropPopulateTensorData(internalTensorData, tensorRequirements, ctxForCleanup->pvaMemHandle);
    if (nvcv_status != NVCV_SUCCESS)
    {
        goto nvcv_wrap_failed;
    }

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorWrapDataConstruct(internalTensorData, (NVCVTensorDataCleanupFunc)InteropUnifiedResourceCleanup,
                                    ctxForCleanup, outTensorHandle),
        nvcv_status, nvcv_wrap_failed);

    /* cuPVA 2.10 onwards does not map an imported CUDA buffer back to the CPU, so publish the
     * host pointer the caller allocated. PvaGet{Tensor,Image}HostPointer reads it back.
     */
    nvcvTensorSetUserPointer(*outTensorHandle, cudaBuf);

    // If nvcvTensorWrapDataConstruct succeeds, it takes ownership of ctxForCleanup and internalTensorData
    // (via ctxForCleanup->tensorData).
    // Set them to NULL to prevent double free by InteropPerformCleanupOnFailure if this path is taken.
    ctxForCleanup      = NULL;
    internalTensorData = NULL; // The data is now owned by NVCV via ctxForCleanup->tensorData

    return NVCV_SUCCESS;

nvcv_wrap_failed:

    return InteropPerformCleanupOnFailure(nvcv_status, ctxForCleanup, (void *)internalTensorData);
}

NVCVStatus nvcvImageWrapCudaBuf(const NVCVImageRequirements *imageRequirements, void *cudaBuf,
                                NVCVImageHandle *outImageHandle)
{
    NVCVStatus nvcv_status = NVCV_SUCCESS;

    InteropCleanupCtx *ctxForCleanup = NULL;
    NVCVImageData *internalImageData = NULL;

    nvcv_status =
        setup_cuda_import_resources(&imageRequirements->mem.cudaMem, cudaBuf, sizeof(NVCVImageData), INTEROP_DATA_IMAGE,
                                    &ctxForCleanup, (void **)&internalImageData, "nvcvImageWrapCudaBuf");

    if (nvcv_status != NVCV_SUCCESS)
    {
        goto nvcv_wrap_failed;
    }

    // Populate NVCVImageData (Image-specific part)
    nvcv_status = InteropPopulateImageData(internalImageData, imageRequirements, ctxForCleanup->pvaMemHandle);
    if (nvcv_status != NVCV_SUCCESS)
    {
        goto nvcv_wrap_failed;
    }

    NVCV_CHECK_ERROR_GOTO(
        nvcvImageWrapDataConstruct(internalImageData, (NVCVImageDataCleanupFunc)InteropUnifiedResourceCleanup,
                                   ctxForCleanup, outImageHandle),
        nvcv_status, nvcv_wrap_failed);

    /* cuPVA 2.10 onwards does not map an imported CUDA buffer back to the CPU, so publish the
     * host pointer the caller allocated. PvaGet{Tensor,Image}HostPointer reads it back.
     */
    nvcvImageSetUserPointer(*outImageHandle, cudaBuf);

    // If nvcvImageWrapDataConstruct succeeds, it takes ownership of ctxForCleanup and internalImageData.
    ctxForCleanup     = NULL;
    internalImageData = NULL; // Data now owned by NVCV via ctxForCleanup->imageData

    return NVCV_SUCCESS;

nvcv_wrap_failed:
    return InteropPerformCleanupOnFailure(nvcv_status, ctxForCleanup, (void *)internalImageData);
}