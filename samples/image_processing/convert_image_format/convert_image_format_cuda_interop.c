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
#include "ref/convert_image_format_ref.h"

#include <OpConvertImageFormat.h>
#include <PvaAllocator.h>
#include <cupva_host.h>
#include <nvcv/Fwd.h>
#include <nvcv/Image.h>
#include <nvcv/alloc/Allocator.h>

int main(int argc, char **argv)
{
    const int32_t imgWidth         = 1920;
    const int32_t imgHeight        = 1080;
    NVCVImageFormat inImageFormat  = NVCV_IMAGE_FORMAT_NV12;
    NVCVImageFormat outImageFormat = NVCV_IMAGE_FORMAT_RGB8;

    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVImageHandle inImageHandle     = NULL;
    void *inImageCudaBuf              = NULL;
    NVCVImageHandle outImageHandle    = NULL;
    void *outImageCudaBuf             = NULL;
    NVCVImageHandle refImageHandle    = NULL;
    void *refImageCudaBuf             = NULL;
    NVCVOperatorHandle operatorHandle = NULL;
    cupvaSyncObj_t sync               = NULL;
    cupvaStream_t stream              = NULL;
    int32_t err                       = 0;

    NVCVImageRequirements inImageRequirements;
    NVCV_CHECK_ERROR_GOTO(nvcvImageCalcRequirementsPva(imgWidth, imgHeight, inImageFormat, 0, 0, &inImageRequirements),
                          err, FreeResources);

    int64_t inImageSizeBytes = 0;
    NVCV_CHECK_ERROR_GOTO(nvcvMemRequirementsCalcTotalSizeBytes(&inImageRequirements.mem.cudaMem, &inImageSizeBytes),
                          err, FreeResources);

    CUDA_CHECK_ERROR_GOTO(cudaHostAlloc((void **)&inImageCudaBuf, inImageSizeBytes, cudaHostAllocMapped), err,
                          FreeResources);

    NVCV_CHECK_ERROR_GOTO(nvcvImageWrapCudaBuf(&inImageRequirements, inImageCudaBuf, &inImageHandle), err,
                          FreeResources);

    NVCVImageRequirements outImageRequirements;
    NVCV_CHECK_ERROR_GOTO(
        nvcvImageCalcRequirementsPva(imgWidth, imgHeight, outImageFormat, 0, 0, &outImageRequirements), err,
        FreeResources);

    int64_t outImageSizeBytes = 0;
    NVCV_CHECK_ERROR_GOTO(nvcvMemRequirementsCalcTotalSizeBytes(&outImageRequirements.mem.cudaMem, &outImageSizeBytes),
                          err, FreeResources);

    CUDA_CHECK_ERROR_GOTO(cudaHostAlloc((void **)&outImageCudaBuf, outImageSizeBytes, cudaHostAllocMapped), err,
                          FreeResources);

    NVCV_CHECK_ERROR_GOTO(nvcvImageWrapCudaBuf(&outImageRequirements, outImageCudaBuf, &outImageHandle), err,
                          FreeResources);

    CUDA_CHECK_ERROR_GOTO(cudaHostAlloc((void **)&refImageCudaBuf, outImageSizeBytes, cudaHostAllocMapped), err,
                          FreeResources);

    NVCV_CHECK_ERROR_GOTO(nvcvImageWrapCudaBuf(&outImageRequirements, refImageCudaBuf, &refImageHandle), err,
                          FreeResources);

    generateConvertImageFormatInput(inImageHandle);

    NVCV_CHECK_ERROR_GOTO(pvaConvertImageFormatCreate(&operatorHandle, &inImageRequirements, &outImageRequirements),
                          err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaSyncObjCreate(&sync, false, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD), err,
                           FreeResources);

    cupvaFence_t fence;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceInit(&fence, sync), err, FreeResources);

    cupvaCmd_t rf;
    CUPVA_CHECK_ERROR_GOTO(CupvaCmdRequestFencesInit(&rf, &fence, 1), err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamCreate(&stream, CUPVA_PVA0, CUPVA_VPU_ANY), err, FreeResources);

    cupvaCmd_t const *cmds[1]  = {&rf};
    cupvaCmdStatus_t status[1] = {NULL};

    NVCV_CHECK_ERROR_GOTO(pvaConvertImageFormatSubmit(operatorHandle, stream, inImageHandle, outImageHandle), err,
                          FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    convertImageFormatRef(inImageHandle, refImageHandle);

    err = checkConvertImageFormatOutput(refImageHandle, outImageHandle);
    if (err)
    {
        printf("Test Fail\n");
    }
    else
    {
        printf("Test Pass\n");
    }

FreeResources: /* clean up all allocated resources */
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvImageDecRef(inImageHandle, NULL);
    nvcvImageDecRef(outImageHandle, NULL);
    nvcvImageDecRef(refImageHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    cudaFreeHost(inImageCudaBuf);
    cudaFreeHost(outImageCudaBuf);
    cudaFreeHost(refImageCudaBuf);

    return err;
}
