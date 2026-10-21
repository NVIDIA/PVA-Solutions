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
#include "ref/gaussian_filter_ref.h"

#include <ImageIO.h>
#include <OpGaussianFilter.h>
#include <PvaAllocator.h>
#include <cupva_host.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>

// Hardcoded filenames for input and output
#define DEFAULT_INPUT_FILE "cornersubpix/kodim08.bin"
#define DEFAULT_OUTPUT_FILE "filtered_kodim08.bin"

int main(int argc, char **argv)
{
    const int32_t imgWidth   = 768;
    const int32_t imgHeight  = 512;
    const int32_t imgChannel = 1;
    const int32_t tensorRank = 3;
    const int32_t kernelSize = 5;
    const float sigmaX       = 1.7f;
    const float sigmaY       = 1.7f;

    // For assets directory
    const int maxPathLength               = MAX_PATH_LENGTH;
    char assetsDirectory[MAX_PATH_LENGTH] = {0};

    // Flag to indicate if we're using an input file
    bool useInputFile = (argc > 1); // Use input file if any command line argument was provided

    const NVCVBorderType borderMode = NVCV_BORDER_CONSTANT;
    const int32_t borderValue       = 99;

    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVTensorHandle inTensorHandle   = NULL;
    void *inTensorCudaBuf             = NULL;
    NVCVTensorHandle outTensorHandle  = NULL;
    void *outTensorCudaBuf            = NULL;
    NVCVTensorHandle refTensorHandle  = NULL;
    void *refTensorCudaBuf            = NULL;
    NVCVOperatorHandle operatorHandle = NULL;
    cupvaSyncObj_t sync               = NULL;
    cupvaStream_t stream              = NULL;
    int32_t err                       = 0;

    NVCVTensorLayout tensorLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HWC", &tensorLayout), err, FreeResources);

    int64_t tensorShape[] = {imgHeight, imgWidth, imgChannel};
    NVCVTensorRequirements tensorRequirements;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank, tensorShape, NVCV_DATA_TYPE_U8, tensorLayout, 0, 0,
                                                        &tensorRequirements),
                          err, FreeResources);

    int64_t inTensorSizeBytes = 0;
    NVCV_CHECK_ERROR_GOTO(nvcvMemRequirementsCalcTotalSizeBytes(&tensorRequirements.mem.cudaMem, &inTensorSizeBytes),
                          err, FreeResources);

    CUDA_CHECK_ERROR_GOTO(cudaHostAlloc((void **)&inTensorCudaBuf, inTensorSizeBytes, cudaHostAllocMapped), err,
                          FreeResources);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorWrapCudaBuf(&tensorRequirements, inTensorCudaBuf, &inTensorHandle), err,
                          FreeResources);

    int64_t outTensorSizeBytes = 0;
    NVCV_CHECK_ERROR_GOTO(nvcvMemRequirementsCalcTotalSizeBytes(&tensorRequirements.mem.cudaMem, &outTensorSizeBytes),
                          err, FreeResources);

    CUDA_CHECK_ERROR_GOTO(cudaHostAlloc((void **)&outTensorCudaBuf, outTensorSizeBytes, cudaHostAllocMapped), err,
                          FreeResources);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorWrapCudaBuf(&tensorRequirements, outTensorCudaBuf, &outTensorHandle), err,
                          FreeResources);

    CUDA_CHECK_ERROR_GOTO(cudaHostAlloc((void **)&refTensorCudaBuf, outTensorSizeBytes, cudaHostAllocMapped), err,
                          FreeResources);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorWrapCudaBuf(&tensorRequirements, refTensorCudaBuf, &refTensorHandle), err,
                          FreeResources);

    // Load input from file if assets directory specified, otherwise generate random input
    if (useInputFile)
    {
        // Get assets directory
        int assetsErr = GetAssetsDirectory(argc, argv, assetsDirectory, maxPathLength);
        if (assetsErr)
        {
            printf("Test Fail: Assets directory could not be found\n");
            goto FreeResources;
        }

        err = loadGaussianFilterInput(inTensorHandle, assetsDirectory, DEFAULT_INPUT_FILE, imgWidth * imgHeight);
        if (err)
        {
            printf("Test Fail: Input files could not be loaded\n");
            goto FreeResources;
        }
    }
    else
    {
        generateRandomGaussianFilterInput(inTensorHandle);
    }

    NVCV_CHECK_ERROR_GOTO(pvaGaussianFilterCreate(&operatorHandle, &tensorRequirements, sigmaX, sigmaY, kernelSize,
                                                  borderMode, borderValue),
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

    NVCV_CHECK_ERROR_GOTO(pvaGaussianFilterSubmit(operatorHandle, stream, inTensorHandle, outTensorHandle), err,
                          FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    gaussianFilterRef(inTensorHandle, refTensorHandle, sigmaX, sigmaY, kernelSize, borderMode, borderValue);

    // Write output to file if assets directory was specified
    if (useInputFile)
    {
        err = saveGaussianFilterOutput(outTensorHandle, ".", DEFAULT_OUTPUT_FILE, imgWidth * imgHeight);
        if (err)
        {
            printf("Test Fail: Output files could not be saved\n");
            goto FreeResources;
        }
    }

    err = checkGaussianFilterOutput(refTensorHandle, outTensorHandle);
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
    nvcvTensorDecRef(inTensorHandle, NULL);
    nvcvTensorDecRef(outTensorHandle, NULL);
    nvcvTensorDecRef(refTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    cudaFreeHost(inTensorCudaBuf);
    cudaFreeHost(outTensorCudaBuf);
    cudaFreeHost(refTensorCudaBuf);
    return err;
}
