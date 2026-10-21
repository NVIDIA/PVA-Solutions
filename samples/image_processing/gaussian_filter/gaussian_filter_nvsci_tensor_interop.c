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

/**
 * @file gaussian_filter_nvsci_tensor.c
 * @brief Gaussian filter sample demonstrating NvSciBufType_Tensor usage.
 *
 * This sample shows how to use NvSciBufType_Tensor buffers with NVCV/PVA operators.
 * Unlike NvSciBufType_RawBuffer, Tensor buffers embed tensor metadata (dimensions,
 * strides, data type), which can be queried to derive NVCVTensorRequirements.
 *
 * Flow:
 *   1. pvaCreateNvSciBufTensor() - Create tensor buffer with dimensions
 *   2. nvcvTensorImportRequirementsFromNvSci() - Derive requirements from buffer
 *   3. nvcvTensorWrapNvSciBuf() - Wrap buffer into NVCV tensor handle
 *   4. pvaGaussianFilterCreate() - Create operator using derived requirements
 */

#include "ErrorCheckMacros.h"
#include "NvSciInteropUtils.h"
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

    printf("Using NvSciBufType_Tensor for buffer allocation\n");

    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NvSciBufModule nvSciBufModule       = NULL;
    NvSciSyncModule nvSciSyncModule     = NULL;
    NvSciSyncCpuWaitContext waitContext = NULL;
    NVCVTensorHandle inTensorHandle     = NULL;
    NvSciBufObj inTensorNvsciBuf        = NULL;
    NVCVTensorHandle outTensorHandle    = NULL;
    NvSciBufObj outTensorNvsciBuf       = NULL;
    NVCVTensorHandle refTensorHandle    = NULL;
    NvSciBufObj refTensorNvsciBuf       = NULL;
    NvSciSyncObj nvSciSyncObj           = NULL;
    NvSciSyncFence nvSciSyncFence       = NvSciSyncFenceInitializer;
    NVCVOperatorHandle operatorHandle   = NULL;
    cupvaSyncObj_t sync                 = NULL;
    cupvaStream_t stream                = NULL;
    int32_t err                         = 0;

    NVSCI_CHECK_ERROR_GOTO(NvSciBufModuleOpen(&nvSciBufModule), err, FreeResources);
    NVSCI_CHECK_ERROR_GOTO(NvSciSyncModuleOpen(&nvSciSyncModule), err, FreeResources);
    NVSCI_CHECK_ERROR_GOTO(NvSciSyncCpuWaitContextAlloc(nvSciSyncModule, &waitContext), err, FreeResources);

    // Create tensor layout for "HWC" format
    NVCVTensorLayout tensorLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HWC", &tensorLayout), err, FreeResources);

    // Tensor dimensions for NvSciBuf
    uint64_t tensorDims[] = {(uint64_t)imgHeight, (uint64_t)imgWidth, (uint64_t)imgChannel};

    // --- Input Tensor ---
    // Step 1: Create NvSciBufType_Tensor buffer
    // Parameters: numDims, dims, dataType, alignmentPerDim (NULL=default), baseAddrAlign (0=default), pixelFormat (0=skip)
    NVSCI_CHECK_ERROR_GOTO(pvaCreateNvSciBufTensor(&inTensorNvsciBuf, nvSciBufModule, tensorRank, tensorDims,
                                                   NvSciDataType_Uint8, NULL, 0, 0),
                           err, FreeResources);

    // Step 2: Derive NVCVTensorRequirements from the buffer
    NVCVTensorRequirements tensorRequirements;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorImportRequirementsFromNvSci(inTensorNvsciBuf, tensorLayout, &tensorRequirements),
                          err, FreeResources);

    // Step 3: Wrap buffer into NVCV tensor handle
    NVCV_CHECK_ERROR_GOTO(nvcvTensorWrapNvSciBuf(&tensorRequirements, inTensorNvsciBuf, &inTensorHandle), err,
                          FreeResources);

    // --- Output Tensor ---
    NVSCI_CHECK_ERROR_GOTO(pvaCreateNvSciBufTensor(&outTensorNvsciBuf, nvSciBufModule, tensorRank, tensorDims,
                                                   NvSciDataType_Uint8, NULL, 0, 0),
                           err, FreeResources);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorWrapNvSciBuf(&tensorRequirements, outTensorNvsciBuf, &outTensorHandle), err,
                          FreeResources);

    // --- Reference Tensor ---
    NVSCI_CHECK_ERROR_GOTO(pvaCreateNvSciBufTensor(&refTensorNvsciBuf, nvSciBufModule, tensorRank, tensorDims,
                                                   NvSciDataType_Uint8, NULL, 0, 0),
                           err, FreeResources);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorWrapNvSciBuf(&tensorRequirements, refTensorNvsciBuf, &refTensorHandle), err,
                          FreeResources);

    // Create sync object
    NVCV_CHECK_ERROR_GOTO(
        pvaCreateNvSciSyncObj(&nvSciSyncObj, nvSciSyncModule, CUPVA_SIGNALER_WAITER, NvSciSyncAccessPerm_WaitOnly), err,
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

    // Step 4: Create operator using derived tensorRequirements
    NVCV_CHECK_ERROR_GOTO(pvaGaussianFilterCreate(&operatorHandle, &tensorRequirements, sigmaX, sigmaY, kernelSize,
                                                  borderMode, borderValue),
                          err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaSyncObjCreateFromNvsci(&sync, nvSciSyncObj, CUPVA_SIGNALER_WAITER), err, FreeResources);

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

    CUPVA_CHECK_ERROR_GOTO(CupvaFenceExport(&fence, &nvSciSyncFence), err, FreeResources);

    NVSCI_CHECK_ERROR_GOTO(NvSciSyncFenceWait(&nvSciSyncFence, waitContext, NvSciSyncFenceMaxTimeout), err,
                           FreeResources);

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
    NvSciBufObjFree(inTensorNvsciBuf);
    NvSciBufObjFree(outTensorNvsciBuf);
    NvSciBufObjFree(refTensorNvsciBuf);
    NvSciBufModuleClose(nvSciBufModule);
    NvSciSyncObjFree(nvSciSyncObj);
    NvSciSyncCpuWaitContextFree(waitContext);
    NvSciSyncModuleClose(nvSciSyncModule);

    return err;
}
