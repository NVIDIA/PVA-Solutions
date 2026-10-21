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

#include "ErrorCheckMacros.h"
#include "ref/histogram_equalization_ref.h"

#include <ImageIO.h>
#include <OpHistogramEqualization.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Fwd.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>
#include <stdio.h>
#include <string.h>

/**
 * @brief Allocate and initialize tensors
 */
static int AllocateAndInitializeTensors(int32_t image_width, int32_t image_height, NVCVDataType data_type,
                                        const char *assetDir, NVCVAllocatorHandle *allocatorHandle,
                                        NVCVTensorHandle *inTensorHandle, NVCVTensorHandle *outTensorHandle,
                                        NVCVTensorHandle *refTensorHandle, NVCVTensorRequirements *tensorRequirements)
{
    int32_t err = 0;
    NVCVTensorLayout tensorLayout;
    int64_t tensorShape[3];

    tensorShape[0] = image_height;
    tensorShape[1] = image_width;
    tensorShape[2] = 1;

    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HWC", &tensorLayout), err, Error);

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(3, tensorShape, data_type, tensorLayout, 0, 0, tensorRequirements), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensorRequirements, *allocatorHandle, inTensorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensorRequirements, *allocatorHandle, outTensorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensorRequirements, *allocatorHandle, refTensorHandle), err, Error);

    // Generate and load input from a PNG file available in assets or generate random input
    if (assetDir != NULL && assetDir[0] != '\0')
    {
        generateHistogramEqualizationFixedImgInput(tensorRequirements, *inTensorHandle, assetDir);
    }
    else
    {
        generateRandomInput(tensorRequirements, *inTensorHandle);
    }

    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(NVCVTensorRequirements *tensorRequirements, NVCVTensorHandle inTensorHandle,
                          NVCVTensorHandle refTensorHandle, NVCVTensorHandle outTensorHandle, int32_t image_width,
                          int32_t image_height, const char *testName)
{
    int32_t err = getHistogramEqualizationRefData(tensorRequirements, inTensorHandle, refTensorHandle);
    if (err)
    {
        printf("%s test failed: Failed to generate reference data\n", testName);
        return err;
    }

    err = checkHistogramEqualizationOutput(tensorRequirements, inTensorHandle, refTensorHandle, outTensorHandle);
    if (err)
    {
        printf("%s test failed\n", testName);
    }

    // Write output files for visual check
    saveHistogramEqualizationOutput(outTensorHandle, "outKodim08.png", image_width, image_height);
    saveHistogramEqualizationOutput(refTensorHandle, "refKodim08.png", image_width, image_height);

    return err;
}

/**
 * @brief Test histogram equalization operation using CUDA stream
 */
int SubmitHistogramEqualizationToCudaStream(int32_t image_width, int32_t image_height, NVCVDataType data_type,
                                            const char *assetDir)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVTensorHandle inTensorHandle     = NULL;
    NVCVTensorHandle outTensorHandle    = NULL;
    NVCVTensorHandle refTensorHandle    = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cudaStream_t stream                 = NULL;
    int32_t err                         = 0;

    NVCVTensorRequirements tensorRequirements;

    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeTensors(image_width, image_height, data_type, assetDir, &allocatorHandle,
                                       &inTensorHandle, &outTensorHandle, &refTensorHandle, &tensorRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaHistogramEqualizationCreate(&operatorHandle, &tensorRequirements), err, FreeResources);

    /* Submit to CUDA stream */
    NVCV_CHECK_ERROR_GOTO(pvaHistogramEqualizationSubmit(operatorHandle, stream, inTensorHandle, outTensorHandle), err,
                          FreeResources);

    /* Synchronize stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(&tensorRequirements, inTensorHandle, refTensorHandle, outTensorHandle, image_width,
                         image_height, "Submit to CUDA stream");

    // clean up all allocated resources
FreeResources:
    cudaStreamDestroy(stream);
    nvcvTensorDecRef(inTensorHandle, NULL);
    nvcvTensorDecRef(outTensorHandle, NULL);
    nvcvTensorDecRef(refTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

/**
 * @brief Test histogram equalization operation using cuPVA stream
 */
int SubmitHistogramEqualizationToCupvaStream(int32_t image_width, int32_t image_height, NVCVDataType data_type,
                                             const char *assetDir)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVTensorHandle inTensorHandle     = NULL;
    NVCVTensorHandle outTensorHandle    = NULL;
    NVCVTensorHandle refTensorHandle    = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cupvaSyncObj_t sync                 = NULL;
    cupvaStream_t stream                = NULL;
    int32_t err                         = 0;

    NVCVTensorRequirements tensorRequirements;

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeTensors(image_width, image_height, data_type, assetDir, &allocatorHandle,
                                       &inTensorHandle, &outTensorHandle, &refTensorHandle, &tensorRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaHistogramEqualizationCreate(&operatorHandle, &tensorRequirements), err, FreeResources);

    /* Create cuPVA stream and synchronization objects */
    CUPVA_CHECK_ERROR_GOTO(CupvaSyncObjCreate(&sync, false, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD), err,
                           FreeResources);

    cupvaFence_t fence;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceInit(&fence, sync), err, FreeResources);

    cupvaCmd_t rf;
    CUPVA_CHECK_ERROR_GOTO(CupvaCmdRequestFencesInit(&rf, &fence, 1), err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamCreate(&stream, CUPVA_PVA0, CUPVA_VPU_ANY), err, FreeResources);

    /* Submit to cuPVA stream */
    cupvaCmd_t const *cmds[1];
    cmds[0] = &rf;
    cupvaCmdStatus_t status[1];
    status[0] = NULL;

    NVCV_CHECK_ERROR_GOTO(pvaHistogramEqualizationSubmit(operatorHandle, stream, inTensorHandle, outTensorHandle), err,
                          FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    /* Wait for completion */
    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(&tensorRequirements, inTensorHandle, refTensorHandle, outTensorHandle, image_width,
                         image_height, "Submit to cuPVA stream");

    // clean up all allocated resources
FreeResources:
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvTensorDecRef(inTensorHandle, NULL);
    nvcvTensorDecRef(outTensorHandle, NULL);
    nvcvTensorDecRef(refTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char **argv)
{
    /* Parameter settings */
    int32_t image_width  = 1920;
    int32_t image_height = 1280;

    char data_type_str[32] = "u8";
    NVCVDataType data_type = NVCV_DATA_TYPE_U8;
    char assetDir[256]     = "";

    // load custom parameters from argv
    for (int i = 1; i < argc; i++)
    {
        // get asset directory: use -a/--assets in command line
        if (!strcmp(argv[i], "-a") || !strcmp(argv[i], "--assets"))
        {
            strncpy(assetDir, argv[i + 1], sizeof(assetDir) - 1);
            assetDir[sizeof(assetDir) - 1] = '\0';
            i++;
        }
        // get data type: use -d/--data_type in command line
        else if (!strcmp(argv[i], "-d") || !strcmp(argv[i], "--data_type"))
        {
            strncpy(data_type_str, argv[i + 1], sizeof(data_type_str) - 1);
            data_type_str[sizeof(data_type_str) - 1] = '\0';

            if (!strcmp(argv[i + 1], "u8"))
            {
                data_type = NVCV_DATA_TYPE_U8;
            }
            else if (!strcmp(argv[i + 1], "s8"))
            {
                data_type = NVCV_DATA_TYPE_S8;
            }
            else if (!strcmp(argv[i + 1], "u16"))
            {
                data_type = NVCV_DATA_TYPE_U16;
            }
            else if (!strcmp(argv[i + 1], "s16"))
            {
                data_type = NVCV_DATA_TYPE_S16;
            }
            else if (!strcmp(argv[i + 1], "u32"))
            {
                data_type = NVCV_DATA_TYPE_U32;
            }
            else if (!strcmp(argv[i + 1], "s32"))
            {
                data_type = NVCV_DATA_TYPE_S32;
            }
            else if (!strcmp(argv[i + 1], "f32"))
            {
                data_type = NVCV_DATA_TYPE_F32;
            }
            else
            {
                printf("Error: Unsupported data type '%s'\n", argv[i + 1]);
                printf("Supported types: u8, s8, u16, s16, u32, s32, f32\n");
                return 1;
            }
            i++;
        }
    }

    int32_t cudaStreamSubmitTestResult  = 0;
    int32_t cupvaStreamSubmitTestResult = 0;

#if CUPVA_HAS_CUDA_STREAM_SUPPORT == 1

    /* Initialize CUDA context*/
    CUDA_CHECK_ERROR_GOTO(cudaSetDevice(0), cudaStreamSubmitTestResult, CheckResults);

    /* Submit to CUDA stream if driver version is greater than or equal to 2007 */
    uint32_t driverVersion;
    CUPVA_CHECK_ERROR_GOTO(CupvaGetDriverVersion(&driverVersion), cudaStreamSubmitTestResult, CheckResults);
    if (driverVersion >= 2007)
    {
        /* Submit to CUDA stream */
        cudaStreamSubmitTestResult = SubmitHistogramEqualizationToCudaStream(image_width, image_height, data_type,
                                                                             assetDir[0] != '\0' ? assetDir : NULL);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult = SubmitHistogramEqualizationToCupvaStream(image_width, image_height, data_type,
                                                                           assetDir[0] != '\0' ? assetDir : NULL);

CheckResults:
    /* Overall result */
    if (cudaStreamSubmitTestResult || cupvaStreamSubmitTestResult)
    {
        printf("Test Fail\n");
    }
    else
    {
        printf("Test Pass\n");
    }

    return (cudaStreamSubmitTestResult || cupvaStreamSubmitTestResult);
}
