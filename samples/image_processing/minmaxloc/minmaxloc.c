/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "OpMinMaxLoc.h"
#include "ref/minmaxloc_ref.h"

#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Fwd.h>
#include <nvcv/alloc/Allocator.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Allocate and initialize tensors
 */
static int AllocateAndInitializeTensors(
    unsigned short image_width, unsigned short image_height, int loc_capacity, NVCVDataType data_type,
    bool use_fixed_input, int min_cnt0, int max_cnt0, const char *assetDir, NVCVAllocatorHandle *allocatorHandle,
    NVCVTensorHandle *srcTensorHandle, NVCVTensorHandle *dstMinValTensorHandle, NVCVTensorHandle *dstMaxValTensorHandle,
    NVCVTensorHandle *dstMinCntTensorHandle, NVCVTensorHandle *dstMaxCntTensorHandle,
    NVCVTensorHandle *dstMinLocTensorHandle, NVCVTensorHandle *dstMaxLocTensorHandle,
    NVCVTensorRequirements *srcTensorRequirements)
{
    int32_t err = 0;
    int32_t tensorRank;
    int64_t tensorShape[3];
    NVCVTensorLayout tensorLayout;
    NVCVTensorRequirements dstTensorRequirements;

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);

    // src
    tensorRank     = 3;
    tensorShape[0] = image_height;
    tensorShape[1] = image_width;
    tensorShape[2] = 1;

    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HWC", &tensorLayout), err, Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(tensorRank, tensorShape, data_type, tensorLayout, 0, 0, srcTensorRequirements),
        err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(srcTensorRequirements, *allocatorHandle, srcTensorHandle), err, Error);

    // dst values
    tensorRank     = 1;
    tensorShape[0] = 1;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("W", &tensorLayout), err, Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(tensorRank, tensorShape, data_type, tensorLayout, 0, 0, &dstTensorRequirements),
        err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&dstTensorRequirements, *allocatorHandle, dstMinValTensorHandle), err,
                          Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&dstTensorRequirements, *allocatorHandle, dstMaxValTensorHandle), err,
                          Error);

    // counts
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank, tensorShape, NVCV_DATA_TYPE_S32, tensorLayout, 0, 0,
                                                        &dstTensorRequirements),
                          err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&dstTensorRequirements, *allocatorHandle, dstMinCntTensorHandle), err,
                          Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&dstTensorRequirements, *allocatorHandle, dstMaxCntTensorHandle), err,
                          Error);

    // min/max locations
    tensorShape[0] = loc_capacity;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank, tensorShape, NVCV_DATA_TYPE_2S16, tensorLayout, 0,
                                                        0, &dstTensorRequirements),
                          err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&dstTensorRequirements, *allocatorHandle, dstMinLocTensorHandle), err,
                          Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&dstTensorRequirements, *allocatorHandle, dstMaxLocTensorHandle), err,
                          Error);

    // generate input image data: fixed or random
    if (use_fixed_input)
    {
        generateMinMaxLocFixedImgInputC(srcTensorRequirements, *srcTensorHandle, assetDir);
    }
    else
    {
        generateRandomMinMaxLocInput(srcTensorRequirements, *srcTensorHandle, min_cnt0, max_cnt0);
    }

    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(NVCVTensorRequirements *srcTensorRequirements, NVCVTensorHandle srcTensorHandle,
                          NVCVTensorHandle dstMinValTensorHandle, NVCVTensorHandle dstMinLocTensorHandle,
                          NVCVTensorHandle dstMinCntTensorHandle, NVCVTensorHandle dstMaxValTensorHandle,
                          NVCVTensorHandle dstMaxLocTensorHandle, NVCVTensorHandle dstMaxCntTensorHandle,
                          int loc_capacity, const char *testName)
{
    int32_t err = checkMinMaxLocOutput(srcTensorRequirements, srcTensorHandle, dstMinValTensorHandle,
                                       dstMinLocTensorHandle, dstMinCntTensorHandle, dstMaxValTensorHandle,
                                       dstMaxLocTensorHandle, dstMaxCntTensorHandle, loc_capacity);
    if (err)
    {
        printf("%s test failed\n", testName);
    }
    return err;
}

/**
 * @brief Test minmaxloc operation using CUDA stream
 */
int SubmitMinMaxLocToCudaStream(unsigned short image_width, unsigned short image_height, int loc_capacity,
                                NVCVDataType data_type, bool use_fixed_input, int min_cnt0, int max_cnt0,
                                const char *assetDir)
{
    NVCVAllocatorHandle allocatorHandle    = NULL;
    NVCVTensorHandle srcTensorHandle       = NULL;
    NVCVTensorHandle dstMinValTensorHandle = NULL;
    NVCVTensorHandle dstMaxValTensorHandle = NULL;
    NVCVTensorHandle dstMinCntTensorHandle = NULL;
    NVCVTensorHandle dstMaxCntTensorHandle = NULL;
    NVCVTensorHandle dstMinLocTensorHandle = NULL;
    NVCVTensorHandle dstMaxLocTensorHandle = NULL;
    NVCVOperatorHandle operatorHandle      = NULL;
    cudaStream_t stream                    = NULL;
    int32_t err                            = 0;

    NVCVTensorRequirements srcTensorRequirements;

    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeTensors(image_width, image_height, loc_capacity, data_type, use_fixed_input, min_cnt0,
                                       max_cnt0, assetDir, &allocatorHandle, &srcTensorHandle, &dstMinValTensorHandle,
                                       &dstMaxValTensorHandle, &dstMinCntTensorHandle, &dstMaxCntTensorHandle,
                                       &dstMinLocTensorHandle, &dstMaxLocTensorHandle, &srcTensorRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaMinMaxLocCreate(&operatorHandle, &srcTensorRequirements), err, FreeResources);

    /* Submit to CUDA stream */
    NVCV_CHECK_ERROR_GOTO(
        pvaMinMaxLocSubmit(operatorHandle, stream, srcTensorHandle, dstMinValTensorHandle, dstMinLocTensorHandle,
                           dstMinCntTensorHandle, dstMaxValTensorHandle, dstMaxLocTensorHandle, dstMaxCntTensorHandle),
        err, FreeResources);

    /* Synchronize stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(&srcTensorRequirements, srcTensorHandle, dstMinValTensorHandle, dstMinLocTensorHandle,
                         dstMinCntTensorHandle, dstMaxValTensorHandle, dstMaxLocTensorHandle, dstMaxCntTensorHandle,
                         loc_capacity, "Submit to CUDA stream");

// clean up all allocated resources
FreeResources:
    cudaStreamDestroy(stream);
    nvcvTensorDecRef(srcTensorHandle, NULL);
    nvcvTensorDecRef(dstMinValTensorHandle, NULL);
    nvcvTensorDecRef(dstMaxValTensorHandle, NULL);
    nvcvTensorDecRef(dstMinCntTensorHandle, NULL);
    nvcvTensorDecRef(dstMaxCntTensorHandle, NULL);
    nvcvTensorDecRef(dstMinLocTensorHandle, NULL);
    nvcvTensorDecRef(dstMaxLocTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

/**
 * @brief Test minmaxloc operation using cuPVA stream
 */
int SubmitMinMaxLocToCupvaStream(unsigned short image_width, unsigned short image_height, int loc_capacity,
                                 NVCVDataType data_type, bool use_fixed_input, int min_cnt0, int max_cnt0,
                                 const char *assetDir)
{
    NVCVAllocatorHandle allocatorHandle    = NULL;
    NVCVTensorHandle srcTensorHandle       = NULL;
    NVCVTensorHandle dstMinValTensorHandle = NULL;
    NVCVTensorHandle dstMaxValTensorHandle = NULL;
    NVCVTensorHandle dstMinCntTensorHandle = NULL;
    NVCVTensorHandle dstMaxCntTensorHandle = NULL;
    NVCVTensorHandle dstMinLocTensorHandle = NULL;
    NVCVTensorHandle dstMaxLocTensorHandle = NULL;
    NVCVOperatorHandle operatorHandle      = NULL;
    cupvaSyncObj_t sync                    = NULL;
    cupvaStream_t stream                   = NULL;
    int32_t err                            = 0;

    NVCVTensorRequirements srcTensorRequirements;

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeTensors(image_width, image_height, loc_capacity, data_type, use_fixed_input, min_cnt0,
                                       max_cnt0, assetDir, &allocatorHandle, &srcTensorHandle, &dstMinValTensorHandle,
                                       &dstMaxValTensorHandle, &dstMinCntTensorHandle, &dstMaxCntTensorHandle,
                                       &dstMinLocTensorHandle, &dstMaxLocTensorHandle, &srcTensorRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaMinMaxLocCreate(&operatorHandle, &srcTensorRequirements), err, FreeResources);

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

    NVCV_CHECK_ERROR_GOTO(
        pvaMinMaxLocSubmit(operatorHandle, stream, srcTensorHandle, dstMinValTensorHandle, dstMinLocTensorHandle,
                           dstMinCntTensorHandle, dstMaxValTensorHandle, dstMaxLocTensorHandle, dstMaxCntTensorHandle),
        err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    /* Wait for completion */
    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(&srcTensorRequirements, srcTensorHandle, dstMinValTensorHandle, dstMinLocTensorHandle,
                         dstMinCntTensorHandle, dstMaxValTensorHandle, dstMaxLocTensorHandle, dstMaxCntTensorHandle,
                         loc_capacity, "Submit to cuPVA stream");

// clean up all allocated resources
FreeResources:
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvTensorDecRef(srcTensorHandle, NULL);
    nvcvTensorDecRef(dstMinValTensorHandle, NULL);
    nvcvTensorDecRef(dstMaxValTensorHandle, NULL);
    nvcvTensorDecRef(dstMinCntTensorHandle, NULL);
    nvcvTensorDecRef(dstMaxCntTensorHandle, NULL);
    nvcvTensorDecRef(dstMinLocTensorHandle, NULL);
    nvcvTensorDecRef(dstMaxLocTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char **argv)
{
    /* Parameter settings */
    unsigned short image_width  = 1921;
    unsigned short image_height = 1027;

    int loc_capacity = 5000;

    char data_type_str[32] = "u8";
    NVCVDataType data_type = NVCV_DATA_TYPE_U8;
    int min_cnt0           = 2002;
    int max_cnt0           = 200002;
    bool use_fixed_input   = false;
    char assetDir[256]     = "";

    // load custom parameters from argv
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "-d") || !strcmp(argv[i], "--data_type"))
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
                printf("Error: Invalid data type '%s'\n", argv[i + 1]);
                printf("Supported types: u8, s8, u16, s16, u32, s32, f32\n");
                return 1;
            }
            i++;
        }
        else if (!strcmp(argv[i], "--min_cnt"))
        {
            min_cnt0 = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "--max_cnt"))
        {
            max_cnt0 = atoi(argv[i + 1]);
            i++;
        }
        // use fixed input image
        else if (!strcmp(argv[i], "--fixed"))
        {
            use_fixed_input = true;
            image_width     = 1920;
            image_height    = 1080;
        }
        // get asset directory: use -a/--assets in command line
        else if (!strcmp(argv[i], "-a") || !strcmp(argv[i], "--assets"))
        {
            strncpy(assetDir, argv[i + 1], sizeof(assetDir) - 1);
            assetDir[sizeof(assetDir) - 1] = '\0';
            i++;
        }
    }

    // Example command line parameters:
    // random input test: -d u8 --min_cnt 2000 --max_cnt 20000
    // fixed input image test: -d f32 --fixed -a ASSETS_DIR
    if (use_fixed_input)
    {
        printf("Use fixed input image: image_width=%d, image_height=%d, data_type=%s\n", image_width, image_height,
               data_type_str);
    }
    else
    {
        printf(
            "Use random generated input: image_width=%d, image_height=%d, data_type=%s, min_cnt=%d, "
            "max_cnt=%d\n",
            image_width, image_height, data_type_str, min_cnt0, max_cnt0);
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
        cudaStreamSubmitTestResult =
            SubmitMinMaxLocToCudaStream(image_width, image_height, loc_capacity, data_type, use_fixed_input, min_cnt0,
                                        max_cnt0, assetDir[0] != '\0' ? assetDir : NULL);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult =
        SubmitMinMaxLocToCupvaStream(image_width, image_height, loc_capacity, data_type, use_fixed_input, min_cnt0,
                                     max_cnt0, assetDir[0] != '\0' ? assetDir : NULL);

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
