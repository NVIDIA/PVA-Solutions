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
#include "ref/orb_desc_extractor_ref.h"

#include <ImageIO.h>
#include <OpORBDescExtractor.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Fwd.h>
#include <nvcv/Image.h>
#include <nvcv/alloc/Allocator.h>
#include <stdlib.h>

/**
 * @brief Allocate and initialize resources
 */
static int AllocateAndInitializeResources(uint32_t srcImageWidth, uint32_t srcImageHeight, int32_t numFeatures,
                                          NVCVImageFormat imageFormat, NVCVAllocatorHandle *allocatorHandle,
                                          NVCVImageHandle *imageHandle, NVCVTensorHandle *inCornersTensorHandle,
                                          NVCVTensorHandle *outDescriptorsTensorHandle,
                                          NVCVTensorHandle *refDescriptorsTensorHandle,
                                          NVCVImageRequirements *inImageRequirements,
                                          NVCVTensorRequirements *inCornersTensorRequirements,
                                          NVCVTensorRequirements *outDescriptorsTensorRequirements)
{
    int32_t err = 0;
    int32_t tensorRank;
    int64_t tensorShape[3];
    NVCVTensorLayout tensorLayout;

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(
        nvcvImageCalcRequirementsPva(srcImageWidth, srcImageHeight, imageFormat, 0, 0, inImageRequirements), err,
        Error);
    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(inImageRequirements, *allocatorHandle, imageHandle), err, Error);

    // Input feature tensor
    tensorRank     = 2;
    tensorShape[0] = numFeatures;
    tensorShape[1] = 2;

    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HW", &tensorLayout), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank, tensorShape, NVCV_DATA_TYPE_F32, tensorLayout, 0, 0,
                                                        inCornersTensorRequirements),
                          err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(inCornersTensorRequirements, *allocatorHandle, inCornersTensorHandle),
                          err, Error);

    // Output descriptor structure
    tensorRank     = 2;
    tensorShape[0] = numFeatures;
    tensorShape[1] = PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HW", &tensorLayout), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank, tensorShape, NVCV_DATA_TYPE_U8, tensorLayout, 0, 0,
                                                        outDescriptorsTensorRequirements),
                          err, Error);

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(outDescriptorsTensorRequirements, *allocatorHandle, outDescriptorsTensorHandle), err,
        Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(outDescriptorsTensorRequirements, *allocatorHandle, refDescriptorsTensorHandle), err,
        Error);

    // Generate input
    generateORBDescExtractorRandomInput(*imageHandle, *inCornersTensorHandle);

    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(NVCVImageHandle imageHandle, NVCVTensorHandle inCornersTensorHandle,
                          NVCVTensorHandle refDescriptorsTensorHandle, NVCVTensorHandle outDescriptorsTensorHandle,
                          uint32_t disableRBRIEFFlag, NVCVBorderType borderType, int32_t borderValue,
                          const char *testName)
{
    ORBDescExtractorRef(imageHandle, inCornersTensorHandle, refDescriptorsTensorHandle, disableRBRIEFFlag, borderType,
                        borderValue);

    int32_t err = checkORBDescExtractorOutput(refDescriptorsTensorHandle, outDescriptorsTensorHandle);
    if (err)
    {
        printf("%s test failed\n", testName);
    }
    return err;
}

/**
 * @brief Test ORB descriptor extractor operation using CUDA stream
 */
int SubmitORBDescExtractorToCudaStream(uint32_t srcImageWidth, uint32_t srcImageHeight, int32_t numFeatures,
                                       uint32_t disableRBRIEFFlag, NVCVImageFormat imageFormat,
                                       NVCVBorderType borderType, int32_t borderValue)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle         = NULL;
    NVCVImageHandle imageHandle                 = NULL;
    NVCVTensorHandle inCornersTensorHandle      = NULL;
    NVCVTensorHandle outDescriptorsTensorHandle = NULL;
    NVCVTensorHandle refDescriptorsTensorHandle = NULL;
    NVCVOperatorHandle operatorHandle           = NULL;
    cudaStream_t stream                         = NULL;
    int32_t err                                 = 0;

    NVCVImageRequirements inImageRequirements;
    NVCVTensorRequirements inCornersTensorRequirements;
    NVCVTensorRequirements outDescriptorsTensorRequirements;

    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    /* Allocate and initialize resources */
    err = AllocateAndInitializeResources(srcImageWidth, srcImageHeight, numFeatures, imageFormat, &allocatorHandle,
                                         &imageHandle, &inCornersTensorHandle, &outDescriptorsTensorHandle,
                                         &refDescriptorsTensorHandle, &inImageRequirements,
                                         &inCornersTensorRequirements, &outDescriptorsTensorRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(
        pvaORBDescExtractorCreate(&operatorHandle, &inImageRequirements, &inCornersTensorRequirements,
                                  &outDescriptorsTensorRequirements, disableRBRIEFFlag, borderType, borderValue),
        err, FreeResources);

    /* Submit to CUDA stream */
    NVCV_CHECK_ERROR_GOTO(pvaORBDescExtractorSubmit(operatorHandle, stream, imageHandle, inCornersTensorHandle,
                                                    outDescriptorsTensorHandle),
                          err, FreeResources);

    /* Synchronize stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(imageHandle, inCornersTensorHandle, refDescriptorsTensorHandle, outDescriptorsTensorHandle,
                         disableRBRIEFFlag, borderType, borderValue, "Submit to CUDA stream");

FreeResources: /* clean up all allocated resources */
    cudaStreamDestroy(stream);
    nvcvImageDecRef(imageHandle, NULL);
    nvcvTensorDecRef(inCornersTensorHandle, NULL);
    nvcvTensorDecRef(outDescriptorsTensorHandle, NULL);
    nvcvTensorDecRef(refDescriptorsTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

/**
 * @brief Test ORB descriptor extractor operation using cuPVA stream
 */
int SubmitORBDescExtractorToCupvaStream(uint32_t srcImageWidth, uint32_t srcImageHeight, int32_t numFeatures,
                                        uint32_t disableRBRIEFFlag, NVCVImageFormat imageFormat,
                                        NVCVBorderType borderType, int32_t borderValue)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle         = NULL;
    NVCVImageHandle imageHandle                 = NULL;
    NVCVTensorHandle inCornersTensorHandle      = NULL;
    NVCVTensorHandle outDescriptorsTensorHandle = NULL;
    NVCVTensorHandle refDescriptorsTensorHandle = NULL;
    NVCVOperatorHandle operatorHandle           = NULL;
    cupvaSyncObj_t sync                         = NULL;
    cupvaStream_t stream                        = NULL;
    int32_t err                                 = 0;

    NVCVImageRequirements inImageRequirements;
    NVCVTensorRequirements inCornersTensorRequirements;
    NVCVTensorRequirements outDescriptorsTensorRequirements;

    /* Allocate and initialize resources */
    err = AllocateAndInitializeResources(srcImageWidth, srcImageHeight, numFeatures, imageFormat, &allocatorHandle,
                                         &imageHandle, &inCornersTensorHandle, &outDescriptorsTensorHandle,
                                         &refDescriptorsTensorHandle, &inImageRequirements,
                                         &inCornersTensorRequirements, &outDescriptorsTensorRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(
        pvaORBDescExtractorCreate(&operatorHandle, &inImageRequirements, &inCornersTensorRequirements,
                                  &outDescriptorsTensorRequirements, disableRBRIEFFlag, borderType, borderValue),
        err, FreeResources);

    /* Create cuPVA stream and synchronization objects */
    CUPVA_CHECK_ERROR_GOTO(CupvaSyncObjCreate(&sync, false, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD), err,
                           FreeResources);

    cupvaFence_t fence;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceInit(&fence, sync), err, FreeResources);

    cupvaCmd_t rf;
    CUPVA_CHECK_ERROR_GOTO(CupvaCmdRequestFencesInit(&rf, &fence, 1), err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamCreate(&stream, CUPVA_PVA0, CUPVA_VPU_ANY), err, FreeResources);

    /* Submit to cuPVA stream */
    cupvaCmd_t const *cmds[1]  = {&rf};
    cupvaCmdStatus_t status[1] = {NULL};

    NVCV_CHECK_ERROR_GOTO(pvaORBDescExtractorSubmit(operatorHandle, stream, imageHandle, inCornersTensorHandle,
                                                    outDescriptorsTensorHandle),
                          err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    /* Wait for completion */
    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(imageHandle, inCornersTensorHandle, refDescriptorsTensorHandle, outDescriptorsTensorHandle,
                         disableRBRIEFFlag, borderType, borderValue, "Submit to cuPVA stream");

FreeResources: /* clean up all allocated resources */
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvImageDecRef(imageHandle, NULL);
    nvcvTensorDecRef(inCornersTensorHandle, NULL);
    nvcvTensorDecRef(outDescriptorsTensorHandle, NULL);
    nvcvTensorDecRef(refDescriptorsTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char **argv)
{
    /* Parameter settings */
    NVCVImageFormat imageFormat = NVCV_IMAGE_FORMAT_U8;
    NVCVBorderType borderType   = NVCV_BORDER_REPLICATE;
    int32_t borderValue         = 0;

    uint32_t srcImageWidth     = 960;
    uint32_t srcImageHeight    = 540;
    int32_t numFeatures        = 50;
    uint32_t disableRBRIEFFlag = 0;

    // Parse command-line arguments
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "-f") || !strcmp(argv[i], "--format"))
        {
            if (!strcmp(argv[i + 1], "u8"))
            {
                printf("ORBDescExtractor sample test: format=U8\n");
                imageFormat = NVCV_IMAGE_FORMAT_U8;
            }
            else if (!strcmp(argv[i + 1], "y8"))
            {
                printf("ORBDescExtractor sample test: format=Y8\n");
                imageFormat = NVCV_IMAGE_FORMAT_Y8;
            }
            else if (!strcmp(argv[i + 1], "y8_er"))
            {
                printf("ORBDescExtractor sample test: format=Y8_ER\n");
                imageFormat = NVCV_IMAGE_FORMAT_Y8_ER;
            }
            else if (!strcmp(argv[i + 1], "s8"))
            {
                printf("ORBDescExtractor sample test: format=S8\n");
                imageFormat = NVCV_IMAGE_FORMAT_S8;
            }
            else if (!strcmp(argv[i + 1], "u16"))
            {
                printf("ORBDescExtractor sample test: format=U16\n");
                imageFormat = NVCV_IMAGE_FORMAT_U16;
            }
            else if (!strcmp(argv[i + 1], "y16"))
            {
                printf("ORBDescExtractor sample test: format=Y16\n");
                imageFormat = NVCV_IMAGE_FORMAT_Y16;
            }
            else if (!strcmp(argv[i + 1], "y16_er"))
            {
                printf("ORBDescExtractor sample test: format=Y16_ER\n");
                imageFormat = NVCV_IMAGE_FORMAT_Y16_ER;
            }
            else if (!strcmp(argv[i + 1], "s16"))
            {
                printf("ORBDescExtractor sample test: format=S16\n");
                imageFormat = NVCV_IMAGE_FORMAT_S16;
            }
            else
            {
                printf("ORBDescExtractor sample test: invalid image format!\n");
                return -1;
            }
            i++;
        }
        else if (!strcmp(argv[i], "-b") || !strcmp(argv[i], "--border_type"))
        {
            if (!strcmp(argv[i + 1], "zero"))
            {
                borderType = NVCV_BORDER_CONSTANT;
            }
            else if (!strcmp(argv[i + 1], "clamp"))
            {
                borderType = NVCV_BORDER_REPLICATE;
            }
            else
            {
                printf("ORBDescExtractor sample test: invalid border type!\n");
                return -1;
            }
            i++;
        }
        else if (!strcmp(argv[i], "-nf") || !strcmp(argv[i], "--num_features"))
        {
            numFeatures = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-dr") || !strcmp(argv[i], "--disable_rbrief"))
        {
            disableRBRIEFFlag = 1;
        }
    }

    printf("ORB DESCRIPTOR EXTRACTOR SAMPLE TEST, w=%d, h=%d, numFeatures=%d, disableRBRIEFFlag=%d\n", srcImageWidth,
           srcImageHeight, numFeatures, disableRBRIEFFlag);

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
        cudaStreamSubmitTestResult = SubmitORBDescExtractorToCudaStream(
            srcImageWidth, srcImageHeight, numFeatures, disableRBRIEFFlag, imageFormat, borderType, borderValue);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult = SubmitORBDescExtractorToCupvaStream(
        srcImageWidth, srcImageHeight, numFeatures, disableRBRIEFFlag, imageFormat, borderType, borderValue);

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
