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
#include "ref/image_flip_ref.h"

#include <OpImageFlip.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Fwd.h>
#include <nvcv/Image.h>
#include <nvcv/alloc/Allocator.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Allocate and initialize images
 */
static int AllocateAndInitializeImages(int32_t image_width, int32_t image_height, NVCVImageFormat imageFormat,
                                       const char *assetDir, NVCVAllocatorHandle *allocatorHandle,
                                       NVCVImageHandle *inImageHandle, NVCVImageHandle *outImageHandle,
                                       NVCVImageHandle *refImageHandle, NVCVImageRequirements *imageRequirements)
{
    int32_t err = 0;

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvImageCalcRequirementsPva(image_width, image_height, imageFormat, 0, 0, imageRequirements),
                          err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(imageRequirements, *allocatorHandle, inImageHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(imageRequirements, *allocatorHandle, outImageHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(imageRequirements, *allocatorHandle, refImageHandle), err, Error);

    // Generate and load input from a PNG file available in assets or generate random input
    if (assetDir != NULL && assetDir[0] != '\0')
    {
        generateImageFlipFixedImgInput(imageRequirements, *inImageHandle, assetDir);
    }
    else
    {
        generateImageFlipRandomInput(imageRequirements, *inImageHandle);
    }

    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(NVCVImageRequirements *imageRequirements, NVCVImageHandle inImageHandle,
                          NVCVImageHandle refImageHandle, NVCVImageHandle outImageHandle,
                          PVAFlipDirection flipDirection, int32_t image_width, int32_t image_height,
                          const char *testName)
{
    getImageFlipRefData(imageRequirements, inImageHandle, refImageHandle, flipDirection);

    int32_t err = checkImageFlipOutput(imageRequirements, inImageHandle, refImageHandle, outImageHandle);
    if (err)
    {
        printf("%s test failed\n", testName);
    }

    // Write output files for visual check
    saveImageFlipOutput(outImageHandle, "out_image_flip.png", image_width, image_height);
    saveImageFlipOutput(refImageHandle, "ref_image_flip.png", image_width, image_height);

    return err;
}

/**
 * @brief Test image flip operation using CUDA stream
 */
int SubmitImageFlipToCudaStream(int32_t image_width, int32_t image_height, NVCVImageFormat imageFormat,
                                const char *assetDir, PVAFlipDirection flipDirection)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVImageHandle inImageHandle       = NULL;
    NVCVImageHandle outImageHandle      = NULL;
    NVCVImageHandle refImageHandle      = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cudaStream_t stream                 = NULL;
    int32_t err                         = 0;

    NVCVImageRequirements imageRequirements;

    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    /* Allocate and initialize images */
    err = AllocateAndInitializeImages(image_width, image_height, imageFormat, assetDir, &allocatorHandle,
                                      &inImageHandle, &outImageHandle, &refImageHandle, &imageRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaImageFlipCreate(&operatorHandle, &imageRequirements), err, FreeResources);

    /* Submit to CUDA stream */
    NVCV_CHECK_ERROR_GOTO(pvaImageFlipSubmit(operatorHandle, stream, inImageHandle, outImageHandle, flipDirection), err,
                          FreeResources);

    /* Synchronize stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(&imageRequirements, inImageHandle, refImageHandle, outImageHandle, flipDirection, image_width,
                         image_height, "Submit to CUDA stream");

    // clean up all allocated resources
FreeResources:
    cudaStreamDestroy(stream);
    nvcvImageDecRef(inImageHandle, NULL);
    nvcvImageDecRef(outImageHandle, NULL);
    nvcvImageDecRef(refImageHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

/**
 * @brief Test image flip operation using cuPVA stream
 */
int SubmitImageFlipToCupvaStream(int32_t image_width, int32_t image_height, NVCVImageFormat imageFormat,
                                 const char *assetDir, PVAFlipDirection flipDirection)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVImageHandle inImageHandle       = NULL;
    NVCVImageHandle outImageHandle      = NULL;
    NVCVImageHandle refImageHandle      = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cupvaSyncObj_t sync                 = NULL;
    cupvaStream_t stream                = NULL;
    int32_t err                         = 0;

    NVCVImageRequirements imageRequirements;

    /* Allocate and initialize images */
    err = AllocateAndInitializeImages(image_width, image_height, imageFormat, assetDir, &allocatorHandle,
                                      &inImageHandle, &outImageHandle, &refImageHandle, &imageRequirements);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaImageFlipCreate(&operatorHandle, &imageRequirements), err, FreeResources);

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

    NVCV_CHECK_ERROR_GOTO(pvaImageFlipSubmit(operatorHandle, stream, inImageHandle, outImageHandle, flipDirection), err,
                          FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    /* Wait for completion */
    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(&imageRequirements, inImageHandle, refImageHandle, outImageHandle, flipDirection, image_width,
                         image_height, "Submit to cuPVA stream");

    // clean up all allocated resources
FreeResources:
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvImageDecRef(inImageHandle, NULL);
    nvcvImageDecRef(outImageHandle, NULL);
    nvcvImageDecRef(refImageHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char **argv)
{
    /* Parameter settings */
    int32_t image_width  = 1920;
    int32_t image_height = 1280;

    NVCVImageFormat imageFormat = NVCV_IMAGE_FORMAT_U8; // Default to U8
    char format_str[32]         = "u8";
    char assetDir[256]          = "";
    PVAFlipDirection flipDirection;
    char flip_str[32];
    bool flipDirectionSet = false;

    // load custom parameters from argv
    for (int i = 1; i < argc; i++)
    {
        // get image width: use -w/--width in command line
        if (!strcmp(argv[i], "-w") || !strcmp(argv[i], "--width"))
        {
            image_width = atoi(argv[i + 1]);
            i++;
        }
        // get image height: use -h/--height in command line
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--height"))
        {
            image_height = atoi(argv[i + 1]);
            i++;
        }
        // get image format: use -f/--format in command line
        else if (!strcmp(argv[i], "-f") || !strcmp(argv[i], "--format"))
        {
            strncpy(format_str, argv[i + 1], sizeof(format_str) - 1);
            format_str[sizeof(format_str) - 1] = '\0';
            if (!strcmp(argv[i + 1], "u8"))
            {
                imageFormat = NVCV_IMAGE_FORMAT_U8;
            }
            else if (!strcmp(argv[i + 1], "y8_er"))
            {
                imageFormat = NVCV_IMAGE_FORMAT_Y8_ER;
            }
            else
            {
                printf("Error: Unsupported format '%s'. Supported formats: u8, y8_er\n", argv[i + 1]);
                return 1;
            }
            i++;
        }
        // get flip direction: use -d/--direction in command line
        else if (!strcmp(argv[i], "-d") || !strcmp(argv[i], "--direction"))
        {
            strncpy(flip_str, argv[i + 1], sizeof(flip_str) - 1);
            flip_str[sizeof(flip_str) - 1] = '\0';
            if (!strcmp(argv[i + 1], "horizontal") || !strcmp(argv[i + 1], "h"))
            {
                flipDirection    = PVA_FLIP_HORIZONTAL;
                flipDirectionSet = true;
            }
            else if (!strcmp(argv[i + 1], "vertical") || !strcmp(argv[i + 1], "v"))
            {
                flipDirection    = PVA_FLIP_VERTICAL;
                flipDirectionSet = true;
            }
            else if (!strcmp(argv[i + 1], "both") || !strcmp(argv[i + 1], "b"))
            {
                flipDirection    = PVA_FLIP_BOTH;
                flipDirectionSet = true;
            }
            else
            {
                printf("Error: Unsupported flip direction '%s'. Supported: horizontal (h), vertical (v), both (b)\n",
                       argv[i + 1]);
                return 1;
            }
            i++;
        }
        // show help
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-help"))
        {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  -w, --width <width>         Image width (default: 1920)\n");
            printf("  -h, --height <height>       Image height (default: 1280)\n");
            printf("  -f, --format <format>       Image format: u8, y8_er (default: u8)\n");
            printf("  -d, --direction <direction> Flip direction: horizontal(h), vertical(v), both(b) [REQUIRED]\n");
            printf("  -a, --assets <directory>    Asset directory for test images (optional)\n");
            printf("  --help                      Show this help message\n");
            printf("\nExamples:\n");
            printf("  %s -d horizontal -w 640 -h 480\n", argv[0]);
            printf("  %s -d vertical -f y8_er -a /test_images/\n", argv[0]);
            printf("  %s -d both -w 1920 -h 1280\n", argv[0]);
            return 0;
        }
        // get asset directory: use -a/--assets in command line
        else if (!strcmp(argv[i], "-a") || !strcmp(argv[i], "--assets"))
        {
            strncpy(assetDir, argv[i + 1], sizeof(assetDir) - 1);
            assetDir[sizeof(assetDir) - 1] = '\0';
            i++;
        }
    }

    // Validate required parameters
    if (!flipDirectionSet)
    {
        printf("Error: Flip direction is required. Use -d/--direction to specify.\n");
        printf("Supported directions: horizontal (h), vertical (v), both (b)\n");
        printf("Use --help for more information.\n");
        return 1;
    }

    printf("Running Image Flip sample with:\n");
    printf("  Image size: %dx%d\n", image_width, image_height);
    printf("  Image format: %s\n", format_str);
    printf("  Flip direction: %s\n", flip_str);
    if (assetDir[0] != '\0')
    {
        printf("  Asset directory: %s\n", assetDir);
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
        cudaStreamSubmitTestResult = SubmitImageFlipToCudaStream(image_width, image_height, imageFormat,
                                                                 assetDir[0] != '\0' ? assetDir : NULL, flipDirection);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult = SubmitImageFlipToCupvaStream(image_width, image_height, imageFormat,
                                                               assetDir[0] != '\0' ? assetDir : NULL, flipDirection);

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
