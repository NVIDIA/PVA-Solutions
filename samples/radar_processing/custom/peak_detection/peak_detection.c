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
#include "peak_detection_param.h"
#include "peak_detection_ref.h"

#include <ImageIO.h>
#include <OpPeakDetection.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Fwd.h>
#include <nvcv/Tensor.h>
#include <nvcv/TensorData.h>
#include <nvcv/TensorLayout.h>
#include <nvcv/alloc/Allocator.h>
#include <stdlib.h>

#define MAX_FILENAME_LENGTH (256)

/**
 * @brief Allocate and initialize tensors
 */
static int AllocateAndInitializeTensors(
    int argc, char **argv, int32_t Nb, int32_t Nr, int32_t NofTx, int32_t NofRx, int32_t repeatFold,
    int32_t useRandomInputs, char *filePostfix, NVCVAllocatorHandle *allocatorHandle,
    NVCVTensorHandle *nciFinalTensorHandle, NVCVTensorHandle *nciRxTensorHandle, NVCVTensorHandle *FT2DTensorHandle,
    NVCVTensorHandle *noiseEstTensorHandle, NVCVTensorHandle *refPeakCountTensorHandle,
    NVCVTensorHandle *refPeakIndexTensorHandle, NVCVTensorHandle *refSnapTensorHandle,
    NVCVTensorHandle *pvaPeakCountTensorHandle, NVCVTensorHandle *pvaPeakIndexTensorHandle,
    NVCVTensorHandle *pvaSnapTensorHandle, NVCVTensorRequirements *tensorRequirementsFT2D,
    NVCVTensorRequirements *tensorRequirementsSnap)
{
    int32_t err = 0;
    char assetsDirectory[MAX_FILENAME_LENGTH];
    char filename[MAX_FILENAME_LENGTH];

    if (!useRandomInputs)
    {
        GetAssetsDirectory(argc, argv, assetsDirectory, MAX_FILENAME_LENGTH);
        strncat(assetsDirectory, "/peak_detection", sizeof(assetsDirectory) - strlen(assetsDirectory) - 1);
        assetsDirectory[sizeof(assetsDirectory) - 1] = '\0';
    }

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);
    NVCVTensorLayout layoutHW;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HW", &layoutHW), err, Error);

    // NCI Final tensor
    {
        int64_t tensorShape[] = {Nb, Nr / repeatFold};
        NVCVTensorRequirements tensorRequirements;

        NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(sizeof(tensorShape) / sizeof(tensorShape[0]), tensorShape,
                                                            NVCV_DATA_TYPE_U32, layoutHW, 0, 0, &tensorRequirements),
                              err, Error);

        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&tensorRequirements, *allocatorHandle, nciFinalTensorHandle), err,
                              Error);

        if (useRandomInputs)
        {
            generateRandomPeakDetectionInput(*nciFinalTensorHandle);
        }
        else
        {
            snprintf(filename, sizeof(filename), "NCI_FINAL_%s.bin", filePostfix);
            NVCV_CHECK_ERROR_GOTO(loadTensorInput(*nciFinalTensorHandle, assetsDirectory, filename), err, Error);
        }
    }

    // NCI Rx tensor
    {
        int64_t tensorShape[] = {Nb, Nr};
        NVCVTensorRequirements tensorRequirements;

        NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(sizeof(tensorShape) / sizeof(tensorShape[0]), tensorShape,
                                                            NVCV_DATA_TYPE_U32, layoutHW, 0, 0, &tensorRequirements),
                              err, Error);

        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&tensorRequirements, *allocatorHandle, nciRxTensorHandle), err,
                              Error);

        if (useRandomInputs)
        {
            generateRandomPeakDetectionInput(*nciRxTensorHandle);
        }
        else
        {
            snprintf(filename, sizeof(filename), "NCI_RX_%s.bin", filePostfix);
            NVCV_CHECK_ERROR_GOTO(loadTensorInput(*nciRxTensorHandle, assetsDirectory, filename), err, Error);
        }
    }

    // Noise estimate tensor
    {
        NVCVTensorLayout layout;
        NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("W", &layout), err, Error);

        int64_t tensorShape[] = {Nb};
        NVCVTensorRequirements tensorRequirements;

        NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(sizeof(tensorShape) / sizeof(tensorShape[0]), tensorShape,
                                                            NVCV_DATA_TYPE_U32, layout, 0, 0, &tensorRequirements),
                              err, Error);

        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&tensorRequirements, *allocatorHandle, noiseEstTensorHandle), err,
                              Error);

        if (useRandomInputs)
        {
            generateRandomPeakDetectionInput(*noiseEstTensorHandle);
        }
        else
        {
            snprintf(filename, sizeof(filename), "NoiseEstimation_%s.bin", filePostfix);
            NVCV_CHECK_ERROR_GOTO(loadTensorInput(*noiseEstTensorHandle, assetsDirectory, filename), err, Error);
        }
    }

    // FT2D tensor
    {
        NVCVTensorLayout layout;
        NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HCW", &layout), err, Error);

        int64_t tensorShape[] = {Nb, NofRx, Nr};
        NVCVTensorRequirements tensorRequirements;

        NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(sizeof(tensorShape) / sizeof(tensorShape[0]), tensorShape,
                                                            NVCV_DATA_TYPE_2S32, layout, 0, 0, &tensorRequirements),
                              err, Error);

        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&tensorRequirements, *allocatorHandle, FT2DTensorHandle), err, Error);

        if (useRandomInputs)
        {
            generateRandomPeakDetectionInput(*FT2DTensorHandle);
        }
        else
        {
            snprintf(filename, sizeof(filename), "FT2D_%s.bin", filePostfix);
            NVCV_CHECK_ERROR_GOTO(loadTensorInput(*FT2DTensorHandle, assetsDirectory, filename), err, Error);
        }

        *tensorRequirementsFT2D = tensorRequirements;
    }

    // Peak count tensors
    {
        NVCVTensorLayout layout;
        NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("W", &layout), err, Error);

        int64_t tensorShape[] = {1};
        NVCVTensorRequirements tensorRequirements;

        NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(sizeof(tensorShape) / sizeof(tensorShape[0]), tensorShape,
                                                            NVCV_DATA_TYPE_U32, layout, 0, 0, &tensorRequirements),
                              err, Error);

        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&tensorRequirements, *allocatorHandle, refPeakCountTensorHandle), err,
                              Error);
        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&tensorRequirements, *allocatorHandle, pvaPeakCountTensorHandle), err,
                              Error);
    }

    // Peak index tensors
    {
        NVCVTensorLayout layout;
        NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HW", &layout), err, Error);

        int64_t tensorShape[] = {3, PEAKDET_MAX_PEAK_COUNT};
        NVCVTensorRequirements tensorRequirements;

        NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(sizeof(tensorShape) / sizeof(tensorShape[0]), tensorShape,
                                                            NVCV_DATA_TYPE_U32, layout, 0, 0, &tensorRequirements),
                              err, Error);

        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&tensorRequirements, *allocatorHandle, refPeakIndexTensorHandle), err,
                              Error);
        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&tensorRequirements, *allocatorHandle, pvaPeakIndexTensorHandle), err,
                              Error);
    }

    // Snap tensors
    {
        NVCVTensorLayout layout;
        NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("WH", &layout), err, Error);

        int64_t tensorShape[] = {PEAKDET_MAX_PEAK_COUNT, NofTx * NofRx};
        NVCVTensorRequirements tensorRequirements;

        NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(sizeof(tensorShape) / sizeof(tensorShape[0]), tensorShape,
                                                            NVCV_DATA_TYPE_2S32, layout, 0, 0, &tensorRequirements),
                              err, Error);
        *tensorRequirementsSnap = tensorRequirements;

        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&tensorRequirements, *allocatorHandle, refSnapTensorHandle), err,
                              Error);
        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&tensorRequirements, *allocatorHandle, pvaSnapTensorHandle), err,
                              Error);
    }

    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(NVCVTensorHandle *inHandles, NVCVTensorHandle *refOutHandles, NVCVTensorHandle *pvaOutHandles,
                          int32_t NofTx, int32_t NofRx, int32_t repeatFold, float maxErrRatio, const char *testName)
{
    peakDetectionRef(inHandles, NofTx, repeatFold, refOutHandles);

    int32_t err = checkPeakDetectionOutput(NofTx, NofRx, refOutHandles[0], refOutHandles[1], refOutHandles[2],
                                           pvaOutHandles[0], pvaOutHandles[1], pvaOutHandles[2], maxErrRatio);
    if (err)
    {
        printf("%s test failed\n", testName);
    }
    return err;
}

/**
 * @brief Test peak detection operation using CUDA stream
 */
int SubmitPeakDetectionToCudaStream(int argc, char **argv, int32_t Nb, int32_t Nr, int32_t NofTx, int32_t NofRx,
                                    int32_t repeatFold, int32_t useRandomInputs, char *filePostfix, float maxErrRatio)
{
    NVCVAllocatorHandle allocatorHandle       = NULL;
    NVCVTensorHandle nciFinalTensorHandle     = NULL;
    NVCVTensorHandle nciRxTensorHandle        = NULL;
    NVCVTensorHandle FT2DTensorHandle         = NULL;
    NVCVTensorHandle noiseEstTensorHandle     = NULL;
    NVCVTensorHandle refPeakCountTensorHandle = NULL;
    NVCVTensorHandle refPeakIndexTensorHandle = NULL;
    NVCVTensorHandle refSnapTensorHandle      = NULL;
    NVCVTensorHandle pvaPeakCountTensorHandle = NULL;
    NVCVTensorHandle pvaPeakIndexTensorHandle = NULL;
    NVCVTensorHandle pvaSnapTensorHandle      = NULL;

    NVCVOperatorHandle operatorHandle = NULL;
    cudaStream_t stream               = NULL;
    int32_t err                       = 0;

    NVCVTensorRequirements tensorRequirementsFT2D;
    NVCVTensorRequirements tensorRequirementsSnap;

    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeTensors(argc, argv, Nb, Nr, NofTx, NofRx, repeatFold, useRandomInputs, filePostfix,
                                       &allocatorHandle, &nciFinalTensorHandle, &nciRxTensorHandle, &FT2DTensorHandle,
                                       &noiseEstTensorHandle, &refPeakCountTensorHandle, &refPeakIndexTensorHandle,
                                       &refSnapTensorHandle, &pvaPeakCountTensorHandle, &pvaPeakIndexTensorHandle,
                                       &pvaSnapTensorHandle, &tensorRequirementsFT2D, &tensorRequirementsSnap);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(
        pvaPeakDetectionCreate(&operatorHandle, &tensorRequirementsFT2D, &tensorRequirementsSnap, NofTx, repeatFold),
        err, FreeResources);

    /* Submit to CUDA stream */
    NVCVTensorHandle inHandles[] = {FT2DTensorHandle, nciRxTensorHandle, nciFinalTensorHandle, noiseEstTensorHandle};
    NVCVTensorHandle pvaOutHandles[] = {pvaPeakCountTensorHandle, pvaPeakIndexTensorHandle, pvaSnapTensorHandle};
    NVCV_CHECK_ERROR_GOTO(pvaPeakDetectionSubmit(operatorHandle, stream, inHandles, pvaOutHandles, 4, 3), err,
                          FreeResources);

    /* Synchronize stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    /* Validate output */
    NVCVTensorHandle refOutHandles[] = {refPeakCountTensorHandle, refPeakIndexTensorHandle, refSnapTensorHandle};
    err = ValidateOutput(inHandles, refOutHandles, pvaOutHandles, NofTx, NofRx, repeatFold, maxErrRatio,
                         "Submit to CUDA stream");
    if (err)
    {
        printf("Submit to CUDA stream test failed\n");
    }

FreeResources: /* clean up all allocated resources */
    cudaStreamDestroy(stream);
    nvcvOperatorDestroy(operatorHandle);
    nvcvTensorDecRef(nciFinalTensorHandle, NULL);
    nvcvTensorDecRef(nciRxTensorHandle, NULL);
    nvcvTensorDecRef(FT2DTensorHandle, NULL);
    nvcvTensorDecRef(noiseEstTensorHandle, NULL);
    nvcvTensorDecRef(refPeakCountTensorHandle, NULL);
    nvcvTensorDecRef(refPeakIndexTensorHandle, NULL);
    nvcvTensorDecRef(refSnapTensorHandle, NULL);
    nvcvTensorDecRef(pvaPeakCountTensorHandle, NULL);
    nvcvTensorDecRef(pvaPeakIndexTensorHandle, NULL);
    nvcvTensorDecRef(pvaSnapTensorHandle, NULL);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

/**
 * @brief Test peak detection operation using cuPVA stream
 */
int SubmitPeakDetectionToCupvaStream(int argc, char **argv, int32_t Nb, int32_t Nr, int32_t NofTx, int32_t NofRx,
                                     int32_t repeatFold, int32_t useRandomInputs, char *filePostfix, float maxErrRatio)
{
    NVCVAllocatorHandle allocatorHandle       = NULL;
    NVCVTensorHandle nciFinalTensorHandle     = NULL;
    NVCVTensorHandle nciRxTensorHandle        = NULL;
    NVCVTensorHandle FT2DTensorHandle         = NULL;
    NVCVTensorHandle noiseEstTensorHandle     = NULL;
    NVCVTensorHandle refPeakCountTensorHandle = NULL;
    NVCVTensorHandle refPeakIndexTensorHandle = NULL;
    NVCVTensorHandle refSnapTensorHandle      = NULL;
    NVCVTensorHandle pvaPeakCountTensorHandle = NULL;
    NVCVTensorHandle pvaPeakIndexTensorHandle = NULL;
    NVCVTensorHandle pvaSnapTensorHandle      = NULL;

    NVCVOperatorHandle operatorHandle = NULL;
    int32_t err                       = 0;
    cupvaStream_t stream              = NULL;
    cupvaSyncObj_t sync               = NULL;

    NVCVTensorRequirements tensorRequirementsFT2D;
    NVCVTensorRequirements tensorRequirementsSnap;

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeTensors(argc, argv, Nb, Nr, NofTx, NofRx, repeatFold, useRandomInputs, filePostfix,
                                       &allocatorHandle, &nciFinalTensorHandle, &nciRxTensorHandle, &FT2DTensorHandle,
                                       &noiseEstTensorHandle, &refPeakCountTensorHandle, &refPeakIndexTensorHandle,
                                       &refSnapTensorHandle, &pvaPeakCountTensorHandle, &pvaPeakIndexTensorHandle,
                                       &pvaSnapTensorHandle, &tensorRequirementsFT2D, &tensorRequirementsSnap);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(
        pvaPeakDetectionCreate(&operatorHandle, &tensorRequirementsFT2D, &tensorRequirementsSnap, NofTx, repeatFold),
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

    NVCVTensorHandle inHandles[] = {FT2DTensorHandle, nciRxTensorHandle, nciFinalTensorHandle, noiseEstTensorHandle};
    NVCVTensorHandle pvaOutHandles[] = {pvaPeakCountTensorHandle, pvaPeakIndexTensorHandle, pvaSnapTensorHandle};
    NVCV_CHECK_ERROR_GOTO(pvaPeakDetectionSubmit(operatorHandle, stream, inHandles, pvaOutHandles, 4, 3), err,
                          FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    /* Wait for completion */
    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    if (!waitSuccess)
    {
        printf("PVA execution did not complete within timeout\n");
        err = -1;
        goto FreeResources;
    }

    /* Validate output */
    NVCVTensorHandle refOutHandles[] = {refPeakCountTensorHandle, refPeakIndexTensorHandle, refSnapTensorHandle};
    err = ValidateOutput(inHandles, refOutHandles, pvaOutHandles, NofTx, NofRx, repeatFold, maxErrRatio,
                         "Submit to cuPVA stream");
    if (err)
    {
        printf("Submit to cuPVA stream test failed\n");
    }

FreeResources: /* clean up all allocated resources */
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvOperatorDestroy(operatorHandle);
    nvcvTensorDecRef(nciFinalTensorHandle, NULL);
    nvcvTensorDecRef(nciRxTensorHandle, NULL);
    nvcvTensorDecRef(FT2DTensorHandle, NULL);
    nvcvTensorDecRef(noiseEstTensorHandle, NULL);
    nvcvTensorDecRef(refPeakCountTensorHandle, NULL);
    nvcvTensorDecRef(refPeakIndexTensorHandle, NULL);
    nvcvTensorDecRef(refSnapTensorHandle, NULL);
    nvcvTensorDecRef(pvaPeakCountTensorHandle, NULL);
    nvcvTensorDecRef(pvaPeakIndexTensorHandle, NULL);
    nvcvTensorDecRef(pvaSnapTensorHandle, NULL);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char **argv)
{
    /* Parameter settings */
    char *filePostfix       = NULL;
    int32_t useRandomInputs = 0;
    const float maxErrRatio = 0.05f;

    if (argc == 1)
    {
        // Use random input, for local tests only.
        useRandomInputs = 1;
    }
    else if (argc == 3)
    {
        filePostfix = "1";
    }
    else if (argc == 4)
    {
        filePostfix = argv[3];
    }
    else
    {
        printf("Usage: %s or %s --assets <assets directory> or %s --assets <assets directory> <filename postfix>\n",
               argv[0], argv[0], argv[0]);
        return -1;
    }
    if (useRandomInputs)
    {
        printf("Using random input, ");
    }

    const int32_t Nb         = 224;
    const int32_t Nr         = 512;
    const int32_t NofTx      = 4;
    const int32_t NofRx      = 4;
    const int32_t repeatFold = 8;
    printf(
        "number of range bins = %d, number of Doppler bins = %d, number of Tx = %d, number of Rx = %d, repeat fold = "
        "%d\n",
        Nb, Nr, NofTx, NofRx, repeatFold);

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
        cudaStreamSubmitTestResult = SubmitPeakDetectionToCudaStream(argc, argv, Nb, Nr, NofTx, NofRx, repeatFold,
                                                                     useRandomInputs, filePostfix, maxErrRatio);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult = SubmitPeakDetectionToCupvaStream(argc, argv, Nb, Nr, NofTx, NofRx, repeatFold,
                                                                   useRandomInputs, filePostfix, maxErrRatio);

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
