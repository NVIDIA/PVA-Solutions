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
#ifdef REF_ONLY /* provided by the build system */
#    define NCI_REF_ONLY
#endif

#include "ref/nci_ref.h"

#include <ErrorCheckMacros.h>
#include <ImageIO.h>

#ifndef NCI_REF_ONLY
#    include <OpNci.h>
#endif
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <limits.h>
#include <math.h>
#include <nvcv/Fwd.h>
#include <nvcv/alloc/Allocator.h>

/**
 * @brief Allocate and initialize tensors
 */
static int AllocateAndInitializeTensors(int argc, char **argv, int32_t Nr, int32_t NofRx, int32_t Ns,
                                        int32_t numDopplerFolds, NVCVAllocatorHandle *allocatorHandle,
                                        NVCVTensorHandle *inFt2dTensorHandle, NVCVTensorHandle *outTensorHandles,
                                        NVCVTensorHandle *refTensorHandles,
                                        NVCVTensorRequirements *inTensorRequirements,
                                        NVCVTensorRequirements *outTensorRequirements, char *assetsDirectory,
                                        int32_t maxPathLength, PVAFFTLayout inputLayout)
{
    int32_t err                           = 0;
    int64_t outNciRxTensorShape[]         = {Nr, Ns};
    int64_t outNciFinalTensorShape[]      = {Nr, Ns / numDopplerFolds};
    int64_t outNoiseEstimateTensorShape[] = {Nr};

    NVCVTensorLayout inTensorLayout;
    int64_t inTensorShape[] = {Nr, NofRx, Ns};
    if (inputLayout == PVA_FFT_LAYOUT_RANGE_DOPPLER_RX)
    {
        inTensorShape[1] = Ns;
        inTensorShape[2] = NofRx;
    }

    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HCW", &inTensorLayout), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);

    // Construct the input FT2D tensor
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(sizeof(inTensorShape) / sizeof(inTensorShape[0]), inTensorShape,
                                      NVCV_DATA_TYPE_2S32, inTensorLayout, 0, 0, inTensorRequirements),
        err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(inTensorRequirements, *allocatorHandle, inFt2dTensorHandle), err, Error);

    // Construct the output and ref NCI_RX tensors
    NVCVTensorLayout outNciRxTensorLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HW", &outNciRxTensorLayout), err, Error);

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(sizeof(outNciRxTensorShape) / sizeof(outNciRxTensorShape[0]), outNciRxTensorShape,
                                      NVCV_DATA_TYPE_U32, outNciRxTensorLayout, 0, 0, &outTensorRequirements[0]),
        err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&outTensorRequirements[0], *allocatorHandle, &outTensorHandles[0]), err,
                          Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&outTensorRequirements[0], *allocatorHandle, &refTensorHandles[0]), err,
                          Error);

    // Construct the output and ref NCI_FINAL tensors
    NVCVTensorLayout outNciFinalTensorLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HW", &outNciFinalTensorLayout), err, Error);

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(sizeof(outNciFinalTensorShape) / sizeof(outNciFinalTensorShape[0]),
                                      outNciFinalTensorShape, NVCV_DATA_TYPE_U32, outNciFinalTensorLayout, 0, 0,
                                      &outTensorRequirements[1]),
        err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&outTensorRequirements[1], *allocatorHandle, &outTensorHandles[1]), err,
                          Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&outTensorRequirements[1], *allocatorHandle, &refTensorHandles[1]), err,
                          Error);

    // Construct the output and ref NOISE_ESTIMATE tensors
    NVCVTensorLayout outNoiseEstimateTensorLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("W", &outNoiseEstimateTensorLayout), err, Error);

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(sizeof(outNoiseEstimateTensorShape) / sizeof(outNoiseEstimateTensorShape[0]),
                                      outNoiseEstimateTensorShape, NVCV_DATA_TYPE_U32, outNoiseEstimateTensorLayout, 0,
                                      0, &outTensorRequirements[2]),
        err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&outTensorRequirements[2], *allocatorHandle, &outTensorHandles[2]), err,
                          Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&outTensorRequirements[2], *allocatorHandle, &refTensorHandles[2]), err,
                          Error);

    generateRandomNciInput(*inFt2dTensorHandle, 16);
    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(NVCVTensorHandle *inFt2dTensorHandle, NVCVTensorHandle *refTensorHandles,
                          NVCVTensorHandle *outTensorHandles, int32_t numDopplerFolds, PVARadarNCIParams *nciParams,
                          uint32_t nciRxTolerance, uint32_t nciFinalTolerance, uint32_t noiseEstimateTolerance,
                          const char *testName)
{
    int32_t err = nciRef(inFt2dTensorHandle, refTensorHandles, 1, 3, numDopplerFolds, nciParams->inputLayout);
    if (err)
    {
        printf("%s test failed: NCI reference function failed\n", testName);
        return err;
    }

    // Compare the output to reference
    err = checkNCIOutput(refTensorHandles[0], outTensorHandles[0], nciRxTolerance);
    err |= checkNCIOutput(refTensorHandles[1], outTensorHandles[1], nciFinalTolerance);
    if (nciParams->noiseEstimationEnabled == true)
    {
        err |= checkNCIOutput(refTensorHandles[2], outTensorHandles[2], noiseEstimateTolerance);
    }
    if (err)
    {
        printf("%s test failed\n", testName);
    }
    return err;
}

/**
 * @brief Test NCI operation using CUDA stream
 */
int SubmitNCIToCudaStream(int argc, char **argv, int32_t Nr, int32_t NofRx, int32_t Ns, int32_t numDopplerFolds,
                          PVARadarNCIParams *nciParams, uint32_t nciRxTolerance, uint32_t nciFinalTolerance,
                          uint32_t noiseEstimateTolerance)
{
    const int32_t maxPathLength = 256;
    char assetsDirectory[maxPathLength];

    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle  = NULL;
    NVCVTensorHandle inFt2dTensorHandle  = NULL;
    NVCVTensorHandle outTensorHandles[3] = {NULL, NULL, NULL};
    NVCVTensorHandle refTensorHandles[3] = {NULL, NULL, NULL};

#ifndef NCI_REF_ONLY
    NVCVOperatorHandle operatorHandle = NULL;
    cudaStream_t stream               = NULL;
#endif
    int32_t err = 0;

    NVCVTensorRequirements inTensorRequirements;
    NVCVTensorRequirements outTensorRequirements[3];

#ifndef NCI_REF_ONLY
    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);
#endif

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeTensors(argc, argv, Nr, NofRx, Ns, numDopplerFolds, &allocatorHandle,
                                       &inFt2dTensorHandle, outTensorHandles, refTensorHandles, &inTensorRequirements,
                                       outTensorRequirements, assetsDirectory, maxPathLength, nciParams->inputLayout);
    if (err)
        goto FreeResources;

    /* Create an array of pointers for outTensorRequirements */
    const NVCVTensorRequirements *outTensorRequirementsPointers[3] = {
        &outTensorRequirements[0], &outTensorRequirements[1], &outTensorRequirements[2]};

#ifndef NCI_REF_ONLY
    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(
        pvaNciCreate(&operatorHandle, &inTensorRequirements, outTensorRequirementsPointers, nciParams), err,
        FreeResources);

    /* Submit to CUDA stream */
    NVCV_CHECK_ERROR_GOTO(pvaNciSubmit(operatorHandle, stream, &inFt2dTensorHandle, outTensorHandles), err,
                          FreeResources);

    /* Synchronize stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);
#endif

    /* Validate output */
    err = ValidateOutput(&inFt2dTensorHandle, refTensorHandles, outTensorHandles, numDopplerFolds, nciParams,
                         nciRxTolerance, nciFinalTolerance, noiseEstimateTolerance, "Submit to CUDA stream");

FreeResources: /* clean up all allocated resources */
#ifndef NCI_REF_ONLY
    cudaStreamDestroy(stream);
    nvcvOperatorDestroy(operatorHandle);
#endif
    nvcvTensorDecRef(inFt2dTensorHandle, NULL);
    nvcvTensorDecRef(outTensorHandles[0], NULL);
    nvcvTensorDecRef(outTensorHandles[1], NULL);
    nvcvTensorDecRef(outTensorHandles[2], NULL);
    nvcvTensorDecRef(refTensorHandles[0], NULL);
    nvcvTensorDecRef(refTensorHandles[1], NULL);
    nvcvTensorDecRef(refTensorHandles[2], NULL);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

/**
 * @brief Test NCI operation using cuPVA stream
 */
int SubmitNCIToCupvaStream(int argc, char **argv, int32_t Nr, int32_t NofRx, int32_t Ns, int32_t numDopplerFolds,
                           PVARadarNCIParams *nciParams, uint32_t nciRxTolerance, uint32_t nciFinalTolerance,
                           uint32_t noiseEstimateTolerance)
{
    const int32_t maxPathLength = 256;
    char assetsDirectory[maxPathLength];

    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle  = NULL;
    NVCVTensorHandle inFt2dTensorHandle  = NULL;
    NVCVTensorHandle outTensorHandles[3] = {NULL, NULL, NULL};
    NVCVTensorHandle refTensorHandles[3] = {NULL, NULL, NULL};

#ifndef NCI_REF_ONLY
    NVCVOperatorHandle operatorHandle = NULL;
    cupvaSyncObj_t sync               = NULL;
    cupvaStream_t stream              = NULL;
#endif
    int32_t err = 0;

    NVCVTensorRequirements inTensorRequirements;
    NVCVTensorRequirements outTensorRequirements[3];

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeTensors(argc, argv, Nr, NofRx, Ns, numDopplerFolds, &allocatorHandle,
                                       &inFt2dTensorHandle, outTensorHandles, refTensorHandles, &inTensorRequirements,
                                       outTensorRequirements, assetsDirectory, maxPathLength, nciParams->inputLayout);
    if (err)
        goto FreeResources;

    /* Create an array of pointers for outTensorRequirements */
    const NVCVTensorRequirements *outTensorRequirementsPointers[3] = {
        &outTensorRequirements[0], &outTensorRequirements[1], &outTensorRequirements[2]};

#ifndef NCI_REF_ONLY
    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(
        pvaNciCreate(&operatorHandle, &inTensorRequirements, outTensorRequirementsPointers, nciParams), err,
        FreeResources);

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

    NVCV_CHECK_ERROR_GOTO(pvaNciSubmit(operatorHandle, stream, &inFt2dTensorHandle, outTensorHandles), err,
                          FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    /* Wait for completion */
    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);
#endif

    /* Validate output */
    err = ValidateOutput(&inFt2dTensorHandle, refTensorHandles, outTensorHandles, numDopplerFolds, nciParams,
                         nciRxTolerance, nciFinalTolerance, noiseEstimateTolerance, "Submit to cuPVA stream");

FreeResources: /* clean up all allocated resources */
#ifndef NCI_REF_ONLY
    if (NULL != stream)
    {
        CupvaStreamDestroy(stream);
    }
    if (NULL != sync)
    {
        CupvaSyncObjDestroy(sync);
    }
    nvcvOperatorDestroy(operatorHandle);
#endif
    nvcvTensorDecRef(inFt2dTensorHandle, NULL);
    nvcvTensorDecRef(outTensorHandles[0], NULL);
    nvcvTensorDecRef(outTensorHandles[1], NULL);
    nvcvTensorDecRef(outTensorHandles[2], NULL);
    nvcvTensorDecRef(refTensorHandles[0], NULL);
    nvcvTensorDecRef(refTensorHandles[1], NULL);
    nvcvTensorDecRef(refTensorHandles[2], NULL);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char **argv)
{
    /* Parameter settings */
    const int32_t Nr              = 256;
    const int32_t NofRx           = 8;
    const int32_t Ns              = 512;
    const int32_t numDopplerFolds = 8;

    PVARadarNCIParams nciParams = {
        .repeatFold             = numDopplerFolds,
        .noiseEstimationEnabled = true,
        .inputLayout            = PVA_FFT_LAYOUT_RANGE_DOPPLER_RX,
    };

    const uint32_t nciRxTolerance         = 3u;
    const uint32_t nciFinalTolerance      = (2u * nciRxTolerance) + 2u;
    const uint32_t noiseEstimateTolerance = nciFinalTolerance * 10u;

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
        cudaStreamSubmitTestResult = SubmitNCIToCudaStream(argc, argv, Nr, NofRx, Ns, numDopplerFolds, &nciParams,
                                                           nciRxTolerance, nciFinalTolerance, noiseEstimateTolerance);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult = SubmitNCIToCupvaStream(argc, argv, Nr, NofRx, Ns, numDopplerFolds, &nciParams,
                                                         nciRxTolerance, nciFinalTolerance, noiseEstimateTolerance);

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
