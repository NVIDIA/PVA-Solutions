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

#include "doa_params.h"
#include "doa_target_processing_ref.h"

#include <ErrorCheckMacros.h>
#include <ImageIO.h>
#include <OpDoaDAFFT.h>
#include <PvaAllocator.h>
#include <PvaOperator.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <limits.h>
#include <nvcv/Fwd.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>
#include <stdbool.h>
#include <string.h>

/**
 * @brief Allocate and initialize tensors
 */
static int AllocateAndInitializeTensors(int argc, char **argv, int32_t numPeaks, int32_t calibVectorQbits,
                                        int32_t fftQbits, const char *peakRawFile, const char *nciFinalFile,
                                        NVCVAllocatorHandle *allocatorHandle, NVCVTensorHandle *inTensorHandles,
                                        NVCVTensorHandle *outTensorHandle, NVCVTensorHandle *refTensorHandle,
                                        NVCVTensorRequirements *inTensorReqPtrs[],
                                        NVCVTensorRequirements *outTensorReqPtr, PVARadarGP *GP, int32_t *calibVector,
                                        char *assetsDirectory, int32_t maxPathLength)
{
    int32_t err = 0;

    populateGP(GP, calibVectorQbits, fftQbits);
    populateCalibVector(calibVector, calibVectorQbits);

    NVCVDataType inDataType[5] = {NVCV_DATA_TYPE_U32, NVCV_DATA_TYPE_2S32, NVCV_DATA_TYPE_U32, NVCV_DATA_TYPE_2S32,
                                  NVCV_DATA_TYPE_U32};
    NVCVDataType outDataType   = NVCV_DATA_TYPE_F32;

    int32_t tensorRank          = 2;
    int64_t inTensorShape[5][2] = {{1, 1}, {1, 16}, {3, DOA_MAX_TARGET_COUNT}, {DOA_MAX_TARGET_COUNT, 16}, {224, 64}};
    int64_t outTensorShape[]    = {7, DOA_MAX_TARGET_COUNT};

    NVCVTensorLayout tensorLayout;

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("W", &tensorLayout), err, Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(1, inTensorShape[0], inDataType[0], tensorLayout, 0, 0, inTensorReqPtrs[0]), err,
        Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(inTensorReqPtrs[0], *allocatorHandle, &inTensorHandles[0]), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HW", &tensorLayout), err, Error);
    for (int i = 1; i < 5; ++i)
    {
        NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank, inTensorShape[i], inDataType[i], tensorLayout,
                                                            0, 0, inTensorReqPtrs[i]),
                              err, Error);
        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(inTensorReqPtrs[i], *allocatorHandle, &inTensorHandles[i]), err,
                              Error);
    }

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(tensorRank, outTensorShape, outDataType, tensorLayout, 0, 0, outTensorReqPtr),
        err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(outTensorReqPtr, *allocatorHandle, outTensorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(outTensorReqPtr, *allocatorHandle, refTensorHandle), err, Error);

    err = GetAssetsDirectory(argc, argv, assetsDirectory, maxPathLength);
    if (err)
    {
        printf("Failed to get assets directory\n");
        return err;
    }

    err = loadPeakCountToTensor(inTensorHandles[0], numPeaks);
    if (err)
    {
        printf("Failed to load peak count to tensor\n");
        return err;
    }

    err = loadCalibVectorToTensorFixedPoint(inTensorHandles[1], calibVector, GP->nofSnap * 2);
    if (err)
    {
        printf("Failed to load calibration vector to tensor\n");
        return err;
    }

    err = loadPeakDataFromRawFile(inTensorHandles[2], inTensorHandles[3], assetsDirectory, peakRawFile, numPeaks);
    if (err)
    {
        printf("Failed to load peak data from raw file\n");
        return err;
    }

    err = loadNciFinalData(inTensorHandles[4], assetsDirectory, nciFinalFile);
    if (err)
    {
        printf("Failed to load NCI FINAL data from raw file\n");
        return err;
    }

    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(NVCVTensorHandle *inTensorHandles, NVCVTensorHandle refTensorHandle,
                          NVCVTensorHandle outTensorHandle, PVARadarGP *GP, int32_t numPeaks, bool writeTarResults,
                          const char *testName)
{
    int32_t err = 0;

    doaTargetProcessingRef(inTensorHandles, refTensorHandle, GP);

    if (writeTarResults)
    {
        printf("Writing reference TAR results to file\n");
        err = writeTarResultsToFile(refTensorHandle, numPeaks, "tar_ref_results.dat");
        if (err)
        {
            printf("Warning: Failed to write TAR results to file\n");
        }

        printf("Writing PVA TAR results to file\n");
        err = writeTarResultsToFile(outTensorHandle, numPeaks, "tar_pva_results.dat");
        if (err)
        {
            printf("Warning: Failed to write TAR results to file\n");
        }
    }
    else
    {
        printf("Skipping TAR results file output (use --write-tar to enable)\n");
    }

    int32_t quickResult = quickCompareOutputWithReference(outTensorHandle, refTensorHandle, GP, numPeaks, 2e-4f);
    if (quickResult)
    {
        printf("%s test failed: Found %d mismatches\n", testName, quickResult);
        return quickResult;
    }

    return 0;
}

/**
 * @brief Test DOA target processing operation using CUDA stream
 */
int SubmitDOATargetProcessingToCudaStream(int argc, char **argv, int32_t numPeaks, int32_t calibVectorQbits,
                                          int32_t fftQbits, bool writeTarResults, const char *peakRawFile,
                                          const char *nciFinalFile)
{
    const int32_t maxPathLength = 256;
    char assetsDirectory[maxPathLength];

    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVTensorHandle inTensorHandles[5] = {NULL, NULL, NULL, NULL, NULL};
    NVCVTensorHandle outTensorHandle    = NULL;
    NVCVTensorHandle refTensorHandle    = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cudaStream_t stream                 = NULL;

    NVCVTensorRequirements inTensorReqs[5];
    NVCVTensorRequirements outTensorReq;
    NVCVTensorRequirements *inTensorReqPtrs[5] = {NULL, NULL, NULL, NULL, NULL};
    NVCVTensorRequirements *outTensorReqPtr    = NULL;

    for (int i = 0; i < 5; ++i)
    {
        inTensorReqPtrs[i] = &inTensorReqs[i];
    }
    outTensorReqPtr = &outTensorReq;

    int32_t inTensorCount = 5;
    int32_t err           = 0;

    PVARadarGP GP;
    int32_t calibVector[16 * 2]; // GP.nofSnap * 2, but GP not initialized yet

    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeTensors(argc, argv, numPeaks, calibVectorQbits, fftQbits, peakRawFile, nciFinalFile,
                                       &allocatorHandle, inTensorHandles, &outTensorHandle, &refTensorHandle,
                                       inTensorReqPtrs, outTensorReqPtr, &GP, calibVector, assetsDirectory,
                                       maxPathLength);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(
        pvaDoaDAFFTCreate(&operatorHandle, (const NVCVTensorRequirements **)inTensorReqPtrs,
                          (const NVCVTensorRequirements *)outTensorReqPtr, inTensorCount, (const PVARadarGP *)&GP),
        err, FreeResources);

    /* Submit to CUDA stream */
    NVCV_CHECK_ERROR_GOTO(pvaDoaDAFFTSubmit(operatorHandle, stream, inTensorHandles, outTensorHandle, inTensorCount),
                          err, FreeResources);

    /* Synchronize stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(inTensorHandles, refTensorHandle, outTensorHandle, &GP, numPeaks, writeTarResults,
                         "Submit to CUDA stream");
    if (err)
    {
        printf("Submit to CUDA stream test failed\n");
    }

FreeResources: /* clean up all allocated resources */
    cudaStreamDestroy(stream);
    for (int i = 0; i < inTensorCount; ++i)
    {
        nvcvTensorDecRef(inTensorHandles[i], NULL);
    }
    nvcvTensorDecRef(outTensorHandle, NULL);
    nvcvTensorDecRef(refTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

/**
 * @brief Test DOA target processing operation using cuPVA stream
 */
int SubmitDOATargetProcessingToCupvaStream(int argc, char **argv, int32_t numPeaks, int32_t calibVectorQbits,
                                           int32_t fftQbits, bool writeTarResults, const char *peakRawFile,
                                           const char *nciFinalFile)
{
    const int32_t maxPathLength = 256;
    char assetsDirectory[maxPathLength];

    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVTensorHandle inTensorHandles[5] = {NULL, NULL, NULL, NULL, NULL};
    NVCVTensorHandle outTensorHandle    = NULL;
    NVCVTensorHandle refTensorHandle    = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;

    NVCVTensorRequirements inTensorReqs[5];
    NVCVTensorRequirements outTensorReq;
    NVCVTensorRequirements *inTensorReqPtrs[5] = {NULL, NULL, NULL, NULL, NULL};
    NVCVTensorRequirements *outTensorReqPtr    = NULL;

    for (int i = 0; i < 5; ++i)
    {
        inTensorReqPtrs[i] = &inTensorReqs[i];
    }
    outTensorReqPtr = &outTensorReq;

    int32_t inTensorCount = 5;
    int32_t err           = 0;

    PVARadarGP GP;
    int32_t calibVector[16 * 2]; // GP.nofSnap * 2, but GP not initialized yet

    cupvaSyncObj_t sync  = NULL;
    cupvaStream_t stream = NULL;

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeTensors(argc, argv, numPeaks, calibVectorQbits, fftQbits, peakRawFile, nciFinalFile,
                                       &allocatorHandle, inTensorHandles, &outTensorHandle, &refTensorHandle,
                                       inTensorReqPtrs, outTensorReqPtr, &GP, calibVector, assetsDirectory,
                                       maxPathLength);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(
        pvaDoaDAFFTCreate(&operatorHandle, (const NVCVTensorRequirements **)inTensorReqPtrs,
                          (const NVCVTensorRequirements *)outTensorReqPtr, inTensorCount, (const PVARadarGP *)&GP),
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
    cupvaCmd_t const *cmds[1];
    cmds[0] = &rf;
    cupvaCmdStatus_t status[1];
    status[0] = NULL;

    NVCV_CHECK_ERROR_GOTO(pvaDoaDAFFTSubmit(operatorHandle, stream, inTensorHandles, outTensorHandle, inTensorCount),
                          err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    /* Wait for completion */
    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(inTensorHandles, refTensorHandle, outTensorHandle, &GP, numPeaks, writeTarResults,
                         "Submit to cuPVA stream");
    if (err)
    {
        printf("Submit to cuPVA stream test failed\n");
    }

FreeResources: /* clean up all allocated resources */
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    for (int i = 0; i < inTensorCount; ++i)
    {
        nvcvTensorDecRef(inTensorHandles[i], NULL);
    }
    nvcvTensorDecRef(outTensorHandle, NULL);
    nvcvTensorDecRef(refTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char **argv)
{
    /* Parameter settings */
    const char peakRawFile[]  = "/doa_da_fft/peak_fixed.dat";
    const char nciFinalFile[] = "/doa_da_fft/nci_final_fixed.dat";
    const int32_t numPeaks    = 71;
    int32_t calibVectorQbits  = 30;
    int32_t fftQbits          = 15;
    bool writeTarResults      = false;

    // Check for optional --write-tar argument
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "--write-tar") == 0)
        {
            writeTarResults = true;
            break;
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
        cudaStreamSubmitTestResult = SubmitDOATargetProcessingToCudaStream(
            argc, argv, numPeaks, calibVectorQbits, fftQbits, writeTarResults, peakRawFile, nciFinalFile);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult = SubmitDOATargetProcessingToCupvaStream(
        argc, argv, numPeaks, calibVectorQbits, fftQbits, writeTarResults, peakRawFile, nciFinalFile);

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
