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

#include "ref/doppler_fft_ref.h"

#include <ErrorCheckMacros.h>
#include <ImageIO.h>
#include <OpDopplerFFT.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <limits.h>
#include <nvcv/Fwd.h>
#include <nvcv/alloc/Allocator.h>

/**
 * @brief Allocate and initialize tensors
 */
static int AllocateAndInitializeTensors(int argc, char **argv, int32_t Nb, int32_t NofRx, int32_t Nr,
                                        PVADopplerFFTParams *params, bool useRawData,
                                        NVCVAllocatorHandle *allocatorHandle, NVCVTensorHandle *inTensorHandle,
                                        NVCVTensorHandle *winTensorHandle, NVCVTensorHandle *outTensorHandle,
                                        NVCVTensorHandle *refTensorHandle, NVCVTensorRequirements *inTensorRequirements,
                                        NVCVTensorRequirements *winTensorRequirements,
                                        NVCVTensorRequirements *outTensorRequirements, char *assetsDirectory,
                                        int32_t maxPathLength)
{
    int32_t err = 0;

    int64_t inTensorShape[]  = {Nr, NofRx, Nb};
    int64_t winTensorShape[] = {Nr};

    if (params->inputLayout == PVA_FFT_LAYOUT_RANGE_RX_DOPPLER)
    {
        inTensorShape[0] = Nb;
        inTensorShape[1] = NofRx;
        inTensorShape[2] = Nr;
    }
    else if (params->inputLayout == PVA_FFT_LAYOUT_RX_DOPPLER_RANGE)
    {
        inTensorShape[0] = NofRx;
        inTensorShape[1] = Nr;
        inTensorShape[2] = Nb;
    }

    int64_t outTensorShape[3];
    switch (params->outputLayout)
    {
    case PVA_FFT_LAYOUT_DOPPLER_RX_RANGE:
        outTensorShape[0] = Nr;
        outTensorShape[1] = NofRx;
        outTensorShape[2] = Nb;
        break;
    case PVA_FFT_LAYOUT_RANGE_RX_DOPPLER:
        outTensorShape[0] = Nb;
        outTensorShape[1] = NofRx;
        outTensorShape[2] = Nr;
        break;
    case PVA_FFT_LAYOUT_RANGE_DOPPLER_RX:
        outTensorShape[0] = Nb;
        outTensorShape[1] = Nr;
        outTensorShape[2] = NofRx;
        break;
    case PVA_FFT_LAYOUT_RX_RANGE_DOPPLER:
        outTensorShape[0] = NofRx;
        outTensorShape[1] = Nb;
        outTensorShape[2] = Nr;
        break;
    default:
        printf("Unsupported output layout %d\n", (int)params->outputLayout);
        return -1;
    }

    NVCVTensorLayout inTensorLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HCW", &inTensorLayout), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(3, inTensorShape, NVCV_DATA_TYPE_2S32, inTensorLayout, 0, 0,
                                                        inTensorRequirements),
                          err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(inTensorRequirements, *allocatorHandle, inTensorHandle), err, Error);

    NVCVTensorLayout winTensorLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("W", &winTensorLayout), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(1, winTensorShape, NVCV_DATA_TYPE_S32, winTensorLayout, 0, 0,
                                                        winTensorRequirements),
                          err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(winTensorRequirements, *allocatorHandle, winTensorHandle), err, Error);

    NVCVTensorLayout outTensorLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HCW", &outTensorLayout), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(3, outTensorShape, NVCV_DATA_TYPE_2S32, outTensorLayout, 0, 0,
                                                        outTensorRequirements),
                          err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(outTensorRequirements, *allocatorHandle, outTensorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(outTensorRequirements, *allocatorHandle, refTensorHandle), err, Error);

    if (useRawData)
    {
        err = GetAssetsDirectory(argc, argv, assetsDirectory, maxPathLength);
        if (err)
        {
            printf("Failed to get assets directory\n");
            goto Error;
        }
        err = loadDopplerFFTRaw(*inTensorHandle, assetsDirectory, "/doppler_fft/pva_ft1d_transp_10_complex_int32.bin",
                                false);
        if (err)
        {
            printf("Failed to load input data\n");
            goto Error;
        }
    }
    else
    {
        generateDopplerFFTInput(*inTensorHandle);
    }
    generateDopplerFFTWindow(*winTensorHandle, params->windowType);

    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(NVCVTensorHandle inTensorHandle, NVCVTensorHandle winTensorHandle,
                          NVCVTensorHandle refTensorHandle, NVCVTensorHandle outTensorHandle,
                          PVADopplerFFTParams *params, bool useRawData, char *assetsDirectory, float maxDiffFloat,
                          const char *testName)
{
    int32_t err = 0;

    if (useRawData)
    {
        err = loadDopplerFFTRaw(refTensorHandle, assetsDirectory, "/doppler_fft/ref_ft2d_10.bin", true);
        if (err)
        {
            printf("%s test failed: Failed to load reference data\n", testName);
            return err;
        }
        dumpDopplerFFTSNR(refTensorHandle, outTensorHandle);
    }
    else
    {
        dopplerFFTRef(inTensorHandle, winTensorHandle, refTensorHandle, params);
    }

    err = checkDopplerFFTOutput(refTensorHandle, outTensorHandle, maxDiffFloat);
    if (err)
    {
        printf("%s test failed\n", testName);
    }
    return err;
}

/**
 * @brief Test doppler FFT operation using CUDA stream
 */
int SubmitDopplerFFTToCudaStream(int argc, char **argv, int32_t Nb, int32_t NofRx, int32_t Nr,
                                 PVADopplerFFTParams *params, bool useRawData, float maxDiffFloat)
{
    const int32_t maxPathLength = 256;
    char assetsDirectory[maxPathLength];

    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVTensorHandle inTensorHandle     = NULL;
    NVCVTensorHandle winTensorHandle    = NULL;
    NVCVTensorHandle outTensorHandle    = NULL;
    NVCVTensorHandle refTensorHandle    = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cudaStream_t stream                 = NULL;
    int32_t err                         = 0;

    NVCVTensorRequirements inTensorRequirements;
    NVCVTensorRequirements winTensorRequirements;
    NVCVTensorRequirements outTensorRequirements;

    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeTensors(argc, argv, Nb, NofRx, Nr, params, useRawData, &allocatorHandle, &inTensorHandle,
                                       &winTensorHandle, &outTensorHandle, &refTensorHandle, &inTensorRequirements,
                                       &winTensorRequirements, &outTensorRequirements, assetsDirectory, maxPathLength);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaDopplerFFTCreate(&operatorHandle, &inTensorRequirements, &winTensorRequirements,
                                              &outTensorRequirements, params),
                          err, FreeResources);

    /* Submit to CUDA stream */
    NVCV_CHECK_ERROR_GOTO(pvaDopplerFFTSubmit(operatorHandle, stream, inTensorHandle, winTensorHandle, outTensorHandle),
                          err, FreeResources);

    /* Synchronize stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(inTensorHandle, winTensorHandle, refTensorHandle, outTensorHandle, params, useRawData,
                         assetsDirectory, maxDiffFloat, "Submit to CUDA stream");

FreeResources: /* clean up all allocated resources */
    cudaStreamDestroy(stream);
    nvcvTensorDecRef(inTensorHandle, NULL);
    nvcvTensorDecRef(winTensorHandle, NULL);
    nvcvTensorDecRef(outTensorHandle, NULL);
    nvcvTensorDecRef(refTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

/**
 * @brief Test doppler FFT operation using cuPVA stream
 */
int SubmitDopplerFFTToCupvaStream(int argc, char **argv, int32_t Nb, int32_t NofRx, int32_t Nr,
                                  PVADopplerFFTParams *params, bool useRawData, float maxDiffFloat)
{
    const int32_t maxPathLength = 256;
    char assetsDirectory[maxPathLength];

    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVTensorHandle inTensorHandle     = NULL;
    NVCVTensorHandle winTensorHandle    = NULL;
    NVCVTensorHandle outTensorHandle    = NULL;
    NVCVTensorHandle refTensorHandle    = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cupvaSyncObj_t sync                 = NULL;
    cupvaStream_t stream                = NULL;
    int32_t err                         = 0;

    NVCVTensorRequirements inTensorRequirements;
    NVCVTensorRequirements winTensorRequirements;
    NVCVTensorRequirements outTensorRequirements;

    /* Allocate and initialize tensors */
    err = AllocateAndInitializeTensors(argc, argv, Nb, NofRx, Nr, params, useRawData, &allocatorHandle, &inTensorHandle,
                                       &winTensorHandle, &outTensorHandle, &refTensorHandle, &inTensorRequirements,
                                       &winTensorRequirements, &outTensorRequirements, assetsDirectory, maxPathLength);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaDopplerFFTCreate(&operatorHandle, &inTensorRequirements, &winTensorRequirements,
                                              &outTensorRequirements, params),
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

    NVCV_CHECK_ERROR_GOTO(pvaDopplerFFTSubmit(operatorHandle, stream, inTensorHandle, winTensorHandle, outTensorHandle),
                          err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    /* Wait for completion */
    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(inTensorHandle, winTensorHandle, refTensorHandle, outTensorHandle, params, useRawData,
                         assetsDirectory, maxDiffFloat, "Submit to cuPVA stream");

FreeResources: /* clean up all allocated resources */
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvTensorDecRef(inTensorHandle, NULL);
    nvcvTensorDecRef(winTensorHandle, NULL);
    nvcvTensorDecRef(outTensorHandle, NULL);
    nvcvTensorDecRef(refTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char **argv)
{
    /* Parameter settings */
    bool useRawData    = false;
    float maxDiffFloat = 0.0f;

    int32_t Nb    = 513;
    int32_t NofRx = 8;
    int32_t Nr    = 512;

    PVADopplerFFTParams params = {.windowType   = PVA_BATCH_FFT_WINDOW_HANNING,
                                  .inputLayout  = PVA_FFT_LAYOUT_DOPPLER_RX_RANGE,
                                  .outputLayout = PVA_FFT_LAYOUT_RANGE_RX_DOPPLER};

    if (argc > 1)
    {
        // Use raw data
        useRawData = true;
    }

    if (useRawData)
    {
        maxDiffFloat = 0.004f;

        Nb    = 224;
        NofRx = 4;
        Nr    = 512;

        params.outputLayout = PVA_FFT_LAYOUT_RANGE_RX_DOPPLER;
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
            SubmitDopplerFFTToCudaStream(argc, argv, Nb, NofRx, Nr, &params, useRawData, maxDiffFloat);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult =
        SubmitDopplerFFTToCupvaStream(argc, argv, Nb, NofRx, Nr, &params, useRawData, maxDiffFloat);

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
