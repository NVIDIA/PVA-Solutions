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

#include "ref/range_fft_ref.h"

#include <ErrorCheckMacros.h>
#include <ImageIO.h>
#include <OpRangeFFT.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <limits.h>
#include <nvcv/Fwd.h>
#include <nvcv/alloc/Allocator.h>

/**
 * @brief Allocate and initialize tensors
 *
 * @param inDataType  Input tensor data type — NVCV_DATA_TYPE_S32 (real) or NVCV_DATA_TYPE_2S32 (complex).
 */
static int AllocateAndInitializeTensors(int argc, char **argv, int32_t Ns, int32_t NofRx, int32_t Nr, int32_t Nb,
                                        PVARangeFFTParams *params, bool useRawData, NVCVDataType inDataType,
                                        NVCVAllocatorHandle *allocatorHandle, NVCVTensorHandle *inTensorHandle,
                                        NVCVTensorHandle *winTensorHandle, NVCVTensorHandle *outTensorHandle,
                                        NVCVTensorHandle *refTensorHandle, NVCVTensorRequirements *inTensorRequirements,
                                        NVCVTensorRequirements *winTensorRequirements,
                                        NVCVTensorRequirements *outTensorRequirements, char *assetsDirectory,
                                        int32_t maxPathLength)
{
    int32_t err = 0;

    int64_t inTensorShape[]  = {Nr, NofRx, Ns};
    int64_t winTensorShape[] = {Ns};
    int64_t outTensorShape[] = {Nr, NofRx, Nb};

    // RX_DOPPLER_RANGE puts the Rx axis outermost: HCW[NofRx, Nr, W] rather than HCW[Nr, NofRx, W].
    if (params->inputLayout == PVA_FFT_LAYOUT_RX_DOPPLER_RANGE)
    {
        inTensorShape[0] = NofRx;
        inTensorShape[1] = Nr;
    }
    if (params->outputLayout == PVA_FFT_LAYOUT_RX_DOPPLER_RANGE)
    {
        outTensorShape[0] = NofRx;
        outTensorShape[1] = Nr;
    }
    // RANGE_RX_DOPPLER puts the chirps innermost: HCW[Nb, NofRx, Nr].
    if (params->outputLayout == PVA_FFT_LAYOUT_RANGE_RX_DOPPLER)
    {
        outTensorShape[0] = Nb;
        outTensorShape[1] = NofRx;
        outTensorShape[2] = Nr;
    }

    NVCVTensorLayout inTensorLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HCW", &inTensorLayout), err, Error);

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(3, inTensorShape, inDataType, inTensorLayout, 0, 0, inTensorRequirements), err,
        Error);

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
        GetAssetsDirectory(argc, argv, assetsDirectory, maxPathLength);
        loadRangeFFTRaw(*inTensorHandle, assetsDirectory, "/range_fft/ref_timedata_10_int32.bin", false);
    }
    else
    {
        generateRangeFFTInput(*inTensorHandle);
    }
    generateRangeFFTWindow(*winTensorHandle, params->windowType);

    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(NVCVTensorHandle inTensorHandle, NVCVTensorHandle winTensorHandle,
                          NVCVTensorHandle refTensorHandle, NVCVTensorHandle outTensorHandle, PVARangeFFTParams *params,
                          bool useRawData, char *assetsDirectory, float maxDiffFloat, const char *testName)
{
    int32_t err = 0;

    if (useRawData)
    {
        loadRangeFFTRaw(refTensorHandle, assetsDirectory, "/range_fft/ref_ft1d_transp_10.bin", true);
        dumpRangeFFTSNR(refTensorHandle, outTensorHandle);
    }
    else
    {
        rangeFFTRef(inTensorHandle, winTensorHandle, refTensorHandle, params);
    }

    err = checkRangeFFTOutput(refTensorHandle, outTensorHandle, maxDiffFloat);
    if (err)
    {
        printf("%s test failed\n", testName);
    }
    return err;
}

/**
 * @brief Test range FFT operation using CUDA stream
 */
int SubmitRangeFFTToCudaStream(int argc, char **argv, int32_t Ns, int32_t NofRx, int32_t Nr, int32_t Nb,
                               PVARangeFFTParams *params, bool useRawData, float maxDiffFloat, NVCVDataType inDataType)
{
    const int32_t maxPathLength = 256;
    char assetsDirectory[maxPathLength];

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

    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    err = AllocateAndInitializeTensors(argc, argv, Ns, NofRx, Nr, Nb, params, useRawData, inDataType, &allocatorHandle,
                                       &inTensorHandle, &winTensorHandle, &outTensorHandle, &refTensorHandle,
                                       &inTensorRequirements, &winTensorRequirements, &outTensorRequirements,
                                       assetsDirectory, maxPathLength);
    if (err)
        goto FreeResources;

    NVCV_CHECK_ERROR_GOTO(pvaRangeFFTCreate(&operatorHandle, &inTensorRequirements, &winTensorRequirements,
                                            &outTensorRequirements, params),
                          err, FreeResources);

    NVCV_CHECK_ERROR_GOTO(pvaRangeFFTSubmit(operatorHandle, stream, inTensorHandle, winTensorHandle, outTensorHandle),
                          err, FreeResources);

    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    err = ValidateOutput(inTensorHandle, winTensorHandle, refTensorHandle, outTensorHandle, params, useRawData,
                         assetsDirectory, maxDiffFloat, "Submit to CUDA stream");

FreeResources:
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
 * @brief Test range FFT operation using cuPVA stream
 */
int SubmitRangeFFTToCupvaStream(int argc, char **argv, int32_t Ns, int32_t NofRx, int32_t Nr, int32_t Nb,
                                PVARangeFFTParams *params, bool useRawData, float maxDiffFloat, NVCVDataType inDataType)
{
    const int32_t maxPathLength = 256;
    char assetsDirectory[maxPathLength];

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

    err = AllocateAndInitializeTensors(argc, argv, Ns, NofRx, Nr, Nb, params, useRawData, inDataType, &allocatorHandle,
                                       &inTensorHandle, &winTensorHandle, &outTensorHandle, &refTensorHandle,
                                       &inTensorRequirements, &winTensorRequirements, &outTensorRequirements,
                                       assetsDirectory, maxPathLength);
    if (err)
        goto FreeResources;

    NVCV_CHECK_ERROR_GOTO(pvaRangeFFTCreate(&operatorHandle, &inTensorRequirements, &winTensorRequirements,
                                            &outTensorRequirements, params),
                          err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaSyncObjCreate(&sync, false, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD), err,
                           FreeResources);

    cupvaFence_t fence;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceInit(&fence, sync), err, FreeResources);

    cupvaCmd_t rf;
    CUPVA_CHECK_ERROR_GOTO(CupvaCmdRequestFencesInit(&rf, &fence, 1), err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamCreate(&stream, CUPVA_PVA0, CUPVA_VPU_ANY), err, FreeResources);

    cupvaCmd_t const *cmds[1];
    cmds[0] = &rf;
    cupvaCmdStatus_t status[1];
    status[0] = NULL;

    NVCV_CHECK_ERROR_GOTO(pvaRangeFFTSubmit(operatorHandle, stream, inTensorHandle, winTensorHandle, outTensorHandle),
                          err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    err = ValidateOutput(inTensorHandle, winTensorHandle, refTensorHandle, outTensorHandle, params, useRawData,
                         assetsDirectory, maxDiffFloat, "Submit to cuPVA stream");

FreeResources:
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

/**
 * @brief Run one named test case (cuPVA stream + optional CUDA stream), print PASSED/FAILED line.
 */
static int RunNamedCase(int argc, char **argv, const char *testName, int32_t Ns, int32_t NofRx, int32_t Nr, int32_t Nb,
                        NVCVDataType inDataType, PVARangeFFTParams *params, bool useRawData, float maxDiffFloat)
{
    int32_t cudaResult  = 0;
    int32_t cupvaResult = 0;

#if CUPVA_HAS_CUDA_STREAM_SUPPORT == 1
    int32_t cudaErr = cudaSetDevice(0);
    if (cudaErr == cudaSuccess)
    {
        uint32_t driverVersion = 0;
        cupvaError_t cupvaErr  = CupvaGetDriverVersion(&driverVersion);
        if (cupvaErr == CUPVA_ERROR_NONE && driverVersion >= 2007)
        {
            cudaResult =
                SubmitRangeFFTToCudaStream(argc, argv, Ns, NofRx, Nr, Nb, params, useRawData, maxDiffFloat, inDataType);
        }
    }
#endif

    cupvaResult =
        SubmitRangeFFTToCupvaStream(argc, argv, Ns, NofRx, Nr, Nb, params, useRawData, maxDiffFloat, inDataType);

    const int32_t failed = (cudaResult != 0 || cupvaResult != 0);
    if (failed)
    {
        printf("%s: FAILED\n", testName);
    }
    else
    {
        printf("%s: PASSED\n", testName);
    }
    return failed;
}

int main(int argc, char **argv)
{
    bool useRawData    = false;
    float maxDiffFloat = 0.0f;

    if (argc > 1)
    {
        useRawData = true;
    }

    PVARangeFFTParams params = {.windowType   = PVA_BATCH_FFT_WINDOW_HANNING,
                                .inputLayout  = PVA_FFT_LAYOUT_DOPPLER_RX_RANGE,
                                .outputLayout = PVA_FFT_LAYOUT_DOPPLER_RX_RANGE};

    int32_t totalFailed = 0;

    if (useRawData)
    {
        // Legacy raw-data path stays real-input only; preserve the original Ns/Nb shape.
        totalFailed += RunNamedCase(argc, argv, "RealInput_Raw", /*Ns=*/512, /*NofRx=*/4, /*Nr=*/512, /*Nb=*/224,
                                    NVCV_DATA_TYPE_S32, &params, /*useRawData=*/true, /*maxDiffFloat=*/0.004f);
    }
    else
    {
        // Real-input cases (existing behaviour).
        totalFailed += RunNamedCase(argc, argv, "RealInput_N1024", /*Ns=*/1024, /*NofRx=*/8, /*Nr=*/512,
                                    /*Nb=*/1024 / 2 + 1, NVCV_DATA_TYPE_S32, &params, false, maxDiffFloat);
        totalFailed += RunNamedCase(argc, argv, "RealInput_N512", /*Ns=*/512, /*NofRx=*/4, /*Nr=*/128,
                                    /*Nb=*/512 / 2 + 1, NVCV_DATA_TYPE_S32, &params, false, maxDiffFloat);

        // Rx-major input and output: HCW[NofRx, Nr, Ns] in, HCW[NofRx, Nr, Nb] out.
        PVARangeFFTParams rxMajorParams = {.windowType   = PVA_BATCH_FFT_WINDOW_HANNING,
                                           .inputLayout  = PVA_FFT_LAYOUT_RX_DOPPLER_RANGE,
                                           .outputLayout = PVA_FFT_LAYOUT_RX_DOPPLER_RANGE};
        totalFailed +=
            RunNamedCase(argc, argv, "RealInput_N512_RxDopplerRange", /*Ns=*/512, /*NofRx=*/4,
                         /*Nr=*/128, /*Nb=*/512 / 2 + 1, NVCV_DATA_TYPE_S32, &rxMajorParams, false, maxDiffFloat);

        // Chirp-innermost output for the Doppler FFT: HCW[Nr, NofRx, Ns] in, HCW[Nb, NofRx, Nr] out. Real input
        // only.
        PVARangeFFTParams chirpInnermostParams = {.windowType   = PVA_BATCH_FFT_WINDOW_HANNING,
                                                  .inputLayout  = PVA_FFT_LAYOUT_DOPPLER_RX_RANGE,
                                                  .outputLayout = PVA_FFT_LAYOUT_RANGE_RX_DOPPLER};
        totalFailed += RunNamedCase(argc, argv, "RealInput_N64_RangeRxDoppler", /*Ns=*/64, /*NofRx=*/8, /*Nr=*/256,
                                    /*Nb=*/64 / 2 + 1, NVCV_DATA_TYPE_S32, &chirpInnermostParams, false, maxDiffFloat);

        // New complex-input cases.  Output range bins = Ns (full complex spectrum).  Complex input is capped at
        // Ns <= 512 due to VMEM tile-buffer sizing (see device/range_fft_param.h:MAX_COMPLEX_FFT_SIZE).
        totalFailed += RunNamedCase(argc, argv, "ComplexInput_N512", /*Ns=*/512, /*NofRx=*/8, /*Nr=*/128,
                                    /*Nb=*/512, NVCV_DATA_TYPE_2S32, &params, false, maxDiffFloat);
        totalFailed += RunNamedCase(argc, argv, "ComplexInput_N256_MultiChannel", /*Ns=*/256, /*NofRx=*/4, /*Nr=*/64,
                                    /*Nb=*/256, NVCV_DATA_TYPE_2S32, &params, false, maxDiffFloat);
    }

    if (totalFailed)
    {
        printf("Test Fail\n");
    }
    else
    {
        printf("Test Pass\n");
    }

    return (totalFailed != 0);
}
