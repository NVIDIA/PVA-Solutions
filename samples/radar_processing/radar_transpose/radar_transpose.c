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

#include "ref/radar_transpose_ref.h"

#include <ErrorCheckMacros.h>
#include <OpRadarTranspose.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Fwd.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

enum
{
    N_RANGE   = 16,
    N_RX      = 4,
    N_DOPPLER = 64,
};

static int AllocateAndInitializeTensors(PVARadarTransposeParams const *params, NVCVAllocatorHandle *allocatorHandle,
                                        NVCVTensorHandle *inTensorHandle, NVCVTensorHandle *outTensorHandle,
                                        NVCVTensorHandle *refTensorHandle, NVCVTensorRequirements *inTensorRequirements,
                                        NVCVTensorRequirements *outTensorRequirements)
{
    int32_t err             = 0;
    int64_t inTensorShape[] = {N_RANGE, N_RX, N_DOPPLER};
    int64_t outTensorShape[3];

    if (params->outputLayout == PVA_FFT_LAYOUT_RANGE_DOPPLER_RX)
    {
        outTensorShape[0] = N_RANGE;
        outTensorShape[1] = N_DOPPLER;
        outTensorShape[2] = N_RX;
    }
    else
    {
        outTensorShape[0] = N_DOPPLER;
        outTensorShape[1] = N_RX;
        outTensorShape[2] = N_RANGE;
    }

    NVCVTensorLayout tensorLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HCW", &tensorLayout), err, Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(3, inTensorShape, NVCV_DATA_TYPE_2S32, tensorLayout, 0, 1, inTensorRequirements),
        err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(3, outTensorShape, NVCV_DATA_TYPE_2S32, tensorLayout, 0, 1,
                                                        outTensorRequirements),
                          err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(allocatorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(inTensorRequirements, *allocatorHandle, inTensorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(outTensorRequirements, *allocatorHandle, outTensorHandle), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(outTensorRequirements, *allocatorHandle, refTensorHandle), err, Error);

    generateRadarTransposeInput(*inTensorHandle);
    radarTransposeRef(*inTensorHandle, *refTensorHandle, params);

Error:
    return err;
}

static int SubmitRadarTransposeToCudaStream(PVARadarTransposeParams const *params)
{
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVTensorHandle inTensorHandle     = NULL;
    NVCVTensorHandle outTensorHandle    = NULL;
    NVCVTensorHandle refTensorHandle    = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cudaStream_t stream                 = NULL;
    int32_t err                         = 0;
    NVCVTensorRequirements inTensorRequirements;
    NVCVTensorRequirements outTensorRequirements;

    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);
    err = AllocateAndInitializeTensors(params, &allocatorHandle, &inTensorHandle, &outTensorHandle, &refTensorHandle,
                                       &inTensorRequirements, &outTensorRequirements);
    if (err)
        goto FreeResources;

    NVCV_CHECK_ERROR_GOTO(
        pvaRadarTransposeCreate(&operatorHandle, &inTensorRequirements, &outTensorRequirements, params), err,
        FreeResources);
    NVCV_CHECK_ERROR_GOTO(pvaRadarTransposeSubmit(operatorHandle, stream, inTensorHandle, outTensorHandle), err,
                          FreeResources);
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    err = checkRadarTransposeOutput(refTensorHandle, outTensorHandle);

FreeResources:
    cudaStreamDestroy(stream);
    nvcvTensorDecRef(inTensorHandle, NULL);
    nvcvTensorDecRef(outTensorHandle, NULL);
    nvcvTensorDecRef(refTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);
    return err;
}

static int SubmitRadarTransposeToCupvaStream(PVARadarTransposeParams const *params)
{
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVTensorHandle inTensorHandle     = NULL;
    NVCVTensorHandle outTensorHandle    = NULL;
    NVCVTensorHandle refTensorHandle    = NULL;
    NVCVOperatorHandle operatorHandle   = NULL;
    cupvaSyncObj_t sync                 = NULL;
    cupvaStream_t stream                = NULL;
    int32_t err                         = 0;
    NVCVTensorRequirements inTensorRequirements;
    NVCVTensorRequirements outTensorRequirements;

    err = AllocateAndInitializeTensors(params, &allocatorHandle, &inTensorHandle, &outTensorHandle, &refTensorHandle,
                                       &inTensorRequirements, &outTensorRequirements);
    if (err)
        goto FreeResources;

    NVCV_CHECK_ERROR_GOTO(
        pvaRadarTransposeCreate(&operatorHandle, &inTensorRequirements, &outTensorRequirements, params), err,
        FreeResources);
    CUPVA_CHECK_ERROR_GOTO(CupvaSyncObjCreate(&sync, false, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD), err,
                           FreeResources);

    cupvaFence_t fence;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceInit(&fence, sync), err, FreeResources);

    cupvaCmd_t requestFence;
    CUPVA_CHECK_ERROR_GOTO(CupvaCmdRequestFencesInit(&requestFence, &fence, 1), err, FreeResources);
    CUPVA_CHECK_ERROR_GOTO(CupvaStreamCreate(&stream, CUPVA_PVA0, CUPVA_VPU_ANY), err, FreeResources);

    NVCV_CHECK_ERROR_GOTO(pvaRadarTransposeSubmit(operatorHandle, stream, inTensorHandle, outTensorHandle), err,
                          FreeResources);

    cupvaCmd_t const *commands[1] = {&requestFence};
    cupvaCmdStatus_t status[1]    = {NULL};
    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, commands, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);
    if (!waitSuccess)
    {
        printf("Failed to wait for RadarTranspose completion\n");
        err = 1;
        goto FreeResources;
    }

    err = checkRadarTransposeOutput(refTensorHandle, outTensorHandle);

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

int main(void)
{
    const PVAFFTLayout layouts[] = {
        PVA_FFT_LAYOUT_RANGE_DOPPLER_RX,
        PVA_FFT_LAYOUT_DOPPLER_RX_RANGE,
    };
    int32_t result = 0;

#if CUPVA_HAS_CUDA_STREAM_SUPPORT == 1
    CUDA_CHECK_ERROR_GOTO(cudaSetDevice(0), result, CheckResults);

    uint32_t driverVersion;
    CUPVA_CHECK_ERROR_GOTO(CupvaGetDriverVersion(&driverVersion), result, CheckResults);
#endif

    for (uint32_t i = 0; i < sizeof(layouts) / sizeof(layouts[0]); ++i)
    {
        PVARadarTransposeParams params = {layouts[i]};

#if CUPVA_HAS_CUDA_STREAM_SUPPORT == 1
        if (driverVersion >= 2007)
            result |= SubmitRadarTransposeToCudaStream(&params);
#endif
        result |= SubmitRadarTransposeToCupvaStream(&params);
    }

CheckResults:
    printf(result == 0 ? "Test Pass\n" : "Test Fail\n");
    return result != 0;
}
