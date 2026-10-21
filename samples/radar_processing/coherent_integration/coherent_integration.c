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

#include "ref/coherent_integration_ref.h"

#include <ErrorCheckMacros.h>
#include <HostPointer.h>
#include <ImageIO.h>
#include <OpCoherentIntegration.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Fwd.h>
#include <nvcv/alloc/Allocator.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define INPUT_FIXTURE "coherent_integration/input_rdm_s32.bin"

/* 2 azimuth x 4 elevation uniform rectangular array, in wavelengths. */
#define RX_AZIMUTH_COUNT 2
#define RX_AZIMUTH_SPACING 4.0f
#define RX_ELEVATION_SPACING 0.5f

enum
{
    SAMPLE_NUM_RANGE_BINS   = 32,
    SAMPLE_NUM_RX           = 8,
    SAMPLE_NUM_DOPPLER_BINS = 512,
    SAMPLE_REPEAT_FOLD      = 16,
    SAMPLE_INPUT_SIZE       = SAMPLE_NUM_RANGE_BINS * SAMPLE_NUM_RX * SAMPLE_NUM_DOPPLER_BINS * 2 * sizeof(int32_t),
};

typedef struct
{
    float azimuthStartDeg;
    float azimuthStepDeg;
    int32_t numAzimuthBins;
    float elevationStartDeg;
    float elevationStepDeg;
    int32_t numElevationBins;
} BeamGrid;

static const BeamGrid kBeamGrid = {-60.0f, 2.0f, 61, -10.0f, 5.0f, 8};

typedef struct
{
    NVCVTensorHandle rxPositions;
    NVCVTensorHandle azimuthBins;
    NVCVTensorHandle elevationBins;
    NVCVTensorHandle steeringVectors;
} SteeringResources;

typedef struct
{
    NVCVAllocatorHandle allocatorHandle;
    NVCVTensorHandle rangeDopplerMap;
    NVCVTensorHandle coherentIntegrationRx;
    NVCVTensorHandle coherentIntegrationFinal;
    NVCVTensorHandle referenceCoherentIntegrationRx;
    NVCVTensorHandle referenceCoherentIntegrationFinal;
    SteeringResources steering;
    NVCVTensorRequirements rangeDopplerMapTensorRequirements;
    NVCVTensorRequirements coherentIntegrationRxTensorRequirements;
    NVCVTensorRequirements coherentIntegrationFinalTensorRequirements;
} CoherentIntegrationSample;

/** @brief Construct a tensor and return its requirements. */
static int32_t ConstructTensor(NVCVAllocatorHandle allocatorHandle, int32_t rank, const int64_t *shape,
                               NVCVDataType dataType, const char *layoutName, NVCVTensorHandle *tensorHandle,
                               NVCVTensorRequirements *requirements)
{
    int32_t err = 0;
    NVCVTensorLayout layout;

    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake(layoutName, &layout), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(rank, shape, dataType, layout, 0, 0, requirements), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(requirements, allocatorHandle, tensorHandle), err, Error);

Error:
    return err;
}

/** @brief Load the packed range-Doppler fixture into the input tensor. */
static int32_t LoadFixtureInput(NVCVTensorHandle tensor, const char *assetsDirectory)
{
    void *input = PvaGetTensorHostPointer(tensor);
    if (input == NULL)
    {
        printf("Failed to map the range-Doppler input tensor.\n");
        return -1;
    }

    return ReadImageBuffer(INPUT_FIXTURE, assetsDirectory, input, SAMPLE_INPUT_SIZE);
}

/** @brief Construct and initialize the sample steering vectors. */
static int32_t BuildSteeringVectors(NVCVAllocatorHandle allocatorHandle, SteeringResources *steering)
{
    int32_t err                          = 0;
    const int64_t rxPositionsShape[]     = {SAMPLE_NUM_RX};
    const int64_t azimuthShape[]         = {kBeamGrid.numAzimuthBins};
    const int64_t elevationShape[]       = {kBeamGrid.numElevationBins};
    const int64_t steeringVectorsShape[] = {kBeamGrid.numAzimuthBins, kBeamGrid.numElevationBins, SAMPLE_NUM_RX};
    NVCVTensorRequirements requirements;

    err = ConstructTensor(allocatorHandle, 1, rxPositionsShape, NVCV_DATA_TYPE_2F32, "W", &steering->rxPositions,
                          &requirements);
    if (err)
        goto Error;
    err = ConstructTensor(allocatorHandle, 1, azimuthShape, NVCV_DATA_TYPE_F32, "W", &steering->azimuthBins,
                          &requirements);
    if (err)
        goto Error;
    err = ConstructTensor(allocatorHandle, 1, elevationShape, NVCV_DATA_TYPE_F32, "W", &steering->elevationBins,
                          &requirements);
    if (err)
        goto Error;
    err = ConstructTensor(allocatorHandle, 3, steeringVectorsShape, NVCV_DATA_TYPE_2S16, "CHW",
                          &steering->steeringVectors, &requirements);
    if (err)
        goto Error;

    float *rxPositions   = PvaGetTensorHostPointer(steering->rxPositions);
    float *azimuthBins   = PvaGetTensorHostPointer(steering->azimuthBins);
    float *elevationBins = PvaGetTensorHostPointer(steering->elevationBins);
    if ((rxPositions == NULL) || (azimuthBins == NULL) || (elevationBins == NULL))
    {
        printf("Failed to map the steering-vector input tensors.\n");
        err = -1;
        goto Error;
    }

    for (int32_t rx = 0; rx < SAMPLE_NUM_RX; rx++)
    {
        rxPositions[2 * rx]     = (float)(rx % RX_AZIMUTH_COUNT) * RX_AZIMUTH_SPACING;
        rxPositions[2 * rx + 1] = (float)(rx / RX_AZIMUTH_COUNT) * RX_ELEVATION_SPACING;
    }
    for (int32_t bin = 0; bin < kBeamGrid.numAzimuthBins; bin++)
    {
        azimuthBins[bin] = kBeamGrid.azimuthStartDeg + ((float)bin * kBeamGrid.azimuthStepDeg);
    }
    for (int32_t bin = 0; bin < kBeamGrid.numElevationBins; bin++)
    {
        elevationBins[bin] = kBeamGrid.elevationStartDeg + ((float)bin * kBeamGrid.elevationStepDeg);
    }

    NVCV_CHECK_ERROR_GOTO(
        pvaCoherentIntegrationComputeSteeringVectors(steering->rxPositions, steering->azimuthBins,
                                                     steering->elevationBins, steering->steeringVectors),
        err, Error);

Error:
    return err;
}

/** @brief Allocate and initialize all resources used by one sample submission. */
static int32_t SetupSampleResources(CoherentIntegrationSample *sample, int argc, char **argv)
{
    char assetsDirectory[MAX_PATH_LENGTH]         = {0};
    int32_t err                                   = 0;
    const int64_t rangeDopplerShape[]             = {SAMPLE_NUM_RANGE_BINS, SAMPLE_NUM_RX, SAMPLE_NUM_DOPPLER_BINS};
    const int64_t coherentIntegrationRxShape[]    = {SAMPLE_NUM_RANGE_BINS, SAMPLE_NUM_DOPPLER_BINS};
    const int64_t coherentIntegrationFinalShape[] = {SAMPLE_NUM_RANGE_BINS,
                                                     SAMPLE_NUM_DOPPLER_BINS / SAMPLE_REPEAT_FOLD};

    memset(sample, 0, sizeof(*sample));
    if (GetAssetsDirectory(argc, argv, assetsDirectory, sizeof(assetsDirectory)) != 0)
    {
        return -1;
    }

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(&sample->allocatorHandle), err, Error);
    err = ConstructTensor(sample->allocatorHandle, 3, rangeDopplerShape, NVCV_DATA_TYPE_2S32, "HCW",
                          &sample->rangeDopplerMap, &sample->rangeDopplerMapTensorRequirements);
    if (err)
        goto Error;
    err = ConstructTensor(sample->allocatorHandle, 2, coherentIntegrationRxShape, NVCV_DATA_TYPE_U32, "HW",
                          &sample->coherentIntegrationRx, &sample->coherentIntegrationRxTensorRequirements);
    if (err)
        goto Error;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&sample->coherentIntegrationRxTensorRequirements, sample->allocatorHandle,
                                              &sample->referenceCoherentIntegrationRx),
                          err, Error);
    err = ConstructTensor(sample->allocatorHandle, 2, coherentIntegrationFinalShape, NVCV_DATA_TYPE_U32, "HW",
                          &sample->coherentIntegrationFinal, &sample->coherentIntegrationFinalTensorRequirements);
    if (err)
        goto Error;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&sample->coherentIntegrationFinalTensorRequirements,
                                              sample->allocatorHandle, &sample->referenceCoherentIntegrationFinal),
                          err, Error);

    err = BuildSteeringVectors(sample->allocatorHandle, &sample->steering);
    if (err)
        goto Error;
    err = LoadFixtureInput(sample->rangeDopplerMap, assetsDirectory);

Error:
    return err;
}

/** @brief Release all resources owned by one sample submission. */
static void FreeSampleResources(CoherentIntegrationSample *sample)
{
    nvcvTensorDecRef(sample->rangeDopplerMap, NULL);
    nvcvTensorDecRef(sample->coherentIntegrationRx, NULL);
    nvcvTensorDecRef(sample->coherentIntegrationFinal, NULL);
    nvcvTensorDecRef(sample->referenceCoherentIntegrationRx, NULL);
    nvcvTensorDecRef(sample->referenceCoherentIntegrationFinal, NULL);
    nvcvTensorDecRef(sample->steering.rxPositions, NULL);
    nvcvTensorDecRef(sample->steering.azimuthBins, NULL);
    nvcvTensorDecRef(sample->steering.elevationBins, NULL);
    nvcvTensorDecRef(sample->steering.steeringVectors, NULL);
    nvcvAllocatorDecRef(sample->allocatorHandle, NULL);
}

/** @brief Compute the reference outputs for the current sample input. */
static int32_t RunReference(CoherentIntegrationSample *sample)
{
    int32_t err = coherentIntegrationTargetRef(
        sample->rangeDopplerMap, sample->steering.steeringVectors, sample->referenceCoherentIntegrationRx,
        sample->referenceCoherentIntegrationFinal, SAMPLE_REPEAT_FOLD, PVA_FFT_LAYOUT_RANGE_RX_DOPPLER);
    if (err)
    {
        printf("Coherent integration C reference failed.\n");
        return err;
    }

    return 0;
}

/** @brief Compare the PVA outputs against the reference outputs. */
static int32_t ComparePvaOutput(const CoherentIntegrationSample *sample)
{
    int32_t err =
        checkCoherentIntegrationOutput(sample->referenceCoherentIntegrationRx, sample->coherentIntegrationRx, 0U);
    err |=
        checkCoherentIntegrationOutput(sample->referenceCoherentIntegrationFinal, sample->coherentIntegrationFinal, 0U);
    return err;
}

/** @brief Submit synchronization commands and wait for the cuPVA stream to complete. */
static int32_t SubmitAndWaitForCupvaStream(cupvaStream_t stream, cupvaSyncObj_t sync)
{
    int32_t err = 0;
    cupvaCmd_t barrier;
    cupvaFence_t fence;
    cupvaCmd_t requestFence;

    CUPVA_CHECK_ERROR_GOTO(CupvaCmdBarrierInit(&barrier), err, Error);
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceInit(&fence, sync), err, Error);
    CUPVA_CHECK_ERROR_GOTO(CupvaCmdRequestFencesInit(&requestFence, &fence, 1), err, Error);
    cupvaCmd_t const *commands[] = {&barrier, &requestFence};
    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, commands, NULL, 2, CUPVA_IN_ORDER, -1, -1), err, Error);
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, NULL), err, Error);

Error:
    return err;
}

/** @brief Run the coherent-integration sample using a CUDA stream. */
int SubmitCoherentIntegrationToCudaStream(int argc, char **argv)
{
    CoherentIntegrationSample sample         = {0};
    NVCVOperatorHandle operatorHandle        = NULL;
    cudaStream_t stream                      = NULL;
    int32_t err                              = 0;
    PVARadarCoherentIntegrationParams params = {
        .repeatFold  = SAMPLE_REPEAT_FOLD,
        .inputLayout = PVA_FFT_LAYOUT_RANGE_RX_DOPPLER,
    };

    err = SetupSampleResources(&sample, argc, argv);
    if (err)
        goto FreeResources;

    err = RunReference(&sample);
    if (err)
        goto FreeResources;

    NVCV_CHECK_ERROR_GOTO(pvaCoherentIntegrationCreate(&operatorHandle, &sample.rangeDopplerMapTensorRequirements,
                                                       &sample.coherentIntegrationRxTensorRequirements,
                                                       &sample.coherentIntegrationFinalTensorRequirements, &params),
                          err, FreeResources);
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);
    NVCV_CHECK_ERROR_GOTO(
        pvaCoherentIntegrationSubmit(operatorHandle, stream, sample.rangeDopplerMap, sample.steering.steeringVectors,
                                     sample.coherentIntegrationRx, sample.coherentIntegrationFinal),
        err, FreeResources);
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    err = ComparePvaOutput(&sample);

FreeResources:
    cudaStreamDestroy(stream);
    nvcvOperatorDestroy(operatorHandle);
    FreeSampleResources(&sample);
    return err;
}

/** @brief Run the coherent-integration sample using a cuPVA stream. */
int SubmitCoherentIntegrationToCupvaStream(int argc, char **argv)
{
    CoherentIntegrationSample sample         = {0};
    NVCVOperatorHandle operatorHandle        = NULL;
    cupvaSyncObj_t sync                      = NULL;
    cupvaStream_t stream                     = NULL;
    int32_t err                              = 0;
    PVARadarCoherentIntegrationParams params = {
        .repeatFold  = SAMPLE_REPEAT_FOLD,
        .inputLayout = PVA_FFT_LAYOUT_RANGE_RX_DOPPLER,
    };

    err = SetupSampleResources(&sample, argc, argv);
    if (err)
        goto FreeResources;

    err = RunReference(&sample);
    if (err)
        goto FreeResources;

    NVCV_CHECK_ERROR_GOTO(pvaCoherentIntegrationCreate(&operatorHandle, &sample.rangeDopplerMapTensorRequirements,
                                                       &sample.coherentIntegrationRxTensorRequirements,
                                                       &sample.coherentIntegrationFinalTensorRequirements, &params),
                          err, FreeResources);
    CUPVA_CHECK_ERROR_GOTO(CupvaSyncObjCreate(&sync, false, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD), err,
                           FreeResources);
    CUPVA_CHECK_ERROR_GOTO(CupvaStreamCreate(&stream, CUPVA_PVA0, CUPVA_VPU_ANY), err, FreeResources);
    NVCV_CHECK_ERROR_GOTO(
        pvaCoherentIntegrationSubmit(operatorHandle, stream, sample.rangeDopplerMap, sample.steering.steeringVectors,
                                     sample.coherentIntegrationRx, sample.coherentIntegrationFinal),
        err, FreeResources);
    err = SubmitAndWaitForCupvaStream(stream, sync);
    if (err == 0)
    {
        err = ComparePvaOutput(&sample);
    }
FreeResources:
    if (stream != NULL)
    {
        CupvaStreamDestroy(stream);
    }
    if (sync != NULL)
    {
        CupvaSyncObjDestroy(sync);
    }
    nvcvOperatorDestroy(operatorHandle);
    FreeSampleResources(&sample);
    return err;
}

int main(int argc, char **argv)
{
    int32_t cudaStreamSubmitResult  = 0;
    int32_t cupvaStreamSubmitResult = 0;

#if CUPVA_HAS_CUDA_STREAM_SUPPORT == 1
    CUDA_CHECK_ERROR_GOTO(cudaSetDevice(0), cudaStreamSubmitResult, CheckResults);

    uint32_t driverVersion;
    CUPVA_CHECK_ERROR_GOTO(CupvaGetDriverVersion(&driverVersion), cudaStreamSubmitResult, CheckResults);
    if (driverVersion >= 2007U)
    {
        cudaStreamSubmitResult = SubmitCoherentIntegrationToCudaStream(argc, argv);
    }
#endif

    cupvaStreamSubmitResult = SubmitCoherentIntegrationToCupvaStream(argc, argv);

CheckResults:
    if (cudaStreamSubmitResult || cupvaStreamSubmitResult)
    {
        printf("Test Fail\n");
    }
    else
    {
        printf("Test Pass\n");
    }

    return (cudaStreamSubmitResult || cupvaStreamSubmitResult);
}
