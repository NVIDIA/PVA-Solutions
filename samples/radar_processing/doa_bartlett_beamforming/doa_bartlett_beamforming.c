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

/**
 * @file doa_bartlett_beamforming.c
 * @brief Sample application and test harness for DOA Bartlett Beamforming + Target Processing operator
 *
 * This sample demonstrates how to use the Bartlett beamforming operator for
 * Direction of Arrival estimation using steering vector correlation on
 * radar detection snapshots, combined with target processing to produce
 * complete target information.
 *
 * The sample provides:
 * - Tensor allocation and initialization
 * - Reference CPU implementation for validation
 * - Operator submission to both CUDA and cuPVA streams
 * - Result comparison and accuracy checking
 *
 * Key Functions:
 * - AllocateAndInitializeTensors(): Sets up input/output tensors
 * - SubmitBartlettToCudaStream(): Submits operator to CUDA stream
 * - SubmitBartlettToCupvaStream(): Submits operator to cuPVA stream
 * - ValidateOutput(): Validates PVA results against CPU reference
 */

#include <ErrorCheckMacros.h>
#include <ImageIO.h>
#include <OpDoaBartlettBeamforming.h>
#include <OpTargetProcessing.h>
#include <PvaAllocator.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <cupva_host_nonsafety.h>
#include <math.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_FILENAME_LENGTH 256

/* Position2D structure for virtual array locations */
typedef struct
{
    float x;
    float y;
} Position2D;

#define NUM_DETECTIONS 97
#define NUM_TX 8
#define NUM_RX 8
#define NUM_AZIMUTH_BINS 81
#define NUM_ELEVATION_BINS 8

#define TOLERANCE 0.1f

#define PRINT_ACCURACY_STATS 0

/* Forward declarations from reference implementation (extern "C" in the header) */
extern void doaBartlettBeamformingRef(
    NVCVTensorHandle detectionCountTensorHandle, NVCVTensorHandle snapshotsTensorHandle,
    NVCVTensorHandle steeringVectorsTensorHandle, NVCVTensorHandle azimuthBinsTensorHandle,
    NVCVTensorHandle elevationBinsTensorHandle, NVCVTensorHandle targetCountTensorHandle,
    NVCVTensorHandle targetAnglesTensorHandle, NVCVTensorHandle targetIndexMapTensorHandle,
    const PVADoaBartlettBeamformingParams *params);

extern void targetProcessingRef(NVCVTensorHandle targetCountTensorHandle, NVCVTensorHandle detectionListTensorHandle,
                                NVCVTensorHandle detectionIndexMapTensorHandle, NVCVTensorHandle anglesTensorHandle,
                                NVCVTensorHandle ddmDopplerOffsetsTensorHandle, NVCVTensorHandle nciTensorHandle,
                                NVCVTensorHandle targetListTensorHandle, const PVATargetProcessingParams *params);

extern int32_t checkTargetProcessingOutput(NVCVTensorHandle targetCountTensorHandle,
                                           NVCVTensorHandle pvaTargetListTensorHandle,
                                           NVCVTensorHandle refTargetListTensorHandle);

/**
 * @brief Structure to hold tensor handles and requirements
 */
typedef struct
{
    NVCVAllocatorHandle *allocatorHandle;
    NVCVTensorHandle *detectionCountTensor;    /* Tensor 0: Detection count (input) */
    NVCVTensorHandle *snapshotsTensor;         /* Tensor 1: Snapshots (input) */
    NVCVTensorHandle *steeringVectorsTensor;   /* Tensor 2: Steering vectors (input) */
    NVCVTensorHandle *azimuthBinsTensor;       /* Tensor 3: Azimuth bins (input) */
    NVCVTensorHandle *elevationBinsTensor;     /* Tensor 4: Elevation bins (input) */
    NVCVTensorHandle *detectionListTensor;     /* Tensor 5: Detection list (input) */
    NVCVTensorHandle *ddmDopplerOffsetsTensor; /* Tensor 6: DDM Doppler offsets (input) */
    NVCVTensorHandle *targetCountTensor;       /* Tensor 7: Target count (output) */
    NVCVTensorHandle *targetAnglesTensor;      /* Tensor 8: Target angles (DOA output) */
    NVCVTensorHandle *targetIndexMapTensor;    /* Tensor 9: Target index map (DOA output) */
    NVCVTensorHandle *targetListTensor;        /* Tensor 10: Target list (TP output) */
    NVCVTensorHandle *vcPosTensor;             /* Virtual channel positions */
    NVCVTensorHandle *refTargetCountTensor;    /* Reference: target count */
    NVCVTensorHandle *refTargetAnglesTensor;   /* Reference: DOA angles */
    NVCVTensorHandle *refTargetIndexMapTensor; /* Reference: target-to-detection map */
    NVCVTensorHandle *refTargetListTensor;     /* Reference: target list */
    NVCVTensorRequirements *snapshotsReqs;
    NVCVTensorRequirements *detectionListReqs;
    NVCVTensorRequirements *ddmDopplerOffsetsReqs;
    NVCVTensorRequirements *targetAnglesReqs;
    NVCVTensorRequirements *targetIndexMapReqs;
    NVCVTensorRequirements *targetListReqs;
    NVCVTensorRequirements *targetCountReqs;
} BartlettTensors;

/**
 * @brief Allocate and initialize all tensors for Bartlett beamforming + Target Processing
 */
static int AllocateAndInitializeTensors(BartlettTensors *tensors, PVADoaBartlettBeamformingParams *params, int argc,
                                        char **argv)
{
    int32_t err = 0;
    int32_t i;

    /* Tensor layouts */
    NVCVTensorLayout layoutW, layoutCHW, layoutHW, layoutNCHW;

    /* Tensor shapes */
    int64_t shape_count[]          = {1};
    int64_t shape_snapshots[]      = {NUM_DETECTIONS, NUM_TX, NUM_RX};
    int64_t shape_steering[]       = {NUM_AZIMUTH_BINS, NUM_ELEVATION_BINS, NUM_TX, NUM_RX};
    int64_t shape_azBins[]         = {NUM_AZIMUTH_BINS};
    int64_t shape_elBins[]         = {NUM_ELEVATION_BINS};
    int64_t shape_detectionList[]  = {NUM_DETECTIONS, 2};
    int64_t shape_ddmDoppler[]     = {1};
    int64_t numTargetAngleParams   = params->enablePowerOutput ? 3 : 2;
    int64_t shape_targetAngles[]   = {numTargetAngleParams, NUM_DETECTIONS};
    int64_t shape_targetIndexMap[] = {NUM_DETECTIONS};
    int64_t targetListHeight       = params->enablePowerOutput ? 8 : 7;
    int64_t shape_targetList[]     = {targetListHeight, NUM_DETECTIONS};

    /* Temporary requirements */
    NVCVTensorRequirements countReqs, steeringReqs, azBinsReqs, elBinsReqs;

    /* Data pointers */
    NVCVTensorData data;
    int32_t *ptr32;
    float *azimuthBins   = NULL;
    float *elevationBins = NULL;
    Position2D *vcPosPtr = NULL;

    /* Assets directory for loading test data */
    char assetsDirectory[MAX_FILENAME_LENGTH] = {0};
    char snapshotFile[MAX_FILENAME_LENGTH + 64];
    int useAssets = (GetAssetsDirectory(argc, argv, assetsDirectory, MAX_FILENAME_LENGTH) == 0);

    /* Construct PVA-specific allocator */
    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(tensors->allocatorHandle), err, Error);

    /* Create tensor layouts */
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("W", &layoutW), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("CHW", &layoutCHW), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HW", &layoutHW), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("NCHW", &layoutNCHW), err, Error);

    /* Tensor 0: Detection count (input) */
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(1, shape_count, NVCV_DATA_TYPE_S32, layoutW, 0, 0, &countReqs),
                          err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&countReqs, *tensors->allocatorHandle, tensors->detectionCountTensor),
                          err, Error);

    /* Tensor 1: Snapshots (input) */
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(3, shape_snapshots, NVCV_DATA_TYPE_2S32, layoutCHW, 0, 0, tensors->snapshotsReqs),
        err, Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(tensors->snapshotsReqs, *tensors->allocatorHandle, tensors->snapshotsTensor), err, Error);

    /* Tensor 2: Steering vectors (input) */
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(4, shape_steering, NVCV_DATA_TYPE_2S16, layoutNCHW, 0, 0, &steeringReqs), err,
        Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&steeringReqs, *tensors->allocatorHandle, tensors->steeringVectorsTensor),
                          err, Error);

    /* Tensor 3: Azimuth bins (input) */
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(1, shape_azBins, NVCV_DATA_TYPE_F32, layoutW, 0, 0, &azBinsReqs), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&azBinsReqs, *tensors->allocatorHandle, tensors->azimuthBinsTensor), err,
                          Error);

    /* Tensor 4: Elevation bins (input) */
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(1, shape_elBins, NVCV_DATA_TYPE_F32, layoutW, 0, 0, &elBinsReqs), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&elBinsReqs, *tensors->allocatorHandle, tensors->elevationBinsTensor),
                          err, Error);

    /* Tensor 5: Detection list (input) */
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(2, shape_detectionList, NVCV_DATA_TYPE_S32, layoutHW, 0, 0,
                                                        tensors->detectionListReqs),
                          err, Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(tensors->detectionListReqs, *tensors->allocatorHandle, tensors->detectionListTensor), err,
        Error);

    /* Tensor 6: DDM Doppler offsets (input) */
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(1, shape_ddmDoppler, NVCV_DATA_TYPE_F32, layoutW, 0, 0,
                                                        tensors->ddmDopplerOffsetsReqs),
                          err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensors->ddmDopplerOffsetsReqs, *tensors->allocatorHandle,
                                              tensors->ddmDopplerOffsetsTensor),
                          err, Error);

    /* Tensor 7: Target count (output) */
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(1, shape_count, NVCV_DATA_TYPE_S32, layoutW, 0, 0, tensors->targetCountReqs), err,
        Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(tensors->targetCountReqs, *tensors->allocatorHandle, tensors->targetCountTensor), err,
        Error);

    /* Tensor 8: Target angles (DOA output) */
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(2, shape_targetAngles, NVCV_DATA_TYPE_F32, layoutHW, 0, 0,
                                                        tensors->targetAnglesReqs),
                          err, Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(tensors->targetAnglesReqs, *tensors->allocatorHandle, tensors->targetAnglesTensor), err,
        Error);

    /* Tensor 9: Target index map (DOA output) */
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(1, shape_targetIndexMap, NVCV_DATA_TYPE_S32, layoutW, 0, 0,
                                                        tensors->targetIndexMapReqs),
                          err, Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(tensors->targetIndexMapReqs, *tensors->allocatorHandle, tensors->targetIndexMapTensor), err,
        Error);

    /* Tensor 10: Target list (Target Processing output) */
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(2, shape_targetList, NVCV_DATA_TYPE_F32, layoutHW, 0, 0, tensors->targetListReqs),
        err, Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(tensors->targetListReqs, *tensors->allocatorHandle, tensors->targetListTensor), err, Error);

    /* Reference tensors for validation */
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(tensors->targetCountReqs, *tensors->allocatorHandle, tensors->refTargetCountTensor), err,
        Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(tensors->targetAnglesReqs, *tensors->allocatorHandle, tensors->refTargetAnglesTensor), err,
        Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(tensors->targetIndexMapReqs, *tensors->allocatorHandle, tensors->refTargetIndexMapTensor),
        err, Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(tensors->targetListReqs, *tensors->allocatorHandle, tensors->refTargetListTensor), err,
        Error);

    /* Virtual channel positions tensor [numTx, numRx] - 2F32 */
    int64_t shapeVcPos[] = {NUM_TX, NUM_RX};
    NVCVTensorRequirements vcPosReqs;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(2, shapeVcPos, NVCV_DATA_TYPE_2F32, layoutHW, 0, 0, &vcPosReqs),
                          err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&vcPosReqs, *tensors->allocatorHandle, tensors->vcPosTensor), err, Error);

    /* Fill detection count */
    nvcvTensorExportData(*tensors->detectionCountTensor, &data);
    CupvaMemGetHostPointer((void **)&ptr32, (void *)data.buffer.strided.basePtr);
    ptr32[0] = NUM_DETECTIONS;

    /* Generate or load snapshot data */
    nvcvTensorExportData(*tensors->snapshotsTensor, &data);
    CupvaMemGetHostPointer((void **)&ptr32, (void *)data.buffer.strided.basePtr);
    {
        int dataLoaded      = 0;
        size_t expectedSize = NUM_DETECTIONS * NUM_TX * NUM_RX * 2 * sizeof(int32_t);

        /* Try to load from assets file if provided */
        if (useAssets)
        {
            FILE *fp;
            snprintf(snapshotFile, sizeof(snapshotFile), "%s/bartlett_beamforming/snapshots.bin", assetsDirectory);
            fp = fopen(snapshotFile, "rb");
            if (fp)
            {
                size_t bytesRead = fread(ptr32, 1, expectedSize, fp);
                fclose(fp);
                if (bytesRead == expectedSize)
                {
                    printf("Loaded snapshot data from: %s\n", snapshotFile);
                    dataLoaded = 1;
                }
            }
        }

        /* Fall back to synthetic data if not loaded */
        if (!dataLoaded)
        {
            const int32_t Q20_SCALE = (1 << 20);
            int32_t det, tx, rx;
            for (det = 0; det < NUM_DETECTIONS; det++)
            {
                for (tx = 0; tx < NUM_TX; tx++)
                {
                    for (rx = 0; rx < NUM_RX; rx++)
                    {
                        int32_t idx_snap    = ((det * NUM_TX + tx) * NUM_RX + rx) * 2;
                        int32_t inputIdx    = tx * NUM_RX + rx;
                        ptr32[idx_snap]     = (inputIdx * Q20_SCALE) / 16;
                        ptr32[idx_snap + 1] = (det * Q20_SCALE) / 8;
                    }
                }
            }
        }
    }

    /* Fill azimuth and elevation bins */
    {
        NVCVTensorData azData, elData;
        float az, el;
        int idx;

        nvcvTensorExportData(*tensors->azimuthBinsTensor, &azData);
        nvcvTensorExportData(*tensors->elevationBinsTensor, &elData);
        CupvaMemGetHostPointer((void **)&azimuthBins, (void *)azData.buffer.strided.basePtr);
        CupvaMemGetHostPointer((void **)&elevationBins, (void *)elData.buffer.strided.basePtr);

        idx = 0;
        for (az = -60.0f; az <= 60.0f && idx < NUM_AZIMUTH_BINS; az += 1.5f, idx++)
        {
            azimuthBins[idx] = az;
        }

        idx = 0;
        for (el = -12.0f; el <= 30.0f && idx < NUM_ELEVATION_BINS; el += 6.0f, idx++)
        {
            elevationBins[idx] = el;
        }
    }

    /* Fill detection list with synthetic data */
    {
        NVCVTensorData detListData;
        int32_t *detListPtr;
        nvcvTensorExportData(*tensors->detectionListTensor, &detListData);
        CupvaMemGetHostPointer((void **)&detListPtr, (void *)detListData.buffer.strided.basePtr);
        for (i = 0; i < NUM_DETECTIONS; i++)
        {
            detListPtr[2 * i]     = i % 256; /* Range index */
            detListPtr[2 * i + 1] = i % 128; /* Doppler index */
        }
    }

    /* Fill DDM Doppler offsets */
    {
        NVCVTensorData ddmData;
        float *ddmPtr;
        nvcvTensorExportData(*tensors->ddmDopplerOffsetsTensor, &ddmData);
        CupvaMemGetHostPointer((void **)&ddmPtr, (void *)ddmData.buffer.strided.basePtr);
        ddmPtr[0] = 0.0f; /* No DDM offset for this test */
    }

    /* Create virtual channel positions and compute steering vectors */
    {
        NVCVTensorData vcPosData;
        int32_t txIdx, rxIdx;

        nvcvTensorExportData(*tensors->vcPosTensor, &vcPosData);
        CupvaMemGetHostPointer((void **)&vcPosPtr, (void *)vcPosData.buffer.strided.basePtr);

        for (txIdx = 0; txIdx < NUM_TX; txIdx++)
        {
            for (rxIdx = 0; rxIdx < NUM_RX; rxIdx++)
            {
                int32_t vcIdx     = txIdx * NUM_RX + rxIdx;
                vcPosPtr[vcIdx].x = txIdx * 0.5f;
                vcPosPtr[vcIdx].y = rxIdx * 0.5f;
            }
        }

        /* Compute steering vectors using C API */
        NVCV_CHECK_ERROR_GOTO(pvaDoaBartlettBeamformingComputeSteeringVectors(
                                  *tensors->vcPosTensor, *tensors->azimuthBinsTensor, *tensors->elevationBinsTensor,
                                  *tensors->steeringVectorsTensor),
                              err, Error);
    }

    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference implementation
 */
static int ValidateOutput(BartlettTensors *tensors, PVADoaBartlettBeamformingParams *params,
                          PVATargetProcessingParams *targetProcessingParams, const char *testName)
{
    int32_t err = 0;

    /* Run CPU reference DOA implementation (DOA-only, produces angles and index map) */
    doaBartlettBeamformingRef(*tensors->detectionCountTensor, *tensors->snapshotsTensor,
                              *tensors->steeringVectorsTensor, *tensors->azimuthBinsTensor,
                              *tensors->elevationBinsTensor, *tensors->refTargetCountTensor,
                              *tensors->refTargetAnglesTensor, *tensors->refTargetIndexMapTensor, params);

    /* Run CPU reference target processing to produce final target list */
    /* Pass NULL for NCI tensor since enableQuadraticInterpolation is false in this sample */
    targetProcessingRef(*tensors->refTargetCountTensor, *tensors->detectionListTensor,
                        *tensors->refTargetIndexMapTensor, *tensors->refTargetAnglesTensor,
                        *tensors->ddmDopplerOffsetsTensor, NULL /* nciTensor */, *tensors->refTargetListTensor,
                        targetProcessingParams);

    /* Compare PVA output with CPU reference */
    err = checkTargetProcessingOutput(*tensors->targetCountTensor, *tensors->targetListTensor,
                                      *tensors->refTargetListTensor);
    if (err)
    {
        printf("%s test failed\n", testName);
    }

    return err;
}

/**
 * @brief Free all tensor resources
 */
static void FreeTensors(BartlettTensors *tensors)
{
    if (*tensors->refTargetListTensor)
        nvcvTensorDecRef(*tensors->refTargetListTensor, NULL);
    if (*tensors->refTargetIndexMapTensor)
        nvcvTensorDecRef(*tensors->refTargetIndexMapTensor, NULL);
    if (*tensors->refTargetAnglesTensor)
        nvcvTensorDecRef(*tensors->refTargetAnglesTensor, NULL);
    if (*tensors->refTargetCountTensor)
        nvcvTensorDecRef(*tensors->refTargetCountTensor, NULL);
    if (*tensors->targetListTensor)
        nvcvTensorDecRef(*tensors->targetListTensor, NULL);
    if (*tensors->targetIndexMapTensor)
        nvcvTensorDecRef(*tensors->targetIndexMapTensor, NULL);
    if (*tensors->targetAnglesTensor)
        nvcvTensorDecRef(*tensors->targetAnglesTensor, NULL);
    if (*tensors->targetCountTensor)
        nvcvTensorDecRef(*tensors->targetCountTensor, NULL);
    if (*tensors->ddmDopplerOffsetsTensor)
        nvcvTensorDecRef(*tensors->ddmDopplerOffsetsTensor, NULL);
    if (*tensors->detectionListTensor)
        nvcvTensorDecRef(*tensors->detectionListTensor, NULL);
    if (*tensors->elevationBinsTensor)
        nvcvTensorDecRef(*tensors->elevationBinsTensor, NULL);
    if (*tensors->azimuthBinsTensor)
        nvcvTensorDecRef(*tensors->azimuthBinsTensor, NULL);
    if (*tensors->steeringVectorsTensor)
        nvcvTensorDecRef(*tensors->steeringVectorsTensor, NULL);
    if (*tensors->snapshotsTensor)
        nvcvTensorDecRef(*tensors->snapshotsTensor, NULL);
    if (*tensors->detectionCountTensor)
        nvcvTensorDecRef(*tensors->detectionCountTensor, NULL);
    if (*tensors->vcPosTensor)
        nvcvTensorDecRef(*tensors->vcPosTensor, NULL);
    if (*tensors->allocatorHandle)
        nvcvAllocatorDecRef(*tensors->allocatorHandle, NULL);
}

/**
 * @brief Test Bartlett Beamforming + Target Processing operation using CUDA stream
 */
int SubmitBartlettToCudaStream(PVADoaBartlettBeamformingParams *params, PVATargetProcessingParams *tpParams, int argc,
                               char **argv)
{
    /* Initialize all resource handles to NULL for safe cleanup on error */
    NVCVAllocatorHandle allocatorHandle      = NULL;
    NVCVTensorHandle detectionCountTensor    = NULL;
    NVCVTensorHandle snapshotsTensor         = NULL;
    NVCVTensorHandle steeringVectorsTensor   = NULL;
    NVCVTensorHandle azimuthBinsTensor       = NULL;
    NVCVTensorHandle elevationBinsTensor     = NULL;
    NVCVTensorHandle detectionListTensor     = NULL;
    NVCVTensorHandle ddmDopplerOffsetsTensor = NULL;
    NVCVTensorHandle targetCountTensor       = NULL;
    NVCVTensorHandle targetAnglesTensor      = NULL;
    NVCVTensorHandle targetIndexMapTensor    = NULL;
    NVCVTensorHandle targetListTensor        = NULL;
    NVCVTensorHandle vcPosTensor             = NULL;
    NVCVTensorHandle refTargetCountTensor    = NULL;
    NVCVTensorHandle refTargetAnglesTensor   = NULL;
    NVCVTensorHandle refTargetIndexMapTensor = NULL;
    NVCVTensorHandle refTargetListTensor     = NULL;
    NVCVOperatorHandle doaOperatorHandle     = NULL;
    NVCVOperatorHandle tpOperatorHandle      = NULL;
    cudaStream_t stream                      = NULL;
    int32_t err                              = 0;
    cudaError_t cudaErr                      = cudaSuccess;

    /* Tensor requirement structures (stack-allocated) */
    NVCVTensorRequirements snapshotsReqs;
    NVCVTensorRequirements detectionListReqs;
    NVCVTensorRequirements ddmDopplerOffsetsReqs;
    NVCVTensorRequirements targetAnglesReqs;
    NVCVTensorRequirements targetIndexMapReqs;
    NVCVTensorRequirements targetListReqs;
    NVCVTensorRequirements targetCountReqs;

    /* Create non-blocking CUDA stream for async execution */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaErr, FreeResources);

    /* Setup tensor structure with pointers to all handles and requirements */
    BartlettTensors tensors = {.allocatorHandle         = &allocatorHandle,
                               .detectionCountTensor    = &detectionCountTensor,
                               .snapshotsTensor         = &snapshotsTensor,
                               .steeringVectorsTensor   = &steeringVectorsTensor,
                               .azimuthBinsTensor       = &azimuthBinsTensor,
                               .elevationBinsTensor     = &elevationBinsTensor,
                               .detectionListTensor     = &detectionListTensor,
                               .ddmDopplerOffsetsTensor = &ddmDopplerOffsetsTensor,
                               .targetCountTensor       = &targetCountTensor,
                               .targetAnglesTensor      = &targetAnglesTensor,
                               .targetIndexMapTensor    = &targetIndexMapTensor,
                               .targetListTensor        = &targetListTensor,
                               .vcPosTensor             = &vcPosTensor,
                               .refTargetCountTensor    = &refTargetCountTensor,
                               .refTargetAnglesTensor   = &refTargetAnglesTensor,
                               .refTargetIndexMapTensor = &refTargetIndexMapTensor,
                               .refTargetListTensor     = &refTargetListTensor,
                               .snapshotsReqs           = &snapshotsReqs,
                               .detectionListReqs       = &detectionListReqs,
                               .ddmDopplerOffsetsReqs   = &ddmDopplerOffsetsReqs,
                               .targetAnglesReqs        = &targetAnglesReqs,
                               .targetIndexMapReqs      = &targetIndexMapReqs,
                               .targetListReqs          = &targetListReqs,
                               .targetCountReqs         = &targetCountReqs};

    /* Allocate all tensors and populate with test data */
    NVCV_CHECK_ERROR_GOTO(AllocateAndInitializeTensors(&tensors, params, argc, argv), err, FreeResources);

    /* Create Bartlett Beamforming DOA operator instance */
    NVCV_CHECK_ERROR_GOTO(
        pvaDoaBartlettBeamformingCreate(&doaOperatorHandle, &snapshotsReqs, &detectionListReqs, &ddmDopplerOffsetsReqs,
                                        &targetAnglesReqs, &targetIndexMapReqs, params),
        err, FreeResources);

    /* Create standalone Target Processing operator instance */
    /* Note: NCI tensor is NULL - quadratic interpolation disabled in this sample */
    NVCV_CHECK_ERROR_GOTO(
        pvaTargetProcessingCreate(&tpOperatorHandle, &detectionListReqs, &targetIndexMapReqs, &targetAnglesReqs,
                                  &ddmDopplerOffsetsReqs, &targetListReqs, NULL /* nciReqs */, tpParams),
        err, FreeResources);

    /* Submit DOA operator to CUDA stream */
    NVCV_CHECK_ERROR_GOTO(pvaDoaBartlettBeamformingSubmit(
                              doaOperatorHandle, stream, detectionCountTensor, snapshotsTensor, steeringVectorsTensor,
                              azimuthBinsTensor, elevationBinsTensor, detectionListTensor, ddmDopplerOffsetsTensor,
                              targetCountTensor, targetAnglesTensor, targetIndexMapTensor),
                          err, FreeResources);

    /* Submit Target Processing operator to CUDA stream */
    NVCV_CHECK_ERROR_GOTO(pvaTargetProcessingSubmit(tpOperatorHandle, stream, targetCountTensor, detectionCountTensor,
                                                    detectionListTensor, targetIndexMapTensor, targetAnglesTensor,
                                                    ddmDopplerOffsetsTensor, NULL /* nciTensor */, targetListTensor),
                          err, FreeResources);

    /* Wait for all operations on the stream to complete */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), cudaErr, FreeResources);

    /* Run CPU reference and compare results */
    err = ValidateOutput(&tensors, params, tpParams, "Submit to CUDA stream");

FreeResources: /* Clean up all allocated resources (safe to call with NULL handles) */
    cudaStreamDestroy(stream);
    nvcvOperatorDestroy(tpOperatorHandle);
    nvcvOperatorDestroy(doaOperatorHandle);
    FreeTensors(&tensors);

    return err;
}

/**
 * @brief Test Bartlett Beamforming + Target Processing operation using cuPVA stream
 */
int SubmitBartlettToCupvaStream(PVADoaBartlettBeamformingParams *params, PVATargetProcessingParams *tpParams, int argc,
                                char **argv)
{
    /* Initialize all resource handles to NULL for safe cleanup on error */
    NVCVAllocatorHandle allocatorHandle      = NULL;
    NVCVTensorHandle detectionCountTensor    = NULL;
    NVCVTensorHandle snapshotsTensor         = NULL;
    NVCVTensorHandle steeringVectorsTensor   = NULL;
    NVCVTensorHandle azimuthBinsTensor       = NULL;
    NVCVTensorHandle elevationBinsTensor     = NULL;
    NVCVTensorHandle detectionListTensor     = NULL;
    NVCVTensorHandle ddmDopplerOffsetsTensor = NULL;
    NVCVTensorHandle targetCountTensor       = NULL;
    NVCVTensorHandle targetAnglesTensor      = NULL;
    NVCVTensorHandle targetIndexMapTensor    = NULL;
    NVCVTensorHandle targetListTensor        = NULL;
    NVCVTensorHandle vcPosTensor             = NULL;
    NVCVTensorHandle refTargetCountTensor    = NULL;
    NVCVTensorHandle refTargetAnglesTensor   = NULL;
    NVCVTensorHandle refTargetIndexMapTensor = NULL;
    NVCVTensorHandle refTargetListTensor     = NULL;
    NVCVOperatorHandle doaOperatorHandle     = NULL;
    NVCVOperatorHandle tpOperatorHandle      = NULL;
    cupvaSyncObj_t sync                      = NULL;
    cupvaStream_t stream                     = NULL;
    int32_t err                              = 0;

    /* Tensor requirement structures (stack-allocated) */
    NVCVTensorRequirements snapshotsReqs;
    NVCVTensorRequirements detectionListReqs;
    NVCVTensorRequirements ddmDopplerOffsetsReqs;
    NVCVTensorRequirements targetAnglesReqs;
    NVCVTensorRequirements targetIndexMapReqs;
    NVCVTensorRequirements targetListReqs;
    NVCVTensorRequirements targetCountReqs;

    /* Setup tensor structure with pointers to all handles and requirements */
    BartlettTensors tensors = {.allocatorHandle         = &allocatorHandle,
                               .detectionCountTensor    = &detectionCountTensor,
                               .snapshotsTensor         = &snapshotsTensor,
                               .steeringVectorsTensor   = &steeringVectorsTensor,
                               .azimuthBinsTensor       = &azimuthBinsTensor,
                               .elevationBinsTensor     = &elevationBinsTensor,
                               .detectionListTensor     = &detectionListTensor,
                               .ddmDopplerOffsetsTensor = &ddmDopplerOffsetsTensor,
                               .targetCountTensor       = &targetCountTensor,
                               .targetAnglesTensor      = &targetAnglesTensor,
                               .targetIndexMapTensor    = &targetIndexMapTensor,
                               .targetListTensor        = &targetListTensor,
                               .vcPosTensor             = &vcPosTensor,
                               .refTargetCountTensor    = &refTargetCountTensor,
                               .refTargetAnglesTensor   = &refTargetAnglesTensor,
                               .refTargetIndexMapTensor = &refTargetIndexMapTensor,
                               .refTargetListTensor     = &refTargetListTensor,
                               .snapshotsReqs           = &snapshotsReqs,
                               .detectionListReqs       = &detectionListReqs,
                               .ddmDopplerOffsetsReqs   = &ddmDopplerOffsetsReqs,
                               .targetAnglesReqs        = &targetAnglesReqs,
                               .targetIndexMapReqs      = &targetIndexMapReqs,
                               .targetListReqs          = &targetListReqs,
                               .targetCountReqs         = &targetCountReqs};

    /* Allocate all tensors and populate with test data */
    NVCV_CHECK_ERROR_GOTO(AllocateAndInitializeTensors(&tensors, params, argc, argv), err, FreeResources);

    /* Create Bartlett Beamforming DOA operator instance */
    NVCV_CHECK_ERROR_GOTO(
        pvaDoaBartlettBeamformingCreate(&doaOperatorHandle, &snapshotsReqs, &detectionListReqs, &ddmDopplerOffsetsReqs,
                                        &targetAnglesReqs, &targetIndexMapReqs, params),
        err, FreeResources);

    /* Create standalone Target Processing operator instance */
    /* Note: NCI tensor is NULL - quadratic interpolation disabled in this sample */
    NVCV_CHECK_ERROR_GOTO(
        pvaTargetProcessingCreate(&tpOperatorHandle, &detectionListReqs, &targetIndexMapReqs, &targetAnglesReqs,
                                  &ddmDopplerOffsetsReqs, &targetListReqs, NULL /* nciReqs */, tpParams),
        err, FreeResources);

    /* Create synchronization object for host-PVA coordination (signaler and waiter mode) */
    CUPVA_CHECK_ERROR_GOTO(CupvaSyncObjCreate(&sync, false, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD), err,
                           FreeResources);

    /* Initialize fence for task completion notification */
    cupvaFence_t fence;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceInit(&fence, sync), err, FreeResources);

    /* Create request-fences command to signal fence after task completion */
    cupvaCmd_t rf;
    CUPVA_CHECK_ERROR_GOTO(CupvaCmdRequestFencesInit(&rf, &fence, 1), err, FreeResources);

    /* Create cuPVA stream on PVA0 hardware using any available VPU */
    CUPVA_CHECK_ERROR_GOTO(CupvaStreamCreate(&stream, CUPVA_PVA0, CUPVA_VPU_ANY), err, FreeResources);

    /* Prepare command array and status array for stream submission */
    cupvaCmd_t const *cmds[1];
    cmds[0] = &rf;
    cupvaCmdStatus_t status[1];
    status[0] = NULL;

    /* Submit DOA operator to cuPVA stream */
    NVCV_CHECK_ERROR_GOTO(pvaDoaBartlettBeamformingSubmit(
                              doaOperatorHandle, stream, detectionCountTensor, snapshotsTensor, steeringVectorsTensor,
                              azimuthBinsTensor, elevationBinsTensor, detectionListTensor, ddmDopplerOffsetsTensor,
                              targetCountTensor, targetAnglesTensor, targetIndexMapTensor),
                          err, FreeResources);

    /* Submit Target Processing operator to cuPVA stream */
    NVCV_CHECK_ERROR_GOTO(pvaTargetProcessingSubmit(tpOperatorHandle, stream, targetCountTensor, detectionCountTensor,
                                                    detectionListTensor, targetIndexMapTensor, targetAnglesTensor,
                                                    ddmDopplerOffsetsTensor, NULL /* nciTensor */, targetListTensor),
                          err, FreeResources);

    /* Submit fence command to stream (in-order execution, no timeout) */
    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    /* Wait for fence to signal (blocks until PVA task completes) */
    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    /* Run CPU reference and compare results */
    err = ValidateOutput(&tensors, params, tpParams, "Submit to cuPVA stream");

FreeResources: /* Clean up all allocated resources (safe to call with NULL handles) */
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvOperatorDestroy(tpOperatorHandle);
    nvcvOperatorDestroy(doaOperatorHandle);
    FreeTensors(&tensors);

    return err;
}

int main(int argc, char **argv)
{
    /* Parse command-line option: --separable enables separable Bartlett mode */
    int enableSeparable = 0;
    int i;
    for (i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--separable") == 0)
        {
            enableSeparable = 1;
        }
    }

    /* Configure Bartlett Beamforming DOA operator parameters */
    PVADoaBartlettBeamformingParams params;
    memset(&params, 0, sizeof(params));
    params.enableInterpolation = true; /* Enable quadratic sub-bin interpolation */
    params.enablePowerOutput   = true; /* Enable peak power (dB) output */

    /* Configure standalone Target Processing operator parameters */
    PVATargetProcessingParams tpParams;
    memset(&tpParams, 0, sizeof(tpParams));
    tpParams.wavelength                   = 0.004f;  /* 4mm, 77GHz */
    tpParams.sweepTime                    = 100e-6f; /* 100 microseconds */
    tpParams.rangeResolution              = 0.1f;    /* 0.1 meters */
    tpParams.numChirps                    = 256;
    tpParams.fftShiftedDoppler            = false;
    tpParams.enableRvDecoupling           = true;
    tpParams.enableQuadraticInterpolation = false; /* Requires NCI input (not yet supported) */

    /* Base FMCW radar parameters for RV decoupling (derived params computed internally)
     * Example values for 77GHz automotive FMCW radar:
     * - fs (sampling frequency) = 10 MHz
     * - nofSamples = 256
     * - PRI (pulse repetition interval) = 100e-6 s
     * - slope = 20 MHz/us = 20e12 Hz/s
     * - c = 3e8 m/s (speed of light)
     * - deltaV = 3.6 km/h (velocity resolution)
     */
    float fs     = 10e6f;   /* 10 MHz sampling rate */
    float PRI    = 100e-6f; /* 100 us PRI */
    float slope  = 20e12f;  /* 20 MHz/us chirp slope */
    float c      = 3e8f;    /* speed of light */
    float deltaV = 3.6f;    /* velocity resolution in km/h */

    tpParams.fs         = fs;                  /* Sampling frequency (Hz) */
    tpParams.nofSamples = 256;                 /* Samples per chirp */
    tpParams.PRI        = PRI;                 /* Pulse Repetition Interval (seconds) */
    tpParams.deltaV     = deltaV;              /* Velocity resolution (km/h) */
    tpParams.contR      = c / (2.0f * slope);  /* Range coupling: c / (2 * slope) */
    tpParams.contVFast  = -c / (2.0f * 77e9f); /* Velocity fast coupling */
    tpParams.contVSlow  = c / (2.0f * 77e9f);  /* Velocity slow coupling */

    /* Separable Bartlett mode: 2-step search (azimuth at el~0, then neighborhood x all elevations)
     * Reduces complexity from O(numAz * numEl) to O(numAz + (2*N+1) * numEl)
     * Enable with: --separable */
    if (enableSeparable)
    {
        params.separableParams.enableSeparable = true;
        params.separableParams.azNeighborhood  = PVA_BARTLETT_AZ_NEIGHBORHOOD_2; /* Search ±2 bins around peak */
        params.separableParams.azSource        = PVA_BARTLETT_AZ_FROM_EL0;       /* Use azimuth from Step 1 */
        printf("Separable Bartlett mode enabled (azNeighborhood=%d)\n", (int)params.separableParams.azNeighborhood);
    }

    /* Initialize test result variables */
    int32_t cudaStreamSubmitTestResult  = 0;
    int32_t cupvaStreamSubmitTestResult = 0;
    cudaError_t cudaErr                 = cudaSuccess;

#if CUPVA_HAS_CUDA_STREAM_SUPPORT == 1

    /* Initialize CUDA device (required for CUDA stream submission) */
    CUDA_CHECK_ERROR_GOTO(cudaSetDevice(0), cudaErr, CheckResults);

    /* Check cuPVA driver version (CUDA stream support requires v2007+) */
    uint32_t driverVersion;
    CUPVA_CHECK_ERROR_GOTO(CupvaGetDriverVersion(&driverVersion), cudaStreamSubmitTestResult, CheckResults);
    if (driverVersion >= 2007)
    {
        /* Run test using CUDA stream (allows interop with CUDA operations) */
        cudaStreamSubmitTestResult = SubmitBartlettToCudaStream(&params, &tpParams, argc, argv);
    }
#endif

    /* Run test using native cuPVA stream (direct PVA control, lower latency) */
    cupvaStreamSubmitTestResult = SubmitBartlettToCupvaStream(&params, &tpParams, argc, argv);

CheckResults:
    /* Print overall test result (pass if both tests passed) */
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
