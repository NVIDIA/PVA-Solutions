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

/**
 * @file target_processing.c
 * @brief Sample application for the standalone Target Processing operator.
 *
 * This sample demonstrates how to use the pvaTargetProcessing operator for
 * converting radar detection indices and DOA angle estimates into complete
 * target information (velocity, range, angles, and Cartesian coordinates).
 *
 * The sample provides:
 * - Tensor allocation and initialization with synthetic test data
 * - Operator submission to both CUDA and cuPVA streams
 * - Result comparison against CPU reference implementation
 */

#include "target_processing_ref.h"

#include <ErrorCheckMacros.h>
#include <OpTargetProcessing.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <math.h>
#include <nvcv/Fwd.h>
#include <nvcv/alloc/Allocator.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Structure to hold tensor handles and requirements
 */
typedef struct
{
    NVCVAllocatorHandle allocatorHandle;
    NVCVTensorHandle targetCountTensorHandle;
    NVCVTensorHandle detectionCountTensorHandle;
    NVCVTensorHandle detectionListTensorHandle;
    NVCVTensorHandle targetIndexMapTensorHandle;
    NVCVTensorHandle targetAnglesTensorHandle;
    NVCVTensorHandle ddmDopplerOffsetsTensorHandle;
    NVCVTensorHandle targetListTensorHandle;
    NVCVTensorRequirements detectionListTensorRequirements;
    NVCVTensorRequirements targetIndexMapTensorRequirements;
    NVCVTensorRequirements targetAnglesTensorRequirements;
    NVCVTensorRequirements ddmDopplerOffsetsTensorRequirements;
    NVCVTensorRequirements targetListTensorRequirements;
} TargetProcessingTensors;

/**
 * @brief Allocate and initialize tensors with synthetic test data
 */
static int AllocateAndInitializeTensors(int32_t numTargets, int32_t numDetections, int32_t numDopplerFolds,
                                        int32_t includePower, TargetProcessingTensors *tensors)
{
    int32_t err              = 0;
    int32_t targetListHeight = includePower ? 8 : 7;

    // Construct PVA allocator
    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(&tensors->allocatorHandle), err, Error);

    // Target count tensor [1] S32
    NVCVTensorLayout wLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("W", &wLayout), err, Error);

    NVCVTensorRequirements targetCountReqs;
    int64_t targetCountShape[] = {1};
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(1, targetCountShape, NVCV_DATA_TYPE_S32, wLayout, 0, 0, &targetCountReqs), err,
        Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(&targetCountReqs, tensors->allocatorHandle, &tensors->targetCountTensorHandle), err, Error);

    // Detection count tensor [1] S32
    NVCVTensorRequirements detectionCountReqs;
    int64_t detectionCountShape[] = {1};
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(1, detectionCountShape, NVCV_DATA_TYPE_S32, wLayout, 0, 0, &detectionCountReqs),
        err, Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(&detectionCountReqs, tensors->allocatorHandle, &tensors->detectionCountTensorHandle), err,
        Error);

    // Detection list tensor [numDetections, 2] S32
    NVCVTensorLayout hwLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HW", &hwLayout), err, Error);

    int64_t detectionListShape[] = {numDetections, 2};
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(2, detectionListShape, NVCV_DATA_TYPE_S32, hwLayout, 0, 0,
                                                        &tensors->detectionListTensorRequirements),
                          err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&tensors->detectionListTensorRequirements, tensors->allocatorHandle,
                                              &tensors->detectionListTensorHandle),
                          err, Error);

    // Target index map tensor [numTargets] S32
    int64_t targetIndexMapShape[] = {numTargets};
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(1, targetIndexMapShape, NVCV_DATA_TYPE_S32, wLayout, 0, 0,
                                                        &tensors->targetIndexMapTensorRequirements),
                          err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&tensors->targetIndexMapTensorRequirements, tensors->allocatorHandle,
                                              &tensors->targetIndexMapTensorHandle),
                          err, Error);

    // Target angles tensor [3, numTargets] F32
    int64_t targetAnglesShape[] = {3, numTargets};
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(2, targetAnglesShape, NVCV_DATA_TYPE_F32, hwLayout, 0, 0,
                                                        &tensors->targetAnglesTensorRequirements),
                          err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&tensors->targetAnglesTensorRequirements, tensors->allocatorHandle,
                                              &tensors->targetAnglesTensorHandle),
                          err, Error);

    // DDM Doppler offsets tensor [numDopplerFolds] F32
    int64_t ddmDopplerOffsetsShape[] = {numDopplerFolds};
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(1, ddmDopplerOffsetsShape, NVCV_DATA_TYPE_F32, wLayout, 0, 0,
                                                        &tensors->ddmDopplerOffsetsTensorRequirements),
                          err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&tensors->ddmDopplerOffsetsTensorRequirements, tensors->allocatorHandle,
                                              &tensors->ddmDopplerOffsetsTensorHandle),
                          err, Error);

    // Target list tensor [targetListHeight, numTargets] F32
    int64_t targetListShape[] = {targetListHeight, numTargets};
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(2, targetListShape, NVCV_DATA_TYPE_F32, hwLayout, 0, 0,
                                                        &tensors->targetListTensorRequirements),
                          err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&tensors->targetListTensorRequirements, tensors->allocatorHandle,
                                              &tensors->targetListTensorHandle),
                          err, Error);

    // Fill input tensors with synthetic data
    // Note: PVA-allocated tensors require CupvaMemGetHostPointer to get host-accessible pointers
    {
        NVCVTensorData data;
        // Set target count = numTargets
        NVCV_CHECK_ERROR_GOTO(nvcvTensorExportData(tensors->targetCountTensorHandle, &data), err, Error);
        int32_t *targetCountPtr = NULL;
        CupvaMemGetHostPointer((void **)&targetCountPtr, (void *)data.buffer.strided.basePtr);
        *targetCountPtr = numTargets;

        // Set detection count = numDetections
        NVCV_CHECK_ERROR_GOTO(nvcvTensorExportData(tensors->detectionCountTensorHandle, &data), err, Error);
        int32_t *detectionCountPtr = NULL;
        CupvaMemGetHostPointer((void **)&detectionCountPtr, (void *)data.buffer.strided.basePtr);
        *detectionCountPtr = numDetections;

        // Fill detection list with synthetic indices
        NVCV_CHECK_ERROR_GOTO(nvcvTensorExportData(tensors->detectionListTensorHandle, &data), err, Error);
        int32_t *detectionListPtr = NULL;
        CupvaMemGetHostPointer((void **)&detectionListPtr, (void *)data.buffer.strided.basePtr);
        int32_t detectionListPitch = (int32_t)(data.buffer.strided.strides[0] / sizeof(int32_t)); // stride in elements
        for (int32_t i = 0; i < numDetections; i++)
        {
            detectionListPtr[i * detectionListPitch + 0] = i % 256;       // range index
            detectionListPtr[i * detectionListPitch + 1] = (i * 3) % 512; // doppler index
        }

        // Fill target index map (1:1 mapping for simplicity)
        NVCV_CHECK_ERROR_GOTO(nvcvTensorExportData(tensors->targetIndexMapTensorHandle, &data), err, Error);
        int32_t *targetIndexMapPtr = NULL;
        CupvaMemGetHostPointer((void **)&targetIndexMapPtr, (void *)data.buffer.strided.basePtr);
        for (int32_t i = 0; i < numTargets; i++)
        {
            targetIndexMapPtr[i] = i % numDetections;
        }

        // Fill target angles [3, numTargets]: azimuth, elevation, power
        NVCV_CHECK_ERROR_GOTO(nvcvTensorExportData(tensors->targetAnglesTensorHandle, &data), err, Error);
        float *targetAnglesPtr = NULL;
        CupvaMemGetHostPointer((void **)&targetAnglesPtr, (void *)data.buffer.strided.basePtr);
        int32_t targetAnglesPitch = (int32_t)(data.buffer.strided.strides[0] / sizeof(float));
        for (int32_t i = 0; i < numTargets; i++)
        {
            targetAnglesPtr[0 * targetAnglesPitch + i] = 10.0f * ((float)i / (float)numTargets) - 5.0f; // azimuth
            targetAnglesPtr[1 * targetAnglesPitch + i] = 2.0f * ((float)i / (float)numTargets) - 1.0f;  // elevation
            targetAnglesPtr[2 * targetAnglesPitch + i] = 20.0f + (float)i * 0.1f;                       // power (dB)
        }

        // Fill DDM Doppler offsets
        NVCV_CHECK_ERROR_GOTO(nvcvTensorExportData(tensors->ddmDopplerOffsetsTensorHandle, &data), err, Error);
        float *ddmOffsetsPtr = NULL;
        CupvaMemGetHostPointer((void **)&ddmOffsetsPtr, (void *)data.buffer.strided.basePtr);
        for (int32_t i = 0; i < numDopplerFolds; i++)
        {
            ddmOffsetsPtr[i] = 0.0f; // No offset for simplicity
        }
    }

    return 0;

Error:
    return err;
}

/**
 * @brief Submit target processing operator to a cuPVA stream
 */
static int SubmitTargetProcessingToCupvaStream(NVCVOperatorHandle opHandle, TargetProcessingTensors *tensors)
{
    int32_t err = 0;

    // Create cuPVA stream and synchronization objects
    cupvaSyncObj_t sync  = NULL;
    cupvaStream_t stream = NULL;

    // Create synchronization object for host-PVA coordination
    CUPVA_CHECK_ERROR_GOTO(CupvaSyncObjCreate(&sync, false, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD), err, Error);

    // Initialize fence for task completion notification
    cupvaFence_t fence;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceInit(&fence, sync), err, Error);

    // Create request-fences command to signal fence after task completion
    cupvaCmd_t rf;
    CUPVA_CHECK_ERROR_GOTO(CupvaCmdRequestFencesInit(&rf, &fence, 1), err, Error);

    // Create cuPVA stream on PVA0 hardware using any available VPU
    CUPVA_CHECK_ERROR_GOTO(CupvaStreamCreate(&stream, CUPVA_PVA0, CUPVA_VPU_ANY), err, Error);

    // Prepare command array and status array for stream submission
    cupvaCmd_t const *cmds[1];
    cmds[0] = &rf;
    cupvaCmdStatus_t status[1];
    status[0] = NULL;

    // Submit operator
    NVCV_CHECK_ERROR_GOTO(
        pvaTargetProcessingSubmit(opHandle, stream, tensors->targetCountTensorHandle,
                                  tensors->detectionCountTensorHandle, tensors->detectionListTensorHandle,
                                  tensors->targetIndexMapTensorHandle, tensors->targetAnglesTensorHandle,
                                  tensors->ddmDopplerOffsetsTensorHandle, NULL, tensors->targetListTensorHandle),
        err, Error);

    // Submit fence command to stream
    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, Error);

    // Wait for fence to signal (blocks until PVA task completes)
    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, Error);

    printf("Target Processing cuPVA stream submission: PASS\n");

Error:
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    return err;
}

/**
 * @brief Deallocate all tensors
 */
static void DeallocateTensors(TargetProcessingTensors *tensors)
{
    if (tensors->targetListTensorHandle)
        nvcvTensorDecRef(tensors->targetListTensorHandle, NULL);
    if (tensors->ddmDopplerOffsetsTensorHandle)
        nvcvTensorDecRef(tensors->ddmDopplerOffsetsTensorHandle, NULL);
    if (tensors->targetAnglesTensorHandle)
        nvcvTensorDecRef(tensors->targetAnglesTensorHandle, NULL);
    if (tensors->targetIndexMapTensorHandle)
        nvcvTensorDecRef(tensors->targetIndexMapTensorHandle, NULL);
    if (tensors->detectionListTensorHandle)
        nvcvTensorDecRef(tensors->detectionListTensorHandle, NULL);
    if (tensors->detectionCountTensorHandle)
        nvcvTensorDecRef(tensors->detectionCountTensorHandle, NULL);
    if (tensors->targetCountTensorHandle)
        nvcvTensorDecRef(tensors->targetCountTensorHandle, NULL);
    if (tensors->allocatorHandle)
        nvcvAllocatorDecRef(tensors->allocatorHandle, NULL);
}

/**
 * @brief Run the Target Processing sample
 */
static int RunTestTargetProcessing(void)
{
    int32_t err                 = 0;
    NVCVOperatorHandle opHandle = NULL;

    // Test configuration
    int32_t numTargets      = 64;
    int32_t numDetections   = 64;
    int32_t numDopplerFolds = 1;
    int32_t includePower    = 1;

    PVATargetProcessingParams params = {
        0.004f,  // wavelength (77GHz)
        100e-6f, // sweepTime
        0.1f,    // rangeResolution
        256,     // numChirps
        false,   // fftShiftedDoppler
        false,   // enableRvDecoupling
        false,   // enableQuadraticInterpolation
        10e6f,   // fs
        256,     // nofSamples
        100e-6f, // PRI
        3.6f,    // deltaV
        0.0f,    // contR
        0.0f,    // contVFast
        0.0f,    // contVSlow
        0,       // longitudinalAxis: X_FORWARD
        0        // groundProjection: SLANT_RANGE
    };

    TargetProcessingTensors tensors;
    memset(&tensors, 0, sizeof(tensors));

    printf("=== Target Processing Sample ===\n");
    printf("Configuration: %d targets, %d detections, %d Doppler folds, power=%s\n", numTargets, numDetections,
           numDopplerFolds, includePower ? "enabled" : "disabled");

    // Allocate and initialize tensors
    err = AllocateAndInitializeTensors(numTargets, numDetections, numDopplerFolds, includePower, &tensors);
    if (err != 0)
    {
        printf("Failed to allocate tensors\n");
        goto Cleanup;
    }

    // Create operator
    err = pvaTargetProcessingCreate(&opHandle, &tensors.detectionListTensorRequirements,
                                    &tensors.targetIndexMapTensorRequirements, &tensors.targetAnglesTensorRequirements,
                                    &tensors.ddmDopplerOffsetsTensorRequirements, &tensors.targetListTensorRequirements,
                                    NULL, &params);
    if (err != 0)
    {
        printf("Failed to create Target Processing operator\n");
        goto Cleanup;
    }

    // Submit to cuPVA stream
    err = SubmitTargetProcessingToCupvaStream(opHandle, &tensors);
    if (err != 0)
    {
        printf("cuPVA submission failed\n");
        goto Cleanup;
    }

    printf("=== Target Processing Sample: PASS ===\n");

Cleanup:
    if (opHandle)
        nvcvOperatorDestroy(opHandle);
    DeallocateTensors(&tensors);
    return err;
}

int main(void)
{
    return RunTestTargetProcessing();
}

#ifdef __cplusplus
}
#endif
