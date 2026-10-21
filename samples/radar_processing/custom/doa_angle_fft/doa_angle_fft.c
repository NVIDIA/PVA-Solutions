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
 * @file doa_angle_fft.c
 * @brief Sample application and test harness for DOA Angle FFT operator
 *
 * This file demonstrates how to use the pvaDoaAngleFFT operator for
 * Direction of Arrival estimation using 2D FFT-based beamforming on
 * radar detection snapshots.
 *
 * The sample provides:
 * - Tensor allocation and initialization
 * - Reference CPU implementation for validation
 * - Operator submission to both CUDA and cuPVA streams
 * - Result comparison and accuracy checking
 *
 * Key Functions:
 * - AllocateAndInitializeTensors(): Sets up input/output tensors
 * - SubmitDoaAngleFFTToCudaStream(): Submits operator to CUDA stream
 * - SubmitDoaAngleFFTToCupvaStream(): Submits operator to cuPVA stream
 * - VerifyDoaAngleFFTResults(): Validates PVA results against CPU reference
 * - RunTestDoaAngleFFT(): Main test entry point
 */

#include "doa_angle_fft_ref.h"

#include <ErrorCheckMacros.h>
#include <ImageIO.h>
#include <OpDoaAngleFFT.h>
#include <OpTargetProcessing.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <limits.h>
#include <nvcv/Fwd.h>
#include <nvcv/alloc/Allocator.h>
#include <stdlib.h>
#include <target_processing_ref.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Structure to hold tensor handles and requirements
 */
typedef struct
{
    NVCVAllocatorHandle *allocatorHandle;
    NVCVTensorHandle *inDetectionCountTensorHandle;
    NVCVTensorHandle *inSnapshotsTensorHandle;
    NVCVTensorHandle *inDetectionListTensorHandle;     // Detection list (range/doppler indices)
    NVCVTensorHandle *inDdmDopplerOffsetsTensorHandle; // DDM Doppler offsets
    NVCVTensorHandle *outTargetCountTensorHandle;
    NVCVTensorHandle *outTargetAnglesTensorHandle;   // Intermediate: DOA angles [3, N] F32
    NVCVTensorHandle *outTargetIndexMapTensorHandle; // Intermediate: target-to-detection map [N] S32
    NVCVTensorHandle *outTargetListTensorHandle;
    NVCVTensorHandle *refTargetCountTensorHandle;
    NVCVTensorHandle *refTargetAnglesTensorHandle;   // Reference: DOA angles
    NVCVTensorHandle *refTargetIndexMapTensorHandle; // Reference: target-to-detection map
    NVCVTensorHandle *refTargetListTensorHandle;     // Reference output
    NVCVTensorRequirements *inDetectionCountTensorRequirements;
    NVCVTensorRequirements *inSnapshotsTensorRequirements;
    NVCVTensorRequirements *inDetectionListTensorRequirements;
    NVCVTensorRequirements *inDdmDopplerOffsetsTensorRequirements;
    NVCVTensorRequirements *outTargetCountTensorRequirements;
    NVCVTensorRequirements *outTargetAnglesTensorRequirements;
    NVCVTensorRequirements *outTargetIndexMapTensorRequirements;
    NVCVTensorRequirements *outTargetListTensorRequirements;
} DoaAngleFFTTensors;

/**
 * @brief Allocate and initialize tensors
 */
static int AllocateAndInitializeTensors(int32_t NofRx, int32_t NofTx, int32_t NofTargets, int32_t NofTargetAngleParams,
                                        int32_t NofDetections, PVADoaAngleFFTParams *params,
                                        DoaAngleFFTTensors *tensors, bool includePower)
{
    int32_t err = 0;

    // Define data types for all tensors
    NVCVDataType inDetectionCountDataType = NVCV_DATA_TYPE_S32;  // Single int32 scalar
    NVCVDataType inSnapshotsDataType      = NVCV_DATA_TYPE_2S32; // Complex int16 (real + imaginary)
    NVCVDataType outTargetCountDataType   = NVCV_DATA_TYPE_S32;  // Single int32 scalar

    // Define tensor ranks (number of dimensions)
    int32_t inDetectionCountTensorRank = 1; // Scalar (1D)
    int32_t inSnapshotsRank            = 3; // 3D: [detections, elevation, azimuth]
    int32_t outTargetCountTensorRank   = 1; // Scalar (1D)

    // Define tensor shapes
    int64_t inDetectionCountTensorShape[] = {1};                        // Single scalar
    int64_t inSnapshotsShape[]            = {NofTargets, NofTx, NofRx}; // [N, H, W]
    int64_t outTargetCountTensorShape[]   = {1};                        // Single scalar

    // Construct PVA-specific allocator
    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(tensors->allocatorHandle), err, Error);

    // Create and allocate input detection count tensor (scalar)
    NVCVTensorLayout inDetectionCountTensorLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("W", &inDetectionCountTensorLayout), err, Error);

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(inDetectionCountTensorRank, inDetectionCountTensorShape, inDetectionCountDataType,
                                      inDetectionCountTensorLayout, 0, 0, tensors->inDetectionCountTensorRequirements),
        err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensors->inDetectionCountTensorRequirements, *tensors->allocatorHandle,
                                              tensors->inDetectionCountTensorHandle),
                          err, Error);

    // Create and allocate input snapshots tensor (CHW layout: [detections, elevation, azimuth])
    NVCVTensorLayout inSnapshotsLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("CHW", &inSnapshotsLayout), err, Error);

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(inSnapshotsRank, inSnapshotsShape, inSnapshotsDataType, inSnapshotsLayout, 0, 0,
                                      tensors->inSnapshotsTensorRequirements),
        err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensors->inSnapshotsTensorRequirements, *tensors->allocatorHandle,
                                              tensors->inSnapshotsTensorHandle),
                          err, Error);

    // Create and allocate output target count tensor (scalar)
    NVCVTensorLayout outTargetCountTensorLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("W", &outTargetCountTensorLayout), err, Error);

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(outTargetCountTensorRank, outTargetCountTensorShape, outTargetCountDataType,
                                      outTargetCountTensorLayout, 0, 0, tensors->outTargetCountTensorRequirements),
        err, Error);

    // Allocate both PVA output and reference (CPU) tensors for target count
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensors->outTargetCountTensorRequirements, *tensors->allocatorHandle,
                                              tensors->outTargetCountTensorHandle),
                          err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensors->outTargetCountTensorRequirements, *tensors->allocatorHandle,
                                              tensors->refTargetCountTensorHandle),
                          err, Error);

    // Create and allocate detection list tensor [NofDetections, 2] - range/doppler indices
    NVCVDataType detectionListDataType = NVCV_DATA_TYPE_S32;
    int32_t detectionListRank          = 2;
    int64_t detectionListShape[]       = {NofDetections == 0 ? 1 : NofDetections, 2};
    NVCVTensorLayout detectionListLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HW", &detectionListLayout), err, Error);

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(detectionListRank, detectionListShape, detectionListDataType, detectionListLayout,
                                      0, 0, tensors->inDetectionListTensorRequirements),
        err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensors->inDetectionListTensorRequirements, *tensors->allocatorHandle,
                                              tensors->inDetectionListTensorHandle),
                          err, Error);

    // Create and allocate DDM Doppler offsets tensor [1] - single offset
    NVCVDataType ddmDopplerOffsetsDataType = NVCV_DATA_TYPE_F32;
    int32_t ddmDopplerOffsetsRank          = 1;
    int64_t ddmDopplerOffsetsShape[]       = {1};
    NVCVTensorLayout ddmDopplerOffsetsLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("W", &ddmDopplerOffsetsLayout), err, Error);

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(ddmDopplerOffsetsRank, ddmDopplerOffsetsShape, ddmDopplerOffsetsDataType,
                                      ddmDopplerOffsetsLayout, 0, 0, tensors->inDdmDopplerOffsetsTensorRequirements),
        err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensors->inDdmDopplerOffsetsTensorRequirements, *tensors->allocatorHandle,
                                              tensors->inDdmDopplerOffsetsTensorHandle),
                          err, Error);

    // Create and allocate target angles tensor [3, NofTargets] - DOA intermediate output
    NVCVDataType targetAnglesDataType = NVCV_DATA_TYPE_F32;
    int32_t targetAnglesRank          = 2;
    int64_t targetAnglesShape[]       = {3, NofTargets};
    NVCVTensorLayout targetAnglesLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HW", &targetAnglesLayout), err, Error);

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(targetAnglesRank, targetAnglesShape, targetAnglesDataType, targetAnglesLayout, 0,
                                      0, tensors->outTargetAnglesTensorRequirements),
        err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensors->outTargetAnglesTensorRequirements, *tensors->allocatorHandle,
                                              tensors->outTargetAnglesTensorHandle),
                          err, Error);

    // Create and allocate target index map tensor [NofTargets] - DOA intermediate output
    NVCVDataType targetIndexMapDataType = NVCV_DATA_TYPE_S32;
    int32_t targetIndexMapRank          = 1;
    int64_t targetIndexMapShape[]       = {NofTargets};
    NVCVTensorLayout targetIndexMapLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("W", &targetIndexMapLayout), err, Error);

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(targetIndexMapRank, targetIndexMapShape, targetIndexMapDataType,
                                      targetIndexMapLayout, 0, 0, tensors->outTargetIndexMapTensorRequirements),
        err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensors->outTargetIndexMapTensorRequirements, *tensors->allocatorHandle,
                                              tensors->outTargetIndexMapTensorHandle),
                          err, Error);

    // Create and allocate target list tensor [7 or 8, NofTargets]
    NVCVDataType targetListDataType = NVCV_DATA_TYPE_F32;
    int32_t targetListRank          = 2;
    int64_t targetListShape[]       = {includePower ? 8 : 7, NofTargets};
    NVCVTensorLayout targetListLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HW", &targetListLayout), err, Error);

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(targetListRank, targetListShape, targetListDataType, targetListLayout, 0, 0,
                                      tensors->outTargetListTensorRequirements),
        err, Error);

    // Allocate both PVA output and reference (CPU) tensors for target list
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensors->outTargetListTensorRequirements, *tensors->allocatorHandle,
                                              tensors->outTargetListTensorHandle),
                          err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensors->outTargetListTensorRequirements, *tensors->allocatorHandle,
                                              tensors->refTargetListTensorHandle),
                          err, Error);

    // Allocate reference (CPU) tensors for DOA intermediate outputs
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensors->outTargetAnglesTensorRequirements, *tensors->allocatorHandle,
                                              tensors->refTargetAnglesTensorHandle),
                          err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensors->outTargetIndexMapTensorRequirements, *tensors->allocatorHandle,
                                              tensors->refTargetIndexMapTensorHandle),
                          err, Error);

    // Generate synthetic input data for testing
    generateDoaAngleFFTInput(*tensors->inSnapshotsTensorHandle, *tensors->inDetectionCountTensorHandle, NofDetections);
    // Initialize detectionList with dummy range/doppler indices
    NVCVTensorData detectionListData;
    nvcvTensorExportData(*tensors->inDetectionListTensorHandle, &detectionListData);
    int32_t *detectionListPtr = NULL;
    CupvaMemGetHostPointer((void **)&detectionListPtr, (void *)detectionListData.buffer.strided.basePtr);
    generateDetectionListInput(*tensors->inDetectionListTensorHandle);

    // Initialize DDM Doppler offsets
    NVCVTensorData ddmData;
    nvcvTensorExportData(*tensors->inDdmDopplerOffsetsTensorHandle, &ddmData);
    float *ddmPtr = NULL;
    CupvaMemGetHostPointer((void **)&ddmPtr, (void *)ddmData.buffer.strided.basePtr);
    *ddmPtr = 0.0f; // No DDM offset for this test

    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(NVCVTensorHandle inDetectionCountTensorHandle, NVCVTensorHandle inSnapshotsTensorHandle,
                          NVCVTensorHandle inDetectionListTensorHandle,
                          NVCVTensorHandle inDdmDopplerOffsetsTensorHandle, NVCVTensorHandle refTargetCountTensorHandle,
                          NVCVTensorHandle refTargetAnglesTensorHandle, NVCVTensorHandle refTargetIndexMapTensorHandle,
                          NVCVTensorHandle outTargetCountTensorHandle, NVCVTensorHandle refTargetListTensorHandle,
                          NVCVTensorHandle outTargetListTensorHandle, PVADoaAngleFFTParams *params,
                          PVATargetProcessingParams *targetProcessingParams, const char *testName)
{
    int32_t err = 0;

    // Run CPU reference DOA implementation (DOA-only, produces angles and index map)
    doaAngleFFTRef(inDetectionCountTensorHandle, inSnapshotsTensorHandle, refTargetCountTensorHandle,
                   refTargetAnglesTensorHandle, refTargetIndexMapTensorHandle, params);

    // Run CPU reference target processing to produce final target list
    // Pass NULL for NCI tensor since enableQuadraticInterpolation is false in this sample
    targetProcessingRef(refTargetCountTensorHandle, inDetectionListTensorHandle, refTargetIndexMapTensorHandle,
                        refTargetAnglesTensorHandle, inDdmDopplerOffsetsTensorHandle, NULL /* nciTensor */,
                        refTargetListTensorHandle, targetProcessingParams);

    // Compare PVA output with CPU reference
    err = checkTargetProcessingOutput(outTargetCountTensorHandle, outTargetListTensorHandle, refTargetListTensorHandle);
    if (err)
    {
        printf("%s test failed\n", testName);
    }
    return err;
}

/**
 * @brief Test DoA Angle FFT operation using CUDA stream
 */
int SubmitDoaAngleFFTToCudaStream(int32_t NofRx, int32_t NofTx, int32_t NofTargets, int32_t NofTargetAngleParams,
                                  int32_t NofDetections, PVADoaAngleFFTParams *params,
                                  PVATargetProcessingParams *targetProcessingParams, bool includePower)
{
    /* Initialize all resource handles to NULL for safe cleanup on error */
    NVCVAllocatorHandle allocatorHandle              = NULL;
    NVCVTensorHandle inDetectionCountTensorHandle    = NULL;
    NVCVTensorHandle inSnapshotsTensorHandle         = NULL;
    NVCVTensorHandle inDetectionListTensorHandle     = NULL;
    NVCVTensorHandle inDdmDopplerOffsetsTensorHandle = NULL;
    NVCVTensorHandle outTargetCountTensorHandle      = NULL;
    NVCVTensorHandle outTargetAnglesTensorHandle     = NULL;
    NVCVTensorHandle outTargetIndexMapTensorHandle   = NULL;
    NVCVTensorHandle outTargetListTensorHandle       = NULL;
    NVCVTensorHandle refTargetCountTensorHandle      = NULL;
    NVCVTensorHandle refTargetAnglesTensorHandle     = NULL;
    NVCVTensorHandle refTargetIndexMapTensorHandle   = NULL;
    NVCVTensorHandle refTargetListTensorHandle       = NULL;
    NVCVOperatorHandle doaOperatorHandle             = NULL;
    NVCVOperatorHandle tpOperatorHandle              = NULL;
    cudaStream_t stream                              = NULL;
    int32_t err                                      = 0;
    cudaError_t cudaErr                              = cudaSuccess;

    // Tensor requirement structures (stack-allocated)
    NVCVTensorRequirements inDetectionCountTensorRequirements;
    NVCVTensorRequirements inSnapshotsTensorRequirements;
    NVCVTensorRequirements inDetectionListTensorRequirements;
    NVCVTensorRequirements inDdmDopplerOffsetsTensorRequirements;
    NVCVTensorRequirements outTargetCountTensorRequirements;
    NVCVTensorRequirements outTargetAnglesTensorRequirements;
    NVCVTensorRequirements outTargetIndexMapTensorRequirements;
    NVCVTensorRequirements outTargetListTensorRequirements;

    // Create non-blocking CUDA stream for async execution
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaErr, FreeResources);

    // Setup tensor structure with pointers to all handles and requirements
    DoaAngleFFTTensors tensors = {.allocatorHandle                       = &allocatorHandle,
                                  .inDetectionCountTensorHandle          = &inDetectionCountTensorHandle,
                                  .inSnapshotsTensorHandle               = &inSnapshotsTensorHandle,
                                  .inDetectionListTensorHandle           = &inDetectionListTensorHandle,
                                  .inDdmDopplerOffsetsTensorHandle       = &inDdmDopplerOffsetsTensorHandle,
                                  .outTargetCountTensorHandle            = &outTargetCountTensorHandle,
                                  .outTargetAnglesTensorHandle           = &outTargetAnglesTensorHandle,
                                  .outTargetIndexMapTensorHandle         = &outTargetIndexMapTensorHandle,
                                  .outTargetListTensorHandle             = &outTargetListTensorHandle,
                                  .refTargetCountTensorHandle            = &refTargetCountTensorHandle,
                                  .refTargetAnglesTensorHandle           = &refTargetAnglesTensorHandle,
                                  .refTargetIndexMapTensorHandle         = &refTargetIndexMapTensorHandle,
                                  .refTargetListTensorHandle             = &refTargetListTensorHandle,
                                  .inDetectionCountTensorRequirements    = &inDetectionCountTensorRequirements,
                                  .inSnapshotsTensorRequirements         = &inSnapshotsTensorRequirements,
                                  .inDetectionListTensorRequirements     = &inDetectionListTensorRequirements,
                                  .inDdmDopplerOffsetsTensorRequirements = &inDdmDopplerOffsetsTensorRequirements,
                                  .outTargetCountTensorRequirements      = &outTargetCountTensorRequirements,
                                  .outTargetAnglesTensorRequirements     = &outTargetAnglesTensorRequirements,
                                  .outTargetIndexMapTensorRequirements   = &outTargetIndexMapTensorRequirements,
                                  .outTargetListTensorRequirements       = &outTargetListTensorRequirements};

    // Allocate all tensors and populate with test data
    NVCV_CHECK_ERROR_GOTO(AllocateAndInitializeTensors(NofRx, NofTx, NofTargets, NofTargetAngleParams, NofDetections,
                                                       params, &tensors, includePower),
                          err, FreeResources);

    // Create DOA Angle FFT operator instance (outputs targetAngles and targetIndexMap)
    NVCV_CHECK_ERROR_GOTO(
        pvaDoaAngleFFTCreate(&doaOperatorHandle, &inSnapshotsTensorRequirements, &inDetectionListTensorRequirements,
                             &inDdmDopplerOffsetsTensorRequirements, &outTargetAnglesTensorRequirements,
                             &outTargetIndexMapTensorRequirements, params),
        err, FreeResources);

    // Create standalone Target Processing operator (converts DOA outputs to final targetList)
    // Note: NCI tensor is NULL - quadratic interpolation disabled in this sample
    NVCV_CHECK_ERROR_GOTO(
        pvaTargetProcessingCreate(&tpOperatorHandle, &inDetectionListTensorRequirements,
                                  &outTargetIndexMapTensorRequirements, &outTargetAnglesTensorRequirements,
                                  &inDdmDopplerOffsetsTensorRequirements, &outTargetListTensorRequirements,
                                  NULL /* nciTensorRequirements */, targetProcessingParams),
        err, FreeResources);

    // Submit DOA operator to CUDA stream (outputs targetAngles and targetIndexMap)
    NVCV_CHECK_ERROR_GOTO(
        pvaDoaAngleFFTSubmit(doaOperatorHandle, stream, inDetectionCountTensorHandle, inSnapshotsTensorHandle,
                             inDetectionListTensorHandle, inDdmDopplerOffsetsTensorHandle, outTargetCountTensorHandle,
                             outTargetAnglesTensorHandle, outTargetIndexMapTensorHandle),
        err, FreeResources);

    // Submit Target Processing operator to CUDA stream (converts DOA outputs to targetList)
    NVCV_CHECK_ERROR_GOTO(pvaTargetProcessingSubmit(
                              tpOperatorHandle, stream, outTargetCountTensorHandle, inDetectionCountTensorHandle,
                              inDetectionListTensorHandle, outTargetIndexMapTensorHandle, outTargetAnglesTensorHandle,
                              inDdmDopplerOffsetsTensorHandle, NULL /* nciTensor */, outTargetListTensorHandle),
                          err, FreeResources);

    // Wait for all operations on the stream to complete
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), cudaErr, FreeResources);

    // Run CPU reference and compare results
    err = ValidateOutput(inDetectionCountTensorHandle, inSnapshotsTensorHandle, inDetectionListTensorHandle,
                         inDdmDopplerOffsetsTensorHandle, refTargetCountTensorHandle, refTargetAnglesTensorHandle,
                         refTargetIndexMapTensorHandle, outTargetCountTensorHandle, refTargetListTensorHandle,
                         outTargetListTensorHandle, params, targetProcessingParams, "Submit to CUDA stream");

FreeResources: // Clean up all allocated resources (safe to call with NULL handles)
    cudaStreamDestroy(stream);
    nvcvTensorDecRef(inDetectionCountTensorHandle, NULL);
    nvcvTensorDecRef(inSnapshotsTensorHandle, NULL);
    nvcvTensorDecRef(inDetectionListTensorHandle, NULL);
    nvcvTensorDecRef(inDdmDopplerOffsetsTensorHandle, NULL);
    nvcvTensorDecRef(outTargetCountTensorHandle, NULL);
    nvcvTensorDecRef(outTargetAnglesTensorHandle, NULL);
    nvcvTensorDecRef(outTargetIndexMapTensorHandle, NULL);
    nvcvTensorDecRef(outTargetListTensorHandle, NULL);
    nvcvTensorDecRef(refTargetCountTensorHandle, NULL);
    nvcvTensorDecRef(refTargetAnglesTensorHandle, NULL);
    nvcvTensorDecRef(refTargetIndexMapTensorHandle, NULL);
    nvcvTensorDecRef(refTargetListTensorHandle, NULL);
    nvcvOperatorDestroy(doaOperatorHandle);
    nvcvOperatorDestroy(tpOperatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

/**
 * @brief Test DoA Angle FFT operation using cuPVA stream
 */
int SubmitDoaAngleFFTToCupvaStream(int32_t NofRx, int32_t NofTx, int32_t NofTargets, int32_t NofTargetAngleParams,
                                   int32_t NofDetections, PVADoaAngleFFTParams *params,
                                   PVATargetProcessingParams *targetProcessingParams, bool includePower)
{
    /* Initialize all resource handles to NULL for safe cleanup on error */
    NVCVAllocatorHandle allocatorHandle              = NULL;
    NVCVTensorHandle inDetectionCountTensorHandle    = NULL;
    NVCVTensorHandle inSnapshotsTensorHandle         = NULL;
    NVCVTensorHandle inDetectionListTensorHandle     = NULL;
    NVCVTensorHandle inDdmDopplerOffsetsTensorHandle = NULL;
    NVCVTensorHandle outTargetAnglesTensorHandle     = NULL;
    NVCVTensorHandle outTargetIndexMapTensorHandle   = NULL;
    NVCVTensorHandle outTargetCountTensorHandle      = NULL;
    NVCVTensorHandle outTargetListTensorHandle       = NULL;
    NVCVTensorHandle refTargetCountTensorHandle      = NULL;
    NVCVTensorHandle refTargetAnglesTensorHandle     = NULL;
    NVCVTensorHandle refTargetIndexMapTensorHandle   = NULL;
    NVCVTensorHandle refTargetListTensorHandle       = NULL;
    NVCVOperatorHandle doaOperatorHandle             = NULL;
    NVCVOperatorHandle tpOperatorHandle              = NULL;
    cupvaSyncObj_t sync                              = NULL;
    cupvaStream_t stream                             = NULL;
    int32_t err                                      = 0;

    // Tensor requirement structures (stack-allocated)
    NVCVTensorRequirements inDetectionCountTensorRequirements;
    NVCVTensorRequirements inSnapshotsTensorRequirements;
    NVCVTensorRequirements inDetectionListTensorRequirements;
    NVCVTensorRequirements inDdmDopplerOffsetsTensorRequirements;
    NVCVTensorRequirements outTargetCountTensorRequirements;
    NVCVTensorRequirements outTargetAnglesTensorRequirements;
    NVCVTensorRequirements outTargetIndexMapTensorRequirements;
    NVCVTensorRequirements outTargetListTensorRequirements;

    // Setup tensor structure with pointers to all handles and requirements
    DoaAngleFFTTensors tensors = {.allocatorHandle                       = &allocatorHandle,
                                  .inDetectionCountTensorHandle          = &inDetectionCountTensorHandle,
                                  .inSnapshotsTensorHandle               = &inSnapshotsTensorHandle,
                                  .inDetectionListTensorHandle           = &inDetectionListTensorHandle,
                                  .inDdmDopplerOffsetsTensorHandle       = &inDdmDopplerOffsetsTensorHandle,
                                  .outTargetCountTensorHandle            = &outTargetCountTensorHandle,
                                  .outTargetAnglesTensorHandle           = &outTargetAnglesTensorHandle,
                                  .outTargetIndexMapTensorHandle         = &outTargetIndexMapTensorHandle,
                                  .outTargetListTensorHandle             = &outTargetListTensorHandle,
                                  .refTargetCountTensorHandle            = &refTargetCountTensorHandle,
                                  .refTargetAnglesTensorHandle           = &refTargetAnglesTensorHandle,
                                  .refTargetIndexMapTensorHandle         = &refTargetIndexMapTensorHandle,
                                  .refTargetListTensorHandle             = &refTargetListTensorHandle,
                                  .inDetectionCountTensorRequirements    = &inDetectionCountTensorRequirements,
                                  .inSnapshotsTensorRequirements         = &inSnapshotsTensorRequirements,
                                  .inDetectionListTensorRequirements     = &inDetectionListTensorRequirements,
                                  .inDdmDopplerOffsetsTensorRequirements = &inDdmDopplerOffsetsTensorRequirements,
                                  .outTargetCountTensorRequirements      = &outTargetCountTensorRequirements,
                                  .outTargetAnglesTensorRequirements     = &outTargetAnglesTensorRequirements,
                                  .outTargetIndexMapTensorRequirements   = &outTargetIndexMapTensorRequirements,
                                  .outTargetListTensorRequirements       = &outTargetListTensorRequirements};

    // Allocate all tensors and populate with test data
    NVCV_CHECK_ERROR_GOTO(AllocateAndInitializeTensors(NofRx, NofTx, NofTargets, NofTargetAngleParams, NofDetections,
                                                       params, &tensors, includePower),
                          err, FreeResources);

    // Create DOA Angle FFT operator instance (outputs targetAngles and targetIndexMap)
    NVCV_CHECK_ERROR_GOTO(
        pvaDoaAngleFFTCreate(&doaOperatorHandle, &inSnapshotsTensorRequirements, &inDetectionListTensorRequirements,
                             &inDdmDopplerOffsetsTensorRequirements, &outTargetAnglesTensorRequirements,
                             &outTargetIndexMapTensorRequirements, params),
        err, FreeResources);

    // Create standalone Target Processing operator (converts DOA outputs to final targetList)
    // Note: NCI tensor is NULL - quadratic interpolation disabled in this sample
    NVCV_CHECK_ERROR_GOTO(
        pvaTargetProcessingCreate(&tpOperatorHandle, &inDetectionListTensorRequirements,
                                  &outTargetIndexMapTensorRequirements, &outTargetAnglesTensorRequirements,
                                  &inDdmDopplerOffsetsTensorRequirements, &outTargetListTensorRequirements,
                                  NULL /* nciTensorRequirements */, targetProcessingParams),
        err, FreeResources);

    // Create synchronization object for host-PVA coordination (signaler and waiter mode)
    CUPVA_CHECK_ERROR_GOTO(CupvaSyncObjCreate(&sync, false, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD), err,
                           FreeResources);

    // Initialize fence for task completion notification
    cupvaFence_t fence;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceInit(&fence, sync), err, FreeResources);

    // Create request-fences command to signal fence after task completion
    cupvaCmd_t rf;
    CUPVA_CHECK_ERROR_GOTO(CupvaCmdRequestFencesInit(&rf, &fence, 1), err, FreeResources);

    // Create cuPVA stream on PVA0 hardware using any available VPU
    CUPVA_CHECK_ERROR_GOTO(CupvaStreamCreate(&stream, CUPVA_PVA0, CUPVA_VPU_ANY), err, FreeResources);

    // Prepare command array and status array for stream submission
    cupvaCmd_t const *cmds[1];
    cmds[0] = &rf;
    cupvaCmdStatus_t status[1];
    status[0] = NULL;

    // Submit DOA operator to cuPVA stream (outputs targetAngles and targetIndexMap)
    NVCV_CHECK_ERROR_GOTO(
        pvaDoaAngleFFTSubmit(doaOperatorHandle, stream, inDetectionCountTensorHandle, inSnapshotsTensorHandle,
                             inDetectionListTensorHandle, inDdmDopplerOffsetsTensorHandle, outTargetCountTensorHandle,
                             outTargetAnglesTensorHandle, outTargetIndexMapTensorHandle),
        err, FreeResources);

    // Submit Target Processing operator to cuPVA stream (converts DOA outputs to targetList)
    NVCV_CHECK_ERROR_GOTO(pvaTargetProcessingSubmit(
                              tpOperatorHandle, stream, outTargetCountTensorHandle, inDetectionCountTensorHandle,
                              inDetectionListTensorHandle, outTargetIndexMapTensorHandle, outTargetAnglesTensorHandle,
                              inDdmDopplerOffsetsTensorHandle, NULL /* nciTensor */, outTargetListTensorHandle),
                          err, FreeResources);

    // Submit fence command to stream (in-order execution, no timeout)
    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    // Wait for fence to signal (blocks until PVA task completes)
    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    // Run CPU reference and compare results
    err = ValidateOutput(inDetectionCountTensorHandle, inSnapshotsTensorHandle, inDetectionListTensorHandle,
                         inDdmDopplerOffsetsTensorHandle, refTargetCountTensorHandle, refTargetAnglesTensorHandle,
                         refTargetIndexMapTensorHandle, outTargetCountTensorHandle, refTargetListTensorHandle,
                         outTargetListTensorHandle, params, targetProcessingParams, "Submit to cuPVA stream");

FreeResources: // Clean up all allocated resources (safe to call with NULL handles)
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvTensorDecRef(inDetectionCountTensorHandle, NULL);
    nvcvTensorDecRef(inSnapshotsTensorHandle, NULL);
    nvcvTensorDecRef(inDetectionListTensorHandle, NULL);
    nvcvTensorDecRef(inDdmDopplerOffsetsTensorHandle, NULL);
    nvcvTensorDecRef(outTargetCountTensorHandle, NULL);
    nvcvTensorDecRef(outTargetAnglesTensorHandle, NULL);
    nvcvTensorDecRef(outTargetIndexMapTensorHandle, NULL);
    nvcvTensorDecRef(outTargetListTensorHandle, NULL);
    nvcvTensorDecRef(refTargetCountTensorHandle, NULL);
    nvcvTensorDecRef(refTargetAnglesTensorHandle, NULL);
    nvcvTensorDecRef(refTargetIndexMapTensorHandle, NULL);
    nvcvTensorDecRef(refTargetListTensorHandle, NULL);
    nvcvOperatorDestroy(doaOperatorHandle);
    nvcvOperatorDestroy(tpOperatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char **argv)
{

    // Default test parameters (antenna array and target configuration)
    int32_t NofRx                = 8;          // Number of receive antennas (azimuth dimension)
    int32_t NofTx                = 8;          // Number of transmit antennas (elevation dimension)
    int32_t NofTargets           = 1;          // Maximum number of targets to detect
    int32_t NofTargetAngleParams = 3;          // Output parameters per target: [azimuth, elevation, power]
    int32_t NofDetections        = NofTargets; // Number of detections to process
    bool includePower            = true;       // Include power in target list

    // Configure DOA Angle FFT operator parameters
    PVADoaAngleFFTParams params;
    params.numVirtualAzimuthElements   = 16; // Virtual array size in azimuth (after MIMO)
    params.numVirtualElevationElements = 4;  // Virtual array size in elevation (after MIMO)
    params.angleEstimationPrecision    = PVA_ANGLE_ESTIMATION_PRECISION_DEFAULT; // Scaling factor 4 (64x16 FFT)
    params.enableLocalPeakDetection    = false; // Use global peak detection (simpler, 1 target per snapshot)
    // Configure Target Processing parameters (separate from DOA params)
    PVATargetProcessingParams targetProcessingParams;
    targetProcessingParams.wavelength                   = 0.004f;  // 4mm, 77GHz
    targetProcessingParams.sweepTime                    = 100e-6f; // 100 microseconds
    targetProcessingParams.rangeResolution              = 0.1f;    // 0.1 meters
    targetProcessingParams.numChirps                    = 256;
    targetProcessingParams.fftShiftedDoppler            = false;
    targetProcessingParams.enableRvDecoupling           = false; // Enable RV decoupling
    targetProcessingParams.enableQuadraticInterpolation = false; // Requires NCI input
    targetProcessingParams.groundProjection             = 0;     // 0 = SLANT_RANGE, 1 = GROUND_PLANE
    targetProcessingParams.longitudinalAxis             = 0;     // 0 = X_FORWARD, 1 = Y_FORWARD

    // Parse command-line arguments if provided (all four must be specified)
    if (argc >= 5)
    {
        char *endptr;
        // Parse and validate NofTargets (max targets)
        int32_t val = strtol(argv[1], &endptr, 10);
        if (*endptr != '\0' || val > INT_MAX || val < INT_MIN || val <= 0)
        {
            printf("Error: Invalid NofTargets value: %s\n", argv[1]);
            return -1;
        }
        NofTargets = (int32_t)val;

        // Parse and validate NofDetections (detections to process, can be 0)
        val = strtol(argv[2], &endptr, 10);
        if (*endptr != '\0' || val > INT_MAX || val < INT_MIN || val < 0)
        {
            printf("Error: Invalid NofDetections value: %s\n", argv[2]);
            return -1;
        }
        NofDetections = (int32_t)val;

        // Parse and validate angleEstimationPrecision
        val = strtol(argv[3], &endptr, 10);
        if (*endptr != '\0' || val > PVA_ANGLE_ESTIMATION_PRECISION_HIGH || val < PVA_ANGLE_ESTIMATION_PRECISION_LOW)
        {
            printf("Error: Invalid angleEstimationPrecision value: %s\n", argv[3]);
            return -1;
        }
        params.angleEstimationPrecision = (int32_t)val;

        // Parse and validate includePower
        val = strtol(argv[4], &endptr, 10);
        if (*endptr != '\0' || val > 1 || val < 0)
        {
            printf("Error: Invalid includePower value: %s\n", argv[4]);
        }
        includePower = (bool)val;

        printf(
            "Parameters set from command line: NofTargets=%d, NofDetections=%d, angleEstimationPrecision=%d, "
            "includePower=%d\n",
            NofTargets, NofDetections, params.angleEstimationPrecision, includePower);

        // Validate detection count doesn't exceed target buffer size
        if (NofDetections > NofTargets)
        {
            printf("Error: NofDetections (%d) cannot be greater than NofTargets (%d)\n", NofDetections, NofTargets);
            return -1;
        }
    }
    else if (argc > 1)
    {
        // Require all parameters if any are provided
        printf("Usage: %s <NofTargets> <NofDetections> <angleEstimationPrecision> <includePower>\n", argv[0]);
        printf("All four parameters are required when specifying command line arguments.\n");
        return -1;
    }

    // Initialize test result variables
    int32_t cudaStreamSubmitTestResult  = 0;
    int32_t cupvaStreamSubmitTestResult = 0;
    cudaError_t cudaErr                 = cudaSuccess;

#if CUPVA_HAS_CUDA_STREAM_SUPPORT == 1

    // Initialize CUDA device (required for CUDA stream submission)
    CUDA_CHECK_ERROR_GOTO(cudaSetDevice(0), cudaErr, CheckResults);

    // Check cuPVA driver version (CUDA stream support requires v2007+)
    uint32_t driverVersion;
    CUPVA_CHECK_ERROR_GOTO(CupvaGetDriverVersion(&driverVersion), cudaStreamSubmitTestResult, CheckResults);
    if (driverVersion >= 2007)
    {
        // Run test using CUDA stream (allows interop with CUDA operations)
        cudaStreamSubmitTestResult =
            SubmitDoaAngleFFTToCudaStream(NofRx, NofTx, NofTargets, NofTargetAngleParams, NofDetections, &params,
                                          &targetProcessingParams, includePower);
    }
#endif

    // Run test using native cuPVA stream (direct PVA control, lower latency)
    cupvaStreamSubmitTestResult = SubmitDoaAngleFFTToCupvaStream(
        NofRx, NofTx, NofTargets, NofTargetAngleParams, NofDetections, &params, &targetProcessingParams, includePower);

CheckResults:
    // Print overall test result (pass if both tests passed)
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

#ifdef __cplusplus
}
#endif