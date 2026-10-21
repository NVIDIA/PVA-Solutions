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
 * @file snapshot_extraction.c
 * @brief Sample application and test harness for Snapshot Extraction operator
 *
 * This file demonstrates how to use the pvaSnapshotExtraction operator for
 * DDM (Doppler Division Multiplexing) disambiguation and snapshot extraction
 * from Range-Doppler maps for radar target detection.
 *
 * The sample provides:
 * - Tensor allocation and initialization
 * - Reference CPU implementation for validation
 * - Operator submission to both CUDA and cuPVA streams
 * - Result comparison and accuracy checking
 *
 * Key Functions:
 * - AllocateAndInitializeTensors(): Sets up input/output tensors
 * - SubmitSnapshotExtractionToCudaStream(): Submits operator to CUDA stream
 * - SubmitSnapshotExtractionToCupvaStream(): Submits operator to cuPVA stream
 * - VerifySnapshotExtractionResults(): Validates PVA results against CPU reference
 * - RunTestSnapshotExtraction(): Main test entry point
 */

#include "snapshot_extraction_ref.h"

#include <ErrorCheckMacros.h>
#include <ImageIO.h>
#include <OpSnapshotExtraction.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <limits.h>
#include <nvcv/Fwd.h>
#include <nvcv/alloc/Allocator.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Structure to hold tensor handles and requirements
 */
typedef struct
{
    NVCVAllocatorHandle *allocatorHandle;
    NVCVTensorHandle *detectionCountTensorHandle;
    NVCVTensorHandle *foldedDetectionListTensorHandle;
    NVCVTensorHandle *ddmDopplerOffsetsTensorHandle;
    NVCVTensorHandle *nciRxTensorHandle;
    NVCVTensorHandle *rangeDopplerMapTensorHandle;
    NVCVTensorHandle *calibrationWeightsTensorHandle;
    NVCVTensorHandle *detectionListTensorHandle;
    NVCVTensorHandle *snapshotsTensorHandle;
    NVCVTensorHandle *refDetectionListTensorHandle;
    NVCVTensorHandle *refSnapshotsTensorHandle;
    NVCVTensorRequirements *detectionCountTensorRequirements;
    NVCVTensorRequirements *foldedDetectionListTensorRequirements;
    NVCVTensorRequirements *ddmDopplerOffsetsTensorRequirements;
    NVCVTensorRequirements *nciRxTensorRequirements;
    NVCVTensorRequirements *rangeDopplerMapTensorRequirements;
    NVCVTensorRequirements *calibrationWeightsTensorRequirements;
    NVCVTensorRequirements *detectionListTensorRequirements;
    NVCVTensorRequirements *snapshotsTensorRequirements;
} SnapshotExtractionTensors;

/**
 * @brief Allocate and initialize tensors
 */
static int AllocateAndInitializeTensors(int32_t numDetections, int32_t numDopplerFolds, int32_t numRangeBins,
                                        int32_t numDopplerBins, int32_t numRx, int32_t numTx, int32_t useCalibration,
                                        PVASnapshotExtractionParams *params, SnapshotExtractionTensors *tensors)
{
    int32_t err = 0;

    // Define data types for all tensors
    NVCVDataType detectionCountDataType      = NVCV_DATA_TYPE_S32;  // Single int32 scalar
    NVCVDataType foldedDetectionListDataType = NVCV_DATA_TYPE_S32;  // Int32 pairs (range, folded doppler)
    NVCVDataType ddmDopplerOffsetsDataType   = NVCV_DATA_TYPE_F32;  // Float32 offsets
    NVCVDataType nciRxDataType               = NVCV_DATA_TYPE_U32;  // Uint32 magnitudes
    NVCVDataType rangeDopplerMapDataType     = NVCV_DATA_TYPE_2S32; // Complex int32
    NVCVDataType detectionListDataType       = NVCV_DATA_TYPE_S32;  // Int32 pairs (range, doppler)
    NVCVDataType snapshotsDataType           = NVCV_DATA_TYPE_2S32; // Complex int32

    // Define tensor ranks (number of dimensions)
    int32_t detectionCountTensorRank = 1; // Scalar (1D)
    int32_t foldedDetectionListRank  = 2; // 2D: [numDetections, 2]
    int32_t ddmDopplerOffsetsRank    = 1; // 1D: [numDopplerFolds]
    int32_t nciRxRank                = 2; // 2D: [numRangeBins, numDopplerBins]
    int32_t rangeDopplerMapRank      = 3; // 3D: Format-dependent
    int32_t detectionListRank        = 2; // 2D: [numDetections, 2]
    int32_t snapshotsRank            = 3; // 3D: [numDetections, numTx, numRx]

    // Define tensor shapes
    int64_t detectionCountTensorShape[] = {1};                            // Single scalar
    int64_t foldedDetectionListShape[]  = {numDetections, 2};             // [N, 2]
    int64_t ddmDopplerOffsetsShape[]    = {numDopplerFolds};              // [M]
    int64_t nciRxShape[]                = {numRangeBins, numDopplerBins}; // [H, W]
    int64_t detectionListShape[]        = {numDetections, 2};             // [N, 2]
    int64_t snapshotsShape[]            = {numDetections, numTx, numRx};  // [N, H, W]

    // Range-Doppler Map shape depends on format
    int64_t rangeDopplerMapShape[3];
    if (params->rangeDopplerFormat == PVA_FFT_LAYOUT_RANGE_RX_DOPPLER)
    {
        // RANGE_RX_DOPPLER format: [numRangeBins, numRx, numDopplerBins] - HCW layout
        rangeDopplerMapShape[0] = numRangeBins;
        rangeDopplerMapShape[1] = numRx;
        rangeDopplerMapShape[2] = numDopplerBins;
    }
    else // PVA_FFT_LAYOUT_RANGE_DOPPLER_RX
    {
        // RANGE_DOPPLER_RX format: [numRangeBins, numDopplerBins, numRx] - HCW layout
        rangeDopplerMapShape[0] = numRangeBins;
        rangeDopplerMapShape[1] = numDopplerBins;
        rangeDopplerMapShape[2] = numRx;
    }

    // Construct PVA-specific allocator
    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(tensors->allocatorHandle), err, Error);

    // Create and allocate detection count tensor (scalar)
    NVCVTensorLayout detectionCountTensorLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("W", &detectionCountTensorLayout), err, Error);

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(detectionCountTensorRank, detectionCountTensorShape, detectionCountDataType,
                                      detectionCountTensorLayout, 0, 0, tensors->detectionCountTensorRequirements),
        err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensors->detectionCountTensorRequirements, *tensors->allocatorHandle,
                                              tensors->detectionCountTensorHandle),
                          err, Error);

    // Create and allocate folded detection list tensor [N, 2]
    NVCVTensorLayout foldedDetectionListLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HW", &foldedDetectionListLayout), err, Error);

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(foldedDetectionListRank, foldedDetectionListShape, foldedDetectionListDataType,
                                      foldedDetectionListLayout, 0, 0, tensors->foldedDetectionListTensorRequirements),
        err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensors->foldedDetectionListTensorRequirements, *tensors->allocatorHandle,
                                              tensors->foldedDetectionListTensorHandle),
                          err, Error);

    // Create and allocate DDM Doppler offsets tensor [M]
    NVCVTensorLayout ddmDopplerOffsetsLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("W", &ddmDopplerOffsetsLayout), err, Error);

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(ddmDopplerOffsetsRank, ddmDopplerOffsetsShape, ddmDopplerOffsetsDataType,
                                      ddmDopplerOffsetsLayout, 0, 0, tensors->ddmDopplerOffsetsTensorRequirements),
        err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensors->ddmDopplerOffsetsTensorRequirements, *tensors->allocatorHandle,
                                              tensors->ddmDopplerOffsetsTensorHandle),
                          err, Error);

    // Create and allocate NCI Rx tensor [H, W]
    NVCVTensorLayout nciRxLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HW", &nciRxLayout), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(nciRxRank, nciRxShape, nciRxDataType, nciRxLayout, 0, 0,
                                                        tensors->nciRxTensorRequirements),
                          err, Error);

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorConstruct(tensors->nciRxTensorRequirements, *tensors->allocatorHandle, tensors->nciRxTensorHandle),
        err, Error);

    // Create and allocate Range-Doppler map tensor [H, C, W]
    NVCVTensorLayout rangeDopplerMapLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HCW", &rangeDopplerMapLayout), err, Error);

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(rangeDopplerMapRank, rangeDopplerMapShape, rangeDopplerMapDataType,
                                      rangeDopplerMapLayout, 0, 0, tensors->rangeDopplerMapTensorRequirements),
        err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensors->rangeDopplerMapTensorRequirements, *tensors->allocatorHandle,
                                              tensors->rangeDopplerMapTensorHandle),
                          err, Error);

    // Create and allocate calibration weights tensor [numTx, numRx] (only if calibration enabled)
    if (useCalibration)
    {
        NVCVTensorLayout calibrationWeightsLayout;
        NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HW", &calibrationWeightsLayout), err, Error);

        NVCVDataType calibrationWeightsDataType = NVCV_DATA_TYPE_2S32; // Complex int32
        int32_t calibrationWeightsRank          = 2;
        int64_t calibrationWeightsShape[]       = {numTx, numRx};

        NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(calibrationWeightsRank, calibrationWeightsShape,
                                                            calibrationWeightsDataType, calibrationWeightsLayout, 0, 0,
                                                            tensors->calibrationWeightsTensorRequirements),
                              err, Error);

        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensors->calibrationWeightsTensorRequirements,
                                                  *tensors->allocatorHandle, tensors->calibrationWeightsTensorHandle),
                              err, Error);

        // Initialize calibration weights using helper function
        NVCVTensorData calibrationWeightsData;
        NVCV_CHECK_ERROR_GOTO(nvcvTensorExportData(*tensors->calibrationWeightsTensorHandle, &calibrationWeightsData),
                              err, Error);

        // Calibration weights will be generated by generateSnapshotExtractionInput()
        printf("Calibration weights tensor allocated for [%d x %d] antenna array\n", numTx, numRx);
    }
    else
    {
        // NULL calibration tensor when not enabled
        *tensors->calibrationWeightsTensorHandle = NULL;
    }

    // Create and allocate detection list tensor [N, 2]
    NVCVTensorLayout detectionListLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HW", &detectionListLayout), err, Error);

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(detectionListRank, detectionListShape, detectionListDataType, detectionListLayout,
                                      0, 0, tensors->detectionListTensorRequirements),
        err, Error);

    // Allocate both PVA output and reference (CPU) tensors for detection list
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensors->detectionListTensorRequirements, *tensors->allocatorHandle,
                                              tensors->detectionListTensorHandle),
                          err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensors->detectionListTensorRequirements, *tensors->allocatorHandle,
                                              tensors->refDetectionListTensorHandle),
                          err, Error);

    // Create and allocate snapshots tensor [N, Tx, Rx]
    NVCVTensorLayout snapshotsLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("CHW", &snapshotsLayout), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(snapshotsRank, snapshotsShape, snapshotsDataType,
                                                        snapshotsLayout, 0, 0, tensors->snapshotsTensorRequirements),
                          err, Error);

    // Allocate both PVA output and reference (CPU) tensors for snapshots
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensors->snapshotsTensorRequirements, *tensors->allocatorHandle,
                                              tensors->snapshotsTensorHandle),
                          err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(tensors->snapshotsTensorRequirements, *tensors->allocatorHandle,
                                              tensors->refSnapshotsTensorHandle),
                          err, Error);

    // Generate synthetic input data for testing (including detection count and DDM doppler offsets)
    generateSnapshotExtractionInput(*tensors->detectionCountTensorHandle, *tensors->foldedDetectionListTensorHandle,
                                    *tensors->ddmDopplerOffsetsTensorHandle, *tensors->rangeDopplerMapTensorHandle,
                                    *tensors->nciRxTensorHandle,
                                    useCalibration ? *tensors->calibrationWeightsTensorHandle : NULL, numDetections,
                                    numDopplerFolds, numRangeBins, numDopplerBins);

    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(NVCVTensorHandle detectionCountTensorHandle, NVCVTensorHandle foldedDetectionListTensorHandle,
                          NVCVTensorHandle ddmDopplerOffsetsTensorHandle, NVCVTensorHandle nciRxTensorHandle,
                          NVCVTensorHandle rangeDopplerMapTensorHandle, NVCVTensorHandle calibrationWeightsTensorHandle,
                          NVCVTensorHandle refDetectionListTensorHandle, NVCVTensorHandle detectionListTensorHandle,
                          NVCVTensorHandle refSnapshotsTensorHandle, NVCVTensorHandle snapshotsTensorHandle,
                          PVASnapshotExtractionParams *params, const char *testName)
{
    int32_t err = 0;

    // Run CPU reference implementation to generate expected results
    snapshotExtractionRef(detectionCountTensorHandle, foldedDetectionListTensorHandle, ddmDopplerOffsetsTensorHandle,
                          nciRxTensorHandle, rangeDopplerMapTensorHandle, calibrationWeightsTensorHandle,
                          refDetectionListTensorHandle, refSnapshotsTensorHandle, params);

    // Compare PVA output with CPU reference
    // Note: Sample test uses same detection count for both PVA and reference
    err =
        checkSnapshotExtractionOutput(detectionCountTensorHandle, detectionCountTensorHandle, detectionListTensorHandle,
                                      refDetectionListTensorHandle, snapshotsTensorHandle, refSnapshotsTensorHandle);
    if (err)
    {
        printf("%s test failed\n", testName);
    }
    return err;
}

/**
 * @brief Test Snapshot Extraction operation using CUDA stream
 */
int SubmitSnapshotExtractionToCudaStream(int32_t numDetections, int32_t numDopplerFolds, int32_t numRangeBins,
                                         int32_t numDopplerBins, int32_t numRx, int32_t numTx, int32_t useCalibration,
                                         PVASnapshotExtractionParams *params)
{
    /* Initialize all resource handles to NULL for safe cleanup on error */
    NVCVAllocatorHandle allocatorHandle              = NULL;
    NVCVTensorHandle detectionCountTensorHandle      = NULL;
    NVCVTensorHandle foldedDetectionListTensorHandle = NULL;
    NVCVTensorHandle ddmDopplerOffsetsTensorHandle   = NULL;
    NVCVTensorHandle nciRxTensorHandle               = NULL;
    NVCVTensorHandle rangeDopplerMapTensorHandle     = NULL;
    NVCVTensorHandle calibrationWeightsTensorHandle  = NULL;
    NVCVTensorHandle detectionListTensorHandle       = NULL;
    NVCVTensorHandle snapshotsTensorHandle           = NULL;
    NVCVTensorHandle refDetectionListTensorHandle    = NULL;
    NVCVTensorHandle refSnapshotsTensorHandle        = NULL;
    NVCVOperatorHandle operatorHandle                = NULL;
    cudaStream_t stream                              = NULL;
    int32_t err                                      = 0;
    cudaError_t cudaErr                              = cudaSuccess;

    // Tensor requirement structures (stack-allocated)
    NVCVTensorRequirements detectionCountTensorRequirements;
    NVCVTensorRequirements foldedDetectionListTensorRequirements;
    NVCVTensorRequirements ddmDopplerOffsetsTensorRequirements;
    NVCVTensorRequirements nciRxTensorRequirements;
    NVCVTensorRequirements rangeDopplerMapTensorRequirements;
    NVCVTensorRequirements calibrationWeightsTensorRequirements;
    NVCVTensorRequirements detectionListTensorRequirements;
    NVCVTensorRequirements snapshotsTensorRequirements;

    // Create non-blocking CUDA stream for async execution
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaErr, FreeResources);

    // Setup tensor structure with pointers to all handles and requirements
    SnapshotExtractionTensors tensors = {
        .allocatorHandle                       = &allocatorHandle,
        .detectionCountTensorHandle            = &detectionCountTensorHandle,
        .foldedDetectionListTensorHandle       = &foldedDetectionListTensorHandle,
        .ddmDopplerOffsetsTensorHandle         = &ddmDopplerOffsetsTensorHandle,
        .nciRxTensorHandle                     = &nciRxTensorHandle,
        .rangeDopplerMapTensorHandle           = &rangeDopplerMapTensorHandle,
        .calibrationWeightsTensorHandle        = &calibrationWeightsTensorHandle,
        .detectionListTensorHandle             = &detectionListTensorHandle,
        .snapshotsTensorHandle                 = &snapshotsTensorHandle,
        .refDetectionListTensorHandle          = &refDetectionListTensorHandle,
        .refSnapshotsTensorHandle              = &refSnapshotsTensorHandle,
        .detectionCountTensorRequirements      = &detectionCountTensorRequirements,
        .foldedDetectionListTensorRequirements = &foldedDetectionListTensorRequirements,
        .ddmDopplerOffsetsTensorRequirements   = &ddmDopplerOffsetsTensorRequirements,
        .nciRxTensorRequirements               = &nciRxTensorRequirements,
        .rangeDopplerMapTensorRequirements     = &rangeDopplerMapTensorRequirements,
        .calibrationWeightsTensorRequirements  = &calibrationWeightsTensorRequirements,
        .detectionListTensorRequirements       = &detectionListTensorRequirements,
        .snapshotsTensorRequirements           = &snapshotsTensorRequirements};

    // Allocate all tensors and populate with test data
    NVCV_CHECK_ERROR_GOTO(AllocateAndInitializeTensors(numDetections, numDopplerFolds, numRangeBins, numDopplerBins,
                                                       numRx, numTx, useCalibration, params, &tensors),
                          err, FreeResources);

    // Create Snapshot Extraction operator instance with tensor requirements and params
    NVCV_CHECK_ERROR_GOTO(
        pvaSnapshotExtractionCreate(
            &operatorHandle, &foldedDetectionListTensorRequirements, &ddmDopplerOffsetsTensorRequirements,
            &nciRxTensorRequirements, &rangeDopplerMapTensorRequirements, &detectionListTensorRequirements,
            &snapshotsTensorRequirements, useCalibration ? tensors.calibrationWeightsTensorRequirements : NULL, params),
        err, FreeResources);

    // Submit operator to CUDA stream (cuPVA runtime handles PVA scheduling)
    NVCV_CHECK_ERROR_GOTO(
        pvaSnapshotExtractionSubmit(operatorHandle, stream, detectionCountTensorHandle, foldedDetectionListTensorHandle,
                                    ddmDopplerOffsetsTensorHandle, nciRxTensorHandle, rangeDopplerMapTensorHandle,
                                    useCalibration ? calibrationWeightsTensorHandle : NULL, detectionListTensorHandle,
                                    snapshotsTensorHandle),
        err, FreeResources);

    // Wait for all operations on the stream to complete
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), cudaErr, FreeResources);

    // Run CPU reference and compare results
    err = ValidateOutput(detectionCountTensorHandle, foldedDetectionListTensorHandle, ddmDopplerOffsetsTensorHandle,
                         nciRxTensorHandle, rangeDopplerMapTensorHandle, calibrationWeightsTensorHandle,
                         refDetectionListTensorHandle, detectionListTensorHandle, refSnapshotsTensorHandle,
                         snapshotsTensorHandle, params, "Submit to CUDA stream");

FreeResources: // Clean up all allocated resources (safe to call with NULL handles)
    cudaStreamDestroy(stream);
    nvcvTensorDecRef(detectionCountTensorHandle, NULL);
    nvcvTensorDecRef(foldedDetectionListTensorHandle, NULL);
    nvcvTensorDecRef(ddmDopplerOffsetsTensorHandle, NULL);
    nvcvTensorDecRef(nciRxTensorHandle, NULL);
    nvcvTensorDecRef(rangeDopplerMapTensorHandle, NULL);
    nvcvTensorDecRef(calibrationWeightsTensorHandle, NULL);
    nvcvTensorDecRef(detectionListTensorHandle, NULL);
    nvcvTensorDecRef(snapshotsTensorHandle, NULL);
    nvcvTensorDecRef(refDetectionListTensorHandle, NULL);
    nvcvTensorDecRef(refSnapshotsTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

/**
 * @brief Test Snapshot Extraction operation using cuPVA stream
 */
int SubmitSnapshotExtractionToCupvaStream(int32_t numDetections, int32_t numDopplerFolds, int32_t numRangeBins,
                                          int32_t numDopplerBins, int32_t numRx, int32_t numTx, int32_t useCalibration,
                                          PVASnapshotExtractionParams *params)
{
    /* Initialize all resource handles to NULL for safe cleanup on error */
    NVCVAllocatorHandle allocatorHandle              = NULL;
    NVCVTensorHandle detectionCountTensorHandle      = NULL;
    NVCVTensorHandle foldedDetectionListTensorHandle = NULL;
    NVCVTensorHandle ddmDopplerOffsetsTensorHandle   = NULL;
    NVCVTensorHandle nciRxTensorHandle               = NULL;
    NVCVTensorHandle rangeDopplerMapTensorHandle     = NULL;
    NVCVTensorHandle calibrationWeightsTensorHandle  = NULL;
    NVCVTensorHandle detectionListTensorHandle       = NULL;
    NVCVTensorHandle snapshotsTensorHandle           = NULL;
    NVCVTensorHandle refDetectionListTensorHandle    = NULL;
    NVCVTensorHandle refSnapshotsTensorHandle        = NULL;
    NVCVOperatorHandle operatorHandle                = NULL;
    cupvaSyncObj_t sync                              = NULL;
    cupvaStream_t stream                             = NULL;
    int32_t err                                      = 0;

    // Tensor requirement structures (stack-allocated)
    NVCVTensorRequirements detectionCountTensorRequirements;
    NVCVTensorRequirements foldedDetectionListTensorRequirements;
    NVCVTensorRequirements ddmDopplerOffsetsTensorRequirements;
    NVCVTensorRequirements nciRxTensorRequirements;
    NVCVTensorRequirements rangeDopplerMapTensorRequirements;
    NVCVTensorRequirements calibrationWeightsTensorRequirements;
    NVCVTensorRequirements detectionListTensorRequirements;
    NVCVTensorRequirements snapshotsTensorRequirements;

    // Setup tensor structure with pointers to all handles and requirements
    SnapshotExtractionTensors tensors = {
        .allocatorHandle                       = &allocatorHandle,
        .detectionCountTensorHandle            = &detectionCountTensorHandle,
        .foldedDetectionListTensorHandle       = &foldedDetectionListTensorHandle,
        .ddmDopplerOffsetsTensorHandle         = &ddmDopplerOffsetsTensorHandle,
        .nciRxTensorHandle                     = &nciRxTensorHandle,
        .rangeDopplerMapTensorHandle           = &rangeDopplerMapTensorHandle,
        .calibrationWeightsTensorHandle        = &calibrationWeightsTensorHandle,
        .detectionListTensorHandle             = &detectionListTensorHandle,
        .snapshotsTensorHandle                 = &snapshotsTensorHandle,
        .refDetectionListTensorHandle          = &refDetectionListTensorHandle,
        .refSnapshotsTensorHandle              = &refSnapshotsTensorHandle,
        .detectionCountTensorRequirements      = &detectionCountTensorRequirements,
        .foldedDetectionListTensorRequirements = &foldedDetectionListTensorRequirements,
        .ddmDopplerOffsetsTensorRequirements   = &ddmDopplerOffsetsTensorRequirements,
        .nciRxTensorRequirements               = &nciRxTensorRequirements,
        .rangeDopplerMapTensorRequirements     = &rangeDopplerMapTensorRequirements,
        .calibrationWeightsTensorRequirements  = &calibrationWeightsTensorRequirements,
        .detectionListTensorRequirements       = &detectionListTensorRequirements,
        .snapshotsTensorRequirements           = &snapshotsTensorRequirements};

    // Allocate all tensors and populate with test data
    NVCV_CHECK_ERROR_GOTO(AllocateAndInitializeTensors(numDetections, numDopplerFolds, numRangeBins, numDopplerBins,
                                                       numRx, numTx, useCalibration, params, &tensors),
                          err, FreeResources);

    // Create Snapshot Extraction operator instance with tensor requirements and params
    NVCV_CHECK_ERROR_GOTO(
        pvaSnapshotExtractionCreate(
            &operatorHandle, &foldedDetectionListTensorRequirements, &ddmDopplerOffsetsTensorRequirements,
            &nciRxTensorRequirements, &rangeDopplerMapTensorRequirements, &detectionListTensorRequirements,
            &snapshotsTensorRequirements, useCalibration ? tensors.calibrationWeightsTensorRequirements : NULL, params),
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

    // Submit operator to cuPVA stream (queues task to PVA)
    NVCV_CHECK_ERROR_GOTO(
        pvaSnapshotExtractionSubmit(operatorHandle, stream, detectionCountTensorHandle, foldedDetectionListTensorHandle,
                                    ddmDopplerOffsetsTensorHandle, nciRxTensorHandle, rangeDopplerMapTensorHandle,
                                    calibrationWeightsTensorHandle, detectionListTensorHandle, snapshotsTensorHandle),
        err, FreeResources);

    // Submit fence command to stream (in-order execution, no timeout)
    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    // Wait for fence to signal (blocks until PVA task completes)
    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    // Run CPU reference and compare results
    err = ValidateOutput(detectionCountTensorHandle, foldedDetectionListTensorHandle, ddmDopplerOffsetsTensorHandle,
                         nciRxTensorHandle, rangeDopplerMapTensorHandle, calibrationWeightsTensorHandle,
                         refDetectionListTensorHandle, detectionListTensorHandle, refSnapshotsTensorHandle,
                         snapshotsTensorHandle, params, "Submit to cuPVA stream");

FreeResources: // Clean up all allocated resources (safe to call with NULL handles)
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvTensorDecRef(detectionCountTensorHandle, NULL);
    nvcvTensorDecRef(foldedDetectionListTensorHandle, NULL);
    nvcvTensorDecRef(ddmDopplerOffsetsTensorHandle, NULL);
    nvcvTensorDecRef(nciRxTensorHandle, NULL);
    nvcvTensorDecRef(rangeDopplerMapTensorHandle, NULL);
    nvcvTensorDecRef(calibrationWeightsTensorHandle, NULL);
    nvcvTensorDecRef(detectionListTensorHandle, NULL);
    nvcvTensorDecRef(snapshotsTensorHandle, NULL);
    nvcvTensorDecRef(refDetectionListTensorHandle, NULL);
    nvcvTensorDecRef(refSnapshotsTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}

int main(int argc, char **argv)
{
    // Test parameters matching radar_pipeline_advanced_test configuration
    // These values are derived from radarCfg.json in public/pipelines/assets/radar_advanced/
    int32_t numDetections                        = 10;  // Number of detections to process (typical test value)
    int32_t numDopplerFolds                      = 16;  // Number of TX antennas / DDM repeat fold factor
    int32_t numRangeBins                         = 512; // Range bins = numSamples/2 = 1024/2
    int32_t numDopplerBins                       = 512; // Doppler bins = numChirps
    int32_t numRx                                = 8;   // Number of RX antennas (2 azimuth × 4 elevation)
    int32_t numTx                                = 8;   // Number of TX antennas
    PVAFFTLayout rangeDopplerFormat              = PVA_FFT_LAYOUT_RANGE_RX_DOPPLER; // Default format
    int32_t useCalibration                       = 0; // Default: use DDMA weights (no calibration)

    // Parse command-line arguments if provided
    if (argc >= 5)
    {
        char *endptr;
        // Parse and validate numDetections
        int32_t val = strtol(argv[1], &endptr, 10);
        if (*endptr != '\0' || val > INT_MAX || val < INT_MIN || val <= 0)
        {
            printf("Error: Invalid numDetections value: %s\n", argv[1]);
            return -1;
        }
        numDetections = (int32_t)val;

        // Parse and validate numDopplerFolds
        val = strtol(argv[2], &endptr, 10);
        if (*endptr != '\0' || val > INT_MAX || val < INT_MIN || val <= 0)
        {
            printf("Error: Invalid numDopplerFolds value: %s\n", argv[2]);
            return -1;
        }
        numDopplerFolds = (int32_t)val;

        // Parse and validate numRx
        val = strtol(argv[3], &endptr, 10);
        if (*endptr != '\0' || val > INT_MAX || val < INT_MIN || val <= 0)
        {
            printf("Error: Invalid numRx value: %s\n", argv[3]);
            return -1;
        }
        numRx = (int32_t)val;

        // Parse and validate numTx
        val = strtol(argv[4], &endptr, 10);
        if (*endptr != '\0' || val > INT_MAX || val < INT_MIN || val <= 0)
        {
            printf("Error: Invalid numTx value: %s\n", argv[4]);
            return -1;
        }
        numTx = (int32_t)val;

        // Parse and validate rangeDopplerFormat (optional 5th argument)
        if (argc >= 6)
        {
            val = strtol(argv[5], &endptr, 10);
            if (*endptr != '\0' || val > INT_MAX || val < INT_MIN)
            {
                printf("Error: Invalid rangeDopplerFormat value: %s\n", argv[5]);
                return -1;
            }
            if (val != PVA_FFT_LAYOUT_RANGE_RX_DOPPLER && val != PVA_FFT_LAYOUT_RANGE_DOPPLER_RX)
            {
                printf("Error: Invalid rangeDopplerFormat value: %d (must be 1 or 2)\n", (int32_t)val);
                return -1;
            }
            rangeDopplerFormat = (PVAFFTLayout)val;
        }

        // Parse and validate useCalibration (optional 6th argument)
        if (argc >= 7)
        {
            val = strtol(argv[6], &endptr, 10);
            if (*endptr != '\0' || val > INT_MAX || val < INT_MIN)
            {
                printf("Error: Invalid useCalibration value: %s\n", argv[6]);
                return -1;
            }
            if (val != 0 && val != 1)
            {
                printf("Error: Invalid useCalibration value: %d (must be 0 or 1)\n", (int32_t)val);
                return -1;
            }
            useCalibration = (int32_t)val;
        }

        printf("Parameters set from command line:\n");
        printf("  numDetections       = %d\n", numDetections);
        printf("  numRangeBins        = %d\n", numRangeBins);
        printf("  numDopplerBins      = %d\n", numDopplerBins);
        printf("  numDopplerFolds     = %d\n", numDopplerFolds);
        printf("  numRx               = %d\n", numRx);
        printf("  numTx               = %d\n", numTx);
        printf("  rangeDopplerFormat  = %d (%s)\n", rangeDopplerFormat,
               rangeDopplerFormat == PVA_FFT_LAYOUT_RANGE_RX_DOPPLER ? "RANGE_RX_DOPPLER" : "RANGE_DOPPLER_RX");
        printf("  useCalibration      = %d (%s)\n", useCalibration,
               useCalibration ? "CALIBRATION WEIGHTS" : "DDMA WEIGHTS");

        // Validate max detections limit
        if (numDetections > 8192)
        {
            printf("Error: numDetections (%d) cannot be greater than 8192\n", numDetections);
            return -1;
        }
    }
    else if (argc > 1)
    {
        // Require all parameters if any are provided
        printf(
            "Usage: %s [<numDetections> <numDopplerFolds> <numRx> <numTx> [<rangeDopplerFormat>] [<useCalibration>]]\n",
            argv[0]);
        printf("\nParameters:\n");
        printf("  numDetections       : Number of detections to process (1-8192)\n");
        printf("  numDopplerFolds     : Number of DDM folds (typically equals numTx)\n");
        printf("  numRx               : Number of RX antennas (supported: 4 or 8)\n");
        printf("  numTx               : Number of TX antennas (supported: 4 or 8)\n");
        printf("                        Supported TX×RX configurations: 4×4, 8×8\n");
        printf("  rangeDopplerFormat  : Range-Doppler map format (optional)\n");
        printf("                        1 = RANGE_RX_DOPPLER (default)\n");
        printf("                        2 = RANGE_DOPPLER_RX\n");
        printf("  useCalibration      : Use calibration weights (optional)\n");
        printf("                        0 = DDMA weights (default)\n");
        printf("                        1 = Calibration weights\n");
        printf("\nFirst 4 parameters are required when specifying command line arguments.\n");
        printf("The 5th parameter (rangeDopplerFormat) and 6th parameter (useCalibration) are optional.\n");
        printf("\nDefault values (if no arguments provided):\n");
        printf("  numDetections=10, numDopplerFolds=16, numRx=8, numTx=8, rangeDopplerFormat=1, useCalibration=0\n");
        return -1;
    }
    else
    {
        // Print the configuration used (default values)
        printf("Parameters set (using defaults):\n");
        printf("  numDetections       = %d\n", numDetections);
        printf("  numDopplerFolds     = %d\n", numDopplerFolds);
        printf("  numRx               = %d\n", numRx);
        printf("  numTx               = %d\n", numTx);
        printf("  rangeDopplerFormat  = %d (%s)\n", rangeDopplerFormat,
               rangeDopplerFormat == PVA_FFT_LAYOUT_RANGE_RX_DOPPLER ? "RANGE_RX_DOPPLER" : "RANGE_DOPPLER_RX");
        printf("  useCalibration      = %d (%s)\n", useCalibration,
               useCalibration ? "CALIBRATION WEIGHTS" : "DDMA WEIGHTS");
    }

    // Configure Snapshot Extraction operator parameters (value-initialize to zero all fields)
    PVASnapshotExtractionParams params = {0};
    params.numDopplerFolds             = numDopplerFolds;
    params.rangeDopplerFormat          = rangeDopplerFormat; // Use format from command line or default

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
        cudaStreamSubmitTestResult = SubmitSnapshotExtractionToCudaStream(
            numDetections, numDopplerFolds, numRangeBins, numDopplerBins, numRx, numTx, useCalibration, &params);
    }
#endif

    // Run test using native cuPVA stream (direct PVA control, lower latency)
    cupvaStreamSubmitTestResult = SubmitSnapshotExtractionToCupvaStream(
        numDetections, numDopplerFolds, numRangeBins, numDopplerBins, numRx, numTx, useCalibration, &params);

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
