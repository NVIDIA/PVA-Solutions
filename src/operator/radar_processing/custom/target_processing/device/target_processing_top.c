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
 * @file target_processing_top.c
 * @brief VPU main kernel for Target Processing
 *
 * This file implements the PVA (Programmable Vision Accelerator) VPU kernel for
 * target processing, which performs Range-Velocity (RV) decoupling and Cartesian
 * coordinate computation for radar targets.
 *
 * Algorithm Overview:
 * 1. Read input data (detection list, target angles, DDM offsets)
 * 2. Process targets in batches:
 *    a. Convert range bin index to physical range
 *    b. Convert Doppler bin index to velocity (with RV decoupling)
 *    c. Convert spherical coordinates (range, angles) to Cartesian (X, Y, Z)
 * 3. Write output target list with all computed properties
 *
 * Key Features:
 * - Double-buffered I/O for overlapping computation and data transfer
 * - Batched processing for vectorization efficiency
 * - SQDF (Sequence Data Flow) for efficient DRAM ↔ VMEM transfers
 */

#include "target_processing_knl.h"
#include "target_processing_param.h"

#include <cupva_device.h>

/* Scalar parameters stored in VMEM bank A */
VMEM(A, float, rangeResolution);
VMEM(A, float, sweepTime);
VMEM(A, float, wavelength);
VMEM(A, int32_t, numChirps);
VMEM(A, int32_t, fftShiftedDoppler);
VMEM(A, int32_t, targetCount);
VMEM(A, int32_t, detectionCount);
VMEM(A, int32_t, maxTargetCount);
VMEM(A, int32_t, power_output_enabled);
VMEM(A, int32_t, longitudinalAxis);
VMEM(A, int32_t, groundProjection);

/* RV Decoupling parameters struct */
VMEM(A, rv_decoupling_params_t, rvParams);

/* Detection list buffer*/
VMEM(A, int32_t, detectionList, RDF_SINGLE(int32_t, DETECTION_LIST_TILE_PITCH, DETECTION_LIST_TILE_HEIGHT));
VMEM(B, int32_t, detectionListRangeIndex, DETECTION_LIST_TILE_HEIGHT);
VMEM(C, int32_t, detectionListDopplerIndex, DETECTION_LIST_TILE_HEIGHT);

VMEM(C, int32_t, rangeBinIndex, TARGET_INDEX_MAP_TILE_PITCH);
VMEM(B, int32_t, dopplerBinIndex, TARGET_INDEX_MAP_TILE_PITCH);

/* DDM Doppler offsets buffer */
VMEM(A, float, ddmDopplerOffsets, DDM_DOPPLER_OFFSETS_TILE_PITCH);

/* Target index map and target angles buffer */
VMEM(B, int32_t, targetIndexMap, RDF_DOUBLE(int32_t, TARGET_INDEX_MAP_TILE_PITCH, TARGET_INDEX_MAP_TILE_HEIGHT));
VMEM(B, float, targetAngles, RDF_DOUBLE(float, TARGET_ANGLES_TILE_PITCH, TARGET_ANGLES_TILE_HEIGHT));

/* Target list buffer */
VMEM(C, float, targetList, RDF_DOUBLE(float, TARGET_LIST_TILE_PITCH, TARGET_LIST_TILE_HEIGHT));

/* Sequence Data Flow (SQDF) handlers for data transfers (VMEM bank A) */
VMEM_SEQDF_HANDLER(A, targetCntDetectionListAndDdmDopplerOffsetsHdl);

VMEM_SEQDF_HANDLER(A, targetIndexMapHdl);
VMEM_SEQDF_HANDLER(A, targetAnglesHdl);
VMEM_SEQDF_HANDLER(A, targetListHdl);

/* External memory pointers for SQDF (VMEM bank A) */
VMEM_POINTER(A, targetIndexMapExtPtr);
VMEM_POINTER(A, targetAnglesExtPtr);
VMEM_POINTER(A, targetListExtPtr);

/* Samplers for range bin index and velocity bin index (VMEM bank A) */
VMEM(A, CupvaSampler, samplerRange, 2);
VMEM(A, CupvaSampler, samplerVelocity, 2);

/* NCI data for quadratic interpolation (VMEM bank A) */
VMEM(A, uint32_t, nciData, NCI_NODE_SIZE *NUM_NCI_NODES);
VMEM_GSDF_HANDLER(A, nciHdl, NUM_NCI_NODES);
VMEM_SEQDF_HANDLER(A, nciPaddingHdl); // SQDF for DRAM→L2 padding transfer
VMEM(A, int32_t, nciPaddedWidth);
VMEM(A, int32_t, nciPaddedHeight);

/* Folded doppler indices for NCI GSDF access */
VMEM(A, int32_t, dopplerFoldedIndex, BATCH_SIZE);

/* Parameter structures for double-buffering (VMEM bank A) */
VMEM(A, target_processing_params_t, params_target_processing, 2);

/**
 * @brief Main VPU kernel entry point for Target Processing
 *
 * This function performs Range-Velocity decoupling and Cartesian coordinate
 * computation for radar targets. It processes targets in batches for efficiency.
 *
 * Processing Steps:
 * 1. Load input data (target count, detection list, angles, DDM offsets)
 * 2. For each batch of targets:
 *    a. Load target data for the batch
 *    b. Compute velocity, range, and Cartesian coordinates
 *    c. Write results to output
 * 3. Final writeback of all results
 */
CUPVA_VPU_MAIN()
{

    // Calculate buffer sizes for double-buffering
    int32_t targetIndexMapTileBufSize = TARGET_INDEX_MAP_TILE_HEIGHT * TARGET_INDEX_MAP_TILE_PITCH;
    int32_t targetAnglesTileBufSize   = TARGET_ANGLES_TILE_HEIGHT * TARGET_ANGLES_TILE_PITCH;
    int32_t targetListTileBufSize     = TARGET_LIST_TILE_HEIGHT * TARGET_LIST_TILE_PITCH;

    // Load the entire detection list and DDM Doppler offsets
    cupvaSQDFOpen(targetCntDetectionListAndDdmDopplerOffsetsHdl);
    cupvaSQDFTrig(targetCntDetectionListAndDdmDopplerOffsetsHdl);
    cupvaSQDFSync(targetCntDetectionListAndDdmDopplerOffsetsHdl);
    cupvaSQDFClose(targetCntDetectionListAndDdmDopplerOffsetsHdl);

    // Initialize parameters structure for both double buffers
    for (int32_t i = 0; i < 2; i++) chess_unroll_loop(*)
    {
        params_target_processing[i].rangeResolution   = rangeResolution;
        params_target_processing[i].sweepTime         = sweepTime;
        params_target_processing[i].wavelength        = wavelength;
        params_target_processing[i].numChirps         = numChirps;
        params_target_processing[i].fftShiftedDoppler = fftShiftedDoppler;
        params_target_processing[i].numTargets        = targetCount;
        params_target_processing[i].numDetections     = detectionCount;

        // Pre-compute derived parameters on VPU (once, outside inner loops)
        params_target_processing[i].invSweepTime = frcp(sweepTime);
        params_target_processing[i].invNumChirps = frcp((float)numChirps);

        // Setup RV decoupling parameters (pointer to struct passed via .set())
        params_target_processing[i].rvParams = &rvParams;

        // Setup external memory pointers for SQDF updates
        params_target_processing[i].targetIndexMapExtPtr = &targetIndexMapExtPtr;
        params_target_processing[i].targetAnglesExtPtr   = &targetAnglesExtPtr;
        params_target_processing[i].targetListExtPtr     = &targetListExtPtr;

        // Setup double-buffered pointers (offset by buffer size for second set)
        params_target_processing[i].targetIndexMap = ((int32_t *)targetIndexMap) + (i * targetIndexMapTileBufSize);
        params_target_processing[i].targetAngles   = ((float *)targetAngles) + (i * targetAnglesTileBufSize);
        params_target_processing[i].targetList     = ((float *)targetList) + (i * targetListTileBufSize);

        params_target_processing[i].ddmDopplerOffsets         = ((float *)ddmDopplerOffsets);
        params_target_processing[i].detectionList             = ((int32_t *)detectionList);
        params_target_processing[i].detectionListRangeIndex   = ((int32_t *)detectionListRangeIndex);
        params_target_processing[i].detectionListDopplerIndex = ((int32_t *)detectionListDopplerIndex);
        params_target_processing[i].rangeBinIndex             = ((int32_t *)rangeBinIndex);
        params_target_processing[i].dopplerBinIndex           = ((int32_t *)dopplerBinIndex);

        // Setup samplers for range bin index and velocity bin index
        params_target_processing[i].samplerRange    = &samplerRange[i];
        params_target_processing[i].samplerVelocity = &samplerVelocity[i];

        params_target_processing[i].power_output_requested = (power_output_enabled == 1) ? true : false;

        // Coordinate system options
        params_target_processing[i].longitudinalAxis = longitudinalAxis;
        params_target_processing[i].groundProjection = groundProjection;

        // NCI data for quadratic interpolation
        // If enableQuadraticInterpolation is true and GSDF is set up, use the NCI buffers
        if (rvParams.enableQuadraticInterpolation && nciPaddedWidth > 0)
        {
            params_target_processing[i].nciData            = (uint32_t *)nciData;
            params_target_processing[i].nciHdl             = nciHdl; // VMEM_GSDF_HANDLER creates pointer array
            params_target_processing[i].dopplerFoldedIndex = dopplerFoldedIndex; // Pre-computed folded indices
            params_target_processing[i].nciWidth           = nciPaddedWidth - 2; // Unpadded NCI width for folding
        }
        else
        {
            params_target_processing[i].nciData            = NULL;
            params_target_processing[i].nciHdl             = NULL;
            params_target_processing[i].dopplerFoldedIndex = NULL;
            params_target_processing[i].nciWidth           = 0;
        }
        params_target_processing[i].detectionListFolded = NULL;
    }

    int32_t numTargets = targetCount;

    // Open target index map and target angles data flow handler
    cupvaSQDFOpen(targetIndexMapHdl);
    cupvaSQDFOpen(targetAnglesHdl);
    cupvaSQDFOpen(targetListHdl);

    numTargets = numTargets > maxTargetCount ? maxTargetCount : numTargets;

    // Set up NCI quadratic interpolation if enabled
    if (rvParams.enableQuadraticInterpolation && nciPaddedWidth > 0)
    {
        // First, transfer NCI data from DRAM to L2 with cyclic padding via SQDF
        // This must complete before GSDF reads from L2
        cupvaSQDFOpen(nciPaddingHdl);
        cupvaSQDFTrig(nciPaddingHdl);
        cupvaSQDFSync(nciPaddingHdl);
        cupvaSQDFClose(nciPaddingHdl);

        // Now open GSDF handler to read from L2 (padded NCI buffer)
        cupvaGSDFOpen(nciHdl, &nciData[0], NCI_NODE_WIDTH, NCI_NODE_SIZE);

        // Reset external base to point to L2 memory (padded NCI)
        ExtMemPointer nciL2Ptr;
        cupvaGetL2ExtMemPointer(&nciL2Ptr, 0U);

        cupvaGSDFResetExtBase(nciHdl, &nciL2Ptr, nciPaddedWidth, nciPaddedHeight, nciPaddedWidth);
        cupvaGSDFCycleBuffers(nciHdl);

        cupvaGSDFResetExtBase(nciHdl, &nciL2Ptr, nciPaddedWidth, nciPaddedHeight, nciPaddedWidth);
        cupvaGSDFCycleBuffers(nciHdl);
    }

    if (numTargets != 0)
    {
        // Process targets in batches
        const int32_t batchSize = BATCH_SIZE;
        int32_t numBatches      = (numTargets + batchSize - 1) / batchSize;

        // Calculate first batch size and start loading
        int32_t firstBatchSize = (numTargets < batchSize) ? numTargets : batchSize;
        updateInputSQDF(targetIndexMapHdl, targetAnglesHdl, 0, firstBatchSize, &params_target_processing[0]);

        detection_list_transpose_init(&params_target_processing[0]);
        detection_list_transpose_kernel(&params_target_processing[0]);

        target_processing_init(&params_target_processing[0], &params_target_processing[1]);

        for (int32_t batchIdx = 0; batchIdx < numBatches; batchIdx++)
        {
            int32_t targetIdx        = batchIdx * batchSize;
            int32_t currentBatchSize = (targetIdx + batchSize <= numTargets) ? batchSize : (numTargets - targetIdx);

            // Determine ping-pong buffer index
            int32_t bufIdx = batchIdx % 2;

            // Wait for current batch loads to complete
            cupvaSQDFSync(targetIndexMapHdl);
            cupvaSQDFSync(targetAnglesHdl);

            // Prefetch next batch if not last iteration (double-buffering)
            if (batchIdx < (numBatches - 1))
            {
                int32_t nextTargetIdx = (batchIdx + 1) * batchSize;
                int32_t nextBatchSize =
                    (nextTargetIdx + batchSize <= numTargets) ? batchSize : (numTargets - nextTargetIdx);
                updateInputSQDF(targetIndexMapHdl, targetAnglesHdl, batchIdx + 1, nextBatchSize,
                                &params_target_processing[(batchIdx & 1) ^ 1]);
            }

            // Process this batch of targets
            target_processing_kernel(&params_target_processing[bufIdx], currentBatchSize);

            /* Wait for previous output transfer to complete (except first iteration) */
            if (batchIdx)
            {
                cupvaSQDFSync(targetListHdl);
            }

            // Trigger output data store for this batch
            updateOutputSQDF(targetListHdl, batchIdx, currentBatchSize, &params_target_processing[bufIdx]);
        }
        // Wait for the last batch's stores to complete
        cupvaSQDFSync(targetListHdl);
    }

    // Close all data flow handlers
    cupvaSQDFClose(targetIndexMapHdl);
    cupvaSQDFClose(targetAnglesHdl);
    cupvaSQDFClose(targetListHdl);

    // Close GSDF for NCI if it was opened
    if (rvParams.enableQuadraticInterpolation && nciPaddedWidth > 0)
    {
        cupvaGSDFClose(nciHdl);
    }

    return 0;
}
