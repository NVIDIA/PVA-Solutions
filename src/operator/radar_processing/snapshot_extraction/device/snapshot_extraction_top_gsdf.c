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
 * @file snapshot_extraction_top_gsdf.c
 * @brief VPU main kernel for Snapshot Extraction with DDM Disambiguation (GSDF variant)
 *
 * This file implements the GSDF-based VPU kernel for extracting antenna snapshots
 * from Range-Doppler maps with DDM (Doppler Division Multiplexing) disambiguation.
 *
 * GSDF Optimization:
 * - For RANGE_DOPPLER_RX format: Uses numTx tiles (each gathering numRx complex values)
 * - For RANGE_RX_DOPPLER format: Uses numTx * numRx tiles (each gathering one complex value)
 * - Reduces memory bandwidth by gathering only required data points
 */

#include "snapshot_extraction_knl.h"
#include "snapshot_extraction_param.h"
#include "snapshot_extraction_types.h"

#include <cupva_device.h>
#include <cupva_device/sequence_dataflow.h>

/* Scalar parameters stored in VMEM */
VMEM(A, int32_t, numDetections);
VMEM(A, int32_t, maxDetections);
VMEM(A, int32_t, numRangeBins);
VMEM(A, int32_t, numDopplerBins);
VMEM(A, int32_t, numRx);
VMEM(A, int32_t, numTx);
VMEM(A, int32_t, numDopplerFolds);
VMEM(A, int32_t, numChirpsPerFold);
VMEM(A, int32_t, rangeBatchSize);
VMEM(A, int32_t, rangeDopplerFormat);
VMEM(A, int32_t, useCalibrationWeights);
VMEM(A, uint32_t, ddmSpaceBitMask);
/* Coefficient buffers (loaded once from DRAM) */
VMEM(A, int32_t, weightsBuf, WEIGHTS_BUF_SIZE);
VMEM(A, int32_t, ddmSpaceBuf, DDM_SPACE_BUF_SIZE);

/* Input buffers (double-buffered for ping-pong) */
VMEM(A, int32_t, rdmBuf, RDF_DOUBLE(int32_t, RDM_BUF_SIZE, 1)); // Gathered RDM data using GSDF
VMEM(B, uint32_t, nciBuf, RDF_DOUBLE(uint32_t, NCI_BATCH_BUF_SIZE, 1));
VMEM(B, int32_t, foldedDetectionListBuf, DETECTION_LIST_BUF_SIZE);

/* Output buffers (double-buffered for snapshot output) */
VMEM(C, int32_t, detectionListOutBuf, DETECTION_LIST_BUF_SIZE);
VMEM(C, int32_t, snapshotsOutBuf, RDF_DOUBLE(int32_t, SNAPSHOTS_BUF_SIZE, 1));

/* Data Flow handlers */
VMEM_SEQDF_HANDLER(A, coeffHdl);
VMEM_GSDF_HANDLER(A, rdmHdl, GSDF_MAX_NUM_TILES); // GSDF for gathering RDM data; host caps tiles per trigger
VMEM_SEQDF_HANDLER(A, nciHdl);
VMEM_SEQDF_HANDLER(A, detectionListHdl);
VMEM_SEQDF_HANDLER(A, snapshotsOutHdl);
VMEM_SEQDF_HANDLER(A, detectionListOutHdl);

/* External memory pointers */
VMEM_POINTER(A, rdmExtPtr);
VMEM_POINTER(A, nciExtPtr);
VMEM_POINTER(A, snapshotsOutExtPtr);

/* Parameter structures for double-buffering */
VMEM(A, snapshot_extraction_params_t, params_snapshot_extraction, 2);

/**
 * @brief Update NCI SQDF to load NCI data for a specific range bin
 */
static inline void updateNciSQDF(SequenceDataFlowHandler &nciHdl, int32_t rangeIdx,
                                 snapshot_extraction_params_t *params)
{
    int64_t nciBase   = params->nci_ext_ptr->base + params->nci_ext_ptr->offset;
    int32_t nciOffset = rangeIdx * params->nci_range_stride;

    cupvaSQDFUpdateAddr(nciHdl, 0, nciBase + nciOffset, MEMTYPE_DRAM, cupvaGetVmemAddress((void *)params->nci_buf),
                        MEMTYPE_VMEM);

    cupvaSQDFFlushAndTrig(nciHdl);
}

/**
 * @brief Update output SQDF to store snapshot for a specific detection
 */
static inline void updateOutputSQDF(SequenceDataFlowHandler &snapshotsOutHdl, int32_t detectionIdx,
                                    snapshot_extraction_params_t *params)
{
    int64_t snapshotsOutBase = params->snapshots_out_ext_ptr->base + params->snapshots_out_ext_ptr->offset;
    int32_t snapshotOffset   = detectionIdx * params->snapshot_stride;

    cupvaSQDFUpdateAddr(snapshotsOutHdl, 0, cupvaGetVmemAddress((void *)params->snapshots_out_buf), MEMTYPE_VMEM,
                        snapshotsOutBase + snapshotOffset, MEMTYPE_DRAM);

    cupvaSQDFFlushAndTrig(snapshotsOutHdl);
}

/**
 * @brief Main VPU kernel entry point for Snapshot Extraction (GSDF variant)
 */
CUPVA_VPU_MAIN()
{
    // Calculate buffer sizes for double-buffering
    const int32_t rdmBufSize       = RDM_BUF_SIZE;
    const int32_t nciBufSize       = NCI_BATCH_BUF_SIZE;
    const int32_t snapshotsBufSize = SNAPSHOTS_BUF_SIZE;

    // Calculate strides
    int32_t nciRangeStride = numDopplerBins * sizeof(uint32_t);
    int32_t snapshotStride = numTx * numRx * COMPLEX_ELEMENT_NUM * sizeof(int32_t);

    // Initialize two parameter sets for double-buffered processing
    for (int32_t i = 0; i < 2; i++)
    {
        // Processing parameters
        params_snapshot_extraction[i].numRx            = numRx;
        params_snapshot_extraction[i].numTx            = numTx;
        params_snapshot_extraction[i].numDopplerFolds  = numDopplerFolds;
        params_snapshot_extraction[i].numDopplerBins   = numDopplerBins;
        params_snapshot_extraction[i].numChirpsPerFold = numChirpsPerFold;
        params_snapshot_extraction[i].rangeIdx         = -1;

        // Double-buffered I/O pointers
        params_snapshot_extraction[i].rdm_buf           = ((int32_t *)rdmBuf) + (i * rdmBufSize);
        params_snapshot_extraction[i].nci_buf           = ((uint32_t *)nciBuf) + (i * nciBufSize);
        params_snapshot_extraction[i].snapshots_out_buf = ((int32_t *)snapshotsOutBuf) + (i * snapshotsBufSize);

        // Shared coefficient buffers
        params_snapshot_extraction[i].weights_buf             = weightsBuf;
        params_snapshot_extraction[i].ddm_space_buf           = ddmSpaceBuf;
        params_snapshot_extraction[i].use_calibration_weights = useCalibrationWeights;

        // External memory pointers
        params_snapshot_extraction[i].rdm_ext_ptr           = &rdmExtPtr;
        params_snapshot_extraction[i].nci_ext_ptr           = &nciExtPtr;
        params_snapshot_extraction[i].snapshots_out_ext_ptr = &snapshotsOutExtPtr;

        // Stride information
        params_snapshot_extraction[i].nci_range_stride = nciRangeStride;
        params_snapshot_extraction[i].snapshot_stride  = snapshotStride;
    }

    // Load coefficients from DRAM
    cupvaSQDFOpen(coeffHdl);
    cupvaSQDFTrig(coeffHdl);
    cupvaSQDFSync(coeffHdl);
    cupvaSQDFClose(coeffHdl);

    // Use DDMspace from VMEM buffer (populated by host via DMA)
    int32_t *DDMspace = ddmSpaceBuf;

    // Load folded detection list from DRAM
    cupvaSQDFOpen(detectionListHdl);
    cupvaSQDFTrig(detectionListHdl);
    // while loading the detection list, copy the weights to all other rows to make it easier to apply weight correction
    if (!useCalibrationWeights)
    {
        int32_t temp_weightsBuf[MAX_NUM_TX * 2];
        for (int32_t i = 0; i < MAX_NUM_TX * 2; i++)
        {
            temp_weightsBuf[i] = weightsBuf[i];
        }

        // Copy first row of weights to all other rows to make it easier to apply weight correction
        for (int32_t i = 0; i < numTx; i++)
        {
            for (int32_t j = 0; j < numRx; j++)
            {
                int32_t index         = (i * numRx + j) * 2;
                weightsBuf[index]     = temp_weightsBuf[i * 2];
                weightsBuf[index + 1] = temp_weightsBuf[i * 2 + 1];
            }
        }
    }
    cupvaSQDFSync(detectionListHdl);
    cupvaSQDFClose(detectionListHdl);

    // Initialize agen configuration for both ping-pong buffers
    applyWeightCorrectionAgenInit(&params_snapshot_extraction[0]);
    applyWeightCorrectionAgenInit(&params_snapshot_extraction[1]);

    // GSDF path: Optimization depends on RDM format
    // For RANGE_DOPPLER_RX: All Rx data are contiguous, use numTx tiles (each tile gathers numRx complex values)
    // For RANGE_RX_DOPPLER: Use numTx * numRx tiles (each tile gathers one complex value)
    int32_t rdmWidth, tileWidth, totalTilesPerDetection;
    if (rangeDopplerFormat == RANGE_DOPPLER_FORMAT_RANGE_DOPPLER_RX)
    {
        // RANGE_DOPPLER_RX format: [numRangeBins, numDopplerBins, numRx]
        rdmWidth               = numDopplerBins * numRx * 2; // 2 for complex (real, imag)
        tileWidth              = numRx * 2;                  // Each tile: numRx complex values
        totalTilesPerDetection = numTx;                      // One tile per Tx
    }
    else // RANGE_DOPPLER_FORMAT_RANGE_RX_DOPPLER
    {
        // RANGE_RX_DOPPLER format: [numRangeBins, numRx, numDopplerBins]
        rdmWidth               = numRx * numDopplerBins * 2; // 2 for complex (real, imag)
        tileWidth              = 2;                          // Each tile: one complex value
        totalTilesPerDetection = numTx * numRx;              // One tile per Tx-Rx pair
    }

    cupvaGSDFOpen(rdmHdl, params_snapshot_extraction[0].rdm_buf, tileWidth, tileWidth);
    cupvaGSDFResetExtBase(rdmHdl, &rdmExtPtr, rdmWidth, numRangeBins, rdmWidth);

    cupvaSQDFOpen(nciHdl);
    cupvaSQDFOpen(snapshotsOutHdl);

    numDetections = numDetections > maxDetections ? maxDetections : numDetections;

    // Handle zero detection count: skip processing if no detections
    if (numDetections != 0)
    {
        // Prefetch first detection's NCI data into ping buffer
        int32_t firstRangeIdx = foldedDetectionListBuf[0];
        updateNciSQDF(nciHdl, firstRangeIdx, &params_snapshot_extraction[0]);
        params_snapshot_extraction[0].rangeIdx = firstRangeIdx;

        // Allocate ping-pong buffers for GSDF gather offsets to avoid race conditions
        // (one for current GSDF in flight, one for next computation)
        int32_t gatherOffsets[2][MAX_NUM_TX * MAX_NUM_RX];

        // Prefetch first detection: sync NCI, compute doppler/offsets, and trigger GSDF
        {
            int32_t detIdx                              = 0;
            int32_t bufferIdx                           = detIdx & 1;
            snapshot_extraction_params_t *currentParams = &params_snapshot_extraction[bufferIdx];

            int32_t rIdx          = foldedDetectionListBuf[detIdx * 2];
            int32_t foldedDoppIdx = foldedDetectionListBuf[detIdx * 2 + 1];

            // Sync first NCI load (triggered in initialization)
            cupvaSQDFSync(nciHdl);

            // Trigger NCI load for second detection if it exists (overlap with det 0 computation)
            if (numDetections > 1)
            {
                int32_t nextRangeIdx                     = foldedDetectionListBuf[2];
                int32_t nextBufferIdx                    = 1;
                snapshot_extraction_params_t *nextParams = &params_snapshot_extraction[nextBufferIdx];
                updateNciSQDF(nciHdl, nextRangeIdx, nextParams);
                nextParams->rangeIdx = nextRangeIdx;
            }

            // Find best TX1 doppler index using NCI data
            int32_t localRIdx = 0;
            int32_t bestDopplerIdxs[MAX_NUM_TX];
            int32_t tx1DoppIdx;
            tx1DoppIdx = gatherNciRxTileComputeDopplerIdx(currentParams->nci_buf, currentParams->numTx,
                                                          currentParams->numDopplerBins, currentParams->numDopplerFolds,
                                                          foldedDoppIdx, ddmSpaceBitMask);

            for (int32_t txIdx = 0; txIdx < currentParams->numTx; txIdx++)
            {
                bestDopplerIdxs[txIdx] = (DDMspace[txIdx] + tx1DoppIdx) % currentParams->numDopplerBins;
            }
            // Write unfolded detection to output buffer
            detectionListOutBuf[detIdx * 2]     = rIdx;
            detectionListOutBuf[detIdx * 2 + 1] = tx1DoppIdx;

            // Compute offsets and trigger GSDF gather for first detection
            computeRdmGatherOffsets(rIdx, bestDopplerIdxs, currentParams->numRx, currentParams->numTx,
                                    currentParams->numDopplerBins, rangeDopplerFormat, gatherOffsets[bufferIdx]);
            int32_t *vmemDest = currentParams->rdm_buf;
            cupvaGSDFUpdateVmemAddr(rdmHdl, vmemDest, tileWidth, tileWidth);
            cupvaGSDFUpdateOffsets1D(rdmHdl, &gatherOffsets[bufferIdx][0]);
            cupvaGSDFTrig(rdmHdl); // Trigger GSDF for det 0
        }

        // Process detections 0 to numDetections-2 with full pipelining
        for (int32_t detIdx = 0; detIdx < (numDetections - 1); detIdx++)
        {
            // Determine which parameter set to use for current detection (ping=0, pong=1)
            int32_t bufferIdx                           = detIdx & 1;
            snapshot_extraction_params_t *currentParams = &params_snapshot_extraction[bufferIdx];

            // Prepare next detection (detIdx + 1) while current GSDF is in flight
            int32_t nextDetIdx                       = detIdx + 1;
            int32_t nextBufferIdx                    = nextDetIdx & 1;
            snapshot_extraction_params_t *nextParams = &params_snapshot_extraction[nextBufferIdx];

            int32_t nextRIdx          = foldedDetectionListBuf[nextDetIdx * 2];
            int32_t nextFoldedDoppIdx = foldedDetectionListBuf[nextDetIdx * 2 + 1];

            // Sync NCI for next detection (this overlaps with current GSDF transfer)
            cupvaSQDFSync(nciHdl);

            // Load NCI data for detection after next (if available)
            if (nextDetIdx < (numDetections - 1))
            {
                int32_t nextNextRangeIdx                     = foldedDetectionListBuf[(nextDetIdx + 1) * 2];
                int32_t nextNextBufferIdx                    = (nextDetIdx + 1) & 1;
                snapshot_extraction_params_t *nextNextParams = &params_snapshot_extraction[nextNextBufferIdx];
                updateNciSQDF(nciHdl, nextNextRangeIdx, nextNextParams);
                nextNextParams->rangeIdx = nextNextRangeIdx;
            }

            // Find best TX1 doppler index for next detection (overlaps with current GSDF transfer)
            int32_t localRIdx = 0;
            int32_t bestDopplerIdxs[MAX_NUM_TX];
            int32_t tx1DoppIdx;
            tx1DoppIdx =
                gatherNciRxTileComputeDopplerIdx(nextParams->nci_buf, nextParams->numTx, nextParams->numDopplerBins,
                                                 nextParams->numDopplerFolds, nextFoldedDoppIdx, ddmSpaceBitMask);

            for (int32_t txIdx = 0; txIdx < nextParams->numTx; txIdx++)
            {
                bestDopplerIdxs[txIdx] = (DDMspace[txIdx] + tx1DoppIdx) % nextParams->numDopplerBins;
            }

            // Write unfolded detection to output buffer
            detectionListOutBuf[nextDetIdx * 2]     = nextRIdx;
            detectionListOutBuf[nextDetIdx * 2 + 1] = tx1DoppIdx;

            // Compute gather offsets for next detection (use nextBufferIdx to avoid race with current GSDF)
            computeRdmGatherOffsets(nextRIdx, bestDopplerIdxs, nextParams->numRx, nextParams->numTx,
                                    nextParams->numDopplerBins, rangeDopplerFormat, gatherOffsets[nextBufferIdx]);

            // Sync GSDF for current detection (most of transfer should be complete by now)
            cupvaGSDFSync(rdmHdl);

            // Trigger GSDF for next detection
            int32_t *vmemDest = nextParams->rdm_buf;
            cupvaGSDFUpdateVmemAddr(rdmHdl, vmemDest, tileWidth, tileWidth);
            cupvaGSDFUpdateOffsets1D(rdmHdl, &gatherOffsets[nextBufferIdx][0]);
            cupvaGSDFTrig(rdmHdl); // Trigger GSDF for next detection

            // Apply weight correction (DDMA or calibration) to gathered RDM data
            applyWeightCorrection(currentParams);

            // Wait for previous snapshot transfer to complete before starting new one
            if (detIdx > 0)
            {
                cupvaSQDFSync(snapshotsOutHdl);
            }

            // Start transferring current snapshot to DRAM
            updateOutputSQDF(snapshotsOutHdl, detIdx, currentParams);
        }

        // Process last detection (numDetections - 1) without prefetching
        {
            int32_t detIdx                              = numDetections - 1;
            int32_t bufferIdx                           = detIdx & 1;
            snapshot_extraction_params_t *currentParams = &params_snapshot_extraction[bufferIdx];

            // Sync GSDF for last detection
            cupvaGSDFSync(rdmHdl);

            // Apply weight correction (DDMA or calibration) to gathered RDM data
            applyWeightCorrection(currentParams);

            // Wait for previous snapshot transfer to complete (only if this is not the first detection)
            if (numDetections > 1)
            {
                cupvaSQDFSync(snapshotsOutHdl);
            }

            // Start transferring final snapshot to DRAM
            updateOutputSQDF(snapshotsOutHdl, detIdx, currentParams);
        }

        // Wait for final snapshot transfer to complete
        cupvaSQDFSync(snapshotsOutHdl);
    }

    // Close data flow handlers
    cupvaGSDFClose(rdmHdl);
    cupvaSQDFClose(nciHdl);
    cupvaSQDFClose(snapshotsOutHdl);

    // Transfer detection list for all detections in a single transfer
    cupvaSQDFOpen(detectionListOutHdl);
    cupvaSQDFTrig(detectionListOutHdl);
    cupvaSQDFSync(detectionListOutHdl);
    cupvaSQDFClose(detectionListOutHdl);

    return 0;
}
