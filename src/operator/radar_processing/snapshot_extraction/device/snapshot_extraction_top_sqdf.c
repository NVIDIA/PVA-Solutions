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
 * @file snapshot_extraction_top_sqdf.c
 * @brief VPU main kernel for Snapshot Extraction with DDM Disambiguation (SQDF variant)
 *
 * This file implements the SQDF-based VPU kernel for extracting antenna snapshots
 * from Range-Doppler maps with DDM (Doppler Division Multiplexing) disambiguation.
 *
 * SQDF Approach:
 * - Loads full RDM row for each range bin
 * - Extracts required snapshot data from loaded row
 * - Simpler dataflow, suitable for small antenna configurations
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
VMEM(A, int32_t, rdmBuf, RDF_DOUBLE(int32_t, RDM_BUF_SIZE, 1)); // Full RDM batch using SQDF
VMEM(A, int32_t, gatheredRdmBuf, RDM_BUF_SIZE);                 // Gathered RDM data using SQDF
VMEM(B, uint32_t, nciBuf, RDF_DOUBLE(uint32_t, NCI_BATCH_BUF_SIZE, 1));
VMEM(B, int32_t, foldedDetectionListBuf, DETECTION_LIST_BUF_SIZE);

/* Output buffers (double-buffered for snapshot output) */
VMEM(C, int32_t, detectionListOutBuf, DETECTION_LIST_BUF_SIZE);
VMEM(C, int32_t, snapshotsOutBuf, RDF_DOUBLE(int32_t, SNAPSHOTS_BUF_SIZE, 1));

/* Data Flow handlers */
VMEM_SEQDF_HANDLER(A, coeffHdl);
VMEM_SEQDF_HANDLER(A, rdmHdl); // SQDF for loading full RDM row
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
 * @brief Update input SQDF to load RDM and NCI data for a specific range bin
 */
static inline void updateInputSQDF(SequenceDataFlowHandler &rdmHdl, SequenceDataFlowHandler &nciHdl, int32_t rangeIdx,
                                   snapshot_extraction_params_t *params)
{
    int64_t rdmBase = params->rdm_ext_ptr->base + params->rdm_ext_ptr->offset;
    int64_t nciBase = params->nci_ext_ptr->base + params->nci_ext_ptr->offset;

    int32_t rdmOffset = rangeIdx * params->rdm_range_stride;
    int32_t nciOffset = rangeIdx * params->nci_range_stride;

    cupvaSQDFUpdateAddr(rdmHdl, 0, rdmBase + rdmOffset, MEMTYPE_DRAM, cupvaGetVmemAddress((void *)params->in_rdm_buf),
                        MEMTYPE_VMEM);
    cupvaSQDFUpdateAddr(nciHdl, 0, nciBase + nciOffset, MEMTYPE_DRAM, cupvaGetVmemAddress((void *)params->nci_buf),
                        MEMTYPE_VMEM);

    cupvaSQDFFlushAndTrig(rdmHdl);
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
 * @brief Main VPU kernel entry point for Snapshot Extraction (SQDF variant)
 */
CUPVA_VPU_MAIN()
{
    // Calculate buffer sizes for double-buffering
    const int32_t rdmBufSize       = RDM_BUF_SIZE;
    const int32_t nciBufSize       = NCI_BATCH_BUF_SIZE;
    const int32_t snapshotsBufSize = SNAPSHOTS_BUF_SIZE;

    // Calculate strides
    int32_t rdmRangeStride = numRx * numDopplerBins * COMPLEX_ELEMENT_NUM * sizeof(int32_t);
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
        params_snapshot_extraction[i].in_rdm_buf        = ((int32_t *)rdmBuf) + (i * rdmBufSize);
        params_snapshot_extraction[i].nci_buf           = ((uint32_t *)nciBuf) + (i * nciBufSize);
        params_snapshot_extraction[i].snapshots_out_buf = ((int32_t *)snapshotsOutBuf) + (i * snapshotsBufSize);

        // Gathered RDM data [numTx, numRx] complex
        params_snapshot_extraction[i].rdm_buf = ((int32_t *)gatheredRdmBuf);

        // Shared coefficient buffers
        params_snapshot_extraction[i].weights_buf             = weightsBuf;
        params_snapshot_extraction[i].ddm_space_buf           = ddmSpaceBuf;
        params_snapshot_extraction[i].use_calibration_weights = useCalibrationWeights;

        // External memory pointers
        params_snapshot_extraction[i].rdm_ext_ptr           = &rdmExtPtr;
        params_snapshot_extraction[i].nci_ext_ptr           = &nciExtPtr;
        params_snapshot_extraction[i].snapshots_out_ext_ptr = &snapshotsOutExtPtr;

        // Stride information
        params_snapshot_extraction[i].rdm_range_stride = rdmRangeStride;
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

    // SQDF path: Load full RDM rows
    cupvaSQDFOpen(rdmHdl);
    cupvaSQDFOpen(nciHdl);
    cupvaSQDFOpen(snapshotsOutHdl);

    numDetections = numDetections > maxDetections ? maxDetections : numDetections;

    // Handle zero detection count: skip processing if no detections
    if (numDetections != 0)
    {
        // Prefetch first detection's range bin into ping buffer (params[0])
        int32_t firstRangeIdx = foldedDetectionListBuf[0];
        updateInputSQDF(rdmHdl, nciHdl, firstRangeIdx, &params_snapshot_extraction[0]);
        params_snapshot_extraction[0].rangeIdx = firstRangeIdx;

        // Process each detection using ping-pong buffers
        for (int32_t detIdx = 0; detIdx < numDetections; detIdx++)
        {
            // Determine which parameter set to use for current detection (ping=0, pong=1)
            int32_t bufferIdx                           = detIdx & 1;
            snapshot_extraction_params_t *currentParams = &params_snapshot_extraction[bufferIdx];

            // Get detection parameters
            int32_t rIdx          = foldedDetectionListBuf[detIdx * 2];
            int32_t foldedDoppIdx = foldedDetectionListBuf[detIdx * 2 + 1];

            // SQDF path: Load both RDM and NCI data
            cupvaSQDFSync(rdmHdl);
            cupvaSQDFSync(nciHdl);

            // Prefetch next detection's range bin if available
            if (detIdx < (numDetections - 1))
            {
                int32_t nextRangeIdx                     = foldedDetectionListBuf[(detIdx + 1) * 2];
                int32_t nextBufferIdx                    = (detIdx + 1) & 1;
                snapshot_extraction_params_t *nextParams = &params_snapshot_extraction[nextBufferIdx];

                updateInputSQDF(rdmHdl, nciHdl, nextRangeIdx, nextParams);
                nextParams->rangeIdx = nextRangeIdx;
            }

            // Local range index is always 0 since we load one range bin at a time
            int32_t localRIdx = 0;

            // Find best TX1 doppler index using NCI data
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

            // SQDF path: Extract snapshot from loaded RDM batch
            extractSnapshot(localRIdx, bestDopplerIdxs, rangeDopplerFormat, currentParams);

            // Apply weight correction (DDMA or calibration) to extracted snapshot
            applyWeightCorrection(currentParams);

            // Wait for previous snapshot transfer to complete
            if (detIdx > 0)
            {
                cupvaSQDFSync(snapshotsOutHdl);
            }

            // Start transferring current snapshot to DRAM
            updateOutputSQDF(snapshotsOutHdl, detIdx, currentParams);
        }

        // Wait for final snapshot transfer to complete
        cupvaSQDFSync(snapshotsOutHdl);
    }

    // Close data flow handlers
    cupvaSQDFClose(rdmHdl);
    cupvaSQDFClose(nciHdl);
    cupvaSQDFClose(snapshotsOutHdl);

    // Transfer detection list for all detections in a single transfer
    cupvaSQDFOpen(detectionListOutHdl);
    cupvaSQDFTrig(detectionListOutHdl);
    cupvaSQDFSync(detectionListOutHdl);
    cupvaSQDFClose(detectionListOutHdl);

    return 0;
}
