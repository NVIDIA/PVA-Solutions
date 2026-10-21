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

#include "snapshot_extraction_knl.h"

#include "snapshot_extraction_param.h"

/**
 * @brief Extract spatial snapshot and apply weight correction (DDMA or calibration)
 */
void extractSnapshot(int32_t rangeIdx, const int32_t *dopplerIndices, int32_t rangeDopplerFormat,
                     snapshot_extraction_params_t *params)
{
    // Gather spatial data from RDM into params->rdm_buf
    // Input: params->in_rdm_buf (full RDM batch for SQDF)
    // Output: params->rdm_buf (gathered snapshot data)
    const int32_t *rangeDopplerMapPtr = params->in_rdm_buf;
    int32_t *gatheredRdm              = params->rdm_buf;
    int32_t numRx                     = params->numRx;
    int32_t numTx                     = params->numTx;
    int32_t numDopplerBins            = params->numDopplerBins;

    for (int32_t txIdx = 0; txIdx < numTx; txIdx++)
    {
        int32_t dopplerIdx = dopplerIndices[txIdx];

        for (int32_t rxIdx = 0; rxIdx < numRx; rxIdx++)
        {
            // Extract complex value from range-doppler map
            // Layout depends on rangeDopplerFormat:
            // - RANGE_DOPPLER_FORMAT_RANGE_RX_DOPPLER: [numRangeBins, numRx, numDopplerBins]
            // - RANGE_DOPPLER_FORMAT_RANGE_DOPPLER_RX: [numRangeBins, numDopplerBins, numRx]
            // Complex S32 (2 int32 per element)
            int32_t rdmIndex;
            if (rangeDopplerFormat == RANGE_DOPPLER_FORMAT_RANGE_RX_DOPPLER)
            {
                // RANGE_RX_DOPPLER format: [R, Rx, D]
                rdmIndex = ((rangeIdx * numRx + rxIdx) * numDopplerBins + dopplerIdx) * 2;
            }
            else
            {
                // RANGE_DOPPLER_RX format: [R, D, Rx]
                rdmIndex = ((rangeIdx * numDopplerBins + dopplerIdx) * numRx + rxIdx) * 2;
            }

            // Gather into rdm_buf
            int32_t gatherIndex          = (txIdx * numRx + rxIdx) * 2;
            gatheredRdm[gatherIndex]     = rangeDopplerMapPtr[rdmIndex];
            gatheredRdm[gatherIndex + 1] = rangeDopplerMapPtr[rdmIndex + 1];
        }
    }
}

void computeRdmGatherOffsets(int32_t rangeIdx, const int32_t *bestDopplerIndices, int32_t numRx, int32_t numTx,
                             int32_t numDopplerBins, int32_t rangeDopplerFormat, int32_t *gatherOffsets)
{
    if (rangeDopplerFormat == RANGE_DOPPLER_FORMAT_RANGE_RX_DOPPLER)
    {
        // RANGE_RX_DOPPLER format: Need 64 offsets (8 Tx × 8 Rx)
        // Formula: offset = ((rangeIdx * numRx + rxIdx) * numDopplerBins + dopplerIdx) * 2

        // Precompute range base: rangeIdx * numRx * numDopplerBins * 2
        int32_t rangeBase = rangeIdx * numRx * numDopplerBins * 2;
        // Process each Tx
        int rxIdx[8]     = {0, 1, 2, 3, 4, 5, 6, 7};
        vintx v_rxIdx    = sign_extend(*(vint *)&rxIdx[0]);
        int32_t rxStride = numDopplerBins * 2;
        vintx v_rxStride = replicatew(rxStride);
        v_rxStride       = vmulwhw(v_rxStride, v_rxIdx, VPU_ROUND_0);

        vintx v_bestDopplerIndices = sign_extend(*(vint *)&bestDopplerIndices[0]);
        vintx v_dopplerOffset      = vmulwhw(v_bestDopplerIndices, replicatew(2), VPU_ROUND_0);
        vintx v_baseOffset_tx      = replicatew(rangeBase) + v_dopplerOffset;
        for (int32_t txIdx = 0; txIdx < numTx; txIdx++)
        {
            // Vector approach: compute all 8 Rx offsets at once
            // offset[rx] = rangeBase + (rxIdx * numDopplerBins * 2) + dopplerOffset
            // Compute base for this Tx: rangeBase + doppler contribution
            int32_t outputIdx                                          = txIdx * numRx;
            int32_t baseOffset                                         = vselectlane(v_baseOffset_tx, txIdx);
            vintx v_gatherOffsets                                      = replicatew(baseOffset) + v_rxStride;
            *(vint chess_storage(RAM_Bb) *)(&gatherOffsets[outputIdx]) = extract(v_gatherOffsets);
        }
    }
    else // RANGE_DOPPLER_FORMAT_RANGE_DOPPLER_RX
    {
        // RANGE_DOPPLER_RX format: Need only 8 offsets (one per Tx)
        // Formula: offset = (rangeIdx * numDopplerBins + dopplerIdx) * numRx * 2

        // Precompute range base: rangeIdx * numDopplerBins * numRx * 2
        int32_t rangeBase     = rangeIdx * numDopplerBins * numRx * 2;
        int32_t dopplerStride = numRx * 2; // Stride per doppler bin

        //gatherOffsets[x] = rangeBase + bestDopplerIndices[x] * dopplerStride;
        vintx v_bestDopplerIndices = sign_extend(*(vint *)&bestDopplerIndices[0]);
        vintx v_dopplerStride      = vmulwhw(v_bestDopplerIndices, replicatew(dopplerStride), VPU_ROUND_0);
        vintx v_gatherOffsets      = replicatew(rangeBase) + v_dopplerStride;
        *(vint chess_storage(RAM_Bb) *)(&gatherOffsets[0]) = extract(v_gatherOffsets);
    }
}

void applyWeightCorrectionAgenInit(snapshot_extraction_params_t *params)
{
    // Initialize agen configurations for weight correction
    // This function creates fully initialized agen objects and extracts their configurations
    // to avoid repeated INIT_AGEN1 calls in the processing loop

    const int32_t vecw  = 8; // chess_elementsof(dvintx) = 8 for int32_t vectors
    int32_t numElements = params->numTx * params->numRx;

    AgenWrapper wrapper;
    wrapper.size = sizeof(int32_t);
    wrapper.n1   = numElements / vecw;
    wrapper.s1   = vecw * 2; // 8 elements per vector, 2 int32 per complex

    // Initialize and extract configuration for weights buffer
    agen agen_weights      = init((void *)params->weights_buf);
    agen_weights.lane_ofst = 0;
    INIT_AGEN1(agen_weights, wrapper);
    params->cfg_weights = extract_agen_cfg(agen_weights);

    // Initialize and extract configuration for gathered RDM buffer
    agen agen_rdm      = init((void *)params->rdm_buf);
    agen_rdm.lane_ofst = 0;
    INIT_AGEN1(agen_rdm, wrapper);
    params->cfg_gathered_rdm = extract_agen_cfg(agen_rdm);

    // Initialize and extract configuration for output snapshots buffer
    agen agen_out      = init((void *)params->snapshots_out_buf);
    agen_out.lane_ofst = 0;
    agen_out.round     = params->use_calibration_weights ? (CALIBRATION_WEIGHTS_QBITS - 16) : (DDMA_WEIGHTS_QBITS - 16);
    INIT_AGEN1(agen_out, wrapper);
    params->cfg_snapshots_out = extract_agen_cfg(agen_out);

    // Set Q-format bits based on calibration mode
    params->qbits = params->use_calibration_weights ? CALIBRATION_WEIGHTS_QBITS : DDMA_WEIGHTS_QBITS;
}

void applyWeightCorrection(snapshot_extraction_params_t *params)
{
    // Apply weight correction using pre-initialized agen config from params
    // This avoids repeated INIT_AGEN1 calls by using init_agen_from_cfg()
    // to recreate agen objects from stored configurations

    int32_t numRx = params->numRx;
    int32_t numTx = params->numTx;

    // Recreate agen objects from pre-extracted configurations
    // Note: We only need to update the base pointers; the configuration is reused
    agen in = init_agen_from_cfg(params->cfg_weights);

    agen in2 = init_agen_from_cfg(params->cfg_gathered_rdm);

    agen out = init_agen_from_cfg(params->cfg_snapshots_out);

    for (int32_t iter = 0; iter < (numTx * numRx) / 8; iter++)
    {
        dvintx dv_calibrationWeights = dvint_load_di(in);
        dvintx dv_gatheredRdm        = dvint_load_di(in2);

        // Apply weight correction: gathered * weight
        // Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        dvintx dv_snapshotOut = dvcmulw_t16(dv_gatheredRdm, dv_calibrationWeights);

        // Store to output snapshot
        vstore_i(dv_snapshotOut, out);
    }
}

// The most time consuming part of this function is to calculate the sum of
// inPtr[ddmSpace[0]%REPEAT_FOLD] + inPtr[ddmSpace[1]%REPEAT_FOLD] + inPtr[ddmSpace[2]%REPEAT_FOLD] + inPtr[ddmSpace[3]%REPEAT_FOLD]
// This is achieved by the following steps:
// 1. Unpack the ddmSpace into the following form, (use ddmSpace={0, 7, 6, 4} as an example)
//    0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0
//    where 0 means the corresponding value in inPtr is used, 1 means the corresponding value in inPtr is not used.
//    Note that the unpacked sequence is of 2 * REPEAT_FOLD to avoid the periodicity access of ddmSpace.
// 2. Then take a position shift to get the bitmask position, e.g. 0, 1, 1, 1, 0, 1, 0, 0 mean position #0, #4, #6 and #7
//    Note that the above sequence is packed into a 32-bit integer, so that can leverage vbitunpackw to generate the per lane mask.
// 3. Use vmux to get the values to be accumulated, and then use vsumr_s to get the sum of the 4 values.
int32_t computeNciRxSumFindMax_16folds(uint32_t *nciRxGatheredPtr, uint32_t numDopplerFolds, uint32_t ddmSpaceBitMask)
{
    // Stores the sum of the 4 values in the DDMspace
    uint32_t sumBuf[MAX_DDM_FOLDS] = {0};
    dvintx dv0                     = zero_extend(*(dvuint *)nciRxGatheredPtr);

    for (int32_t i = 0, j = numDopplerFolds - 1; i < numDopplerFolds; i++, j--)
    {
        vintx v_mask0 = vbitunpackw(ddmSpaceBitMask);
        vintx v_mask1 = vbitunpackw(ddmSpaceBitMask >> 8);

        dvintx dv_mask;
        dv_mask.lo = v_mask0;
        dv_mask.hi = v_mask1;

        ddmSpaceBitMask = ddmSpaceBitMask >> 1;

        dvintx dv1 = dvmux(dv_mask, 0, dv0);
        sumBuf[j]  = vsumr_s(dv1.lo) + vsumr_s(dv1.hi);
    }

    // Find index of the max value in the sumBuf.
    vintx v2 = zero_extend(*(vuint *)(&sumBuf[0]));
    vintx v3 = zero_extend(*(vuint *)(&sumBuf[8]));
    int32_t index1, max1;
    vmaxrid_s(v2, max1, index1);
    int32_t index2, max2;
    vmaxrid_s(v3, max2, index2);
    int32_t index = max1 >= max2 ? index1 : index2 + 8;

    return index;
}

// The most time consuming part of this function is to calculate the sum of
// inPtr[ddmSpace[0]%REPEAT_FOLD] + inPtr[ddmSpace[1]%REPEAT_FOLD] + inPtr[ddmSpace[2]%REPEAT_FOLD] + inPtr[ddmSpace[3]%REPEAT_FOLD]
// This is achieved by the following steps:
// 1. Unpack the ddmSpace into the following form, (use ddmSpace={0, 7, 6, 4} as an example)
//    0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0
//    where 0 means the corresponding value in inPtr is used, 1 means the corresponding value in inPtr is not used.
//    Note that the unpacked sequence is of 2 * REPEAT_FOLD to avoid the periodicity access of ddmSpace.
// 2. Then take a position shift to get the bitmask position, e.g. 0, 1, 1, 1, 0, 1, 0, 0 mean position #0, #4, #6 and #7
//    Note that the above sequence is packed into a 32-bit integer, so that can leverage vbitunpackw to generate the per lane mask.
// 3. Use vmux to get the values to be accumulated, and then use vsumr_s to get the sum of the 4 values.
int32_t computeNciRxSumFindMax_8folds(uint32_t *nciRxGatheredPtr, uint32_t numDopplerFolds, uint32_t ddmSpaceBitMask)
{
    // Stores the sum of the 4 values in the DDMspace
    uint32_t sumBuf[MAX_DDM_FOLDS] = {0};
    vintx v0                       = zero_extend(*(vuint *)nciRxGatheredPtr);

    for (int32_t i = 0, j = numDopplerFolds - 1; i < numDopplerFolds; i++, j--)
    {
        vintx v_mask    = vbitunpackw(ddmSpaceBitMask);
        ddmSpaceBitMask = ddmSpaceBitMask >> 1;

        vintx v1  = vmux(v_mask, 0, v0);
        sumBuf[j] = vsumr_s(v1);
    }

    // Find index of the max value in the sumBuf.
    vintx v2 = zero_extend(*(vuint *)sumBuf);
    int32_t index, placeholder;
    vmaxrid_s(v2, placeholder, index);

    return index;
}

uint32_t gatherNciRxTileComputeDopplerIdx(uint32_t *nciRxTilePtr, uint32_t numTx, uint32_t numDopplerBins,
                                          uint32_t numDopplerFolds, uint32_t dopplerFoldIdx, uint32_t ddmSpaceBitMask)
{
    uint32_t nciRxTileBufGather[MAX_DDM_FOLDS] = {0};
    for (int32_t i = 0; i < numDopplerFolds; i++)
    {
        nciRxTileBufGather[i] =
            nciRxTilePtr[(dopplerFoldIdx + ((numDopplerBins / numDopplerFolds) * i)) % numDopplerBins];
    }

    int32_t index;
    if (numDopplerFolds > 8)
    {
        index = computeNciRxSumFindMax_16folds(nciRxTileBufGather, numDopplerFolds, ddmSpaceBitMask);
    }
    else
    {
        index = computeNciRxSumFindMax_8folds(nciRxTileBufGather, numDopplerFolds, ddmSpaceBitMask);
    }

    uint32_t dopplerIdx = dopplerFoldIdx + index * (numDopplerBins / numDopplerFolds);
    return dopplerIdx;
}