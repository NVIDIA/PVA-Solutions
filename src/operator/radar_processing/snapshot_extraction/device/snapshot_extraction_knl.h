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

#ifndef SNAPSHOT_EXTRACTION_KNL_H
#define SNAPSHOT_EXTRACTION_KNL_H

#include "snapshot_extraction_types.h"

#include <cupva_device.h> /* Main device-side header file */
#include <stdint.h>

/**
 * @brief Compute GSDF gather offsets for RDM data based on bestDopplerIndices
 *
 * Computes 1D byte offsets into the RDM tensor for gathering required complex values.
 * RDM layout format depends on rangeDopplerFormat:
 *   - RANGE_DOPPLER_FORMAT_RANGE_RX_DOPPLER: [numRangeBins, numRx, numDopplerBins] complex S32
 *     Each tile gathers one complex value, generates numTx * numRx offsets
 *   - RANGE_DOPPLER_FORMAT_RANGE_DOPPLER_RX: [numRangeBins, numDopplerBins, numRx] complex S32
 *     All Rx data are contiguous, each tile gathers numRx complex values, generates numTx offsets (optimized)
 * For GSDF, viewed as flattened 2D with appropriate width and height based on format.
 *
 * @param rangeIdx Range index of the detection
 * @param bestDopplerIndices Best doppler indices for each Tx [numTx]
 * @param numRx Number of Rx antennas
 * @param numTx Number of Tx antennas
 * @param numDopplerBins Total number of doppler bins
 * @param rangeDopplerFormat RDM memory layout format (RANGE_DOPPLER_FORMAT_RANGE_RX_DOPPLER or RANGE_DOPPLER_FORMAT_RANGE_DOPPLER_RX)
 * @param gatherOffsets Output: 1D offsets for GSDF gather [numTx * numRx] for RANGE_RX_DOPPLER, [numTx] for RANGE_DOPPLER_RX
 */
void computeRdmGatherOffsets(int32_t rangeIdx, const int32_t *bestDopplerIndices, int32_t numRx, int32_t numTx,
                             int32_t numDopplerBins, int32_t rangeDopplerFormat, int32_t *gatherOffsets);

/**
 * @brief Extract spatial snapshot from Range-Doppler Map
 *
 * This function gathers spatial data from params->in_rdm_buf into params->rdm_buf.
 * The gathered data is stored in [numTx, numRx] complex layout, ready for weight correction.
 * All parameters are accessed from the params structure.
 *
 * @param rangeIdx Range index of the detection
 * @param dopplerIndices Doppler indices for each Tx [numTx]
 * @param rangeDopplerFormat RDM memory layout format (RANGE_DOPPLER_FORMAT_RANGE_RX_DOPPLER or RANGE_DOPPLER_FORMAT_RANGE_DOPPLER_RX)
 * @param params Snapshot extraction parameters (contains in_rdm_buf as input, numRx, numTx, numDopplerBins, rdm_buf for gathered data output)
 */
void extractSnapshot(int32_t rangeIdx, const int32_t *dopplerIndices, int32_t rangeDopplerFormat,
                     snapshot_extraction_params_t *params);

/**
 * @brief Initialize agen configuration for weight correction
 *
 * Creates fully initialized agen objects for weights, gathered RDM, and output snapshots buffers,
 * then extracts their configurations to avoid repeated INIT_AGEN1 calls in the processing loop.
 * This follows the pattern used in doa_angle_fft (find_max_val_loc_init).
 *
 * @param params Snapshot extraction parameters (output: cfg_weights, cfg_gathered_rdm,
 *               cfg_snapshots_out, and qbits will be set)
 */
void applyWeightCorrectionAgenInit(snapshot_extraction_params_t *params);

/**
 * @brief Apply weight correction using agen config from params

 * @param params Snapshot extraction parameters (contains cfg_weights, cfg_gathered_rdm,
 *               cfg_snapshots_out, and qbits)
 */
void applyWeightCorrection(snapshot_extraction_params_t *params);

uint32_t gatherNciRxTileComputeDopplerIdx(uint32_t *nciRxTilePtr, uint32_t numTx, uint32_t numDopplerBins,
                                          uint32_t numDopplerFolds, uint32_t dopplerFoldIdx, uint32_t ddmSpaceBitMask);

#endif // SNAPSHOT_EXTRACTION_KNL_H
