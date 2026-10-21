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

#ifndef SNAPSHOT_EXTRACTION_TYPES_H
#define SNAPSHOT_EXTRACTION_TYPES_H

#include "snapshot_extraction_param.h"

#include <cupva_device.h>

/**
 * @brief Parameter structure for snapshot extraction processing
 *
 * This structure holds all parameters and buffer pointers needed for
 * snapshot extraction processing with ping-pong double buffering.
 */
typedef struct snapshot_extraction_params_s
{
    // Processing parameters
    int32_t numRx;            // Number of RX antennas
    int32_t numTx;            // Number of physical TX antennas (for snapshot output)
    int32_t numDopplerFolds;  // Number of DDM fold candidates (for disambiguation)
    int32_t numDopplerBins;   // Number of doppler bins
    int32_t numChirpsPerFold; // Number of chirps per fold
    int32_t rangeIdx;         // Currently loaded range bin index

    // VMEM buffer pointers (double-buffered for ping-pong)
    int32_t *in_rdm_buf; // Full RDM batch buffer [1, numRx, numDopplerBins] complex (via SQDF)
    int32_t *rdm_buf;    // Gathered RDM data [numTx, numRx] complex (via GSDF) or gathered manually in case of SQDF

    uint32_t *nci_buf;          // NCI batch buffer [1, numDopplerBins]
    int32_t *snapshots_out_buf; // Snapshot output buffer [1, numTx, numRx] complex

    // Shared coefficient buffers (not double-buffered)
    int32_t *weights_buf;   // DDMA or calibration weights [numTx x numRx] complex
    int32_t *ddm_space_buf; // DDM space [numTx]

    // Flag to indicate which weights to use
    int32_t use_calibration_weights; // 0=use DDMA weights, 1=use calibration weights

    // External memory pointers (DRAM addresses for SQDF)
    ExtMemPointer *rdm_ext_ptr;           // RDM source DRAM address
    ExtMemPointer *nci_ext_ptr;           // NCI source DRAM address
    ExtMemPointer *snapshots_out_ext_ptr; // Snapshot output DRAM address

    // Stride information
    int32_t rdm_range_stride; // Bytes per range bin in RDM (only needed for SQDF)
    int32_t nci_range_stride; // Bytes per range bin in NCI
    int32_t snapshot_stride;  // Bytes per snapshot

    // Agen configurations for weight correction (fully initialized, extracted from agen objects)
    AgenCFG cfg_weights;       // Agen configuration for weights buffer
    AgenCFG cfg_gathered_rdm;  // Agen configuration for gathered RDM buffer
    AgenCFG cfg_snapshots_out; // Agen configuration for output snapshots buffer
    int32_t qbits;             // Q-format bits for fixed-point arithmetic
} snapshot_extraction_params_t;

#endif // SNAPSHOT_EXTRACTION_TYPES_H
