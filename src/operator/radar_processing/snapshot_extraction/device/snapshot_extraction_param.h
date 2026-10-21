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

#ifndef SNAPSHOT_EXTRACTION_PARAM_H
#define SNAPSHOT_EXTRACTION_PARAM_H

// Complex element has 2 components (real, imaginary)
#define COMPLEX_ELEMENT_NUM (2)

// Range-Doppler map format definitions (matching PVAFFTLayout enum)
#define RANGE_DOPPLER_FORMAT_RANGE_RX_DOPPLER \
    (1) // [numRangeBins, numRx, numDopplerBins] = PVA_FFT_LAYOUT_RANGE_RX_DOPPLER
#define RANGE_DOPPLER_FORMAT_RANGE_DOPPLER_RX \
    (2) // [numRangeBins, numDopplerBins, numRx] = PVA_FFT_LAYOUT_RANGE_DOPPLER_RX

// Range batch size for processing (set to 1 for processing one range bin at a time)
#define RANGE_BATCH_SIZE (1)

// Maximum detections that can be stored in VMEM
#define MAX_DETECTIONS (8192)

// Maximum number of Tx antennas (physical antennas for snapshot output)
#define MAX_NUM_TX (8)

// Maximum number of DDM folds (for DDM disambiguation - can be larger than numTx)
#define MAX_DDM_FOLDS (16)

// Maximum number of Rx antennas
#define MAX_NUM_RX (8)

// Maximum number of virtual antennas
#define MAX_NUM_VIRTUAL_ANTENNAS (MAX_NUM_TX * MAX_NUM_RX)

// Maximum number of doppler bins
#define MAX_NUM_DOPPLER_BINS (512)

// Maximum number of range bins in a batch
#define MAX_RANGE_BINS_PER_BATCH (RANGE_BATCH_SIZE)

// Fixed-point Q-format bits
#define DDMA_WEIGHTS_QBITS (16)
#define DDMA_WEIGHTS_SCALE (1 << DDMA_WEIGHTS_QBITS)

#define CALIBRATION_WEIGHTS_QBITS (28)
#define CALIBRATION_WEIGHTS_SCALE (1LL << CALIBRATION_WEIGHTS_QBITS)

// Buffer sizes
#define WEIGHTS_BUF_SIZE (MAX_NUM_TX * MAX_NUM_RX * COMPLEX_ELEMENT_NUM)
#define DDM_DOPPLER_OFFSETS_BUF_SIZE (MAX_NUM_TX)
#define DDM_SPACE_BUF_SIZE (MAX_NUM_TX)

// In case of SQDF-based RDM loading: Load entire range row [rangeBatchSize, numRx, numDopplerBins] complex
// GSDF-based RDM gathering: Only gather required data [numTx, numRx] complex. This is same size as snapshot output buffer since we gather exactly what we need
#define RDM_BUF_SIZE (MAX_RANGE_BINS_PER_BATCH * MAX_NUM_RX * MAX_NUM_DOPPLER_BINS * COMPLEX_ELEMENT_NUM)

// NCI batch buffer size: [rangeBatchSize, numDopplerBins]
#define NCI_BATCH_BUF_SIZE (MAX_RANGE_BINS_PER_BATCH * MAX_NUM_DOPPLER_BINS)

// Detection list buffer: [maxDetections, 2]
#define DETECTION_LIST_BUF_SIZE (MAX_DETECTIONS * 2)

// Snapshot buffer: [1, numTx, numRx] complex
#define SNAPSHOTS_BUF_SIZE (1 * MAX_NUM_TX * MAX_NUM_RX * COMPLEX_ELEMENT_NUM)

// Maximum tiles per GSDF trigger (hardware limitation)
#define MAX_GSDF_TILES_PER_TRIGGER (32)

#endif // SNAPSHOT_EXTRACTION_PARAM_H
