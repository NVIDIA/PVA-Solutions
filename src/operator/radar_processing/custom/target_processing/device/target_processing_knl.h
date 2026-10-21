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
 * @file target_processing_knl.h
 * @brief Kernel function declarations for Target Processing
 *
 * This header defines the kernel functions used by the Target Processing component
 * to perform Range-Velocity (RV) decoupling and Cartesian coordinate computation.
 */

#ifndef TARGET_PROCESSING_KNL_H
#define TARGET_PROCESSING_KNL_H

#include "target_processing_param.h"

#include <cupva_device.h>

/**
 * @brief Parameter structure for target processing kernel
 */
typedef struct
{
    float rangeResolution;     // Range resolution (meters)
    float sweepTime;           // Sweep time (seconds)
    float wavelength;          // Wavelength (meters)
    int32_t numChirps;         // Number of chirps
    int32_t fftShiftedDoppler; // FFT shifted doppler flag (0 or 1)
    int32_t numTargets;        // Number of targets to process
    int32_t numDetections;     // Number of detections to process

    // Pre-computed derived parameters (computed on host to avoid VPU divisions)
    float invSweepTime; // 1.0 / sweepTime (for Doppler frequency calculation)
    float invNumChirps; // 1.0 / numChirps (for normalization)

    // Coordinate system options (for spherical-to-Cartesian conversion)
    int32_t longitudinalAxis; // 0 = X_FORWARD (X=forward, Y=lateral), 1 = Y_FORWARD (Y=forward, X=lateral)
    int32_t groundProjection; // 0 = SLANT_RANGE (use range directly), 1 = GROUND_PLANE (scale by cos(elevation))

    // External memory pointers for SQDF updates
    ExtMemPointer *targetIndexMapExtPtr;
    ExtMemPointer *targetAnglesExtPtr;
    ExtMemPointer *targetListExtPtr;

    // Buffer pointers for SQDF updates
    int32_t *targetIndexMap;
    float *targetAngles;
    float *targetList;
    bool power_output_requested;

    float *ddmDopplerOffsets;

    int32_t *detectionList;
    int32_t *detectionListRangeIndex;
    int32_t *detectionListDopplerIndex;
    int32_t *dopplerBinIndex;
    int32_t *dopplerFoldedIndex; // Folded doppler indices for NCI GSDF access
    int32_t *rangeBinIndex;
    CupvaSampler *samplerRange;
    CupvaSampler *samplerVelocity;

    // AGEN configurations for detection list transpose
    AgenCFG cfgsLoadDetectionList;
    AgenCFG cfgsStoreDetectionListRangeIndex;
    AgenCFG cfgsStoreDetectionListDopplerIndex;

    AgenCFG cfgsLoadRangeBinIndex;
    AgenCFG cfgsLoadDopplerBinIndex;

    // AGEN configurations for target processing
    AgenCFG cfgsLoadIndexMap;
    AgenCFG cfgsLoadAngles;
    AgenCFG cfgsStoreIndexMap;
    AgenCFG cfgsStoreAngles;
    AgenCFG cfgsStoreRange;
    AgenCFG cfgsStoreVelocity;
    AgenCFG cfgsStoreXYZ;
    AgenCFG cfgsStorePower;

    // Pointer to RV decoupling parameters (set in VPU main)
    rv_decoupling_params_t *rvParams;

    // NCI data for quadratic interpolation
    uint32_t *nciData;   // Pointer to NCI data in VMEM (from GSDF)
    GSDFHandler *nciHdl; // GSDF handler for NCI tiles
    AgenCFG cfgsLoadNci; // AGEN configuration for loading NCI
    int32_t nciWidth;    // Unpadded NCI width for doppler index folding

    // Folded detection list (optional, for debugging)
    int32_t *detectionListFolded;
} target_processing_params_t;

void detection_list_transpose_init(void *args);
void detection_list_transpose_kernel(void *args);

void target_processing_init(void *args0, void *args1);

/**
 * @brief Process targets to compute velocity, range, and Cartesian coordinates
 *
 * This function performs Range-Velocity decoupling and Cartesian coordinate
 * computation for radar targets. It converts detection indices and angles
 * into physical measurements.
 *
 * @param params Processing parameters
 * @param batchSize Number of targets to process in this batch
 */
void target_processing_kernel(const target_processing_params_t *params, int32_t batchSize);

/**
 * @brief Update input SQDF for loading target index map and angles
 *
 * Configures Sequence Data Flow to read target index map and angles from DRAM to VMEM
 * for the current batch iteration. Handles partial batches by adjusting tile size.
 *
 * @param targetIndexMapHdl SQDF handler for target index map input
 * @param targetAnglesHdl SQDF handler for target angles input
 * @param batchIdx Current batch iteration index
 * @param batchSize Number of valid targets in the batch
 * @param params Parameter structure with buffer and external memory pointers
 */
void updateInputSQDF(SequenceDataFlowHandler &targetIndexMapHdl, SequenceDataFlowHandler &targetAnglesHdl,
                     int32_t batchIdx, int32_t batchSize, target_processing_params_t *params);

/**
 * @brief Update output SQDF for writing computed target list
 *
 * Configures Sequence Data Flow to write computed target list from VMEM to DRAM.
 * Handles partial batches by adjusting tile size for the last batch.
 *
 * @param targetListHdl SQDF handler for target list output
 * @param batchIdx Current batch iteration index
 * @param batchSize Number of valid targets in the batch
 * @param params Parameter structure with buffer and external memory pointers
 */
void updateOutputSQDF(SequenceDataFlowHandler &targetListHdl, int32_t batchIdx, int32_t batchSize,
                      target_processing_params_t *params);

#endif // TARGET_PROCESSING_KNL_H
