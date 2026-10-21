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
 * @file doa_angle_estimation_knl.h
 * @brief VPU kernel functions for angle estimation from 2D FFT peaks
 *
 * This file contains functions for estimating target azimuth and elevation angles
 * from detected peaks in the 2D FFT power spectrum. The angle estimation pipeline:
 *
 * 1. gather_neighbours: Extract 3x3 neighborhood around detected peak
 * 2. compute_indices: Perform quadratic interpolation for sub-bin peak location
 * 3. compute_angles_init: Initialize 1D samplers for index-to-angle conversion
 * 4. compute_angles: Convert interpolated indices to angle values (degrees) and power (dB)
 *
 * These functions are separated from FFT processing to enable modular reuse and
 * potential future support for alternative angle estimation methods.
 */

#ifndef DOA_ANGLE_ESTIMATION_KNL_H
#define DOA_ANGLE_ESTIMATION_KNL_H

#include <RadarDefinitions.h>
#include <cupva_device.h> /* Main device-side header file */

/**
 * @brief Parameter structure for DOA Angle FFT processing
 *
 * Encapsulates all configuration and buffer pointers needed for
 * 2D FFT-based Direction of Arrival angle estimation.
 */
typedef struct doa_angle_fft_params_s
{
    // The FFT and peak-search agen configs live in doa_angle_fft_top.c, per input buffer and slot.
    AgenCFG cfgs_compute_index_map; // AGEN configuration for index map output
    AgenCFG cfgs_compute_angles;    // AGEN configuration for angle output
    AgenCFG cfgs_compute_power_db;  // AGEN configuration for power output in dB

    // Samplers for 1D linear interpolation (bin index -> angle conversion)
    CupvaSampler *sampler_azimuth;   // Azimuth angle interpolation sampler
    CupvaSampler *sampler_elevation; // Elevation angle interpolation sampler

    // FFT and processing parameters
    int32_t log2N_azimuth_size;   // Log2 of azimuth FFT size (e.g., 5 for 32-point FFT)
    int32_t log2N_elevation_size; // Log2 of elevation FFT size

    // Stage structure for each axis, factorized on the host and delivered by SQDF.
    FFTFactorization *factorization_azimuth;
    FFTFactorization *factorization_elevation;
    int32_t window_shift_qbits;  // Q-bits to shift after window multiplication
    int32_t twiddle_shift_qbits; // Q-bits to shift after twiddle multiplication

    // Buffer pointers
    float *power_spectrum; // Power spectrum buffer (output of 2D FFT)
    int32_t *angle_bins;   // Pre-computed angle bins LUT (azimuth + elevation)
    float epsilon;         // Small value to avoid division by zero

    // Input/output data buffers (VMEM)
    float *dst_target_angles_buf;      // Output target angles [azimuth, elevation, power]
    int32_t *dst_target_index_map_buf; // Output target index map [azimuth_idx, elevation_idx]
    int32_t *dst_target_idx;           // Running count of output targets

    // External memory pointers (DRAM addresses for SQDF)
    ExtMemPointer *src_snapshots_ext_ptr;        // Source DRAM address for snapshots
    ExtMemPointer *dst_target_angles_ext_ptr;    // Destination DRAM address for angles
    ExtMemPointer *dst_target_index_map_ext_ptr; // Destination DRAM address for index map
} doa_angle_fft_params_t;

/**
 * @brief Initialize AGEN for finding maximum value location in power spectrum
 *
 * Configures address generator to scan the 2D power spectrum and
 * find the location of the peak (maximum) value.
 *
 * @param power_spectrum Pointer to 2D power spectrum buffer
 * @param log2N Log2 of FFT size (height dimension)
 * @param cfgs Array to store AGEN configuration
 * @param line_width Width of power spectrum (azimuth dimension)
 * @param pad Padding for memory alignment
 */
void find_max_val_loc_init(float *power_spectrum, int32_t log2N, AgenCFG *cfgs, int32_t line_width, int32_t pad);

/**
 * @brief Find the location of the maximum value in the power spectrum
 *
 * Reports the first position, in row-major scan order, whose power equals `value`.
 *
 * @param cfgs AGEN configuration from find_max_val_loc_init
 * @param n FFT size (height dimension)
 * @param iter Number of iterations (width / vector_width)
 * @param value Maximum value to search for
 * @param log2_width Log2 of the azimuth bin count, to split the position without a divide
 * @param p_val_loc_x [out] X coordinate (azimuth bin) of the peak, element 0 only
 * @param p_val_loc_y [out] Y coordinate (elevation bin) of the peak, element 0 only
 */
void find_max_val_loc(AgenCFG *cfgs, int32_t n, int32_t iter, float value, int32_t log2_width,
                      unsigned short *p_val_loc_x, unsigned short *p_val_loc_y);

/**
 * @brief Gather 3x3 neighborhood values around detected peak for interpolation
 *
 * Extracts the power spectrum values at and around the peak location
 * (3 neighbors in azimuth, 3 neighbors in elevation) needed for
 * quadratic interpolation to find sub-bin peak location.
 *
 * Uses circular indexing to handle edge cases and wraparound correctly.
 *
 * @param args Pointer to doa_angle_fft_params_t structure
 * @param peakAzimuthIndex Detected peak azimuth bin index
 * @param peakElevationIndex Detected peak elevation bin index
 * @param pY1 [out] Left/lower neighbor values (azimuth + elevation)
 * @param pY2 [out] Center peak values (azimuth + elevation)
 * @param pY3 [out] Right/upper neighbor values (azimuth + elevation)
 * @param pshiftedIdx [out] Adjusted indices for interpolation (azimuth + elevation)
 */
void gather_neighbours(void *args, unsigned short peakAzimuthIndex, unsigned short peakElevationIndex, float *pY1,
                       float *pY2, float *pY3, int32_t *pshiftedIdx);

/**
 * @brief Compute interpolated peak indices using quadratic interpolation
 *
 * Uses the 3x3 neighborhood values to perform quadratic interpolation
 * separately for azimuth and elevation dimensions, producing sub-bin
 * accurate peak locations.
 *
 * Interpolation formula: x_peak = x_center + (y1 - y3) / (2 * (y1 - 2*y2 + y3))
 *
 * @param args Pointer to doa_angle_fft_params_t structure
 * @param pY1 Left/lower neighbor values (azimuth + elevation)
 * @param pY2 Center peak values (azimuth + elevation)
 * @param pY3 Right/upper neighbor values (azimuth + elevation)
 * @param pshiftedIdx Adjusted indices for interpolation (azimuth + elevation)
 * @param interpAzimuthIndex [out] Interpolated azimuth indices (Q16 fixed-point, batch of 8)
 * @param interpElevationIndex [out] Interpolated elevation indices (Q16 fixed-point, batch of 8)
 */
void compute_indices(void *args, float *pY1, float *pY2, float *pY3, int32_t *pshiftedIdx, int32_t *interpAzimuthIndex,
                     int32_t *interpElevationIndex);

/**
 * @brief Initialize 1D samplers for angle interpolation
 *
 * Sets up CupvaSampler objects that perform 1D linear interpolation
 * to convert from fractional bin indices to actual angle values in degrees.
 * Uses pre-computed angle bin lookup tables.
 *
 * This function also initializes AGEN configurations for writing output
 * angles and index maps.
 *
 * @param args0 Pointer to doa_angle_fft_params_t structure (buffer 0)
 * @param args1 Pointer to doa_angle_fft_params_t structure (buffer 1)
 * @param interpAzimuthIndex Dummy azimuth indices for sampler initialization
 * @param interpElevationIndex Dummy elevation indices for sampler initialization
 * @param peakAzimuth Dummy output for azimuth angles
 * @param peakElevation Dummy output for elevation angles
 */
void compute_angles_init(void *args0, void *args1, int32_t *interpAzimuthIndex, int32_t *interpElevationIndex,
                         int32_t *peakAzimuth, int32_t *peakElevation);

/**
 * @brief Convert interpolated indices to angle values and write output
 *
 * Uses pre-initialized 1D samplers to convert fractional bin indices
 * to actual azimuth and elevation angles in degrees. Also converts
 * power values from Q8 fixed-point to dB scale (floating-point).
 *
 * Output format per target: [azimuth_degrees, elevation_degrees, power_dB]
 *
 * @param args Pointer to doa_angle_fft_params_t structure
 * @param interpAzimuthIndex Interpolated azimuth indices (Q16 fixed-point, batch of 8)
 * @param interpElevationIndex Interpolated elevation indices (Q16 fixed-point, batch of 8)
 * @param max_val Peak power values (Q8 fixed-point, batch of 8)
 * @param peakAzimuth [out] Azimuth angles in degrees (batch of 8)
 * @param peakElevation [out] Elevation angles in degrees (batch of 8)
 * @param batch_size Number of valid entries in the batch (1-8)
 */
void compute_angles(void *args, int32_t *interpAzimuthIndex, int32_t *interpElevationIndex, float *max_val,
                    int32_t *peakAzimuth, int32_t *peakElevation, int32_t batch_size);

/**
 * @brief Load the next group of detection snapshots into an input slot buffer
 *
 * @param srcSnapshots SQDF handler for input data
 * @param iter Index of the first detection in the group
 * @param dst Input group buffer; detection i lands in slot i
 * @param params DRAM address of the snapshots
 * @param count Detections in this group, at most `groups`
 * @param groups Detections in a full group (G)
 * @param tileWidthBytes Snapshot row width, for the narrowed tile of a short last group
 */
void updateInputSQDF(SequenceDataFlowHandler &srcSnapshots, int32_t iter, int32_t *dst, doa_angle_fft_params_t *params,
                     int32_t count, int32_t groups, int32_t tileWidthBytes);

/**
 * @brief Update output SQDF for computed angles and indices
 *
 * Configures Sequence Data Flow to write computed angles and index maps from VMEM to DRAM.
 * Handles partial batches by adjusting tile size for the last batch.
 *
 * @param dstTargetAngles SQDF handler for angle output
 * @param dstTargetIndexMap SQDF handler for index map output
 * @param iter Current batch iteration
 * @param offset Offset within the batch
 * @param isLastWidth Number of valid targets in the batch
 * @param params Parameter structure with buffer pointers
 */
void updateOutputSQDF(SequenceDataFlowHandler &dstTargetAngles, SequenceDataFlowHandler &dstTargetIndexMap,
                      int32_t iter, int32_t offset, int32_t isLastWidth, doa_angle_fft_params_t *params);

#endif /* DOA_ANGLE_ESTIMATION_KNL_H */
