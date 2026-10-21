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
 * @file doa_angle_estimation_knl.c
 * @brief VPU kernel implementation for angle estimation from 2D FFT peaks
 *
 * This file implements the angle estimation pipeline that converts detected
 * peaks in the 2D FFT power spectrum into physical azimuth and elevation angles.
 *
 * Processing Pipeline:
 * 1. Peak Detection → 2D bin indices (integer)
 * 2. Gather Neighbors → Extract 3x3 neighborhood values
 * 3. Quadratic Interpolation → Sub-bin peak location (Q16 fixed-point)
 * 4. Index-to-Angle Conversion → Physical angles in degrees (float32)
 * 5. Power Conversion → dB scale (float32)
 *
 * Key Features:
 * - Sub-bin accuracy via quadratic interpolation
 * - Circular indexing for wraparound handling
 * - Vectorized processing (8 targets per batch)
 * - Hardware-accelerated 1D sampling for angle lookup
 */

#include "doa_angle_estimation_knl.h"

#include "doa_angle_fft_param.h"

void find_max_val_loc_init(float *power_spectrum, int32_t log2N, AgenCFG *cfgs, int32_t line_width, int32_t pad)
{
    // Initialize the kernel test data
    const int32_t vecw = chess_elementsof(dvintx);
    uint32_t height    = 1 << log2N;

    AgenWrapper wrapper;

    agen in      = init(power_spectrum);
    wrapper.size = sizeof(float);
    wrapper.n1   = line_width / vecw;
    wrapper.n2   = height;
    wrapper.s1   = vecw;
    wrapper.s2   = line_width;
    INIT_AGEN2(in, wrapper);

    cfgs[0] = extract_agen_cfg(in);
}

/* The smallest scan-order position p whose power equals `value`, as (p % width, p / width). A max-reduce
 * over negated positions picks the earliest match; `value` came from the spectrum, so one exists. */
void find_max_val_loc(AgenCFG *cfgs, int32_t n, int32_t iter, float value, int32_t log2_width,
                      unsigned short *p_val_loc_x, unsigned short *p_val_loc_y)
{
    agen a_src                   = init_agen_from_cfg(cfgs[0]);
    unsigned short p_lane_id[32] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

    const int32_t vecw    = chess_elementsof(dvintx);
    const int32_t niter_w = iter >> 1;
    const int32_t niter   = niter_w * n;
    const int32_t width   = niter_w * vecw; /* azimuth bins per row */

    /* Loses every max against a real negated position, which bottom out at -(width * n). */
    const vshortx sentinel = replicateh(-30000);
    vshortx neg_pos        = replicateh(0) - zero_extend(*(vushort *)p_lane_id);

    /* Four independent accumulators: a single one would serialise the loop on the max chain.
     * niter is (az / 16) * el, a multiple of four for every supported shape. */
    vshortx acc0 = sentinel;
    vshortx acc1 = sentinel;
    vshortx acc2 = sentinel;
    vshortx acc3 = sentinel;

    chess_separator_scheduler();

    for (int32_t i = 0; i < niter; i += 4) chess_prepare_for_pipelining
    {
        dvfloatx d0 = dvfloat_load(a_src);
        dvfloatx d1 = dvfloat_load(a_src);
        dvfloatx d2 = dvfloat_load(a_src);
        dvfloatx d3 = dvfloat_load(a_src);

        vshortx f0 = vdemote((dvintx)(d0 == value));
        vshortx p0 = neg_pos;
        neg_pos    = neg_pos - (short)vecw;
        vshortx f1 = vdemote((dvintx)(d1 == value));
        vshortx p1 = neg_pos;
        neg_pos    = neg_pos - (short)vecw;
        vshortx f2 = vdemote((dvintx)(d2 == value));
        vshortx p2 = neg_pos;
        neg_pos    = neg_pos - (short)vecw;
        vshortx f3 = vdemote((dvintx)(d3 == value));
        vshortx p3 = neg_pos;
        neg_pos    = neg_pos - (short)vecw;

        acc0 = vmax(acc0, vmux(f0, p0, sentinel));
        acc1 = vmax(acc1, vmux(f1, p1, sentinel));
        acc2 = vmax(acc2, vmux(f2, p2, sentinel));
        acc3 = vmax(acc3, vmux(f3, p3, sentinel));
    }

    int neg_idx = 0;
    int lane    = 0;
    vmaxrid_s(vmax(vmax(acc0, acc1), vmax(acc2, acc3)), neg_idx, lane);

    // width (the azimuth bin count) is a power of two.
    const int32_t idx = -neg_idx;
    p_val_loc_x[0]    = (unsigned short)(idx & (width - 1));
    p_val_loc_y[0]    = (unsigned short)(idx >> log2_width);
}

/**
 * @brief Quadratic interpolation to find sub-bin peak location
 *
 * Given three neighboring values (y1, y2, y3) where y2 is the peak,
 * fits a parabola and finds the peak offset from y2's position.
 *
 * Formula: delta = (y1 - y3) / (2 * (y1 - 2*y2 + y3))
 *
 * @param y1 Left/lower neighbor value
 * @param y2 Center peak value
 * @param y3 Right/upper neighbor value
 * @param delta [out] Interpolated offset in Q16 fixed-point format (range: -0.5 to 0.5)
 */
static inline void quadraticInterpolation(dvfloatx &y1, dvfloatx &y2, dvfloatx &y3, dvintx &delta)
{
    // Use Q16 format for interpolated indices (16 fractional bits for high precision)
    const int32_t INTERP_QBITS = ANGLE_QBITS; // Use same precision as angles
    const float INTERP_SCALE   = ANGLE_SCALE;
    dvfloatx zero_f;
    zero_f.lo = replicatef(0.0f);
    zero_f.hi = replicatef(0.0f);

    //dvfloatx y1_f = dvintx_dvfp(y1);
    //dvfloatx y2_f = dvintx_dvfp(y2);
    //dvfloatx y3_f = dvintx_dvfp(y3);

    dvfloatx y1_f = y1;
    dvfloatx y2_f = y2;
    dvfloatx y3_f = y3;

    dvfloatx vNumerator_f = (y1_f - y3_f);
    dvfloatx vDen_f       = (y1_f - (y2_f * 2.0f) + y3_f) * 2.0f;
    dvfloatx vDenRcp_f    = dvfrcp(vDen_f);

    // delta = num / den = num * rcp(den)
    // Use vmux to avoid NaN when den == 0: select 0.0 where den is zero
    dvintx denZeroMask = (vDen_f == zero_f);
    dvfloatx delta_f   = dvmux(denZeroMask, zero_f, (vNumerator_f * vDenRcp_f) * INTERP_SCALE);
    delta              = dvfp_dvintx_round(delta_f);
}

/**
 * @brief Gather 3x3 neighborhood values around detected peak for interpolation
 *
 * Extracts the power spectrum values at and around the peak location
 * (3 neighbors in azimuth, 3 neighbors in elevation) needed for
 * quadratic interpolation to find sub-bin peak location.
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
                       float *pY2, float *pY3, int32_t *pshiftedIdx)
{
    doa_angle_fft_params_t *param = (doa_angle_fft_params_t *)args;
    float *power_spectrum         = (float *)param->power_spectrum;
    int32_t numElevBins           = (1 << param->log2N_elevation_size);
    int32_t numAzBins             = (1 << param->log2N_azimuth_size);

    // Convert unshifted peak indices to shifted coordinates early
    int32_t azIdx   = peakAzimuthIndex;
    int32_t elevIdx = peakElevationIndex;

    // Shift to center DC at the center of the spectrum (FFT shift)
    int32_t shiftedAzIdx   = (azIdx + numAzBins / 2) % numAzBins;
    int32_t shiftedElevIdx = (elevIdx + numElevBins / 2) % numElevBins;

    // Quadratic interpolation for azimuth (horizontal neighbors) using circular indexing
    // Always safe to access neighbors with circular indexing
    {
        // Use circular indexing for neighbors (handles wraparound correctly)
        int32_t az_left   = (azIdx - 1 + numAzBins) % numAzBins; // Left neighbor
        int32_t az_center = azIdx;                               // Peak
        int32_t az_right  = (azIdx + 1) % numAzBins;             // Right neighbor

        // Extract power values from 2D power spectrum (row-major layout)
        float y1 = *(power_spectrum + (elevIdx * numAzBins) + az_left);   // Left neighbor
        float y2 = *(power_spectrum + (elevIdx * numAzBins) + az_center); // Peak
        float y3 = *(power_spectrum + (elevIdx * numAzBins) + az_right);  // Right neighbor

        // Store azimuth neighbors and shifted index (first OUTPUT_BATCH_SIZE elements)
        pY1[0]         = y1;
        pY2[0]         = y2;
        pY3[0]         = y3;
        pshiftedIdx[0] = shiftedAzIdx;
    }

    // Quadratic interpolation for elevation (vertical neighbors) using circular indexing
    // Always safe to access neighbors with circular indexing
    {
        // Use circular indexing for neighbors (handles wraparound correctly)
        int32_t elev_lower  = (elevIdx - 1 + numElevBins) % numElevBins; // Lower neighbor
        int32_t elev_center = elevIdx;                                   // Peak
        int32_t elev_upper  = (elevIdx + 1) % numElevBins;               // Upper neighbor

        // Extract power values from 2D power spectrum (row-major layout)
        float y1 = *(power_spectrum + (elev_lower * numAzBins) + azIdx);  // Lower neighbor
        float y2 = *(power_spectrum + (elev_center * numAzBins) + azIdx); // Peak
        float y3 = *(power_spectrum + (elev_upper * numAzBins) + azIdx);  // Upper neighbor

        // Store elevation neighbors and shifted index (second OUTPUT_BATCH_SIZE elements)
        pY1[OUTPUT_BATCH_SIZE]         = y1;
        pY2[OUTPUT_BATCH_SIZE]         = y2;
        pY3[OUTPUT_BATCH_SIZE]         = y3;
        pshiftedIdx[OUTPUT_BATCH_SIZE] = shiftedElevIdx;
    }

    return;
}

/**
 * @brief Compute interpolated peak indices using quadratic interpolation
 *
 * Uses the 3x3 neighborhood values to perform quadratic interpolation
 * separately for azimuth and elevation dimensions, producing sub-bin
 * accurate peak locations.
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
                     int32_t *interpElevationIndex)
{
    doa_angle_fft_params_t *param = (doa_angle_fft_params_t *)args;
    int32_t numElevBins           = (1 << param->log2N_elevation_size);
    int32_t numAzBins             = (1 << param->log2N_azimuth_size);

    // Use Q16 format for interpolated indices (16 fractional bits for high precision)
    const int32_t INTERP_QBITS = ANGLE_QBITS; // Use same precision as angles
    const int32_t INTERP_SCALE = ANGLE_SCALE;

    // Setup range limits for clamping (prevent out-of-bounds indices)
    dvintx dvInterpRangeMax;
    dvInterpRangeMax.lo = replicatew((numAzBins - 1) * INTERP_SCALE);
    dvInterpRangeMax.hi = replicatew((numElevBins - 1) * INTERP_SCALE);

    dvintx dvDelta;
    dvintx dvInterpIdx;
    dvintx dvInterpAzIdx;
    dvintx dvInterpElevIdx;

    // Load input arrays as double-width vectors (azimuth in .lo, elevation in .hi)
    dvintx dvShiftedIdx = sign_extend(*(dvint *)pshiftedIdx);
    dvfloatx dvY1       = sign_extend(*(dvfloat *)pY1);
    dvfloatx dvY2       = sign_extend(*(dvfloat *)pY2);
    dvfloatx dvY3       = sign_extend(*(dvfloat *)pY3);

    dvintx dvInterpScale;
    dvInterpScale.lo = replicatew(INTERP_SCALE);
    dvInterpScale.hi = replicatew(INTERP_SCALE);

    // Perform quadratic interpolation to find sub-bin offset
    quadraticInterpolation(dvY1, dvY2, dvY3, dvDelta);

    // Clamp delta to reasonable range (±4 pixels in Q16)
    dvDelta = dvmax(dvmin(dvDelta, 4 * INTERP_SCALE), -4 * INTERP_SCALE);

    // Convert integer index to Q16 fixed-point: interpIndex = shiftedIdx * INTERP_SCALE + delta
    dvInterpIdx = dvmulwhw(dvInterpScale, dvShiftedIdx);
    dvInterpIdx = dvInterpIdx + dvDelta;

    // Clamp to valid range [0, (numBins-1) * INTERP_SCALE]
    dvInterpIdx = dvmax(dvmin(dvInterpIdx, dvInterpRangeMax), 0);

    // Store results (azimuth and elevation indices)
    *(vint chess_storage(RAM_Bb) *)(interpAzimuthIndex)   = extract(dvInterpIdx.lo);
    *(vint chess_storage(RAM_Bb) *)(interpElevationIndex) = extract(dvInterpIdx.hi);

    // Return indices already in shifted coordinates (deltas added directly)
    return;
}

/**
 * @brief Initialize 1D samplers for angle interpolation
 *
 * Sets up CupvaSampler objects that perform 1D linear interpolation
 * to convert from fractional bin indices to actual angle values in degrees.
 * Uses pre-computed angle bin lookup tables.
 *
 * @param args0 Pointer to doa_angle_fft_params_t structure (buffer 0)
 * @param args1 Pointer to doa_angle_fft_params_t structure (buffer 1)
 * @param interpAzimuthIndex Dummy azimuth indices for sampler initialization
 * @param interpElevationIndex Dummy elevation indices for sampler initialization
 * @param peakAzimuth Dummy output for azimuth angles
 * @param peakElevation Dummy output for elevation angles
 */
void compute_angles_init(void *args0, void *args1, int32_t *interpAzimuthIndex, int32_t *interpElevationIndex,
                         int32_t *peakAzimuth, int32_t *peakElevation)
{
    doa_angle_fft_params_t *param0 = (doa_angle_fft_params_t *)args0;
    doa_angle_fft_params_t *param1 = (doa_angle_fft_params_t *)args1;
    int32_t *angleBins             = param0->angle_bins;
    int32_t numAzBins              = (1 << param0->log2N_azimuth_size);
    int32_t numElevBins            = (1 << param0->log2N_elevation_size);

    // Initialize samplers for buffer 0
    {
        CupvaSampler *az_sampler   = param0->sampler_azimuth;
        CupvaSampler *elev_sampler = param0->sampler_elevation;

        // Setup azimuth sampler: maps azimuth bin indices to angles (degrees)
        CupvaSamplerInput1D const az_input = {.data           = angleBins,
                                              .type           = SAMPLER_INPUT_TYPE_S32,
                                              .length         = (uint32_t)(numAzBins),
                                              .outOfRangeMode = SAMPLER_OUT_OF_RANGE_CONSTANT};

        CupvaSamplerIndices1D const az_indices = {.data               = interpAzimuthIndex,
                                                  .type               = SAMPLER_INDEX_TYPE_U32,
                                                  .width              = (uint32_t)(OUTPUT_BATCH_SIZE),
                                                  .height             = 1,
                                                  .fractionalBits     = ANGLE_QBITS,
                                                  .fractionalHandling = SAMPLER_FRAC_HANDLING_INTERPOLATE};

        CupvaSamplerOutput const az_output = {.data      = peakAzimuth,
                                              .pitch     = OUTPUT_BATCH_SIZE,
                                              .transMode = TRANS_MODE_NONE};

        cupvaSamplerSetup(az_sampler, &az_input, &az_indices, &az_output);

        // Setup elevation sampler: maps elevation bin indices to angles (degrees)
        CupvaSamplerInput1D const elev_input = {.data           = angleBins + numAzBins,
                                                .type           = SAMPLER_INPUT_TYPE_S32,
                                                .length         = (uint32_t)(numElevBins),
                                                .outOfRangeMode = SAMPLER_OUT_OF_RANGE_CONSTANT};

        CupvaSamplerIndices1D const elev_indices = {.data               = interpElevationIndex,
                                                    .type               = SAMPLER_INDEX_TYPE_U32,
                                                    .width              = (uint32_t)(OUTPUT_BATCH_SIZE),
                                                    .height             = 1,
                                                    .fractionalBits     = ANGLE_QBITS,
                                                    .fractionalHandling = SAMPLER_FRAC_HANDLING_INTERPOLATE};

        CupvaSamplerOutput const elev_output = {.data      = peakElevation,
                                                .pitch     = OUTPUT_BATCH_SIZE,
                                                .transMode = TRANS_MODE_NONE};
        cupvaSamplerSetup(elev_sampler, &elev_input, &elev_indices, &elev_output);
    }

    // Initialize AGEN configurations for output angles (buffer 0 and buffer 1)
    {
        AgenWrapper wrapper;

        // Buffer 0: Setup AGEN for writing angles
        agen out0    = init(param0->dst_target_angles_buf);
        wrapper.size = sizeof(float);
        wrapper.n1   = OUT_TARGET_ANGLE_TILE_HEIGHT - 1; // Skip the last row i.e. power
        wrapper.s1   = OUT_TARGET_ANGLE_TILE_PITCH;
        INIT_AGEN1(out0, wrapper);

        AgenCFG *cfgs0 = &param0->cfgs_compute_angles;
        cfgs0[0]       = extract_agen_cfg(out0);

        // Buffer 1: Setup AGEN for writing angles
        agen out1    = init(param1->dst_target_angles_buf);
        wrapper.size = sizeof(float);
        wrapper.n1   = OUT_TARGET_ANGLE_TILE_HEIGHT - 1; // Skip the last row i.e. power
        wrapper.s1   = OUT_TARGET_ANGLE_TILE_PITCH;
        INIT_AGEN1(out1, wrapper);

        AgenCFG *cfgs1 = &param1->cfgs_compute_angles;
        cfgs1[0]       = extract_agen_cfg(out1);
    }

    // Initialize AGEN configurations for output peak power in dB (buffer 0 and buffer 1)
    {
        AgenWrapper wrapper;

        // Buffer 0: Setup AGEN for writing peak power in dB
        agen out0 =
            init(param0->dst_target_angles_buf + (OUT_TARGET_ANGLE_TILE_HEIGHT - 1) * OUT_TARGET_ANGLE_TILE_PITCH);
        wrapper.size = sizeof(float);
        wrapper.n1   = 1;
        wrapper.s1   = OUT_TARGET_ANGLE_TILE_PITCH;
        INIT_AGEN1(out0, wrapper);

        AgenCFG *cfgs0 = &param0->cfgs_compute_power_db;
        cfgs0[0]       = extract_agen_cfg(out0);

        // Buffer 1: Setup AGEN for writing peak power in dB
        agen out1 =
            init(param1->dst_target_angles_buf + (OUT_TARGET_ANGLE_TILE_HEIGHT - 1) * OUT_TARGET_ANGLE_TILE_PITCH);
        wrapper.size = sizeof(float);
        wrapper.n1   = 1;
        wrapper.s1   = OUT_TARGET_ANGLE_TILE_PITCH;
        INIT_AGEN1(out1, wrapper);

        AgenCFG *cfgs1 = &param1->cfgs_compute_power_db;
        cfgs1[0]       = extract_agen_cfg(out1);
    }

    // Initialize AGEN configurations for output index map (buffer 0 and buffer 1)
    {
        AgenWrapper wrapper;

        // Buffer 0: Setup AGEN for writing index map
        agen out0    = init(param0->dst_target_index_map_buf);
        wrapper.size = sizeof(int32_t);
        wrapper.n1   = 1;
        wrapper.s1   = OUT_TARGET_INDEX_MAP_TILE_PITCH;
        INIT_AGEN1(out0, wrapper);

        AgenCFG *cfgs0 = &param0->cfgs_compute_index_map;
        cfgs0[0]       = extract_agen_cfg(out0);

        // Buffer 1: Setup AGEN for writing index map
        agen out1    = init(param1->dst_target_index_map_buf);
        wrapper.size = sizeof(int32_t);
        wrapper.n1   = 1;
        wrapper.s1   = OUT_TARGET_INDEX_MAP_TILE_PITCH;
        INIT_AGEN1(out1, wrapper);

        AgenCFG *cfgs1 = &param1->cfgs_compute_index_map;
        cfgs1[0]       = extract_agen_cfg(out1);
    }

    return;
}

/**
 * @brief Convert interpolated indices to angle values and write output
 *
 * Uses pre-initialized 1D samplers to convert fractional bin indices
 * to actual azimuth and elevation angles in degrees. Also converts
 * power values to dB scale and writes results to output buffers.
 *
 * @param args Pointer to doa_angle_fft_params_t structure
 * @param interpAzimuthIndex Interpolated azimuth indices (Q16 fixed-point, batch of 8)
 * @param interpElevationIndex Interpolated elevation indices (Q16 fixed-point, batch of 8)
 * @param max_val Peak power values (float, batch of 8)
 * @param peakAzimuth [out] Azimuth angles in degrees (batch of 8)
 * @param peakElevation [out] Elevation angles in degrees (batch of 8)
 * @param batch_size Number of valid entries in the batch (1-8)
 */
void compute_angles(void *args, int32_t *interpAzimuthIndex, int32_t *interpElevationIndex, float *max_val,
                    int32_t *peakAzimuth, int32_t *peakElevation, int32_t batch_size)
{
    doa_angle_fft_params_t *param = (doa_angle_fft_params_t *)args;

    CupvaSampler *az_sampler   = param->sampler_azimuth;
    CupvaSampler *elev_sampler = param->sampler_elevation;

    // Start azimuth sampler (asynchronous)
    cupvaSamplerStart(az_sampler);

    // Conversion factors for fixed-point to float
    const float INV_POWER_SCALE = 1.0f / (((float)(1LL << POWER_QBITS)) * ((float)(1LL << POWER_QBITS)));
    const float INV_ANGLE_SCALE = 1.0f / (float)(ANGLE_SCALE);

    // Initialize AGENs for output
    agen_C out_angles   = init_agen_C_from_cfg(param->cfgs_compute_angles);
    agen_C out_power_db = init_agen_C_from_cfg(param->cfgs_compute_power_db);
    agen_C out_idx_map  = init_agen_C_from_cfg(param->cfgs_compute_index_map);

    // Generate target indices [0, 1, 2, 3, 4, 5, 6, 7] + base offset
    int32_t target_idx[OUTPUT_BATCH_SIZE] = {0, 1, 2, 3, 4, 5, 6, 7};
    vintx vtarget_idx                     = sign_extend(*(vint *)target_idx);

    // Convert power values from Q40 float to float, then to dB scale: 10 * log10(power)
    vfloatx vmax_val_float     = sign_extend(*(vfloat *)max_val);
    vfloatx vmax_power_db      = vmax_val_float * INV_POWER_SCALE;
    vfloatx vmax_power_db_log2 = vflog2(vmax_power_db + param->epsilon);
    vfloatx v10                = replicatef(10.0f);
    vfloatx vlog2_recip_10     = vfrcp(vflog2(v10));

    // Add global target index offset and write index map
    vtarget_idx                  = vtarget_idx + replicatew(*param->dst_target_idx);
    vfloatx vpeak_power_db_float = v10 * vmax_power_db_log2 * vlog2_recip_10;
    *param->dst_target_idx       = *param->dst_target_idx + batch_size;
    vstore(vtarget_idx, out_idx_map);
    // Write power in dB to output
    vstore(vpeak_power_db_float, out_power_db);

    // Wait for azimuth sampler to complete
    cupvaSamplerWait();

    // Start elevation sampler (asynchronous)
    cupvaSamplerStart(elev_sampler);
    chess_separator_scheduler();

    // Convert azimuth from Q16 fixed-point to float degrees and write to output
    vintx vpeak_azimuth               = sign_extend(*(vint *)peakAzimuth);
    vfloatx vpeak_azimuth_float       = vintx_vfp(vpeak_azimuth);
    vfloatx vpeak_azimuth_float_angle = vpeak_azimuth_float * INV_ANGLE_SCALE;
    vstore(vpeak_azimuth_float_angle, out_angles);

    // Wait for elevation sampler to complete
    cupvaSamplerWait();
    chess_separator_scheduler();

    // Convert elevation from Q16 fixed-point to float degrees and write to output
    vintx vpeak_elevation               = sign_extend(*(vint *)peakElevation);
    vfloatx vpeak_elevation_float       = vintx_vfp(vpeak_elevation);
    vfloatx vpeak_elevation_float_angle = vpeak_elevation_float * INV_ANGLE_SCALE;
    vstore(vpeak_elevation_float_angle, out_angles);

    return;
}

/* Loads count consecutive detections into consecutive slots in one transfer. A short final group shrinks the tile height. */
void updateInputSQDF(SequenceDataFlowHandler &srcSnapshots, int32_t iter, int32_t *dst, doa_angle_fft_params_t *params,
                     int32_t count, int32_t groups, int32_t tileWidthBytes)
{
    ExtMemPointer *snapExtAddr = params->src_snapshots_ext_ptr;

    int64_t snapBase   = snapExtAddr->base + snapExtAddr->offset;
    int32_t snapOffset = iter * 8 * 8 * 2 * sizeof(int32_t);

    cupvaSQDFUpdateAddr(srcSnapshots, 0, snapBase + snapOffset, MEMTYPE_DRAM, cupvaGetVmemAddress((void *)dst),
                        MEMTYPE_VMEM);

    if (count != groups)
    {
        cupvaSQDFUpdateTileSize(srcSnapshots, 0, tileWidthBytes, IN_SLOT_HEIGHT * count);
    }

    cupvaSQDFFlushAndTrig(srcSnapshots);
}

void updateOutputSQDF(SequenceDataFlowHandler &dstTargetAngles, SequenceDataFlowHandler &dstTargetIndexMap,
                      int32_t iter, int32_t offset, int32_t isLastWidth, doa_angle_fft_params_t *params)
{
    ExtMemPointer *dstTargetAnglesExtAddr   = params->dst_target_angles_ext_ptr;
    ExtMemPointer *dstTargetIndexMapExtAddr = params->dst_target_index_map_ext_ptr;

    int64_t dstTargetAnglesBase     = dstTargetAnglesExtAddr->base + dstTargetAnglesExtAddr->offset;
    int64_t dstTargetIndexMapBase   = dstTargetIndexMapExtAddr->base + dstTargetIndexMapExtAddr->offset;
    int32_t dstTargetAnglesOffset   = ((iter * OUTPUT_BATCH_SIZE) + offset) * sizeof(float);
    int32_t dstTargetIndexMapOffset = ((iter * OUTPUT_BATCH_SIZE) + offset) * sizeof(int32_t);

    cupvaSQDFUpdateAddr(dstTargetAngles, 0, cupvaGetVmemAddress((void *)params->dst_target_angles_buf), MEMTYPE_VMEM,
                        dstTargetAnglesBase + dstTargetAnglesOffset, MEMTYPE_DRAM);
    cupvaSQDFUpdateAddr(dstTargetIndexMap, 0, cupvaGetVmemAddress((void *)params->dst_target_index_map_buf),
                        MEMTYPE_VMEM, dstTargetIndexMapBase + dstTargetIndexMapOffset, MEMTYPE_DRAM);

    if (isLastWidth != OUTPUT_BATCH_SIZE)
    {
        cupvaSQDFUpdateTileSize(dstTargetAngles, 0, isLastWidth * sizeof(float), OUT_TARGET_ANGLE_TILE_HEIGHT);
        cupvaSQDFUpdateTileSize(dstTargetIndexMap, 0, isLastWidth * sizeof(int32_t), OUT_TARGET_INDEX_MAP_TILE_HEIGHT);
    }

    cupvaSQDFFlushAndTrig(dstTargetAngles);
    cupvaSQDFFlushAndTrig(dstTargetIndexMap);
}
