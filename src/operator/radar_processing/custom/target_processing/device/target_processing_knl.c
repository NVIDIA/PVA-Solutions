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
 * @file target_processing_knl.c
 * @brief VPU kernel implementation for Target Processing
 *
 * This file implements the PVA (Programmable Vision Accelerator) VPU kernel for
 * target processing, which performs Range-Velocity (RV) decoupling and Cartesian
 * coordinate computation from radar detection data.
 *
 * Algorithm Overview:
 * 1. Read detection list (range and Doppler indices) for each target
 * 2. Read target angles (azimuth, elevation, power) from DOA processing
 * 3. Convert range indices to physical range (meters)
 * 4. Convert Doppler indices to velocity (m/s) with RV decoupling
 * 5. Convert spherical coordinates (range, azimuth, elevation) to Cartesian (X, Y, Z)
 * 6. Write output target list with all computed properties
 *
 * Key Features:
 * - Batched processing for efficiency
 * - Support for FFT-shifted and non-shifted Doppler
 * - DDM Doppler offset correction for multi-TX radar
 */

#include "target_processing_knl.h"

#include <cupva_device.h>
#define ENABLE_VECTOR_CODE_VERSION 1

/**
 * @brief Compute fmod for float values
 *
 * This helper function computes the floating-point remainder of x/y.
 *
 * @param x Dividend
 * @param y Divisor
 * @return Remainder of x/y
 */
static inline float fmodf_custom(float x, float y)
{
    return x - ((int)(x / y)) * y;
}

static inline dvfloatx dvfmodf_custom(dvfloatx x, float y)
{
    return x - dvintx_dvfp(dvfp_dvintx_trunc(x * frcp(y))) * y;
}

static inline dvintx dvintmod_custom(dvintx x, int32_t y)
{
    return x - dvmulwhw(dvfp_dvintx_trunc(dvintx_dvfp(x) * frcp((float)y)), y);
}

// ============================================================================
// RV Decoupling Functions
// Algorithm: FMCW coupling correction with 5-hypothesis velocity disambiguation
// ============================================================================

/**
 * @brief Compute velocity with 5-hypothesis disambiguation loop
 *
 * It searches through 5 velocity hypotheses to find one within the unambiguous
 * velocity range.
 *
 * @param vTempVIn Base velocity values (f_fast * contVFast - f_slow * contVSlow)
 * @param vTempVOut Output velocity values (in m/s, before km/h conversion)
 * @param tempVCoeff Array of 5 hypothesis coefficients
 * @param velocityAmbHigh High velocity bound (negative value in m/s)
 * @param velocityAmbLow Low velocity bound (positive value in m/s)
 */
static inline void calcVelocityRvDecoupling(dvfloatx &vTempVIn, dvfloatx &vTempVOut, const float *tempVCoeff,
                                            float velocityAmbHigh, float velocityAmbLow)
{
    vTempVOut.lo = replicatef(0.0f);
    vTempVOut.hi = replicatef(0.0f);

    for (int i = 0; i < 5; i++)
    {
        dvfloatx vTempV = vTempVIn - tempVCoeff[i];
        dvintx found    = (vTempV >= velocityAmbHigh) && (vTempV <= velocityAmbLow) && (vTempVOut == 0.0f);
        vTempVOut       = dvmux(found, vTempV, vTempVOut);
    }
}

/**
 * @brief Compute range and velocity using RV decoupling (vectorized version)
 *
 * Applies FMCW coupling correction to compute accurate range and velocity
 * from interpolated range and Doppler bin indices.
 *
 * @param dvInterpolatedRangeIdx   Interpolated range bin indices (input)
 * @param dvInterpolatedDopplerIdx Interpolated Doppler bin indices (input)
 * @param rvParams                 RV decoupling parameters (Infineon-style)
 * @param dvRange                  Output range in meters
 * @param dvVelocity               Output velocity in km/h
 */
static inline void computeRvDecoupling(dvfloatx dvInterpolatedRangeIdx, dvfloatx dvInterpolatedDopplerIdx,
                                       const rv_decoupling_params_t *rvParams, dvfloatx &dvRange, dvfloatx &dvVelocity)
{
    // Extract parameters from struct
    const float f_fast_delta    = rvParams->f_fast_delta;
    const float f_slow_delta    = rvParams->f_slow_delta;
    const float contR           = rvParams->contR;
    const float contVFast       = rvParams->contVFast;
    const float contVSlow       = rvParams->contVSlow;
    const float velocityAmbHigh = rvParams->velocityAmbHigh;
    const float velocityAmbLow  = rvParams->velocityAmbLow;
    const float *tempVCoeff     = rvParams->tempVCoeff;

    // Compute f_fast and f_slow (using interpolated indices, +1 for 1-based indexing)
    dvfloatx dvFast = (dvInterpolatedRangeIdx + 1.0f) * f_fast_delta;
    dvfloatx dvSlow = (dvInterpolatedDopplerIdx + 1.0f) * f_slow_delta;

    // Range with FMCW coupling correction
    dvRange = (dvFast - dvSlow) * contR;

    // Velocity with 5-hypothesis disambiguation
    dvfloatx dvTempVBase = dvFast * contVFast - dvSlow * contVSlow;
    dvfloatx dvTempV;
    calcVelocityRvDecoupling(dvTempVBase, dvTempV, tempVCoeff, velocityAmbHigh, velocityAmbLow);

    // Convert from m/s to km/h
    dvVelocity = dvTempV * 3.6f;
}

// ============================================================================
// Quadratic Interpolation Functions
// Algorithm: Parabolic fit on 3 points to find sub-bin peak location
// ============================================================================

/**
 * @brief Vectorized quadratic interpolation for sub-bin accuracy
 *
 * Given 3 samples (y0, y1, y2) centered at y1 (the peak), compute the
 * fractional offset 'p' from the center to the true peak location.
 * Uses parabolic (quadratic) fit: p = (y0 - y2) / (2 * (y0 - 2*y1 + y2))
 *
 * @param y0 Value at index -1 (before peak)
 * @param y1 Value at index 0 (peak)
 * @param y2 Value at index +1 (after peak)
 * @param p Output: fractional offset from center, in range [-1, 1]
 */
static inline void quadraticInterpolation(dvfloatx &y0, dvfloatx &y1, dvfloatx &y2, dvfloatx &p)
{
    dvfloatx vDen    = y0 - y1 * 2.0f + y2;
    dvfloatx vDenRcp = dvfrcp(vDen * 2.0f);
    p                = (y0 - y2) * vDenRcp;
    // Clamp to valid range [-1, 1]
    dvfloatx one, negOne;
    one.lo          = replicatef(1.0f);
    one.hi          = replicatef(1.0f);
    negOne.lo       = replicatef(-1.0f);
    negOne.hi       = replicatef(-1.0f);
    dvintx gtOne    = (p > 1.0f);
    dvintx ltNegOne = (p < -1.0f);
    p               = dvmux(gtOne, one, p);
    p               = dvmux(ltNegOne, negOne, p);
}

void detection_list_transpose_init(void *args)
{
    target_processing_params_t *param = (target_processing_params_t *)args;
    // Initialize AGEN configurations for input detection list
    {
        AgenWrapper wrapper;

        // Buffer 0: Setup AGEN for reading detection list
        agen in0     = init(param->detectionList);
        wrapper.size = sizeof(int32_t);
        wrapper.n1   = ((param->numDetections * DETECTION_LIST_TILE_PITCH) + (16 - 1)) / 16;
        wrapper.s1   = 16;
        INIT_AGEN1(in0, wrapper);

        AgenCFG *cfgs0 = &param->cfgsLoadDetectionList;
        cfgs0[0]       = extract_agen_cfg(in0);
    }

    // Initialize AGEN configurations for storing range index of detection list
    {
        AgenWrapper wrapper;

        // Buffer 0: Setup AGEN for storing range index of detection list
        agen out0    = init(param->detectionListRangeIndex);
        wrapper.size = sizeof(int32_t);
        wrapper.n1   = (param->numDetections + (16 - 1)) / 16;
        wrapper.s1   = 16;
        INIT_AGEN1(out0, wrapper);

        AgenCFG *cfgs0 = &param->cfgsStoreDetectionListRangeIndex;
        cfgs0[0]       = extract_agen_cfg(out0);
    }

    // Initialize AGEN configurations for storing Doppler index of detection list
    {
        AgenWrapper wrapper;

        // Buffer 0: Setup AGEN for storing Doppler index of detection list
        agen out0    = init(param->detectionListDopplerIndex);
        wrapper.size = sizeof(int32_t);
        wrapper.n1   = (param->numDetections + (16 - 1)) / 16;
        wrapper.s1   = 16;
        INIT_AGEN1(out0, wrapper);

        AgenCFG *cfgs0 = &param->cfgsStoreDetectionListDopplerIndex;
        cfgs0[0]       = extract_agen_cfg(out0);
    }
}

#if ENABLE_VECTOR_CODE_VERSION
void detection_list_transpose_kernel(void *args)
{
    target_processing_params_t *param = (target_processing_params_t *)args;

    agen a_src         = init_agen_from_cfg(param->cfgsLoadDetectionList);
    agen a_dst_range   = init_agen_from_cfg(param->cfgsStoreDetectionListRangeIndex);
    agen a_dst_doppler = init_agen_from_cfg(param->cfgsStoreDetectionListDopplerIndex);
    // Perform detection list transpose
    dvintx dvDetectionList_0;
    dvintx dvDetectionList_1;
    dvintx dvDetectionListRangeIndex;
    dvintx dvDetectionListDopplerIndex;

    for (int i = 0; i < (param->numDetections + (16 - 1)) / 16; i++) chess_loop_range(16, )
    chess_prepare_for_pipelining
    {
        dvintx dvDetectionList_0 = dvint_load_di(a_src);
        dvintx dvDetectionList_1 = dvint_load_di(a_src);

        dvDetectionListRangeIndex.lo = dvDetectionList_0.lo;
        dvDetectionListRangeIndex.hi = dvDetectionList_1.lo;

        dvDetectionListDopplerIndex.lo = dvDetectionList_0.hi;
        dvDetectionListDopplerIndex.hi = dvDetectionList_1.hi;

        vstore(dvDetectionListRangeIndex, a_dst_range);
        vstore(dvDetectionListDopplerIndex, a_dst_doppler);
    }
}
#else
void detection_list_transpose_kernel(void *args)
{
    target_processing_params_t *param  = (target_processing_params_t *)args;
    const int32_t *detectionList       = param->detectionList;
    int32_t *detectionListRangeIndex   = param->detectionListRangeIndex;
    int32_t *detectionListDopplerIndex = param->detectionListDopplerIndex;

    for (int i = 0; i < param->numDetections; i++)
    {
        detectionListRangeIndex[i]   = detectionList[i * 2 + 0];
        detectionListDopplerIndex[i] = detectionList[i * 2 + 1];
    }
}
#endif

void target_processing_init(void *args0, void *args1)
{
    target_processing_params_t *param0 = (target_processing_params_t *)args0;
    target_processing_params_t *param1 = (target_processing_params_t *)args1;

    // Initialize samplers for range bin index and velocity bin index (buffer 0)
    {
        CupvaSampler *range_sampler    = param0->samplerRange;
        CupvaSampler *velocity_sampler = param0->samplerVelocity;

        // Setup range sampler: maps detection list indices to range bin indices
        CupvaSamplerInput1D const range_input = {.data           = param0->detectionListRangeIndex,
                                                 .type           = SAMPLER_INPUT_TYPE_S32,
                                                 .length         = (uint32_t)(param0->numDetections),
                                                 .outOfRangeMode = SAMPLER_OUT_OF_RANGE_CONSTANT};

        CupvaSamplerIndices1D const range_indices = {.data               = param0->targetIndexMap,
                                                     .type               = SAMPLER_INDEX_TYPE_U32,
                                                     .width              = (uint32_t)(BATCH_SIZE),
                                                     .height             = 1,
                                                     .fractionalBits     = 0,
                                                     .fractionalHandling = SAMPLER_FRAC_HANDLING_ROUND};

        CupvaSamplerOutput const range_output = {.data      = param0->rangeBinIndex,
                                                 .pitch     = BATCH_SIZE,
                                                 .transMode = TRANS_MODE_NONE};

        cupvaSamplerSetup(range_sampler, &range_input, &range_indices, &range_output);

        // Setup velocity sampler: maps detection list indices to velocity bin indices
        CupvaSamplerInput1D const velocity_input = {.data           = param0->detectionListDopplerIndex,
                                                    .type           = SAMPLER_INPUT_TYPE_S32,
                                                    .length         = (uint32_t)(param0->numDetections),
                                                    .outOfRangeMode = SAMPLER_OUT_OF_RANGE_CONSTANT};

        CupvaSamplerIndices1D const velocity_indices = {.data               = param0->targetIndexMap,
                                                        .type               = SAMPLER_INDEX_TYPE_U32,
                                                        .width              = (uint32_t)(BATCH_SIZE),
                                                        .height             = 1,
                                                        .fractionalBits     = 0,
                                                        .fractionalHandling = SAMPLER_FRAC_HANDLING_ROUND};

        CupvaSamplerOutput const velocity_output = {.data      = param0->dopplerBinIndex,
                                                    .pitch     = BATCH_SIZE,
                                                    .transMode = TRANS_MODE_NONE};
        cupvaSamplerSetup(velocity_sampler, &velocity_input, &velocity_indices, &velocity_output);
    }

    // Initialize samplers for range bin index and velocity bin index (buffer 1)
    {
        CupvaSampler *range_sampler    = param1->samplerRange;
        CupvaSampler *velocity_sampler = param1->samplerVelocity;

        // Setup range sampler: maps detection list indices to range bin indices
        CupvaSamplerInput1D const range_input = {.data           = param1->detectionListRangeIndex,
                                                 .type           = SAMPLER_INPUT_TYPE_S32,
                                                 .length         = (uint32_t)(param1->numDetections),
                                                 .outOfRangeMode = SAMPLER_OUT_OF_RANGE_CONSTANT};

        CupvaSamplerIndices1D const range_indices = {.data               = param1->targetIndexMap,
                                                     .type               = SAMPLER_INDEX_TYPE_U32,
                                                     .width              = (uint32_t)(BATCH_SIZE),
                                                     .height             = 1,
                                                     .fractionalBits     = 0,
                                                     .fractionalHandling = SAMPLER_FRAC_HANDLING_ROUND};

        CupvaSamplerOutput const range_output = {.data      = param1->rangeBinIndex,
                                                 .pitch     = BATCH_SIZE,
                                                 .transMode = TRANS_MODE_NONE};

        cupvaSamplerSetup(range_sampler, &range_input, &range_indices, &range_output);

        // Setup velocity sampler: maps detection list indices to velocity bin indices
        CupvaSamplerInput1D const velocity_input = {.data           = param1->detectionListDopplerIndex,
                                                    .type           = SAMPLER_INPUT_TYPE_S32,
                                                    .length         = (uint32_t)(param1->numDetections),
                                                    .outOfRangeMode = SAMPLER_OUT_OF_RANGE_CONSTANT};

        CupvaSamplerIndices1D const velocity_indices = {.data               = param1->targetIndexMap,
                                                        .type               = SAMPLER_INDEX_TYPE_U32,
                                                        .width              = (uint32_t)(BATCH_SIZE),
                                                        .height             = 1,
                                                        .fractionalBits     = 0,
                                                        .fractionalHandling = SAMPLER_FRAC_HANDLING_ROUND};

        CupvaSamplerOutput const velocity_output = {.data      = param1->dopplerBinIndex,
                                                    .pitch     = BATCH_SIZE,
                                                    .transMode = TRANS_MODE_NONE};
        cupvaSamplerSetup(velocity_sampler, &velocity_input, &velocity_indices, &velocity_output);
    }

    // Initialize AGEN configurations for input angles (buffer 0 and buffer 1)
    {
        AgenWrapper wrapper;

        // Buffer 0: Setup AGEN for reading angles
        agen in0     = init(param0->targetAngles);
        wrapper.size = sizeof(float);
        wrapper.n1   = TARGET_ANGLES_TILE_HEIGHT;
        wrapper.s1   = TARGET_ANGLES_TILE_PITCH;
        wrapper.n2   = (TARGET_ANGLES_TILE_WIDTH + (16 - 1)) / 16;
        wrapper.s2   = 16;
        INIT_AGEN2(in0, wrapper);

        AgenCFG *cfgs0 = &param0->cfgsLoadAngles;
        cfgs0[0]       = extract_agen_cfg(in0);

        // Buffer 1: Setup AGEN for reading angles
        agen in1     = init(param1->targetAngles);
        wrapper.size = sizeof(float);
        wrapper.n1   = TARGET_ANGLES_TILE_HEIGHT;
        wrapper.s1   = TARGET_ANGLES_TILE_PITCH;
        wrapper.n2   = (TARGET_ANGLES_TILE_WIDTH + (16 - 1)) / 16;
        wrapper.s2   = 16;
        INIT_AGEN2(in1, wrapper);

        AgenCFG *cfgs1 = &param1->cfgsLoadAngles;
        cfgs1[0]       = extract_agen_cfg(in1);
    }

    // Initialize AGEN configurations for input range bin index (buffer 0 and buffer 1)
    {
        AgenWrapper wrapper;

        // Buffer 0: Setup AGEN for reading range bin index
        agen in0     = init(param0->rangeBinIndex);
        wrapper.size = sizeof(int32_t);
        wrapper.n1   = 1;
        wrapper.s1   = TARGET_INDEX_MAP_TILE_PITCH;
        wrapper.n2   = (TARGET_INDEX_MAP_TILE_WIDTH + (16 - 1)) / 16;
        wrapper.s2   = 16;
        INIT_AGEN2(in0, wrapper);

        AgenCFG *cfgs0 = &param0->cfgsLoadRangeBinIndex;
        cfgs0[0]       = extract_agen_cfg(in0);

        // Buffer 1: Setup AGEN for reading range bin index
        agen in1     = init(param1->rangeBinIndex);
        wrapper.size = sizeof(int32_t);
        wrapper.n1   = 1;
        wrapper.s1   = TARGET_INDEX_MAP_TILE_PITCH;
        wrapper.n2   = (TARGET_INDEX_MAP_TILE_WIDTH + (16 - 1)) / 16;
        wrapper.s2   = 16;
        INIT_AGEN2(in1, wrapper);

        AgenCFG *cfgs1 = &param1->cfgsLoadRangeBinIndex;
        cfgs1[0]       = extract_agen_cfg(in1);
    }

    // Initialize AGEN configurations for input doppler bin index (buffer 0 and buffer 1)
    {
        AgenWrapper wrapper;

        // Buffer 0: Setup AGEN for reading doppler bin index
        agen in0     = init(param0->dopplerBinIndex);
        wrapper.size = sizeof(int32_t);
        wrapper.n1   = 1;
        wrapper.s1   = TARGET_INDEX_MAP_TILE_PITCH;
        wrapper.n2   = (TARGET_INDEX_MAP_TILE_WIDTH + (16 - 1)) / 16;
        wrapper.s2   = 16;
        INIT_AGEN2(in0, wrapper);

        AgenCFG *cfgs0 = &param0->cfgsLoadDopplerBinIndex;
        cfgs0[0]       = extract_agen_cfg(in0);

        // Buffer 1: Setup AGEN for reading doppler bin index
        agen in1     = init(param1->dopplerBinIndex);
        wrapper.size = sizeof(int32_t);
        wrapper.n1   = 1;
        wrapper.s1   = TARGET_INDEX_MAP_TILE_PITCH;
        wrapper.n2   = (TARGET_INDEX_MAP_TILE_WIDTH + (16 - 1)) / 16;
        wrapper.s2   = 16;
        INIT_AGEN2(in1, wrapper);

        AgenCFG *cfgs1 = &param1->cfgsLoadDopplerBinIndex;
        cfgs1[0]       = extract_agen_cfg(in1);
    }

    // Initialize AGEN configurations for output velocity (buffer 0 and buffer 1)
    {
        AgenWrapper wrapper;

        // Buffer 0: Setup AGEN for writing velocity
        agen out0    = init(param0->targetList + 0 * TARGET_LIST_TILE_PITCH);
        wrapper.size = sizeof(float);
        wrapper.n1   = 1;
        wrapper.s1   = TARGET_LIST_TILE_PITCH;
        wrapper.n2   = (TARGET_LIST_TILE_WIDTH + (16 - 1)) / 16;
        wrapper.s2   = 16;
        INIT_AGEN2(out0, wrapper);

        AgenCFG *cfgs0 = &param0->cfgsStoreVelocity;
        cfgs0[0]       = extract_agen_cfg(out0);

        // Buffer 1: Setup AGEN for writing velocity
        agen out1    = init(param1->targetList + 0 * TARGET_LIST_TILE_PITCH);
        wrapper.size = sizeof(float);
        wrapper.n1   = 1;
        wrapper.s1   = TARGET_LIST_TILE_PITCH;
        wrapper.n2   = (TARGET_LIST_TILE_WIDTH + (16 - 1)) / 16;
        wrapper.s2   = 16;
        INIT_AGEN2(out1, wrapper);

        AgenCFG *cfgs1 = &param1->cfgsStoreVelocity;
        cfgs1[0]       = extract_agen_cfg(out1);
    }

    // Initialize AGEN configurations for output range (buffer 0 and buffer 1)
    {
        AgenWrapper wrapper;

        // Buffer 0: Setup AGEN for writing range
        agen out0    = init(param0->targetList + 1 * TARGET_LIST_TILE_PITCH);
        wrapper.size = sizeof(float);
        wrapper.n1   = 1;
        wrapper.s1   = TARGET_LIST_TILE_PITCH;
        wrapper.n2   = (TARGET_LIST_TILE_WIDTH + (16 - 1)) / 16;
        wrapper.s2   = 16;
        INIT_AGEN2(out0, wrapper);

        AgenCFG *cfgs0 = &param0->cfgsStoreRange;
        cfgs0[0]       = extract_agen_cfg(out0);

        // Buffer 1: Setup AGEN for writing range
        agen out1    = init(param1->targetList + 1 * TARGET_LIST_TILE_PITCH);
        wrapper.size = sizeof(float);
        wrapper.n1   = 1;
        wrapper.s1   = TARGET_LIST_TILE_PITCH;
        wrapper.n2   = (TARGET_LIST_TILE_WIDTH + (16 - 1)) / 16;
        wrapper.s2   = 16;
        INIT_AGEN2(out1, wrapper);

        AgenCFG *cfgs1 = &param1->cfgsStoreRange;
        cfgs1[0]       = extract_agen_cfg(out1);
    }

    // Initialize AGEN configurations for output angles (buffer 0 and buffer 1)
    {
        AgenWrapper wrapper;

        // Buffer 0: Setup AGEN for writing angles
        agen out0    = init(param0->targetList + 2 * TARGET_LIST_TILE_PITCH);
        wrapper.size = sizeof(float);
        wrapper.n1   = 2;
        wrapper.s1   = TARGET_LIST_TILE_PITCH;
        wrapper.n2   = (TARGET_LIST_TILE_WIDTH + (16 - 1)) / 16;
        wrapper.s2   = 16;
        INIT_AGEN2(out0, wrapper);

        AgenCFG *cfgs0 = &param0->cfgsStoreAngles;
        cfgs0[0]       = extract_agen_cfg(out0);

        // Buffer 1: Setup AGEN for writing angles
        agen out1    = init(param1->targetList + 2 * TARGET_LIST_TILE_PITCH);
        wrapper.size = sizeof(float);
        wrapper.n1   = 2;
        wrapper.s1   = TARGET_LIST_TILE_PITCH;
        wrapper.n2   = (TARGET_LIST_TILE_WIDTH + (16 - 1)) / 16;
        wrapper.s2   = 16;
        INIT_AGEN2(out1, wrapper);

        AgenCFG *cfgs1 = &param1->cfgsStoreAngles;
        cfgs1[0]       = extract_agen_cfg(out1);
    }

    // Initialize AGEN configurations for output x, y, z (buffer 0 and buffer 1)
    {
        AgenWrapper wrapper;

        // Buffer 0: Setup AGEN for writing x, y, z
        agen out0    = init(param0->targetList + 4 * TARGET_LIST_TILE_PITCH);
        wrapper.size = sizeof(float);
        wrapper.n1   = 3;
        wrapper.s1   = TARGET_LIST_TILE_PITCH;
        wrapper.n2   = (TARGET_LIST_TILE_WIDTH + (16 - 1)) / 16;
        wrapper.s2   = 16;
        INIT_AGEN2(out0, wrapper);

        AgenCFG *cfgs0 = &param0->cfgsStoreXYZ;
        cfgs0[0]       = extract_agen_cfg(out0);

        // Buffer 1: Setup AGEN for writing x, y, z
        agen out1    = init(param1->targetList + 4 * TARGET_LIST_TILE_PITCH);
        wrapper.size = sizeof(float);
        wrapper.n1   = 3;
        wrapper.s1   = TARGET_LIST_TILE_PITCH;
        wrapper.n2   = (TARGET_LIST_TILE_WIDTH + (16 - 1)) / 16;
        wrapper.s2   = 16;
        INIT_AGEN2(out1, wrapper);

        AgenCFG *cfgs1 = &param1->cfgsStoreXYZ;
        cfgs1[0]       = extract_agen_cfg(out1);
    }

    // Initialize AGEN configurations for output peak power in dB (buffer 0 and buffer 1)
    {
        AgenWrapper wrapper;

        // Buffer 0: Setup AGEN for writing peak power in dB
        agen out0    = init(param0->targetList + 7 * TARGET_LIST_TILE_PITCH);
        wrapper.size = sizeof(float);
        wrapper.n1   = 1;
        wrapper.s1   = TARGET_LIST_TILE_PITCH;
        wrapper.n2   = (TARGET_LIST_TILE_WIDTH + (16 - 1)) / 16;
        wrapper.s2   = 16;
        INIT_AGEN2(out0, wrapper);

        AgenCFG *cfgs0 = &param0->cfgsStorePower;
        cfgs0[0]       = extract_agen_cfg(out0);

        // Buffer 1: Setup AGEN for writing peak power in dB
        agen out1    = init(param1->targetList + 7 * TARGET_LIST_TILE_PITCH);
        wrapper.size = sizeof(float);
        wrapper.n1   = 1;
        wrapper.s1   = TARGET_LIST_TILE_PITCH;
        wrapper.n2   = (TARGET_LIST_TILE_WIDTH + (16 - 1)) / 16;
        wrapper.s2   = 16;
        INIT_AGEN2(out1, wrapper);

        AgenCFG *cfgs1 = &param1->cfgsStorePower;
        cfgs1[0]       = extract_agen_cfg(out1);
    }

    // Initialize AGEN configurations for NCI loading (buffer 0 and buffer 1)
    // NCI data is loaded from VMEM buffer filled by GSDF
    // Each tile is NCI_NODE_WIDTH x NCI_NODE_HEIGHT (3x3) with 9 uint32_t values
    if (param0->nciData != NULL)
    {
        AgenWrapper wrapper;

        // Setup AGEN for loading NCI tiles via transposed load
        // lane_ofst=1 gives 68-byte pitch (17 elements) between lanes
        int32_t vecw   = 16;
        int32_t niter1 = (16 + vecw - 1) / vecw;

        agen in0      = init(param0->nciData);
        in0.lane_ofst = 1;
        wrapper.size  = sizeof(uint32_t);
        wrapper.n1    = 9;
        wrapper.s1    = 1;
        wrapper.n2    = niter1;
        wrapper.s2    = NCI_NODE_SIZE * vecw;
        wrapper.n3    = 1;
        wrapper.s3    = 0;
        INIT_AGEN3(in0, wrapper);

        AgenCFG *cfgs0 = &param0->cfgsLoadNci;
        cfgs0[0]       = extract_agen_cfg(in0);
    }

    if (param1->nciData != NULL)
    {
        AgenWrapper wrapper;

        // Setup AGEN for loading NCI tiles via transposed load
        int32_t vecw   = 16;
        int32_t niter1 = (16 + vecw - 1) / vecw;

        agen in1      = init(param1->nciData);
        in1.lane_ofst = 1;
        wrapper.size  = sizeof(uint32_t);
        wrapper.n1    = 9;
        wrapper.s1    = 1;
        wrapper.n2    = niter1;
        wrapper.s2    = NCI_NODE_SIZE * vecw;
        wrapper.n3    = 1;
        wrapper.s3    = 0;
        INIT_AGEN3(in1, wrapper);

        AgenCFG *cfgs1 = &param1->cfgsLoadNci;
        cfgs1[0]       = extract_agen_cfg(in1);
    }

    return;
}

/**
 * @brief Process targets to compute velocity, range, and Cartesian coordinates
 *
 * This function performs the core target processing computations:
 * - Range computation from range bin index
 * - Velocity computation from Doppler bin index with RV decoupling
 * - Cartesian coordinate computation from spherical coordinates
 *
 * @param params Processing parameters
 * @param targetIdx Starting target index for this batch
 * @param batchSize Number of targets to process in this batch
 */
#if ENABLE_VECTOR_CODE_VERSION
void target_processing_kernel(const target_processing_params_t *params, int32_t batchSize)
{
    CupvaSampler *range_sampler    = params->samplerRange;
    CupvaSampler *velocity_sampler = params->samplerVelocity;

    int32_t *targetIndexMap = params->targetIndexMap;
    if (batchSize != BATCH_SIZE)
    {
        for (int32_t i = batchSize; i < BATCH_SIZE; i++)
        {
            targetIndexMap[i] = targetIndexMap[0];
        }
    }

    cupvaSamplerStart(range_sampler);
    // Extract parameters
    const float rangeResolution     = params->rangeResolution;
    const float wavelength          = params->wavelength;
    const int32_t numChirps         = params->numChirps;
    const int32_t fftShiftedDoppler = params->fftShiftedDoppler;
    const int32_t numTargets        = params->numTargets;

    // Access data from params structure
    int32_t *detectionListRangeIndex   = params->detectionListRangeIndex;
    int32_t *detectionListDopplerIndex = params->detectionListDopplerIndex;
    const float *targetAngles          = params->targetAngles;
    const float *ddmDopplerOffsets     = params->ddmDopplerOffsets;
    float *targetList                  = params->targetList;

    // Pre-computed derived parameters (computed on host to avoid expensive VPU divisions)
    const float invSweepTime = params->invSweepTime; // 1.0 / sweepTime
    const float invNumChirps = params->invNumChirps; // 1.0 / numChirps

    int32_t *rangeBinIndexList   = params->rangeBinIndex;
    int32_t *dopplerBinIndexList = params->dopplerBinIndex;

    agen a_load_target_angles  = init_agen_from_cfg(params->cfgsLoadAngles);
    agen a_store_target_angles = init_agen_from_cfg(params->cfgsStoreAngles);
    agen a_store_power         = init_agen_from_cfg(params->cfgsStorePower);

    agen a_load_range_bin_index = init_agen_from_cfg(params->cfgsLoadRangeBinIndex);
    agen a_store_range          = init_agen_from_cfg(params->cfgsStoreRange);

    agen a_store_xyz = init_agen_from_cfg(params->cfgsStoreXYZ);

    agen a_load_doppler_bin_index = init_agen_from_cfg(params->cfgsLoadDopplerBinIndex);
    agen a_store_velocity         = init_agen_from_cfg(params->cfgsStoreVelocity);

    // Extract RV decoupling parameters
    const rv_decoupling_params_t *rvParams     = params->rvParams;
    const int32_t enableRvDecoupling           = rvParams->enableRvDecoupling;
    const int32_t enableQuadraticInterpolation = rvParams->enableQuadraticInterpolation;

    // Initialize NCI agen if quadratic interpolation is enabled
    agen a_load_nci;
    if (enableQuadraticInterpolation && params->nciData != NULL)
    {
        a_load_nci = init_agen_from_cfg(params->cfgsLoadNci);
    }

    // Process each target in the first batch
    {
        // Read target angles (azimuth, elevation, power)
        dvfloatx dvTargetAnglesAzimuth   = dvfloat_load(a_load_target_angles);
        dvfloatx dvTargetAnglesElevation = dvfloat_load(a_load_target_angles);
        dvfloatx dvTargetAnglesPower     = dvfloat_load(a_load_target_angles);

        dvfloatx dvAzimuthNormalized   = dvTargetAnglesAzimuth * frcp(360.0f);
        dvfloatx dvElevationNormalized = dvTargetAnglesElevation * frcp(360.0f);

        // Compute Cartesian coordinates
        dvfloatx dvCosEl = dvfcos(dvElevationNormalized);
        dvfloatx dvSinEl = dvfsin(dvElevationNormalized);
        dvfloatx dvCosAz = dvfcos(dvAzimuthNormalized);
        dvfloatx dvSinAz = dvfsin(dvAzimuthNormalized);

        // store target angles (azimuth, elevation, power)
        vstore(dvTargetAnglesAzimuth, a_store_target_angles);
        vstore(dvTargetAnglesElevation, a_store_target_angles);
        vstore(dvTargetAnglesPower, a_store_power);

        // 1. Load range and doppler bin indices
        cupvaSamplerWait();
        cupvaSamplerStart(velocity_sampler);
        dvintx dvRangeBinIndex = dvint_load(a_load_range_bin_index);
        cupvaSamplerWait();
        dvintx dvDopplerBinIndex = dvint_load(a_load_doppler_bin_index);

        // Compute folded doppler indices for NCI GSDF access
        if (enableQuadraticInterpolation && params->dopplerFoldedIndex != NULL && params->nciWidth > 0)
        {
            int32_t nciWidth = params->nciWidth;
            for (int32_t k = 0; k < 16; k++)
            {
                params->dopplerFoldedIndex[k] = dopplerBinIndexList[k] % nciWidth;
            }
        }

        // Convert bin indices to float for calculations
        dvfloatx dvRangeBinFloat   = dvintx_dvfp(dvRangeBinIndex);
        dvfloatx dvDopplerBinFloat = dvintx_dvfp(dvDopplerBinIndex);

        // Apply quadratic interpolation if enabled (adds fractional offsets to bin indices)
        dvfloatx dvInterpolatedRangeIdx   = dvRangeBinFloat;
        dvfloatx dvInterpolatedDopplerIdx = dvDopplerBinFloat;

        if (enableQuadraticInterpolation && params->nciData != NULL && params->nciHdl != NULL)
        {
            // Compute folded doppler indices for first batch
            int32_t nciWidth = params->nciWidth;
            for (int32_t k = 0; k < 16; k++)
            {
                params->dopplerFoldedIndex[k] = dopplerBinIndexList[k] % nciWidth;
            }

            // Trigger GSDF to fetch first batch of NCI tiles from L2 into VMEM
            int32_t numNciTiles = 16;
            cupvaGSDFUpdateTiles(params->nciHdl, params->dopplerFoldedIndex, rangeBinIndexList, numNciTiles);
            cupvaGSDFTrig(params->nciHdl);
            cupvaGSDFSync(params->nciHdl);

            // Re-init AGEN to point to start of VMEM buffer
            a_load_nci = init_agen_from_cfg(params->cfgsLoadNci);

            // Load 3x3 NCI neighborhood using transposed load
            dvintx v11 = dvuint_load_transp(a_load_nci);
            dvintx v12 = dvuint_load_transp(a_load_nci);
            dvintx v13 = dvuint_load_transp(a_load_nci);
            dvintx v21 = dvuint_load_transp(a_load_nci);
            dvintx v22 = dvuint_load_transp(a_load_nci);
            dvintx v23 = dvuint_load_transp(a_load_nci);
            dvintx v31 = dvuint_load_transp(a_load_nci);
            dvintx v32 = dvuint_load_transp(a_load_nci);
            dvintx v33 = dvuint_load_transp(a_load_nci);

            // Convert NCI values to float
            dvfloatx v12Float, v21Float, v22Float, v23Float, v32Float;
            v12Float.lo = vint_vfp(v12.lo);
            v12Float.hi = vint_vfp(v12.hi);
            v21Float.lo = vint_vfp(v21.lo);
            v21Float.hi = vint_vfp(v21.hi);
            v22Float.lo = vint_vfp(v22.lo);
            v22Float.hi = vint_vfp(v22.hi);
            v23Float.lo = vint_vfp(v23.lo);
            v23Float.hi = vint_vfp(v23.hi);
            v32Float.lo = vint_vfp(v32.lo);
            v32Float.hi = vint_vfp(v32.hi);

            // Compute quadratic interpolation offsets
            dvfloatx pr, pd;
            quadraticInterpolation(v12Float, v22Float, v32Float, pr); // Range interpolation
            quadraticInterpolation(v21Float, v22Float, v23Float, pd); // Doppler interpolation

            // Apply interpolation offsets to get sub-bin accuracy
            dvInterpolatedRangeIdx   = dvRangeBinFloat + pr;
            dvInterpolatedDopplerIdx = dvDopplerBinFloat + pd;
        }

        dvfloatx dvRange, dvVelocity;

        if (enableRvDecoupling)
        {
            // RV Decoupling Mode: FMCW coupling correction with velocity disambiguation
            computeRvDecoupling(dvInterpolatedRangeIdx, dvInterpolatedDopplerIdx, rvParams, dvRange, dvVelocity);
        }
        else
        {
            // ================================================================
            // Standard Mode (original implementation)
            // When quadratic interpolation is enabled, uses sub-bin indices
            // ================================================================

            // Compute range from (interpolated) range bin index
            dvRange = dvInterpolatedRangeIdx * rangeResolution;

            // Compute velocity from Doppler bin index
            float ddmDopplerOffset      = ddmDopplerOffsets[0];
            dvintx dvAdjustedDopplerIdx = dvDopplerBinIndex;

            // Apply FFT shift if not already shifted
            if (!fftShiftedDoppler)
            {
                dvAdjustedDopplerIdx = dvintmod_custom(dvDopplerBinIndex + (numChirps / 2), numChirps);
            }
            // Normalize Doppler frequency: (idx / numChirps) - 0.5
            dvfloatx dvNormalizedDopplerFreq = (dvintx_dvfp(dvAdjustedDopplerIdx) * invNumChirps) - 0.5f;

            // Apply DDM Doppler offset correction
            dvfloatx dvDopplerValue = dvfmodf_custom((dvNormalizedDopplerFreq + ddmDopplerOffset) + 0.5f, 1.0f) - 0.5f;

            // Convert to actual frequency and velocity
            dvfloatx dvDopplerFrequency = dvDopplerValue * invSweepTime;
            dvVelocity                  = dvDopplerFrequency * wavelength * frcp(2.0f);
        }

        // Store range and velocity
        vstore(dvRange, a_store_range);
        vstore(dvVelocity, a_store_velocity);

        // 2. Convert spherical coordinates to Cartesian
        dvfloatx dvZ           = dvRange * dvSinEl;
        dvfloatx dvGroundRange = (params->groundProjection == 1) ? dvRange * dvCosEl : dvRange;
        dvfloatx dvX           = dvGroundRange * ((params->longitudinalAxis == 1) ? dvSinAz : dvCosAz);
        dvfloatx dvY           = dvGroundRange * ((params->longitudinalAxis == 1) ? dvCosAz : dvSinAz);

        vstore(dvX, a_store_xyz);
        vstore(dvY, a_store_xyz);
        vstore(dvZ, a_store_xyz);
    }

    // Process each target in the batch except the first batch
    for (int32_t i = 16; i < batchSize; i += 16) chess_loop_range(16, )
    chess_prepare_for_pipelining
    {
        // Read target angles (azimuth, elevation, power)
        dvfloatx dvTargetAnglesAzimuth   = dvfloat_load(a_load_target_angles);
        dvfloatx dvTargetAnglesElevation = dvfloat_load(a_load_target_angles);
        dvfloatx dvTargetAnglesPower     = dvfloat_load(a_load_target_angles);
        // store target angles (azimuth, elevation, power)
        vstore(dvTargetAnglesAzimuth, a_store_target_angles);
        vstore(dvTargetAnglesElevation, a_store_target_angles);
        vstore(dvTargetAnglesPower, a_store_power);

        // 1. Load range and doppler bin indices
        dvintx dvRangeBinIndex   = dvint_load(a_load_range_bin_index);
        dvintx dvDopplerBinIndex = dvint_load(a_load_doppler_bin_index);

        // Convert bin indices to float for calculations
        dvfloatx dvRangeBinFloat   = dvintx_dvfp(dvRangeBinIndex);
        dvfloatx dvDopplerBinFloat = dvintx_dvfp(dvDopplerBinIndex);

        // Apply quadratic interpolation if enabled (adds fractional offsets to bin indices)
        dvfloatx dvInterpolatedRangeIdx   = dvRangeBinFloat;
        dvfloatx dvInterpolatedDopplerIdx = dvDopplerBinFloat;

        if (enableQuadraticInterpolation && params->nciData != NULL && params->nciHdl != NULL)
        {
            // Compute folded doppler indices for this batch
            int32_t nciWidth = params->nciWidth;
            for (int32_t k = 0; k < 16; k++)
            {
                params->dopplerFoldedIndex[k] = dopplerBinIndexList[i + k] % nciWidth;
            }

            // Trigger GSDF for this group of 16 elements
            a_load_nci          = init_agen_from_cfg(params->cfgsLoadNci);
            int32_t numNciTiles = 16;
            cupvaGSDFUpdateTiles(params->nciHdl, params->dopplerFoldedIndex, rangeBinIndexList + i, numNciTiles);
            cupvaGSDFTrig(params->nciHdl);
            cupvaGSDFSync(params->nciHdl);

            // Load 3x3 NCI neighborhood using transposed load
            dvintx v11 = dvuint_load_transp(a_load_nci);
            dvintx v12 = dvuint_load_transp(a_load_nci);
            dvintx v13 = dvuint_load_transp(a_load_nci);
            dvintx v21 = dvuint_load_transp(a_load_nci);
            dvintx v22 = dvuint_load_transp(a_load_nci);
            dvintx v23 = dvuint_load_transp(a_load_nci);
            dvintx v31 = dvuint_load_transp(a_load_nci);
            dvintx v32 = dvuint_load_transp(a_load_nci);
            dvintx v33 = dvuint_load_transp(a_load_nci);

            // Convert NCI values to float
            dvfloatx v12Float, v21Float, v22Float, v23Float, v32Float;
            v12Float.lo = vint_vfp(v12.lo);
            v12Float.hi = vint_vfp(v12.hi);
            v21Float.lo = vint_vfp(v21.lo);
            v21Float.hi = vint_vfp(v21.hi);
            v22Float.lo = vint_vfp(v22.lo);
            v22Float.hi = vint_vfp(v22.hi);
            v23Float.lo = vint_vfp(v23.lo);
            v23Float.hi = vint_vfp(v23.hi);
            v32Float.lo = vint_vfp(v32.lo);
            v32Float.hi = vint_vfp(v32.hi);

            // Compute quadratic interpolation offsets
            dvfloatx pr, pd;
            quadraticInterpolation(v12Float, v22Float, v32Float, pr);
            quadraticInterpolation(v21Float, v22Float, v23Float, pd);

            // Apply interpolation offsets
            dvInterpolatedRangeIdx   = dvRangeBinFloat + pr;
            dvInterpolatedDopplerIdx = dvDopplerBinFloat + pd;
        }

        dvfloatx dvRange, dvVelocity;

        if (enableRvDecoupling)
        {
            // RV Decoupling Mode: FMCW coupling correction with velocity disambiguation
            computeRvDecoupling(dvInterpolatedRangeIdx, dvInterpolatedDopplerIdx, rvParams, dvRange, dvVelocity);
        }
        else
        {
            // ================================================================
            // Standard Mode (original implementation)
            // When quadratic interpolation is enabled, uses sub-bin indices
            // ================================================================

            // Compute range from (interpolated) range bin index
            dvRange = dvInterpolatedRangeIdx * rangeResolution;

            // Compute velocity from Doppler bin index
            float ddmDopplerOffset      = ddmDopplerOffsets[0];
            dvintx dvAdjustedDopplerIdx = dvDopplerBinIndex;

            // Apply FFT shift if not already shifted
            if (!fftShiftedDoppler)
            {
                dvAdjustedDopplerIdx = dvintmod_custom(dvDopplerBinIndex + (numChirps / 2), numChirps);
            }
            // Normalize Doppler frequency: (idx / numChirps) - 0.5
            dvfloatx dvNormalizedDopplerFreq = (dvintx_dvfp(dvAdjustedDopplerIdx) * invNumChirps) - 0.5f;

            // Apply DDM Doppler offset correction
            dvfloatx dvDopplerValue = dvfmodf_custom((dvNormalizedDopplerFreq + ddmDopplerOffset) + 0.5f, 1.0f) - 0.5f;

            // Convert to actual frequency and velocity
            dvfloatx dvDopplerFrequency = dvDopplerValue * invSweepTime;
            dvVelocity                  = dvDopplerFrequency * wavelength * frcp(2.0f);
        }

        // Store range and velocity
        vstore(dvRange, a_store_range);
        vstore(dvVelocity, a_store_velocity);

        // 2. Convert spherical coordinates to Cartesian
        dvfloatx dvAzimuthNormalized   = dvTargetAnglesAzimuth * frcp(360.0f);
        dvfloatx dvElevationNormalized = dvTargetAnglesElevation * frcp(360.0f);

        dvfloatx dvCosEl = dvfcos(dvElevationNormalized);
        dvfloatx dvSinEl = dvfsin(dvElevationNormalized);
        dvfloatx dvCosAz = dvfcos(dvAzimuthNormalized);
        dvfloatx dvSinAz = dvfsin(dvAzimuthNormalized);

        dvfloatx dvZ           = dvRange * dvSinEl;
        dvfloatx dvGroundRange = (params->groundProjection == 1) ? dvRange * dvCosEl : dvRange;
        dvfloatx dvX           = dvGroundRange * ((params->longitudinalAxis == 1) ? dvSinAz : dvCosAz);
        dvfloatx dvY           = dvGroundRange * ((params->longitudinalAxis == 1) ? dvCosAz : dvSinAz);

        vstore(dvX, a_store_xyz);
        vstore(dvY, a_store_xyz);
        vstore(dvZ, a_store_xyz);
    }
}
#else
/**
 * @brief Scalar velocity calculation with 5-hypothesis disambiguation (for non-vector version)
 */
static inline float calcVelocityRvDecouplingScalar(float tempVIn, const float *tempVCoeff, float velocityAmbHigh,
                                                   float velocityAmbLow)
{
    for (int i = 0; i < 5; i++)
    {
        float tempV = tempVIn - tempVCoeff[i];
        if (tempV >= velocityAmbHigh && tempV <= velocityAmbLow)
        {
            return tempV;
        }
    }
    return 0.0f;
}

/**
 * @brief Compute range and velocity using RV decoupling (scalar version)
 *
 * Applies FMCW coupling correction to compute accurate range and velocity
 * from range and Doppler bin indices.
 *
 * @param rangeBinIndex   Range bin index (input)
 * @param dopplerBinIndex Doppler bin index (input)
 * @param rvParams        RV decoupling parameters
 * @param range           Output range in meters
 * @param velocity        Output velocity in km/h
 */
static inline void computeRvDecouplingScalar(int32_t rangeBinIndex, int32_t dopplerBinIndex,
                                             const rv_decoupling_params_t *rvParams, float &range, float &velocity)
{
    // Extract parameters from struct
    const float f_fast_delta    = rvParams->f_fast_delta;
    const float f_slow_delta    = rvParams->f_slow_delta;
    const float contR           = rvParams->contR;
    const float contVFast       = rvParams->contVFast;
    const float contVSlow       = rvParams->contVSlow;
    const float velocityAmbHigh = rvParams->velocityAmbHigh;
    const float velocityAmbLow  = rvParams->velocityAmbLow;
    const float *tempVCoeff     = rvParams->tempVCoeff;

    // Compute f_fast and f_slow (using integer indices, +1 for 1-based indexing)
    float f_fast = ((float)rangeBinIndex + 1.0f) * f_fast_delta;
    float f_slow = ((float)dopplerBinIndex + 1.0f) * f_slow_delta;

    // Range with FMCW coupling correction
    range = (f_fast - f_slow) * contR;

    // Velocity with 5-hypothesis disambiguation
    float tempVBase = f_fast * contVFast - f_slow * contVSlow;
    float tempV     = calcVelocityRvDecouplingScalar(tempVBase, tempVCoeff, velocityAmbHigh, velocityAmbLow);

    // Convert from m/s to km/h
    velocity = tempV * 3.6f;
}

void target_processing_kernel(const target_processing_params_t *params, int32_t batchSize)
{
    // Extract parameters
    const float rangeResolution     = params->rangeResolution;
    const float wavelength          = params->wavelength;
    const int32_t numChirps         = params->numChirps;
    const int32_t fftShiftedDoppler = params->fftShiftedDoppler;
    const int32_t numTargets        = params->numTargets;

    // Access data from params structure
    int32_t *detectionListRangeIndex   = params->detectionListRangeIndex;
    int32_t *detectionListDopplerIndex = params->detectionListDopplerIndex;
    int32_t *targetIndexMap            = params->targetIndexMap;
    const float *targetAngles          = params->targetAngles;
    const float *ddmDopplerOffsets     = params->ddmDopplerOffsets;
    float *targetList                  = params->targetList;

    // Pre-computed derived parameters (computed on host to avoid expensive VPU divisions)
    const float invSweepTime = params->invSweepTime; // 1.0 / sweepTime
    const float invNumChirps = params->invNumChirps; // 1.0 / numChirps

    // RV Decoupling parameters
    const rv_decoupling_params_t *rvParams = params->rvParams;
    const int32_t enableRvDecoupling       = rvParams->enableRvDecoupling;

    int32_t *rangeBinIndexList   = params->rangeBinIndex;
    int32_t *dopplerBinIndexList = params->dopplerBinIndex;

    CupvaSampler *range_sampler    = params->samplerRange;
    CupvaSampler *velocity_sampler = params->samplerVelocity;

    if (batchSize != BATCH_SIZE)
    {
        for (int32_t i = batchSize; i < BATCH_SIZE; i++)
        {
            targetIndexMap[i] = targetIndexMap[0];
        }
    }

    for (int32_t i = 0; i < batchSize; i++)
    {
        // Get detection index for this target
        int32_t detectionIdx = targetIndexMap[i];

        rangeBinIndexList[i]   = detectionListRangeIndex[detectionIdx];
        dopplerBinIndexList[i] = detectionListDopplerIndex[detectionIdx];
    }

    // Process each target in the batch
    for (int32_t i = 0; i < batchSize; i++)
    {
        // Read target angles (azimuth, elevation, power)
        float azimuthDeg   = targetAngles[0 * BATCH_SIZE + i];
        float elevationDeg = targetAngles[1 * BATCH_SIZE + i];
        float peakPowerDb  = targetAngles[2 * BATCH_SIZE + i];

        // Get bin indices
        int32_t rangeBinIndex   = rangeBinIndexList[i];
        int32_t dopplerBinIndex = dopplerBinIndexList[i];

        float range, velocity;

        if (enableRvDecoupling)
        {
            // RV Decoupling Mode: FMCW coupling correction with velocity disambiguation
            computeRvDecouplingScalar(rangeBinIndex, dopplerBinIndex, rvParams, range, velocity);
        }
        else
        {
            // ================================================================
            // Standard Mode (original implementation)
            // ================================================================

            // Compute range from range bin index
            range = (float)rangeBinIndex * rangeResolution;

            // Compute velocity from Doppler bin index
            int32_t adjustedDopplerIdx = dopplerBinIndex;

            // Apply FFT shift if not already shifted
            if (!fftShiftedDoppler)
            {
                adjustedDopplerIdx = (dopplerBinIndex + numChirps / 2) % numChirps;
            }

            // Normalize Doppler frequency: (idx / numChirps) - 0.5
            float normalizedDopplerFreq = ((float)adjustedDopplerIdx * invNumChirps) - 0.5f;

            // Apply DDM Doppler offset correction
            float ddmDopplerOffset = ddmDopplerOffsets[0];
            float dopplerValue     = fmodf_custom(normalizedDopplerFreq + ddmDopplerOffset + 0.5f, 1.0f) - 0.5f;

            // Convert to actual frequency and velocity
            float dopplerFrequency = dopplerValue * invSweepTime;
            velocity               = (dopplerFrequency * wavelength) / 2.0f;
        }

        // Convert spherical coordinates to Cartesian
        float azimuthNormalized   = azimuthDeg / 360.0f;
        float elevationNormalized = elevationDeg / 360.0f;

        float cosEl = fcos(elevationNormalized);
        float sinEl = fsin(elevationNormalized);
        float cosAz = fcos(azimuthNormalized);
        float sinAz = fsin(azimuthNormalized);

        float z           = range * sinEl;
        float groundRange = (params->groundProjection == 1) ? range * cosEl : range;
        float x           = groundRange * ((params->longitudinalAxis == 1) ? sinAz : cosAz);
        float y           = groundRange * ((params->longitudinalAxis == 1) ? cosAz : sinAz);

        // Write output target list
        // Target list format: [velocity, range, azimuth, elevation, x, y, z, power]
        targetList[0 * BATCH_SIZE + i] = velocity;
        targetList[1 * BATCH_SIZE + i] = range;
        targetList[2 * BATCH_SIZE + i] = azimuthDeg;
        targetList[3 * BATCH_SIZE + i] = elevationDeg;
        targetList[4 * BATCH_SIZE + i] = x;
        targetList[5 * BATCH_SIZE + i] = y;
        targetList[6 * BATCH_SIZE + i] = z;
        targetList[7 * BATCH_SIZE + i] = peakPowerDb;
    }
}
#endif
/**
 * @brief Update input SQDF for loading target index map and angles
 *
 * Configures Sequence Data Flow to read target index map and angles from DRAM to VMEM
 * for the current batch iteration. This function updates the SQDF addresses to point
 * to the correct batch location in external memory and the appropriate double-buffer
 * in VMEM. Handles partial batches by adjusting tile size.
 *
 * @param targetIndexMapHdl SQDF handler for target index map input
 * @param targetAnglesHdl SQDF handler for target angles input
 * @param batchIdx Current batch iteration index
 * @param batchSize Number of valid targets in the batch
 * @param params Parameter structure with buffer and external memory pointers
 */
void updateInputSQDF(SequenceDataFlowHandler &targetIndexMapHdl, SequenceDataFlowHandler &targetAnglesHdl,
                     int32_t batchIdx, int32_t batchSize, target_processing_params_t *params)
{
    ExtMemPointer *targetIndexMapExtAddr = params->targetIndexMapExtPtr;
    ExtMemPointer *targetAnglesExtAddr   = params->targetAnglesExtPtr;

    // Calculate offsets for current batch
    int64_t targetIndexMapBase   = targetIndexMapExtAddr->base + targetIndexMapExtAddr->offset;
    int64_t targetAnglesBase     = targetAnglesExtAddr->base + targetAnglesExtAddr->offset;
    int32_t targetIndexMapOffset = batchIdx * BATCH_SIZE * sizeof(int32_t);
    int32_t targetAnglesOffset   = batchIdx * BATCH_SIZE * sizeof(float);

    // Update SQDF addresses for target index map (DRAM -> VMEM)
    cupvaSQDFUpdateAddr(targetIndexMapHdl, 0, targetIndexMapBase + targetIndexMapOffset, MEMTYPE_DRAM,
                        cupvaGetVmemAddress((void *)params->targetIndexMap), MEMTYPE_VMEM);

    // Update SQDF addresses for target angles (DRAM -> VMEM)
    cupvaSQDFUpdateAddr(targetAnglesHdl, 0, targetAnglesBase + targetAnglesOffset, MEMTYPE_DRAM,
                        cupvaGetVmemAddress((void *)params->targetAngles), MEMTYPE_VMEM);

    // Handle partial batches by adjusting tile size for the last batch
    if (batchSize != BATCH_SIZE)
    {
        cupvaSQDFUpdateTileSize(targetIndexMapHdl, 0, batchSize * sizeof(int32_t), TARGET_INDEX_MAP_TILE_HEIGHT);
        cupvaSQDFUpdateTileSize(targetAnglesHdl, 0, batchSize * sizeof(float), TARGET_ANGLES_TILE_HEIGHT);
    }

    // Flush and trigger both transfers
    cupvaSQDFFlushAndTrig(targetIndexMapHdl);
    cupvaSQDFFlushAndTrig(targetAnglesHdl);
}

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
                      target_processing_params_t *params)
{
    ExtMemPointer *targetListExtAddr = params->targetListExtPtr;

    // Calculate offset for current batch
    int64_t targetListBase   = targetListExtAddr->base + targetListExtAddr->offset;
    int32_t targetListOffset = batchIdx * BATCH_SIZE * sizeof(float);

    // Update SQDF addresses for target list (VMEM -> DRAM)
    cupvaSQDFUpdateAddr(targetListHdl, 0, cupvaGetVmemAddress((void *)params->targetList), MEMTYPE_VMEM,
                        targetListBase + targetListOffset, MEMTYPE_DRAM);

    // Handle partial batches by adjusting tile size for the last batch
    if (batchSize != BATCH_SIZE)
    {
        cupvaSQDFUpdateTileSize(
            targetListHdl, 0, batchSize * sizeof(float),
            params->power_output_requested ? TARGET_LIST_TILE_HEIGHT : (TARGET_LIST_TILE_HEIGHT - 1));
    }

    // Flush and trigger transfer
    cupvaSQDFFlushAndTrig(targetListHdl);
}
