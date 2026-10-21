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

#ifndef PVA_SOLUTIONS_PRIV_DOA_PARAMS_H
#define PVA_SOLUTIONS_PRIV_DOA_PARAMS_H

#include <stdint.h>

#define STYPE int32_t
#define DVTYPE dvint
#define DVTYPEX dvintx
#define DVUTYPE dvuint
#define UTYPE uint32_t
#define VTYPEX vintx
#define VTYPE vint
#define DVLOAD(x) dvint_load(x)
#define DVLOAD_DI(x) dvint_load_di(x)
#define VLOAD(x) vint_load(x)
#define DVLOADF(x) dvfloat_load(x)
#define DVULOAD_TRANSP(x) dvuint_load_transp(x)

#define CALIB_VECTOR_WIDTH (16 * ELEMENT_NUM_OF_COMPLEX)
#define AINDICES_WIDTH 256
#define DOA_MAX_TARGET_COUNT 8192

#define PEAK_INDICES_HEIGHT 3
#define PEAK_SNAP_HEIGHT 8
#define PEAK_SNAP_WIDTH CALIB_VECTOR_WIDTH
#define PEAK_SNAP_PROCESSED_TRANSP_WIDTH (PEAK_SNAP_HEIGHT * ELEMENT_NUM_OF_COMPLEX + 1)
// #define PEAK_SNAP_PROCESSED_WIDTH (CALIB_VECTOR_WIDTH)

#define PEAK_SNAP_UNROLLING 2
#define PEAK_TILE_HEIGHT 128

#define PEAK_ITERS (PEAK_TILE_HEIGHT / PEAK_SNAP_HEIGHT)

#define NCI_NODE_WIDTH 3
#define NCI_NODE_HEIGHT 3
#define NCI_NODE_SIZE (16 + 1)
#define NUM_NCI_NODES 32

#define ATAN_TABLE_SIZE 513
#define ASIN_TABLE_SIZE ATAN_TABLE_SIZE
#define Q15_90_DEGREES (int32_t)(90.0f * (1 << 15))
#define Q15_180_DEGREES (int32_t)(180.0f * (1 << 15))
#ifndef PI_CONST_F
#    define PI_CONST_F 3.14159265358979323846f
#endif
#define ANGLE_FLOAT_SCALE (PI_CONST_F / 180.0f)

#define NUM_TARGET_ROWS 7
#define TARGET_DET_VELOCITY_OFFSET 0
#define TARGET_DET_RANGE_OFFSET 1
#define TARGET_DET_AZIMUTH_OFFSET 2
#define TARGET_DET_ELEVATION_OFFSET 3
#define TARGET_DET_X_OFFSET 4
#define TARGET_DET_Y_OFFSET 5
#define TARGET_DET_Z_OFFSET 6

#include "fft_params.h"

typedef struct
{
    /** @brief Number of range bins in the radar system */
    int32_t nr;
    /** @brief Number of Doppler fold repetitions for velocity ambiguity resolution */
    int32_t repeatFold;
    /** @brief Number of snapshots used for array processing */
    int32_t nofSnap;
    /** @brief Number of antenna elements in the array aperture */
    int32_t nofAperture;
    /** @brief Number of FFT points for angle processing (typically 256) */
    int32_t nofAFFT;
    /** @brief Inter-element spacing in the antenna array (in wavelengths) */
    float dis;
    /** @brief Sampling frequency of the radar system (Hz) */
    float fs;
    /** @brief Number of samples per chirp/ramp */
    int32_t nofSamples;
    /** @brief Pulse Repetition Interval (seconds) */
    float PRI;
    /** @brief Number of chirp ramps for Doppler processing */
    int32_t nofRamps;
    /** @brief Velocity resolution (m/s) */
    float deltaV;
    /** @brief Array indices for upper antenna elements in DOA processing */
    int32_t apUpperInd[128];
    /** @brief Snapshot indices for upper antenna elements */
    int32_t apUpper[128];
    /** @brief Array indices for lower antenna elements in DOA processing */
    int32_t apLowerInd[128];
    /** @brief Snapshot indices for lower antenna elements */
    int32_t apLower[128];
    /** @brief Elevation inter-element spacing (in wavelengths) */
    float disE;
    /** @brief Fast velocity conversion factor for range-Doppler mapping */
    float contVFast;
    /** @brief Slow velocity conversion factor for range-Doppler mapping */
    float contVSlow;
    /** @brief Range conversion factor for range-Doppler mapping */
    float contR;
    /** @brief Number of detected peaks/targets to process */
    int32_t numPeaks;
    /** @brief Number of rows in the NCI (Non-Coherent Integration) matrix */
    int32_t numNCIRows;
    /** @brief Number of columns in the NCI matrix */
    int32_t numNCICols;
    /** @brief Number of peaks remaining */
    int32_t remainPeaks;
    /** @brief Number of peak iterations */
    int32_t numPeakIters;
    /** @brief Number of peak iterations remaining */
    int32_t remainPeakIters;
    /** @brief Elevation factor for elevation angle calculation */
    float elevationFactor;
    /** @brief Fast frequency delta */
    float f_fast_delta;
    /** @brief Slow frequency delta */
    float f_slow_delta;
    /** @brief Velocity ambiguity high */
    float velocityAmbHigh;
    /** @brief Velocity ambiguity low */
    float velocityAmbLow;
    /** @brief Velocity tempV coefficients */
    float tempVCoeff[5];
    /** @brief Number of Q bits for the snapshot tensor */
    int32_t calibVectorQbits;
    /** @brief Number of Q bits for the FFT tensor */
    int32_t fftQbits;

    int32_t nciPaddedWidth;
    int32_t nciPaddedHeight;
} PvaRadarGPPriv;

#endif // PVA_SOLUTIONS_PRIV_DOA_PARAMS_H