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

#ifndef PVA_SOLUTIONS_PRIV_DOA_KNL_H
#define PVA_SOLUTIONS_PRIV_DOA_KNL_H

#include "doa_fft_knl.h"
#include "doa_params.h"

#include <cupva_device.h>
#include <stdint.h>

typedef struct
{
    // Array of 58 agent configurations for DOA processing
    AgenCFG cfgs[58];
    // Pointer to PVA radar general purpose private data
    PvaRadarGPPriv *GP;
    // Parameters for batched FFT operations
    param_fft_batched_t *fftParams;
    // Sampler for phase data processing
    CupvaSampler *samplerPhase;
    // Sampler for elevation angle data processing
    CupvaSampler *samplerElevation;
    // Number of iterations to clear FFT input buffers
    int32_t clearFFTInputIters;
    // Number of snapshot iterations for data collection
    int32_t snapIters;
    // Number of iterations for second sub-bin processing
    int32_t subBinIters2;
    // Number of iterations for first sub-bin processing
    int32_t subBinIters1;
    // Total number of sub-bin processing iterations
    int32_t subBinTotalIters;
    // First iteration count for NCI (Noise Covariance Inverse) FFT
    int32_t NCIfftIter1;
    // Second iteration count for NCI FFT
    int32_t NCIfftIter2;
    // Pointer to peak snapshot data
    int32_t *peakSnapPtr;
    // Pointer to calibration vector data
    int32_t *calibVectorPtr;
    // Pointer to twiddle factors table for FFT
    int32_t *twiddleTable;
    // Pointer to angle index lookup table
    float *aIndexTable;

    // Pointer to transposed processed snapshot data
    int32_t *snapProcessedTranspPtr;
    // Pointer to processed snapshot data
    int32_t *snapProcessedPtr;
    // Pointer to peak indices array
    uint32_t *peakIndicesPtr;
    // Pointer to range index adjusted data
    int32_t *rindPtr;
    // Pointer to Doppler fold index adjusted data
    int32_t *dfoldindPtr;
    // Pointer to Doppler index adjusted data
    int32_t *dindPtr;
    // Pointer to final NCI (Noise Covariance Inverse) data
    uint32_t *nciFinalPtr;
    // Pointer to upper bound data
    int32_t *upperPtr;
    // Pointer to lower bound data
    int32_t *lowerPtr;
    // Pointer to upper FFT data
    int32_t *fftUpperPtr;
    // Pointer to lower FFT data
    int32_t *fftLowerPtr;
    // Pointer to temporary FFT buffer
    int32_t *fftTmpPtr;
    // Pointer to maximum upper FFT values
    int32_t *fftUpperMaxPtr;
    // Pointer to maximum lower FFT values
    int32_t *fftLowerMaxPtr;
    // Pointer to arctangent lookup table
    int32_t *atanTablePtr;
    // Pointer to arcsine lookup table
    int32_t *asinTablePtr;
    // Pointer to index data
    int32_t *indexPtr;
    // Pointer to angle data
    int32_t *anglePtr;
    // Pointer to elevation index data
    int32_t *elevationIndexPtr;
    // Pointer to elevation index sign data
    int32_t *elevationIndexSignPtr;
    // Pointer to elevation angle data
    int32_t *elevationAnglePtr;
    // Pointer to final target results data
    float *targetResultsPtr;
    // Pointer to peak snapshot data
    int32_t **snapTilePtr;
    // Pointer to peak indices data
    uint32_t **indicesTilePtr;
    // Pointer to external memory pointers
    ExtMemPointer **extMemPtrs;
    // Handler for GSDF (Generalized Sparse Data Flow) NCI processing
    GSDFHandler *nciHdl;
} DOAParams;

void updateInputSQDF(SequenceDataFlowHandler &inSQDFIndices, SequenceDataFlowHandler &inSQDFData, int32_t iter,
                     int32_t pingPong, DOAParams *params);

void updateOutputSQDF(SequenceDataFlowHandler &outSQDF, ExtMemPointer &targetExtAddr, int32_t iter, DOAParams *params);

void clearFFTInputInit(int32_t &cfg_idx, DOAParams *params);

void processSnapDataInit(int32_t &cfg_idx, DOAParams *params);

void calcAzimuthInit(int32_t &cfg_idx, DOAParams *params);

void processFFTInit(int32_t &cfg_idx, DOAParams *params);

void DOAKnl(DOAParams *params, bool isLastIter);

#endif // PVA_SOLUTIONS_PRIV_DOA_KNL_H
