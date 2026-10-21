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

#include "doa_knl.h"
#include "doa_params.h"

#include <cupva_device.h>

// VMEM buffer declarations for calibvector and aindex table
/* Stage structure for the shared radix-4 engine, factorized on the host. */
VMEM(A, FFTFactorization, fftFactorization);
VMEM(A, int32_t, calibvector, CALIB_VECTOR_WIDTH); // 16 complex Q15 values (real + imag)
VMEM(A, uint32_t, peakIndices, RDF_DOUBLE(uint32_t, PEAK_TILE_HEIGHT, PEAK_INDICES_HEIGHT)); // 3D peak indices array
VMEM(A, int32_t, peakSnap,
     RDF_DOUBLE(int32_t, PEAK_SNAP_WIDTH, PEAK_TILE_HEIGHT)); // 2D peak snapshots array (complex Q15 values)
VMEM(C, int32_t, snapProcessedTransp,
     PEAK_SNAP_PROCESSED_TRANSP_WIDTH *PEAK_SNAP_HEIGHT);                // 2D peak snapshots array (complex Q20 values)
VMEM(A, int32_t, snapProcessed, PEAK_SNAP_HEIGHT *PEAK_SNAP_WIDTH);      // 2D peak snapshots array (complex Q20 values)
VMEM(B, int32_t, upper, FFT_TILE_SIZE);                                  // 2D FFT tile array (complex Q20 values)
VMEM(B, int32_t, lower, FFT_TILE_SIZE);                                  // 2D FFT tile array (complex Q15 values)
VMEM(A, int32_t, fftUpper, FFT_TILE_SIZE);                               // 2D FFT tile array (complex Q20 values)
VMEM(C, int32_t, fftLower, FFT_TILE_SIZE);                               // 2D FFT tile array (complex Q20 values)
VMEM(C, int32_t, fftTmp, FFT_TILE_SIZE_PAD);                             // 2D FFT tile array (complex Q20 values)
VMEM(B, int32_t, fftUpperMax, PEAK_SNAP_HEIGHT *ELEMENT_NUM_OF_COMPLEX); // 2D FFT tile array (complex Q20 values)
VMEM(C, int32_t, fftLowerMax, PEAK_SNAP_HEIGHT *ELEMENT_NUM_OF_COMPLEX); // 2D FFT tile array (complex Q20 values)
VMEM(C, int32_t, angle, PEAK_SNAP_WIDTH);                                // 2D angle array (complex Q15 values)
VMEM(C, int32_t, angleIndex, PEAK_SNAP_WIDTH);                           // 2D index array (complex Q15 values)
VMEM(B, uint32_t, nciFinal, NUM_NCI_NODES *NCI_NODE_SIZE);               // 3D NCI final array
VMEM(B, int32_t, elevationIndex, PEAK_TILE_HEIGHT);                      // 2D elevation array (complex Q15 values)
VMEM(B, int32_t, elevationIndexSign, PEAK_TILE_HEIGHT);                  // 2D elevation array (complex Q15 values)
VMEM(B, int32_t, elevationAngle, PEAK_TILE_HEIGHT);                      // 2D elevation array (complex Q15 values)
VMEM(B, float, targetResults, RDF_DOUBLE(float, NUM_TARGET_ROWS, PEAK_TILE_HEIGHT)); // Target
VMEM(A, CupvaSampler, samplerPhase);                                                 // Target detection results array
VMEM(A, CupvaSampler, samplerElevation);                                             // Target detection results array

VMEM_POINTER(A, indicesExtAddr);
VMEM_POINTER(A, snapExtAddr);
VMEM_POINTER(A, targetExtAddr);

// VMEM declaration for radar processing parameters
VMEM(A, PvaRadarGPPriv, GP);             // Radar general parameters structure
VMEM(A, DOAParams, params);              // Radar general parameters structure
VMEM(A, param_fft_batched_t, fftParams); // FFT parameters structure
VMEM(A, uint32_t, numPeaks);             // Number of detected peaks

// SQDF handle declaration
VMEM_SEQDF_HANDLER(A, InSqdfHdl);
VMEM_SEQDF_HANDLER(A, InSnapIndicesHdl);
VMEM_SEQDF_HANDLER(A, InSnapDataHdl);
VMEM_SEQDF_HANDLER(A, targetHdl);

// GSDF handle declaration for nci_final data transfer
VMEM_GSDF_HANDLER(A, nciHdl, NUM_NCI_NODES);

alignas(64) int32_t chess_storage(DMb % 1
                                  : chess_segment(A)) twiddle[TWIDDLE_SIZE] = {
#include "doa_fft_coeff.h"
};

alignas(64) int32_t chess_storage(DMb % 1
                                  : chess_segment(B)) atan_table[ATAN_TABLE_SIZE] = {
#include "doa_atan_table.h"
};

alignas(64) int32_t chess_storage(DMb % 1
                                  : chess_segment(B)) asin_table[ASIN_TABLE_SIZE] = {
#include "doa_asin_table.h"
};

alignas(64) uint32_t chess_storage(DMb % 1
                                   : chess_segment(A)) aindicesTbl[AINDICES_WIDTH] = {
#include "doa_aindices_table.h"
};

CUPVA_VPU_MAIN()
{
    int32_t err = 0;
    // open SQDF write data flow
    cupvaSQDFOpen(InSqdfHdl);

    // TrigSQDF:
    cupvaSQDFTrig(InSqdfHdl);
    cupvaSQDFSync(InSqdfHdl);

    cupvaSQDFClose(InSqdfHdl);

    // open SQDF read data flow for peak snapshots and indices
    cupvaSQDFOpen(InSnapIndicesHdl);
    cupvaSQDFOpen(InSnapDataHdl);
    // open SQDF write data flow for target results
    cupvaSQDFOpen(targetHdl);

    uint32_t *indicesTilePtr[2]  = {&peakIndices[0], &peakIndices[PEAK_TILE_HEIGHT * PEAK_INDICES_HEIGHT]};
    int32_t *snapTilePtr[2]      = {&peakSnap[0], &peakSnap[PEAK_SNAP_WIDTH * PEAK_TILE_HEIGHT]};
    float *targetResultsPtr[2]   = {&targetResults[0], &targetResults[PEAK_TILE_HEIGHT * NUM_TARGET_ROWS]};
    ExtMemPointer *extMemPtrs[2] = {&indicesExtAddr, &snapExtAddr};

    GP.numPeaks        = numPeaks;
    GP.numPeakIters    = (numPeaks + PEAK_TILE_HEIGHT - 1) / PEAK_TILE_HEIGHT;
    GP.remainPeakIters = ((numPeaks % PEAK_TILE_HEIGHT) + PEAK_SNAP_HEIGHT - 1) / PEAK_SNAP_HEIGHT;
    GP.remainPeaks     = numPeaks % PEAK_TILE_HEIGHT;

    params.GP                     = &GP;
    params.samplerPhase           = &samplerPhase;
    params.samplerElevation       = &samplerElevation;
    params.snapProcessedTranspPtr = &snapProcessedTransp[0];
    params.snapProcessedPtr       = &snapProcessed[0];
    params.calibVectorPtr         = &calibvector[0];
    params.twiddleTable           = (int32_t *)twiddle;
    params.aIndexTable            = (float *)&aindicesTbl[0];
    params.upperPtr               = &upper[0];
    params.lowerPtr               = &lower[0];
    params.fftUpperPtr            = &fftUpper[0];
    params.fftLowerPtr            = &fftLower[0];
    params.fftTmpPtr              = &fftTmp[0];
    params.fftUpperMaxPtr         = &fftUpperMax[0];
    params.fftLowerMaxPtr         = &fftLowerMax[0];
    params.nciFinalPtr            = &nciFinal[0];
    params.elevationIndexPtr      = &elevationIndex[0];
    params.elevationIndexSignPtr  = &elevationIndexSign[0];
    params.elevationAnglePtr      = &elevationAngle[0];
    params.indicesTilePtr         = indicesTilePtr;
    params.snapTilePtr            = snapTilePtr;
    params.extMemPtrs             = extMemPtrs;

    params.anglePtr         = &angle[0];
    params.indexPtr         = &angleIndex[0];
    params.atanTablePtr     = &atan_table[0];
    params.asinTablePtr     = &asin_table[0];
    params.nciHdl           = nciHdl;
    fftParams.cfgs          = &params.cfgs[0];
    fftParams.log2N         = FFT_LOG2_NUM_POINTS;
    fftParams.nbatch        = 8;
    fftParams.shift_qbits   = TWIDDLE_QBITS - 16;
    fftParams.factorization = &fftFactorization;
    params.fftParams        = &fftParams;

    int32_t cfg_idx = 0;
    clearFFTInputInit(cfg_idx, &params);
    processSnapDataInit(cfg_idx, &params);
    processFFTInit(cfg_idx, &params);
    calcAzimuthInit(cfg_idx, &params);

    int32_t pingPong        = 0;
    params.peakSnapPtr      = snapTilePtr[pingPong];
    params.peakIndicesPtr   = indicesTilePtr[pingPong];
    params.targetResultsPtr = targetResultsPtr[pingPong];

    // open GSDF read data flow for nci_final data
    {

        bool status = cupvaGSDFOpen(nciHdl, &nciFinal[0], NCI_NODE_WIDTH, NCI_NODE_SIZE);

        ExtMemPointer ptr;
        cupvaGetL2ExtMemPointer(&ptr, 0U);
        status &= cupvaGSDFResetExtBase(nciHdl, &ptr, GP.nciPaddedWidth, GP.nciPaddedHeight, GP.nciPaddedWidth);
        cupvaGSDFCycleBuffers(nciHdl);
        status &= cupvaGSDFResetExtBase(nciHdl, &ptr, GP.nciPaddedWidth, GP.nciPaddedHeight, GP.nciPaddedWidth);
        cupvaGSDFCycleBuffers(nciHdl);

        if (status != true || numPeaks == 0)
        {
            err = (status == false) ? 1 : 0;
            goto DOA_EXIT;
        }
    }

    updateInputSQDF(InSnapIndicesHdl, InSnapDataHdl, 0, pingPong, &params);

    cupvaFloatingPointNANErrorEnabled(false);

    // Process tiles using RDF
    for (int i = 0; i < GP.numPeakIters; i++)
    {
        cupvaSQDFSync(InSnapIndicesHdl);
        cupvaSQDFSync(InSnapDataHdl);

        if (i != GP.numPeakIters - 1)
        {
            updateInputSQDF(InSnapIndicesHdl, InSnapDataHdl, i + 1, pingPong ^ 1, &params);
        }

        // Process the acquired tile data
        // DoaKnl processing would go here
        DOAKnl(&params, i == (GP.numPeakIters - 1));

        if (i != 0)
        {
            cupvaSQDFSync(targetHdl);
        }

        updateOutputSQDF(targetHdl, targetExtAddr, i, &params);

        pingPong                = !pingPong;
        params.peakSnapPtr      = snapTilePtr[pingPong];
        params.peakIndicesPtr   = indicesTilePtr[pingPong];
        params.targetResultsPtr = targetResultsPtr[pingPong];
    }

    if (GP.numPeakIters > 0)
    {
        cupvaSQDFSync(targetHdl);
    }

DOA_EXIT:
    // close GSDF read data flow
    cupvaGSDFClose(nciHdl);

    // close SQDF read data flow
    cupvaSQDFClose(InSnapIndicesHdl);
    cupvaSQDFClose(InSnapDataHdl);
    // close SQDF write data flow for target results
    cupvaSQDFClose(targetHdl);

    return err;
}
