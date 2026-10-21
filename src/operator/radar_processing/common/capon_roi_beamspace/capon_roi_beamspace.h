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

/**
 * @file capon_roi_beamspace.h
 *
 * @brief Host-side beamspace geometry and steering-vector LUT for the Capon ROI preprocessing operator.
 *
 * The steering LUT is constant for a given array and steering grid, so applications build it once on the
 * host (caponROIPreprocessSteeringLut) and pass it to pvaCaponROIPreprocessSubmit, which gathers one block
 * per detection. The CPU reference of the operator (samples/.../capon_roi_preprocess/ref) shares this
 * geometry, so the gathered steering is bit-identical to the reference's.
 */

#ifndef CAPON_ROI_BEAMSPACE_H
#define CAPON_ROI_BEAMSPACE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Size of the per-TX DDM decode arrays in the (POD) config: the most TX the operator accepts.
#define CAPON_ROI_PREPROC_MAX_TX 8

// ============================================================================
// Configuration
// ============================================================================

/**
 * @brief Geometry and tuning for the Capon ROI beamspace projection.
 *
 * Plain C POD (no default member initializers): zero-initialize (e.g.
 * `CaponROIPreprocConfig cfg = {0};`) then set every field.
 */
typedef struct CaponROIPreprocConfig
{
    // Element-space array geometry (physical antennas).
    int32_t numTx; ///< transmit channels
    int32_t numRx; ///< receive channels
    /// RX array layout when numAzVirt == 0: [numAzimuthElements, numElevationElements]. numTx * numAzimuthElements
    /// is then the virtual azimuth aperture (beamAzSize), and numRx must equal
    /// numAzimuthElements * numElevationElements. Ignored when numAzVirt != 0.
    int32_t numAzimuthElements;
    /// Elevation rows of the virtual array (numElVirt), in both modes.
    int32_t numElevationElements;
    /// 0: the virtual array is laid out from the TX and RX positions, numTx * numAzimuthElements wide.
    /// Nonzero: the virtual array is an explicit grid numAzVirt wide and numElevationElements tall (the operator's
    /// numAzVirt x numElVirt); channel vc0 = rx*numTx + tx sits at column vc0 % numAzVirt and row vc0 / numAzVirt,
    /// txSpacing apart in azimuth and rxElSpacing in elevation. numTx * numRx may then be smaller than the grid
    /// (the remaining channels are zero), and numRx is not tied to numAzimuthElements * numElevationElements.
    int32_t numAzVirt;

    int32_t numChirps; ///< Doppler dimension of the range-Doppler cube

    // ROI / steering grid (operator limits: svDim <= 16, numAzBins 1-16, numElBins 1-8, ROI inside the
    // beamspace).
    int32_t roiAzSize; ///< ROI azimuth size in physical beamspace bins
    int32_t roiElSize; ///< ROI elevation size in physical beamspace bins (svDim = roiAzSize*roiElSize)
    int32_t numAzBins; ///< steering-grid azimuth directions per ROI (numStVecAz)
    int32_t numElBins; ///< steering-grid elevation directions per ROI (numStVecEl)

    // Doppler selection + snapshot construction.
    int32_t enableFBA; ///< non-zero: forward-backward averaging (doubles snapshot count)
    /// Doppler window around the detection, >= 2 (effective = 2*floor(N/2) bins).
    int32_t dopplerNeighborhood;
    /// Snapshot height (operator's H; a multiple of 8 in [8, 64]). The FBA-expanded snapshot count must be
    /// in [1, H]; the remaining rows are written as zero, as the operator does.
    int32_t numDopplerBinsCovMat;

    // Antenna element spacings in wavelengths (used to synthesize steering vectors).
    double txSpacing;
    double rxAzSpacing;
    double rxElSpacing;

    // Fixed-point output Q formats (fractional bits) the operator was created with, each in [0, 30].
    int32_t inputQBits;    ///< snapshotsROI quantization
    int32_t steeringQBits; ///< steeringVectorsROI quantization

    /// Doppler shift per TX in bins; any value, applied modulo numChirps.
    int32_t relDopplerOffset[CAPON_ROI_PREPROC_MAX_TX];
    /// Decode weight per TX, ddmaWeightRe[tx] + j*ddmaWeightIm[tx]; any magnitude.
    float ddmaWeightRe[CAPON_ROI_PREPROC_MAX_TX];
    float ddmaWeightIm[CAPON_ROI_PREPROC_MAX_TX];
} CaponROIPreprocConfig;

/**
 * @brief Build the full quantized steering-vector lookup table the operator gathers from.
 *
 * One steering block per ROI start position, laid out HW [numRoiPos, svBlockInts] with
 * numRoiPos = (beamAz-roiAz+1) * (beamEl-roiEl+1), ROI index = roiElStart*numRoiAzPos + roiAzStart,
 * and each block in NCHW [svDim, numElBins, numAzBins] complex 2S32. Uses the same steering math as
 * caponROIPreprocess(), so a device gather of this LUT is bit-identical to the reference steering.
 *
 * @param lut Destination, pre-sized to numRoiPos * svDim * numElBins * numAzBins * 2 int32.
 * @return 0 on success; negative on invalid configuration.
 */
int caponROIPreprocessSteeringLut(const CaponROIPreprocConfig *cfg, int32_t *lut);

#ifdef __cplusplus
}
#endif

#endif // CAPON_ROI_BEAMSPACE_H
