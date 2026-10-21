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
 * @file capon_roi_preprocess_ref.h
 *
 * @brief Scalar reference for the Capon ROI preprocessing (projection-core) operator.
 *
 * Host-side ROI beamspace projection that produces the input tensors of the Capon (MVDR)
 * DoA-on-ROI stage, which consumes snapshots and steering vectors already projected into a
 * per-detection ROI beamspace.
 *
 * Stage names and numbers follow the operator documentation (radarcaponroipreprocess.rst).
 * Per detection, this reference runs:
 *
 *   - Coarse DoA (stage 5): element-space snapshot -> conj -> zero-padded 2D DFT, shifted ->
 *     peak beamspace bin.
 *   - ROI selection (stage 6): an roiAzSize x roiElSize ROI centered on the peak bin (start
 *     peak - (roiSize-1)/2), clamped to the beamspace.
 *   - Doppler window, DDM decode and forward-backward averaging (stages 1-3): per-Doppler-bin
 *     virtual-array snapshots from the range-Doppler cube.
 *   - Azimuth and elevation projection (stages 7-8): each snapshot onto the ROI beamspace
 *     -> Y (svDim x H).
 *   - Output quantization (stage 9): Y -> snapshotsROI in Q(inputQBits).
 *   - Steering block (stage 10): the steering search grid projected onto the same ROI, the block
 *     the operator gathers from its steering LUT.
 *
 * Stage 4 (block-float quantization) is a device-precision step with no equivalent here: all
 * internal math is double precision, and only the final tensor writes quantize to
 * Q(inputQBits) / Q(steeringQBits).
 *
 * Outputs are written directly in the operator's tensor layouts, 2S32 (interleaved re/im):
 *   - snapshotsROI       : NHW  [roiNum, numDopplerBinsCovMat, svDim]
 *   - steeringVectorsROI : NCHW [roiNum, svDim, numElBins, numAzBins]
 *                          (elevation-major: position = el * numAzBins + az)
 *
 * The beamspace transforms are direct DFTs (no FFT dependency). The API is plain C (extern "C",
 * POD structs) so C drivers can call it.
 */

#ifndef CAPON_ROI_PREPROCESS_REF_H
#define CAPON_ROI_PREPROCESS_REF_H

#include "capon_roi_beamspace.h" // CaponROIPreprocConfig and caponROIPreprocessSteeringLut()

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Host inputs to the projection, all dequantized to float.
 *
 * Pointers are borrowed; the caller owns the storage. Complex data is interleaved [re, im].
 */
typedef struct CaponROIPreprocInput
{
    int32_t numDetections; ///< number of ROIs to process (one per CFAR detection)

    /// Per-detection element-space snapshot used only for coarse-DoA ROI selection.
    /// Layout CHW [numDetections, numTx, numRx], complex interleaved.
    const float *snapshots;

    /// Range-Doppler cube. Layout [numRangeBins, numChirps, numRx], complex interleaved. Used to
    /// re-extract per-Doppler-bin snapshots.
    const float *rangeDopplerCube;
    int32_t numRangeBins;

    /// Detection list [numDetections, 2] = (rangeIdx, dopplerIdx), 0-based.
    const int32_t *detectionList;
} CaponROIPreprocInput;

/**
 * @brief Fixed-point (2S32, interleaved re/im) outputs in operator tensor layout.
 *
 * Buffers must be pre-sized by the caller:
 *   - snapshotsROI       : roiNum * numDopplerBinsCovMat * svDim * 2 int32
 *   - steeringVectorsROI : roiNum * svDim * numElBins * numAzBins * 2 int32
 * where svDim = roiAzSize * roiElSize and roiNum = numDetections.
 */
typedef struct CaponROIPreprocOutput
{
    int32_t *snapshotsROI;
    int32_t *steeringVectorsROI;
} CaponROIPreprocOutput;

// ============================================================================
// Entry points
// ============================================================================

/**
 * @brief Run the ROI beamspace projection for every detection.
 *
 * Runs the stages listed in the file overview. Fills @p out with the quantized snapshots and
 * steering vectors the Capon operator consumes.
 *
 * Rejects the configurations the operator rejects for the parameters both share (the limits in
 * OpCaponROIPreprocess.h).
 *
 * @return 0 on success; negative on invalid configuration: -1 bad counts (numTx/numRx outside [1, 8],
 *         numChirps outside [1, 512], numDetections outside [0, 1024]), -2/-3 RX or virtual-array
 *         mismatch, -4 NULL argument, -6 svDim != roiAzSize*roiElSize,
 *         -7 dopplerNeighborhood < 2 or its FBA-expanded snapshot count above numDopplerBinsCovMat,
 *         -8 another operator limit.
 */
int caponROIPreprocess(const CaponROIPreprocConfig *cfg, const CaponROIPreprocInput *in, CaponROIPreprocOutput *out);

#ifdef __cplusplus
}
#endif

#endif // CAPON_ROI_PREPROCESS_REF_H
