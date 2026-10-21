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
 * @file OpCaponROIPreprocess.h
 *
 * @brief Capon DoA-on-ROI beamspace-projection preprocessing operator.
 *
 * Produces the ROI beamspace snapshots and steering vectors consumed by a Capon (MVDR)
 * DoA-on-ROI stage. Runs the beamspace projection on the PVA VPU.
 *
 * @note Create generates and quantizes the DFT twiddle tables (with
 * PVACaponROIPreprocessParams::useCustomTwiddles, Submit quantizes the caller's). Per detection, every
 * stage runs on the VPU: (1) Doppler window, (2) DDM decode with
 * PVACaponROIPreprocessParams::relDopplerOffset / ddmaWeight, (3) forward-backward averaging,
 * (4) block-float quantization of the virtual-array snapshots, (5) coarse DoA over the beamspace from the element snapshot (direct DFT, or batched FFT-64 with
 * PVACaponROIPreprocessParams::coarseDoAUseFft), (6) ROI selection, (7) azimuth and (8) elevation
 * projection of the ROI, (9) output quantization to Q(inputQBits), and (10) steering-block gather
 * from the LUT. LOCAL Doppler mode only.
 *
 * @note Submit reads detectionCount (and waz and wel when PVACaponROIPreprocessParams::useCustomTwiddles
 * is set) through host pointers when it is called, not when the stream reaches it. Those tensors must
 * be host-accessible and fully written before Submit is called, and a previous Submit on the same
 * handle must have completed, since it reuses the handle's internal buffers. The other inputs are read
 * by the PVA when the stream reaches the work.
 *
 * @note Only the first detectionCount detections are processed. The input rows from detectionCount up
 * to maxDet are still transferred by the DMA, so they must be valid memory, but their values are not
 * used and need no padding. The rows of snapshotsROI and steeringVectorsROI from detectionCount up to
 * maxDet are unspecified (they may hold stale data); read only the first detectionCount rows.
 *
 * CPU reference: public/samples/radar_processing/capon_roi_preprocess/ref/capon_roi_preprocess_ref.cpp
 *
 * @defgroup PVA_OPERATOR_ALGORITHM_RADAR_CAPON_ROI_PREPROCESS RadarCaponROIPreprocess
 * @{
 */

#ifndef PVA_SOLUTIONS_OPCAPONROIPREPROCESS_H
#define PVA_SOLUTIONS_OPCAPONROIPREPROCESS_H

#include <PvaOperator.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructs the Capon ROI preprocessing (projection-core) operator.
 *
 * \b Limitations:
 * - numTx and numRx in [1, 8], and numTx*numRx <= numAzVirt*numElVirt.
 * - ddmaWeightRe and ddmaWeightIm finite for every TX in [0, numTx).
 * - numAzVirt in [1, 69], numElVirt in [1, 35], and numAzVirt*numElVirt <= 69.
 * - beamAz in [1, 128], beamEl in [1, 64]; beamAz*numAzVirt <= 4096 unless a coarse-DoA grid is used.
 * - coarseElVirt (when nonzero) in [numElVirt, 35]; a grid (coarseElVirt != numElVirt) requires
 *   numElVirt == 1 and the direct-DFT coarse DoA.
 * - roiAz in [1, beamAz], roiEl in [1, beamEl], and svDim == roiAz*roiEl <= 16.
 * - roiAz*numElVirt*numDopplerBinsCovMat <= 1024.
 * - numAzBins in [1, 16], numElBins in [1, 8].
 * - numDopplerBinsCovMat (H) a multiple of 8 in [8, 64]; dopplerNeighborhood >= 2, and
 *   2*floor(dopplerNeighborhood/2), doubled when enableFBA, must not exceed H (the remaining
 *   snapshot rows are zero).
 * - rangeDopplerPlanes: every value a detection reads (its row among the first detectionCount, at the
 *   chirps of its Doppler window shifted by each TX's relDopplerOffset) must decode to real and
 *   imaginary parts that are finite and below 2^24 in magnitude. Not checked: other values give that
 *   detection unspecified (saturated) snapshots.
 * - elementSnapshots (first detectionCount rows): real and imaginary parts below 4 in magnitude,
 *   since coarse DoA converts them to 16-bit Q13. For numElVirt == 1 the azimuth beamspace power is
 *   also kept in 32 bits, which needs the sum of |elementSnapshots| over a row below 180 (about 181
 *   less a rounding margin, for twiddles of complex magnitude at most 1; for example below 2.8 per
 *   element at numAzVirt == 64). Not checked; larger values saturate or wrap and can select the
 *   wrong ROI. Scaling a row by a common factor does not change its ROI.
 * - Custom waz/wel (useCustomTwiddles) entries finite with complex magnitude at most 1; Submit rejects
 *   others.
 * - For numElVirt > 1 the coarse DoA ranks the 2D beamspace on fixed-point power, with no
 *   floating-point refine (numElVirt == 1 re-scores its strongest candidates in FP32): when two
 *   peaks are within about 1e-3 in relative power, it can select another ROI than a
 *   double-precision reference.
 * - inputQBits and steeringQBits in [0, 30]. snapshotsROI values are not saturated: a projected
 *   snapshot of magnitude 2^(31-inputQBits) or more wraps.
 * - numChirps in [1, 512].
 * - maxDet in [1, 1024]; above 256 it is tiled tileCols detections per DMA line, where tileCols is
 *   its smallest divisor >= ceil(maxDet/256) (there must be one <= 256), and tileCols times the
 *   larger of 2*numAzVirt*coarseElVirt and 2*numChirps*numRx must not exceed 65535.
 * - coarseDoAUseFft requires useCustomTwiddles == false, numElVirt == 1, beamAz == 64,
 *   numAzVirt == 64, no coarse-DoA grid and the full coarse-DoA field of view.
 * - The coarse-DoA field of view must be an ordered bin range inside [0, beamAz) x [0, beamEl).
 * - Every tensor densely packed (no padding between rows).
 *
 * @param [out] handle Where the operator instance handle is written. Must not be NULL.
 * @param [in] wazReqs        Az twiddle table, HW [1, beamAz*numAzVirt] complex, NVCV_DATA_TYPE_2F32.
 *                            Used only when params->useCustomTwiddles is set; may be NULL otherwise.
 * @param [in] welReqs        El twiddle table, HW [1, beamEl*coarseElVirt] complex, NVCV_DATA_TYPE_2F32
 *                            (coarseElVirt = numElVirt when no coarse-DoA grid is used).
 *                            Used only when params->useCustomTwiddles is set; may be NULL otherwise.
 * @param [in] detectionCountReqs Number of detections to process, W [1], NVCV_DATA_TYPE_S32; the value
 *                            must be in [0, maxDet]. The values of the per-detection input rows at or
 *                            beyond it are not used (the rows must still be valid memory), and their
 *                            snapshotsROI / steeringVectorsROI rows are unspecified.
 * @param [in] detDopplersReqs Per-detection detected Doppler bin, HW [1, maxDet], NVCV_DATA_TYPE_S32.
 * @param [in] elementSnapshotsReqs Per-detection element snapshot for coarse DoA, HW
 *                            [maxDet, numAzVirt*coarseElVirt] complex NVCV_DATA_TYPE_2F32; vc-index
 *                            rx*numTx+tx, or position y*numAzVirt+x on a coarse-DoA grid.
 * @param [in] rangeDopplerPlanesReqs Per-detection range-Doppler plane, HW [maxDet, numChirps*numRx]
 *                            complex NVCV_DATA_TYPE_2F32 (host-gathered cube[rangeIdx]).
 * @param [in] steeringLutReqs Precomputed per-ROI-position steering LUT, HW
 *                            [numRoiPos, svDim*numElBins*numAzBins] complex NVCV_DATA_TYPE_2S32,
 *                            where numRoiPos = (beamAz-roiAz+1)*(beamEl-roiEl+1) and the block for
 *                            ROI start (roiAzStart, roiElStart) lives at row
 *                            roiElStart*(beamAz-roiAz+1)+roiAzStart, quantized to Q(steeringQBits).
 *                            Constant across frames; the device gathers one block per detection.
 * @param [in] snapshotsROIReqs Output snapshots, NHW [maxDet, numDopplerBinsCovMat(H), svDim],
 *                            NVCV_DATA_TYPE_2S32.
 * @param [in] steeringVectorsROIReqs Output steering vectors, NCHW [maxDet, svDim, numElBins, numAzBins],
 *                            NVCV_DATA_TYPE_2S32 (the LUT block selected for each detection's ROI).
 * @param [in] params Capon ROI preprocessing parameters. \ref PVACaponROIPreprocessParams. Must not be NULL.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle, params or a required tensor requirement is NULL, or a
 *                                    parameter or tensor requirement is out of range.
 * @retval NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, e.g. building its program or data flows.
 * @retval NVCV_SUCCESS                Operator created successfully.
 */
NVCVStatus pvaCaponROIPreprocessCreate(
    NVCVOperatorHandle *handle, NVCVTensorRequirements const *const wazReqs,
    NVCVTensorRequirements const *const welReqs, NVCVTensorRequirements const *const detectionCountReqs,
    NVCVTensorRequirements const *const detDopplersReqs, NVCVTensorRequirements const *const elementSnapshotsReqs,
    NVCVTensorRequirements const *const rangeDopplerPlanesReqs, NVCVTensorRequirements const *const steeringLutReqs,
    NVCVTensorRequirements const *const snapshotsROIReqs, NVCVTensorRequirements const *const steeringVectorsROIReqs,
    PVACaponROIPreprocessParams const *const params);

#ifdef __cplusplus
}

/**
 * Submits the Capon ROI preprocessing (projection-core) operator to a cuPVA stream.
 *
 * @param [in] handle       Operator handle. Must not be NULL.
 * @param [in] stream       Valid cuPVA stream.
 * @param [in] waz          Az twiddle table (see Create); may be NULL unless useCustomTwiddles is set.
 * @param [in] wel          El twiddle table (see Create); may be NULL unless useCustomTwiddles is set.
 * @param [in] detectionCount Number of detections to process (see Create).
 * @param [in] detDopplers  Per-detection detected Doppler bin (see Create).
 * @param [in] elementSnapshots Per-detection element snapshots for coarse DoA (see Create).
 * @param [in] rangeDopplerPlanes Per-detection range-Doppler planes (see Create).
 * @param [in] steeringLut Precomputed per-ROI-position steering LUT (see Create).
 * @param [out] snapshotsROI Output ROI beamspace snapshots (see Create).
 * @param [out] steeringVectorsROI Output per-detection steering vectors (see Create).
 *
 * @note detectionCount and custom waz/wel are read on the CPU during this call; see the file-level
 * note.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle or a required tensor is NULL, a tensor does not match the
 *                                    requirements passed to Create, a tensor read on the CPU is not
 *                                    host-accessible, detectionCount is outside [0, maxDet], or a
 *                                    custom waz/wel entry is not finite or has a complex magnitude
 *                                    above 1.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, e.g. updating its data flows.
 * @retval NVCV_ERROR_DEVICE           Submitting the PVA command failed.
 * @retval NVCV_SUCCESS                Operation submitted successfully.
 */
NVCVStatus pvaCaponROIPreprocessSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle const waz,
                                       NVCVTensorHandle const wel, NVCVTensorHandle const detectionCount,
                                       NVCVTensorHandle const detDopplers, NVCVTensorHandle const elementSnapshots,
                                       NVCVTensorHandle const rangeDopplerPlanes, NVCVTensorHandle const steeringLut,
                                       NVCVTensorHandle snapshotsROI, NVCVTensorHandle steeringVectorsROI);

/**
 * Submits the Capon ROI preprocessing (projection-core) operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle       Operator handle. Must not be NULL.
 * @param [in] stream       Valid CUDA stream.
 * @param [in] waz          Az twiddle table (see Create); may be NULL unless useCustomTwiddles is set.
 * @param [in] wel          El twiddle table (see Create); may be NULL unless useCustomTwiddles is set.
 * @param [in] detectionCount Number of detections to process (see Create).
 * @param [in] detDopplers  Per-detection detected Doppler bin (see Create).
 * @param [in] elementSnapshots Per-detection element snapshots for coarse DoA (see Create).
 * @param [in] rangeDopplerPlanes Per-detection range-Doppler planes (see Create).
 * @param [in] steeringLut Precomputed per-ROI-position steering LUT (see Create).
 * @param [out] snapshotsROI Output ROI beamspace snapshots (see Create).
 * @param [out] steeringVectorsROI Output per-detection steering vectors (see Create).
 *
 * @note detectionCount and custom waz/wel are read on the CPU during this call; see the file-level
 * note.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle or a required tensor is NULL, a tensor does not match the
 *                                    requirements passed to Create, a tensor read on the CPU is not
 *                                    host-accessible, detectionCount is outside [0, maxDet], or a
 *                                    custom waz/wel entry is not finite or has a complex magnitude
 *                                    above 1.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, e.g. updating its data flows.
 * @retval NVCV_ERROR_DEVICE           Submitting the PVA command failed.
 * @retval NVCV_SUCCESS                Operation submitted successfully.
 */
NVCVStatus pvaCaponROIPreprocessSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle const waz,
                                       NVCVTensorHandle const wel, NVCVTensorHandle const detectionCount,
                                       NVCVTensorHandle const detDopplers, NVCVTensorHandle const elementSnapshots,
                                       NVCVTensorHandle const rangeDopplerPlanes, NVCVTensorHandle const steeringLut,
                                       NVCVTensorHandle snapshotsROI, NVCVTensorHandle steeringVectorsROI);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPCAPONROIPREPROCESS_H */
