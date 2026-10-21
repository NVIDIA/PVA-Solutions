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
 * @file OpDoaCaponROI.h
 *
 * @brief Defines the Capon (MVDR) ROI DoA operator.
 *
 * This operator estimates the direction of arrival (DoA) for each beamspace ROI using Capon
 * beamforming. It reports up to three interior local peaks, ordered by descending absolute power.
 * For each peak, it returns the absolute power and the azimuth and elevation bin indices. Bin
 * indices are integer-valued when sub-bin interpolation is disabled and may be fractional when it
 * is enabled. An invalid peak is marked by PVA_DOA_CAPON_ROI_INVALID_PEAK_POWER (-1.0f) in the
 * peak-power output; valid peaks always have positive power. The bin indices of an invalid peak are
 * undefined and must be ignored.
 *
 * \b Limitations:
 *  - numROIs must be in the range [1, 4104].
 *  - svDim must be 16.
 *  - numAzBins must be 16.
 *  - numElBins must be 8.
 *  - numPeaksPerROI must be 3.
 *  - numDopplerBinsCovMat must be positive.
 *  - If numDopplerBinsCovMat is not divisible by 32, its remainder must be at least 12 and
 *    divisible by 4.
 *
 * @defgroup PVA_OPERATOR_ALGORITHM_RADAR_DOA_CAPON_ROI RadarCaponDoAROI (Direction of Arrival)
 * @{
 */

#ifndef PVA_SOLUTIONS_OPDOACAPONROI_H
#define PVA_SOLUTIONS_OPDOACAPONROI_H

#include <PvaOperator.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Maximum number of ROIs supported by the Capon ROI DoA operator. */
#define PVA_DOA_CAPON_ROI_MAX_ROIS (4104)

/**
 * Constructs the Capon ROI DoA operator.
 *
 * The operator allocates its own device scratch for the intermediate L_inv matrices passed between its
 * internal stages; this scratch is released by nvcvOperatorDestroy.
 *
 * \b Parameters
 *
 * @param [out] handle Pointer to storage for the operator instance handle.
 *                     + Must not be NULL.
 *
 * @param [in] snapshotsTensorRequirements Pointer to the NVCVTensorRequirements structure that describes the
 *                                         snapshots tensor.
 *                                         + Data Layout: [NHW]
 *                                         + Tensor Rank: 3
 *                                         + Tensor Shape: [numROIs, numDopplerBinsCovMat, svDim]
 *                                         + Data Type: NVCV_DATA_TYPE_2S32
 *                                         + The two S32 channels store the fixed-point real and imaginary
 *                                           components of each snapshot sample.
 *                                         + Each ROI must be contiguous; padding between ROIs is permitted.
 *
 *
 * @param [in] steeringVectorsTensorRequirements Pointer to the NVCVTensorRequirements structure that describes the
 *                                               steering-vector tensor.
 *                                               + Data Layout: [NCHW]
 *                                               + Tensor Rank: 4
 *                                               + Tensor Shape: [numROIs, svDim, numElBins, numAzBins]
 *                                               + Data Type: NVCV_DATA_TYPE_2S32
 *                                               + The two S32 channels store the fixed-point real and imaginary
 *                                                 components of each steering-vector element.
 *                                               + The tensor must be contiguous.
 *
 * @param [in] peakPowerTensorRequirements Pointer to the NVCVTensorRequirements structure that describes the
 *                                         peak-power output tensor.
 *                                         + Data Layout: [HW]
 *                                         + Tensor Rank: 2
 *                                         + Tensor Shape: [numROIs, numPeaksPerROI]
 *                                         + Data Type: NVCV_DATA_TYPE_F32
 *                                         + Each ROI's values must be contiguous; padding between ROIs is permitted.
 *
 * @param [in] azimuthBinIndicesTensorRequirements Pointer to the NVCVTensorRequirements structure that describes the
 *                                                 azimuth-bin-index output tensor.
 *                                                 + Data Layout: [HW]
 *                                                 + Tensor Rank: 2
 *                                                 + Tensor Shape: [numROIs, numPeaksPerROI]
 *                                                 + Data Type: NVCV_DATA_TYPE_F32
 *                                                 + Each ROI's values must be contiguous; padding between ROIs is
 *                                                   permitted.
 *
 * @param [in] elevationBinIndicesTensorRequirements Pointer to the NVCVTensorRequirements structure that describes
 *                                                   the elevation-bin-index output tensor.
 *                                                   + Data Layout: [HW]
 *                                                   + Tensor Rank: 2
 *                                                   + Tensor Shape: [numROIs, numPeaksPerROI]
 *                                                   + Data Type: NVCV_DATA_TYPE_F32
 *                                                   + Each ROI's values must be contiguous; padding between ROIs is
 *                                                     permitted.
 *
 * @param [in] params Pointer to the PVADoaCaponROIParams structure that contains the operator configuration.
 *                    + Must not be NULL.
 *                    + See \ref PVADoaCaponROIParams.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT     An argument is null or violates the operator requirements.
 * @retval NVCV_ERROR_OUT_OF_MEMORY        Not enough memory to create the operator.
 * @retval NVCV_SUCCESS                    Operator created successfully.
 */
NVCVStatus pvaDoaCaponROICreate(NVCVOperatorHandle *handle,
                                NVCVTensorRequirements const *const snapshotsTensorRequirements,
                                NVCVTensorRequirements const *const steeringVectorsTensorRequirements,
                                NVCVTensorRequirements const *const peakPowerTensorRequirements,
                                NVCVTensorRequirements const *const azimuthBinIndicesTensorRequirements,
                                NVCVTensorRequirements const *const elevationBinIndicesTensorRequirements,
                                PVADoaCaponROIParams const *const params);

#ifdef __cplusplus
}

/**
 * Submits the Capon ROI DoA operator to a cuPVA stream.
 *
 * All tensors must match the requirements passed to pvaDoaCaponROICreate.
 *
 * @param [in] handle Handle to the operator instance returned by pvaDoaCaponROICreate.
 *                    + Must not be NULL.
 *
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] snapshotsROI Input tensor handle containing the complex snapshot ROIs.
 *
 * @param [in] steeringVectorsROI Input tensor handle containing the complex steering vectors.
 *
 * @param [out] peakPower Output tensor handle containing the peak powers.
 *
 * @param [out] azimuthBinIndices Output tensor handle containing the azimuth bin indices.
 *
 * @param [out] elevationBinIndices Output tensor handle containing the elevation bin indices.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT An argument is null or a tensor does not match the operator configuration.
 * @retval NVCV_ERROR_NOT_COMPATIBLE    The handle does not refer to a Capon ROI operator.
 * @retval NVCV_ERROR_DEVICE           Command submission failed.
 * @retval NVCV_ERROR_INTERNAL         An internal operator error occurred.
 * @retval NVCV_SUCCESS                Operation submitted successfully.
 */
NVCVStatus pvaDoaCaponROISubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle const snapshotsROI,
                                NVCVTensorHandle const steeringVectorsROI, NVCVTensorHandle peakPower,
                                NVCVTensorHandle azimuthBinIndices, NVCVTensorHandle elevationBinIndices);

/**
 * Submits the Capon ROI DoA operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * All tensors must match the requirements passed to pvaDoaCaponROICreate.
 *
 * @param [in] handle Handle to the operator instance returned by pvaDoaCaponROICreate.
 *                    + Must not be NULL.
 *
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] snapshotsROI Input tensor handle containing the complex snapshot ROIs.
 *
 * @param [in] steeringVectorsROI Input tensor handle containing the complex steering vectors.
 *
 * @param [out] peakPower Output tensor handle containing the peak powers.
 *
 * @param [out] azimuthBinIndices Output tensor handle containing the azimuth bin indices.
 *
 * @param [out] elevationBinIndices Output tensor handle containing the elevation bin indices.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT An argument is null or a tensor does not match the operator configuration.
 * @retval NVCV_ERROR_NOT_COMPATIBLE    The handle does not refer to a Capon ROI operator.
 * @retval NVCV_ERROR_DEVICE           Command submission failed.
 * @retval NVCV_ERROR_INTERNAL         An internal operator error occurred.
 * @retval NVCV_SUCCESS                Operation submitted successfully.
 */
NVCVStatus pvaDoaCaponROISubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle const snapshotsROI,
                                NVCVTensorHandle const steeringVectorsROI, NVCVTensorHandle peakPower,
                                NVCVTensorHandle azimuthBinIndices, NVCVTensorHandle elevationBinIndices);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPDOACAPONROI_H */
