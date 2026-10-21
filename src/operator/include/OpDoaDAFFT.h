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
 * @file OpDoaDAFFT.h
 *
 * @brief Defines types and functions for the DoA DA FFT (phase-compensated dual-aperture FFT) operator.
 * @defgroup PVA_RADAR_OPERATOR_ALGORITHM_RADAR_DOA_DA_FFT RadarDoaDaFFT
 * @{
 */

#ifndef PVA_SOLUTIONS_OP_DOA_DA_FFT_H
#define PVA_SOLUTIONS_OP_DOA_DA_FFT_H

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
 * Constructs and an instance of the DOA target processing operator.
 *
 * \b Limitations:
 *  - The FFT size for DOA processing is fixed to 256.
 *  - The calibration vector is fixed to 16 complex elements.
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     Must not be NULL.
 *
 * @param [in] inTensorRequirements Pointer to the NVCVTensorRequirements structure array which contains input Tensor layout rank, shape and data type information.
 *
 *    NVCVTensorRequirement | dtype               | rank | layout | shape
 *    -------------------   | ------------------- | ---- | ------ | -----------
 *    0                     | NVCV_DATA_TYPE_U32  | 2   | W       | [1]
 *    1                     | NVCV_DATA_TYPE_2S32 | 2   | HW      | [1, PVA_RADAR_NUM_TOTAL_ANTENNA_ELEMENTS]
 *    2                     | NVCV_DATA_TYPE_U32  | 2   | HW      | [PVA_RADAR_PEAKDET_NUM_PEAK_INDICES, PVA_RADAR_MAX_TARGET_COUNT]
 *    3                     | NVCV_DATA_TYPE_2S32 | 2   | HW      | [PVA_RADAR_MAX_TARGET_COUNT, PVA_RADAR_NUM_TOTAL_ANTENNA_ELEMENTS]
 *    4                     | NVCV_DATA_TYPE_U32  | 2   | HW      | [Nb, Nr/PVA_RADAR_DOPPLER_FOLD_COUNT]
 *
 *    PVA_RADAR_NUM_TOTAL_ANTENNA_ELEMENTS: Number of total antenna elements.
 *    PVA_RADAR_PEAKDET_NUM_PEAK_INDICES: Number of peak indices (3 values: range bin index, folded doppler bin index, and unfolded doppler bin index).
 *    PVA_RADAR_MAX_TARGET_COUNT: Maximum possible number of input peaks. Should be less than or equal to DOA_MAX_TARGET_COUNT.
 *    Nb: Number of range bins.
 *    Nr: Number of ramps or chirps.
 *    PVA_RADAR_DOPPLER_FOLD_COUNT: Number of doppler folds.
 *
 *    Tensor 0: Number of detected peaks.
 *    Tensor 1: calibration vector.
 *    Tensor 2: peak indices.
 *    Tensor 3: peak snap data.
 *    Tensor 4: final NCI data.
 *
 * @param [in] outTensorRequirements Pointer to the NVCVTensorRequirements structure which contains output Tensor layout rank, shape and data type information.
 *
 *   \b Output:
 *      Data Layout:    [HW]
 *      W:  Maximum possible number of peaks to be detected.
 *      H:  Number of target detection properties. Currently DOA supports 7 target detection properties.
 *
 *
 *      Batches:        [1]
 *      Rank:           2 for [HW].
 *
 *    Tensor 0: DOA output.
 *
 *    Data Type        | Allowed
 *    ---------------- | -------------
 *    8bit  Unsigned   | No
 *    8bit  Signed     | No
 *    16bit Unsigned   | No
 *    16bit Signed     | No
 *    32bit Unsigned   | No
 *    32bit Signed     | No
 *    32bit Float      | Yes
 *    64bit Float      | No
 *    2x32bit Unsigned | No
 *    2x32bit Signed   | No
 *
 *   \b Output \b Tensor \b Layout:
 *
 *   The output tensor contains 7 rows of target detection properties, with each row containing
 *   up to W (maximum number of peaks) target values:
 *
 *    Row Index | Property         | Unit  | Description
 *    --------- | ---------------- | ----- | -----------
 *    0         | Velocity         | km/h  | Target radial velocity (positive: moving away, negative: approaching)
 *    1         | Range            | m     | Target distance from radar
 *    2         | Azimuth Angle    | deg   | Horizontal angle (-90° to +90°, 0° is boresight)
 *    3         | Elevation Angle  | deg   | Vertical angle relative to horizontal plane
 *    4         | X Coordinate     | m     | Lateral position (X = Range × sin(Azimuth))
 *    5         | Y Coordinate     | m     | Longitudinal position (Y = Range × cos(Azimuth))
 *    6         | Z Coordinate     | m     | Vertical position (Z = Range × sin(Elevation))
 *
 *   Example: For a tensor with shape [7, 128], the first 128 float values are velocities for up to 128 targets,
 *   the next 128 values are ranges, and so on.
 *
 * @param [in] inTensorCount Number of input tensors.
 * @param [in] GP Pointer to global radar parameter structure.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT     Handle is null or some parameter is outside valid range.
 * @retval NVCV_ERROR_OUT_OF_MEMORY        Not enough memory to create the operator.
 * @retval NVCV_ERROR_INVALID_IMAGE_FORMAT Image format is invalid.
 * @retval NVCV_SUCCESS                    Operation executed successfully.
 */
NVCVStatus pvaDoaDAFFTCreate(NVCVOperatorHandle *handle, const NVCVTensorRequirements **inTensorRequirements,
                             const NVCVTensorRequirements *outTensorRequirements, int inTensorCount,
                             const PVARadarGP *GP);

#ifdef __cplusplus
}

/**
 * Submits the DOA operator to a cuPVA stream.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 *
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] inHandles Pointer to array of input NVCVTensorHandle.
 *
 * @param [out] outHandle Output tensor handle.
 *
 * @param[in] inTensorCount Number of input tensors.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaDoaDAFFTSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle *inHandles,
                             NVCVTensorHandle outHandle, int inTensorCount);

/**
 * Submits the DOA operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 *
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] inHandles Pointer to array of input NVCVTensorHandle.
 *
 * @param [out] outHandle Output tensor handle.
 *
 * @param[in] inTensorCount Number of input tensors.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaDoaDAFFTSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle *inHandles,
                             NVCVTensorHandle outHandle, int inTensorCount);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OP_DOA_DA_FFT_H */
