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
 * @file OpDopplerFFT.h
 *
 * @brief Defines types and functions to handle the DopplerFFT operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_RADAR_DOPPLERFFT RadarDopplerFFT
 * @{
 */

#ifndef PVA_SOLUTIONS_OPDOPPLERFFT_H
#define PVA_SOLUTIONS_OPDOPPLERFFT_H

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
 * Constructs an instance of the DopplerFFT operator.
 *
 *   \b Limitations:
 *
 *      The number of chirps must be factorizable using only factors of 2, 3, 4, and 5, in no more than six FFT stages.
 *      If outputLayout is PVA_FFT_LAYOUT_DOPPLER_RX_RANGE: the number of chirps must be in the range [2, 1024].
 *      Otherwise: the number of chirps must be in the range [2, 1024).
 *      The number of Doppler bins can't be greater than the number of chirps.
 *      The number of output range bins and RX channels must equal those of the input.
 *      The number of RX channels must not exceed 256, and the number of range bins must not exceed 2048 * G, where G (1 to 8)
 *      is the number of range-bin groups the operator packs into one tile.
 *      The DMA line pitch limits (number of RX channels * number of range bins) to 8191 for a PVA_FFT_LAYOUT_DOPPLER_RX_RANGE
 *      input or output, the number of range bins to 8191 for a PVA_FFT_LAYOUT_RX_DOPPLER_RANGE input, and
 *      (number of RX channels * number of Doppler bins) to 8191 for a PVA_FFT_LAYOUT_RANGE_RX_DOPPLER output.
 *      The DMA advance limits (number of chirps * number of range bins) to 1048575 for a PVA_FFT_LAYOUT_RX_DOPPLER_RANGE
 *      input, and (number of range bins * number of Doppler bins) to 1048575 for a PVA_FFT_LAYOUT_RX_RANGE_DOPPLER output.
 *      If outputLayout is PVA_FFT_LAYOUT_RANGE_DOPPLER_RX: exactly 8 RX channels and the number of chirps in the range [32, 900].
 *      If inputLayout is PVA_FFT_LAYOUT_RANGE_RX_DOPPLER:
 *        - the number of chirps must be less than 1024 and the number of range bins a multiple of 8;
 *        - (number of RX channels * number of chirps) must not exceed 32767;
 *        - outputLayout must not be PVA_FFT_LAYOUT_RANGE_DOPPLER_RX.
 *      The input, window and output tensors must be densely packed, with no padding between dimensions.
 *      inputLayout must be PVA_FFT_LAYOUT_DOPPLER_RX_RANGE, PVA_FFT_LAYOUT_RANGE_RX_DOPPLER or PVA_FFT_LAYOUT_RX_DOPPLER_RANGE.
 *      outputLayout must be PVA_FFT_LAYOUT_DOPPLER_RX_RANGE, PVA_FFT_LAYOUT_RANGE_RX_DOPPLER, PVA_FFT_LAYOUT_RANGE_DOPPLER_RX
 *      or PVA_FFT_LAYOUT_RX_RANGE_DOPPLER.
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] inTensorRequirements Pointer to the NVCVTensorRequirements structure which contains input Tensor layout rank, shape and data type information.
 *
 *  \b Input:
 *      Data Layout:    [HCW, NHCW]
 *
 *      If inputLayout is PVA_FFT_LAYOUT_DOPPLER_RX_RANGE (the default):
 *          W: Number of range bins.
 *          C: Number of RX channels.
 *          H: Number of chirps.
 *
 *      If inputLayout is PVA_FFT_LAYOUT_RANGE_RX_DOPPLER:
 *          W: Number of chirps.
 *          C: Number of RX channels.
 *          H: Number of range bins.
 *
 *      If inputLayout is PVA_FFT_LAYOUT_RX_DOPPLER_RANGE:
 *          W: Number of range bins.
 *          C: Number of chirps.
 *          H: Number of RX channels.
 *
 *      Batches:        [1]
 *
 *      Rank:           3 for [HCW] or 4 for [NHCW].
 *
 *    Data Type        | Allowed
 *    ---------------- | -------------
 *    8bit  Unsigned   | No
 *    8bit  Signed     | No
 *    16bit Unsigned   | No
 *    16bit Signed     | No
 *    32bit Unsigned   | No
 *    32bit Signed     | No
 *    32bit Float      | No
 *    64bit Float      | No
 *    2x16bit Unsigned | No
 *    2x16bit Signed   | No
 *    2x32bit Unsigned | No
 *    2x32bit Signed   | Yes
 *
 * @param [in] winTensorRequirements Pointer to the NVCVTensorRequirements structure which contains window Tensor layout rank, shape and data type information.
 *
 *  \b Input:
 *      Data Layout:    [W, NW]
 *          W: Number of coefficients. Must be the same as the number of chirps in the input tensor.
 *
 *      Batches:        [1]
 *
 *      Rank:           1 for [W] or 2 for [NW].
 *
 *    Data Type        | Allowed
 *    ---------------- | -------------
 *    8bit  Unsigned   | No
 *    8bit  Signed     | No
 *    16bit Unsigned   | No
 *    16bit Signed     | No
 *    32bit Unsigned   | No
 *    32bit Signed     | Yes
 *    32bit Float      | No
 *    64bit Float      | No
 *    2x16bit Unsigned | No
 *    2x16bit Signed   | No
 *    2x32bit Unsigned | No
 *    2x32bit Signed   | No
 *
 * @param [in] outTensorRequirements Pointer to the NVCVTensorRequirements structure which contains output Tensor layout rank, shape and data type information.
 *
 *   \b Output:
 *      Data Layout:    [HCW], [NHCW] only when N = 1
 *      If outputLayout is PVA_FFT_LAYOUT_DOPPLER_RX_RANGE:
 *          H: Number of Doppler bins.
 *          C: Number of RX channels.
 *          W: Number of range bins.
 *      If outputLayout is PVA_FFT_LAYOUT_RANGE_RX_DOPPLER:
 *          H: Number of range bins.
 *          C: Number of RX channels.
 *          W: Number of Doppler bins.
 *      If outputLayout is PVA_FFT_LAYOUT_RANGE_DOPPLER_RX:
 *          H: Number of range bins.
 *          C: Number of Doppler bins.
 *          W: Number of RX channels.
 *      If outputLayout is PVA_FFT_LAYOUT_RX_RANGE_DOPPLER:
 *          H: Number of RX channels.
 *          C: Number of range bins.
 *          W: Number of Doppler bins.
 *
 *      Batches:        [1]
 *
 *      Rank:           3 for [HCW] or 4 for [NHCW] only when N = 1.
 *
 *    Data Type        | Allowed
 *    ---------------- | -------------
 *    8bit  Unsigned   | No
 *    8bit  Signed     | No
 *    16bit Unsigned   | No
 *    16bit Signed     | No
 *    32bit Unsigned   | No
 *    32bit Signed     | No
 *    32bit Float      | No
 *    64bit Float      | No
 *    2x16bit Unsigned | No
 *    2x16bit Signed   | No
 *    2x32bit Unsigned | No
 *    2x32bit Signed   | Yes
 *
 *   \b Input/Output \b Dependency:
 *
 *     Property      |  Input == Output
 *    -------------- | -------------
 *     Data Layout   | N/A
 *     Data Type     | Yes
 *     Number        | Yes
 *     Channels      | Yes
 *     Width         | N/A
 *     Height        | N/A
 *
 * @param [in] params Pointer to the DopplerFFT parameters. \ref PVADopplerFFTParams. Pass NULL to use the operator's default parameters
 *                    (Hanning window, PVA_FFT_LAYOUT_DOPPLER_RX_RANGE input, PVA_FFT_LAYOUT_RANGE_RX_DOPPLER output).
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT     Handle is null or some parameter is outside valid range.
 * @retval NVCV_ERROR_OUT_OF_MEMORY        Not enough memory to create the operator.
 * @retval NVCV_ERROR_INVALID_IMAGE_FORMAT Image format is invalid.
 * @retval NVCV_SUCCESS                    Operation executed successfully.
 */
NVCVStatus pvaDopplerFFTCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements const *const inTensorRequirements,
                               NVCVTensorRequirements const *const winTensorRequirements,
                               NVCVTensorRequirements const *const outTensorRequirements,
                               PVADopplerFFTParams const *const params);

#ifdef __cplusplus
}

/**
 * Submits the DopplerFFT operator to a cuPVA stream.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 *
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] in Input tensor handle.
 *
 * @param [in] win Window tensor handle.
 *
 * @param [out] out Output tensor handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaDopplerFFTSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle const in,
                               NVCVTensorHandle const win, NVCVTensorHandle out);

/**
 * Submits the DopplerFFT operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in Input tensor handle.
 * @param [in] win Window tensor handle.
 * @param [out] out Output tensor handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaDopplerFFTSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle const in,
                               NVCVTensorHandle const win, NVCVTensorHandle out);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPDOPPLERFFT_H */