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
 * @file OpRangeFFT.h
 *
 * @brief Defines types and functions to handle the RangeFFT operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_RADAR_RANGEFFT RadarRangeFFT
 * @{
 */

#ifndef PVA_SOLUTIONS_OPRANGEFFT_H
#define PVA_SOLUTIONS_OPRANGEFFT_H

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
 * Constructs an instance of the RangeFFT operator.
 *
 *   \b Limitations:
 *
 *      For real input (S32): the supported number of samples must be in the range [4, 1024], must be even and must be factorizable by only factors of 2, 3, 4, and 5
 *      For complex input (2S32): the supported number of samples must be in the range [2, 512] and must be factorizable by only factors of 2, 3, 4, and 5.
 *      For real input (S32): the number of range bins can't be greater than (the number of samples / 2 + 1).
 *      For complex input (2S32): the number of range bins can't be greater than the number of samples.
 *      If inputLayout or outputLayout is PVA_FFT_LAYOUT_RX_DOPPLER_RANGE, or outputLayout is PVA_FFT_LAYOUT_RANGE_RX_DOPPLER: the number of chirps must be a multiple of 8.
 *      If inputLayout is PVA_FFT_LAYOUT_RX_DOPPLER_RANGE: (number of RX channels * number of chirps) must not exceed 2048.
 *      G below (1 to 8) follows from the arguments: it is the largest value, up to the maximum for the number of samples below, for
 *      which the number of chirps is a multiple of 8 * G and, for a PVA_FFT_LAYOUT_RX_DOPPLER_RANGE output with
 *      (number of RX channels * number of chirps) above 2048 * G, the conditions listed for that output hold. The maximum is:
 *        - real input: 8 for 4 to 108 samples, 7 for 120 to 144, 6 for 150 to 160, 5 for 162 (6 with a PVA_FFT_LAYOUT_RANGE_RX_DOPPLER
 *          output), 5 for 180 to 192, 4 for 200 to 240, 3 for 250 to 324, 2 for 360 to 486, and 1 for 500 to 1024;
 *        - complex input: 8 for 2 to 54 samples, 7 for 60 to 72, 6 for 75 to 81, 5 for 90 to 96, 4 for 100 to 120, 3 for 125 to 162,
 *          2 for 180 to 243, and 1 for 250 to 512.
 *      If inputLayout or outputLayout is PVA_FFT_LAYOUT_DOPPLER_RX_RANGE: at most 256 RX channels, and the number of chirps must not
 *        exceed 2048 * G.
 *      If inputLayout is PVA_FFT_LAYOUT_DOPPLER_RX_RANGE: (number of RX channels * number of samples) must not exceed 65535 for real
 *        input and 32767 for complex input.
 *      If outputLayout is PVA_FFT_LAYOUT_DOPPLER_RX_RANGE: (number of RX channels * number of range bins) must not exceed 32767.
 *      If outputLayout is PVA_FFT_LAYOUT_RANGE_RX_DOPPLER: the input must be real (S32), and (number of RX channels * number of chirps)
 *        must not exceed 2048 * G.
 *      If outputLayout is PVA_FFT_LAYOUT_RX_DOPPLER_RANGE and (number of RX channels * number of chirps) exceeds 2048 * G:
 *        - the number of range bins must be the maximum: (the number of samples / 2 + 1) for real input, the number of samples for complex input;
 *        - for real input the number of samples must be a multiple of 16; for complex input it must be one more than a multiple of 8;
 *        - (number of RX channels * number of chirps / (8 * G)) must be the product of an even number and another number, both no greater
 *          than 256, with the even number times (64 * G * number of range bins) no greater than 8388607.
 *      The input, window and output tensors must be densely packed, with no padding between dimensions.
 *      inputLayout must be PVA_FFT_LAYOUT_DOPPLER_RX_RANGE or PVA_FFT_LAYOUT_RX_DOPPLER_RANGE.
 *      outputLayout must be PVA_FFT_LAYOUT_DOPPLER_RX_RANGE, PVA_FFT_LAYOUT_RX_DOPPLER_RANGE or PVA_FFT_LAYOUT_RANGE_RX_DOPPLER.
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
 *          W: Number of samples.
 *          C: Number of RX channels.
 *          H: Number of chirps.
 *
 *      If inputLayout is PVA_FFT_LAYOUT_RX_DOPPLER_RANGE:
 *          W: Number of samples.
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
 *    32bit Signed     | Yes  (real input)
 *    32bit Float      | No
 *    64bit Float      | No
 *    2x16bit Unsigned | No
 *    2x16bit Signed   | No
 *    2x32bit Unsigned | No
 *    2x32bit Signed   | Yes  (complex input)
 *
 * @param [in] winTensorRequirements Pointer to the NVCVTensorRequirements structure which contains window Tensor layout rank, shape and data type information.
 *
 *  \b Input:
 *      Data Layout:    [W, NW]
 *          W: Number of coefficients. Must be the same as the number of samples in the input tensor.
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
 *      Data Layout:    [HCW, NHCW]
 *
 *      If outputLayout is PVA_FFT_LAYOUT_DOPPLER_RX_RANGE (the default):
 *          W: Number of range bins.
 *          C: Number of RX channels.
 *          H: Number of chirps.
 *
 *      If outputLayout is PVA_FFT_LAYOUT_RX_DOPPLER_RANGE:
 *          W: Number of range bins.
 *          C: Number of chirps.
 *          H: Number of RX channels.
 *
 *      If outputLayout is PVA_FFT_LAYOUT_RANGE_RX_DOPPLER (real input only):
 *          W: Number of chirps.
 *          C: Number of RX channels.
 *          H: Number of range bins.
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
 *   \b Input/Output \b Dependency:
 *
 *     Property      |  Input == Output
 *    -------------- | -------------
 *     Data Layout   | Yes
 *     Data Type     | No
 *     Number        | Yes
 *     Channels      | Layout-dependent (see below)
 *     Width         | N/A
 *     Height        | Layout-dependent (see below)
 *
 *   The input and output always have the same number of RX channels and chirps, but the tensor dimension
 *   that carries each depends on the layout, so C and H are equal only when both ends put them in the same
 *   place. inputLayout and outputLayout are independent: a PVA_FFT_LAYOUT_RX_DOPPLER_RANGE input may produce
 *   a PVA_FFT_LAYOUT_DOPPLER_RX_RANGE output and vice versa.
 *
 * @param [in] params Pointer to the RangeFFT parameters. \ref PVARangeFFTParams.
 *
 *      windowType:   PVA_BATCH_FFT_WINDOW_HANNING or PVA_BATCH_FFT_WINDOW_USER_DEFINED.
 *      inputLayout:  PVA_FFT_LAYOUT_DOPPLER_RX_RANGE (default) or PVA_FFT_LAYOUT_RX_DOPPLER_RANGE.
 *      outputLayout: PVA_FFT_LAYOUT_DOPPLER_RX_RANGE (default), PVA_FFT_LAYOUT_RX_DOPPLER_RANGE or
 *                    PVA_FFT_LAYOUT_RANGE_RX_DOPPLER (real input only).
 *
 *      A NULL params pointer selects the defaults, which are the first enumerator of each field.
 *      Layouts other than those listed above are not supported by this operator.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT     Handle is null or some parameter is outside valid range.
 * @retval NVCV_ERROR_OUT_OF_MEMORY        Not enough memory to create the operator.
 * @retval NVCV_ERROR_INVALID_IMAGE_FORMAT Image format is invalid.
 * @retval NVCV_SUCCESS                    Operation executed successfully.
 */
NVCVStatus pvaRangeFFTCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements const *const inTensorRequirements,
                             NVCVTensorRequirements const *const winTensorRequirements,
                             NVCVTensorRequirements const *const outTensorRequirements,
                             PVARangeFFTParams const *const params);

#ifdef __cplusplus
}

/**
 * Submits the RangeFFT operator to a cuPVA stream.
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
NVCVStatus pvaRangeFFTSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle const in,
                             NVCVTensorHandle const win, NVCVTensorHandle out);

/**
 * Submits the RangeFFT operator to a CUDA stream.
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
NVCVStatus pvaRangeFFTSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle const in,
                             NVCVTensorHandle const win, NVCVTensorHandle out);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPRANGEFFT_H */