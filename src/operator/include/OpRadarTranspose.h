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
 * @file OpRadarTranspose.h
 *
 * @brief Defines types and functions to handle the RadarTranspose operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_RADAR_TRANSPOSE RadarTranspose
 * @{
 */

#ifndef PVA_SOLUTIONS_OPRADARTRANSPOSE_H
#define PVA_SOLUTIONS_OPRADARTRANSPOSE_H

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
 * Parameters for the RadarTranspose operator.
 */
typedef struct PVARadarTransposeParams
{
    /**
     * Target output layout for the transpose operation. Supported values are PVA_FFT_LAYOUT_RANGE_DOPPLER_RX and
     * PVA_FFT_LAYOUT_DOPPLER_RX_RANGE.
     *
     * PVA_FFT_LAYOUT_RANGE_RX_DOPPLER is the fixed input layout and is not supported as an output layout.
     */
    PVAFFTLayout outputLayout;
} PVARadarTransposeParams;

/**
 * Constructs an instance of the RadarTranspose operator.
 *
 * Transposes data from PVA_FFT_LAYOUT_RANGE_RX_DOPPLER to the requested output layout.
 *
 *   \b Limitations:
 *
 *      - nRange (range bins) and nDoppler (Doppler bins) must be in [1, 1024]. nRx (RX channels) must be in [1, 256].
 *      - Input and output tensors must be densely packed, with no padding between dimensions.
 *      - outputLayout must be PVA_FFT_LAYOUT_RANGE_DOPPLER_RX or PVA_FFT_LAYOUT_DOPPLER_RX_RANGE.
 *      - For PVA_FFT_LAYOUT_DOPPLER_RX_RANGE, nRange must be a multiple of 4, and a multiple of 8 above 512.
 *      - Some other combinations of nRange, nRx and nDoppler are not supported, for example all three at their
 *        maximum. Create rejects them with NVCV_ERROR_INVALID_ARGUMENT.
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] inTensorRequirements Pointer to the NVCVTensorRequirements structure which contains input Tensor
 *                                  layout rank, shape and data type information.
 *                                  + Must not be NULL.
 *
 *  \b Input:
 *      Data Layout:    [HCW], [NHCW] only when N = 1
 *          H: Number of range bins.
 *          C: Number of RX channels.
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
 * @param [in] outTensorRequirements Pointer to the NVCVTensorRequirements structure which contains output Tensor
 *                                   layout rank, shape and data type information.
 *                                   + Must not be NULL.
 *
 *  \b Output:
 *      Data Layout:    [HCW], [NHCW] only when N = 1
 *      If outputLayout is PVA_FFT_LAYOUT_RANGE_DOPPLER_RX:
 *          H: Number of range bins.
 *          C: Number of Doppler bins.
 *          W: Number of RX channels.
 *      If outputLayout is PVA_FFT_LAYOUT_DOPPLER_RX_RANGE:
 *          H: Number of Doppler bins.
 *          C: Number of RX channels.
 *          W: Number of range bins.
 *      The H, C and W dimensions must exactly match the permutation selected by outputLayout.
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
 *  \b Input/Output \b Dependency:
 *
 *    Property      | Input == Output
 *    --------------| -------------
 *    Data Layout   | Yes
 *    Data Type     | Yes
 *    Number        | Yes
 *    Channels      | No, permuted per outputLayout (see Output)
 *    Width         | No, permuted per outputLayout (see Output)
 *    Height        | No, permuted per outputLayout (see Output)
 *
 * @param [in] params Pointer to PVARadarTransposeParams.
 *                    + Must not be NULL.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT A required pointer is null, a tensor requirement does not use a supported
 *                                     layout, rank, shape or data type, or a parameter is invalid.
 * @retval NVCV_ERROR_OUT_OF_MEMORY    Not enough memory.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaRadarTransposeCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements const *const inTensorRequirements,
                                   NVCVTensorRequirements const *const outTensorRequirements,
                                   PVARadarTransposeParams const *const params);

#ifdef __cplusplus
}

/**
 * Submits the RadarTranspose operator to a cuPVA stream.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 *
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] in Input tensor handle.
 *
 * @param [out] out Output tensor handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null or not a RadarTranspose operator, a tensor base pointer is
 *                                     null, or a tensor does not match the requirements used to create the operator.
 * @retval NVCV_ERROR_INTERNAL         Updating the data flows or submitting to the stream failed.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaRadarTransposeSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle const in,
                                   NVCVTensorHandle out);

/**
 * Submits the RadarTranspose operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in Input tensor handle.
 * @param [out] out Output tensor handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null or not a RadarTranspose operator, a tensor base pointer is
 *                                     null, or a tensor does not match the requirements used to create the operator.
 * @retval NVCV_ERROR_INTERNAL         Updating the data flows or submitting to the stream failed.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaRadarTransposeSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle const in,
                                   NVCVTensorHandle out);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPRADARTRANSPOSE_H */
