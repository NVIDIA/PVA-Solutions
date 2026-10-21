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
 * @file OpHistogramEqualization.h
 *
 * @brief Defines types and functions to handle the Histogram Equalization operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_HISTOGRAMEQUALIZATION HistogramEqualization
 * @{
 */

#ifndef PVA_SOLUTIONS_OPHISTOGRAMEQUALIZATION_H
#define PVA_SOLUTIONS_OPHISTOGRAMEQUALIZATION_H

#include <PvaOperator.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/// @brief Parameters for Histogram Equalization operator.

/**
  * Constructs and an instance of the Histogram Equalization operator.
  * This operator finds a new mapping palette based on the original histogram of a grayscale image using the cumulative distribution function to transform the original pixel value to a new pixel value.
  *
  * \b Limitations:
  * This operator is designed to support only grayscale images based on their gray levels.
  * The input image width be at least 129 pixels.
  * The input image height be at least 17 pixels.
  *
  * \b Input:
  *    Data Layout:    [HWC]
  *
  *    Data Type      | Allowed
  *    -------------- | -------------
  *    8bit  Unsigned | Yes
  *    8bit  Signed   | No
  *    16bit Unsigned | No
  *    16bit Signed   | No
  *    32bit Unsigned | No
  *    32bit Signed   | No
  *    32bit Float    | No
  *    64bit Float    | No
  *
  *
  *  \b Output:
  *      Data Layout:    [HWC]
  *
  *
  *    Data Type      | Allowed
  *    -------------- | -------------
  *    8bit  Unsigned | Yes
  *    8bit  Signed   | No
  *    16bit Unsigned | No
  *    16bit Signed   | No
  *    32bit Unsigned | No
  *    32bit Signed   | No
  *    32bit Float    | No
  *    64bit Float    | No
  *
  *
  *   \b Input/Output \b Dependency:
  *
  *     Property      |  Input == Output
  *    -------------- | -------------
  *     Data Layout   | Yes
  *     Data Type     | Yes
  *     Number        | Yes
  *     Channels      | Yes
  *     Width         | Yes
  *     Height        | Yes
  *
  * @param [out] handle Where the operator instance handle will be written to.
  *                     + Must not be NULL.
  *
  * @param [in] tensorRequirements Pointer to the NVCVTensorRequirements structure which contains input Tensor layout rank, shape and data type information.
  *
  * @retval NVCV_ERROR_INVALID_ARGUMENT     Handle is null or tensorRequirements are not valid or outside a valid range.
  * @retval NVCV_ERROR_OUT_OF_MEMORY        Not enough memory to create the operator.
  * @retval NVCV_SUCCESS                    Operation executed successfully.
  */
NVCVStatus pvaHistogramEqualizationCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements *tensorRequirements);

#ifdef __cplusplus
}

/**
  * Submits the HistogramEqualization operator to a cuPVA stream.
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
  * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
  * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
  * @retval NVCV_SUCCESS                Operation executed successfully.
  */
NVCVStatus pvaHistogramEqualizationSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle in,
                                          NVCVTensorHandle out);

/**
 * Submits the HistogramEqualization operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in Input tensor handle.
 * @param [out] out Output tensor handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaHistogramEqualizationSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                          NVCVTensorHandle out);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPHISTOGRAMEQUALIZATION_H */