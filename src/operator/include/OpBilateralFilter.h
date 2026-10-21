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
 * @file OpBilateralFilter.h
 *
 * @brief Defines types and functions to handle the Template Matching operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_BILATERALFILTER BilateralFilter
 * @{
 */

#ifndef PVA_SOLUTIONS_OPBILATERALFILTER_H
#define PVA_SOLUTIONS_OPBILATERALFILTER_H

#include <PvaOperator.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/BorderType.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Constructs and an instance of the BilateralFilter operator.
 *
 * Limitations:
 *
 * Input tensor:
 *     Data Layout:    [CHW], [HWC], [NCHW], [NHWC] only when
 *     C:              [1]
 *     N:              [1]
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      8bit  Unsigned | Yes
 *      8bit  Signed   | No
 *      16bit Unsigned | No
 *      16bit Signed   | No
 *      32bit Unsigned | No
 *      32bit Signed   | No
 *      32bit Float    | No
 *      64bit Float    | No
 *
 * Output tensor:
 *     Data Layout:    [CHW], [HWC], [NCHW], [NHWC] only when
 *     C:              [1]
 *     N:              [1]
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      8bit  Unsigned | Yes
 *      8bit  Signed   | No
 *      16bit Unsigned | No
 *      16bit Signed   | No
 *      32bit Unsigned | No
 *      32bit Signed   | No
 *      32bit Float    | No
 *      64bit Float    | No
 *
 * Input/Output dependency
 *      Property      |  Input == Output
 *     -------------- | -------------
 *      Data Layout   | Yes
 *      Data Type     | Yes
 *      Number        | Yes
 *      Channels      | Yes
 *      Width         | Yes
 *      Height        | Yes
 *
 * Parameters
 *
 *      kernelSize = 3 or 5 or 7
 *      borderMode = NVCV_BORDER_CONSTANT or NVCV_BORDER_REPLICATE
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] tensorRequirements Pointer to the NVCVTensorRequirements structure which contains Tensor rank, shape, layout and data type information.
 *
 * @param [in] kernelSize The size (width and height) of filter kernel.
 *
 * @param [in] borderMode Border mode to be used when accessing elements outside input image.
 *
 * @param [in] borderValue Constant border value to be used when borderMode is NVCV_BORDER_CONSTANT. Ignored otherwise.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaBilateralFilterCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements *tensorRequirements,
                                    const int32_t kernelSize, const NVCVBorderType borderMode,
                                    const int32_t borderValue);

#ifdef __cplusplus
}

/**
 * Submits the BilateralFilter operator to a cuPVA stream.
 *
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] in Input tensor handle.
 *
 * @param [in] sigmaRange Standard deviation in color space.
 *                        Must be > 0.
 *
 * @param [in] sigmaSpace Standard deviation in the coordinate space.
 *                        Must be > 0.
 *
 * @param [out] out Output tensor handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaBilateralFilterSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle in,
                                    const float sigmaRange, const float sigmaSpace, NVCVTensorHandle out);

/**
 * Submits the BilateralFilter operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in Input tensor handle.
 * @param [in] sigmaRange Standard deviation in color space.
 * @param [in] sigmaSpace Standard deviation in the coordinate space.
 * @param [out] out Output tensor handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaBilateralFilterSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                    const float sigmaRange, const float sigmaSpace, NVCVTensorHandle out);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPBILATERALFILTER_H */