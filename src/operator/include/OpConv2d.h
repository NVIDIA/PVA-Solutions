/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OpConv2d.h
 *
 * @brief Defines types and functions to handle the 2D convolution operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_CONV2D Conv2d
 *
 * Compatibility: Requires PVA SDK 2.6.0 and later.
 *
 * @{
 */

#ifndef PVA_SOLUTIONS_OPCONV2D_H
#define PVA_SOLUTIONS_OPCONV2D_H

#include <PvaOperator.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/BorderType.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Constructs an instance of the Conv2d operator.
 *
 * \b Limitations:
 *
 * \b Input:
 *      Data Layout:    [kHWC]
 *      Channels:       [1]
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      8bit  Unsigned | Yes
 *      8bit  Signed   | Yes
 *      16bit Unsigned | Yes
 *      16bit Signed   | Yes
 *      32bit Unsigned | No
 *      32bit Signed   | No
 *      32bit Float    | No
 *      64bit Float    | No
 *
 * \b Output:
 *      Data Layout:    [kHWC]
 *      Channels:       [1]
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      8bit  Unsigned | Yes
 *      8bit  Signed   | Yes
 *      16bit Unsigned | Yes
 *      16bit Signed   | Yes
 *      32bit Unsigned | No
 *      32bit Signed   | No
 *      32bit Float    | No
 *      64bit Float    | No
 *
 * \b Input/Output \b Dependency:
 *      Property    | Input == Output
 *     -------------| -------------
 *      Data Layout | Yes
 *      Data Type   | Yes
 *      Number      | Yes
 *      Channels    | Yes
 *      Width       | Yes
 *      Height      | Yes
 *
 * \b Kernel \b Coefficients:
 *      Property           | Requirement
 *     ------------------- | -------------
 *      Data Layout        | [HWC] where C=1, H=W (square tensor)
 *      Kernel Size        | 3x3, 5x5, or 7x7 only (Kernel size = H = W)
 *      Data Type Width    | Must match tensorRequirements data type width
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      8bit  Unsigned | Yes
 *      8bit  Signed   | Yes
 *      16bit Unsigned | Yes
 *      16bit Signed   | Yes
 *      32bit Unsigned | No
 *      32bit Signed   | No
 *      32bit Float    | No
 *      64bit Float    | No
 *
 * \b Parameters
 *
 *      borderMode = NVCV_BORDER_CONSTANT or NVCV_BORDER_REPLICATE
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] tensorRequirements Pointer to the NVCVTensorRequirements structure which contains Tensor rank, shape, layout and data type information.
 *
 * @param [in] borderMode Border mode to be used when accessing elements outside input image.
 *
 * @param [in] borderValue Constant border value to be used when borderMode is NVCV_BORDER_CONSTANT. Ignored otherwise.
 *
 * @param [in] kernelCoefficients Tensor handle containing the kernel coefficients. The kernel size is automatically extracted from the tensor dimensions.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaConv2dCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements *tensorRequirements,
                           const NVCVBorderType borderMode, int32_t borderValue, NVCVTensorHandle kernelCoefficients);

#ifdef __cplusplus
}

/**
 * Submits the Conv2d operator to a cuPVA stream.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] in Input tensor handle.
 *
 * @param [out] out Output tensor handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaConv2dSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out);

/**
 * Submits the Conv2d operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in Input tensor handle.
 * @param [out] out Output tensor handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaConv2dSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPCONV2D_H */