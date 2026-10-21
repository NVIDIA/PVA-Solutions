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
 * @file OpMedianFilter.h
 *
 * @brief Defines types and functions to handle the median filter operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_MEDIANFILTER MedianFilter
 * @{
 */

#ifndef PVA_SOLUTIONS_OPMEDIANFILTER_H
#define PVA_SOLUTIONS_OPMEDIANFILTER_H

#include <PvaOperator.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/BorderType.h>
#include <nvcv/Image.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Constructs an instance of the MedianFilter operator.
 *
 * \b Limitations:
 *
 * \b Input:
 *      Image sizes and formats are specified in inImageRequirements.
 *
 *      Image Format   | Allowed
 *      -------------- | -------------
 *      U8             | Yes
 *      S8             | Yes
 *      U16            | Yes
 *      S16            | Yes
 *      Y8             | Yes
 *      Y8_ER          | Yes
 *      Y16            | Yes
 *      Y16_ER         | Yes
 *      U32            | Yes
 *      S32            | Yes
 *      2S16           | Yes
 *
 * \b Output:
 *      Image sizes and formats are specified in outImageRequirements.
 *
 * \b Input/Output \b Dependency:
 *      Property    | Input == Output
 *     -------------| -------------
 *       Width      |  Yes
 *       Height     |  Yes
 *       Format     |  Yes
 *
 * \b Kernel \b Size:
 *      Kernel Size   | Allowed
 *      ------------- | -------------
 *      3x3           |  Yes
 *      5x5           |  Yes
 *      5x7           |  Yes
 *      7x7           |  Yes
 *      11x11         |  No
 *
 * \b Kernel \b Type:
 *      Kernel Type     | Allowed
 *      --------------- | -------------
 *      Box             |  Yes
 *      Cross           |  No
 *      Checkered       |  No
 *
 * \b Border \b Type:
 *      Border Type        | Allowed
 *      ------------------ | -------------
 *      Zero               | Yes
 *      Clamp              | Yes
 *      Mirror             | No
 *      Reflect            | No
 *      Limited            | No
 *
 * \b Parameters
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] inImageRequirements Pointer to the input image NVCVImageRequirements structure which contains image shape,
 *                                 strides, and format information.
 *
 * @param [in] kernelRequirements Pointer to the kernel tensor NVCVTensorRequirements structure which contains tensor layout, rank,
 *                                 shape, and strides information.
 *                                 + Data Layout:    [HW]
 *                                 + Data Type:      NVCV_DATA_TYPE_S8
 *
 * @param [in] outImageRequirements Pointer to the output image NVCVImageRequirements structure which contains image shape,
 *                                  strides, and format information.
 *
 * @param [in] borderType Border type to use.
 *
 * @param [in] borderValue Border value to use for NVCV_BORDER_CONSTANT.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaMedianFilterCreate(NVCVOperatorHandle *handle, NVCVImageRequirements *inImageRequirements,
                                 NVCVTensorRequirements *kernelRequirements,
                                 NVCVImageRequirements *outImageRequirements, NVCVBorderType borderType,
                                 int32_t borderValue);

#ifdef __cplusplus
}

/**
 * Submits the MedianFilter operator to a cuPVA stream.
 *
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] in Input image handle.
 *
 * @param [in] kernel Kernel data tensor handle.
 *
 * @param [out] out Output image handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaMedianFilterSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVImageHandle in,
                                 NVCVTensorHandle kernel, NVCVImageHandle out);

/**
 * Submits the MedianFilter operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in Input image handle.
 * @param [in] kernel Kernel data tensor handle.
 * @param [out] out Output image handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaMedianFilterSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageHandle in,
                                 NVCVTensorHandle kernel, NVCVImageHandle out);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPMEDIANFILTER_H */