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
 * @file OpORBDescExtractor.h
 *
 * @brief Defines types and functions to handle the ORB descriptor extractor operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_ORBDESCEXTRACTOR ORBDescExtractor
 * @{
 */

#ifndef PVA_SOLUTIONS_OPORBDESCEXTRACTOR_H
#define PVA_SOLUTIONS_OPORBDESCEXTRACTOR_H

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

/** Constructs an instance of the ORBDescExtractor operator.
 * \b Limitations:
 *      Only support feature points from a single image.
 *
 * \b Input \b Image:
 *      Input image sizes and formats are specified in inImageRequirements.
 *
 *      Image Format   | Allowed
 *      -------------- | -------------
 *      U8             | Yes
 *      Y8             | Yes
 *      Y8_ER          | Yes
 *      S8             | Yes
 *      U16            | Yes
 *      Y16            | Yes
 *      Y16_ER         | Yes
 *      S16            | Yes
 *
 * \b Border \b Type:
 *      Border Type        | Allowed
 *      ------------------ | -------------
 *      Constant           | Yes
 *      Clamp              | Yes
 *      Mirror             | No
 *      Reflect            | No
 *      Limited            | No
 *
 * \b Parameters:
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] inImageRequirements Pointer to the input image NVCVImageRequirements structure which contains
 *                                   image shape, strides, and format information.
 *
 * @param[in] inCornersTensorRequirements Pointer to the input corners NVCVTensorRequirements structure which contains
 *                                          tensor rank, shape, layout and data type information. Corner coordinates
 *                                          are in XY-interleaved format.
 *
 *      Data Layout:    [HW]
 *      Data Type:      NVCV_DATA_TYPE_F32
 *      Height:         Number of features
 *      Width:          2
 *
 * @param[in] outDescriptorsTensorRequirements Pointer to the output descriptors NVCVTensorRequirements.
 *                                             2D tensor with 2nd dimension size \ref PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH.
 *
 *      Data Layout:    [HW]
 *      Data Type:      NVCV_DATA_TYPE_U8
 *      Height:         Number of features
 *      Width:          PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH
 *
 * @param[in] disableRBRIEFFlag Binary flag to control whether to use the rotationally-invariant BRIEF algorithm.
 *                  + Valid values are as follows:
 *                    -  0: Default, enable rotationally-invariant BRIEF.
 *                    -  1 : Disable rotationally-invariant BRIEF.
 *
 * @param [in] borderType Border type to use.
 *
 * @param [in] borderValue Border value to use for NVCV_BORDER_CONSTANT.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval NVCV_SUCCESS
 */
NVCVStatus pvaORBDescExtractorCreate(NVCVOperatorHandle *handle, const NVCVImageRequirements *inImageRequirements,
                                     const NVCVTensorRequirements *inCornersTensorRequirements,
                                     const NVCVTensorRequirements *outDescriptorsTensorRequirements,
                                     uint32_t disableRBRIEFFlag, NVCVBorderType borderType, int32_t borderValue);

#ifdef __cplusplus
}

/**
 * Submits the ORBDescExtractor operator to a cuPVA stream.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] in Input image handle.
 *
 * @param[in] inCornersTensorHandle Input corners tensor handle.
 *
 * @param[out] outDescriptorsTensorHandle Output descriptors tensor handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaORBDescExtractorSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVImageHandle in,
                                     NVCVTensorHandle inCornersTensorHandle,
                                     NVCVTensorHandle outDescriptorsTensorHandle);

/**
 * Submits the ORBDescExtractor operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in Input image handle.
 * @param [in] inCornersTensorHandle Input corners tensor handle.
 * @param [out] outDescriptorsTensorHandle Output descriptors tensor handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaORBDescExtractorSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageHandle in,
                                     NVCVTensorHandle inCornersTensorHandle,
                                     NVCVTensorHandle outDescriptorsTensorHandle);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPORBDESCEXTRACTOR_H */