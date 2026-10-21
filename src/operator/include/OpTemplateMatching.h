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
 * @file OpTemplateMatching.h
 *
 * @brief Defines types and functions to handle the Template Matching operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_TEMPLATEMATCHING TemplateMatching
 * @{
 */

#ifndef PVA_SOLUTIONS_OPTEMPLATEMATCHING_H
#define PVA_SOLUTIONS_OPTEMPLATEMATCHING_H

#include <PvaOperator.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Constructs and an instance of the TemplateMatching operator.
 *
 * \b Limitations:
 *
 * Input image tensor:
 *      + If the input image tensor has a resolution W × H, then W >= 94 and H >= 94.
 *      + Data Layout:    [CHW], [HWC], [NCHW], [NHWC] only when
 *       - C:              [1]
 *       - N:              [1]
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
 * Input template tensor:
 *      + If the template image tensor has a resolution w × h, then 0 < w <= 31 and 0 < h <= 31.
 *      + The template is matched only over translations, so matching is neither rotation- nor
 *        scale-invariant. A rotated or resized instance of the template is not detected.
 *      + Data Layout:    [CHW], [HWC], [NCHW], [NHWC] only when
 *       - C:              [1]
 *       - N:              [1]
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
 *      + The output tensor is the normalized sum of squared differences (score) for every possible location of the
 *      template inside the input image tensor. The best match is the lowest score, and an exact match scores zero.
 *      + If the input image tensor has a resolution W × H and input template tensor has a resolution w × h, then the
 *      output image has a resolution (W - w + 1) × (H - h + 1).
 *      + Data Layout:    [CHW], [HWC], [NCHW], [NHWC] only when
 *       - C:              [1]
 *       - N:              [1]
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      8bit  Unsigned | No
 *      8bit  Signed   | No
 *      16bit Unsigned | No
 *      16bit Signed   | No
 *      32bit Unsigned | No
 *      32bit Signed   | No
 *      32bit Float    | Yes
 *      64bit Float    | No
 *
 * Input/Output dependency
 *      Property      |  Input == Output
 *     -------------- | -------------
 *      Data Layout   | Yes
 *      Data Type     | No
 *      Number        | Yes
 *      Channels      | Yes
 *      Width         | No
 *      Height        | No
 *
 *
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] inImageTensorRequirements Pointer to the NVCVTensorRequirements structure for input image tensor which
 * contains Tensor rank, shape, layout and data type information.
 *
 * @param [in] inTemplateTensorRequirements Pointer to the NVCVTensorRequirements structure for input template tensor
 * which contains Tensor rank, shape, layout and data type information.
 *
 * @param [in] outTensorRequirements Pointer to the NVCVTensorRequirements structure for output tensor which contains
 * Tensor rank, shape, layout and data type information.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaTemplateMatchingCreate(NVCVOperatorHandle *handle,
                                     NVCVTensorRequirements const *const inImageTensorRequirements,
                                     NVCVTensorRequirements const *const inTemplateTensorRequirements,
                                     NVCVTensorRequirements const *const outTensorRequirements);

#ifdef __cplusplus
}

/**
 * Submits the TemplateMatching operator to a cuPVA stream.
 *
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] inImage Input image tensor handle.
 *
 * @param [in] inTemplate Input template tensor handle.
 *
 * @param [out] out Output tensor handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaTemplateMatchingSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle inImage,
                                     NVCVTensorHandle inTemplate, NVCVTensorHandle out);

/**
 * Submits the TemplateMatching operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] inImage Input image tensor handle.
 * @param [in] inTemplate Input template tensor handle.
 * @param [out] out Output tensor handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaTemplateMatchingSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle inImage,
                                     NVCVTensorHandle inTemplate, NVCVTensorHandle out);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPTEMPLATEMATCHING_H */