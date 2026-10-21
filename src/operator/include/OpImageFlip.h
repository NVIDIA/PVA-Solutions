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
 * @file OpImageFlip.h
 *
 * @brief Defines types and functions to handle the image flip operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_IMAGEFLIP ImageFlip
 * @{
 */

#ifndef PVA_SOLUTIONS_OPIMAGEFLIP_H
#define PVA_SOLUTIONS_OPIMAGEFLIP_H

#include <PvaOperator.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Image.h>
#include <nvcv/Status.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Constructs an instance of the ImageFlip operator.
  *
  * This operator flips an image horizontally, vertically, or both directions.
  *
  * \b Limitations:
  * - This operator is designed to support only grayscale images based on their gray levels.
  * - Input image and output image should be of U8 type. This includes the following formats:
  *      NVCV_IMAGE_FORMAT_U8, NVCV_IMAGE_FORMAT_Y8 or NVCV_IMAGE_FORMAT_Y8_ER.
  *
  * \b Input:
  *      Input image sizes and format are specified in imageRequirements.
  *
  *      Image Format   | Allowed
  *      -------------- | -------------
  *      U8             | Yes
  *      Y8             | Yes
  *      Y8_ER          | Yes
  *
  * \b Output:
  *      Output image has the same format and dimensions as the input image.
  *
  *      Image Format   | Allowed
  *      -------------- | -------------
  *      U8             | Yes
  *      Y8             | Yes
  *      Y8_ER          | Yes
  *
  * \b Input/Output \b Dependency:
  *      Property    | Input == Output
  *     -------------| -------------
  *      Format      | Yes
  *      Width       | Yes
  *      Height      | Yes
  *
  * \b Parameters
  *
  * @param [out] handle Where the operator instance handle will be written to.
  *                     + Must not be NULL.
  *
  * @param [in] imageRequirements Pointer to the NVCVImageRequirements structure which contains image width, height and format information.
  *
  * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null or imageRequirements are invalid.
  * @retval NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
  * @retval NVCV_SUCCESS                Operation executed successfully.
  */
NVCVStatus pvaImageFlipCreate(NVCVOperatorHandle *handle, NVCVImageRequirements *imageRequirements);

#ifdef __cplusplus
}

/**
  * Submits the ImageFlip operator to a cuPVA stream.
  *
  * @param [in] handle Handle to the operator.
  *                    + Must not be NULL.
  *
  * @param [in] stream Handle to a valid cuPVA stream.
  *
  * @param [in] in Input image handle.
  *
  * @param [out] out Output image handle.
  *
  * @param [in] flipDirection Direction of the flip operation.
  *                           + PVA_FLIP_HORIZONTAL: Flip horizontally (left-right)
  *                           + PVA_FLIP_VERTICAL: Flip vertically (up-down)
  *                           + PVA_FLIP_BOTH: Flip both directions (180° rotation)
  *
  * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null, images are invalid, or flipDirection is invalid.
  * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
  * @retval NVCV_SUCCESS                Operation executed successfully.
  */
NVCVStatus pvaImageFlipSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVImageHandle in, NVCVImageHandle out,
                              PVAFlipDirection flipDirection);

/**
 * Submits the ImageFlip operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in Input image handle.
 * @param [out] out Output image handle.
 * @param [in] flipDirection Direction of the flip operation.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null, images are invalid, or flipDirection is invalid.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaImageFlipSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageHandle in, NVCVImageHandle out,
                              PVAFlipDirection flipDirection);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPIMAGEFLIP_H */
