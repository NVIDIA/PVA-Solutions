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
 * @file OpGaussianPyramid.h
 *
 * @brief Defines types and functions to handle the GaussianPyramid operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_GAUSSIANPYRAMID GaussianPyramid
 * @{
 */

#ifndef PVA_SOLUTIONS_OPGAUSSIANPYRAMID_H
#define PVA_SOLUTIONS_OPGAUSSIANPYRAMID_H

#include <PvaOperator.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/BorderType.h>
#include <nvcv/Image.h>
#include <nvcv/Status.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructs an instance of the GaussianPyramid operator.
 *
 * Limitations:
 *
 *   Input/Output:
 *      Image format             | Allowed
 *      ------------------------ | -------
 *      NVCV_IMAGE_FORMAT_U8     | Yes
 *      NVCV_IMAGE_FORMAT_S8     | Yes
 *      NVCV_IMAGE_FORMAT_Y8     | Yes
 *      NVCV_IMAGE_FORMAT_Y8_ER  | Yes
 *      NVCV_IMAGE_FORMAT_U16    | Yes
 *      NVCV_IMAGE_FORMAT_S16    | Yes
 *      NVCV_IMAGE_FORMAT_Y16    | Yes
 *      NVCV_IMAGE_FORMAT_Y16_ER | Yes
 *
 *      The order of returned output image pyramid is from high to low resolution (fine to coarse).
 *      Each pyramid level has half the width and height of the previous level (downsampled by a factor of 2).
 *
 *      Image size constraints:
 *      - The minimum supported width and height of input image is 2x4 pixels.
 *      - Image height modulo tile height cannot be 1.
 *
 *      The function will return an error if the input image is too small or violates the height constraint.
 *
 *  Gaussian kernel:
 *      The Gaussian kernel size only supports 5x5.
 *      The implemented 5x5 Gaussian kernel is:
 *          |  1   4   6   4   1  |
 *          |  4  16  24  16   4  |
 *          |  6  24  36  24   6  |
 *          |  4  16  24  16   4  |
 *          |  1   4   6   4   1  |
 *
 *  Parameters:
 *     numLevels:
 *      - The supported range of pyramid levels is 2-3.
 *      - Level 0 is a copy of the input image.
 *      - For numLevels=2: returns level 0 (input copy) and level 1 (downsampled).
 *      - For numLevels=3: returns level 0 (input copy), level 1, and level 2 (both downsampled).
 *
 *     borderMode:
 *      - NVCV_BORDER_CONSTANT
 *
 *     borderValue:
 *      - Must be 0.
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] inputImageRequirements Pointer to the NVCVImageRequirements structure which contains input image width, height and format information.
 *
 * @param [in] numLevels Number of pyramid levels to compute.
 *
 * @param [in] borderMode Border mode to be used when accessing elements outside input image.
 *
 * @param [in] borderValue Constant border value to be used when borderMode is NVCV_BORDER_CONSTANT.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT     Handle is null or some parameter is outside valid range.
 * @retval NVCV_ERROR_OUT_OF_MEMORY        Not enough memory to create the operator.
 * @retval NVCV_ERROR_INVALID_IMAGE_FORMAT Image format is invalid.
 * @retval NVCV_SUCCESS                    Operation executed successfully.
 */
NVCVStatus pvaGaussianPyramidCreate(NVCVOperatorHandle *handle,
                                    NVCVImageRequirements const *const inputImageRequirements, const int32_t numLevels,
                                    const NVCVBorderType borderMode, const int32_t borderValue);

#ifdef __cplusplus
}

/**
 * Submits the GaussianPyramid operator to a cuPVA stream.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 *
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] inImageHandle Input image handle.
 *
 * @param [out] outImageHandles Pointer to array of output images.
 *                              + The size of the array is equal to numLevels.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaGaussianPyramidSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVImageHandle inImageHandle,
                                    NVCVImageHandle *outImageHandles);

/**
 * Submits the GaussianPyramid operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] inImageHandle Input image handle.
 * @param [out] outImageHandles Pointer to array of output images.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaGaussianPyramidSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageHandle inImageHandle,
                                    NVCVImageHandle *outImageHandles);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPGAUSSIANPYRAMID_H */