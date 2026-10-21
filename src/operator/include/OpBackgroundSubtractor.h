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
 * @file OpBackgroundSubtractor.h
 *
 * @brief Defines types and functions to handle the background subtraction operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_BACKGROUNDSUBTRACTOR BackgroundSubtractor
 * @{
 */

#ifndef PVA_SOLUTIONS_OPBACKGROUNDSUBTRACTOR_H
#define PVA_SOLUTIONS_OPBACKGROUNDSUBTRACTOR_H

#include <PvaOperator.h>
#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Image.h>
#include <nvcv/Status.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Constructs an instance of the BackgroundSubtractor operator.
 *
 * \b Limitations:
 *      Input image and output background image should be of U8 type. This includes the following formats:
 *      NVCV_IMAGE_FORMAT_U8, NVCV_IMAGE_FORMAT_Y8 or NVCV_IMAGE_FORMAT_Y8_ER.
 *
 * \b Input:
 *      Input is an image in U8 format.
 *
 * \b Outputs:
 *      There are two outputs:
 *      - Output mask in U8 format. When shadow detection is disabled, output is binary with 0 indicating
 *      background and 255 indicating foreground. When shadow detection is enabled, output is a mask with 0 indicating
 *      background, shadowPixelValue indicating shadow, and 255 indicating foreground.
 *      - Optional output background image in the same format as the input image.
 *
 * \b Input/Output \b Dependency:
 *      Property    | Input == Output
 *     -------------| -------------
 *      Data Layout | Yes
 *      Data Type   | Yes
 *      Channels    | Yes
 *      Width       | Yes
 *      Height      | Yes
 *
 * \b Parameters
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] imageRequirements Pointer to the NVCVImageRequirements structure which contains image shape,
 *                               and format information. This format will be applicable for both the input image
 *                               and the output background image. This can be NVCV_IMAGE_FORMAT_U8, NVCV_IMAGE_FORMAT_Y8
 *                               or NVCV_IMAGE_FORMAT_Y8_ER.

 * @param [in] maskRequirements Pointer to the background mask output NVCVImageRequirements structure.
 * @param [in] gmmDataType The data type used for the GMM model parameters (mean, variance, weights).
 *                         - PVA_GMM_DATA_TYPE_FP32: Use 32-bit floating point for GMM parameters
 *                         - PVA_GMM_DATA_TYPE_FP16: Use 16-bit floating point for GMM parameters
 *                         PVA_GMM_DATA_TYPE_FP32 yields higher background modeling accuracy but results in longer execution time.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval NVCV_SUCCESS                Operation executed successfully.
*/
NVCVStatus pvaBackgroundSubtractorCreate(NVCVOperatorHandle *handle, NVCVImageRequirements *imageRequirements,
                                         NVCVImageRequirements *maskRequirements, PVAGMMDataType gmmDataType);

#ifdef __cplusplus
}

/**
 * Submits the BackgroundSubtractor operator to a cuPVA stream.
 *
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] in Input image handle.
 *
 * @param [out] outMask Output mask handle. Every pixel is 0 or 255 or shadowPixelValue (if shadow detection is enabled).
 * 0 indicates background, 255 indicates foreground, and shadowPixelValue indicates shadow.
 *
 * @param [out] outBackgroundImage Optional output background image handle. If background image is not required, pass NULL.
 *
 * @param [in] learningRate Learning rate that controls how fast the background model adapts to changes.
 * - Range: [0.0, 1.0]
 * - A value of 0.0 means the background model is static (no updates)
 * - A value of 1.0 means the model is completely reinitialized from the current frame
 * - Lower values result in a more stable background model that adapts slowly
 * - Higher values allow the model to adapt quickly to changes but may be more sensitive to noise
 * Setting the learning rate to 1.0 will reset the gaussian models.
 *
 * @param [in] varThreshold varThreshold determines how close a pixel must be to a model to be
 * classified as belonging to that model, for the purpose of classifying a pixel as background.
 * Higher values mean that pixels further from the mean will be classified as belonging to that
 * model. Default value is 16. This parameter does not affect how the model is updated.
 *
 * Determining if a pixel is background happens as follows. We go over each model in
 * decreasing order of weight while checking the following 2 conditions:
 *
 * 1. Pixel must be close to the current model using the following criterion:
 *    (pixel - model.mean)^2 < varThreshold * model.variance
 * 2. Total weight of all preceeding models should be lower than 0.9.
 *
 * @param [in] enableShadowDetection Controls whether the algorithm detects and marks shadow regions.
 * - false: Shadow detection disabled
 * - true: Shadow detection enabled
 *
 * When enabled, the algorithm will attempt to identify pixels that are darker versions
 * of the background (shadows) and mark them with shadowPixelValue instead of classifying
 * them as foreground.
 *
 * @param [in] shadowPixelValue The pixel value used to mark detected shadow regions in the output mask.
 * - Typically set to 127 (gray) to distinguish from:
 *   * Background (0, black)
 *   * Foreground (255, white)
 *
 * Only relevant when detectShadow is enabled (set to true).
 * Setting this to 0 while detectShadow is enabled effectively disables shadow detection,
 * as shadow pixels will be indistinguishable from background.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaBackgroundSubtractorSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVImageHandle in,
                                         NVCVImageHandle outMask, NVCVImageHandle outBackgroundImage,
                                         float learningRate, float varThreshold, bool enableShadowDetection,
                                         uint8_t shadowPixelValue);

/**
 * Submits the BackgroundSubtractor operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in Input image handle.
 * @param [out] outMask Output mask handle.
 * @param [out] outBackgroundImage Optional output background image handle.
 * @param [in] learningRate Learning rate [0.0, 1.0].
 * @param [in] varThreshold Variance threshold for background classification.
 * @param [in] enableShadowDetection Enable shadow detection.
 * @param [in] shadowPixelValue Pixel value for detected shadows.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaBackgroundSubtractorSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageHandle in,
                                         NVCVImageHandle outMask, NVCVImageHandle outBackgroundImage,
                                         float learningRate, float varThreshold, bool enableShadowDetection,
                                         uint8_t shadowPixelValue);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPBACKGROUNDSUBTRACTOR_H */
