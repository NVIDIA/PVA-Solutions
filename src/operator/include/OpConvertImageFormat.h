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
 * @file OpConvertImageFormat.h
 *
 * @brief Defines types and functions to handle the ConvertImageFormat operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_CONVERTIMAGEFORMAT ConvertImageFormat
 * @{
 */

#ifndef PVA_SOLUTIONS_OPCONVERTIMAGEFORMAT_H
#define PVA_SOLUTIONS_OPCONVERTIMAGEFORMAT_H

#include <PvaOperator.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Image.h>
#include <nvcv/Status.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructs an instance of the ConvertImageFormat operator.
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] inImageRequirements Pointer to the NVCVImageRequirements structure which contains input image width, height and format information.
 *
 *   \b Input:
 *    This structure must be filled by using nvcvImageCalcRequirementsPva().
 *
 * @param [in] outImageRequirements Pointer to the NVCVImageRequirements structure which contains output image width, height and format information.
 *
 *   \b Output:
 *    This structure must be filled by using nvcvImageCalcRequirementsPva().
 *
 *   \b Input/Output \b Dependency:
 *
 *     Property      |  Input == Output
 *    -------------- | -------------
 *     Width         | Yes
 *     Height        | Yes
 *     Image format  | No
 *
 *    The supported combinations of input and output image formats for conversion are shown in the following table.
 *    Rows represent the input image format and columns represent the output image format.
 *
 *     in/out  | NV12   | NV12_ER | YUYV   | UYVY   | VYUY   | YUV8p  | BGR8   | RGB8   | BGRA8  | RGBA8  | BGR8p  | RGB8p
 *    ---------|--------|---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------
 *      NV12   | -      | -       | -      | -      | -      | -      | yes    | yes    | -      | -      | yes    | yes
 *      NV12_ER| -      | -       | -      | -      | -      | -      | yes    | yes    | -      | yes    | -      | -
 *      YUYV   | yes    | yes     | -      | -      | -      | -      | yes    | yes    | -      | -      | yes    | yes
 *      UYVY   | yes    | yes     | -      | -      | -      | -      | yes    | yes    | -      | -      | yes    | yes
 *      VYUY   | yes    | yes     | -      | -      | -      | -      | yes    | yes    | -      | -      | yes    | yes
 *      YUV8p  | yes    | yes     | yes    | yes    | yes    | -      | -      | -      | -      | -      | -      | -
 *      BGR8   | yes    | -       | yes    | yes    | yes    | -      | -      | -      | yes    | -      | -      | -
 *      RGB8   | yes    | -       | yes    | yes    | yes    | -      | -      | -      | -      | yes    | -      | -
 *      BGRA8  | -      | -       | -      | -      | -      | -      | yes    | -      | -      | -      | -      | -
 *      RGBA8  | -      | -       | -      | -      | -      | -      | -      | yes    | -      | -      | -      | -
 *      BGR8p  | yes    | -       | yes    | yes    | yes    | -      | -      | -      | -      | -      | -      | -
 *      RGB8p  | yes    | -       | yes    | yes    | yes    | -      | -      | -      | -      | -      | -      | -
 *
 *    NOTES:
 *    - `NV12_ER` is the BT.601 full-range (0-255 luma, 0-255 chroma) variant of NV12. Exposed paths:
 *      YUV → NV12_ER (chroma resample) and NV12_ER → RGB8 / BGR8 / RGBA8 (full-range YUV→RGB, no Y-16
 *      offset). The reverse RGB/BGR → NV12_ER and the planar NV12_ER → RGB8p/BGR8p paths are still
 *      pending and tracked as a follow-up.
 *    - For RGBA8 / BGRA8 output, alpha is set to 255 in every pixel.
 *    - `RGBA8 -> RGB8` and `BGRA8 -> BGR8` are single-pass (the alpha lane is dropped).
 *    - Single-pass studio-range `NV12 -> RGBA8` is not yet supported; chain `NV12 -> RGB8 -> RGBA8`
 *      (the full-range `NV12_ER -> RGBA8` path is direct).
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT     Handle is null or some parameter is outside valid range.
 * @retval NVCV_ERROR_OUT_OF_MEMORY        Not enough memory to create the operator.
 * @retval NVCV_ERROR_INVALID_IMAGE_FORMAT Image format is invalid.
 * @retval NVCV_SUCCESS                    Operation executed successfully.
 */
NVCVStatus pvaConvertImageFormatCreate(NVCVOperatorHandle *handle,
                                       NVCVImageRequirements const *const inImageRequirements,
                                       NVCVImageRequirements const *const outImageRequirements);

#ifdef __cplusplus
}

/**
 * Submits the ConvertImageFormat operator to a cuPVA stream.
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
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaConvertImageFormatSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVImageHandle const in,
                                       NVCVImageHandle out);

/**
 * Submits the ConvertImageFormat operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in Input image handle.
 * @param [out] out Output image handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaConvertImageFormatSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageHandle const in,
                                       NVCVImageHandle out);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPCONVERTIMAGEFORMAT_H */
