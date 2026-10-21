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
 * @file OpBlurFilterROI.h
 *
 * @brief Defines types and functions to handle the BlurFilterROI operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_BLURFILTERROI BlurFilterROI
 * @{
 */

#ifndef PVA_SOLUTIONS_OPBLURFILTERROI_H
#define PVA_SOLUTIONS_OPBLURFILTERROI_H

#include <PvaOperator.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Image.h>
#include <nvcv/Rect.h>
#include <nvcv/Size.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
  * Constructs an instance of the BlurFilterROI operator.
  *
  * \b Limitations:
  *
  *    The maximum supported width and height of ROI rectangle is 256 pixels.
  *    The maximum supported number of ROI rectangles is 256.
  *    The ROI rectangles are assumed inside the image. There is no checking or error handling due to the performance reason.
  *
  * \b Input:
  *    Image Format: NVCV_IMAGE_FORMAT_NV12, NVCV_IMAGE_FORMAT_NV12_ER, NVCV_IMAGE_FORMAT_NV12_BL or NVCV_IMAGE_FORMAT_NV12_ER_BL
  *
  * \b Output:
  *    Image Format: NVCV_IMAGE_FORMAT_NV12, NVCV_IMAGE_FORMAT_NV12_ER, NVCV_IMAGE_FORMAT_NV12_BL or NVCV_IMAGE_FORMAT_NV12_ER_BL
  *
  * \b Input/Output \b Dependency:
  *     Property      |  Input == Output
  *    -------------- | -------------
  *     Width         |  Yes
  *     Height        |  Yes
  *     Image format  |  No
  *
  * @param [out] handle Where the operator instance handle will be written to.
  *                     + Must not be NULL.
  *
  * @param [in] inImageRequirements Pointer to the NVCVImageRequirements structure which contains input image width, height and format information.
  *    This structure must be filled by using nvcvImageCalcRequirementsPva().
  *
  * @param [in] outImageRequirements Pointer to the NVCVImageRequirements structure which contains output image width, height and format information.
  *    This structure must be filled by using nvcvImageCalcRequirementsPva().
  *
  * @param [in] rectTensorRequirements Pointer to the NVCVTensorRequirements structure which contains Tensor rank, shape, layout and data type information.
  *    This structure must be filled by using nvcvTensorCalcRequirementsPva().
  *    The tensor should be a 1D tensor (rank=1) and the data type should correspond to NVCV_DATA_TYPE_4S32.
  *
  * @param [in] blockSize The size of width and height to be used when splitting the ROI rectangle into multiple blocks.
  *                       For each ROI rectangle, the blur filter will use the value on the top-left corner of the block to fill in the whole block.
  *
  * @retval NVCV_ERROR_INVALID_ARGUMENT     Handle is null or some parameter is outside valid range.
  * @retval NVCV_ERROR_OUT_OF_MEMORY        Not enough memory to create the operator.
  * @retval NVCV_ERROR_INVALID_IMAGE_FORMAT Image format is invalid.
  * @retval NVCV_SUCCESS                    Operation executed successfully.
  */
NVCVStatus pvaBlurFilterROICreate(NVCVOperatorHandle *handle, NVCVImageRequirements const *const inImageRequirements,
                                  NVCVImageRequirements const *const outImageRequirements,
                                  NVCVTensorRequirements const *const rectTensorRequirements,
                                  const NVCVSize2D blockSize);

#ifdef __cplusplus
}

/**
  * Submits the BlurFilterROI operator to a cuPVA stream.
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
  * @param [in] rect Input tensor handle to store ROI rectangles.
  *                  Each element in the tensor is an NVCVRectI structure, representing {x, y, width, height} in NVCV_DATA_TYPE_4S32 datatype.
  *
  * @param [in] numRects Input variable to store number of ROI rectangles.
  *
  * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
  * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
  * @retval NVCV_SUCCESS                Operation executed successfully.
  */
NVCVStatus pvaBlurFilterROISubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVImageHandle const in,
                                  NVCVImageHandle out, NVCVTensorHandle const rect, const size_t numRects);

/**
 * Submits the BlurFilterROI operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in Input image handle.
 * @param [out] out Output image handle.
 * @param [in] rect Input tensor handle to store ROI rectangles.
 * @param [in] numRects Input variable to store number of ROI rectangles.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaBlurFilterROISubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageHandle const in,
                                  NVCVImageHandle out, NVCVTensorHandle const rect, const size_t numRects);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPBLURFILTERROI_H */
