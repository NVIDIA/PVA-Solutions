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
 * @file OpRemap.h
 *
 * @brief Defines types and functions to handle the remap operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_REMAP Remap
 * @{
 */

#ifndef PVA_SOLUTIONS_OPREMAP_H
#define PVA_SOLUTIONS_OPREMAP_H

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

/** Constructs an instance of the Remap operator.
 *
 * The Remap algorithm applies a generic geometric transformation to an image using a warp map.
 * For each pixel in the output image, the warp map specifies the corresponding (possibly non-integer) source location in the input image.
 *
 * Specifically, for output[i][j], the warp map provides a float coordinate warpMap[i][j] = {inX, inY}
 *
 * - When using nearest neighbor interpolation, the output pixel is computed by rounding the float coordinate to the nearest integer:
 *       output[i][j] = input[round(inY)][round(inX)]
 *     where round(inX) = floor(inX + 0.5) and round(inY) = floor(inY + 0.5)
 *
 * - When using bilinear interpolation, the output pixel is computed by bilinearly interpolating the four neighboring input pixels surrounding the float coordinate:
 *         output[i][j] = top + (bottom - top) * fracY
 *         top = topLeft + (topRight - topLeft) * fracX
 *         bottom = bottomLeft + (bottomRight - bottomLeft) * fracX
 *     where the four neighboring input pixels are:
 *         - topLeft = input[floor(inY)][floor(inX)]
 *         - topRight = input[floor(inY)][ceil(inX)]
 *         - bottomLeft = input[ceil(inY)][floor(inX)]
 *         - bottomRight = input[ceil(inY)][ceil(inX)]
 *     The interpolation weights {fracX, fracY} are determined by the fractional part of inX and inY.
 *
 * \b Limitations:
 *     Remap is often used to perform a geometric transformation to an image using a smoothly varying warp map. To do this efficiently on PVA, we process in a
 *     regular fixed-size raster scan tile sequence across the output image (the output tile size is 64x30). For each output tile, the size of the required input tile
 *     is defined by the minimal and maximal X and Y values of the warp map within that tile. The entire input tile is brought into VMEM at once, and is double
 *     buffered for efficient pipelining. We define the space for one input tile in VMEM to be as large as possible, which comes out to 32kB. Therefore, the warp map
 *     contents must meet the following constraint for each tile: (maxX - minX + 1) * (maxY - minY + 1) * BytesPerPixel < 32KB.
 *     pvaRemapValidate can be used to check whether the warp map satisfies the above constraint. It will return errors if any input tile exceeds 32KB.
 *
 * \b Input:
 *      Image sizes and formats are specified in inRequirements.
 *
 *      Image Format   | Allowed
 *      -------------- | -------------
 *      U8             | Yes
 *      NV12           | Yes
 *
 * \b Output:
 *      Image sizes and formats are specified in outRequirements.
 *
 *      Image Format   | Allowed
 *      -------------- | -------------
 *      U8             | Yes
 *      NV12           | Yes
 *
 * \b WarpMap:
 *      Data Layout:    [kHWC]
 *      Channels:       [2]
 *      Value Range:    (-inf, +inf)
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      32bit Float    | Yes
 *
 * \b Parameters
 *      borderMode = NVCV_BORDER_CONSTANT or NVCV_BORDER_REPLICATE
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] inRequirements Pointer to the input image NVCVImageRequirements structure.
 *
 * @param [in] outRequirements Pointer to the output image NVCVImageRequirements structure.
 *
 * @param [in] warpMapRequirements Pointer to the warp map NVCVTensorRequirements structure.
 *
 * @param [in] interpType Interpolation type to be used when accessing non integer pixel values. \ref PVAInterpolationType.
 *
 * @param [in] borderMode Border mode to be used when accessing elements outside input image.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT       Possible cases include:
 *                                            1) The handle, stream, input, or output is either NULL or points to an invalid address.
 *                                            2) The input or output image does not meet the requirements used to create the operator handle.
 *                                            3) The metadata of the input does not correspond to the correct format.
 * @retval NVCV_ERROR_OUT_OF_MEMORY          Possible cases include:
 *                                            1) Failed to allocate memory for the operator.
 *                                            2) Failed to allocate memory for holding input tile buffer.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaRemapCreate(NVCVOperatorHandle *handle, NVCVImageRequirements const *const inRequirements,
                          NVCVImageRequirements const *const outRequirements,
                          NVCVTensorRequirements const *const warpMapRequirements,
                          PVAInterpolationType const interpType, NVCVBorderType const borderMode);

/**
 * Validates the warp map. Return errors if any input tile exceeds 32KB. This API is mainly for debugging purpose.
 *
 * @param [in] warpMapHandle Handle to the warp map tensor.
 *
 * @param [in] inWidth Width of the input image.
 *
 * @param [in] inHeight Height of the input image.
 *
 * @param [in] outWidth Width of the output image.
 *
 * @param [in] outHeight Height of the output image.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Invalid warp map due to input tile size exceeding 32KB.
 * @retval NVCV_SUCCESS                Valid warp map.
 */
NVCVStatus pvaRemapValidate(NVCVTensorHandle warpMapHandle, int32_t inWidth, int32_t inHeight, int32_t outWidth,
                            int32_t outHeight);

#ifdef __cplusplus
}

/**
 * Submits the Remap operator to a cuPVA stream.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] in Input image handle.
 *
 * @param [out] out Output image handle.
 *
 * @param [in] warpMap Warp map tensor handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaRemapSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVImageHandle in, NVCVImageHandle out,
                          NVCVTensorHandle warpMap);

/**
 * Submits the Remap operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in Input image handle.
 * @param [out] out Output image handle.
 * @param [in] warpMap Warp map tensor handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaRemapSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageHandle in, NVCVImageHandle out,
                          NVCVTensorHandle warpMap);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPREMAP_H */
