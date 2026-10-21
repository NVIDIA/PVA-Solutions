/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OpNms.h
 *
 * @brief Defines types and functions to handle the Non-Maximum Suppression operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_NMS NMS (Non-Maximum Suppression)
 *
 * Compatibility: Requires PVA SDK 2.6.0 and later.
 *
 * @{
 */

#ifndef PVA_SOLUTIONS_OPNMS_H
#define PVA_SOLUTIONS_OPNMS_H

#include <PvaOperator.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/BorderType.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Constructs an instance of the Non-Maximum Suppression (NMS) operator.
 *
 * For each pixel an nmsSize x nmsSize window centered on that pixel is searched. If the pixel is the
 * maximum of the window it is copied to the output, otherwise the output pixel is set to zero. Ties
 * are resolved by position: among equal maxima within the window, the pixel that appears last in
 * raster order (left to right, top to bottom) wins, that is, the one with the largest row index
 * and, if those are equal, the largest column index. A rectangular plateau of equal values
 * therefore leaves exactly one surviving pixel, at its bottom-right corner, rather than keeping all
 * of them.
 *
 * \b Limitations:
 * - nmsSize must be 3 or 5.
 * - Only NVCV_BORDER_CONSTANT and NVCV_BORDER_REPLICATE are supported.
 * - The suppression decision is exact only where the center pixel differs from every other pixel of
 *   its window by more than 4 for a 3x3 window or 12 for a 5x5 window; see \b Accuracy below.
 *
 * \b Input:
 *      Data Layout:    [kHWC]
 *      Channels:       [1]
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      8bit  Unsigned | No
 *      8bit  Signed   | No
 *      16bit Unsigned | No
 *      16bit Signed   | No
 *      32bit Unsigned | No
 *      32bit Signed   | Yes
 *      32bit Float    | No
 *      64bit Float    | No
 *
 * \b Output:
 *      Data Layout:    [kHWC]
 *      Channels:       [1]
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      8bit  Unsigned | No
 *      8bit  Signed   | No
 *      16bit Unsigned | No
 *      16bit Signed   | No
 *      32bit Unsigned | No
 *      32bit Signed   | Yes
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
 * \b Parameters
 *
 *      Data Type  = NVCV_DATA_TYPE_S32 only
 *      nmsSize    = 3 or 5 (window is nmsSize x nmsSize)
 *      borderMode = NVCV_BORDER_CONSTANT or NVCV_BORDER_REPLICATE
 *
 * \b Accuracy:
 *
 *      The window maximum is computed with a position-dependent bias added to each tap, which
 *      implements the bottom-right tie-break. Because the bias is added to the pixel value, the
 *      decision is exact only where the center pixel differs from every other pixel of its window
 *      by more than B = (nmsSize * nmsSize - 1) / 2, i.e. 4 for a 3x3 window and 12 for a 5x5
 *      window. Within that margin the result can err in either direction: a lower neighbor can
 *      suppress the center pixel, or a higher one can fail to suppress it.
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] tensorRequirements Pointer to the NVCVTensorRequirements structure which contains Tensor rank, shape, layout and data type information.
 *                     + Must not be NULL.
 *
 * @param [in] nmsSize Suppression window size. Must be 3 or 5.
 *
 * @param [in] borderMode Border mode to be used when accessing elements outside input image.
 *
 * @param [in] borderValue Constant border value to be used when borderMode is NVCV_BORDER_CONSTANT. Ignored otherwise.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaNmsCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements *tensorRequirements, int32_t nmsSize,
                        const NVCVBorderType borderMode, int32_t borderValue);

#ifdef __cplusplus
}

/**
 * Submits the NMS operator to a cuPVA stream.
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
NVCVStatus pvaNmsSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out);

/**
 * Submits the NMS operator to a CUDA stream.
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
NVCVStatus pvaNmsSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPNMS_H */
