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
 * @file OpMinMaxLoc.h
 *
 * @brief Defines types and functions to handle the Min/Max Location operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_MINMAXLOC MinMaxLoc
 * @{
 */

#ifndef PVA_SOLUTIONS_OPMINMAXLOC_H
#define PVA_SOLUTIONS_OPMINMAXLOC_H

#include <PvaOperator.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Constructs and an instance of MinMaxLoc operator. It calculates location coordinates achieving min/max values on the input image.
 *
 * \b Limitations:
 *      1. The operator supports arbitrary image sizes, but achieves the best performance when image width is divisible by tile width 64.
 *      2. The min/max count number accumulates for all locations achieving the min/max value. But the number of min/max locations to return is limited by the user defined capacity.
 *
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] tensorRequirements NVCVTensorRequirements structure to specify input parameters:
 *
 *      layout: [HWC, NHWC, CHW, NCHW], where N=1 and C=1.
 *      rank: 3 for [HWC, CHW], and 4 for [NHWC, NCHW].
 *      shape: input tensor shape. For example, for [HWC] layout, shape={image_height, image_width, channel_number}.
 *      dtype: input tensor data type, must be one of:
 *      {NVCV_DATA_TYPE_U8, NVCV_DATA_TYPE_U16, NVCV_DATA_TYPE_U32, NVCV_DATA_TYPE_F32}.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaMinMaxLocCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements *tensorRequirements);

#ifdef __cplusplus
}

/**
 * Submits the MinMaxLoc operator to a cuPVA stream.
 *
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] in Input image tensor handle. Currently only supports [HWC] tensor layout.
 *
 * [Note] The operator supports arbitrary image sizes, but achieves the best performance when image width is divisible by tile width 64.
 *
 * \b Limitations:
 *
 * \b Input:
 *      Data Layout:    [HWC, NHWC, CHW, NCHW]
 *      Batches(N):     [1]
 *      Channels(C):    [1]
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      8bit  Unsigned | Yes
 *      8bit  Signed   | Yes
 *      16bit Unsigned | Yes
 *      16bit Signed   | Yes
 *      32bit Unsigned | Yes
 *      32bit Signed   | Yes
 *      32bit Float    | Yes
 *      64bit Float    | No
 *
 *
 * @param [out] minVal Output tensor handle for min values. For current [HWC] input layout with 1 channel, the output tensor is of 1 number.
 *
 *      Data Layout:    [W]
 *      Length:         1
 *      Data Type:      the same as input data type
 *
 * @param [out] minLoc Output tensor handle for min location coordinates. Maximum number of locations is locCapacity defined in pvaMinMaxLocCreate.
 *
 *      Data Layout:    [W]
 *      Channel:        1
 *      Length:         locCapacity
 *      Data Type:      NVCV_DATA_TYPE_2S16 (16 bit signed int, XY-interleaved)
 *
 * @param [out] numMin Output tensor handle for min location counts. For current [HWC] input layout with 1 channel, the output tensor is of 1 number.
 *  [Note] The count number will accumulate for all the min/max locations in the input, thus could be larger than the location tensor capacity.
 *
 *      Data Layout:    [W]
 *      Length:         1
 *      Range:          [1, image_height * image_width]
 *      Data Type:      NVCV_DATA_TYPE_S32
 *
 * @param [out] maxVal Output tensor handle for max values. For current [HWC] input layout with 1 channel, the output tensor is of 1 number.
 *
 *      Data Layout:    [W]
 *      Length:         1
 *      Data Type:      the same as input data type
 *
 * @param [out] maxLoc Output tensor handle for max location coordinates.  Maximum number of locations is locCapacity defined in pvaMinMaxLocCreate.
 *
 *      Data Layout:    [W]
 *      Channel:        1
 *      Length:         locCapacity
 *      Data Type:      NVCV_DATA_TYPE_2S16 (16 bit signed int, XY-interleaved)
 *
 * @param [out] numMax Output tensor handle for max location counts. For current [HWC] input layout with 1 channel, the output tensor is of 1 number.
 *  [Note] the count number will accumulate for all the min/max locations in the input, thus could be larger than the location tensor capacity.
 *
 *      Data Layout:    [W]
 *      Length:         1
 *      Range:          [1, image_height * image_width]
 *      Data Type:      NVCV_DATA_TYPE_S32
 *
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaMinMaxLocSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle in,
                              NVCVTensorHandle minVal, NVCVTensorHandle minLoc, NVCVTensorHandle numMin,
                              NVCVTensorHandle maxVal, NVCVTensorHandle maxLoc, NVCVTensorHandle numMax);

/**
 * Submits the MinMaxLoc operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in Input image tensor handle.
 * @param [out] minVal Output tensor handle for min values.
 * @param [out] minLoc Output tensor handle for min location coordinates.
 * @param [out] numMin Output tensor handle for min location counts.
 * @param [out] maxVal Output tensor handle for max values.
 * @param [out] maxLoc Output tensor handle for max location coordinates.
 * @param [out] numMax Output tensor handle for max location counts.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaMinMaxLocSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                              NVCVTensorHandle minVal, NVCVTensorHandle minLoc, NVCVTensorHandle numMin,
                              NVCVTensorHandle maxVal, NVCVTensorHandle maxLoc, NVCVTensorHandle numMax);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPMINMAXLOC_H */
