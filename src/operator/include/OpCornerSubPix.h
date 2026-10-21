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
 * @file OpCornerSubPix.h
 *
 * @brief Defines types and functions to handle the Sub-Pixel Corner Detection operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_CORNERSUBPIX CornerSubPix
 * @{
 */

#ifndef PVA_SOLUTIONS_CORNER_SUB_PIX_H
#define PVA_SOLUTIONS_CORNER_SUB_PIX_H

#include <PvaOperator.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Size.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Constructs an instance of the CornerSubPix operator.
 *
 * \b Limitations:
 *
 * \b Image:
 *     Data Layout:    [CHW], [HWC], [NCHW], [NHWC] only when
 *     C:              [1]
 *     N:              [1]
 *
 *     Data Type      | Allowed
 *     -------------- | -------------
 *     8bit  Unsigned | Yes
 *     8bit  Signed   | No
 *     16bit Unsigned | No
 *     16bit Signed   | No
 *     32bit Unsigned | No
 *     32bit Signed   | No
 *     32bit Float    | No
 *     64bit Float    | No
 *
 * \b Corners:
 *     Data Layout:    [W], [HW], [CHW], [NCHW] only when
 *     C:              [1]
 *     N:              [1]
 *     Data Type      | Allowed
 *     -------------- | -------------
 *     8bit  Unsigned | No
 *     8bit  Signed   | No
 *     16bit Unsigned | No
 *     16bit Signed   | No
 *     32bit Unsigned | No
 *     32bit Signed   | No
 *     32bit Float    | Yes
 *     64bit Float    | No
 *
 * \b Parameters:
 *
 *     winSize:
 *      - winWidth  = 3 or 4
 *      - winHeight = 3 or 4
 *      - winWidth  = winHeight
 *     zeroZone:
 *      - zeroZoneWidth < winWidth
 *      - zeroZoneHeight < winHeight
 *      - negative zeroZoneWidth or zeroZoneHeight means no zeroZone
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] cornerParams Pointer to the NVCVTensorRequirements structure which contains [in/out] corners Tensor rank, shape, layout and data type information.
 *
 * @param [in] imageParams Pointer to the NVCVTensorRequirements structure which contains [in] image Tensor rank, shape, layout and data type information.
 *
 * @param [in] winSize Half of the side length of the search window.
 *
 * @param [in] zeroZone Half of the size of the dead region in the middle of the search zone over which the summation is not done.
 *                      It is used sometimes to avoid possible singularities of the autocorrelation matrix.
 *                      The negative indicates that there is no such a size.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_OUT_OF_MEMORY    No enough memory to create the operator.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaCornerSubPixCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements *cornerParams,
                                 NVCVTensorRequirements *imageParams, const NVCVSize2D winSize,
                                 const NVCVSize2D zeroZone);

#ifdef __cplusplus
}

/**
 * Submits the CornerSubPix operator to a cuPVA stream.
 *
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] inCorners Input tensor handle.
 *                       Data Type: NVCV_DATA_TYPE_F32
 *                       Data Layout: [W], [HW], [CHW], [NCHW] only when C = 1 and N = 1
 *
 * @param [out] outCorners Output tensor handle.
 *                         Data Type: NVCV_DATA_TYPE_F32
 *                         Data Layout: [W], [HW], [CHW], [NCHW] only when C = 1 and N = 1
 *
 * @param [in] image Input tensor handle.
 *                   Data Type: NVCV_DATA_TYPE_U8
 *                   Data Layout: [CHW], [HWC], [NCHW], [NHWC] only when C = 1 and N = 1
 *
 * @param [in] maxIters The process of corner position refinement stops after maxIters iterations, must >= 6.
 *
 * @param [in] eps The process of corner position refinement stops when the corner position moves by less than eps on some iteration.
 *
 * @param [in] numCorners The number of [in/out] corners, valid range (0, 4096].
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside the valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaCornerSubPixSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle inCorners,
                                 NVCVTensorHandle outCorners, NVCVTensorHandle image, const int32_t maxIters,
                                 const float eps, const int32_t numCorners);

/**
 * Submits the CornerSubPix operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] inCorners Input tensor handle.
 * @param [out] outCorners Output tensor handle.
 * @param [in] image Input tensor handle.
 * @param [in] maxIters The process of corner position refinement stops after maxIters iterations.
 * @param [in] eps The process of corner position refinement stops when the corner position moves by less than eps.
 * @param [in] numCorners The number of corners.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside the valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaCornerSubPixSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle inCorners,
                                 NVCVTensorHandle outCorners, NVCVTensorHandle image, const int32_t maxIters,
                                 const float eps, const int32_t numCorners);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_CORNER_SUB_PIX_H */