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
 * @file OpDistanceTransform.h
 *
 * @brief Defines types and functions to handle the distance transform operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_DISTANCE_TRANSFORM DistanceTransform
 * @{
 */

#ifndef PVA_SOLUTIONS_DISTANCE_TRANSFORM_H
#define PVA_SOLUTIONS_DISTANCE_TRANSFORM_H

#include <PvaOperator.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Constructs an instance of the DistanceTransform operator.
 *  The operator computes the minimum distance of each off-pixel in a binary image to the nearest on pixel.
 *  It is commonly used in object segmentation, shape analysis, and skeletonization.
 *
 * \b Limitations:
 *
 * \b Input:
 *     Apply to the inImage parameter of the \ref pvaDistanceTransformSubmit.
 *     Data Layout: [kHWC]
 *     Channels:    [1]
 *
 *     Data Type      | Allowed
 *     -------------- | -------------
 *     8bit  Unsigned | No
 *     8bit  Signed   | No
 *     16bit Unsigned | Yes
 *     16bit Signed   | No
 *     32bit Unsigned | No
 *     32bit Signed   | No
 *     32bit Float    | No
 *     64bit Float    | No
 *
 * \b Output:
 *     Apply to the outDistance and outLabel parameters of the \ref pvaDistanceTransformSubmit.
 *     Data Layout:    [kHWC]
 *     Channels:       [1]
 *
 *     Data Type      | Allowed
 *     -------------- | -------------
 *     8bit  Unsigned | No
 *     8bit  Signed   | No
 *     16bit Unsigned | Yes
 *     16bit Signed   | No
 *     32bit Unsigned | No
 *     32bit Signed   | No
 *     32bit Float    | No
 *     64bit Float    | No
 *
 * \b Input/Output \b Dependency:
 *     Property    | Input == Output
 *     ------------| -------------
 *     Data Layout | Yes
 *     Data Type   | Yes
 *     Number      | Yes
 *     Channels    | Yes
 *     Width       | Yes
 *     Height      | Yes
 *
 * \b Width/Height:
 *     Any value from 32 to 1024
 *
 * \b Number \b of \b labels:
 *     Up to 16 labels valued from 0 to 15
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] tensorRequirements Pointer to the NVCVTensorRequirements structure which contains input image tensor rank, shape, layout and data type information.
 *
 * @param [in] distanceType Type of distance that the operation computes. \ref PVADistanceType. Current implementation only supports Euclidean distance. The distance value is in UQ13.3 fixed-point format.
 *
 * @retval NVCV_ERROR_NOT_IMPLEMENTED  The distance type is not implementated.
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaDistanceTransformCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements *tensorRequirements,
                                      PVADistanceType distanceType);

#ifdef __cplusplus
}

/**
 * Submits the DistanceTransform operator to a cuPVA stream.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 *
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] inImage Input image with tensor handle.
 *                     The image contains pixel label values in uint16 format. Pixels that are off (not labeled) should be set to 0xFFFF.
 *                     Label values between 0 and 15 (inclusive) denote active pixels in the image.
 *                     The caller must ensure input labels fall within this supported range. Labels outside this range will be ignored and produce undefined results.
 *
 * @param [in] maxDistance The max distance of the operator in the fixed-point format of UQ13.3. Any distance greater is saturated to this value.
 *                         The max distance should be in the range from 0.0 to (image_width + image_height).0. Setting to 0.0 means using the default
 *                         value (image_width + image_height).0 (need to convert to UQ13.3 format by left shifting 3 bits).
 *
 * @param [out] outDistance Output distance tensor handle. The distance from off pixel to the nearest on pixel is notated as UQ13.3.
 *
 * @param [out] outLabel Output label tensor handle. The label value of each pixel marks the Voronoi regions.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside the valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaDistanceTransformSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, const NVCVTensorHandle inImage,
                                      uint16_t maxDistance, NVCVTensorHandle outDistance, NVCVTensorHandle outLabel);

/**
 * Submits the DistanceTransform operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] inImage Input image with tensor handle.
 * @param [in] maxDistance The max distance of the operator in the fixed-point format of UQ13.3.
 * @param [out] outDistance Output distance tensor handle.
 * @param [out] outLabel Output label tensor handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside the valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaDistanceTransformSubmit(NVCVOperatorHandle handle, cudaStream_t stream, const NVCVTensorHandle inImage,
                                      uint16_t maxDistance, NVCVTensorHandle outDistance, NVCVTensorHandle outLabel);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_DISTANCE_TRANSFORM_H */