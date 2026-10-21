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
 * @file OpWarpPerspective.h
 *
 * @brief Defines types and functions to handle the warp perspective operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_WARPPERSPECTIVE WarpPerspective
 * @{
 */

#ifndef PVA_SOLUTIONS_OPWARPPERSPECTIVE_H
#define PVA_SOLUTIONS_OPWARPPERSPECTIVE_H

#include <PvaOperator.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Image.h>
#include <nvcv/Status.h>

#ifdef __cplusplus
extern "C" {
#endif

/// 3x3 row-major matrix for warping remap from output pixel to input
typedef float WarpMatrixType[3][3];

/** Constructs an instance of the WarpPerspective operator.
 *
 * Perspective Warp algorithm allows for correcting perspective distortion caused by camera misalignment with respect to the object plane being captured.
 * If the camera position, tilt and pan relative to the frame are known, a 3x3 pespective transform can be derived.
 * For perspective transformation, it will perserve collinearity and incidence. When the third row of warping matrix is [0, 0, 1], the transformation is also affine.
 * Affine transformation will also maintain parallelism.
 *
 * \b Limitations:
 *      - When having fixed output tile shape, the related input tile can be whole image theoretically for certain perspective transformations.
 *      For PVA implementations, only 32KB can be afforded for intput tile size, and the host sanity check will throw exceptions if it fails.
 *      - Currently only NVCV_BORDER_CONSTANT is supported when accessing input pixels outside the image boundary.
 *
 * \b Input:
 *      Image sizes and formats are specified in inImageRequirements.
 *
 *      Image Format   | Allowed
 *      -------------- | -------------
 *      U8             | Yes
 *      NV12           | Yes
 *      NV12_ER        | Yes  (reuses the NV12 kernel; warp is range-agnostic)
 *
 * \b Output:
 *      Image sizes and formats are specified in outImageRequirements.
 *
 * \b Input/Output \b Dependency:
 *      Property    | Input == Output
 *     -------------| -------------
 *      Format      | Yes
 *
 * \b Parameters
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] inRequirements Pointer to the NVCVImageRequirements structure which contains image shape,
 *                                strides, and format information.

 * @param [in] outRequirements Pointer to the output image NVCVImageRequirements structure.
 *
 * @param [in] interpType Interpolation type to be used when accessing non integer pixel values. \ref PVAInterpolationType.
 *      - PVA_INTERPOLATION_NN or PVA_INTERPOLATION_LINEAR
 *
 * @param [in] warpTransformationType to indicate if the warp transformation is perspective or affine. \ref PVAWarpTransformationType.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT       Possible cases include:
 *                                            1) The handle, stream, input, or output is either NULL or points to an invalid address.
 *                                            2) The input or output image does not meet the requirements used to create the operator handle.
 *                                            3) The metadata of the input does not correspond to the correct NV12 format (e.g., number of planes not equal to 2).
 * @retval NVCV_ERROR_OUT_OF_MEMORY          Possible cases include:
 *                                            1) Failed to allocate memory for the operator.
 *                                            2) Failed to allocate memory for holding input tile buffer, which the host initializes then sends it for the device's use.
 * @retval NVCV_SUCCESS                Operation executed successfully.
*/
NVCVStatus pvaWarpPerspectiveCreate(NVCVOperatorHandle *handle, NVCVImageRequirements *inRequirements,
                                    NVCVImageRequirements *outRequirements, PVAInterpolationType interpType,
                                    PVAWarpTransformationType warpTransformationType);

#ifdef __cplusplus
}

/**
 * Submits the WarpPerspective operator to a cuPVA stream.
 *
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] in Input image handle.
 *
 * @param [in] invWarpMatrix 3x3 row-major matrix for warping remap, input coordinates are computed via invWarpMatrix * [x, y, 1]^T
 *
 * @param [out] out Output image handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaWarpPerspectiveSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVImageHandle in,
                                    WarpMatrixType invWarpMatrix, NVCVImageHandle out);

/**
 * Submits the WarpPerspective operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in Input image handle.
 * @param [in] invWarpMatrix 3x3 row-major matrix for warping remap.
 * @param [out] out Output image handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaWarpPerspectiveSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageHandle in,
                                    WarpMatrixType invWarpMatrix, NVCVImageHandle out);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPWARPPERSPECTIVE_H */