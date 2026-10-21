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
 * @file OpContourTracing.h
 *
 * @brief Defines types and functions to handle the ContourTracing operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_CONTOUR_TRACING ContourTracing
 * @{
 */

#ifndef PVA_SOLUTIONS_OPCONTOURTRACING_H
#define PVA_SOLUTIONS_OPCONTOURTRACING_H

#include <PvaOperator.h>
#include <cuda_runtime.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Number of entries in the contour tracing direction lookup table. */
#define PVA_CONTOUR_TRACING_DIRECTION_TABLE_ENTRIES 256U

/**
 * Constructs an instance of the ContourTracing operator.
 *
 * Given a binary image (foreground is nonzero, background is 0), the ContourTracing operator walks the
 * boundary of each connected foreground component and outputs its boundary pixels as an ordered
 * sequence of (x, y) coordinate pairs. The caller supplies one or more bounding-box ROIs (via the
 * rect tensor) that each enclose a single connected component. There is no restriction on the input
 * image size. Each ROI width and height must be even and no greater than 128. One-pixel-wide bridge
 * cases are supported. If no contour is found within an ROI, its length output is 0 and no valid
 * contour coordinates are produced for that ROI.
 *
 * \b Input tensor:
 *      Data Layout:    [HWC]
 *          H: Image height
 *          W: Image width
 *          C: 1
 *
 *      Data Type      | Allowed
 *      -------------- | ---------
 *      8bit Unsigned  | Yes
 *
 * \b Rect tensor:
 *      Data Layout:    [W]
 *          W: Maximum number of bounding rectangles, must <= 32
 *      Data Type:      [4S32]
 *
 * \b Output tensor:
 *      Data Layout:    [HWC]
 *          H: Maximum number of bounding rectangles, must <= 32
 *          W: Maximum number of contour pixels per bounding rectangle, must <= 16384
 *          C: 2 (x and y coordinates)
 *      Data Type:      [16bit Unsigned]
 *
 * \b Len tensor:
 *      Data Layout:    [W]
 *          W: Maximum number of bounding rectangles, must <= 32
 *      Data Type:      [16bit Unsigned]
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] inTensorRequirements Pointer to the NVCVTensorRequirements structure for the input tensor.
 *
 * @param [in] rectTensorRequirements Pointer to the NVCVTensorRequirements structure for the bounding rectangle tensor.
 *
 * @param [in] outTensorRequirements Pointer to the NVCVTensorRequirements structure for the output tensor.
 *
 * @param [in] lenTensorRequirements Pointer to the NVCVTensorRequirements structure for the contour length tensor.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaContourTracingCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements const *const inTensorRequirements,
                                   NVCVTensorRequirements const *const rectTensorRequirements,
                                   NVCVTensorRequirements const *const outTensorRequirements,
                                   NVCVTensorRequirements const *const lenTensorRequirements);

/**
 * Builds the next direction lookup table for contour tracing.
 *
 * For each 8-neighbor bitmap values in [0, 255], packs the next move direction for each of the
 * 4 possible previous directions into one uint16_t entry (4 bits per previous direction).
 *
 * @param [out] table Caller-allocated array of at least
 *                    PVA_CONTOUR_TRACING_DIRECTION_TABLE_ENTRIES elements.
 *                    Must not be NULL.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Table is null.
 * @retval NVCV_SUCCESS                Table written successfully.
 */
NVCVStatus pvaContourTracingBuildDirectionTable(uint16_t *table);

#ifdef __cplusplus
}

/**
 * Submits the ContourTracing operator to a cuPVA stream.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 *
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] inTensor Input tensor handle.
 *
 * @param [in] rectTensor Bounding rectangle tensor handle.
 *                        Each element is an NVCVRectI structure, representing {x, y, width, height} in NVCV_DATA_TYPE_4S32 datatype.
 *
 * @param [out] outTensor Output tensor handle.
 *
 * @param [out] lenTensor Contour length tensor handle.
 *
 * @param [in] numRects Number of bounding rectangles to process.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaContourTracingSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle const inTensor,
                                   NVCVTensorHandle const rectTensor, NVCVTensorHandle outTensor,
                                   NVCVTensorHandle lenTensor, const int32_t numRects);

/**
 * Submits the ContourTracing operator to a CUDA stream.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 *
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] inTensor Input tensor handle.
 *
 * @param [in] rectTensor Bounding rectangle tensor handle.
 *                        Each element is an NVCVRectI structure, representing {x, y, width, height} in NVCV_DATA_TYPE_4S32 datatype.
 *
 * @param [out] outTensor Output tensor handle.
 *
 * @param [out] lenTensor Contour length tensor handle.
 *
 * @param [in] numRects Number of bounding rectangles to process.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaContourTracingSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle const inTensor,
                                   NVCVTensorHandle const rectTensor, NVCVTensorHandle outTensor,
                                   NVCVTensorHandle lenTensor, const int32_t numRects);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPCONTOURTRACING_H */
