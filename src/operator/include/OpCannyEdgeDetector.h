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
 * @file OpCannyEdgeDetector.h
 *
 * @brief Defines types and functions to handle the Canny edge detection operation.
 *        This operator includes the following steps:
 *        1. Compute the gradient intensity and orientation of the image using the Sobel filter.
 *        2. Non-maximum suppression to thin the edges.
 *        3. Double thresholding to detect strong and weak edges.
 *        4. Edge tracking by hysteresis to retain only those weak edges that are connected to strong ones; all other weak edges are removed.
 *        Note that pre-processing to remove noise in the image is not included in this operator.
 *
 * @defgroup PVA_OPERATOR_ALGORITHM_CANNY_EDGEDETECTOR CannyEdgeDetector
 * @{
 */

#ifndef PVA_SOLUTIONS_CANNY_EDGE_DETECTOR_H
#define PVA_SOLUTIONS_CANNY_EDGE_DETECTOR_H

#ifdef __cplusplus
#    include <cuda_runtime.h>
#endif

#include <PvaOperator.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Constructs an instance of the CannyEdgeDetector operator.
 *
 * \b Limitations:
 *    Edge tracking by hysteresis uses the CCL (Connected Component Labeling) approach.
 *    The maximum number of labels is limited to 65536.
 *    If the number of labels exceeds this limit, every pixel in edgeMap is set to 1.
 *    Normal edgeMap values are 0 for non-edge pixels and 255 for edge pixels.
 *
 * \b Image:
 *    Data Layout:    [CHW], [NCHW] only when
 *    C:              [1]
 *    N:              [1]
 *
 *    Data Type      | Allowed
 *    -------------- | --------------
 *    8bit  Unsigned | Yes
 *    8bit  Signed   | No
 *    16bit Unsigned | No
 *    16bit Signed   | No
 *    32bit Unsigned | No
 *    32bit Signed   | No
 *    32bit Float    | No
 *    64bit Float    | No
 *
 *    The output edgeMap NVCVTensorRequirements could be deduced from input imageParams.
 *
 * \b Input/Output \b Dependency:
 *    Property       | Input == Output
 *    -------------- | --------------
 *    Data Layout    | Yes
 *    Data Type      | Yes
 *    Number         | Yes
 *    Channels       | Yes
 *    Width          | Yes
 *    Height         | Yes
 *
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] imageParams Pointer to the NVCVTensorRequirements structure which contains [in] image Tensor rank, shape, layout and data type information.
 *
 * @param [in] gradientSize The Sobel kernel size. Must be 3 or 5 or 7.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_OUT_OF_MEMORY    No enough memory to create the operator.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaCannyEdgeDetectorCreate(NVCVOperatorHandle *handle, const NVCVTensorRequirements *imageParams,
                                      const int32_t gradientSize);

#ifdef __cplusplus
}

/**
 * Submits the CannyEdgeDetector operator to a cuPVA stream.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] cmdStatus Pointer to the array of cuPVA command status.
 *                       Used to track the execution status of the operation.
 *
 * @param [in] image Input tensor handle of image.
 *                   Data Type: NVCV_DATA_TYPE_U8
 *                   Data Layout: [CHW], [NCHW] only when C = 1 and N = 1
 *
 * @param [out] edgeMap Output tensor handle of linked edges.
 *                      Data Type: NVCV_DATA_TYPE_U8
 *                      Data Layout: [CHW], [NCHW] only when C = 1 and N = 1
 *                      Each pixel in the edgeMap tensor has one of the following values:
 *
 *                      Value | Description
 *                      ----- | ----------------------------
 *                      0     | Non-edge
 *                      1     | CCL region overflow sentinel
 *                      255   | Edge
 *
 * @param [in] thresholdStrong Strong threshold for the edge hysteresis procedure.
 *                             If gradient intensity is larger than the strong threshold, edge type is set to strong.
 *
 * @param [in] thresholdWeak Weak threshold for the edge hysteresis procedure.
 *                           If gradient intensity is larger than the weak threshold and less than or equal to the strong threshold,
 *                           edge type is set to weak.
 *                           If gradient intensity is less than or equal to the weak threshold, edge type is set to non-edge.
 *                           Reasonble values for double thresholding are thresholdStrong > thresholdWeak > 0.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside the valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaCannyEdgeDetectorSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, cupvaCmdStatus_t *cmdStatus,
                                      NVCVTensorHandle image, NVCVTensorHandle edgeMap, const int32_t thresholdStrong,
                                      const int32_t thresholdWeak);

/**
 * Submits the CannyEdgeDetector operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] cmdStatus Pointer to the array of cuPVA command status.
 * @param [in] image Input tensor handle of image.
 * @param [out] edgeMap Output tensor handle of linked edges.
 * @param [in] thresholdStrong Strong threshold for the edge hysteresis procedure.
 * @param [in] thresholdWeak Weak threshold for the edge hysteresis procedure.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside the valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaCannyEdgeDetectorSubmit(NVCVOperatorHandle handle, cudaStream_t stream, cupvaCmdStatus_t *cmdStatus,
                                      NVCVTensorHandle image, NVCVTensorHandle edgeMap, const int32_t thresholdStrong,
                                      const int32_t thresholdWeak);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_CANNY_EDGE_DETECTOR_H */