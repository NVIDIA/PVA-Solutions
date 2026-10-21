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
 * @file OpFastCorner.h
 *
 * @brief This header file defines the types and functions necessary for implementing
 *        the FAST (Features from Accelerated Segment Test) corner detection algorithm.
 *        The FAST corner detection method is known for its computational efficiency
 *        and is particularly well-suited for real-time applications such as video
 *        processing and robotics.
 *
 * @defgroup PVA_OPERATOR_ALGORITHM_FAST_CORNER FastCornerDetector
 * @{
 */

#ifndef PVA_SOLUTIONS_FASTCORNER_H
#define PVA_SOLUTIONS_FASTCORNER_H

#include <PvaOperator.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/BorderType.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @cond SKIP_THIS */
#define PVA_FAST_CORNER_MAX_TOPK_FEATURES (250000)
/** @endcond */

/*
 * The FAST corner detection algorithm evaluates a candidate pixel by examining
 * a circular area of surrounding pixels. A pixel is classified as a corner if
 * a specified number \( N \) of contiguous pixels in the circle are either
 * significantly brighter or darker than the candidate pixel, based on a defined
 * intensity threshold.
 *
 * The algorithm may apply non-maximum suppression to eliminate redundant detections,
 * keeping only the candidate pixel if its intensity is larger than that of its
 * immediate neighbors.
 *
 */
/**
 * @brief Structure to hold parameters for the FAST corner detector.
 */
typedef struct PvaFastCornerDetectorParamsRec
{
    /// The radius of the circle around a pixel used for corner detection.
    int32_t circleRadius;

    /// The arc length defines the number of contiguous pixels on the circle
    /// that are considered during corner evaluation.
    int32_t arcLength;

    /// Specifies whether non-maximum suppression should be applied to
    /// refine corner detection results.
    uint8_t nonMaxSuppression;

    /// Number of highest-scoring corners to return using TopK feature selection.
    /// When set to 0 (default), TopK is disabled and all detected corners are returned.
    /// When set to a value in [1, PVA_FAST_CORNER_MAX_TOPK_FEATURES], the operator uses
    /// histogram-based score thresholding to return approximately the K highest-scoring corners.
    int32_t numTopKFeatures;
} PvaFastCornerDetectorParams;

/** Constructs an instance of the FAST corner operator.
 *
 *
 * \b Limitations:
 *      This release only supports specific parameter values: circleRadius=3,
 *      arcLength=9.
 *
 * \b Input:
 *      Data Layout:    [HW]
 *      Channels:       [1]
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      8-bit  Unsigned| Yes
 *      8-bit  Signed  | Yes
 *      16-bit Unsigned| Yes
 *      16-bit Signed  | Yes
 *
 *
 * \b Parameters:
 *
 *      params:
 *          - Typical combinations for circleRadius and arcLength include {1, 5},
 *            {2, 7}, {3, 9}.
 *          - nonMaxSuppression can be set to 0 (off) or 1 (on).
 *          - numTopKFeatures: set to 0 to disable TopK (default). Set to a value
 *            in [1, PVA_FAST_CORNER_MAX_TOPK_FEATURES] to enable histogram-based
 *            score thresholding that returns approximately the K highest-scoring corners.
 *
 *      borderMode:  NVCV_BORDER_CONSTANT or NVCV_BORDER_REPLICATE.
 *
 *
 * @param [out] handle Pointer to where the operator instance handle will be written.
 *                     Must not be NULL.
 *
 * @param [in] tensorRequirements Pointer to a structure containing tensor rank, shape,
 *                                layout, and data type information.
 *
 * @param [in] params Parameters for performing FAST corner detection. \ref PvaFastCornerDetectorParams.
 *
 * @param [in] borderMode Mode used for handling borders during processing.
 *
 * @param [in] borderValue Constant value used when borderMode is NVCV_BORDER_CONSTANT; ignored otherwise.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_OUT_OF_MEMORY          Possible cases include:
 *                                            1) Failed to allocate memory for the operator.
 *                                            2) Failed to allocate memory for holding intermediate buffer, which the host initializes then sends it for the device's use.
 * @retval NVCV_SUCCESS                      Operation executed successfully.
 */
NVCVStatus pvaFastCornerDetectorCreate(NVCVOperatorHandle *handle,
                                       NVCVTensorRequirements const *const tensorRequirements,
                                       PvaFastCornerDetectorParams const *const params, NVCVBorderType borderMode,
                                       int32_t borderValue);

#ifdef __cplusplus
}

/**
 * Submits the FastCornerDetector operator to a cuPVA stream.
 *
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] in Input tensor handle containing image data.
 *
 * @param [in] intensityThreshold Threshold used to determine if pixels in the circle are
 *                                considered brighter or darker than the central pixel's intensity.
 *                                + Must be non-negative.
 *
 * @param [out] loc Output tensor handle for storing coordinates of detected corners.
 *                  Maximum number of locations ("locCapacity") is inferred from tensor size.
 *
 *      Data Layout:    [W]
 *      Channel:        1
 *      Length:         locCapacity
 *      Data Type:      NVCV_DATA_TYPE_2F32 (XY-interleaved)
 *
 * @param [out] numLoc Output tensor handle for storing count of detected corners.
 *
 *      Data Layout:    [W]
 *      Length:         1
 *      Range:          [0, "image height" x "image width"]
 *      Data Type:      NVCV_DATA_TYPE_S32
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT       Possible cases include:
 *                                            1) The handle or stream or in or out is either NULL or points to an invalid address.
 *                                            2) The input or output tensor does not meet the requirements used to create the operator handle.
 * @retval NVCV_SUCCESS                      Operation executed successfully.
 */
NVCVStatus pvaFastCornerDetectorSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle in,
                                       int32_t intensityThreshold, NVCVTensorHandle loc, NVCVTensorHandle numLoc);

/**
 * Submits the FastCornerDetector operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in Input tensor handle containing image data.
 * @param [in] intensityThreshold Threshold used to determine if pixels in the circle are considered brighter or darker.
 * @param [out] loc Output tensor handle for storing coordinates of detected corners.
 * @param [out] numLoc Output tensor handle for storing count of detected corners.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT       Possible cases include invalid parameters.
 * @retval NVCV_SUCCESS                      Operation executed successfully.
 */
NVCVStatus pvaFastCornerDetectorSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                       int32_t intensityThreshold, NVCVTensorHandle loc, NVCVTensorHandle numLoc);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_FASTCORNER_H */
