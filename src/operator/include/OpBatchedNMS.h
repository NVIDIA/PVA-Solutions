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
 * @file OpBatchedNMS.h
 *
 * @brief Defines types and functions to handle the Batched Non-Maximum Suppression operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_BATCHED_NMS BatchedNMS
 * @{
 */

#ifndef PVA_SOLUTIONS_OPBATCHEDNMS_H
#define PVA_SOLUTIONS_OPBATCHEDNMS_H

#include <PvaOperator.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructs an instance of the Batched NMS operator.
 *
 * This operator performs per-class Non-Maximum Suppression on bounding boxes.
 *
 * Algorithm:
 *  - For each class:
 *    1. Sort boxes by confidence score (descending)
 *    2. Keep highest-confidence box
 *    3. Suppress overlapping boxes (IoU > threshold)
 *    4. Repeat until no boxes remain
 *  - Collect all surviving boxes from all classes
 *  - Stop after maxOutputBoxes survivors in class first-appearance order.
 *    Survivors are confidence-sorted within each class, not globally across classes.
 *
 * \b Limitations:
 *  - Max input boxes: 8192
 *  - Max output boxes: 1000
 *  - Max boxes per class: 4096. Only the first 4096 boxes of a class in input order take part
 *    in suppression; later boxes of that class are ignored regardless of their scores.
 *  - Runtime input count is clamped to [0, maxInputBoxes].
 *  - Input class identifiers must be in [0, 299]. Boxes with other identifiers are ignored.
 *  - Input scores must be finite and greater than 0. Boxes are ordered by the FP16 bit pattern
 *    of their scores, so zero, negative, infinite or NaN scores give an undefined order.
 *    The order of boxes with equal scores is unspecified.
 *  - Only the first count entries of the output tensors are written. Entries at and after the
 *    output count keep their previous contents.
 *  - Box tensors must have contiguous rows (strides[1] equal to the element size); row padding
 *    (strides[0] larger than the row size) is supported. One-dimensional tensors must be
 *    tightly packed.
 *  - Tensor layouts are not checked. The samples use "HW" for the box tensors and "W" for the
 *    one-dimensional tensors. The tensors passed at submission must match the requirements
 *    passed to pvaBatchedNMSCreate(), including the layout.
 *
 * \b Input Tensors (Structure-of-Arrays format):
 *    Tensor Name  | dtype               | rank | layout | shape
 *    ------------ | ------------------- | ---- | ------ | -----
 *    boxes        | NVCV_DATA_TYPE_F32  | 2    | HW     | [4, maxInputBoxes]
 *    scores       | NVCV_DATA_TYPE_F16  | 1    | W      | [maxInputBoxes]
 *    classes      | NVCV_DATA_TYPE_S16  | 1    | W      | [maxInputBoxes]
 *    count        | NVCV_DATA_TYPE_S32  | 1    | W      | [1]
 *
 *  boxes format: rows [left, top, right, bottom] in pixel coordinates
 *  count: actual number of input boxes (can be < maxInputBoxes)
 *
 * \b Output Tensors (Structure-of-Arrays format):
 *    Tensor Name  | dtype               | rank | layout | shape
 *    ------------ | ------------------- | ---- | ------ | -----
 *    boxes        | NVCV_DATA_TYPE_F32  | 2    | HW     | [4, maxOutputBoxes]
 *    scores       | NVCV_DATA_TYPE_F32  | 1    | W      | [maxOutputBoxes]
 *    classes      | NVCV_DATA_TYPE_S32  | 1    | W      | [maxOutputBoxes]
 *    count        | NVCV_DATA_TYPE_S32  | 1    | W      | [1]
 *
 *  count: number of boxes kept, in [0, maxOutputBoxes]
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] inBoxesReqs Pointer to NVCVTensorRequirements for the input boxes tensor.
 *                         + Must not be NULL.
 *
 * @param [in] inScoresReqs Pointer to NVCVTensorRequirements for the input scores tensor.
 *                          + Must not be NULL.
 *
 * @param [in] inClassesReqs Pointer to NVCVTensorRequirements for the input classes tensor.
 *                           + Must not be NULL.
 *
 * @param [in] inCountReqs Pointer to NVCVTensorRequirements for the input count tensor.
 *                         + Must not be NULL.
 *
 * @param [in] outBoxesReqs Pointer to NVCVTensorRequirements for the output boxes tensor.
 *                          + Must not be NULL.
 *
 * @param [in] outScoresReqs Pointer to NVCVTensorRequirements for the output scores tensor.
 *                           + Must not be NULL.
 *
 * @param [in] outClassesReqs Pointer to NVCVTensorRequirements for the output classes tensor.
 *                            + Must not be NULL.
 *
 * @param [in] outCountReqs Pointer to NVCVTensorRequirements for the output count tensor.
 *                          + Must not be NULL.
 *
 * @param [in] iouThreshold IoU threshold for suppression (e.g., 0.45).
 *                          Boxes with IoU > threshold are suppressed.
 *                          + Must be in the range [0, 1].
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT The possible cases might be the following:
 *                                      1) handle or any of the tensor requirements is NULL.
 *                                      2) iouThreshold is outside [0, 1] or is NaN.
 *                                      3) A box tensor does not have rank 2, data type F32 or
 *                                         shape[0] equal to 4.
 *                                      4) maxInputBoxes is outside [1, 8192] or maxOutputBoxes is
 *                                         outside [1, 1000].
 *                                      5) A box tensor has strides[1] different from 4 or a row
 *                                         stride strides[0] smaller than the row size.
 *                                      6) A one-dimensional tensor does not have rank 1, has a data
 *                                         type other than the one listed above, or its extent is not
 *                                         maxInputBoxes, maxOutputBoxes or 1 as listed above.
 *                                      7) A one-dimensional tensor is not tightly packed.
 * @retval NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator.
 * @retval NVCV_SUCCESS                Operator created successfully.
 */
NVCVStatus pvaBatchedNMSCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements const *inBoxesReqs,
                               NVCVTensorRequirements const *inScoresReqs, NVCVTensorRequirements const *inClassesReqs,
                               NVCVTensorRequirements const *inCountReqs, NVCVTensorRequirements const *outBoxesReqs,
                               NVCVTensorRequirements const *outScoresReqs,
                               NVCVTensorRequirements const *outClassesReqs, NVCVTensorRequirements const *outCountReqs,
                               float iouThreshold);

#ifdef __cplusplus
}

/**
 * Submits the Batched NMS operator to a cuPVA stream.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 *
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] inBoxes Input boxes tensor [4, maxInputBoxes].
 *                     + Must not be NULL.
 *
 * @param [in] inScores Input scores tensor [maxInputBoxes].
 *                      + Must not be NULL.
 *
 * @param [in] inClasses Input classes tensor [maxInputBoxes].
 *                       + Must not be NULL.
 *
 * @param [in] inCount Input count tensor [1] containing the number of valid input boxes.
 *                     + Must not be NULL.
 *
 * @param [out] outBoxes Output boxes tensor [4, maxOutputBoxes].
 *                       + Must not be NULL.
 *
 * @param [out] outScores Output scores tensor [maxOutputBoxes].
 *                        + Must not be NULL.
 *
 * @param [out] outClasses Output classes tensor [maxOutputBoxes].
 *                         + Must not be NULL.
 *
 * @param [out] outCount Output count tensor [1] containing the number of output boxes.
 *                       + Must not be NULL.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT The possible cases might be the following:
 *                                      1) handle or any tensor is NULL, or points to an invalid address.
 *                                      2) A tensor does not meet the requirements used to create the
 *                                         operator handle.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, submission failed.
 * @retval NVCV_SUCCESS                Operation submitted successfully.
 */
NVCVStatus pvaBatchedNMSSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle inBoxes,
                               NVCVTensorHandle inScores, NVCVTensorHandle inClasses, NVCVTensorHandle inCount,
                               NVCVTensorHandle outBoxes, NVCVTensorHandle outScores, NVCVTensorHandle outClasses,
                               NVCVTensorHandle outCount);

/**
 * Submits the Batched NMS operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] inBoxes Input boxes tensor.
 * @param [in] inScores Input scores tensor.
 * @param [in] inClasses Input classes tensor.
 * @param [in] inCount Input count tensor.
 * @param [out] outBoxes Output boxes tensor.
 * @param [out] outScores Output scores tensor.
 * @param [out] outClasses Output classes tensor.
 * @param [out] outCount Output count tensor.
 *
 * @retval NVCV_ERROR_NOT_IMPLEMENTED  CUDA stream submission is not supported on this platform.
 * @retval NVCV_ERROR_INVALID_ARGUMENT Invalid handle or tensor.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, submission failed.
 * @retval NVCV_SUCCESS                Operation submitted successfully.
 */
NVCVStatus pvaBatchedNMSSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle inBoxes,
                               NVCVTensorHandle inScores, NVCVTensorHandle inClasses, NVCVTensorHandle inCount,
                               NVCVTensorHandle outBoxes, NVCVTensorHandle outScores, NVCVTensorHandle outClasses,
                               NVCVTensorHandle outCount);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPBATCHEDNMS_H */
