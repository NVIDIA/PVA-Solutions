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
 * @file OpYoloV5Decode.h
 *
 * @brief Defines types and functions to handle the YOLOv5 decode operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_YOLOV5_DECODE YoloV5Decode
 * @{
 */

#ifndef PVA_SOLUTIONS_OPYOLOV5DECODE_H
#define PVA_SOLUTIONS_OPYOLOV5DECODE_H

#include <PvaOperator.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Raw YOLOv5 head tensor layout. The layout is selected when the operator is created. */
typedef enum
{
    PVA_YOLOV5_DECODE_INPUT_CHW = 0, /**< [3 * (classes + 5), H, W]. */
    PVA_YOLOV5_DECODE_INPUT_HWC = 1  /**< [H, W, 3 * (classes + 5)]. */
} PVAYoloV5DecodeInputLayout;

/**
 * Constructs an instance of the YOLOv5 decode operator.
 *
 * The operator decodes the raw output of one YOLOv5 detection head (one scale) into
 * boxes, scores and class identifiers, keeping the detections whose score is above a
 * confidence threshold.
 *
 * \b Semantics:
 *
 *  The prediction tensor holds 3 anchors x (numClasses + 5) values per grid cell. For
 *  anchor a of cell (x, y) the values are, in order: tx, ty, tw, th, obj and one logit
 *  cls[c] per class. In CHW layout value k of anchor a is channel a * (numClasses + 5) + k;
 *  in HWC layout it is element a * (numClasses + 5) + k of the cell. The anchor row of
 *  anchor a of cell (x, y) is i = (y * W + x) * 3 + a, and row i of the anchors tensor
 *  holds [grid_x, grid_y, anchor_w, anchor_h, stride] for it, with anchor_w, anchor_h and
 *  stride in pixels. With sigmoid(v) = 1 / (1 + exp(-v)):
 *
 *      cx    = (2 * sigmoid(tx) - 0.5 + grid_x) * stride
 *      cy    = (2 * sigmoid(ty) - 0.5 + grid_y) * stride
 *      w     = (2 * sigmoid(tw))^2 * anchor_w
 *      h     = (2 * sigmoid(th))^2 * anchor_h
 *      score = sigmoid(obj) * sigmoid(cls[c])
 *
 *  One detection is emitted for every class c with score > confThreshold (strictly
 *  greater), so an anchor can produce several detections that share the same box. The
 *  box is written as
 *
 *      left   = max(cx - w / 2, 0)
 *      top    = max(cy - h / 2, 0)
 *      right  = min(cx + w / 2, imgWidth)
 *      bottom = min(cy + h / 2, imgHeight)
 *
 *  the score is stored as FP16 and the class as its index c. Detections are written in
 *  ascending anchor row order and, within one anchor row, in ascending class order,
 *  starting at the position given by the count tensor. Once maxOutputBoxes detections
 *  are stored the remaining detections are dropped without an error and the count
 *  stays at maxOutputBoxes.
 *
 *  The count tensor is an in/out append cursor, so the outputs can be shared by the
 *  operators of all scales of a model: initialize it to zero before the first scale and
 *  submit the scales in order. An incoming count outside [0, maxOutputBoxes] is clamped
 *  to that range.
 *
 * \b Limitations:
 *  - Exactly 80 classes and three anchors per grid cell are supported.
 *  - The prediction and anchors tensors must be tightly packed.
 *  - The rows of the boxes tensor may be padded (strides[0] >= maxOutputBoxes * sizeof(float));
 *    its elements, and those of the scores, classes and count tensors, must be contiguous.
 *  - maxOutputBoxes must not exceed 5120.
 *  - The layout fields of the tensor requirements are not checked; the layout of the
 *    prediction tensor is selected by inputLayout.
 *
 * \b Input Tensors:
 *    Tensor Name  | dtype               | rank | layout           | shape
 *    ------------ | ------------------- | ---- | ---------------- | -----
 *    predictions  | NVCV_DATA_TYPE_F32  | 3    | CHW or HWC       | [255, H, W] or [H, W, 255]
 *    anchors      | NVCV_DATA_TYPE_F32  | 2    | HW               | [H * W * 3, 5]
 *
 * \b Output Tensors:
 *    Tensor Name  | dtype               | rank | layout | shape
 *    ------------ | ------------------- | ---- | ------ | -----
 *    boxes        | NVCV_DATA_TYPE_F32  | 2    | HW     | [4, maxOutputBoxes]
 *    scores       | NVCV_DATA_TYPE_F16  | 1    | W      | [maxOutputBoxes]
 *    classes      | NVCV_DATA_TYPE_S16  | 1    | W      | [maxOutputBoxes]
 *    count        | NVCV_DATA_TYPE_S32  | 1    | W      | [1]
 *
 *  The boxes tensor uses structure-of-arrays storage: row 0 holds left, row 1 top,
 *  row 2 right and row 3 bottom.
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] inPredReqs Pointer to the NVCVTensorRequirements of the predictions tensor.
 *                        + Must not be NULL.
 *
 * @param [in] inAnchorReqs Pointer to the NVCVTensorRequirements of the anchors tensor.
 *                          + Must not be NULL.
 *
 * @param [in] outBoxesReqs Pointer to the NVCVTensorRequirements of the boxes tensor.
 *                          + Must not be NULL.
 *
 * @param [in] outScoresReqs Pointer to the NVCVTensorRequirements of the scores tensor.
 *                           + Must not be NULL.
 *
 * @param [in] outClassesReqs Pointer to the NVCVTensorRequirements of the classes tensor.
 *                            + Must not be NULL.
 *
 * @param [in] outCountReqs Pointer to the NVCVTensorRequirements of the count tensor.
 *                          + Must not be NULL.
 *
 * @param [in] numClasses Number of object classes. Must be 80.
 *
 * @param [in] confThreshold Confidence threshold for filtering detections (e.g., 0.25).
 *                           + Must be in the range (0, 1), exclusive.
 *
 * @param [in] imgHeight Network input height in pixels. Must be positive.
 *
 * @param [in] imgWidth Network input width in pixels. Must be positive.
 *
 * @param [in] inputLayout Layout of the prediction tensor.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT The possible cases might be the following:
 *                                      1) handle or a tensor requirements pointer is NULL.
 *                                      2) numClasses is not 80.
 *                                      3) confThreshold is not in (0, 1), or imgHeight or imgWidth
 *                                         is not positive.
 *                                      4) inputLayout is not a PVAYoloV5DecodeInputLayout value.
 *                                      5) A tensor has the wrong rank, data type or shape, or
 *                                         violates a documented limitation above.
 * @retval NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator.
 * @retval NVCV_SUCCESS                Operator created successfully.
 */
NVCVStatus pvaYoloV5DecodeCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements const *inPredReqs,
                                 NVCVTensorRequirements const *inAnchorReqs, NVCVTensorRequirements const *outBoxesReqs,
                                 NVCVTensorRequirements const *outScoresReqs,
                                 NVCVTensorRequirements const *outClassesReqs,
                                 NVCVTensorRequirements const *outCountReqs, int numClasses, float confThreshold,
                                 int imgHeight, int imgWidth, PVAYoloV5DecodeInputLayout inputLayout);

#ifdef __cplusplus
}

/**
 * Submits the YOLOv5 decode operator to a cuPVA stream.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] predictions Raw prediction tensor for one YOLOv5 scale.
 *                         + Must not be NULL.
 * @param [in] anchors Anchors tensor for the same scale.
 *                     + Must not be NULL.
 * @param [out] boxes Decoded boxes tensor, shared by all scales.
 *                    + Must not be NULL.
 * @param [out] scores Decoded scores tensor, shared by all scales.
 *                     + Must not be NULL.
 * @param [out] classes Decoded class identifiers tensor, shared by all scales.
 *                      + Must not be NULL.
 * @param [in,out] count Append cursor. Initialize it to zero before the first scale.
 *                       + Must not be NULL.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT The possible cases might be the following:
 *                                      1) handle or a tensor handle is NULL, or points to an invalid
 *                                         address.
 *                                      2) A tensor does not meet the requirements used to create the
 *                                         operator handle.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator.
 * @retval NVCV_SUCCESS                Operation submitted successfully.
 */
NVCVStatus pvaYoloV5DecodeSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle predictions,
                                 NVCVTensorHandle anchors, NVCVTensorHandle boxes, NVCVTensorHandle scores,
                                 NVCVTensorHandle classes, NVCVTensorHandle count);

/**
 * Submits the YOLOv5 decode operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] predictions Raw prediction tensor for one YOLOv5 scale.
 * @param [in] anchors Anchors tensor for the same scale.
 * @param [out] boxes Decoded boxes tensor, shared by all scales.
 * @param [out] scores Decoded scores tensor, shared by all scales.
 * @param [out] classes Decoded class identifiers tensor, shared by all scales.
 * @param [in,out] count Append cursor. Initialize it to zero before the first scale.
 *
 * @retval NVCV_ERROR_NOT_IMPLEMENTED  CUDA stream submission is not supported on this platform.
 * @retval NVCV_ERROR_INVALID_ARGUMENT The possible cases might be the following:
 *                                      1) handle or a tensor handle is NULL, or points to an invalid
 *                                         address.
 *                                      2) A tensor does not meet the requirements used to create the
 *                                         operator handle.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator.
 * @retval NVCV_SUCCESS                Operation submitted successfully.
 */
NVCVStatus pvaYoloV5DecodeSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle predictions,
                                 NVCVTensorHandle anchors, NVCVTensorHandle boxes, NVCVTensorHandle scores,
                                 NVCVTensorHandle classes, NVCVTensorHandle count);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPYOLOV5DECODE_H */
