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
 * @file OpNhwcToNchwLetterboxNormalize.h
 *
 * @brief Defines types and functions to handle the fused HWC-to-CHW letterbox and normalize operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_NHWC_TO_NCHW_LETTERBOX_NORMALIZE NhwcToNchwLetterboxNormalize
 * @{
 */

#ifndef PVA_SOLUTIONS_OPNHWCTONCHWLETTERBOXNORMALIZE_H
#define PVA_SOLUTIONS_OPNHWCTONCHWLETTERBOXNORMALIZE_H

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
 * @enum PVANhwcToNchwLetterboxNormalizePaddingPolicy
 * @brief Defines how the letterbox padding region of the output tensor is initialized.
 */
typedef enum
{
    /**
     * The operator writes padValue into the padding region on every submission, before the
     * PVA image-region write and after earlier work on the supplied stream. The fill is done
     * on the host through the output tensor's host mapping. On a CUDA stream it is enqueued
     * with cudaLaunchHostFunc() and Submit stays asynchronous. cuPVA has no host-callback
     * command, so on a cuPVA stream Submit submits a fence, waits for earlier stream work to
     * complete, fills the padding and then submits the PVA command; Submit is therefore
     * synchronous with respect to earlier work on that stream.
     */
    PVA_NHWC_TO_NCHW_PADDING_RESTORE_ON_SUBMIT = 0,
    /**
     * The caller initializes the padding region of every output tensor with the padding
     * value before submitting it. The operator writes only the image region and padValue
     * is unused.
     */
    PVA_NHWC_TO_NCHW_PADDING_PREFILLED = 1
} PVANhwcToNchwLetterboxNormalizePaddingPolicy;

/**
 * Constructs an instance of the NhwcToNchwLetterboxNormalize operator.
 *
 * Converts a single interleaved uint8 RGB image into a planar float32 tensor, centering it
 * in a larger output (letterboxing) and scaling its values. The input must already be
 * resized to imgWidth x imgHeight; the operator does not resize. Given an input of shape
 * [imgHeight, imgWidth, 3] and an output of shape [3, outHeight, outWidth],
 *
 *     padTop  = (outHeight - imgHeight) / 2
 *     padLeft = (outWidth  - imgWidth)  / 2
 *
 *     out[c, y + padTop, x + padLeft] = (float)in[y, x, c] * scale
 *
 * for 0 <= y < imgHeight, 0 <= x < imgWidth, 0 <= c < 3. Every other output element is
 * padding and holds padValue. An odd leftover row or column goes to the bottom or right
 * padding. No mean or standard-deviation normalization and no channel swap is applied:
 * input channel c is written to output plane c.
 *
 * \b Limitations:
 *      Both tensors are rank 3 and describe a single image; there is no batch (N) dimension.
 *
 *      - Input pixels are tightly packed within each row: strides[2] must be 1 and
 *        strides[1] must be 3. Rows may be followed by padding: strides[0] must be at
 *        least imgWidth * 3.
 *      - The output tensor must be tightly packed.
 *      - imgWidth must not exceed outWidth and imgHeight must not exceed outHeight.
 *      - scale and padValue must be finite.
 *
 *      With PVA_NHWC_TO_NCHW_PADDING_RESTORE_ON_SUBMIT, Submit on a cuPVA stream waits
 *      for earlier work on that stream before returning.
 *
 * \b Input:
 *      Data Layout:    [HWC] interleaved
 *      Dimensions:     imgHeight x imgWidth x 3
 *      Data Type:      8bit Unsigned (NVCV_DATA_TYPE_U8)
 *
 * \b Output:
 *      Data Layout:    [CHW] planar
 *      Dimensions:     3 x outHeight x outWidth
 *      Data Type:      32bit Float (NVCV_DATA_TYPE_F32)
 *
 * @param [out] handle        Where the operator instance handle will be written to.
 *                            + Must not be NULL.
 *
 * @param [in]  inTensorReqs  Requirements of the input tensor (imgHeight x imgWidth x 3, U8, HWC).
 *                            + Must not be NULL.
 *
 * @param [in]  outTensorReqs Requirements of the output tensor (3 x outHeight x outWidth, F32, CHW).
 *                            + Must not be NULL.
 *
 * @param [in]  scale         Multiplier applied to every image pixel (e.g. 1.0 / 255.0).
 *
 * @param [in]  padValue      Value written to padding elements (e.g. 114.0 / 255.0). Unused with
 *                            PVA_NHWC_TO_NCHW_PADDING_PREFILLED.
 *
 * @param [in]  paddingPolicy How the padding region is initialized, see
 *                            PVANhwcToNchwLetterboxNormalizePaddingPolicy.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT The possible cases might be the following:
 *                                      1) handle, inTensorReqs or outTensorReqs is NULL, or points to an
 *                                         invalid address.
 *                                      2) paddingPolicy is not a PVANhwcToNchwLetterboxNormalizePaddingPolicy
 *                                         value.
 *                                      3) scale or padValue is not finite.
 *                                      4) A tensor does not have rank 3, the documented layout or data type,
 *                                         or 3 channels, or has a non-positive extent.
 *                                      5) The tensors violate a documented limitation above: input larger
 *                                         than the output, an input row pitch below imgWidth * 3, or a
 *                                         non-packed pixel or output layout.
 * @retval NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaNhwcToNchwLetterboxNormalizeCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements const *inTensorReqs,
                                                 NVCVTensorRequirements const *outTensorReqs, float scale,
                                                 float padValue,
                                                 PVANhwcToNchwLetterboxNormalizePaddingPolicy paddingPolicy);

#ifdef __cplusplus
}

/**
 * Submits the NhwcToNchwLetterboxNormalize operator to a cuPVA stream.
 *
 * @param [in]  handle Handle to the operator.
 *                     + Must not be NULL.
 *
 * @param [in]  stream Handle to a valid cuPVA stream.
 *
 * @param [in]  in     Input tensor handle. Must match the input tensor requirements passed to
 *                     pvaNhwcToNchwLetterboxNormalizeCreate().
 *                     + Must not be NULL.
 *
 * @param [out] out    Output tensor handle. Must match the output tensor requirements passed to
 *                     pvaNhwcToNchwLetterboxNormalizeCreate().
 *                     + Must not be NULL.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT The possible cases might be the following:
 *                                      1) handle, in or out is NULL, or points to an invalid address.
 *                                      2) The input or output tensor does not meet the requirements used to
 *                                         create the operator handle.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, including a failure to map the output
 *                                     tensor to the host or to wait for earlier stream work.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaNhwcToNchwLetterboxNormalizeSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle in,
                                                 NVCVTensorHandle out);

/**
 * Submits the NhwcToNchwLetterboxNormalize operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * With PVA_NHWC_TO_NCHW_PADDING_RESTORE_ON_SUBMIT this path enqueues the padding fill with
 * cudaLaunchHostFunc(), so it uses the CUDA runtime.
 *
 * @param [in]  handle Handle to the operator.
 *                     + Must not be NULL.
 *
 * @param [in]  stream Handle to a valid CUDA stream.
 *
 * @param [in]  in     Input tensor handle. Must match the input tensor requirements passed to
 *                     pvaNhwcToNchwLetterboxNormalizeCreate().
 *                     + Must not be NULL.
 *
 * @param [out] out    Output tensor handle. Must match the output tensor requirements passed to
 *                     pvaNhwcToNchwLetterboxNormalizeCreate().
 *                     + Must not be NULL.
 *
 * @retval NVCV_ERROR_NOT_IMPLEMENTED  CUDA stream submission is not supported on this platform.
 * @retval NVCV_ERROR_INVALID_ARGUMENT The possible cases might be the following:
 *                                      1) handle, in or out is NULL, or points to an invalid address.
 *                                      2) The input or output tensor does not meet the requirements used to
 *                                         create the operator handle.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, including a failure to map the output
 *                                     tensor to the host or to enqueue the padding fill.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaNhwcToNchwLetterboxNormalizeSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                                 NVCVTensorHandle out);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPNHWCTONCHWLETTERBOXNORMALIZE_H */
